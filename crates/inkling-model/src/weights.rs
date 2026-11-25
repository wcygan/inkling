//! Exact-mode weights: every text tensor in checkpoint precision, in one device arena.
//!
//! - Dense weights stay BF16; experts stay NVFP4 (layers 0–19) or MXFP8 (layers 20–23).
//! - Q|R|gate|sink rows are concatenated into `layers.{l}.attn.qrgs` [6552, 2304] and K|V into
//!   `layers.{l}.attn.kv` [2048, 2304] by placing the raw copies back to back (squid5090 weights.zig).
//! - Per-expert global scales `amax[e] / outer_scale_denominator` are computed on the host and stored as
//!   `layers.{l}.mlp.experts.{w13,w2}_global_scale` F32 [128]; the raw amax/denominator tensors are not
//!   uploaded.
//! - Vision-tower tensors are skipped.

use crate::config::{dims, is_global_layer};
use crate::loader::{FileCopy, LoadOptions, LoadStats, stream_to_device};
use crate::safetensors::{Dtype, SafetensorsFile, TensorInfo};
use inkling_runtime::{Arena, ArenaPlan, DevSlice, Slot, Stream};
use std::collections::HashMap;
use std::path::Path;

pub struct Weights {
    /// The resident arena (every weight, unless the load was split).
    pub arena: Arena,
    /// Weights only needed while building quantized copies (`load_exact_split`); freed by
    /// `free_transient`. Lookups of a freed weight panic.
    transient: std::sync::Mutex<Option<Arena>>,
    transient_bytes: usize,
    /// name -> (slot, in the transient arena)
    slots: HashMap<String, (Slot, bool)>,
    pub stats: LoadStats,
}

impl Weights {
    fn new(arena: Arena, transient: Option<Arena>, slots: HashMap<String, (Slot, bool)>, stats: LoadStats) -> Weights {
        let transient_bytes = transient.as_ref().map_or(0, Arena::total_bytes);
        Weights { arena, transient: std::sync::Mutex::new(transient), transient_bytes, slots, stats }
    }

    /// The device slice for a weight, by its name without the `model.llm.` prefix.
    pub fn get(&self, name: &str) -> DevSlice {
        match self.try_get(name) {
            Some(s) => s,
            None if self.slots.contains_key(name) => panic!("weight {name} was freed (transient after packing)"),
            None => panic!("no weight named {name}"),
        }
    }

    pub fn try_get(&self, name: &str) -> Option<DevSlice> {
        match self.slots.get(name) {
            Some(&(s, false)) => Some(self.arena.get(s)),
            Some(&(s, true)) => self.transient.lock().unwrap().as_ref().map(|a| a.get(s)),
            None => None,
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.slots.keys().map(String::as_str)
    }

    /// Bytes on the device: resident, and transient while it is still held.
    pub fn device_bytes(&self) -> (usize, usize) {
        let held = self.transient.lock().unwrap().is_some();
        (self.arena.total_bytes(), if held { self.transient_bytes } else { 0 })
    }

    /// Frees the transient arena. Device pointers into it that were resolved earlier dangle: only call
    /// this once nothing that runs will read those weights. Returns the bytes freed.
    pub fn free_transient(&self) -> usize {
        self.transient.lock().unwrap().take().map_or(0, |a| a.total_bytes())
    }
}

/// Which weights fp8 serving (`INKLING_NVFP4_PARTS=experts,head,router,ctrl,dense_fp8` with DSpark and
/// wide prefill) no longer reads once `Nvfp4Weights`, `Fp8Dense` and `DsparkWeights` are built
/// (see the serve memory breakdown): the BF16 head, the
/// BF16 attention projections, the BF16 router and controllers, and the MXFP8 experts of layers 20-23
/// (repacked to NVFP4). Layers 0-19 keep their native NVFP4 experts, which the kernels read in place.
pub fn fp8_serving_transient(name: &str) -> bool {
    if name == "unembed.weight" {
        return true;
    }
    let Some(rest) = name.strip_prefix("layers.") else { return false };
    let Some((l, tail)) = rest.split_once('.') else { return false };
    let Ok(l) = l.parse::<usize>() else { return false };
    matches!(tail, "attn.qrgs" | "attn.kv" | "attn.wo_ud.weight" | "mlp.gate.weight")
        || tail.starts_with("controllers.")
        || (l >= 20 && tail.starts_with("mlp.experts."))
}

/// DSpark drafter weights not read after `DsparkWeights::prepare` packs them: the BF16 projections.
pub fn mtp_transient(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("layers.") else { return false };
    let Some((_, tail)) = rest.split_once('.') else { return false };
    matches!(tail, "qkv" | "gu" | "kv_ctx" | "self_attn.o_proj.weight" | "mlp.down_proj.weight")
}

/// Builds the resident and (when any slot is transient) the transient arena from `(key, bytes,
/// transient)` reservations; returns the arenas and the slot map.
fn plan_split(reservations: &[(String, usize, bool)]) -> Result<(Arena, Option<Arena>, HashMap<String, (Slot, bool)>), String> {
    let mut resident = ArenaPlan::new();
    let mut transient = ArenaPlan::new();
    let mut slots = HashMap::new();
    let mut any = false;
    for (key, bytes, t) in reservations {
        let slot = if *t { transient.reserve(key.clone(), *bytes) } else { resident.reserve(key.clone(), *bytes) };
        any |= *t;
        slots.insert(key.clone(), (slot, *t));
    }
    let r = resident.allocate().map_err(|e| e.to_string())?;
    let t = if any { Some(transient.allocate().map_err(|e| e.to_string())?) } else { None };
    Ok((r, t, slots))
}

fn slice_of(r: &Arena, t: &Option<Arena>, (slot, transient): (Slot, bool)) -> DevSlice {
    if transient { t.as_ref().unwrap().get(slot) } else { r.get(slot) }
}

const PREFIX: &str = "model.llm.";

/// Row-concatenated attention projections, in output order: Q | R | gate | sink, and K | V.
const CONCAT_GROUPS: [(&str, &[&str]); 2] =
    [("qrgs", &["wq_du", "wr_du", "gate_attn", "attn_sink_proj"]), ("kv", &["wk_dv", "wv_dv"])];

/// Shape and dtype checks for the tensors whose layout the kernels depend on.
fn check_layer(f: &SafetensorsFile, l: usize) -> Result<(), String> {
    use dims::*;
    let p = |s: &str| format!("{PREFIX}layers.{l}.{s}");
    let bf = Dtype::BF16;
    for (name, shape) in [
        ("attn.wq_du.weight", vec![Q_DIM, HIDDEN]),
        ("attn.wr_du.weight", vec![R_DIM, HIDDEN]),
        ("attn.gate_attn.weight", vec![Q_DIM, HIDDEN]),
        ("attn.attn_sink_proj.weight", vec![HEADS, HIDDEN]),
        ("attn.wk_dv.weight", vec![KV_DIM, HIDDEN]),
        ("attn.wv_dv.weight", vec![KV_DIM, HIDDEN]),
        ("attn.wo_ud.weight", vec![HIDDEN, Q_DIM]),
        ("attn.q_l2_clip.scale", vec![HEADS, HEAD_DIM]),
        ("attn.rel_l2_clip.scale", vec![HEADS, REL_DIM]),
        ("attn.rel_logits_proj.proj", vec![REL_DIM, REL_EXTENT]),
        ("attn_sconv.weight", vec![HIDDEN, 1, CONV_TAPS]),
        ("mlp_sconv.weight", vec![HIDDEN, 1, CONV_TAPS]),
        ("mlp_norm.weight", vec![HIDDEN]),
        ("mlp.gate.weight", vec![EXPERTS, HIDDEN]),
    ] {
        f.expect(&p(name), bf, &shape)?;
    }
    f.expect(&p("mlp.gate.bias"), Dtype::F32, &[EXPERTS])?;
    if is_global_layer(l) {
        for n in ["q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"] {
            f.expect(&p(n), bf, &[HIDDEN])?;
        }
        for kv in ["k", "v"] {
            f.expect(&p(&format!("attn.{kv}_sconv.static_w")), bf, &[CONV_TAPS, KV_DIM])?;
            f.expect(&p(&format!("attn.{kv}_sconv.weight_proj")), bf, &[KV_HEADS * CONV_TAPS, HIDDEN])?;
        }
    } else {
        f.expect(&p("attn_norm.weight"), bf, &[HIDDEN])?;
        for kv in ["k", "v"] {
            f.expect(&p(&format!("attn.{kv}_sconv.weight")), bf, &[KV_DIM, 1, CONV_TAPS])?;
        }
    }
    if (LOOP_START..LOOP_END).contains(&l) {
        for c in ["attn", "k", "v", "mlp"] {
            f.expect(&p(&format!("controllers.{c}.down_weight")), bf, &[CONTROLLER_RANK, 2 * HIDDEN])?;
            f.expect(&p(&format!("controllers.{c}.forget_weight")), bf, &[HIDDEN, CONTROLLER_RANK])?;
            f.expect(&p(&format!("controllers.{c}.gain_weight")), bf, &[HIDDEN, CONTROLLER_RANK])?;
        }
    }
    let e = |s: &str| p(&format!("mlp.experts.{s}"));
    if l < NVFP4_EXPERT_LAYERS {
        f.expect(&e("w13_weight"), Dtype::U8, &[EXPERTS, 2 * INTERMEDIATE, HIDDEN / 2])?;
        f.expect(&e("w13_weight_scale"), Dtype::F8E4M3, &[EXPERTS, 2 * INTERMEDIATE, HIDDEN / 16])?;
        f.expect(&e("w2_weight"), Dtype::U8, &[EXPERTS, HIDDEN, INTERMEDIATE / 2])?;
        f.expect(&e("w2_weight_scale"), Dtype::F8E4M3, &[EXPERTS, HIDDEN, INTERMEDIATE / 16])?;
    } else {
        f.expect(&e("w13_weight"), Dtype::F8E4M3, &[EXPERTS, 2 * INTERMEDIATE, HIDDEN])?;
        f.expect(&e("w13_weight_scale"), Dtype::U8, &[EXPERTS, 2 * INTERMEDIATE, HIDDEN / 32])?;
        f.expect(&e("w2_weight"), Dtype::F8E4M3, &[EXPERTS, HIDDEN, INTERMEDIATE])?;
        f.expect(&e("w2_weight_scale"), Dtype::U8, &[EXPERTS, HIDDEN, INTERMEDIATE / 32])?;
    }
    Ok(())
}

fn is_derived_input(name: &str) -> bool {
    name.ends_with("_amax") || name.ends_with("_outer_scale_denominator")
}

/// Plans, allocates and loads the exact-mode weights from `model_dir`.
pub fn load_exact(model_dir: &Path, stream: &Stream, opts: LoadOptions) -> Result<Weights, String> {
    load_exact_split(model_dir, stream, opts, |_| false)
}

/// As `load_exact`, with the weights for which `transient(key)` holds in a second arena that
/// `Weights::free_transient` can release after the quantized copies are built.
pub fn load_exact_split(model_dir: &Path, stream: &Stream, opts: LoadOptions, transient: impl Fn(&str) -> bool) -> Result<Weights, String> {
    let path = model_dir.join("model-00001-of-00001.safetensors");
    let f = SafetensorsFile::open(&path)?;
    f.expect(&format!("{PREFIX}embed.weight"), Dtype::BF16, &[dims::VOCAB_PADDED, dims::HIDDEN])?;
    f.expect(&format!("{PREFIX}unembed.weight"), Dtype::BF16, &[dims::VOCAB_PADDED, dims::HIDDEN])?;
    for l in 0..dims::LAYERS {
        check_layer(&f, l)?;
    }

    // (key, bytes, transient)
    let mut reservations: Vec<(String, usize, bool)> = Vec::new();
    // (tensor, key index, byte offset inside the slot)
    let mut placements: Vec<(&TensorInfo, usize, usize)> = Vec::new();
    let mut host_uploads: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut reserve = |key: String, bytes: usize| -> usize {
        let t = transient(&key);
        reservations.push((key, bytes, t));
        reservations.len() - 1
    };

    let mut grouped: HashMap<String, ()> = HashMap::new();
    for l in 0..dims::LAYERS {
        for (group, parts) in CONCAT_GROUPS {
            let names: Vec<String> = parts.iter().map(|s| format!("{PREFIX}layers.{l}.attn.{s}.weight")).collect();
            let infos: Vec<&TensorInfo> = names.iter().map(|n| f.get(n).unwrap()).collect();
            let total: u64 = infos.iter().map(|t| t.bytes).sum();
            let k = reserve(format!("layers.{l}.attn.{group}"), total as usize);
            let mut off = 0usize;
            for (n, t) in names.iter().zip(&infos) {
                placements.push((t, k, off));
                off += t.bytes as usize;
                grouped.insert(n.clone(), ());
            }
        }
        for w in ["w13", "w2"] {
            let amax_name = format!("{PREFIX}layers.{l}.mlp.experts.{w}_weight_amax");
            let den_name = format!("{PREFIX}layers.{l}.mlp.experts.{w}_weight_outer_scale_denominator");
            if let (Some(amax), Some(den)) = (f.get(&amax_name), f.get(&den_name)) {
                let amax = f.read_f32(amax)?;
                let den = f.read_f32(den)?;
                if amax.len() != dims::EXPERTS || den.len() != 1 || den[0].is_nan() || den[0] <= 0.0 {
                    return Err(format!("layer {l} {w}: bad amax/denominator shapes or values"));
                }
                let scale: Vec<u8> = amax.iter().flat_map(|a| (a / den[0]).to_le_bytes()).collect();
                let k = reserve(format!("layers.{l}.mlp.experts.{w}_global_scale"), scale.len());
                host_uploads.push((k, scale));
            }
        }
    }
    for t in f.tensors() {
        let Some(short) = t.name.strip_prefix(PREFIX) else { continue }; // vision tower
        if grouped.contains_key(&t.name) || is_derived_input(&t.name) {
            continue;
        }
        let k = reserve(short.to_owned(), t.bytes as usize);
        placements.push((t, k, 0));
    }

    let (arena, trans, slots) = plan_split(&reservations)?;
    let at = |k: usize| slice_of(&arena, &trans, slots[&reservations[k].0]);
    let copies: Vec<FileCopy> =
        placements.iter().map(|&(t, k, off)| FileCopy { offset: t.offset, dst: at(k).sub(off, t.bytes as usize) }).collect();
    for (k, bytes) in &host_uploads {
        at(*k).upload(bytes).map_err(|e| e.to_string())?;
    }
    let stats = stream_to_device(&path, &copies, stream, opts)?;
    Ok(Weights::new(arena, trans, slots, stats))
}

/// DSpark drafter layers in `mtp.safetensors`.
pub const MTP_LAYERS: usize = 3;
pub const MTP_INTERMEDIATE: usize = 9216;
pub const MTP_MARKOV_RANK: usize = 256;

/// Row-concatenated DSpark projections (squid5090 `load_dspark`): `qkv` = q|k|v [5120, 2304],
/// `gu` = gate|up [18432, 2304], `kv_ctx` = k_ctx|v_ctx [2048, 2304].
const MTP_GROUPS: [(&str, &[&str]); 3] = [
    ("qkv", &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    ("gu", &["mlp.gate_proj", "mlp.up_proj"]),
    ("kv_ctx", &["self_attn.k_proj_ctx", "self_attn.v_proj_ctx"]),
];

fn check_mtp(f: &SafetensorsFile) -> Result<(), String> {
    use dims::*;
    let bf = Dtype::BF16;
    f.expect("fuse_weight", bf, &[4, 4])?;
    for (n, shape) in [
        ("mask_embed", vec![HIDDEN]),
        ("norm.weight", vec![HIDDEN]),
        ("markov_head.dflare_norm.weight", vec![HIDDEN]),
        ("markov_head.hidden_out.bias", vec![HIDDEN]),
        ("markov_head.hidden_out.weight", vec![HIDDEN, MTP_MARKOV_RANK]),
        ("markov_head.markov_w1.weight", vec![VOCAB_PADDED, MTP_MARKOV_RANK]),
        ("markov_head.rnn_proj.bias", vec![3 * MTP_MARKOV_RANK]),
        ("markov_head.rnn_proj.weight", vec![3 * MTP_MARKOV_RANK, 2 * MTP_MARKOV_RANK + 2 * HIDDEN]),
    ] {
        f.expect(n, bf, &shape)?;
    }
    for l in 0..MTP_LAYERS {
        let p = |s: &str| format!("layers.{l}.{s}");
        for (n, shape) in [
            ("dflare_norm.weight", vec![HIDDEN]),
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
            ("self_attn.q_norm.weight", vec![HEAD_DIM]),
            ("self_attn.k_norm.weight", vec![HEAD_DIM]),
            ("self_attn.k_norm_ctx.weight", vec![HEAD_DIM]),
            ("self_attn.q_proj.weight", vec![Q_DIM, HIDDEN]),
            ("self_attn.k_proj.weight", vec![KV_DIM, HIDDEN]),
            ("self_attn.v_proj.weight", vec![KV_DIM, HIDDEN]),
            ("self_attn.k_proj_ctx.weight", vec![KV_DIM, HIDDEN]),
            ("self_attn.v_proj_ctx.weight", vec![KV_DIM, HIDDEN]),
            ("self_attn.o_proj.weight", vec![HIDDEN, Q_DIM]),
            ("mlp.gate_proj.weight", vec![MTP_INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", vec![MTP_INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, MTP_INTERMEDIATE]),
        ] {
            f.expect(&p(n), bf, &shape)?;
        }
    }
    Ok(())
}

/// Loads the DSpark drafter (`mtp.safetensors`, all BF16) into its own arena. Names are as in the file,
/// plus the concatenated `layers.{l}.{qkv,gu,kv_ctx}`; the concatenated parts are not stored separately.
pub fn load_mtp(model_dir: &Path, stream: &Stream, opts: LoadOptions) -> Result<Weights, String> {
    load_mtp_split(model_dir, stream, opts, |_| false)
}

/// As `load_mtp`, with the weights for which `transient(key)` holds in a freeable second arena.
pub fn load_mtp_split(model_dir: &Path, stream: &Stream, opts: LoadOptions, transient: impl Fn(&str) -> bool) -> Result<Weights, String> {
    let path = model_dir.join("mtp.safetensors");
    let f = SafetensorsFile::open(&path)?;
    check_mtp(&f)?;
    let mut reservations: Vec<(String, usize, bool)> = Vec::new();
    let mut placements: Vec<(&TensorInfo, usize, usize)> = Vec::new();
    let mut reserve = |key: String, bytes: usize| -> usize {
        let t = transient(&key);
        reservations.push((key, bytes, t));
        reservations.len() - 1
    };
    let mut grouped: HashMap<String, ()> = HashMap::new();
    for l in 0..MTP_LAYERS {
        for (group, parts) in MTP_GROUPS {
            let names: Vec<String> = parts.iter().map(|s| format!("layers.{l}.{s}.weight")).collect();
            let infos: Vec<&TensorInfo> = names.iter().map(|n| f.get(n).unwrap()).collect();
            let total: u64 = infos.iter().map(|t| t.bytes).sum();
            let k = reserve(format!("layers.{l}.{group}"), total as usize);
            let mut off = 0usize;
            for (n, t) in names.iter().zip(&infos) {
                placements.push((t, k, off));
                off += t.bytes as usize;
                grouped.insert(n.clone(), ());
            }
        }
    }
    for t in f.tensors() {
        if grouped.contains_key(&t.name) {
            continue;
        }
        let k = reserve(t.name.clone(), t.bytes as usize);
        placements.push((t, k, 0));
    }
    let (arena, trans, slots) = plan_split(&reservations)?;
    let at = |k: usize| slice_of(&arena, &trans, slots[&reservations[k].0]);
    let copies: Vec<FileCopy> =
        placements.iter().map(|&(t, k, off)| FileCopy { offset: t.offset, dst: at(k).sub(off, t.bytes as usize) }).collect();
    let stats = stream_to_device(&path, &copies, stream, opts)?;
    Ok(Weights::new(arena, trans, slots, stats))
}

/// Reads back up to `window` bytes at the start, middle and end of every loaded tensor and compares them
/// with the file. Returns the number of bytes compared.
pub fn verify_readback(model_dir: &Path, w: &Weights, window: usize) -> Result<u64, String> {
    let f = SafetensorsFile::open(&model_dir.join("model-00001-of-00001.safetensors"))?;
    let mut compared = 0u64;
    let mut place = |t: &TensorInfo, dst: DevSlice| -> Result<(), String> {
        let n = t.bytes as usize;
        if n == 0 {
            return Ok(());
        }
        let w = window.min(n);
        for start in [0, n / 2 - (n / 2).min(w / 2), n - w] {
            let mut file_bytes = vec![0u8; w];
            f.read_at(t.offset + start as u64, &mut file_bytes).map_err(|e| e.to_string())?;
            let mut dev_bytes = vec![0u8; w];
            dst.download_bytes_at(start, &mut dev_bytes).map_err(|e| e.to_string())?;
            if file_bytes != dev_bytes {
                return Err(format!("{}: device bytes differ from the file at +{start}", t.name));
            }
            compared += w as u64;
        }
        Ok(())
    };
    for l in 0..dims::LAYERS {
        for (group, parts) in CONCAT_GROUPS {
            let base = w.get(&format!("layers.{l}.attn.{group}"));
            let mut off = 0usize;
            for p in parts {
                let t = f.get(&format!("{PREFIX}layers.{l}.attn.{p}.weight")).unwrap();
                place(t, base.sub(off, t.bytes as usize))?;
                off += t.bytes as usize;
            }
        }
    }
    for t in f.tensors() {
        let Some(short) = t.name.strip_prefix(PREFIX) else { continue };
        if let Some(dst) = w.try_get(short) {
            place(t, dst)?;
        }
    }
    Ok(compared)
}
