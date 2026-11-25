//! Wide prefill: prompt chunks of T = 16-256 tokens per step, in the shippable fp8 mode's precisions
//! except the dense projections and the router, which run as cuBLAS BF16 GEMMs on the checkpoint's BF16
//! weights (a stand-in until the FP8 prefill GEMMs exist).
//!
//! Per visit, the chunk runs the decode path's per-token kernels with T rows (grid over tokens), copies
//! of the short-conv kernels that only read the rings (`prefill_*` in `prefill_attn.rs` /
//! `prefill_mega.rs`; `prefill_ring_store` writes the chunk's last ring slots afterwards), attention as
//! T / 8 sub-steps of the B = 8 verify kernels (the chunk's K/V are all stored first, so each sub-step sees
//! the keys before it), and the MoE grouped by expert (`prefill_moe.rs`). No LM head: the body of the
//! prompt needs no logits; its last token runs through the B = 1 graph as before.
//!
//! The decode and verify graphs are untouched. Each chunk size has its own captured graph; the host sets
//! `d_pos` (the chunk's first position) and the tokens before each launch, as `prefill_chunked` does.

use crate::cublas::Blas;
use crate::dspark::Dspark;
use crate::forward::Forward;
use crate::nvfp4::{Nvfp4Weights, Packed};
use crate::state::State;
use inkling_model::weights::Weights;
use inkling_model::{SCHEDULE, dims, is_global_layer};
use inkling_runtime::{Args, Arena, ArenaPlan, GraphExec, Kernel, KernelArg, LaunchConfig, Module, PinnedBuffer, PtrPack, Slot, Stream, capture};

const H: usize = dims::HIDDEN;
const NULL: u64 = 0;
/// Chunk sizes with a captured graph, largest first (INKLING_WIDE_MAX caps them, default 2048; 1024 restores the
/// earlier buffers, about 0.37 GiB less).
const CHUNKS: [usize; 8] = [2048, 1024, 512, 256, 128, 64, 32, 16];
/// Tokens per fused global attention launch (see the D + E loop).
const FUSE_MAX: usize = 1024;
/// Dynamic shared memory of the `prefill_gemm.rs` kernels (`GEMM_TCX_SMEM`: two half-step stages and the
/// decoded weights).
const DENSE_TC_SMEM: u32 = (2 * (128 * 72 * 2 + 128 * 80) + 128 * 72 * 2) as u32;
const T_MAX: usize = 2048;
/// Tokens per K/V-store + attention slice of a global visit (local visits: `state::local_slice()`).
const SLICE: usize = 256;
const WORKSPACE: usize = 32 << 20;
/// Dynamic shared memory of `prefill_moe_{up,down}_mma` (`MOE_MMA_SMEM`: two (128 + 64) x 128 f16 stages).
/// Dynamic shared memory of `prefill_attn_flash2` (`FLASH2_SMEM`: two 32-key K/V stages, the R rows).
const FLASH2_SMEM: u32 = (2 * 2 * 32 * 136 * 2 + 96 * 16 * 2) as u32;
/// Most key splits per global-layer query block (INKLING_WIDE_SPLITS, default 8).
const MAX_SPLITS: usize = 8;
/// Dynamic shared memory of `prefill_attn_flash3` (`FLASH3_SMEM`: two 64-key K/V stages).
const FLASH3_SMEM: u32 = (2 * 2 * 64 * 136 * 2 + 2 * 64 * 4 * 2) as u32;
/// Dynamic shared memory of `prefill_attn_flash4` (`FLASH4_SMEM`: two 80-row K/V windows, scales, FP8 band).
/// Dynamic shared memory of `prefill_attn_flash5` (`FLASH5_SMEM`: one 64-key K/V stage).
const FLASH5_SMEM: u32 = (2 * 64 * 136 * 2 + 64 * 4 * 2) as u32;
const FLASH4_SMEM: u32 = (2 * 2 * 80 * 136 * 2 + 2 * 80 * 4 * 2 + 2 * 16 * 136 * 2) as u32;

fn global_splits() -> i32 {
    std::env::var("INKLING_WIDE_SPLITS").ok().and_then(|v| v.parse().ok()).unwrap_or(MAX_SPLITS).clamp(1, MAX_SPLITS) as i32
}

const MOE_MMA_SMEM: u32 = 2 * (128 + 64) * 128 * 2;
/// Dynamic shared memory of `prefill_moe_{up,down}_bs` (`MOE_BS_SMEM`: two stages of 64 + 128 code rows and scales).
const MOE_BS_SMEM: u32 = 2 * ((64 + 128) * 80 + (64 + 128) * 8);
/// Visits whose updated hidden state feeds the drafter (forward.rs `TAP_VISITS`).
const TAP_VISITS: [usize; 3] = [5, 23, 41];

/// `CtrlNVFP4Ptr3` (192 B), as in forward.rs.
#[derive(Clone, Copy)]
struct CtrlNvfp4([[Packed; 3]; 3]);

impl KernelArg for CtrlNvfp4 {
    const SIZE: usize = 192;
    fn write(&self, out: &mut [u8]) {
        for (m, streams) in self.0.iter().enumerate() {
            let base = 64 * m;
            for (s, p) in streams.iter().enumerate() {
                out[base + 8 * s..base + 8 * s + 8].copy_from_slice(&p.data.ptr.to_ne_bytes());
                out[base + 24 + 8 * s..base + 32 + 8 * s].copy_from_slice(&p.scales.ptr.to_ne_bytes());
                out[base + 48 + 4 * s..base + 52 + 4 * s].copy_from_slice(&p.gs.to_ne_bytes());
            }
        }
    }
}

struct K {
    embed_norm: Kernel,
    rmsnorm: Kernel,
    extract_block: Kernel,
    fused_depth_norm: Kernel,
    controller: Kernel,
    bulk8: Kernel,
    bulk8_fp8: Kernel,
    tail8: Kernel,
    merge: Kernel,
    post_next: Kernel,
    attn_post: Kernel,
    post_res: Kernel,
    local_prep: Kernel,
    global_prep: Kernel,
    ring_store: Kernel,
    to_bf16: Kernel,
    sub_pos: Kernel,
    route_select: Kernel,
    moe_group: Kernel,
    moe_up: Kernel,
    moe_down: Kernel,
    /// `prefill_attn_flash` (one launch per visit), unless INKLING_WIDE_FLASH=0 (then T / 8 verify sub-steps).
    flash: Option<Kernel>,
    /// `prefill_attn_flash2` + `prefill_attn_merge2`, opt-in with INKLING_WIDE_FLASH=2 (it adds two hard
    /// flips over v1 on gate-long with FP8 KV); INKLING_WIDE_FLASH=0 selects the B = 8 verify sub-steps.
    flash2: Option<[Kernel; 2]>,
    /// `prefill_attn_flash3` (v1's arithmetic and order with cp.async staging; bit-identical to v1), the
    /// default (INKLING_WIDE_FLASH=1 selects v1); `prefill_rel16` builds the f16 rel tables; `prefill_cmp_h` is the shadow check.
    flash3: Option<Kernel>,
    rel16_k: Option<Kernel>,
    cmp_h: Option<Kernel>,
    flash3_any: Option<Kernel>,
    /// `prefill_attn_flash4`: flash3 with two query blocks per CTA sharing each staged K/V window
    /// (bit-identical to v1), selected with INKLING_WIDE_FLASH=4.
    flash4: Option<Kernel>,
    flash4_any: Option<Kernel>,
    /// `prefill_attn_flash5`: flash3 with one K/V stage so two CTAs fit per SM (bit-identical to v1),
    /// selected with INKLING_WIDE_FLASH=5.
    flash5: Option<Kernel>,
    flash5_any: Option<Kernel>,
    shadow5: bool,
    /// INKLING_WIDE_SHADOW=3 (or 4): run flash3 (or flash4) beside v1 and count mismatching outputs (global / local visits).
    shadow: bool,
    shadow4: bool,
    /// FP8 dense prefill GEMMs (`prefill_dense.rs`, MT = 1, 2, 4), unless INKLING_WIDE_DENSE=bf16.
    qkv_fp8: Option<[Kernel; 3]>,
    o_fp8: Option<[Kernel; 3]>,
    /// `qkvrg_fp8_prefill_tcx` + `linear_fp8_prefill_k3072_tcx` (`prefill_gemm.rs`): the MT = 4 split-K
    /// kernels' arithmetic in 128 x 128 tiles, bit-identical; used for chunks above 32 tokens unless
    /// INKLING_WIDE_DENSE_TC=0.
    dense_tc: Option<[Kernel; 2]>,
    /// Tiled grouped MoE (`prefill_moe_mma.rs`), only with INKLING_WIDE_MOE=mma: at T <= 1024 the
    /// decode-structured grouped kernels (`prefill_moe.rs`) measured faster (2K 0.37 vs 0.46 s, 8K 1.28 vs 1.46 s).
    moe_mma: Option<[Kernel; 3]>,
    /// Block-scaled FP4 tensor-core MoE (`prefill_moe_bs.rs`: tiles, activation quantizer, up, down), only with
    /// INKLING_WIDE_MOE=bs: it quantizes the activations to NVFP4, so it changes the arithmetic.
    moe_bs: Option<[Kernel; 4]>,
    /// `prefill_route_logits_nvfp4` (the decode router's arithmetic), unless INKLING_WIDE_ROUTER=bf16.
    router8: Option<Kernel>,
}

struct Bufs {
    hidden: u64,
    prelude: u64,
    partial: u64,
    saved: u64,
    prior_last: u64,
    residual: u64,
    o_proj: u64,
    mlp_in: u64,
    mlp_normed: u64,
    mlp_normed_h: u64,
    blocks: u64,
    attn_in: [u64; 3],
    attn_in_h: [u64; 3],
    xb: [u64; 3],
    qrgs: u64,
    k_proj: u64,
    v_proj: u64,
    attn_gated: u64,
    router_logits: u64,
    route_ids: u64,
    route_w: u64,
    offs: u64,
    pairs: u64,
    act: u64,
    slots: u64,
    tokens: u64,
    sub_pos: u64,
    /// DSpark taps, [4][T][HIDDEN] for a chunk of T (visits 5, 23, 41 and the final normed state).
    taps: u64,
    attn_gated_h: u64,
    tiles: u64,
    tiles_n: u64,
    attn_part: u64,
    rel16: u64,
    shadow_h: u64,
    counters: u64,
    /// NVFP4 activations of the block-scaled MoE: codes ([8 t][INTERMEDIATE / 2] bytes at most) and scales.
    qa_codes: u64,
    qa_scales: u64,
}

pub struct WidePrefill<'a> {
    st: &'a State,
    q: &'a Nvfp4Weights,
    w: &'a Weights,
    k: K,
    b: Bufs,
    arena: Arena,
    blas: Blas,
    graphs: Vec<(usize, GraphExec)>,
    tokens_slice: inkling_runtime::DevSlice,
    stream: &'a Stream,
    dspark: Option<&'a Dspark<'a>>,
    /// Per layer: FP8 (data, scales) of QRGS, K, V, O.
    fp8: Vec<[u64; 8]>,
    /// (start, size) of the last wide chunk launched (for `flash_parity`).
    last_wide: std::cell::Cell<(usize, usize)>,
}

/// Prompts shorter than this (INKLING_WIDE_MIN, default 256 tokens) keep trunk's B <= 8 chunked prefill:
/// they gain little from wide chunks (512 tokens: 0.46 -> 0.14 s), and the wide path's different summation
/// order flipped one short held-out case (essay, 39 tokens).
fn wide_min() -> usize {
    std::env::var("INKLING_WIDE_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(256)
}

/// f32 to IEEE half bits, round to nearest even (finite, normal-range inputs such as rel_proj).
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - exp);
        let r = (m + 0xfff + ((m >> 13) & 1)) >> 13;
        return sign | r as u16;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    let r = (man + 0xfff + ((man >> 13) & 1)) >> 13;
    let v = ((exp as u32) << 10) + r;
    sign | v.min(0x7c00) as u16
}

/// The flash3 kernel for the shadow check, whether or not it is selected.
fn m3(k: &K) -> Option<Kernel> {
    k.flash3_any
}

/// One attention launch per global visit and chunk instead of per 256-token slice (default; INKLING_WIDE_GFUSE=0
/// restores the per-slice launches).
fn gfuse() -> bool {
    std::env::var("INKLING_WIDE_GFUSE").map_or(true, |v| v != "0")
}

/// The grouped MoE kernels with two row tiles per CTA (default; INKLING_WIDE_MOE_R2C=0 disables).
fn moe_r2c() -> bool {
    std::env::var("INKLING_WIDE_MOE_R2C").map_or(true, |v| v != "0")
}

/// Tokens each controller cluster runs in turn in a chunk of `t` (bit-identical to one cluster per token, with
/// the weights loaded once per cluster): INKLING_WIDE_CTRL_TB, default 16, capped so that at least 128 clusters
/// run at once; 0 selects the decode kernel with one cluster per token.
fn ctrl_tb(t: usize) -> usize {
    let tb = std::env::var("INKLING_WIDE_CTRL_TB").ok().and_then(|v| v.parse().ok()).unwrap_or(16usize);
    if tb == 0 { 0 } else { tb.min(t / 128).max(1) }
}

fn chunk_sizes() -> Vec<usize> {
    let max = std::env::var("INKLING_WIDE_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(T_MAX);
    CHUNKS.iter().copied().filter(|&t| t <= max).collect()
}

fn ptr<T>(arena: &Arena, s: Slot) -> u64 {
    let _ = std::marker::PhantomData::<T>;
    arena.get(s).ptr
}

impl<'a> WidePrefill<'a> {
    /// Resolves the kernels, allocates the chunk buffers and captures one graph per chunk size. Needs
    /// the fp8 / nvfp4 mode with NVFP4 experts and controllers (`fwd` is only checked for that).
    pub fn new(
        m: &Module,
        w: &'a Weights,
        q: &'a Nvfp4Weights,
        fwd: &Forward,
        st: &'a State,
        stream: &'a Stream,
        dspark: Option<&'a Dspark<'a>>,
    ) -> Result<WidePrefill<'a>, String> {
        if !(fwd.parts.experts && fwd.parts.ctrl) {
            return Err("wide prefill needs NVFP4 experts and controllers (--mode fp8 or nvfp4)".into());
        }
        if std::env::var_os("INKLING_WIDE_PREFILL").is_some_and(|v| v == "0") {
            return Err("INKLING_WIDE_PREFILL=0".into());
        }
        let f = |n: &str| m.function(n).ok_or_else(|| format!("kernel {n} missing from the cubin"));
        let k = K {
            embed_norm: f("embed_norm")?,
            rmsnorm: f("rmsnorm")?,
            extract_block: f("extract_block")?,
            fused_depth_norm: f("fused_depth_norm")?,
            controller: if ctrl_tb(T_MAX) > 0 { f("prefill_controller_nvfp4_cl8")? } else { f("mega_controller_nvfp4_cl8")? },
            bulk8: f("global_attn_bulk_splitk_bf16kv_b8")?,
            bulk8_fp8: f("global_attn_bulk_splitk_fp8kv_b8")?,
            tail8: f("local_tail_attn_tc_bf16kv_b8")?,
            merge: f("attn_merge_sink_gate")?,
            post_next: f("prefill_mlp_post_next_depth_norm")?,
            attn_post: f("prefill_attn_post_mlp_in")?,
            post_res: f("prefill_mlp_post_res")?,
            local_prep: f("prefill_local_sconv_prep_store")?,
            global_prep: f("prefill_global_sconv_prep_store_bf16kv")?,
            ring_store: f("prefill_ring_store")?,
            to_bf16: f("prefill_f32_to_bf16")?,
            sub_pos: f("prefill_sub_pos")?,
            route_select: f("prefill_route_select")?,
            moe_group: f("prefill_moe_group")?,
            // `_r2c`: two row tiles per CTA sharing each activation load, bit-identical (INKLING_WIDE_MOE_R2C=0
            // selects the one-tile kernels).
            moe_up: if moe_r2c() { f("prefill_moe_up_nvfp4_r2c")? } else { f("prefill_moe_up_nvfp4")? },
            moe_down: if moe_r2c() { f("prefill_moe_down_nvfp4_r2c")? } else { f("prefill_moe_down_nvfp4")? },
            qkv_fp8: (|| Some([m.function("qkvrg_fp8_prefill_mt1")?, m.function("qkvrg_fp8_prefill_mt2")?, m.function("qkvrg_fp8_prefill_mt4")?]))()
                .filter(|_| fwd.fp8.is_some() && std::env::var("INKLING_WIDE_DENSE").map_or(true, |v| v != "bf16")),
            dense_tc: (|| Some([m.function("qkvrg_fp8_prefill_tcx")?, m.function("linear_fp8_prefill_k3072_tcx")?]))()
                .filter(|_| std::env::var("INKLING_WIDE_DENSE_TC").map_or(true, |v| v != "0")),
            o_fp8: (|| Some([m.function("linear_fp8_prefill_k3072_mt1")?, m.function("linear_fp8_prefill_k3072_mt2")?, m.function("linear_fp8_prefill_k3072_mt4")?]))(),
            moe_mma: (|| Some([m.function("prefill_moe_tiles")?, m.function("prefill_moe_up_mma")?, m.function("prefill_moe_down_mma")?]))()
                .filter(|_| std::env::var("INKLING_WIDE_MOE").is_ok_and(|v| v == "mma")),
            moe_bs: (|| Some([m.function("prefill_moe_tiles")?, m.function("prefill_quant_f16_to_nvfp4")?, m.function("prefill_moe_up_bs")?, m.function("prefill_moe_down_bs")?]))()
                .filter(|_| std::env::var("INKLING_WIDE_MOE").is_ok_and(|v| v == "bs")),
            router8: m.function("prefill_route_logits_nvfp4").filter(|_| fwd.parts.router && std::env::var("INKLING_WIDE_ROUTER").map_or(true, |v| v != "bf16")),
            flash2: (|| Some([m.function("prefill_attn_flash2")?, m.function("prefill_attn_merge2")?]))()
                .filter(|_| std::env::var("INKLING_WIDE_FLASH").is_ok_and(|v| v == "2")),
            flash3: m.function("prefill_attn_flash3").filter(|_| std::env::var("INKLING_WIDE_FLASH").map_or(true, |v| v == "3")),
            rel16_k: m.function("prefill_rel16"),
            cmp_h: m.function("prefill_cmp_h"),
            flash3_any: m.function("prefill_attn_flash3"),
            flash4: m.function("prefill_attn_flash4").filter(|_| std::env::var("INKLING_WIDE_FLASH").is_ok_and(|v| v == "4")),
            flash4_any: m.function("prefill_attn_flash4"),
            flash5: m.function("prefill_attn_flash5").filter(|_| std::env::var("INKLING_WIDE_FLASH").is_ok_and(|v| v == "5")),
            flash5_any: m.function("prefill_attn_flash5"),
            shadow5: std::env::var("INKLING_WIDE_SHADOW").is_ok_and(|v| v == "5"),
            shadow: std::env::var("INKLING_WIDE_SHADOW").is_ok_and(|v| v == "3" || v == "4" || v == "5"),
            shadow4: std::env::var("INKLING_WIDE_SHADOW").is_ok_and(|v| v == "4"),
            flash: m.function("prefill_attn_flash").filter(|_| std::env::var_os("INKLING_WIDE_FLASH").is_none_or(|v| v != "0")),
        };
        if let Some([f2, _]) = k.flash2 {
            f2.set_max_dynamic_shared(FLASH2_SMEM).map_err(|e| e.to_string())?;
        }
        if let Some(f3) = m.function("prefill_attn_flash3") {
            f3.set_max_dynamic_shared(FLASH3_SMEM).map_err(|e| e.to_string())?;
        }
        for name in ["qkvrg_fp8_prefill_tcx", "linear_fp8_prefill_k3072_tcx"] {
            if let Some(f) = m.function(name) {
                f.set_max_dynamic_shared(DENSE_TC_SMEM).map_err(|e| e.to_string())?;
            }
        }
        if let Some(f5) = m.function("prefill_attn_flash5") {
            f5.set_max_dynamic_shared(FLASH5_SMEM).map_err(|e| e.to_string())?;
        }
        if let Some(f4) = m.function("prefill_attn_flash4") {
            f4.set_max_dynamic_shared(FLASH4_SMEM).map_err(|e| e.to_string())?;
        }
        if let Some([_, _, ku, kd]) = k.moe_bs {
            ku.set_max_dynamic_shared(MOE_BS_SMEM).map_err(|e| e.to_string())?;
            kd.set_max_dynamic_shared(MOE_BS_SMEM).map_err(|e| e.to_string())?;
        }
        if let Some([_, ku, kd]) = k.moe_mma {
            ku.set_max_dynamic_shared(MOE_MMA_SMEM).map_err(|e| e.to_string())?;
            kd.set_max_dynamic_shared(MOE_MMA_SMEM).map_err(|e| e.to_string())?;
        }
        let t = chunk_sizes()[0];
        let mut p = ArenaPlan::new();
        let fh = |p: &mut ArenaPlan, n: &str| p.reserve_of::<f32>(n, t * H);
        let s_hidden = fh(&mut p, "hidden");
        let s_prelude = fh(&mut p, "prelude");
        let s_partial = fh(&mut p, "partial");
        let s_saved = fh(&mut p, "saved");
        let s_prior = fh(&mut p, "prior_last");
        let s_residual = fh(&mut p, "residual");
        let s_oproj = fh(&mut p, "o_proj");
        let s_mlp_in = fh(&mut p, "mlp_in");
        let s_mlp_normed = fh(&mut p, "mlp_normed");
        let s_mlp_normed_h = p.reserve_of::<u16>("mlp_normed_h", t * H);
        let s_blocks = p.reserve_of::<f32>("blocks", t * 4 * H);
        let s_attn_in = [fh(&mut p, "attn_in0"), fh(&mut p, "attn_in1"), fh(&mut p, "attn_in2")];
        let s_attn_in_h = [0, 1, 2].map(|i| p.reserve_of::<u16>(format!("attn_in_h{i}"), t * H));
        let s_xb = [0, 1, 2].map(|i| p.reserve_of::<u16>(format!("xb{i}"), t * dims::Q_DIM));
        let s_qrgs = p.reserve_of::<f32>("qrgs", t * dims::QRGS_ROWS);
        let s_k = p.reserve_of::<f32>("k_proj", t * dims::KV_DIM);
        let s_v = p.reserve_of::<f32>("v_proj", t * dims::KV_DIM);
        let s_gated = p.reserve_of::<f32>("attn_gated", t * dims::Q_DIM);
        let s_rl = p.reserve_of::<f32>("router_logits", t * dims::EXPERTS);
        let s_ids = p.reserve_of::<i32>("route_ids", t * dims::TOP_K);
        let s_rw = p.reserve_of::<f32>("route_w", t * dims::TOP_K);
        let s_offs = p.reserve_of::<i32>("offs", dims::EXPERTS + 1);
        let s_pairs = p.reserve_of::<i32>("pairs", t * dims::TOP_K);
        let s_act = p.reserve_of::<u16>("act", t * dims::TOP_K * dims::INTERMEDIATE);
        let s_slots = p.reserve_of::<f32>("slots", t * dims::TOP_K * H);
        let s_tokens = p.reserve_of::<i32>("tokens", t);
        let s_sub = p.reserve_of::<i32>("sub_pos", t / 8);
        let s_taps = p.reserve_of::<f32>("taps", 4 * t * H);
        let s_gh = p.reserve_of::<u16>("attn_gated_h", t * dims::Q_DIM);
        let s_tiles = p.reserve_of::<i32>("tiles", 2 * (t * dims::TOP_K / 64 + dims::EXPERTS));
        let s_tiles_n = p.reserve_of::<i32>("tiles_n", 4);
        // prefill_attn_flash2 key-split partials: (m, d) for up to 1024 tokens x 24 heads x 16 splits,
        // then the o rows for up to MAX_SPLITS splits.
        // Per-layer relative-bias tables for prefill_attn_flash2: rel_proj [16][1024] BF16 as f16 [1024][16].
        let s_rel16 = p.reserve_of::<u16>("rel16", dims::LAYERS * 1024 * 16);
        let s_shadow = p.reserve_of::<u16>("shadow_h", t * dims::Q_DIM);
        let s_counters = p.reserve_of::<u32>("shadow_counters", 4);
        let s_part = p.reserve_of::<f32>("attn_part", 2 * 1024 * dims::HEADS * 16 + 1024 * dims::HEADS * MAX_SPLITS * dims::HEAD_DIM);
        let s_qc = p.reserve_aligned("qa_codes", if k.moe_bs.is_some() { t * dims::TOP_K * dims::INTERMEDIATE / 2 } else { 16 }, 256);
        let s_qs = p.reserve_aligned("qa_scales", if k.moe_bs.is_some() { t * dims::TOP_K * dims::INTERMEDIATE / 16 } else { 16 }, 256);
        let s_ws = p.reserve_aligned("cublas_ws", WORKSPACE, 256);
        let arena = p.allocate().map_err(|e| e.to_string())?;
        let g = |s| ptr::<u8>(&arena, s);
        let b = Bufs {
            hidden: g(s_hidden),
            prelude: g(s_prelude),
            partial: g(s_partial),
            saved: g(s_saved),
            prior_last: g(s_prior),
            residual: g(s_residual),
            o_proj: g(s_oproj),
            mlp_in: g(s_mlp_in),
            mlp_normed: g(s_mlp_normed),
            mlp_normed_h: g(s_mlp_normed_h),
            blocks: g(s_blocks),
            attn_in: s_attn_in.map(g),
            attn_in_h: s_attn_in_h.map(g),
            xb: s_xb.map(g),
            qrgs: g(s_qrgs),
            k_proj: g(s_k),
            v_proj: g(s_v),
            attn_gated: g(s_gated),
            router_logits: g(s_rl),
            route_ids: g(s_ids),
            route_w: g(s_rw),
            offs: g(s_offs),
            pairs: g(s_pairs),
            act: g(s_act),
            slots: g(s_slots),
            tokens: g(s_tokens),
            sub_pos: g(s_sub),
            taps: g(s_taps),
            attn_gated_h: g(s_gh),
            tiles: g(s_tiles),
            tiles_n: g(s_tiles_n),
            attn_part: g(s_part),
            rel16: g(s_rel16),
            shadow_h: g(s_shadow),
            counters: g(s_counters),
            qa_codes: g(s_qc),
            qa_scales: g(s_qs),
        };
        if let Some(kr) = k.rel16_k {
            for l in 0..dims::LAYERS {
                let mut a = Args::new();
                a.push(w.get(&format!("layers.{l}.attn.rel_logits_proj.proj")).ptr).push(arena.get(s_rel16).ptr + (l * 1024 * 16 * 2) as u64);
                let cfg = LaunchConfig::new((64u32, 1u32), 256u32);
                unsafe { kr.launch(&cfg, stream, &mut a) }.map_err(|e| e.to_string())?;
            }
            arena.get(s_counters).fill_u8(0, stream).map_err(|e| e.to_string())?;
            stream.synchronize().map_err(|e| e.to_string())?;
        } else {
            let mut host = vec![0u16; dims::LAYERS * 1024 * 16];
            for l in 0..dims::LAYERS {
                let src: Vec<u16> = w.get(&format!("layers.{l}.attn.rel_logits_proj.proj")).download(16 * 1024)?;
                for k in 0..16 {
                    for d in 0..1024 {
                        host[(l * 1024 + d) * 16 + k] = f32_to_f16_bits(f32::from_bits((src[k * 1024 + d] as u32) << 16));
                    }
                }
            }
            arena.get(s_rel16).upload(&host).map_err(|e| e.to_string())?;
        }
        let blas = Blas::new(stream.raw(), g(s_ws), WORKSPACE)?;
        let tokens_slice = arena.get(s_tokens);
        let fp8 = match (&fwd.fp8, k.qkv_fp8) {
            (Some(f), Some(_)) => f
                .layers
                .iter()
                .map(|l| {
                    let (wk, wv) = (l.kv.rows(0, dims::KV_DIM), l.kv.rows(dims::KV_DIM, dims::KV_DIM));
                    [l.qrgs.data.ptr, l.qrgs.scales.ptr, wk.data.ptr, wk.scales.ptr, wv.data.ptr, wv.scales.ptr, l.wo.data.ptr, l.wo.scales.ptr]
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut wp = WidePrefill { st, q, w, k, b, arena, blas, graphs: Vec::new(), tokens_slice, stream, dspark, fp8, last_wide: std::cell::Cell::new((0, 0)) };
        for t in chunk_sizes() {
            if std::env::var_os("INKLING_WIDE_EAGER").is_none() {
                let gexec = capture::<String>(stream, |s| wp.record(s, t))?;
                gexec.upload(stream).map_err(|e| e.to_string())?;
                wp.graphs.push((t, gexec));
            }
        }
        Ok(wp)
    }

    pub fn chunk_sizes(&self) -> Vec<usize> {
        chunk_sizes()
    }

    pub fn total_bytes(&self) -> usize {
        self.arena.total_bytes()
    }

    fn go(&self, s: &Stream, k: Kernel, grid: (u32, u32), block: u32, mut args: Args) -> Result<(), String> {
        let cfg = LaunchConfig::new(grid, block);
        unsafe { k.launch(&cfg, s, &mut args) }.map_err(|e| e.to_string())
    }

    fn go_smem(&self, s: &Stream, k: Kernel, grid: impl Into<inkling_runtime::launch::Dim>, block: u32, smem: u32, mut args: Args) -> Result<(), String> {
        let cfg = LaunchConfig::new(grid, block).shared(smem);
        unsafe { k.launch(&cfg, s, &mut args) }.map_err(|e| e.to_string())
    }

    /// The controller for `t` tokens and `streams` streams: clusters of 8 CTAs that each run `ctrl_tb(t)` tokens
    /// in turn, or one cluster per token with INKLING_WIDE_CTRL_TB=0.
    fn go_ctrl(&self, s: &Stream, t: usize, streams: u32, mut a: Args) -> Result<(), String> {
        let tb = ctrl_tb(t);
        if tb == 0 {
            return self.go(s, self.k.controller, (8 * t as u32, streams), 288, a);
        }
        a.push(tb as i32).push(t as i32);
        self.go(s, self.k.controller, (8 * t.div_ceil(tb) as u32, streams), 288, a)
    }

    fn to_bf16(&self, s: &Stream, src: u64, dst: u64, n: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.push(src).push(dst).push(n as i32);
        self.go(s, self.k.to_bf16, (n.div_ceil(1024) as u32, 1), 256, a)
    }

    fn ring_store(&self, s: &Stream, src: u64, slotted: bool, width: usize, ring: u64, t: usize) -> Result<(), String> {
        let mut a = Args::new();
        a.push(src).push(slotted).push(width as i32).push(ring).push(t as i32).push(self.d_pos());
        self.go(s, self.k.ring_store, (t.min(16) as u32, 1), 256, a)
    }

    fn d_pos(&self) -> u64 {
        self.st.get(self.st.s.d_pos).ptr
    }

    fn wt(&self, name: &str) -> u64 {
        self.w.get(name).ptr
    }

    /// Records one chunk of `t` tokens (`tokens[0..t)` at positions `*d_pos + i`), leaving `d_pos` alone.
    pub fn record(&self, s: &Stream, t: usize) -> Result<(), String> {
        let (b, k, st) = (&self.b, &self.k, self.st);
        let null3 = PtrPack([NULL; 3]);
        let tu = t as u32;
        let d_pos = self.d_pos();
        let split_buf = st.get(st.s.split_buf).ptr;

        let mut a = Args::new();
        a.push(self.wt("embed.weight")).push(self.wt("embed_norm.weight")).push(b.tokens).push(b.hidden);
        self.go(s, k.embed_norm, (tu, 1), 256, a)?;
        let mut a = Args::new();
        a.push(d_pos).push(b.sub_pos).push((t / 8) as i32);
        self.go(s, k.sub_pos, (1, 1), 256, a)?;

        let tap = |i: usize| if self.dspark.is_some() { b.taps + (i * t * H * 4) as u64 } else { NULL };
        let cross_q = self.wt("middle_cycle_cross_pass_queries");
        let cross_kv_q = self.wt("middle_cycle_cross_pass_kv_queries");
        let mut block_count: i32 = 0;
        let mut prev_mlp_pending = false;
        for v in SCHEDULE.iter() {
            let vi = v.index;
            let l = v.layer;
            let lp = |n: &str| self.wt(&format!("layers.{l}.{n}"));
            let global = is_global_layer(l);
            let lq = &self.q.layers[l];
            let vs = &st.visits[vi];
            let use_prior = v.controllers;
            let boundary = l % dims::BLOCK_LAYERS == 0;
            let k_cache = st.get(vs.k_cache).ptr;
            let v_cache = st.get(vs.v_cache).ptr;
            let kv_fp8 = vs.kv_fp8.map(|sl| st.get(sl).ptr).unwrap_or(NULL);
            let in_norms = if global {
                [lp("q_input_norm.weight"), lp("k_input_norm.weight"), lp("v_input_norm.weight")]
            } else {
                [lp("attn_norm.weight"), NULL, NULL]
            };
            let attn_query = if l > 0 { self.wt(&format!("attn_res.attn_queries.{}", l - 1)) } else { NULL };

            if v.loop_restart {
                let mut a = Args::new();
                a.push(b.blocks).push(4i32).push(2i32).push(b.saved).push(t as i32);
                self.go(s, k.extract_block, ((t * H).div_ceil(256) as u32, 1), 256, a)?;
                let mut a = Args::new();
                a.push(b.hidden).push(NULL).push(b.prior_last).push(H as i32).push(H as i32).push(dims::RMS_EPS);
                self.go(s, k.rmsnorm, (tu, 1), 256, a)?;
                let bytes = t * H * 4;
                let pre = self.arena_slice(b.prelude, bytes);
                pre.copy_to(&self.arena_slice(b.hidden, bytes), s).map_err(|e| e.to_string())?;
                block_count = 1;
            }

            // A. Depth norm (with the previous visit's MLP post when pending).
            let partial_dst = if boundary { b.partial } else { NULL };
            let n_out: u32 = if global { 3 } else { 1 };
            let (norms, outs) = match (use_prior, global) {
                (false, true) => (PtrPack(in_norms), PtrPack(b.attn_in)),
                (false, false) => (PtrPack([in_norms[0], NULL, NULL]), PtrPack([b.attn_in[0], NULL, NULL])),
                (true, true) => (null3, PtrPack(b.attn_in)),
                (true, false) => (null3, PtrPack([b.mlp_in, NULL, NULL])),
            };
            let fp8 = !self.fp8.is_empty();
            let outs_h = match (fp8 && !use_prior, global) {
                (true, true) => PtrPack(b.attn_in_h),
                (true, false) => PtrPack([b.attn_in_h[0], NULL, NULL]),
                (false, _) => null3,
            };
            if prev_mlp_pending {
                let pl = SCHEDULE[vi - 1].layer;
                let prev_ring = st.get(st.visits[vi - 1].mlp_ring).ptr;
                let mut a = Args::new();
                a.push(b.slots).push(true).push(self.wt(&format!("layers.{pl}.mlp_sconv.weight"))).push(prev_ring);
                a.push(b.residual).push(b.hidden).push(if vi == dims::LOOP_START { b.prelude } else { TAP_VISITS.iter().position(|&x| x == vi).map_or(NULL, tap) }).push(NULL);
                a.push(partial_dst).push(b.blocks).push(block_count).push(attn_query).push(norms).push(outs).push(outs_h).push(d_pos);
                self.go(s, k.post_next, (tu, n_out), 576, a)?;
                self.ring_store(s, b.slots, true, H, prev_ring, t)?;
            } else {
                let mut a = Args::new();
                a.push(b.hidden).push(partial_dst).push(b.blocks).push(block_count).push(attn_query).push(norms).push(outs).push(outs_h);
                self.go(s, k.fused_depth_norm, (tu, n_out), 256, a)?;
            }

            // B. Attention controllers.
            let loc = l.wrapping_sub(dims::LOOP_START);
            if use_prior {
                let c = lq.ctrl.expect("nvfp4 controllers on a loop visit");
                let xs = if global { PtrPack(b.attn_in) } else { PtrPack([b.mlp_in; 3]) };
                let cpq = PtrPack([
                    cross_q + (2 * loc * H * 4) as u64,
                    cross_kv_q + (2 * loc * H * 4) as u64,
                    cross_kv_q + ((2 * loc + 1) * H * 4) as u64,
                ]);
                let post = if global { PtrPack(in_norms) } else { PtrPack([in_norms[0]; 3]) };
                let mut a = Args::new();
                a.push(xs).push(b.saved).push(b.prior_last).push(cpq).push(CtrlNvfp4([0, 1, 2].map(|m| [c[0][m], c[1][m], c[2][m]])));
                a.push(post).push(PtrPack(b.attn_in)).push(PtrPack(b.attn_in_h));
                self.go_ctrl(s, t, 3, a)?;
            }

            // C. Projections: FP8 prefill GEMMs on the fp8 mode's weights, or cuBLAS BF16 on the checkpoint's.
            let distinct = global || use_prior;
            let mt = if t <= 16 { 0 } else if t <= 32 { 1 } else { 2 };
            let rows_per_cta = 16usize << mt;
            if let (Some(kq), true) = (k.qkv_fp8, fp8) {
                let f = &self.fp8[l];
                let (in_kh, in_vh) = if distinct { (b.attn_in_h[1], b.attn_in_h[2]) } else { (b.attn_in_h[0], b.attn_in_h[0]) };
                let mut a = Args::new();
                for &p in &f[..6] {
                    a.push(p);
                }
                a.push(b.attn_in_h[0]).push(in_kh).push(in_vh).push(b.qrgs).push(b.k_proj).push(b.v_proj).push(t as i32);
                match k.dense_tc {
                    Some([tq, _]) if mt == 2 => self.go_smem(s, tq, (68u32, t.div_ceil(128) as u32), 256, DENSE_TC_SMEM, a)?,
                    _ => self.go(s, kq[mt], (269, t.div_ceil(rows_per_cta) as u32), 192, a)?,
                }
            } else {
                self.to_bf16(s, b.attn_in[0], b.xb[0], t * H)?;
                let (xk, xv) = if distinct {
                    self.to_bf16(s, b.attn_in[1], b.xb[1], t * H)?;
                    self.to_bf16(s, b.attn_in[2], b.xb[2], t * H)?;
                    (b.xb[1], b.xb[2])
                } else {
                    (b.xb[0], b.xb[0])
                };
                let kv = lp("attn.kv");
                self.blas.linear_f32(b.xb[0], lp("attn.qrgs"), b.qrgs, t, H, dims::QRGS_ROWS, dims::QRGS_ROWS)?;
                self.blas.linear_f32(xk, kv, b.k_proj, t, H, dims::KV_DIM, dims::KV_DIM)?;
                self.blas.linear_f32(xv, kv + (dims::KV_DIM * H * 2) as u64, b.v_proj, t, H, dims::KV_DIM, dims::KV_DIM)?;
            }

            // D + E, per slice (the local caches hold WINDOW + local_slice() positions): short conv,
            // clip and KV store of the slice (rings read only), then its attention, which sees every key
            // before it. The ring slots are written after the last slice.
            let k_ring = st.get(vs.k_ring).ptr;
            let v_ring = st.get(vs.v_ring).ptr;
            let window: i32 = if global { 0 } else { dims::WINDOW as i32 };
            let rel_proj = lp("attn.rel_logits_proj.proj");
            // Global visits with a 16-token flash kernel: store every slice first, then one attention launch
            // over the whole chunk (4x the CTAs of a 256-token slice; 128 CTAs left 42 of 170 SMs idle).
            // Same values: queries never read later rows (causal mask), and the 2304-slot global ring still
            // holds every row from the chunk's first query - 1023 on.
            let fuse = global && gfuse() && (k.flash3.is_some() || k.flash4.is_some() || k.flash5.is_some() || (k.shadow && k.flash.is_some()));
            // Local visits store and attend `local_slice()` tokens per launch; global ones 256 (fused below).
            let slice = if global { SLICE } else { crate::state::local_slice() };
            for j0 in (0..t).step_by(slice) {
                let tj = slice.min(t - j0);
                let mut a = Args::new();
                a.push(b.qrgs).push(b.k_proj).push(b.v_proj);
                if global {
                    a.push(b.attn_in[1]).push(b.attn_in[2]);
                }
                a.push(if global { lp("attn.k_sconv.static_w") } else { lp("attn.k_sconv.weight") });
                a.push(if global { lp("attn.v_sconv.static_w") } else { lp("attn.v_sconv.weight") });
                if global {
                    a.push(lp("attn.k_sconv.weight_proj")).push(lp("attn.v_sconv.weight_proj"));
                }
                a.push(k_ring).push(v_ring).push(lp("attn.q_l2_clip.scale")).push(lp("attn.rel_l2_clip.scale")).push(k_cache).push(v_cache);
                if global {
                    a.push(kv_fp8).push(NULL);
                }
                a.push(vs.kv_cap as i32).push(d_pos).push(j0 as i32);
                self.go(s, if global { k.global_prep } else { k.local_prep }, (64, tj as u32), 128, a)?;
                // Fused launches cover up to FUSE_MAX tokens: the 2304-slot FP8-mode ring holds the group's
                // first query - 1023 .. its last row (<= 2047 rows) only up to 1024 tokens per group.
                let g0 = j0 / FUSE_MAX * FUSE_MAX;
                if fuse && j0 + tj < (g0 + FUSE_MAX).min(t) {
                    continue;
                }
                let (j0, tj) = if fuse { (g0, j0 + tj - g0) } else { (j0, tj) };

                if let (Some([f2, m2]), true) = (k.flash2, tj % 32 == 0) {
                    // Long-context flash (32-token query blocks, cp.async K/V stages); global layers split
                    // their key range over `GLOBAL_SPLITS` CTAs and merge with the sink and gate.
                    let splits: i32 = if global { global_splits() } else { 1 };
                    let (og, ogh) = if fp8 { (NULL, b.attn_gated_h) } else { (b.attn_gated, NULL) };
                    let mut a = Args::new();
                    a.push(b.qrgs).push(b.rel16 + (l * 1024 * 16 * 2) as u64).push(k_cache).push(v_cache).push(vs.ring_cap as i32).push(window).push(d_pos);
                    a.push(og).push(ogh).push(j0 as i32).push(splits).push(b.attn_part).push(kv_fp8).push(vs.kv_cap as i32);
                    self.go_smem(s, f2, (dims::KV_HEADS as u32, (tj / 32) as u32, splits as u32), 192, FLASH2_SMEM, a)?;
                    if splits > 1 {
                        let mut a = Args::new();
                        a.push(b.qrgs).push(b.attn_part).push(splits).push(og).push(ogh).push(j0 as i32);
                        self.go(s, m2, (tj as u32, dims::HEADS as u32), 128, a)?;
                    }
                    continue;
                }
                if let (true, false, Some(f5)) = (fuse, k.shadow, k.flash5_any.filter(|_| k.flash3.is_some())) {
                    // Fused global launch with the default flash: flash5 (flash3's arithmetic, one stage, two
                    // CTAs per SM), so the chunk's 8 x t/16 CTAs run in ~2 rounds on 340 CTA slots.
                    let mut a = Args::new();
                    a.push(b.qrgs).push(b.rel16 + (l * 1024 * 16 * 2) as u64).push(k_cache).push(v_cache).push(vs.ring_cap as i32).push(window).push(d_pos);
                    a.push(if fp8 { NULL } else { b.attn_gated }).push(if fp8 { b.attn_gated_h } else { NULL }).push(j0 as i32);
                    a.push(kv_fp8).push(vs.kv_cap as i32);
                    self.go_smem(s, f5, (dims::KV_HEADS as u32, (tj / 16) as u32), 192, FLASH5_SMEM, a)?;
                    continue;
                }
                // Local launches of 512 tokens or more also take flash5 (bit-identical to flash3): their 256+ CTAs
                // fill the GPU two per SM. Below that flash3 is faster (128 CTAs, one per SM).
                let local5 = k.flash5_any.filter(|_| !global && tj >= 512 && k.flash3.is_some() && !k.shadow);
                if let Some(f5) = k.flash5.or(local5) {
                    let mut a = Args::new();
                    a.push(b.qrgs).push(b.rel16 + (l * 1024 * 16 * 2) as u64).push(k_cache).push(v_cache).push(vs.ring_cap as i32).push(window).push(d_pos);
                    a.push(if fp8 { NULL } else { b.attn_gated }).push(if fp8 { b.attn_gated_h } else { NULL }).push(j0 as i32);
                    a.push(kv_fp8).push(vs.kv_cap as i32);
                    self.go_smem(s, f5, (dims::KV_HEADS as u32, (tj / 16) as u32), 192, FLASH5_SMEM, a)?;
                    continue;
                }
                if let Some(f4) = k.flash4 {
                    let mut a = Args::new();
                    a.push(b.qrgs).push(b.rel16 + (l * 1024 * 16 * 2) as u64).push(k_cache).push(v_cache).push(vs.ring_cap as i32).push(window).push(d_pos);
                    a.push(if fp8 { NULL } else { b.attn_gated }).push(if fp8 { b.attn_gated_h } else { NULL }).push(j0 as i32);
                    a.push(kv_fp8).push(vs.kv_cap as i32).push(tj as i32);
                    self.go_smem(s, f4, (dims::KV_HEADS as u32, tj.div_ceil(32) as u32), 384, FLASH4_SMEM, a)?;
                    continue;
                }
                if let Some(f3) = k.flash3 {
                    let mut a = Args::new();
                    a.push(b.qrgs).push(b.rel16 + (l * 1024 * 16 * 2) as u64).push(k_cache).push(v_cache).push(vs.ring_cap as i32).push(window).push(d_pos);
                    a.push(if fp8 { NULL } else { b.attn_gated }).push(if fp8 { b.attn_gated_h } else { NULL }).push(j0 as i32);
                    a.push(kv_fp8).push(vs.kv_cap as i32);
                    self.go_smem(s, f3, (dims::KV_HEADS as u32, (tj / 16) as u32), 192, FLASH3_SMEM, a)?;
                    continue;
                }
                if let (true, Some(f3), Some(kc), true) = (k.shadow, m3(&self.k), k.cmp_h, fp8) {
                    // Shadow: flash3 into a scratch f16 output, compared after v1 below.
                    let mut a = Args::new();
                    a.push(b.qrgs).push(b.rel16 + (l * 1024 * 16 * 2) as u64).push(k_cache).push(v_cache).push(vs.ring_cap as i32).push(window).push(d_pos);
                    a.push(NULL).push(b.shadow_h).push(j0 as i32).push(kv_fp8).push(vs.kv_cap as i32);
                    if let (true, Some(f5)) = (k.shadow5, k.flash5_any) {
                        self.go_smem(s, f5, (dims::KV_HEADS as u32, (tj / 16) as u32), 192, FLASH5_SMEM, a)?;
                    } else if let (true, Some(f4)) = (k.shadow4, k.flash4_any) {
                        a.push(tj as i32);
                        self.go_smem(s, f4, (dims::KV_HEADS as u32, tj.div_ceil(32) as u32), 384, FLASH4_SMEM, a)?;
                    } else {
                        self.go_smem(s, f3, (dims::KV_HEADS as u32, (tj / 16) as u32), 192, FLASH3_SMEM, a)?;
                    }
                    let fk = k.flash.ok_or("shadow needs prefill_attn_flash")?;
                    let mut a = Args::new();
                    a.push(b.qrgs).push(rel_proj).push(k_cache).push(v_cache).push(vs.ring_cap as i32).push(window).push(d_pos);
                    a.push(NULL).push(b.attn_gated_h).push(j0 as i32).push(kv_fp8).push(vs.kv_cap as i32);
                    self.go(s, fk, (dims::KV_HEADS as u32, (tj / 16) as u32), 192, a)?;
                    let n = tj * dims::Q_DIM;
                    let off = (j0 * dims::Q_DIM * 2) as u64;
                    let mut a = Args::new();
                    a.push(b.attn_gated_h + off).push(b.shadow_h + off).push(n as i32).push(b.counters).push(if global { 0i32 } else { 1i32 });
                    self.go(s, kc, (n.div_ceil(256) as u32, 1), 256, a)?;
                    continue;
                }
                if let Some(fk) = k.flash {
                    let mut a = Args::new();
                    a.push(b.qrgs).push(rel_proj).push(k_cache).push(v_cache).push(vs.ring_cap as i32).push(window).push(d_pos);
                    a.push(if fp8 { NULL } else { b.attn_gated }).push(if fp8 { b.attn_gated_h } else { NULL }).push(j0 as i32);
                    a.push(kv_fp8).push(vs.kv_cap as i32);
                    self.go(s, fk, (dims::KV_HEADS as u32, (tj / 16) as u32), 192, a)?;
                    continue;
                }
                // T / 8 sub-steps of the B = 8 verify kernels.
                for j in j0 / 8..(j0 + tj) / 8 {
                    let pos_j = b.sub_pos + 4 * j as u64;
                    let q_j = b.qrgs + (j * 8 * dims::QRGS_ROWS * 4) as u64;
                    if global {
                        let mut a = Args::new();
                        if kv_fp8 != NULL {
                            let (cap, kd) = (vs.kv_cap as u64, dims::KV_DIM as u64);
                            a.push(q_j).push(kv_fp8).push(kv_fp8 + cap * kd).push(kv_fp8 + 2 * cap * kd).push(kv_fp8 + 2 * cap * kd + cap * dims::KV_HEADS as u64 * 2);
                            a.push(vs.kv_cap as i32).push(pos_j).push(split_buf);
                            self.go(s, k.bulk8_fp8, (8, 64), 128, a)?;
                        } else {
                            a.push(q_j).push(k_cache).push(v_cache).push(NULL).push(NULL).push(vs.kv_cap as i32).push(pos_j).push(split_buf);
                            self.go(s, k.bulk8, (8, 64), 128, a)?;
                        }
                    }
                    let mut a = Args::new();
                    a.push(q_j).push(rel_proj).push(k_cache).push(v_cache).push(NULL).push(NULL).push(vs.ring_cap as i32);
                    a.push(window).push(pos_j).push(split_buf);
                    self.go(s, k.tail8, (8, 17), 256, a)?;
                    let mut a = Args::new();
                    a.push(q_j).push(split_buf);
                    a.push(if fp8 { NULL } else { b.attn_gated + (j * 8 * dims::Q_DIM * 4) as u64 });
                    a.push(if fp8 { b.attn_gated_h + (j * 8 * dims::Q_DIM * 2) as u64 } else { NULL }).push(window).push(pos_j);
                    self.go(s, k.merge, (dims::HEADS as u32, 8), 128, a)?;
                }
            }
            self.ring_store(s, b.k_proj, false, dims::KV_DIM, k_ring, t)?;
            self.ring_store(s, b.v_proj, false, dims::KV_DIM, v_ring, t)?;

            // F. Output projection.
            if let (Some(ko), true) = (k.o_fp8, fp8) {
                let f = &self.fp8[l];
                let mut a = Args::new();
                a.push(f[6]).push(f[7]).push(b.attn_gated_h).push(b.o_proj).push(t as i32);
                // prefill-dense v2: MT = 1, 2 run 144 x 256-thread CTAs, MT = 4 72 x 128.
                let (gx, threads) = if mt == 2 { (72, 128) } else { (144, 256) };
                match k.dense_tc {
                    Some([_, to]) if mt == 2 => self.go_smem(s, to, (18u32, t.div_ceil(128) as u32), 256, DENSE_TC_SMEM, a)?,
                    _ => self.go(s, ko[mt], (gx, t.div_ceil(rows_per_cta) as u32), threads, a)?,
                }
            } else {
                self.to_bf16(s, b.attn_gated, b.xb[0], t * dims::Q_DIM)?;
                self.blas.linear_f32(b.xb[0], lp("attn.wo_ud.weight"), b.o_proj, t, dims::Q_DIM, H, H)?;
            }

            // G. Attention post-process and MLP input.
            let attn_ring = st.get(vs.attn_ring).ptr;
            let mlp_norm = lp("mlp_norm.weight");
            let mut a = Args::new();
            a.push(b.o_proj).push(lp("attn_sconv.weight")).push(attn_ring).push(b.hidden).push(b.partial).push(b.blocks);
            a.push(block_count).push(boundary).push(b.residual).push(self.wt(&format!("attn_res.mlp_queries.{l}")));
            a.push(if use_prior { NULL } else { mlp_norm }).push(if use_prior { b.mlp_in } else { b.mlp_normed });
            a.push(if use_prior { NULL } else { b.mlp_normed_h }).push(d_pos);
            self.go(s, k.attn_post, (tu, 1), 576, a)?;
            self.ring_store(s, b.o_proj, false, H, attn_ring, t)?;
            if boundary {
                block_count += 1;
            }

            // H. MLP controller.
            if use_prior {
                let c = lq.ctrl.expect("nvfp4 controllers on a loop visit")[3];
                let mut a = Args::new();
                a.push(PtrPack([b.mlp_in; 3])).push(b.saved).push(b.prior_last).push(PtrPack([cross_q + ((2 * loc + 1) * H * 4) as u64; 3]));
                a.push(CtrlNvfp4([0, 1, 2].map(|i| [c[i]; 3]))).push(PtrPack([mlp_norm; 3])).push(PtrPack([b.mlp_normed; 3]));
                a.push(PtrPack([b.mlp_normed_h, NULL, NULL]));
                self.go_ctrl(s, t, 1, a)?;
            }

            // I. Router and experts grouped by expert. The router runs the decode path's NVFP4 arithmetic (same
            // logits as decode), or a cuBLAS BF16 GEMM (INKLING_WIDE_ROUTER=bf16).
            if let Some(kr) = k.router8 {
                let mut a = Args::new();
                a.push(b.mlp_normed_h).push(lq.router.data.ptr).push(lq.router.scales.ptr).push(lq.router.gs).push(b.router_logits);
                self.go(s, kr, (64, (t / 8) as u32), 32, a)?;
            } else {
                self.to_bf16(s, b.mlp_normed, b.xb[0], t * H)?;
                self.blas.linear_f32(b.xb[0], lp("mlp.gate.weight"), b.router_logits, t, H, dims::EXPERTS, dims::EXPERTS)?;
            }
            let mut a = Args::new();
            a.push(b.router_logits).push(lp("mlp.gate.bias")).push(b.route_ids).push(b.route_w).push(t as i32);
            self.go(s, k.route_select, (t.div_ceil(8) as u32, 1), 256, a)?;
            let mut a = Args::new();
            a.push(b.route_ids).push((t * dims::TOP_K) as i32).push(b.offs).push(b.pairs);
            self.go(s, k.moe_group, (1, 1), 1024, a)?;
            let e = &lq.experts;
            if let Some([kt, kq, ku, kd]) = k.moe_bs {
                let max_tiles = (t * dims::TOP_K / 64 + dims::EXPERTS) as u32;
                let mut a = Args::new();
                a.push(b.offs).push(b.tiles).push(b.tiles_n);
                self.go(s, kt, (1, 1), 32, a)?;
                let quant = |src: u64, values: usize| -> Result<(), String> {
                    let mut a = Args::new();
                    a.push(src).push(b.qa_codes).push(b.qa_scales).push((values / 16) as i32);
                    self.go(s, kq, ((values / 16).div_ceil(256) as u32, 1), 256, a)
                };
                quant(b.mlp_normed_h, t * H)?;
                let mut a = Args::new();
                a.push(b.qa_codes).push(b.qa_scales).push(b.offs).push(b.pairs).push(b.tiles).push(b.tiles_n).push(e.w13.ptr).push(e.w13_scales.ptr).push(e.w13_factor.ptr).push(b.act);
                self.go_smem(s, ku, ((2 * dims::INTERMEDIATE / 128) as u32, max_tiles), 256, MOE_BS_SMEM, a)?;
                quant(b.act, t * dims::TOP_K * dims::INTERMEDIATE)?;
                let mut a = Args::new();
                a.push(b.qa_codes).push(b.qa_scales).push(b.offs).push(b.pairs).push(b.tiles).push(b.tiles_n).push(b.route_w).push(e.w2.ptr).push(e.w2_scales.ptr).push(e.w2_factor.ptr).push(b.slots);
                self.go_smem(s, kd, ((H / 128) as u32, max_tiles), 256, MOE_BS_SMEM, a)?;
            } else if let Some([kt, ku, kd]) = k.moe_mma {
                let max_tiles = (t * dims::TOP_K / 64 + dims::EXPERTS) as u32;
                let mut a = Args::new();
                a.push(b.offs).push(b.tiles).push(b.tiles_n);
                self.go(s, kt, (1, 1), 32, a)?;
                let mut a = Args::new();
                a.push(b.mlp_normed_h).push(b.offs).push(b.pairs).push(b.tiles).push(b.tiles_n).push(e.w13.ptr).push(e.w13_scales.ptr).push(e.w13_factor.ptr).push(b.act);
                self.go_smem(s, ku, ((2 * dims::INTERMEDIATE / 128) as u32, max_tiles), 256, MOE_MMA_SMEM, a)?;
                let mut a = Args::new();
                a.push(b.act).push(b.offs).push(b.pairs).push(b.tiles).push(b.tiles_n).push(b.route_w).push(e.w2.ptr).push(e.w2_scales.ptr).push(e.w2_factor.ptr).push(b.slots);
                self.go_smem(s, kd, ((H / 128) as u32, max_tiles), 256, MOE_MMA_SMEM, a)?;
            } else {
                let mut a = Args::new();
                a.push(b.mlp_normed_h).push(b.offs).push(b.pairs).push(e.w13.ptr).push(e.w13_scales.ptr).push(e.w13_factor.ptr).push(b.act);
                let rt = if moe_r2c() { 2 } else { 1 };
                self.go(s, k.moe_up, ((2 * dims::INTERMEDIATE / 16 / rt) as u32, dims::EXPERTS as u32), 576, a)?;
                let mut a = Args::new();
                a.push(b.act).push(b.offs).push(b.pairs).push(b.route_w).push(e.w2.ptr).push(e.w2_scales.ptr).push(e.w2_factor.ptr).push(b.slots);
                self.go(s, k.moe_down, ((H / 16 / rt) as u32, dims::EXPERTS as u32), 288, a)?;
            }

            // J. Flush before a loop restart.
            if v.flush_mlp {
                let ring = st.get(vs.mlp_ring).ptr;
                let mut a = Args::new();
                a.push(b.slots).push(true).push(lp("mlp_sconv.weight")).push(ring).push(b.residual).push(b.hidden).push(d_pos);
                self.go(s, k.post_res, (tu, 1), 576, a)?;
                self.ring_store(s, b.slots, true, H, ring, t)?;
                prev_mlp_pending = false;
            } else {
                prev_mlp_pending = true;
            }
        }

        // Final pending MLP post + output depth attention + final norm (the drafter's last tap); no LM head.
        let last = dims::LAYERS - 1;
        let ring = st.get(st.visits[dims::VISITS - 1].mlp_ring).ptr;
        let mut a = Args::new();
        a.push(b.slots).push(true).push(self.wt(&format!("layers.{last}.mlp_sconv.weight"))).push(ring).push(b.residual).push(b.hidden);
        a.push(NULL).push(tap(3)).push(NULL).push(b.blocks).push(4i32).push(self.wt("attn_res.output_query"));
        a.push(PtrPack([self.wt("norm.weight"), NULL, NULL])).push(PtrPack([b.mlp_normed, NULL, NULL])).push(null3).push(d_pos);
        self.go(s, k.post_next, (tu, 1), 576, a)?;
        self.ring_store(s, b.slots, true, H, ring, t)?;

        // DSpark commit (f ring, draft context KV) for every position, as T / 8 B = 8 commits: each gets
        // its 8 tokens' taps and its start position in d_pos (restored to the chunk start afterwards).
        if let Some(ds) = self.dspark.filter(|ds| ds.has_wide_commit()) {
            // One commit for the chunk. `slots` is free after the last MLP post: [8 t][H] f32 holds the mix
            // ([3 t][H]) and the K/V projections ([t][2 KV_DIM]).
            ds.record_commit_wide(s, t, b.taps, b.slots, b.slots + (3 * t * H * 4) as u64)?;
        } else if let Some(ds) = self.dspark {
            let dpos = st.get(st.s.d_pos).sub(0, 4);
            let dtaps = ds.taps();
            for j in 0..t / 8 {
                self.arena_slice(b.sub_pos + 4 * j as u64, 4).copy_to(&dpos, s).map_err(|e| e.to_string())?;
                for i in 0..4 {
                    let src = self.arena_slice(b.taps + ((i * t + 8 * j) * H * 4) as u64, 8 * H * 4);
                    src.copy_to(&inkling_runtime::DevSlice { ptr: dtaps + (i * 8 * H * 4) as u64, bytes: 8 * H * 4 }, s).map_err(|e| e.to_string())?;
                }
                ds.record_commit(s, 8)?;
            }
            self.arena_slice(b.sub_pos, 4).copy_to(&dpos, s).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn arena_slice(&self, p: u64, bytes: usize) -> inkling_runtime::DevSlice {
        let whole = self.arena.whole();
        whole.sub((p - whole.ptr) as usize, bytes)
    }

    /// Prefill of `prompt` from position 0. See `prefill_range`.
    pub fn prefill(&self, graphs: &[(i32, GraphExec)], g1: &GraphExec, prompt: &[u32]) -> Result<u32, String> {
        Ok(self.prefill_range(graphs, g1, prompt, 0, prompt.len(), &mut || true)?.expect("tick never stops"))
    }

    /// Prefill of `prompt[start..end)` at positions `start..end`, with positions `0..start` already in the
    /// cache (a server's extend or checkpoint restore). Wide chunks while at least 16 tokens remain before
    /// the last one (when the range has at least `INKLING_WIDE_MIN` tokens), then the B = 8/4/2 `graphs`,
    /// and the last token through `g1`, which advances `d_pos` and feeds the next token into `d_tokens[0]`.
    /// With a drafter attached (`new(.., Some(dspark))`) the wide chunks write its taps and commit, and
    /// `graphs` / `g1` should be the spec prefill graphs (with commit).
    ///
    /// Every ~512 tokens the host syncs and calls `tick`; if it returns false the prefill stops and returns
    /// None (the cache then holds a prefix of the range). Otherwise returns the greedy token after `end`.
    pub fn prefill_range(
        &self,
        graphs: &[(i32, GraphExec)],
        g1: &GraphExec,
        prompt: &[u32],
        start: usize,
        end: usize,
        tick: &mut dyn FnMut() -> bool,
    ) -> Result<Option<u32>, String> {
        let st = self.st;
        let s = self.stream;
        if start >= end || end > prompt.len() {
            return Err(format!("prefill range {start}..{end} of a {}-token prompt", prompt.len()));
        }
        if end > st.max_ctx {
            return Err(format!("context overflow: {end} positions needed but the state holds {} (raise --max-ctx)", st.max_ctx));
        }
        let wide_ok = end - start >= wide_min();
        // (start, size, wide)
        let mut chunks: Vec<(usize, usize, bool)> = Vec::new();
        let mut i = start;
        while i + 1 < end {
            let left = end - 1 - i;
            if let Some(t) = chunk_sizes().into_iter().find(|&t| t <= left).filter(|_| wide_ok) {
                chunks.push((i, t, true));
                i += t;
            } else {
                let b = graphs.iter().map(|(b, _)| *b as usize).find(|&b| b <= left).unwrap_or(1);
                chunks.push((i, b, false));
                i += b;
            }
        }
        chunks.push((end - 1, 1, false));
        let n = end;
        let mut host = PinnedBuffer::alloc(4 * n + 4 * chunks.len())?;
        {
            let h = host.as_mut_slice();
            for (j, &t) in prompt[..n].iter().enumerate() {
                h[4 * j..4 * j + 4].copy_from_slice(&(t as i32).to_ne_bytes());
            }
            for (c, &(c0, _, _)) in chunks.iter().enumerate() {
                let o = 4 * n + 4 * c;
                h[o..o + 4].copy_from_slice(&(c0 as i32).to_ne_bytes());
            }
        }
        let d_tok = st.get(st.s.d_tokens);
        let d_pos = st.get(st.s.d_pos).sub(0, 4);
        let mut since_sync = 0;
        for (c, &(c0, size, wide)) in chunks.iter().enumerate() {
            host.copy_range_to_device_async(4 * n + 4 * c, &d_pos, s)?;
            if wide {
                self.last_wide.set((c0, size));
                host.copy_range_to_device_async(4 * c0, &self.tokens_slice.sub(0, 4 * size), s)?;
                match self.graphs.iter().find(|(t, _)| *t == size) {
                    Some((_, g)) => g.launch(s)?,
                    None => self.record(s, size)?,
                }
            } else {
                host.copy_range_to_device_async(4 * c0, &d_tok.sub(0, 4 * size), s)?;
                if size == 1 {
                    g1.launch(s)?;
                } else {
                    graphs.iter().find(|(gb, _)| *gb as usize == size).expect("graph for chunk").1.launch(s)?;
                }
            }
            since_sync += size;
            if since_sync >= 512 && c + 1 < chunks.len() {
                since_sync = 0;
                s.synchronize()?;
                if !tick() {
                    return Ok(None);
                }
            }
        }
        s.synchronize()?;
        Ok(Some(st.get(st.s.d_next_tokens).download::<i32>(1)?[0] as u32))
    }

    /// Kernel parity of the long-context flash kernel against v1 on real data: after a prefill, reruns the
    /// last visit's attention (visit 47, global) for the last wide chunk's queries with
    /// `prefill_attn_flash` (oracle) and `prefill_attn_flash2` at `splits`, both in f32, and returns
    /// (max |d|, rms of the oracle, rms of the difference, chunk start, tokens).
    pub fn flash_parity(&self, splits: i32) -> Result<(f32, f32, f32, usize, usize), String> {
        let (k1, [k2, m2]) = (self.k.flash.ok_or("no prefill_attn_flash")?, self.k.flash2.ok_or("no prefill_attn_flash2")?);
        let (c0, t) = self.last_wide.get();
        if t % 32 != 0 {
            return Err(format!("last wide chunk has {t} tokens"));
        }
        let (st, s, b) = (self.st, self.stream, &self.b);
        let vi = dims::VISITS - 1;
        let l = SCHEDULE[vi].layer;
        let vs = &st.visits[vi];
        let (kc, vc) = (st.get(vs.k_cache).ptr, st.get(vs.v_cache).ptr);
        let kv_fp8 = vs.kv_fp8.map(|sl| st.get(sl).ptr).unwrap_or(NULL);
        let rel = self.wt(&format!("layers.{l}.attn.rel_logits_proj.proj"));
        let n = t * dims::Q_DIM;
        let o1 = inkling_runtime::DeviceBuffer::alloc(4 * n)?;
        let o2 = inkling_runtime::DeviceBuffer::alloc(4 * n)?;
        st.get(st.s.d_pos).upload(&[c0 as i32])?;
        for j0 in (0..t).step_by(SLICE) {
            let tj = SLICE.min(t - j0);
            let mut a = Args::new();
            a.push(b.qrgs).push(rel).push(kc).push(vc).push(vs.ring_cap as i32).push(0i32).push(self.d_pos()).push(o1.slice().ptr).push(NULL).push(j0 as i32);
            a.push(kv_fp8).push(vs.kv_cap as i32);
            self.go(s, k1, (dims::KV_HEADS as u32, (tj / 16) as u32), 192, a)?;
            let og = if splits > 1 { NULL } else { o2.slice().ptr };
            let mut a = Args::new();
            a.push(b.qrgs).push(b.rel16 + (l * 1024 * 16 * 2) as u64).push(kc).push(vc).push(vs.ring_cap as i32).push(0i32).push(self.d_pos());
            a.push(og).push(NULL).push(j0 as i32).push(splits).push(b.attn_part).push(kv_fp8).push(vs.kv_cap as i32);
            self.go_smem(s, k2, (dims::KV_HEADS as u32, (tj / 32) as u32, splits as u32), 192, FLASH2_SMEM, a)?;
            if splits > 1 {
                let mut a = Args::new();
                a.push(b.qrgs).push(b.attn_part).push(splits).push(o2.slice().ptr).push(NULL).push(j0 as i32);
                self.go(s, m2, (tj as u32, dims::HEADS as u32), 128, a)?;
            }
        }
        s.synchronize()?;
        let r: Vec<f32> = o1.slice().download(n)?;
        let c: Vec<f32> = o2.slice().download(n)?;
        let (mut maxd, mut sr, mut sd) = (0f32, 0f64, 0f64);
        for i in 0..n {
            let d = (r[i] - c[i]).abs();
            maxd = maxd.max(d);
            sr += (r[i] as f64).powi(2);
            sd += (d as f64).powi(2);
        }
        Ok((maxd, (sr / n as f64).sqrt() as f32, (sd / n as f64).sqrt() as f32, c0, t))
    }

    /// Shadow-check counters since the last call (INKLING_WIDE_SHADOW=3), then reset:
    /// (global mismatches, global max |d|, local mismatches, local max |d|).
    pub fn shadow_counters(&self) -> Result<(u32, f32, u32, f32), String> {
        self.stream.synchronize()?;
        let c = self.arena_slice(self.b.counters, 16);
        let v: Vec<u32> = c.download(4)?;
        c.fill_u8(0, self.stream)?;
        self.stream.synchronize()?;
        Ok((v[0], f32::from_bits(v[1]), v[2], f32::from_bits(v[3])))
    }
}
