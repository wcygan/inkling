//! DSpark speculative decoding (nvfp4 mode): the drafter's weights and state, and recorders for the
//! commit (draft context KV from the target's taps), the 7-draft block and the verify bookkeeping.
//!
//! Launch order and parameters follow squid5090's `launch_dspark_commit` / `launch_dspark_draft` /
//! `launch_spec_verify`.

use crate::nvfp4::{Nvfp4Weights, Packed, pack_dense};
use crate::state::{B_MAX, State};
use inkling_model::dims;
use inkling_model::weights::{MTP_INTERMEDIATE, MTP_LAYERS, MTP_MARKOV_RANK, Weights};
use inkling_runtime::{Arena, ArenaPlan, Args, DevSlice, Event, Kernel, KernelArg, LaunchConfig, Module, Slot, Stream};

const H: usize = dims::HIDDEN;
const NULL: u64 = 0;
pub const NUM_DRAFTS: usize = 7;
const BLOCK: usize = NUM_DRAFTS + 1; // anchor + 7 masks
const QKV: usize = dims::Q_DIM + 2 * dims::KV_DIM; // 5120
const RNN: usize = 3 * MTP_MARKOV_RANK; // 768
const MARKOV_BLOCKS: usize = 340;
const UNEMBED_BLOCKS: u32 = 680;
/// Draft local layers (0 and 2): SWA window + 8.
pub const DRAFT_LOCAL_CAP: usize = dims::WINDOW + 8;

fn is_local(l: usize) -> bool {
    l == 0 || l == 2
}

struct DLayer {
    qkv: Packed,
    wo: Packed,
    gu: Packed,
    down: Packed,
    kv_ctx: Packed,
    input_norm: u64,
    post_norm: u64,
    dflare_norm: u64,
    q_norm: u64,
    k_norm: u64,
    k_norm_ctx: u64,
}

/// The drafter's weights: BF16 file tensors, NVFP4 packs, the split RNN projection and the folded
/// Markov head over the draft vocabulary.
pub struct DsparkWeights {
    pub mtp: Weights,
    pub arena: Arena,
    layers: Vec<DLayer>,
    fuse_alpha: u64,
    mask_embed: u64,
    norm_w: u64,
    markov_dflare_norm: u64,
    markov_w1: u64,
    rnn_bias: u64,
    w_sp: u64,
    w_h: u64,
    w_f: u64,
    unembed_draft: Packed,
    fold_w: Packed,
    fold_b: u64,
    /// Draft vocabulary: rows of `unembed_draft` / `fold_w`.
    pub vocab: usize,
    /// Draft row -> token id, when the draft vocabulary is a shortlist.
    vocab_ids: u64,
    pub prepare_seconds: f64,
}

fn launch(k: Kernel, grid: impl Into<inkling_runtime::launch::Dim>, block: impl Into<inkling_runtime::launch::Dim>, pdl: bool, stream: &Stream, mut args: Args) -> Result<(), String> {
    let cfg = LaunchConfig::new(grid, block).pdl(pdl);
    unsafe { k.launch(&cfg, stream, &mut args) }.map_err(|e| e.to_string())
}

fn kern(m: &Module, n: &str) -> Result<Kernel, String> {
    m.function(n).ok_or_else(|| format!("kernel {n} missing from the cubin"))
}

impl DsparkWeights {
    /// Packs the drafter for nvfp4 mode. `shortlist` = draft-vocabulary token ids (sorted, unique, all
    /// < VOCAB); `None` uses the full 200,058-token vocabulary and the target's packed LM head.
    pub fn prepare(
        module: &Module,
        main: &Weights,
        q: &Nvfp4Weights,
        mtp: Weights,
        shortlist: Option<&[i32]>,
        stream: &Stream,
    ) -> Result<DsparkWeights, String> {
        let t = std::time::Instant::now();
        let g = |n: &str| mtp.get(n);
        let vocab = shortlist.map_or(dims::VOCAB, <[i32]>::len);
        let mut plan = ArenaPlan::new();
        let nv = |plan: &mut ArenaPlan, name: String, rows: usize, k: usize| {
            (plan.reserve(format!("{name}.nvfp4"), rows * k / 2), plan.reserve(format!("{name}.scales"), rows * k / 16), rows, k)
        };
        let shapes = [("qkv", QKV, H), ("wo", H, dims::Q_DIM), ("gu", 2 * MTP_INTERMEDIATE, H), ("down", H, MTP_INTERMEDIATE), ("kv_ctx", 2 * dims::KV_DIM, H)];
        let layer_slots: Vec<Vec<(Slot, Slot, usize, usize)>> = (0..MTP_LAYERS)
            .map(|l| shapes.iter().map(|&(n, r, k)| nv(&mut plan, format!("ds.{l}.{n}"), r, k)).collect())
            .collect();
        let fuse_alpha = plan.reserve_of::<f32>("ds.fuse_alpha", 16);
        let w_sp = plan.reserve_of::<u16>("ds.w_sp", RNN * 2 * MTP_MARKOV_RANK);
        let w_h = plan.reserve_of::<u16>("ds.w_h", RNN * H);
        let w_f = plan.reserve_of::<u16>("ds.w_f", RNN * H);
        let fold_b = plan.reserve_of::<f32>("ds.fold_b", vocab);
        let fold = nv(&mut plan, "ds.fold_w".into(), vocab, MTP_MARKOV_RANK);
        let unembed_pruned = shortlist.map(|_| nv(&mut plan, "ds.unembed_draft".into(), vocab, H));
        let ids = shortlist.map(|s| plan.reserve_of::<i32>("ds.vocab_ids", s.len()));
        let arena = plan.allocate().map_err(|e| e.to_string())?;

        // fuse_weight [4, 4] -> row-wise softmax (f64, as squid5090).
        let fw: Vec<u16> = g("fuse_weight").download(16)?;
        let mut alpha = [0f32; 16];
        for r in 0..4 {
            let v: Vec<f64> = (0..4).map(|c| f32::from_bits((fw[r * 4 + c] as u32) << 16) as f64).collect();
            let mx = v.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = v.iter().map(|x| (x - mx).exp()).collect();
            let s: f64 = e.iter().sum();
            for c in 0..4 {
                alpha[r * 4 + c] = (e[c] / s) as f32;
            }
        }
        arena.get(fuse_alpha).upload(&alpha)?;

        let packed = |(d, s, rows, k): (Slot, Slot, usize, usize), src: DevSlice| -> Result<Packed, String> {
            let (data, scales) = (arena.get(d), arena.get(s));
            let gs = pack_dense(module, src, rows, k, data, scales, stream)?;
            Ok(Packed { data, scales, gs, rows, k })
        };
        let mut layers = Vec::new();
        for (l, sl) in layer_slots.into_iter().enumerate() {
            let p = |s: &str| g(&format!("layers.{l}.{s}"));
            let pp = |s: &str| p(s).ptr;
            layers.push(DLayer {
                qkv: packed(sl[0], p("qkv"))?,
                wo: packed(sl[1], p("self_attn.o_proj.weight"))?,
                gu: packed(sl[2], p("gu"))?,
                down: packed(sl[3], p("mlp.down_proj.weight"))?,
                kv_ctx: packed(sl[4], p("kv_ctx"))?,
                input_norm: pp("input_layernorm.weight"),
                post_norm: pp("post_attention_layernorm.weight"),
                dflare_norm: pp("dflare_norm.weight"),
                q_norm: pp("self_attn.q_norm.weight"),
                k_norm: pp("self_attn.k_norm.weight"),
                k_norm_ctx: pp("self_attn.k_norm_ctx.weight"),
            });
        }

        // rnn_proj [768, 512 + 2 * 2304] -> w_sp, w_h, w_f.
        let mut a = Args::new();
        a.push(g("markov_head.rnn_proj.weight")).push(arena.get(w_sp)).push(arena.get(w_h)).push(arena.get(w_f));
        launch(kern(module, "z_split_dspark_rnn")?, RNN as u32, 256, false, stream, a)?;

        // Draft unembed rows (BF16 source for the fold, NVFP4 for the base logits).
        let unembed_bf16 = main.get("unembed.weight");
        let row_bytes = H * 2;
        let gathered = match shortlist {
            None => None,
            Some(s) => {
                let buf = inkling_runtime::DeviceBuffer::alloc(s.len() * row_bytes)?;
                for (i, &tok) in s.iter().enumerate() {
                    unembed_bf16.sub(tok as usize * row_bytes, row_bytes).copy_to(&buf.slice().sub(i * row_bytes, row_bytes), stream)?;
                }
                arena.get(ids.unwrap()).upload(s)?;
                Some(buf)
            }
        };
        let src_rows = gathered.as_ref().map_or(unembed_bf16.sub(0, vocab * row_bytes), |b| b.slice());
        let unembed_draft = match unembed_pruned {
            None => q.unembed,
            Some(sl) => packed(sl, src_rows)?,
        };

        // Fold the Markov head's hidden_out into the draft unembed: fold_b = U b, fold_w = U W (BF16).
        let fold_tmp = inkling_runtime::DeviceBuffer::alloc(vocab * MTP_MARKOV_RANK * 2)?;
        let mut a = Args::new();
        a.push(src_rows).push(g("markov_head.hidden_out.bias")).push(arena.get(fold_b)).push(vocab as i32).push(H as i32);
        launch(kern(module, "z_compute_dspark_fold_bias")?, vocab.div_ceil(8) as u32, (32u32, 8u32), false, stream, a)?;
        let mut a = Args::new();
        a.push(src_rows)
            .push(g("markov_head.hidden_out.weight"))
            .push(fold_tmp.slice())
            .push(vocab as i32)
            .push(H as i32)
            .push(MTP_MARKOV_RANK as i32);
        launch(kern(module, "z_compute_dspark_fold_w")?, vocab as u32, 256, false, stream, a)?;
        let fold_w = packed(fold, fold_tmp.slice())?;
        stream.synchronize()?;
        drop(gathered);
        drop(fold_tmp);

        let gp = |n: &str| g(n).ptr;
        Ok(DsparkWeights {
            layers,
            fuse_alpha: arena.get(fuse_alpha).ptr,
            mask_embed: gp("mask_embed"),
            norm_w: gp("norm.weight"),
            markov_dflare_norm: gp("markov_head.dflare_norm.weight"),
            markov_w1: gp("markov_head.markov_w1.weight"),
            rnn_bias: gp("markov_head.rnn_proj.bias"),
            w_sp: arena.get(w_sp).ptr,
            w_h: arena.get(w_h).ptr,
            w_f: arena.get(w_f).ptr,
            unembed_draft,
            fold_w,
            fold_b: arena.get(fold_b).ptr,
            vocab,
            vocab_ids: ids.map_or(NULL, |s| arena.get(s).ptr),
            arena,
            mtp,
            prepare_seconds: t.elapsed().as_secs_f64(),
        })
    }
}

macro_rules! ds_slots {
    ($v:ident; $($name:ident: $ty:ty = $count:expr),* $(,)?) => {
        struct DSlots { $($name: Slot,)* }
        fn plan_ds(p: &mut ArenaPlan, $v: usize) -> DSlots {
            DSlots { $($name: p.reserve_of::<$ty>(stringify!($name), $count),)* }
        }
    };
}

ds_slots! {
    vocab;
    f_ring: f32 = 16 * H,
    taps: f32 = 4 * B_MAX * H,
    mix: f32 = 3 * B_MAX * H,
    kv: f32 = B_MAX * 2 * dims::KV_DIM,
    block_x: f32 = BLOCK * H,
    normed: f32 = BLOCK * H,
    normed_h: u16 = BLOCK * H,
    qkv: f32 = BLOCK * QKV,
    attn_out_h: u16 = BLOCK * dims::Q_DIM,
    gu: f32 = BLOCK * 2 * MTP_INTERMEDIATE,
    swiglu_h: u16 = BLOCK * MTP_INTERMEDIATE,
    f_prev: f32 = H,
    c_f: f32 = RNN,
    c_h: f32 = NUM_DRAFTS * RNN,
    rnn_state: f32 = MTP_MARKOV_RANK,
    raw_rnn: f32 = RNN,
    blk_val: f32 = MARKOV_BLOCKS,
    blk_idx: i32 = MARKOV_BLOCKS,
    blk_cm: f32 = MARKOV_BLOCKS,
    blk_csum: f32 = MARKOV_BLOCKS,
    base_logits: f32 = NUM_DRAFTS * vocab,
    d_anchor: i32 = 1,
    d_drafts: i32 = B_MAX,
    d_conf: f32 = B_MAX,
    d_accepted: i32 = B_MAX,
    d_num_acc: i32 = 1,
    d_width: i32 = 1,
}

struct DKernels {
    mix: Kernel,
    f32_lin: [Kernel; 4],
    store_ctx: Kernel,
    store_ctx_fp8: Kernel,
    bulk_fp8: Kernel,
    init_block: Kernel,
    rmsnorm_h: Kernel,
    mma_k2304: Kernel,
    mma_k3072_add: Kernel,
    mma_k9216_add: Kernel,
    qk_rope: Kernel,
    bulk: Kernel,
    block_attn: Kernel,
    swiglu: Kernel,
    copy_f_prev: Kernel,
    bf16_b1: Kernel,
    bf16_b7: Kernel,
    base_logits: Kernel,
    rnn_proj: Kernel,
    stage1: Kernel,
    stage2: Kernel,
    assemble: Kernel,
    spec_verify: Kernel,
    /// The wide prefill's whole-chunk commit: mix, linear, BF16 store, FP8 store (`record_commit_wide`).
    wide: Option<[Kernel; 4]>,
}

/// The drafter's recorders, bound to its weights, its state and the target's state.
pub struct Dspark<'a> {
    pub w: &'a DsparkWeights,
    pub arena: Arena,
    s: DSlots,
    k: DKernels,
    /// Draft KV caches for the 3 layers: (k, v, cap).
    ctx: [(u64, u64, usize); MTP_LAYERS],
    /// FP8 draft context for the global layer (INKLING_DRAFT_KV_FP8=1): its (K scales, V scales); that
    /// layer's ctx K/V are then E4M3 code caches (u8, one f16 scale per slot and KV head).
    ctx_fp8: Option<(u64, u64)>,
    split_buf: u64,
    d_pos: u64,
    d_tokens: u64,
    d_next_tokens: u64,
    embed: u64,
    embed_norm: u64,
    pub pdl: bool,
    /// Run the commit on `aux` (forked from and joined back to the step's stream with events), so it
    /// overlaps the LM head, as squid5090 does. Works eagerly and under stream capture.
    pub commit_on_aux: bool,
    aux: Stream,
    fork: Event,
    join: Event,
}

fn p<T: KernelArg>(a: &mut Args, v: T) {
    a.push(v);
}

impl<'a> Dspark<'a> {
    pub fn new(module: &Module, main: &Weights, w: &'a DsparkWeights, st: &State) -> Result<Dspark<'a>, String> {
        let f = |n: &str| kern(module, n);
        let k = DKernels {
            mix: f("dspark_mix")?,
            f32_lin: [f("linear_nvfp4_f32_b1")?, f("linear_nvfp4_f32_b2")?, f("linear_nvfp4_f32_b4")?, f("linear_nvfp4_f32_b8")?],
            store_ctx: f("dspark_store_ctx_kv_bf16")?,
            store_ctx_fp8: f("dspark_store_ctx_kv_fp8")?,
            bulk_fp8: f("dspark_global_attn_bulk_splitk_fp8kv")?,
            init_block: f("dspark_init_block")?,
            rmsnorm_h: f("dspark_rmsnorm_h")?,
            mma_k2304: f("linear_nvfp4_mma_b8_k2304")?,
            mma_k3072_add: f("linear_nvfp4_mma_b8_k3072_add")?,
            mma_k9216_add: f("linear_nvfp4_mma_b8_k9216_add")?,
            qk_rope: f("dspark_block_qk_rope")?,
            bulk: f("dspark_global_attn_bulk_splitk_bf16kv")?,
            block_attn: f("dspark_block_attn")?,
            swiglu: f("dspark_swiglu")?,
            copy_f_prev: f("copy_f_prev_zero_state")?,
            bf16_b1: f("linear_bf16_nosmem_t_b1")?,
            bf16_b7: f("linear_bf16_nosmem_t_b7")?,
            base_logits: f("dspark_base_logits_nvfp4_mma_k7")?,
            rnn_proj: f("markov_rnn_proj")?,
            stage1: f("markov_fold_argmax_stage1_nvfp4")?,
            stage2: f("markov_fold_argmax_stage2")?,
            assemble: f("z_assemble_verify_tokens")?,
            spec_verify: f("spec_verify")?,
            wide: (|| {
                Some([
                    module.function("prefill_dspark_mix")?,
                    module.function("prefill_linear_nvfp4_f32")?,
                    module.function("prefill_dspark_store_ctx_kv_bf16")?,
                    module.function("prefill_dspark_store_ctx_kv_fp8")?,
                ])
            })()
            .filter(|_| std::env::var("INKLING_WIDE_COMMIT").map_or(true, |v| v != "0")),
        };
        let mut plan = ArenaPlan::new();
        let s = plan_ds(&mut plan, w.vocab);
        let caps = [DRAFT_LOCAL_CAP, st.max_ctx + B_MAX, DRAFT_LOCAL_CAP];
        let draft_fp8 = std::env::var_os("INKLING_DRAFT_KV_FP8").is_some_and(|v| v != "0");
        let fp8_scales = draft_fp8.then(|| plan.reserve_of::<u16>("ds.ctx_fp8_scales", 2 * caps[1] * dims::KV_HEADS));
        let kv_slots: Vec<(Slot, Slot, usize)> = caps
            .iter()
            .enumerate()
            .map(|(l, &cap)| {
                assert_eq!(is_local(l), cap == DRAFT_LOCAL_CAP);
                if draft_fp8 && !is_local(l) {
                    (
                        plan.reserve_of::<u8>(format!("ds.{l}.k8"), cap * dims::KV_DIM),
                        plan.reserve_of::<u8>(format!("ds.{l}.v8"), cap * dims::KV_DIM),
                        cap,
                    )
                } else {
                    (
                        plan.reserve_of::<u16>(format!("ds.{l}.k_cache"), cap * dims::KV_DIM),
                        plan.reserve_of::<u16>(format!("ds.{l}.v_cache"), cap * dims::KV_DIM),
                        cap,
                    )
                }
            })
            .collect();
        let arena = plan.allocate().map_err(|e| e.to_string())?;
        let ctx_fp8 = fp8_scales.map(|sl| {
            let ks = arena.get(sl).ptr;
            (ks, ks + (caps[1] * dims::KV_HEADS * 2) as u64)
        });
        let ctx = [0, 1, 2].map(|l| {
            let (kk, vv, cap) = kv_slots[l];
            (arena.get(kk).ptr, arena.get(vv).ptr, cap)
        });
        let q = |slot| st.get(slot).ptr;
        Ok(Dspark {
            w,
            ctx,
            ctx_fp8,
            split_buf: q(st.s.split_buf),
            d_pos: q(st.s.d_pos),
            d_tokens: q(st.s.d_tokens),
            d_next_tokens: q(st.s.d_next_tokens),
            embed: main.get("embed.weight").ptr,
            embed_norm: main.get("embed_norm.weight").ptr,
            arena,
            s,
            k,
            pdl: true,
            commit_on_aux: true,
            aux: Stream::new().map_err(|e| e.to_string())?,
            fork: Event::new_untimed().map_err(|e| e.to_string())?,
            join: Event::new_untimed().map_err(|e| e.to_string())?,
        })
    }

    fn g(&self, slot: Slot) -> u64 {
        self.arena.get(slot).ptr
    }

    /// The drafter's ring buffers (the f ring and the local layers' draft KV), for state snapshots.
    pub fn rolling_slices(&self) -> Vec<DevSlice> {
        let mut out = vec![self.arena.get(self.s.f_ring)];
        for &(k, v, cap) in &self.ctx {
            if cap == DRAFT_LOCAL_CAP {
                let bytes = cap * dims::KV_DIM * 2;
                out.push(DevSlice { ptr: k, bytes });
                out.push(DevSlice { ptr: v, bytes });
            }
        }
        out
    }

    pub fn slice(&self, which: DsBuf) -> DevSlice {
        let s = match which {
            DsBuf::Anchor => self.s.d_anchor,
            DsBuf::Drafts => self.s.d_drafts,
            DsBuf::Conf => self.s.d_conf,
            DsBuf::Accepted => self.s.d_accepted,
            DsBuf::NumAccepted => self.s.d_num_acc,
            DsBuf::Width => self.s.d_width,
        };
        self.arena.get(s)
    }

    /// Taps buffer `[4][B][H]` the target forward writes at visits 5, 23, 41 and after the final MLP.
    pub fn taps(&self) -> u64 {
        self.g(self.s.taps)
    }

    /// Zeroes the draft state that must not carry over between sequences (the f ring).
    pub fn reset(&self, stream: &Stream) -> Result<(), String> {
        self.arena.get(self.s.f_ring).fill_u8(0, stream)?;
        Ok(())
    }

    fn go(&self, k: Kernel, grid: impl Into<inkling_runtime::launch::Dim>, block: impl Into<inkling_runtime::launch::Dim>, pdl: bool, stream: &Stream, a: Args) -> Result<(), String> {
        launch(k, grid, block, pdl && self.pdl, stream, a)
    }

    /// `launch_dspark_commit`: from the taps of the `bsz` tokens just run by the target at positions
    /// `*d_pos + b`, writes the f ring and the draft context KV of every layer.
    pub fn record_commit(&self, stream: &Stream, bsz: i32) -> Result<(), String> {
        let (w, s) = (self.w, &self.s);
        let mut a = Args::new();
        p(&mut a, self.taps());
        p(&mut a, bsz);
        p(&mut a, w.fuse_alpha);
        for l in &w.layers {
            p(&mut a, l.dflare_norm);
        }
        p(&mut a, w.markov_dflare_norm);
        p(&mut a, self.g(s.mix));
        p(&mut a, self.g(s.f_ring));
        p(&mut a, self.d_pos);
        self.go(self.k.mix, (bsz as u32, 4u32), 256, false, stream, a)?;
        let bk = match bsz {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => return Err(format!("commit batch {bsz}")),
        };
        for (l, lw) in w.layers.iter().enumerate() {
            let kv = &lw.kv_ctx;
            let mut a = Args::new();
            p(&mut a, kv.data);
            p(&mut a, kv.scales);
            p(&mut a, kv.gs);
            p(&mut a, self.g(s.mix) + (l * bsz as usize * H * 4) as u64);
            p(&mut a, self.g(s.kv));
            p(&mut a, kv.rows as i32);
            p(&mut a, H as i32);
            self.go(self.k.f32_lin[bk], (kv.rows.div_ceil(4) as u32, 1u32), 128, false, stream, a)?;
            let (ck, cv, cap) = self.ctx[l];
            let mut a = Args::new();
            p(&mut a, self.g(s.kv));
            p(&mut a, lw.k_norm_ctx);
            p(&mut a, ck);
            p(&mut a, cv);
            let fp8 = self.ctx_fp8.filter(|_| !is_local(l));
            p(&mut a, fp8.map_or(NULL, |s| s.0));
            p(&mut a, fp8.map_or(NULL, |s| s.1));
            p(&mut a, cap as i32);
            p(&mut a, self.d_pos);
            let kern = if fp8.is_some() { self.k.store_ctx_fp8 } else { self.k.store_ctx };
            self.go(kern, (bsz as u32, dims::KV_HEADS as u32), 128, false, stream, a)?;
        }
        Ok(())
    }

    /// Whether `record_commit_wide` is available (the kernels are in the cubin and INKLING_WIDE_COMMIT is not 0).
    pub fn has_wide_commit(&self) -> bool {
        self.k.wide.is_some()
    }

    /// The commit for a whole wide-prefill chunk: `t` tokens at positions `*d_pos + i` with their taps in
    /// `taps` ([4][t][H] f32), in 1 + 2 * 3 launches instead of 7 per 8 tokens. `mix` ([3][t][H] f32) and `kv`
    /// ([t][2 KV_DIM] f32) are scratch. Per token the arithmetic is `record_commit`'s, and what is left in the
    /// f ring and in the local layers' rings is what B = 8 commits in order leave: the last 16 tokens, and the
    /// last `cap` tokens (earlier ones would be overwritten). So the draft state is bit-identical.
    pub fn record_commit_wide(&self, stream: &Stream, t: usize, taps: u64, mix: u64, kv_buf: u64) -> Result<(), String> {
        let [k_mix, k_lin, k_store, k_store_fp8] = self.k.wide.ok_or("no wide commit kernels")?;
        let w = self.w;
        let mut a = Args::new();
        p(&mut a, taps);
        p(&mut a, t as i32);
        p(&mut a, w.fuse_alpha);
        for l in &w.layers {
            p(&mut a, l.dflare_norm);
        }
        p(&mut a, w.markov_dflare_norm);
        p(&mut a, mix);
        p(&mut a, self.g(self.s.f_ring));
        p(&mut a, self.d_pos);
        self.go(k_mix, (t as u32, 4u32), 256, false, stream, a)?;
        // Tokens per CTA of the linear: enough rows of CTAs to fill the GPU, few weight decodes.
        let tb = (t / 16).clamp(1, 32);
        for (l, lw) in w.layers.iter().enumerate() {
            let kv = &lw.kv_ctx;
            let mut a = Args::new();
            p(&mut a, kv.data);
            p(&mut a, kv.scales);
            p(&mut a, kv.gs);
            p(&mut a, mix + (l * t * H * 4) as u64);
            p(&mut a, kv_buf);
            p(&mut a, kv.rows as i32);
            p(&mut a, H as i32);
            p(&mut a, t as i32);
            p(&mut a, tb as i32);
            self.go(k_lin, (kv.rows.div_ceil(4) as u32, t.div_ceil(tb) as u32), 128, false, stream, a)?;
            let (ck, cv, cap) = self.ctx[l];
            let off = t.saturating_sub(cap);
            let mut a = Args::new();
            p(&mut a, kv_buf);
            p(&mut a, lw.k_norm_ctx);
            p(&mut a, ck);
            p(&mut a, cv);
            let fp8 = self.ctx_fp8.filter(|_| !is_local(l));
            p(&mut a, fp8.map_or(NULL, |s| s.0));
            p(&mut a, fp8.map_or(NULL, |s| s.1));
            p(&mut a, cap as i32);
            p(&mut a, self.d_pos);
            p(&mut a, off as i32);
            let kern = if fp8.is_some() { k_store_fp8 } else { k_store };
            self.go(kern, ((t - off) as u32, dims::KV_HEADS as u32), 128, false, stream, a)?;
        }
        Ok(())
    }

    /// Starts the commit for `bsz` tokens: on the aux stream (forked from `stream`) when
    /// `commit_on_aux`, else inline. Call [`Dspark::finish_commit`] on `stream` before anything that
    /// must follow the commit (the next step, `inc_pos`, `spec_verify`).
    pub fn start_commit(&self, stream: &Stream, bsz: i32) -> Result<(), String> {
        if !self.commit_on_aux {
            return self.record_commit(stream, bsz);
        }
        self.fork.record(stream).map_err(|e| e.to_string())?;
        self.aux.wait(&self.fork).map_err(|e| e.to_string())?;
        self.record_commit(&self.aux, bsz)?;
        self.join.record(&self.aux).map_err(|e| e.to_string())
    }

    pub fn finish_commit(&self, stream: &Stream) -> Result<(), String> {
        if self.commit_on_aux { stream.wait(&self.join).map_err(|e| e.to_string()) } else { Ok(()) }
    }

    /// `linear_nvfp4_mma_b8_*` over the 8-row block.
    fn mma(&self, stream: &Stream, k: Kernel, w: &Packed, x_h: u64, y: u64, threads: u32) -> Result<(), String> {
        let mut a = Args::new();
        p(&mut a, w.data);
        p(&mut a, w.scales);
        p(&mut a, w.gs);
        p(&mut a, x_h);
        p(&mut a, y);
        p(&mut a, w.rows as i32);
        p(&mut a, 0u32);
        self.go(k, (w.rows.div_ceil(16) as u32, 1u32), threads, true, stream, a)
    }

    fn rmsnorm_h(&self, stream: &Stream, rows: u32, x: u64, w: u64, out_f: u64, out_h: u64) -> Result<(), String> {
        let mut a = Args::new();
        p(&mut a, x);
        p(&mut a, w);
        p(&mut a, out_f);
        p(&mut a, out_h);
        self.go(self.k.rmsnorm_h, rows, 256, true, stream, a)
    }

    /// `launch_dspark_draft(num_drafts = 7)`: 7 drafts after the anchor `d_anchor` at position `*d_pos`,
    /// into `d_drafts[0..7)` with confidences in `d_conf[0..7)`.
    pub fn record_draft(&self, stream: &Stream) -> Result<(), String> {
        let (w, s, k) = (self.w, &self.s, &self.k);
        let (block_x, normed, normed_h, qkv) = (self.g(s.block_x), self.g(s.normed), self.g(s.normed_h), self.g(s.qkv));
        let (attn_out_h, gu, swiglu_h) = (self.g(s.attn_out_h), self.g(s.gu), self.g(s.swiglu_h));

        let mut a = Args::new();
        p(&mut a, self.embed);
        p(&mut a, self.embed_norm);
        p(&mut a, w.mask_embed);
        p(&mut a, self.g(s.d_anchor));
        p(&mut a, block_x);
        self.go(k.init_block, BLOCK as u32, 256, true, stream, a)?;

        for (l, lw) in w.layers.iter().enumerate() {
            let (ck, cv, cap) = self.ctx[l];
            let local = is_local(l);
            self.rmsnorm_h(stream, BLOCK as u32, block_x, lw.input_norm, NULL, normed_h)?;
            self.mma(stream, k.mma_k2304, &lw.qkv, normed_h, qkv, 192)?;
            let mut a = Args::new();
            p(&mut a, qkv);
            p(&mut a, lw.q_norm);
            p(&mut a, lw.k_norm);
            p(&mut a, self.d_pos);
            self.go(k.qk_rope, (BLOCK as u32, (dims::HEADS + dims::KV_HEADS) as u32), 128, true, stream, a)?;
            if !local {
                let mut a = Args::new();
                p(&mut a, qkv);
                p(&mut a, ck);
                p(&mut a, cv);
                p(&mut a, NULL);
                p(&mut a, NULL);
                if let Some((ks, vs)) = self.ctx_fp8 {
                    let mut a = Args::new();
                    p(&mut a, qkv);
                    p(&mut a, ck);
                    p(&mut a, cv);
                    p(&mut a, ks);
                    p(&mut a, vs);
                    p(&mut a, cap as i32);
                    p(&mut a, self.d_pos);
                    p(&mut a, self.split_buf);
                    self.go(k.bulk_fp8, (dims::KV_HEADS as u32, 64u32), 128, true, stream, a)?;
                } else {
                p(&mut a, cap as i32);
                p(&mut a, self.d_pos);
                p(&mut a, self.split_buf);
                self.go(k.bulk, (dims::KV_HEADS as u32, 64u32), 128, true, stream, a)?;
                }
            }
            let mut a = Args::new();
            p(&mut a, qkv);
            p(&mut a, ck);
            p(&mut a, cv);
            p(&mut a, if local { NULL } else { self.split_buf });
            p(&mut a, cap as i32);
            p(&mut a, if local { dims::WINDOW as i32 } else { 0i32 });
            p(&mut a, self.d_pos);
            p(&mut a, NULL);
            p(&mut a, attn_out_h);
            self.go(k.block_attn, (dims::HEADS as u32, BLOCK as u32), 256, true, stream, a)?;
            self.mma(stream, k.mma_k3072_add, &lw.wo, attn_out_h, block_x, 256)?;
            self.rmsnorm_h(stream, BLOCK as u32, block_x, lw.post_norm, NULL, normed_h)?;
            self.mma(stream, k.mma_k2304, &lw.gu, normed_h, gu, 192)?;
            let mut a = Args::new();
            p(&mut a, gu);
            p(&mut a, NULL);
            p(&mut a, swiglu_h);
            p(&mut a, BLOCK as i32);
            self.go(k.swiglu, ((BLOCK * MTP_INTERMEDIATE).div_ceil(256) as u32, 1u32), 256, true, stream, a)?;
            self.mma(stream, k.mma_k9216_add, &lw.down, swiglu_h, block_x, 256)?;
        }

        // Final norm on rows 1..7 -> normed (f32, for rnn_w_h) and normed_h (f16, for the unembed).
        self.rmsnorm_h(stream, NUM_DRAFTS as u32, block_x + (H * 4) as u64, w.norm_w, normed, normed_h)?;
        let mut a = Args::new();
        p(&mut a, self.g(s.f_ring));
        p(&mut a, self.d_pos);
        p(&mut a, self.g(s.f_prev));
        p(&mut a, self.g(s.rnn_state));
        self.go(k.copy_f_prev, (H.div_ceil(256) as u32, 1u32), 256, true, stream, a)?;
        for (kern, x, y, rows, pdl) in [(k.bf16_b1, self.g(s.f_prev), self.g(s.c_f), w.w_f, false), (k.bf16_b7, normed, self.g(s.c_h), w.w_h, true)] {
            let mut a = Args::new();
            p(&mut a, rows);
            p(&mut a, x);
            p(&mut a, y);
            p(&mut a, RNN as i32);
            p(&mut a, H as i32);
            self.go(kern, ((RNN / 4) as u32, 1u32), 128, pdl, stream, a)?;
        }
        let u = &w.unembed_draft;
        let mut a = Args::new();
        p(&mut a, u.data);
        p(&mut a, u.scales);
        p(&mut a, u.gs);
        p(&mut a, normed_h);
        p(&mut a, w.fold_b);
        p(&mut a, self.g(s.base_logits));
        p(&mut a, w.vocab as i32);
        p(&mut a, 0u32);
        self.go(k.base_logits, UNEMBED_BLOCKS, 192, true, stream, a)?;

        let drafts = self.g(s.d_drafts);
        for i in 0..NUM_DRAFTS {
            let mut a = Args::new();
            p(&mut a, w.w_sp);
            p(&mut a, w.rnn_bias);
            p(&mut a, w.markov_w1);
            p(&mut a, self.g(s.c_h) + (i * RNN * 4) as u64);
            p(&mut a, self.g(s.c_f));
            p(&mut a, if i == 0 { self.g(s.d_anchor) } else { drafts + ((i - 1) * 4) as u64 });
            p(&mut a, self.g(s.rnn_state));
            p(&mut a, self.g(s.raw_rnn));
            self.go(k.rnn_proj, 24u32, 128, true, stream, a)?;
            let fw = &w.fold_w;
            let mut a = Args::new();
            p(&mut a, fw.data);
            p(&mut a, fw.scales);
            p(&mut a, fw.gs);
            p(&mut a, self.g(s.base_logits) + (i * w.vocab * 4) as u64);
            p(&mut a, NULL); // fold_b is already in the base logits
            p(&mut a, self.g(s.raw_rnn));
            p(&mut a, self.g(s.rnn_state));
            p(&mut a, self.g(s.blk_val));
            p(&mut a, self.g(s.blk_idx));
            p(&mut a, self.g(s.blk_cm));
            p(&mut a, self.g(s.blk_csum));
            p(&mut a, w.vocab as i32);
            self.go(k.stage1, MARKOV_BLOCKS as u32, 256, true, stream, a)?;
            let mut a = Args::new();
            p(&mut a, self.g(s.blk_val));
            p(&mut a, self.g(s.blk_idx));
            p(&mut a, self.g(s.blk_cm));
            p(&mut a, self.g(s.blk_csum));
            p(&mut a, w.vocab_ids);
            p(&mut a, drafts + (i * 4) as u64);
            p(&mut a, self.g(s.d_conf) + (i * 4) as u64);
            self.go(k.stage2, 1u32, 256, true, stream, a)?;
        }
        Ok(())
    }

    /// `z_assemble_verify_tokens`: d_tokens = [anchor, drafts[0..num_drafts)].
    pub fn record_assemble(&self, stream: &Stream, num_drafts: i32) -> Result<(), String> {
        let mut a = Args::new();
        p(&mut a, self.g(self.s.d_anchor));
        p(&mut a, self.g(self.s.d_drafts));
        p(&mut a, self.d_tokens);
        p(&mut a, num_drafts);
        self.go(self.k.assemble, 1u32, (num_drafts + 1) as u32, false, stream, a)
    }

    /// `z_adaptive_select_width` (CUDA C++, from its own cubin): picks the verify width 2, 4 or 8 from
    /// the leading drafts with confidence >= `threshold`, writes `d_tokens` and the width, and sets the
    /// switch node `handle` to case 0, 1 or 2. Grid 1, 1 thread.
    pub fn record_select_width(&self, stream: &Stream, select: Kernel, threshold: f32, handle: u64) -> Result<(), String> {
        let mut a = Args::new();
        p(&mut a, self.g(self.s.d_anchor));
        p(&mut a, self.g(self.s.d_drafts));
        p(&mut a, self.g(self.s.d_conf));
        p(&mut a, threshold);
        p(&mut a, self.d_tokens);
        p(&mut a, self.g(self.s.d_width));
        p(&mut a, handle);
        self.go(select, 1u32, 1u32, false, stream, a)
    }

    /// `spec_verify` for a verify of width `bsz`: accepted tokens, count, `*d_pos += n`, next anchor.
    pub fn record_spec_verify(&self, stream: &Stream, bsz: i32) -> Result<(), String> {
        let mut a = Args::new();
        p(&mut a, self.d_next_tokens);
        p(&mut a, self.g(self.s.d_drafts));
        p(&mut a, bsz);
        p(&mut a, self.g(self.s.d_accepted));
        p(&mut a, self.g(self.s.d_num_acc));
        p(&mut a, self.d_pos);
        p(&mut a, self.g(self.s.d_anchor));
        self.go(self.k.spec_verify, 1u32, 1u32, false, stream, a)
    }

    pub fn total_bytes(&self) -> usize {
        self.arena.total_bytes()
    }
}

#[derive(Clone, Copy)]
pub enum DsBuf {
    Anchor,
    Drafts,
    Conf,
    Accepted,
    NumAccepted,
    Width,
}
