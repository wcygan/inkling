//! The exact-mode B=1 token step: 835 launches, issued in squid5090's order with its parameter lists. The same code runs
//! eagerly (prefill, debugging) or under stream capture (decode graph).

use crate::dspark::Dspark;
use crate::fp8::Fp8Dense;
use crate::nvfp4::{Nvfp4Weights, Packed};
use crate::state::{B_MAX, LM_HEAD_BLOCKS, State};
use inkling_model::weights::Weights;
use inkling_model::{SCHEDULE, dims, is_global_layer};
use inkling_runtime::{Args, DevSlice, Kernel, KernelArg, LaunchConfig, Module, PtrPack, Stream};

const H: usize = dims::HIDDEN;
const NULL: u64 = 0;
type P3 = PtrPack<3>;
type P9 = PtrPack<9>;

/// Weight precision of the token step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// BF16 dense weights and LM head, experts as stored (NVFP4 layers 0–19, MXFP8 layers 20–23).
    Exact,
    /// Everything NVFP4 (dense re-quantised at load, MXFP8 experts converted), half-precision
    /// attention inputs. squid5090's `--mode nvfp4`.
    Nvfp4,
}

/// Which weights run in NVFP4 in nvfp4 mode. Each is an independent accuracy trade (spec gate G3);
/// parts left off run the exact-mode kernels on the BF16 / checkpoint weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4Parts {
    /// Q|R|gate|sink, K|V and O projections (half-precision attention inputs).
    pub dense: bool,
    /// Cross-pass controllers.
    pub ctrl: bool,
    pub router: bool,
    /// LM head.
    pub head: bool,
    /// MXFP8 experts of layers 20–23 converted to NVFP4.
    pub experts: bool,
    /// With `dense`: the dense projections run in FP8 (E4M3 + f32 row scale) instead of NVFP4
    /// (half-precision attention inputs, f32 accumulation). Supported at every B.
    pub dense_fp8: bool,
}

impl Nvfp4Parts {
    pub const ALL: Nvfp4Parts = Nvfp4Parts { dense: true, ctrl: true, router: true, head: true, experts: true, dense_fp8: false };
    pub const NONE: Nvfp4Parts = Nvfp4Parts { dense: false, ctrl: false, router: false, head: false, experts: false, dense_fp8: false };
    /// FP8 dense projections, everything else NVFP4. Supported at B > 1.
    pub const BATCH_FP8_DENSE: Nvfp4Parts = Nvfp4Parts { dense: true, ctrl: true, router: true, head: true, experts: true, dense_fp8: true };
    /// Route B: exact BF16 dense projections, everything else NVFP4. Supported at B > 1.
    pub const BATCH_EXACT_DENSE: Nvfp4Parts = Nvfp4Parts { dense: false, ctrl: true, router: true, head: true, experts: true, dense_fp8: false };

    /// Parses `dense,ctrl,router,head,experts` (or `all` / `none`).
    pub fn parse(spec: &str) -> Result<Nvfp4Parts, String> {
        let mut p = Nvfp4Parts::NONE;
        for part in spec.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            match part {
                "all" => p = Nvfp4Parts::ALL,
                "none" => p = Nvfp4Parts::NONE,
                "dense" => p.dense = true,
                "dense_fp8" => {
                    p.dense = true;
                    p.dense_fp8 = true;
                }
                "ctrl" => p.ctrl = true,
                "router" => p.router = true,
                "head" => p.head = true,
                "experts" => p.experts = true,
                other => return Err(format!("unknown nvfp4 part {other}")),
            }
        }
        Ok(p)
    }
}

/// `CtrlNVFP4Ptr3` (192 B): for down, forget, gain: `data[3]`, `scales[3]`, `gs[3]` (f32), 4 B pad.
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

fn ptr(s: DevSlice) -> u64 {
    s.ptr
}

/// Kernels resolved by name from the inkling-kernels cubin.
pub struct Kernels {
    embed_norm: Kernel,
    rmsnorm: Kernel,
    extract_block: Kernel,
    fused_depth_norm: Kernel,
    mlp_post_next: Kernel,
    attn_post_mlp_in: Kernel,
    mlp_post_res: Kernel,
    ctrl_prep: Kernel,
    ctrl_down: Kernel,
    ctrl_up: Kernel,
    batched_rmsnorm: Kernel,
    linear: Kernel,
    local_prep: Kernel,
    global_prep: Kernel,
    bulk_attn: Kernel,
    tail_attn: Kernel,
    merge: Kernel,
    route_logits: Kernel,
    // Loaded with the module but no longer launched: `route_select_fast` replaces them.
    #[allow(dead_code)]
    route_select: Kernel,
    moe_up: [Kernel; 2],
    moe_down: [Kernel; 2],
    lm_head1: Kernel,
    lm_head2: Kernel,
    inc_pos: Kernel,
    qkvrg_nvfp4: Kernel,
    linear_nvfp4_k3072: Kernel,
    route_logits_nvfp4: Kernel,
    controller_nvfp4: Kernel,
    /// `mega_controller_nvfp4_cl8` (8-CTA clusters, grid (8B, n), 288 threads), if the cubin has it.
    controller_nvfp4_cl8: Option<Kernel>,
    /// `mega_post_next_ctrl_cl8` / `mega_attn_post_ctrl_cl8`: the loop-visit depth-norm / post kernel
    /// fused into the cl8 controller, if the cubin has them.
    post_next_ctrl: Option<Kernel>,
    attn_post_ctrl: Option<Kernel>,
    lm_head1_nvfp4: Kernel,
    /// B = 2, 4, 8 variants (index 0, 1, 2).
    qkvrg_mma: [Kernel; 3],
    /// Exact BF16 linears at B = 2, 4, 8 (dense-off batched steps), if the cubin has them.
    linear_b: Option<[Kernel; 3]>,
    /// FP8 dense at B = 1, 2, 4, 8 (index 0..4): fused Q|R|gate|sink + K + V, and O (K = 3072).
    qkvrg_fp8: Option<[Kernel; 4]>,
    linear_fp8_k3072: Option<[Kernel; 4]>,
    linear_mma_k3072: [Kernel; 3],
    bulk_attn_b: [Kernel; 3],
    /// FP8 KV bulk (prototype), B = 1/2/4/8.
    bulk_fp8: [Kernel; 4],
    tail_attn_b: [Kernel; 3],
    route_logits_nvfp4_b: [Kernel; 3],
    #[allow(dead_code)]
    route_select_b: [Kernel; 3],
    /// small-ops (`router.rs`): the redux top-8 select, bit-identical to `route_select_b`.
    route_select_fast: [Kernel; 4],
    lm_head_mma: [Kernel; 3],
}

/// Resolves the B = 2, 4, 8 variants of a kernel name pattern (`{}` is replaced by B).
fn v(f: &dyn Fn(&str) -> Result<Kernel, String>, pattern: &str) -> Result<[Kernel; 3], String> {
    Ok([f(&pattern.replace("{}", "2"))?, f(&pattern.replace("{}", "4"))?, f(&pattern.replace("{}", "8"))?])
}

/// The B = 1, 2, 4, 8 variants of a kernel name pattern, if all are in the cubin.
fn v4(m: &Module, pattern: &str) -> Option<[Kernel; 4]> {
    Some([m.function(&pattern.replace("{}", "1"))?, m.function(&pattern.replace("{}", "2"))?, m.function(&pattern.replace("{}", "4"))?, m.function(&pattern.replace("{}", "8"))?])
}

/// Index of a B = 1, 2, 4, 8 variant.
fn bi4(bsz: i32) -> usize {
    match bsz {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => panic!("unsupported batch {bsz}"),
    }
}

/// Index of a B > 1 variant.
fn bi(bsz: i32) -> usize {
    match bsz {
        2 => 0,
        4 => 1,
        8 => 2,
        _ => panic!("unsupported batch {bsz}"),
    }
}

impl Kernels {
    pub fn resolve(m: &Module) -> Result<Kernels, String> {
        let f = |name: &str| m.function(name).ok_or_else(|| format!("kernel {name} missing from the cubin"));
        Ok(Kernels {
            embed_norm: f("embed_norm")?,
            rmsnorm: f("rmsnorm")?,
            extract_block: f("extract_block")?,
            fused_depth_norm: f("fused_depth_norm")?,
            mlp_post_next: f("mega_mlp_post_next_depth_norm")?,
            attn_post_mlp_in: f("mega_attn_post_mlp_in")?,
            mlp_post_res: f("mega_mlp_post_res")?,
            ctrl_prep: f("ctrl_prep_cat_batched")?,
            ctrl_down: f("ctrl_down_batched_b1")?,
            ctrl_up: f("ctrl_up_merge_batched")?,
            batched_rmsnorm: f("batched_rmsnorm")?,
            linear: f("linear_bf16_nosmem_t_b1")?,
            local_prep: f("mega_local_sconv_prep_store")?,
            global_prep: f("mega_global_sconv_prep_store_bf16kv")?,
            bulk_attn: f("global_attn_bulk_splitk_bf16kv_b1")?,
            tail_attn: f("local_tail_attn_tc_bf16kv_b1")?,
            merge: f("attn_merge_sink_gate")?,
            route_logits: f("route_logits_parallel_b1")?,
            route_select: f("route_select_and_group_b1")?,
            moe_up: [f("moe_up_tc_nvfp4")?, f("moe_up_tc_mxfp8")?],
            moe_down: [f("moe_down_tc_nvfp4")?, f("moe_down_tc_mxfp8")?],
            lm_head1: f("lm_head_stage1_bf16_b1")?,
            lm_head2: f("lm_head_stage2_argmax")?,
            inc_pos: f("inc_pos_and_copy_tok")?,
            qkvrg_nvfp4: f("qkvrg_fused_nvfp4_b1_pdl")?,
            linear_nvfp4_k3072: f("linear_nvfp4_b1_pdl_k3072")?,
            route_logits_nvfp4: f("route_logits_nvfp4_parallel_b1")?,
            controller_nvfp4: f("mega_controller_nvfp4")?,
            controller_nvfp4_cl8: m.function("mega_controller_nvfp4_cl8"),
            post_next_ctrl: m.function("mega_post_next_ctrl_cl8"),
            attn_post_ctrl: m.function("mega_attn_post_ctrl_cl8"),
            lm_head1_nvfp4: f("lm_head_stage1_nvfp4_b1")?,
            qkvrg_mma: v(&f, "qkvrg_fused_nvfp4_mma_b{}")?,
            linear_b: v(&f, "linear_bf16_nosmem_t_b{}").ok(),
            qkvrg_fp8: v4(m, "qkvrg_fused_fp8_mma_b{}"),
            linear_fp8_k3072: v4(m, "linear_fp8_mma_b{}_k3072"),
            linear_mma_k3072: v(&f, "linear_nvfp4_mma_b{}_k3072")?,
            bulk_attn_b: v(&f, "global_attn_bulk_splitk_bf16kv_b{}")?,
            bulk_fp8: [
                f("global_attn_bulk_splitk_fp8kv_b1")?,
                f("global_attn_bulk_splitk_fp8kv_b2")?,
                f("global_attn_bulk_splitk_fp8kv_b4")?,
                f("global_attn_bulk_splitk_fp8kv_b8")?,
            ],
            tail_attn_b: v(&f, "local_tail_attn_tc_bf16kv_b{}")?,
            route_logits_nvfp4_b: v(&f, "route_logits_nvfp4_parallel_b{}")?,
            route_select_b: v(&f, "route_select_and_group_b{}")?,
            route_select_fast: [
                f("route_select_fast_b1")?,
                f("route_select_fast_b2")?,
                f("route_select_fast_b4")?,
                f("route_select_fast_b8")?,
            ],
            lm_head_mma: v(&f, "lm_head_nvfp4_mma_stage1_b{}")?,
        })
    }
}

/// One layer's weights, resolved once.
struct LayerW {
    global: bool,
    /// Local: `{attn_norm, -, -}`; global: `{q,k,v}_input_norm`.
    in_norms: [u64; 3],
    mlp_norm: u64,
    qrgs: u64,
    w_k: u64,
    w_v: u64,
    w_o: u64,
    k_sconv: u64,
    v_sconv: u64,
    k_sconv_proj: u64,
    v_sconv_proj: u64,
    q_clip: u64,
    rel_clip: u64,
    rel_proj: u64,
    attn_sconv: u64,
    mlp_sconv: u64,
    /// `[attn, k, v, mlp]` controllers as `(down, forget, gain)`; layers 6–17 only.
    ctrl: Option<[[u64; 3]; 4]>,
    router_w: u64,
    router_b: u64,
    /// Expert format: 0 NVFP4, 1 MXFP8.
    fmt: usize,
    w13: u64,
    w13_scales: u64,
    w13_factor: u64,
    w2: u64,
    w2_scales: u64,
    w2_factor: u64,
    attn_query: u64,
    mlp_query: u64,
}

struct ModelW {
    embed: u64,
    embed_norm: u64,
    final_norm: u64,
    unembed: u64,
    output_query: u64,
    cross_pass_q: u64,
    cross_pass_kv_q: u64,
    layers: Vec<LayerW>,
}

fn resolve_weights(w: &Weights) -> ModelW {
    let g = |n: &str| ptr(w.get(n));
    let layers = (0..dims::LAYERS)
        .map(|l| {
            let p = |s: &str| g(&format!("layers.{l}.{s}"));
            let global = is_global_layer(l);
            let fmt = if l < dims::NVFP4_EXPERT_LAYERS { 0 } else { 1 };
            let kv = w.get(&format!("layers.{l}.attn.kv"));
            let ctrl = (dims::LOOP_START..dims::LOOP_END).contains(&l).then(|| {
                ["attn", "k", "v", "mlp"].map(|c| {
                    ["down", "forget", "gain"].map(|m| p(&format!("controllers.{c}.{m}_weight")))
                })
            });
            LayerW {
                global,
                in_norms: if global {
                    [p("q_input_norm.weight"), p("k_input_norm.weight"), p("v_input_norm.weight")]
                } else {
                    [p("attn_norm.weight"), NULL, NULL]
                },
                mlp_norm: p("mlp_norm.weight"),
                qrgs: p("attn.qrgs"),
                w_k: kv.ptr,
                w_v: kv.ptr + (dims::KV_DIM * H * 2) as u64,
                w_o: p("attn.wo_ud.weight"),
                k_sconv: if global { p("attn.k_sconv.static_w") } else { p("attn.k_sconv.weight") },
                v_sconv: if global { p("attn.v_sconv.static_w") } else { p("attn.v_sconv.weight") },
                k_sconv_proj: if global { p("attn.k_sconv.weight_proj") } else { NULL },
                v_sconv_proj: if global { p("attn.v_sconv.weight_proj") } else { NULL },
                q_clip: p("attn.q_l2_clip.scale"),
                rel_clip: p("attn.rel_l2_clip.scale"),
                rel_proj: p("attn.rel_logits_proj.proj"),
                attn_sconv: p("attn_sconv.weight"),
                mlp_sconv: p("mlp_sconv.weight"),
                ctrl,
                router_w: p("mlp.gate.weight"),
                router_b: p("mlp.gate.bias"),
                fmt,
                w13: p("mlp.experts.w13_weight"),
                w13_scales: p("mlp.experts.w13_weight_scale"),
                w13_factor: if fmt == 0 { p("mlp.experts.w13_global_scale") } else { NULL },
                w2: p("mlp.experts.w2_weight"),
                w2_scales: p("mlp.experts.w2_weight_scale"),
                w2_factor: if fmt == 0 { p("mlp.experts.w2_global_scale") } else { NULL },
                attn_query: if l > 0 { g(&format!("attn_res.attn_queries.{}", l - 1)) } else { NULL },
                mlp_query: g(&format!("attn_res.mlp_queries.{l}")),
            }
        })
        .collect();
    ModelW {
        embed: g("embed.weight"),
        embed_norm: g("embed_norm.weight"),
        final_norm: g("norm.weight"),
        unembed: g("unembed.weight"),
        output_query: g("attn_res.output_query"),
        cross_pass_q: g("middle_cycle_cross_pass_queries"),
        cross_pass_kv_q: g("middle_cycle_cross_pass_kv_queries"),
        layers,
    }
}

/// Buffer addresses from the state arena.
struct Buf {
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
    attn_gated_h: u64,
    qrgs: u64,
    k_proj: u64,
    v_proj: u64,
    attn_gated: u64,
    split_buf: u64,
    ctrl_msg: u64,
    ctrl_cat: u64,
    ctrl_d: u64,
    router_logits: u64,
    route_ids: u64,
    route_w: u64,
    moe_act_h: u64,
    moe_down: u64,
    lm_blk_val: u64,
    lm_blk_idx: u64,
    lm_logits: u64,
    sample_cfg: u64,
    d_pos: u64,
    d_tokens: u64,
    d_next_tokens: u64,
}

pub struct Forward<'a> {
    mode: Mode,
    /// Parts that run in NVFP4 (all of them in nvfp4 mode unless narrowed for G3 measurements).
    pub parts: Nvfp4Parts,
    q: Option<&'a Nvfp4Weights>,
    k: Kernels,
    w: ModelW,
    st: &'a State,
    b: Buf,
    hidden_slice: DevSlice,
    prelude_slice: DevSlice,
    /// Launch the kernels squid5090 marks PDL with programmatic dependent launch.
    pub pdl: bool,
    /// Write the softcapped logits of each step to `lm_logits` (parity runs). Greedy decode leaves this
    /// off, as squid5090 does: the LM head then only reduces per-block maxima. Implemented by passing a
    /// null `SampleConfig*`, which makes stage 1 write logits and stage 2 take the plain argmax.
    pub write_logits: bool,
    /// DSpark speculative decoding: when set, every step writes the drafter's taps (visits 5, 23, 41 and
    /// the final hidden state) and ends with its commit (draft context KV for the step's positions).
    pub dspark: Option<&'a Dspark<'a>>,
    /// Run the NVFP4 controller as `mega_controller_nvfp4_cl8` (+4.9% at B=1,
    /// bit-identical output). On by default when the cubin has it; `INKLING_CTRL_V2=0` falls back.
    pub ctrl_v2: bool,
    /// On loop visits, run the depth-norm / post kernel inside the cl8 controller (one launch instead
    /// of two; bit-identical). Needs `ctrl_v2` and NVFP4 controllers; `INKLING_FUSE_CTRL=0` turns it off.
    pub fuse_ctrl: bool,
    /// FP8 dense weights, packed at construction when `INKLING_NVFP4_PARTS` names `dense_fp8`.
    pub fp8: Option<Fp8Dense>,
}

/// Visits whose updated hidden state feeds the drafter (squid5090 `mtp_target_visits[0..3]`).
const TAP_VISITS: [usize; 3] = [5, 23, 41];

struct Launcher<'s> {
    stream: &'s Stream,
    pdl: bool,
    /// Tokens in this step (B): 1 for decode, 2/4/8 for verify or chunked prefill.
    bsz: i32,
}

impl Launcher<'_> {
    fn go(&self, k: Kernel, grid: (u32, u32), block: u32, pdl: bool, mut args: Args) -> Result<(), String> {
        let cfg = LaunchConfig::new(grid, block).pdl(pdl && self.pdl);
        unsafe { k.launch(&cfg, self.stream, &mut args) }.map_err(|e| e.to_string())
    }
}

fn a<T: KernelArg>(args: &mut Args, v: T) {
    args.push(v);
}

impl<'a> Forward<'a> {
    pub fn new(module: &Module, weights: &Weights, st: &'a State) -> Result<Forward<'a>, String> {
        Forward::with_mode(module, weights, None, st)
    }

    /// `nvfp4 = Some(..)` selects nvfp4 mode.
    pub fn with_mode(
        module: &Module,
        weights: &Weights,
        nvfp4: Option<&'a Nvfp4Weights>,
        st: &'a State,
    ) -> Result<Forward<'a>, String> {
        let s = &st.s;
        let q = |slot| ptr(st.get(slot));
        let b = Buf {
            hidden: q(s.hidden),
            prelude: q(s.prelude),
            partial: q(s.partial),
            saved: q(s.saved),
            prior_last: q(s.prior_last),
            residual: q(s.residual),
            o_proj: q(s.o_proj),
            mlp_in: q(s.mlp_in),
            mlp_normed: q(s.mlp_normed),
            mlp_normed_h: q(s.mlp_normed_h),
            blocks: q(s.blocks),
            attn_in: [q(s.attn_in0), q(s.attn_in1), q(s.attn_in2)],
            attn_in_h: [q(s.attn_in_h0), q(s.attn_in_h1), q(s.attn_in_h2)],
            attn_gated_h: q(s.attn_gated_h),
            qrgs: q(s.qrgs),
            k_proj: q(s.k_proj),
            v_proj: q(s.v_proj),
            attn_gated: q(s.attn_gated),
            split_buf: q(s.split_buf),
            ctrl_msg: q(s.ctrl_msg),
            ctrl_cat: q(s.ctrl_cat),
            ctrl_d: q(s.ctrl_d),
            router_logits: q(s.router_logits),
            route_ids: q(s.route_ids),
            route_w: q(s.route_w),
            moe_act_h: q(s.moe_act_h),
            moe_down: q(s.moe_down),
            lm_blk_val: q(s.lm_blk_val),
            lm_blk_idx: q(s.lm_blk_idx),
            lm_logits: q(s.lm_logits),
            sample_cfg: q(s.sample_cfg),
            d_pos: q(s.d_pos),
            d_tokens: q(s.d_tokens),
            d_next_tokens: q(s.d_next_tokens),
        };
        Ok(Forward {
            mode: if nvfp4.is_some() { Mode::Nvfp4 } else { Mode::Exact },
            parts: if nvfp4.is_some() { Nvfp4Parts::ALL } else { Nvfp4Parts::NONE },
            q: nvfp4,
            k: Kernels::resolve(module)?,
            w: resolve_weights(weights),
            st,
            hidden_slice: st.get(s.hidden),
            prelude_slice: st.get(s.prelude),
            b,
            pdl: true,
            write_logits: false,
            dspark: None,
            ctrl_v2: std::env::var("INKLING_CTRL_V2").map_or(true, |v| v != "0"),
            fuse_ctrl: std::env::var("INKLING_FUSE_CTRL").map_or(true, |v| v != "0"),
            fp8: match (nvfp4, std::env::var("INKLING_NVFP4_PARTS")) {
                (Some(_), Ok(p)) if p.split(',').any(|x| x.trim() == "dense_fp8") => {
                    let s = Stream::new().map_err(|e| e.to_string())?;
                    Some(Fp8Dense::pack(module, weights, &s)?)
                }
                _ => None,
            },
        })
    }

    /// The four-kernel cross-pass controller (`launch_controller_merge_batched`).
    #[allow(clippy::too_many_arguments)]
    fn controllers(&self, l: &Launcher, n: u32, xs: P3, outs: P3, outs_h: P3, cpq: P3, cws: P9, post: P3) -> Result<(), String> {
        let (b, k) = (&self.b, &self.k);
        let mut x = Args::new();
        a(&mut x, xs);
        a(&mut x, b.saved);
        a(&mut x, b.prior_last);
        a(&mut x, cpq);
        a(&mut x, b.ctrl_msg);
        a(&mut x, b.ctrl_cat);
        a(&mut x, l.bsz);
        l.go(k.ctrl_prep, (l.bsz as u32, n), 256, false, x)?;
        let mut x = Args::new();
        a(&mut x, cws);
        a(&mut x, b.ctrl_cat);
        a(&mut x, b.ctrl_d);
        l.go(k.ctrl_down, (16, n), 128, false, x)?;
        let mut x = Args::new();
        a(&mut x, cws);
        a(&mut x, xs);
        a(&mut x, b.ctrl_msg);
        a(&mut x, b.ctrl_d);
        a(&mut x, outs);
        a(&mut x, l.bsz);
        l.go(k.ctrl_up, (36, n), 256, false, x)?;
        let mut x = Args::new();
        a(&mut x, outs);
        a(&mut x, outs_h);
        a(&mut x, post);
        l.go(k.batched_rmsnorm, (l.bsz as u32, n), 256, false, x)
    }

    /// One-launch NVFP4 cross-pass controller (`mega_controller_nvfp4`), grid (B, n), block 1024.
    #[allow(clippy::too_many_arguments)]
    fn controller_nvfp4(&self, l: &Launcher, n: u32, xs: P3, cpq: P3, cws: CtrlNvfp4, post: P3, outs: P3, outs_h: P3) -> Result<(), String> {
        let b = &self.b;
        let mut x = Args::new();
        a(&mut x, xs);
        a(&mut x, b.saved);
        a(&mut x, b.prior_last);
        a(&mut x, cpq);
        a(&mut x, cws);
        a(&mut x, post);
        a(&mut x, outs);
        a(&mut x, outs_h);
        match self.k.controller_nvfp4_cl8.filter(|_| self.ctrl_v2) {
            // PDL: the cl8 kernel prefetches its weights before griddepcontrol.wait.
            Some(k) => l.go(k, (8 * l.bsz as u32, n), 288, true, x),
            None => l.go(self.k.controller_nvfp4, (l.bsz as u32, n), 1024, false, x),
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    fn linear(&self, l: &Launcher, w: u64, x: u64, y: u64, n: usize, kdim: usize) -> Result<(), String> {
        let mut args = Args::new();
        a(&mut args, w);
        a(&mut args, x);
        a(&mut args, y);
        a(&mut args, n as i32);
        a(&mut args, kdim as i32);
        let kern = if l.bsz == 1 {
            self.k.linear
        } else {
            self.k.linear_b.ok_or("cubin lacks linear_bf16_nosmem_t_b{2,4,8}")?[bi(l.bsz)]
        };
        l.go(kern, (n.div_ceil(4) as u32, 1), 128, false, args)
    }

    /// Records one token step for the token in `d_tokens[0]` at position `*d_pos`, ending with the argmax
    /// in `d_next_tokens[0]`, `*d_pos += 1` and `d_tokens[0] = d_next_tokens[0]`.
    pub fn record_step(&self, stream: &Stream) -> Result<(), String> {
        self.record_batch(stream, 1)
    }

    /// Records one step over `bsz` tokens `d_tokens[0..bsz)` at positions `*d_pos + b` (the verify step
    /// or a prefill chunk). B = 1 also advances `d_pos` and feeds the argmax back into `d_tokens[0]`;
    /// B > 1 leaves `d_pos` alone and writes the argmax of every position to `d_next_tokens[0..bsz)`.
    /// B > 1 needs nvfp4 mode with every part on (the B > 1 kernels exist only for that path).
    pub fn record_batch(&self, stream: &Stream, bsz: i32) -> Result<(), String> {
        if bsz != 1 {
            if ![2, 4, 8].contains(&bsz) {
                return Err(format!("batch {bsz} not supported (1, 2, 4 or 8)"));
            }
            if ![Nvfp4Parts::ALL, Nvfp4Parts::BATCH_EXACT_DENSE, Nvfp4Parts::BATCH_FP8_DENSE].contains(&self.parts) {
                return Err("batches above 1 need nvfp4 mode with all parts in NVFP4, all but dense, or FP8 dense".into());
            }
        }
        let l = Launcher { stream, pdl: self.pdl, bsz };
        let (b, k, w) = (&self.b, &self.k, &self.w);
        let null3 = PtrPack([NULL; 3]);
        let pt = self.parts;
        let qw = self.q;
        let f8 = if pt.dense_fp8 {
            let f = self.fp8.as_ref().ok_or("dense_fp8: FP8 weights not packed (set INKLING_NVFP4_PARTS before building the forward)")?;
            let kq = self.k.qkvrg_fp8.ok_or("cubin lacks the FP8 dense kernels")?;
            let ko = self.k.linear_fp8_k3072.ok_or("cubin lacks the FP8 dense kernels")?;
            Some((f, kq[bi4(bsz)], ko[bi4(bsz)]))
        } else {
            None
        };
        let taps = self.dspark.map(|d| d.taps());
        let tap = |i: usize| taps.map_or(NULL, |t| t + (i * l.bsz as usize * H * 4) as u64);

        let mut x = Args::new();
        a(&mut x, w.embed);
        a(&mut x, w.embed_norm);
        a(&mut x, b.d_tokens);
        a(&mut x, b.hidden);
        l.go(k.embed_norm, (l.bsz as u32, 1), 256, false, x)?;

        let mut block_count: i32 = 0;
        let mut prev_mlp_pending = false;
        for v in SCHEDULE.iter() {
            let vi = v.index;
            let lw = &w.layers[v.layer];
            let lq_all = qw.map(|q| &q.layers[v.layer]);
            let on = |b: bool| lq_all.filter(|_| b);
            let (lq_dense, lq_ctrl, lq_router, lq_exp) = (on(pt.dense), on(pt.ctrl), on(pt.router), on(pt.experts));
            let vs = &self.st.visits[vi];
            let use_prior = v.controllers;
            let boundary = v.layer % dims::BLOCK_LAYERS == 0;
            let k_cache = ptr(self.st.get(vs.k_cache));
            let v_cache = ptr(self.st.get(vs.v_cache));
            let kv_fp8 = vs.kv_fp8.map(|s| ptr(self.st.get(s))).unwrap_or(NULL);

            if v.loop_restart {
                let mut x = Args::new();
                a(&mut x, b.blocks);
                a(&mut x, 4i32);
                a(&mut x, 2i32);
                a(&mut x, b.saved);
                a(&mut x, l.bsz);
                l.go(k.extract_block, (((l.bsz as usize * H).div_ceil(256)) as u32, 1), 256, false, x)?;
                let mut x = Args::new();
                a(&mut x, b.hidden);
                a(&mut x, NULL);
                a(&mut x, b.prior_last);
                a(&mut x, H as i32);
                a(&mut x, H as i32);
                a(&mut x, dims::RMS_EPS);
                l.go(k.rmsnorm, (l.bsz as u32, 1), 256, false, x)?;
                // All B tokens: hidden = prelude (squid5090 copies B * 2304 floats).
                let bytes = l.bsz as usize * H * 4;
                self.prelude_slice.sub(0, bytes).copy_to(&self.hidden_slice.sub(0, bytes), stream).map_err(|e| e.to_string())?;
                block_count = 1;
            }

            // A. Depth norm.
            let partial_dst = if boundary { b.partial } else { NULL };
            let n_out: u32 = if lw.global { 3 } else { 1 };
            let (norms, outs) = match (use_prior, lw.global) {
                (false, true) => (PtrPack(lw.in_norms), PtrPack(b.attn_in)),
                (false, false) => (PtrPack([lw.in_norms[0], NULL, NULL]), PtrPack([b.attn_in[0], NULL, NULL])),
                (true, true) => (null3, PtrPack(b.attn_in)),
                (true, false) => (null3, PtrPack([b.mlp_in, NULL, NULL])),
            };
            let outs_h = match (pt.dense && !use_prior, lw.global) {
                (true, true) => PtrPack(b.attn_in_h),
                (true, false) => PtrPack([b.attn_in_h[0], NULL, NULL]),
                (false, _) => null3,
            };
            let tap_slot = TAP_VISITS.iter().position(|&t| t == vi);
            if taps.is_some() && tap_slot.is_some() && !prev_mlp_pending {
                return Err(format!("DSpark tap visit {vi} has no pending MLP post to write it"));
            }
            // Loop visits with NVFP4 cl8 controllers: the depth norm runs inside the attention controller.
            let fused_kernels = match (self.k.post_next_ctrl, self.k.attn_post_ctrl, self.k.controller_nvfp4_cl8) {
                (Some(a1), Some(a2), Some(_)) if self.ctrl_v2 && self.fuse_ctrl && use_prior && lq_ctrl.is_some() => Some((a1, a2)),
                _ => None,
            };
            // The fused attention-controller grid is 24B CTAs of 153 registers x 288 threads (one per SM):
            // at B = 8 that is 192 CTAs, more than the 170 SMs, and measured 1.1-1.3x slower than the
            // pair it replaces, so B = 8 keeps the two launches.
            let fuse_a = prev_mlp_pending && fused_kernels.is_some() && l.bsz <= 4;
            if fuse_a {
                let prev = &w.layers[SCHEDULE[vi - 1].layer];
                let ctrl = lq_ctrl.and_then(|lq| lq.ctrl).expect("nvfp4 controllers on a loop visit");
                let loc = v.layer - dims::LOOP_START;
                let cpq = PtrPack([
                    w.cross_pass_q + (2 * loc * H * 4) as u64,
                    w.cross_pass_kv_q + (2 * loc * H * 4) as u64,
                    w.cross_pass_kv_q + ((2 * loc + 1) * H * 4) as u64,
                ]);
                let cw = CtrlNvfp4([0, 1, 2].map(|m| [ctrl[0][m], ctrl[1][m], ctrl[2][m]]));
                let post = if lw.global { PtrPack(lw.in_norms) } else { PtrPack([lw.in_norms[0]; 3]) };
                let mut x = Args::new();
                a(&mut x, b.moe_down);
                a(&mut x, prev.mlp_sconv);
                a(&mut x, ptr(self.st.get(self.st.visits[vi - 1].mlp_ring)));
                a(&mut x, b.residual);
                a(&mut x, b.hidden);
                a(&mut x, if vi == dims::LOOP_START { b.prelude } else { tap_slot.map_or(NULL, tap) });
                a(&mut x, partial_dst);
                a(&mut x, b.blocks);
                a(&mut x, block_count);
                a(&mut x, lw.attn_query);
                a(&mut x, if lw.global { 3i32 } else { 1i32 });
                a(&mut x, if lw.global { NULL } else { b.mlp_in });
                a(&mut x, b.saved);
                a(&mut x, b.prior_last);
                a(&mut x, cpq);
                a(&mut x, cw);
                a(&mut x, post);
                a(&mut x, PtrPack(b.attn_in));
                a(&mut x, PtrPack(b.attn_in_h));
                a(&mut x, b.d_pos);
                l.go(fused_kernels.unwrap().0, (8 * l.bsz as u32, 3), 288, true, x)?;
            } else if prev_mlp_pending {
                let prev = &w.layers[SCHEDULE[vi - 1].layer];
                let mut x = Args::new();
                a(&mut x, b.moe_down);
                a(&mut x, false);
                a(&mut x, prev.mlp_sconv);
                a(&mut x, ptr(self.st.get(self.st.visits[vi - 1].mlp_ring)));
                a(&mut x, b.residual);
                a(&mut x, b.hidden);
                a(&mut x, if vi == dims::LOOP_START { b.prelude } else { tap_slot.map_or(NULL, tap) });
                a(&mut x, NULL);
                a(&mut x, partial_dst);
                a(&mut x, b.blocks);
                a(&mut x, block_count);
                a(&mut x, lw.attn_query);
                a(&mut x, norms);
                a(&mut x, outs);
                a(&mut x, outs_h);
                a(&mut x, b.d_pos);
                l.go(k.mlp_post_next, (l.bsz as u32, n_out), 576, true, x)?;
            } else {
                let mut x = Args::new();
                a(&mut x, b.hidden);
                a(&mut x, partial_dst);
                a(&mut x, b.blocks);
                a(&mut x, block_count);
                a(&mut x, lw.attn_query);
                a(&mut x, norms);
                a(&mut x, outs);
                a(&mut x, outs_h);
                l.go(k.fused_depth_norm, (l.bsz as u32, n_out), 256, false, x)?;
            }

            // B. Attention controllers.
            if use_prior && !fuse_a {
                let ctrl = lw.ctrl.expect("controller weights on a loop visit");
                let loc = v.layer - dims::LOOP_START;
                let xs = if lw.global { PtrPack(b.attn_in) } else { PtrPack([b.mlp_in; 3]) };
                let cpq = PtrPack([
                    w.cross_pass_q + (2 * loc * H * 4) as u64,
                    w.cross_pass_kv_q + (2 * loc * H * 4) as u64,
                    w.cross_pass_kv_q + ((2 * loc + 1) * H * 4) as u64,
                ]);
                let cws = PtrPack([
                    ctrl[0][0], ctrl[1][0], ctrl[2][0], ctrl[0][1], ctrl[1][1], ctrl[2][1], ctrl[0][2], ctrl[1][2],
                    ctrl[2][2],
                ]);
                let post = if lw.global { PtrPack(lw.in_norms) } else { PtrPack([lw.in_norms[0]; 3]) };
                if let Some(lq) = lq_ctrl {
                    let c = lq.ctrl.expect("nvfp4 controllers on a loop visit");
                    let cw = CtrlNvfp4([0, 1, 2].map(|m| [c[0][m], c[1][m], c[2][m]]));
                    self.controller_nvfp4(&l, 3, xs, cpq, cw, post, PtrPack(b.attn_in), PtrPack(b.attn_in_h))?;
                } else {
                    self.controllers(&l, 3, xs, PtrPack(b.attn_in), null3, cpq, cws, post)?;
                }
            }

            // C. Projections.
            let distinct = lw.global || use_prior;
            let in_k = if distinct { b.attn_in[1] } else { b.attn_in[0] };
            let in_v = if distinct { b.attn_in[2] } else { b.attn_in[0] };
            if let (Some(_), Some((f, kq, _))) = (lq_dense, f8) {
                let fl = &f.layers[v.layer];
                let (wk, wv) = (fl.kv.rows(0, dims::KV_DIM), fl.kv.rows(dims::KV_DIM, dims::KV_DIM));
                let (in_kh, in_vh) =
                    if distinct { (b.attn_in_h[1], b.attn_in_h[2]) } else { (b.attn_in_h[0], b.attn_in_h[0]) };
                let mut x = Args::new();
                for m in [fl.qrgs, wk, wv] {
                    a(&mut x, m.data);
                    a(&mut x, m.scales);
                }
                a(&mut x, b.attn_in_h[0]);
                a(&mut x, in_kh);
                a(&mut x, in_vh);
                a(&mut x, b.qrgs);
                a(&mut x, b.k_proj);
                a(&mut x, b.v_proj);
                a(&mut x, 0u32);
                l.go(kq, (538, 1), 192, true, x)?;
            } else if let Some(lq) = lq_dense {
                let (wq, wk, wv) = (lq.qrgs, lq.kv.rows(0, dims::KV_DIM), lq.kv.rows(dims::KV_DIM, dims::KV_DIM));
                let (in_kh, in_vh) =
                    if distinct { (b.attn_in_h[1], b.attn_in_h[2]) } else { (b.attn_in_h[0], b.attn_in_h[0]) };
                let mut x = Args::new();
                for p in [wq, wk, wv] {
                    a(&mut x, p.data);
                    a(&mut x, p.scales);
                    a(&mut x, p.gs);
                }
                a(&mut x, b.attn_in_h[0]);
                a(&mut x, in_kh);
                a(&mut x, in_vh);
                a(&mut x, b.qrgs);
                a(&mut x, b.k_proj);
                a(&mut x, b.v_proj);
                a(&mut x, 0u32);
                let rows = dims::QRGS_ROWS + 2 * dims::KV_DIM;
                if l.bsz == 1 {
                    l.go(k.qkvrg_nvfp4, (rows.div_ceil(8) as u32, 1), 128, true, x)?;
                } else {
                    l.go(k.qkvrg_mma[bi(l.bsz)], (538, 1), 192, true, x)?;
                }
            } else {
                self.linear(&l, lw.qrgs, b.attn_in[0], b.qrgs, dims::QRGS_ROWS, H)?;
                self.linear(&l, lw.w_k, in_k, b.k_proj, dims::KV_DIM, H)?;
                self.linear(&l, lw.w_v, in_v, b.v_proj, dims::KV_DIM, H)?;
            }

            // D. Short conv, clip, KV store.
            let mut x = Args::new();
            a(&mut x, b.qrgs);
            a(&mut x, b.k_proj);
            a(&mut x, b.v_proj);
            if lw.global {
                a(&mut x, b.attn_in[1]);
                a(&mut x, b.attn_in[2]);
            }
            a(&mut x, lw.k_sconv);
            a(&mut x, lw.v_sconv);
            if lw.global {
                a(&mut x, lw.k_sconv_proj);
                a(&mut x, lw.v_sconv_proj);
            }
            a(&mut x, ptr(self.st.get(vs.k_ring)));
            a(&mut x, ptr(self.st.get(vs.v_ring)));
            a(&mut x, lw.q_clip);
            a(&mut x, lw.rel_clip);
            a(&mut x, k_cache);
            a(&mut x, v_cache);
            if lw.global {
                a(&mut x, kv_fp8);
                a(&mut x, NULL);
            }
            a(&mut x, vs.kv_cap as i32);
            a(&mut x, b.d_pos);
            l.go(if lw.global { k.global_prep } else { k.local_prep }, (64, l.bsz as u32), 128, true, x)?;

            // E. Attention.
            let window: i32 = if lw.global { 0 } else { dims::WINDOW as i32 };
            if lw.global {
                let mut x = Args::new();
                a(&mut x, b.qrgs);
                let threads = if l.bsz == 8 { 128 } else { 64 };
                if kv_fp8 != NULL {
                    // FP8 shadow: K codes, V codes, K scales, V scales (state::kv_fp8_bytes).
                    let cap = vs.kv_cap as u64;
                    let kd = dims::KV_DIM as u64;
                    a(&mut x, kv_fp8);
                    a(&mut x, kv_fp8 + cap * kd);
                    a(&mut x, kv_fp8 + 2 * cap * kd);
                    a(&mut x, kv_fp8 + 2 * cap * kd + cap * dims::KV_HEADS as u64 * 2);
                } else {
                    a(&mut x, k_cache);
                    a(&mut x, v_cache);
                    a(&mut x, NULL);
                    a(&mut x, NULL);
                }
                a(&mut x, vs.kv_cap as i32);
                a(&mut x, b.d_pos);
                a(&mut x, b.split_buf);
                let kern = match (kv_fp8 != NULL, l.bsz) {
                    (false, 1) => k.bulk_attn,
                    (false, bsz) => k.bulk_attn_b[bi(bsz)],
                    (true, bsz) => k.bulk_fp8[bsz.trailing_zeros() as usize],
                };
                l.go(kern, (8, 64), threads, true, x)?;
            }
            let mut x = Args::new();
            a(&mut x, b.qrgs);
            a(&mut x, lw.rel_proj);
            a(&mut x, k_cache);
            a(&mut x, v_cache);
            a(&mut x, NULL);
            a(&mut x, NULL);
            a(&mut x, vs.ring_cap as i32);
            a(&mut x, window);
            a(&mut x, b.d_pos);
            a(&mut x, b.split_buf);
            if l.bsz == 1 {
                l.go(k.tail_attn, (8, 17), 128, true, x)?;
            } else {
                l.go(k.tail_attn_b[bi(l.bsz)], (8, 17), if l.bsz == 8 { 256 } else { 128 }, true, x)?;
            }
            let mut x = Args::new();
            a(&mut x, b.qrgs);
            a(&mut x, b.split_buf);
            a(&mut x, if pt.dense { NULL } else { b.attn_gated });
            a(&mut x, if pt.dense { b.attn_gated_h } else { NULL });
            a(&mut x, window);
            a(&mut x, b.d_pos);
            l.go(k.merge, (dims::HEADS as u32, l.bsz as u32), 128, true, x)?;

            // F. Output projection.
            if let (Some(_), Some((f, _, ko))) = (lq_dense, f8) {
                let wo = f.layers[v.layer].wo;
                let mut x = Args::new();
                a(&mut x, wo.data);
                a(&mut x, wo.scales);
                a(&mut x, b.attn_gated_h);
                a(&mut x, b.o_proj);
                a(&mut x, H as i32);
                a(&mut x, 0u32);
                l.go(ko, (H.div_ceil(16) as u32, 1), 256, true, x)?;
            } else if let Some(lq) = lq_dense {
                let mut x = Args::new();
                a(&mut x, lq.wo.data);
                a(&mut x, lq.wo.scales);
                a(&mut x, lq.wo.gs);
                a(&mut x, b.attn_gated_h);
                a(&mut x, b.o_proj);
                a(&mut x, H as i32);
                if l.bsz == 1 {
                    a(&mut x, dims::Q_DIM as i32);
                    a(&mut x, 0u32);
                    l.go(k.linear_nvfp4_k3072, (H.div_ceil(8) as u32, 1), 128, true, x)?;
                } else {
                    // The MMA variants take no K parameter.
                    a(&mut x, 0u32);
                    l.go(k.linear_mma_k3072[bi(l.bsz)], (H.div_ceil(16) as u32, 1), 256, true, x)?;
                }
            } else {
                self.linear(&l, lw.w_o, b.attn_gated, b.o_proj, H, dims::Q_DIM)?;
            }

            // G. Attention post-process and MLP input (on loop visits, fused with the MLP controller).
            if let Some((_, k_attn_post_ctrl)) = fused_kernels {
                let ctrl = lq_ctrl.and_then(|lq| lq.ctrl).expect("nvfp4 controllers on a loop visit")[3];
                let loc = v.layer - dims::LOOP_START;
                let mut x = Args::new();
                a(&mut x, b.o_proj);
                a(&mut x, lw.attn_sconv);
                a(&mut x, ptr(self.st.get(vs.attn_ring)));
                a(&mut x, b.hidden);
                a(&mut x, b.partial);
                a(&mut x, b.blocks);
                a(&mut x, block_count);
                a(&mut x, boundary);
                a(&mut x, b.residual);
                a(&mut x, lw.mlp_query);
                a(&mut x, b.mlp_in);
                a(&mut x, b.saved);
                a(&mut x, b.prior_last);
                a(&mut x, PtrPack([w.cross_pass_q + ((2 * loc + 1) * H * 4) as u64; 3]));
                a(&mut x, CtrlNvfp4([0, 1, 2].map(|i| [ctrl[i]; 3])));
                a(&mut x, PtrPack([lw.mlp_norm; 3]));
                a(&mut x, PtrPack([b.mlp_normed; 3]));
                a(&mut x, PtrPack([b.mlp_normed_h, NULL, NULL]));
                a(&mut x, b.d_pos);
                l.go(k_attn_post_ctrl, (8 * l.bsz as u32, 1), 288, true, x)?;
            } else {
                let mut x = Args::new();
                a(&mut x, b.o_proj);
                a(&mut x, lw.attn_sconv);
                a(&mut x, ptr(self.st.get(vs.attn_ring)));
                a(&mut x, b.hidden);
                a(&mut x, b.partial);
                a(&mut x, b.blocks);
                a(&mut x, block_count);
                a(&mut x, boundary);
                a(&mut x, b.residual);
                a(&mut x, lw.mlp_query);
                a(&mut x, if use_prior { NULL } else { lw.mlp_norm });
                a(&mut x, if use_prior { b.mlp_in } else { b.mlp_normed });
                a(&mut x, if use_prior { NULL } else { b.mlp_normed_h });
                a(&mut x, b.d_pos);
                l.go(k.attn_post_mlp_in, (l.bsz as u32, 1), 576, true, x)?;
            }
            if boundary {
                block_count += 1;
            }

            // H. MLP controller.
            if use_prior && fused_kernels.is_none() {
                let ctrl = lw.ctrl.expect("controller weights on a loop visit");
                let loc = v.layer - dims::LOOP_START;
                let m = ctrl[3];
                let cpq = PtrPack([w.cross_pass_q + ((2 * loc + 1) * H * 4) as u64; 3]);
                if let Some(lq) = lq_ctrl {
                    let c = lq.ctrl.expect("nvfp4 controllers on a loop visit")[3];
                    let cw = CtrlNvfp4([0, 1, 2].map(|i| [c[i]; 3]));
                    self.controller_nvfp4(
                        &l,
                        1,
                        PtrPack([b.mlp_in; 3]),
                        cpq,
                        cw,
                        PtrPack([lw.mlp_norm; 3]),
                        PtrPack([b.mlp_normed; 3]),
                        PtrPack([b.mlp_normed_h, NULL, NULL]),
                    )?;
                } else {
                    self.controllers(
                        &l,
                        1,
                        PtrPack([b.mlp_in; 3]),
                        PtrPack([b.mlp_normed; 3]),
                        PtrPack([b.mlp_normed_h, NULL, NULL]),
                        cpq,
                        PtrPack([m[0], m[0], m[0], m[1], m[1], m[1], m[2], m[2], m[2]]),
                        PtrPack([lw.mlp_norm; 3]),
                    )?;
                }
            }

            // I. Router and experts.
            if let Some(lq) = lq_router {
                let mut x = Args::new();
                a(&mut x, b.mlp_normed_h);
                a(&mut x, lq.router.data);
                a(&mut x, lq.router.scales);
                a(&mut x, lq.router.gs);
                a(&mut x, b.router_logits);
                a(&mut x, NULL);
                a(&mut x, 0u32);
                let kern = if l.bsz == 1 { k.route_logits_nvfp4 } else { k.route_logits_nvfp4_b[bi(l.bsz)] };
                l.go(kern, (64, 1), 32, true, x)?;
            } else {
                let mut x = Args::new();
                a(&mut x, b.mlp_normed);
                a(&mut x, lw.router_w);
                a(&mut x, b.router_logits);
                a(&mut x, NULL);
                l.go(k.route_logits, (64, 1), 64, false, x)?;
            }
            let mut x = Args::new();
            a(&mut x, b.router_logits);
            a(&mut x, lw.router_b);
            a(&mut x, b.route_ids);
            a(&mut x, b.route_w);
            a(&mut x, NULL);
            a(&mut x, 0u32);
            // The redux select (bit-identical to `route_select_and_group_b*`).
            let kern = k.route_select_fast[l.bsz.trailing_zeros() as usize];
            l.go(kern, (1, 1), 32 * l.bsz as u32, true, x)?;
            let (fmt, w13, w13s, w13f, w2, w2s, w2f) = match lq_exp {
                Some(lq) => {
                    let e = lq.experts;
                    (0, e.w13.ptr, e.w13_scales.ptr, e.w13_factor.ptr, e.w2.ptr, e.w2_scales.ptr, e.w2_factor.ptr)
                }
                None => (lw.fmt, lw.w13, lw.w13_scales, lw.w13_factor, lw.w2, lw.w2_scales, lw.w2_factor),
            };
            let mut x = Args::new();
            a(&mut x, b.mlp_normed_h);
            a(&mut x, b.route_ids);
            a(&mut x, w13);
            a(&mut x, w13s);
            a(&mut x, w13f);
            a(&mut x, l.bsz);
            a(&mut x, b.moe_act_h);
            l.go(k.moe_up[fmt], ((2 * dims::INTERMEDIATE / 16) as u32, (l.bsz as usize * dims::TOP_K) as u32), 576, true, x)?;
            let mut x = Args::new();
            a(&mut x, b.moe_act_h);
            a(&mut x, b.route_ids);
            a(&mut x, b.route_w);
            a(&mut x, w2);
            a(&mut x, w2s);
            a(&mut x, w2f);
            a(&mut x, l.bsz);
            a(&mut x, b.moe_down);
            l.go(k.moe_down[fmt], ((H / 16) as u32, 1), 576, true, x)?;

            // J. Flush before a loop restart.
            if v.flush_mlp {
                let mut x = Args::new();
                a(&mut x, b.moe_down);
                a(&mut x, false);
                a(&mut x, lw.mlp_sconv);
                a(&mut x, ptr(self.st.get(vs.mlp_ring)));
                a(&mut x, b.residual);
                a(&mut x, b.hidden);
                a(&mut x, b.d_pos);
                l.go(k.mlp_post_res, (l.bsz as u32, 1), 576, true, x)?;
                prev_mlp_pending = false;
            } else {
                prev_mlp_pending = true;
            }
        }

        // Final: pending MLP post + output depth-attention + final norm, LM head, argmax, advance.
        let last = &w.layers[dims::LAYERS - 1];
        let mut x = Args::new();
        a(&mut x, b.moe_down);
        a(&mut x, false);
        a(&mut x, last.mlp_sconv);
        a(&mut x, ptr(self.st.get(self.st.visits[dims::VISITS - 1].mlp_ring)));
        a(&mut x, b.residual);
        a(&mut x, b.hidden);
        a(&mut x, NULL);
        a(&mut x, tap(3));
        a(&mut x, NULL);
        a(&mut x, b.blocks);
        a(&mut x, 4i32);
        a(&mut x, w.output_query);
        a(&mut x, PtrPack([w.final_norm, NULL, NULL]));
        a(&mut x, PtrPack([b.mlp_normed, NULL, NULL]));
        a(&mut x, null3);
        a(&mut x, b.d_pos);
        l.go(k.mlp_post_next, (l.bsz as u32, 1), 576, true, x)?;

        if let Some(d) = self.dspark {
            // The commit reads the taps and d_pos only; it overlaps the LM head on the aux stream.
            d.start_commit(stream, l.bsz)?;
        }
        let cfg = if self.write_logits { NULL } else { b.sample_cfg };
        let mut x = Args::new();
        if l.bsz > 1 {
            let q = qw.expect("nvfp4 weights for a batched step");
            a(&mut x, q.unembed.data.ptr);
            a(&mut x, q.unembed.scales.ptr);
            a(&mut x, q.unembed.gs);
            a(&mut x, b.mlp_normed);
            a(&mut x, b.lm_logits);
            a(&mut x, b.lm_blk_val);
            a(&mut x, b.lm_blk_idx);
            a(&mut x, cfg);
            a(&mut x, dims::VOCAB as i32);
            a(&mut x, 0u32);
            l.go(k.lm_head_mma[bi(l.bsz)], (LM_HEAD_BLOCKS as u32, 1), 192, true, x)?;
            x = Args::new();
        }
        let (head_w, head_s, head_gs) = match qw.filter(|_| pt.head) {
            Some(q) => (q.unembed.data.ptr, q.unembed.scales.ptr, q.unembed.gs),
            None => (w.unembed, NULL, 1.0f32),
        };
        a(&mut x, head_w);
        a(&mut x, head_s);
        a(&mut x, NULL);
        a(&mut x, head_gs);
        a(&mut x, b.mlp_normed);
        a(&mut x, b.lm_logits);
        a(&mut x, b.lm_blk_val);
        a(&mut x, b.lm_blk_idx);
        a(&mut x, cfg);
        a(&mut x, dims::VOCAB as i32);
        if l.bsz > 1 {
            // Stage 1 already ran as the MMA variant above.
        } else if pt.head {
            l.go(k.lm_head1_nvfp4, (LM_HEAD_BLOCKS as u32, 1), 256, true, x)?;
        } else {
            l.go(k.lm_head1, (LM_HEAD_BLOCKS as u32, 1), 256, false, x)?;
        }
        let mut x = Args::new();
        a(&mut x, b.lm_blk_val);
        a(&mut x, b.lm_blk_idx);
        a(&mut x, b.lm_logits);
        a(&mut x, b.d_next_tokens);
        a(&mut x, cfg);
        a(&mut x, b.d_pos);
        a(&mut x, dims::VOCAB as i32);
        a(&mut x, l.bsz);
        l.go(k.lm_head2, (l.bsz as u32, 1), 256, true, x)?;
        if let Some(d) = self.dspark {
            // Joined before inc_pos: the commit stores positions *d_pos + b.
            d.finish_commit(stream)?;
        }
        if l.bsz > 1 {
            return Ok(());
        }
        let mut x = Args::new();
        a(&mut x, b.d_pos);
        a(&mut x, b.d_tokens);
        a(&mut x, b.d_next_tokens);
        l.go(k.inc_pos, (1, 1), 1, false, x)
    }

    /// Logits written by the last step (softcapped only when sampling is configured; greedy leaves
    /// them unwritten, as in squid5090).
    pub fn logits_slice(&self) -> DevSlice {
        self.st.get(self.st.s.lm_logits).sub(0, dims::VOCAB * 4)
    }

    pub fn max_batch(&self) -> usize {
        B_MAX
    }
}
