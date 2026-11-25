//! DSpark drafter, speculative commit/verify and the load-time fold kernels
//! (nvfp4 mode, BF16 draft KV).
//!
//! Ports of squid5090 @ 3ee093c `cuda/kernels_dspark.cu`:
//! - `dspark_mix_kernel` (10) as `dspark_mix`;
//! - `dspark_store_ctx_kv_kernel<0>` (66) as `dspark_store_ctx_kv_bf16`;
//! - `dspark_init_block_kernel` (118), `dspark_block_qk_rope_kernel` (159),
//!   `dspark_rmsnorm_h_kernel` (199) under the same names without `_kernel`;
//! - `dspark_global_attn_bulk_splitk_kernel<0>` (233) as `dspark_global_attn_bulk_splitk_bf16kv`;
//! - `dspark_block_attn_kernel` (261), `dspark_swiglu_kernel` (382), `markov_rnn_proj_kernel` (398);
//! - `markov_fold_argmax_stage1_kernel<2>` (641) as `markov_fold_argmax_stage1_nvfp4`;
//! - `markov_fold_argmax_stage2_kernel` (743), `copy_f_prev_zero_state_kernel` (799),
//!   `spec_verify_kernel` (810);
//!
//! and `cuda/cuda_abi.cu`: `z_compute_dspark_fold_bias_kernel` (169),
//! `z_compute_dspark_fold_w_kernel` (182), `z_split_dspark_rnn_kernel` (197)
//! and `z_assemble_verify_tokens_kernel` (219). `z_adaptive_select_width_kernel` (226)
//! stays CUDA C++ (`cuda/z_adaptive_select_width.cu`): it calls the device
//! runtime's `cudaGraphSetConditional`, which cuda-oxide cannot declare (its
//! LLVM verifier rejects the unknown external symbol, and inline PTX cannot
//! declare a module-scope `.extern .func`).
//!
//! Every place nvcc contracts a product into an add is written as an explicit
//! `fmaf` in nvcc's operand order (read from its PTX); every product nvcc keeps
//! separate from a following add, subtract or shuffle reduction is `mul_rn`.

use crate::attn::{BULK_SMEM, bulk_tile_load, run_tc_bulk_splitk, run_tc_bulk_splitk_pre};
use crate::common::*;
use crate::nvfp4::block16_dot;
use crate::{FINAL_SOFTCAP, HIDDEN, cvt_e4m3_to_float, fast_sigmoid, fast_tanh, fmaf, float2half, half_warp_sum, ldg_u2, ldg_u8, pdl_trigger, pdl_wait};
use cuda_device::thread::{blockDim_x, blockDim_y, blockIdx_x, blockIdx_y, gridDim_x, sync_threads, threadIdx_x, threadIdx_y};
use cuda_device::{SharedArray, device, kernel, launch_bounds, warp};

const QKV_STRIDE: usize = Q_DIM + 2 * KV_DIM; // 5120
const MTP_INTERMEDIATE: usize = 9216;
const MTP_MAX_DRAFTS: usize = 7;
const MTP_MARKOV_RANK: usize = 256;
const MARKOV_BLOCKS: usize = 340;
const INV_HEAD_DIM: f32 = 1.0 / HEAD_DIM as f32; // exact: nvcc turns `/ 128.0f` into `* 2^-7`

unsafe extern "C" {
    fn __nv_powf(x: f32, y: f32) -> f32;
    fn __nv_sinf(x: f32) -> f32;
    fn __nv_cosf(x: f32) -> f32;
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct F4([f32; 4]);

#[inline(always)]
unsafe fn ld_f4(p: *const f32) -> [f32; 4] {
    unsafe { (*(p as *const F4)).0 }
}

/// Block-wide sum of 8 warp partials held in `sm[0..8]`, in nvcc's order:
/// `tot = 0; for i: tot += sm[i]`.
#[inline(always)]
unsafe fn sum8(sm: *const f32) -> f32 {
    let mut tot = 0.0f32;
    let mut i = 0;
    while i < 8 {
        tot += unsafe { *sm.add(i) };
        i += 1;
    }
    tot
}

/// NeoX RoPE (theta 1e6) of one pair at position `pos`, frequency index `i`:
/// returns `(x0 c - x1 s, x0 s + x1 c)` as nvcc's SASS evaluates it. The PTX
/// keeps `x0 * c - x1 * s` as two `mul.f32` and a `sub.f32`, but ptxas fuses
/// it into `FFMA(x0, c, -(x1 * s))`, so the port writes that FMA explicitly.
#[inline(always)]
fn rope_pair(x0: f32, x1: f32, pos: i32, i: usize) -> (f32, f32) {
    let e = mul_rn((2 * i) as f32, INV_HEAD_DIM);
    let freq = rcp_rn(unsafe { __nv_powf(1_000_000.0, e) });
    let ang = mul_rn(pos as f32, freq);
    let c = unsafe { __nv_cosf(ang) };
    let s = unsafe { __nv_sinf(ang) };
    (fmaf(x0, c, -mul_rn(x1, s)), fmaf(x1, c, mul_rn(x0, s)))
}

/// Per-head (128-thread) RMS: `rsqrt(sum(v^2) / 128 + eps)` with nvcc's
/// `fma(sum, 2^-7, eps)`.
#[inline(always)]
unsafe fn head_inv_rms(val: f32, sm: *mut f32, inv: *mut f32) -> f32 {
    unsafe {
        let tid = threadIdx_x() as usize;
        let sq = warp_sum(mul_rn(val, val));
        if (tid & 31) == 0 {
            *sm.add(tid >> 5) = sq;
        }
        sync_threads();
        if tid == 0 {
            let t = *sm + *sm.add(1) + *sm.add(2) + *sm.add(3);
            *inv = rsqrtf(fmaf(t, INV_HEAD_DIM, RMS_EPS));
        }
        sync_threads();
        *inv
    }
}

// ---------------------------------------------------------------------------
// Commit: mix the target taps, project, store the draft context KV
// ---------------------------------------------------------------------------

/// `dspark_mix_kernel`: grid (B, 4), 256 threads. Mixes the 4 target taps
/// with `fuse_alpha[m]`, RMSNorm, then the per-m norm; m < 3 writes
/// `out_mix[m, b]`, m = 3 writes `f_ring[(pos + b) & 15]`.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dspark_mix(
    taps: *const f32,
    b_count: i32,
    fuse_alpha: *const f32,
    n0: *const u16,
    n1: *const u16,
    n2: *const u16,
    n3: *const u16,
    out_mix: *mut f32,
    f_ring: *mut f32,
    d_pos: *const i32,
) {
    static mut BUF: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
    static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let buf = SharedArray::as_raw_mut_ptr(&raw mut BUF);
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv = SharedArray::as_raw_mut_ptr(&raw mut INV);
        let b = blockIdx_x() as usize;
        let m = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let bc = b_count as usize;
        let a0 = *fuse_alpha.add(m * 4);
        let a1 = *fuse_alpha.add(m * 4 + 1);
        let a2 = *fuse_alpha.add(m * 4 + 2);
        let a3 = *fuse_alpha.add(m * 4 + 3);
        let t0 = taps.add(b * HIDDEN);
        let t1 = taps.add((bc + b) * HIDDEN);
        let t2 = taps.add((2 * bc + b) * HIDDEN);
        let t3 = taps.add((3 * bc + b) * HIDDEN);

        let mut sum_sq = 0.0f32;
        let mut d = tid;
        while d < HIDDEN {
            // nvcc: fma(a3, t3, fma(a2, t2, fma(a0, t0, a1 * t1)))
            let v = fmaf(a3, *t3.add(d), fmaf(a2, *t2.add(d), fmaf(a0, *t0.add(d), mul_rn(a1, *t1.add(d)))));
            *buf.add(d) = v;
            sum_sq = fmaf(v, v, sum_sq);
            d += 256;
        }
        sum_sq = warp_sum(sum_sq);
        if (tid & 31) == 0 {
            *sm.add(tid >> 5) = sum_sq;
        }
        sync_threads();
        if tid == 0 {
            *inv = rsqrtf(div_rn(sum8(sm), HIDDEN as f32) + RMS_EPS);
        }
        sync_threads();

        let nw = if m == 0 {
            n0
        } else if m == 1 {
            n1
        } else if m == 2 {
            n2
        } else {
            n3
        };
        let dst = if m < 3 {
            out_mix.add((m * bc + b) * HIDDEN)
        } else {
            f_ring.add((((*d_pos) + b as i32) & 15) as usize * HIDDEN)
        };
        let s = *inv;
        let mut d = tid;
        while d < HIDDEN {
            *dst.add(d) = *buf.add(d) * s * bf16_to_f32(*nw.add(d));
            d += 256;
        }
    }
}

/// `dspark_mix` for a whole wide-prefill chunk (grid (T, 4), taps [4][T][H]): the same arithmetic per token, but
/// only the chunk's last 16 tokens write the 16-slot f ring (what B = 8 commits in order leave there).
/// `dspark_mix_kernel`: grid (B, 4), 256 threads. Mixes the 4 target taps
/// with `fuse_alpha[m]`, RMSNorm, then the per-m norm; m < 3 writes
/// `out_mix[m, b]`, m = 3 writes `f_ring[(pos + b) & 15]`.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_dspark_mix(
    taps: *const f32,
    b_count: i32,
    fuse_alpha: *const f32,
    n0: *const u16,
    n1: *const u16,
    n2: *const u16,
    n3: *const u16,
    out_mix: *mut f32,
    f_ring: *mut f32,
    d_pos: *const i32,
) {
    static mut BUF: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
    static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let buf = SharedArray::as_raw_mut_ptr(&raw mut BUF);
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv = SharedArray::as_raw_mut_ptr(&raw mut INV);
        let b = blockIdx_x() as usize;
        let m = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let bc = b_count as usize;
        if m == 3 && b + 16 < bc {
            return;
        }
        let a0 = *fuse_alpha.add(m * 4);
        let a1 = *fuse_alpha.add(m * 4 + 1);
        let a2 = *fuse_alpha.add(m * 4 + 2);
        let a3 = *fuse_alpha.add(m * 4 + 3);
        let t0 = taps.add(b * HIDDEN);
        let t1 = taps.add((bc + b) * HIDDEN);
        let t2 = taps.add((2 * bc + b) * HIDDEN);
        let t3 = taps.add((3 * bc + b) * HIDDEN);

        let mut sum_sq = 0.0f32;
        let mut d = tid;
        while d < HIDDEN {
            // nvcc: fma(a3, t3, fma(a2, t2, fma(a0, t0, a1 * t1)))
            let v = fmaf(a3, *t3.add(d), fmaf(a2, *t2.add(d), fmaf(a0, *t0.add(d), mul_rn(a1, *t1.add(d)))));
            *buf.add(d) = v;
            sum_sq = fmaf(v, v, sum_sq);
            d += 256;
        }
        sum_sq = warp_sum(sum_sq);
        if (tid & 31) == 0 {
            *sm.add(tid >> 5) = sum_sq;
        }
        sync_threads();
        if tid == 0 {
            *inv = rsqrtf(div_rn(sum8(sm), HIDDEN as f32) + RMS_EPS);
        }
        sync_threads();

        let nw = if m == 0 {
            n0
        } else if m == 1 {
            n1
        } else if m == 2 {
            n2
        } else {
            n3
        };
        let dst = if m < 3 {
            out_mix.add((m * bc + b) * HIDDEN)
        } else {
            f_ring.add((((*d_pos) + b as i32) & 15) as usize * HIDDEN)
        };
        let s = *inv;
        let mut d = tid;
        while d < HIDDEN {
            *dst.add(d) = *buf.add(d) * s * bf16_to_f32(*nw.add(d));
            d += 256;
        }
    }
}

/// `store_kv_head_128<1>`: one KV head (128 threads, value `x` each) as E4M3 codes plus an f16 scale per
/// (slot, head): CTA amax through `sm[4]`, scale = max(amax / 448, 1e-8).
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn store_kv_head_fp8(x: f32, codes: *mut u8, scales: *mut u16, slot: usize, h: usize, tid: usize, sm: *mut f32) {
    unsafe {
        let wm = warp_max(fmaxf(x, -x));
        if (tid & 31) == 0 {
            *sm.add(tid >> 5) = wm;
        }
        sync_threads();
        let amax = fmaxf(fmaxf(*sm, *sm.add(1)), fmaxf(*sm.add(2), *sm.add(3)));
        let sc = fmaxf(div_rn(amax, 448.0), 1e-8);
        if tid == 0 {
            *scales.add(slot * NUM_KV_HEADS + h) = float2half(sc);
        }
        *codes.add(slot * KV_DIM + h * HEAD_DIM + tid) = crate::nvfp4::f32_to_e4m3(div_rn(x, sc));
    }
}

/// `dspark_store_ctx_kv_kernel<KV_FMT>` body (BF16 or FP8 caches).
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn store_ctx_kv_body<const FP8: bool>(
    kv: *const f32,
    k_norm: *const u16,
    k_cache: *mut u16,
    v_cache: *mut u16,
    k_scale: *mut u16,
    v_scale: *mut u16,
    cap: i32,
    d_pos: *const i32,
    b_off: usize,
    sm: *mut f32,
    inv: *mut f32,
    kn: *mut f32,
) {
    unsafe {
        let b = blockIdx_x() as usize + b_off;
        let h = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let pos = *d_pos + b as i32;
        let slot = (pos % cap) as usize;
        let kp = kv.add(b * 2 * KV_DIM + h * HEAD_DIM);
        let vp = kv.add(b * 2 * KV_DIM + KV_DIM + h * HEAD_DIM);
        let o = slot * KV_DIM + h * HEAD_DIM + tid;

        if FP8 {
            store_kv_head_fp8(*vp.add(tid), v_cache as *mut u8, v_scale, slot, h, tid, sm);
        } else {
            *v_cache.add(o) = f32_to_bf16(*vp.add(tid));
        }
        sync_threads();

        let kv_val = *kp.add(tid);
        let ir = head_inv_rms(kv_val, sm, inv);
        *kn.add(tid) = kv_val * ir * bf16_to_f32(*k_norm.add(tid));
        sync_threads();
        if tid < 64 {
            let (r0, r1) = rope_pair(*kn.add(tid), *kn.add(tid + 64), pos, tid);
            *kn.add(tid) = r0;
            *kn.add(tid + 64) = r1;
        }
        sync_threads();
        if FP8 {
            store_kv_head_fp8(*kn.add(tid), k_cache as *mut u8, k_scale, slot, h, tid, sm);
        } else {
            *k_cache.add(o) = f32_to_bf16(*kn.add(tid));
        }
    }
}

macro_rules! store_ctx_kernel {
    ($name:ident, $fp8:expr, $doc:literal, $off:ident) => {
        #[doc = $doc]
        #[kernel]
        #[launch_bounds(128)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            kv: *const f32,
            k_norm: *const u16,
            k_cache: *mut u16,
            v_cache: *mut u16,
            k_scale: *mut u16,
            v_scale: *mut u16,
            cap: i32,
            d_pos: *const i32,
            $off: i32,
        ) {
            store_ctx_kernel!(@body $fp8, kv, k_norm, k_cache, v_cache, k_scale, v_scale, cap, d_pos, $off as usize);
        }
    };
    ($name:ident, $fp8:expr, $doc:literal) => {
        #[doc = $doc]
        #[kernel]
        #[launch_bounds(128)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            kv: *const f32,
            k_norm: *const u16,
            k_cache: *mut u16,
            v_cache: *mut u16,
            k_scale: *mut u16,
            v_scale: *mut u16,
            cap: i32,
            d_pos: *const i32,
        ) {
            store_ctx_kernel!(@body $fp8, kv, k_norm, k_cache, v_cache, k_scale, v_scale, cap, d_pos, 0usize);
        }
    };
    (@body $fp8:expr, $kv:ident, $k_norm:ident, $k_cache:ident, $v_cache:ident, $k_scale:ident, $v_scale:ident, $cap:ident, $d_pos:ident, $b_off:expr) => {
        {
            static mut SM: SharedArray<f32, 4> = SharedArray::UNINIT;
            static mut INV: SharedArray<f32, 1> = SharedArray::UNINIT;
            static mut KN: SharedArray<f32, { HEAD_DIM }, 16> = SharedArray::UNINIT;
            unsafe {
                store_ctx_kv_body::<$fp8>(
                    $kv,
                    $k_norm,
                    $k_cache,
                    $v_cache,
                    $k_scale,
                    $v_scale,
                    $cap,
                    $d_pos,
                    $b_off,
                    SharedArray::as_raw_mut_ptr(&raw mut SM),
                    SharedArray::as_raw_mut_ptr(&raw mut INV),
                    SharedArray::as_raw_mut_ptr(&raw mut KN),
                )
            }
        }
    };
}

store_ctx_kernel!(dspark_store_ctx_kv_bf16, false, "`dspark_store_ctx_kv_kernel<0>`: grid (B, 8), 128 threads. V is stored as is; K gets a per-head RMSNorm and NeoX RoPE, both into BF16 caches. The scale pointers are unused for BF16.");
store_ctx_kernel!(dspark_store_ctx_kv_fp8, true, "`dspark_store_ctx_kv_kernel<1>`: as the BF16 kernel, into E4M3 code caches (u8) with an f16 scale per (slot, KV head).");
store_ctx_kernel!(prefill_dspark_store_ctx_kv_bf16, false, "`dspark_store_ctx_kv_bf16` for the wide prefill: CTA x is token `b_off + x` of the chunk (kv [T][2 KV_DIM]), so a launch can cover only the last `cap` tokens of a chunk. Same arithmetic.", b_off);
store_ctx_kernel!(prefill_dspark_store_ctx_kv_fp8, true, "`dspark_store_ctx_kv_fp8` with a token offset, as `prefill_dspark_store_ctx_kv_bf16`.", b_off);
// ---------------------------------------------------------------------------
// Draft block: 8 rows (anchor + 7 masks) through 3 dense layers
// ---------------------------------------------------------------------------

/// `dspark_init_block_kernel`: grid 8, 256 threads. Row 0 = RMSNorm(embed[anchor])
/// * embed_norm; rows 1..7 = mask_embed.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn dspark_init_block(
    embed: *const u16,
    embed_norm: *const u16,
    mask_embed: *const u16,
    d_anchor: *const i32,
    block_x: *mut f32,
) {
    static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv = SharedArray::as_raw_mut_ptr(&raw mut INV);
        let r = blockIdx_x() as usize;
        let tid = threadIdx_x() as usize;
        let dst = block_x.add(r * HIDDEN);
        if r == 0 {
            let row = embed.add(*d_anchor as usize * HIDDEN);
            let mut sum_sq = 0.0f32;
            let mut d = tid;
            while d < HIDDEN {
                let v = bf16_to_f32(*row.add(d));
                sum_sq = fmaf(v, v, sum_sq);
                d += 256;
            }
            sum_sq = warp_sum(sum_sq);
            if (tid & 31) == 0 {
                *sm.add(tid >> 5) = sum_sq;
            }
            sync_threads();
            if tid == 0 {
                *inv = rsqrtf(div_rn(sum8(sm), HIDDEN as f32) + RMS_EPS);
            }
            sync_threads();
            let s = *inv;
            let mut d = tid;
            while d < HIDDEN {
                *dst.add(d) = bf16_to_f32(*row.add(d)) * s * bf16_to_f32(*embed_norm.add(d));
                d += 256;
            }
        } else {
            let mut d = tid;
            while d < HIDDEN {
                *dst.add(d) = bf16_to_f32(*mask_embed.add(d));
                d += 256;
            }
        }
    }
}

/// `dspark_block_qk_rope_kernel`: grid (8, 32), 128 threads. In place on
/// qkv [8, 5120]: heads 0..23 are Q, 24..31 are K; per-head RMSNorm, then
/// NeoX RoPE at position p + row.
#[kernel]
#[launch_bounds(128)]
pub unsafe fn dspark_block_qk_rope(qkv: *mut f32, q_norm: *const u16, k_norm: *const u16, d_pos: *const i32) {
    static mut SM: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut INV: SharedArray<f32, 1> = SharedArray::UNINIT;
    static mut VN: SharedArray<f32, { HEAD_DIM }, 16> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv = SharedArray::as_raw_mut_ptr(&raw mut INV);
        let vn = SharedArray::as_raw_mut_ptr(&raw mut VN);
        let r = blockIdx_x() as usize;
        let h = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let pos = *d_pos + r as i32;
        let (vec, nw) = if h < NUM_HEADS {
            (qkv.add(r * QKV_STRIDE + h * HEAD_DIM), q_norm)
        } else {
            (qkv.add(r * QKV_STRIDE + Q_DIM + (h - NUM_HEADS) * HEAD_DIM), k_norm)
        };
        let val = *vec.add(tid);
        let ir = head_inv_rms(val, sm, inv);
        *vn.add(tid) = val * ir * bf16_to_f32(*nw.add(tid));
        sync_threads();
        if tid < 64 {
            let (r0, r1) = rope_pair(*vn.add(tid), *vn.add(tid + 64), pos, tid);
            *vec.add(tid) = r0;
            *vec.add(tid + 64) = r1;
        }
    }
}

/// `dspark_rmsnorm_h_kernel`: grid rows, 256 threads. f16 output (and an
/// optional f32 copy) for the NVFP4 linears.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn dspark_rmsnorm_h(x: *const f32, w: *const u16, out_f: *mut f32, out_h: *mut u16) {
    static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv = SharedArray::as_raw_mut_ptr(&raw mut INV);
        let r = blockIdx_x() as usize;
        let tid = threadIdx_x() as usize;
        let xb = x.add(r * HIDDEN);
        let mut sq = 0.0f32;
        let mut d = tid;
        while d < HIDDEN {
            let v = *xb.add(d);
            sq = fmaf(v, v, sq);
            d += 256;
        }
        sq = warp_sum(sq);
        if (tid & 31) == 0 {
            *sm.add(tid >> 5) = sq;
        }
        sync_threads();
        if tid == 0 {
            *inv = rsqrtf(div_rn(sum8(sm), HIDDEN as f32) + RMS_EPS);
        }
        sync_threads();
        let ir = *inv;
        let of = if out_f.is_null() { core::ptr::null_mut() } else { out_f.add(r * HIDDEN) };
        let oh = out_h.add(r * HIDDEN);
        let mut d = tid;
        while d < HIDDEN {
            let v = *xb.add(d) * ir * bf16_to_f32(*w.add(d));
            *oh.add(d) = float2half(v);
            if !of.is_null() {
                *of.add(d) = v;
            }
            d += 256;
        }
    }
}

/// `dspark_global_attn_bulk_splitk_kernel<0>`: grid (8, 64), 128 threads.
/// Tensor-core split-K attention of all 24 draft queries (8 rows x 3 GQA)
/// over the global layer's whole context [first, p).
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dspark_global_attn_bulk_splitk_bf16kv(
    qkv: *const f32,
    ck: *const u16,
    cv: *const u16,
    _cks: *const u16,
    _cvs: *const u16,
    cap: i32,
    d_pos: *const i32,
    split_buf: *mut f32,
) {
    static mut SMEM: SharedArray<u8, { BULK_SMEM }, 16> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let p = *d_pos;
        if p <= 0 {
            return;
        }
        let first = if p > cap { p - cap } else { 0 };
        let total_len = p - first;
        let split = blockIdx_y() as i32;
        let chunk = (total_len + KV_SPLIT_K as i32 - 1) / KV_SPLIT_K as i32;
        let t_start = first + split * chunk;
        let t_end = core::cmp::min(p, t_start + chunk);
        // Q_CNT = 24 = 3 x 8 rows: the B = 8 instance with the DSpark row stride.
        run_tc_bulk_splitk::<8, QKV_STRIDE>(
            SharedArray::as_raw_mut_ptr(&raw mut SMEM),
            qkv,
            ck,
            cv,
            cap,
            blockIdx_x() as usize,
            split as usize,
            t_start,
            t_end,
            split_buf,
        );
    }
}

/// `dspark_global_attn_bulk_splitk_kernel<1>`: the draft global bulk over FP8 context caches (u8 E4M3 codes,
/// f16 scales per slot and KV head). Grid (8, 64), 128 threads.
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dspark_global_attn_bulk_splitk_fp8kv(
    qkv: *const f32,
    ck: *const u16,
    cv: *const u16,
    cks: *const u16,
    cvs: *const u16,
    cap: i32,
    d_pos: *const i32,
    split_buf: *mut f32,
) {
    static mut SMEM: SharedArray<u8, { BULK_SMEM }, 16> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let p = *d_pos;
        if p <= 0 {
            return;
        }
        let first = if p > cap { p - cap } else { 0 };
        let total_len = p - first;
        let split = blockIdx_y() as i32;
        let chunk = (total_len + KV_SPLIT_K as i32 - 1) / KV_SPLIT_K as i32;
        let t_start = first + split * chunk;
        let t_end = core::cmp::min(p, t_start + chunk);
        dspark_bulk_fp8_range(qkv, ck, cv, cks, cvs, cap, split, t_start, t_end, split_buf, SharedArray::as_raw_mut_ptr(&raw mut SMEM));
    }
}

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn dspark_bulk_fp8_range(
    qkv: *const f32,
    ck: *const u16,
    cv: *const u16,
    cks: *const u16,
    cvs: *const u16,
    cap: i32,
    split: i32,
    t_start: i32,
    t_end: i32,
    split_buf: *mut f32,
    smem: *mut u8,
) {
    unsafe {
        let kvh = blockIdx_x() as usize;
        let first = if t_start < t_end {
            bulk_tile_load::<true>(ck, cv, cks, cvs, cap, kvh, threadIdx_x() as usize, 128, t_start, core::cmp::min(32, t_end - t_start) as usize)
        } else {
            bulk_tile_load::<true>(ck, cv, cks, cvs, cap, kvh, threadIdx_x() as usize, 128, t_start, 0)
        };
        // Q_CNT = 24 = 3 x 8 rows: the B = 8 instance with the DSpark row stride.
        run_tc_bulk_splitk_pre::<8, QKV_STRIDE, true>(smem, qkv, ck, cv, cks, cvs, cap, kvh, split as usize, t_start, t_end, split_buf, first);
    }
}
/// Context positions whose K/V loads are in flight ahead of `dspark_block_attn`'s softmax chain.
const CTX_PREFETCH: usize = 4;

/// One online-softmax step for a 4-dim slice: returns the updated state.
#[inline(always)]
fn softmax_step(dot: f32, v: [f32; 4], m_val: &mut f32, d_val: &mut f32, acc: &mut [f32; 4]) {
    let score = mul_rn(warp_sum(dot), INV_SQRT_HEAD_DIM);
    let m_new = fmaxf(*m_val, score);
    let alpha = fast_expf(*m_val - m_new);
    let pw = fast_expf(score - m_new);
    *d_val = fmaf(*d_val, alpha, pw);
    let mut i = 0;
    while i < 4 {
        acc[i] = fmaf(pw, v[i], mul_rn(acc[i], alpha));
        i += 1;
    }
    *m_val = m_new;
}

/// `fmaf(q.x, k.x, fmaf(q.y, k.y, fmaf(q.z, k.z, q.w * k.w)))`.
#[inline(always)]
fn dot4(q: [f32; 4], k: [f32; 4]) -> f32 {
    fmaf(q[0], k[0], fmaf(q[1], k[1], fmaf(q[2], k[2], mul_rn(q[3], k[3]))))
}

/// `dspark_block_attn_kernel`: grid (24, 8), 256 threads (warp w = block
/// row w). Local layers (window > 0) also attend to the BF16 SWA context;
/// the global layer (window 0) merges the 64 bulk split-K partials instead.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dspark_block_attn(
    qkv: *const f32,
    ck: *const u16,
    cv: *const u16,
    split_buf: *const f32,
    cap: i32,
    window: i32,
    d_pos: *const i32,
    out: *mut f32,
    out_h: *mut u16,
) {
    static mut SM_M: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SM_D: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SM_ACC: SharedArray<f32, { 8 * HEAD_DIM }, 16> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let sm_m = SharedArray::as_raw_mut_ptr(&raw mut SM_M);
        let sm_d = SharedArray::as_raw_mut_ptr(&raw mut SM_D);
        let sm_acc = SharedArray::as_raw_mut_ptr(&raw mut SM_ACC);
        let h = blockIdx_x() as usize;
        let r = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let kvh = h / GQA_RATIO;
        let p = *d_pos;
        let has_bulk = window == 0 && !split_buf.is_null() && p > 0;
        let first = if window > 0 && p > window { p - window } else { 0 };

        let d_off = lane * 4;
        let qv = ld_f4(qkv.add(r * QKV_STRIDE + h * HEAD_DIM + d_off));
        let mut acc = [0.0f32; 4];
        let mut m_val = -1e30f32;
        let mut d_val = 0.0f32;

        // 1. Context KV [first, p) for the local layers.
        // Positions t = first + warp_i + 8j, processed in order. The K/V loads run
        // CTX_PREFETCH positions ahead of the softmax chain (slots t % cap stay inside
        // the cache even past p; those loads are never used), so the load latency is
        // not exposed once per position. Same arithmetic in the same order as before.
        if !has_bulk && window > 0 {
            let kvo = kvh * HEAD_DIM + d_off;
            let ldkv = |t: i32| -> ([u32; 2], [u32; 2]) {
                let off = (((t % cap) as usize) * KV_DIM + kvo) * 2;
                (ldg_u2((ck as *const u8).add(off)), ldg_u2((cv as *const u8).add(off)))
            };
            let mut t = first + warp_i as i32;
            let mut kv = [([0u32; 2], [0u32; 2]); CTX_PREFETCH];
            let mut j = 0;
            #[unroll]
            while j < CTX_PREFETCH {
                kv[j] = ldkv(t + 8 * j as i32);
                j += 1;
            }
            while t < p {
                let mut j = 0;
                #[unroll]
                while j < CTX_PREFETCH {
                    let tj = t + 8 * j as i32;
                    let (k4, v4) = kv[j];
                    kv[j] = ldkv(tj + 8 * CTX_PREFETCH as i32);
                    if tj < p {
                        let (k0, k1) = bf16x2_to_f32x2(k4[0]);
                        let (k2, k3) = bf16x2_to_f32x2(k4[1]);
                        let (v0, v1) = bf16x2_to_f32x2(v4[0]);
                        let (v2, v3) = bf16x2_to_f32x2(v4[1]);
                        softmax_step(dot4(qv, [k0, k1, k2, k3]), [v0, v1, v2, v3], &mut m_val, &mut d_val, &mut acc);
                    }
                    j += 1;
                }
                t += 8 * CTX_PREFETCH as i32;
            }
        }

        // 2. The 8 bidirectional block rows (warp w takes row w).
        {
            let base = qkv.add(warp_i * QKV_STRIDE + Q_DIM + kvh * HEAD_DIM + d_off);
            let bk = ld_f4(base);
            let bv = ld_f4(base.add(KV_DIM));
            softmax_step(dot4(qv, bk), bv, &mut m_val, &mut d_val, &mut acc);
        }

        if lane == 0 {
            *sm_m.add(warp_i) = m_val;
            *sm_d.add(warp_i) = d_val;
        }
        let mut i = 0;
        while i < 4 {
            *sm_acc.add(warp_i * HEAD_DIM + d_off + i) = acc[i];
            i += 1;
        }
        sync_threads();

        if tid < HEAD_DIM {
            let mut g_m = *sm_m;
            let mut w = 1;
            #[unroll]
            while w < 8 {
                g_m = fmaxf(g_m, *sm_m.add(w));
                w += 1;
            }
            let base_hr = (h * MAX_VERIFY_B + r) * KV_TOTAL_SPLITS;
            let s_m_in = split_buf.add(SPLIT_M_OFF);
            let s_d_in = split_buf.add(SPLIT_D_OFF);
            let s_o_in = split_buf.add(SPLIT_O_OFF);
            if has_bulk {
                let mut s = 0;
                #[unroll]
                while s < KV_SPLIT_K {
                    g_m = fmaxf(g_m, *s_m_in.add(base_hr + s));
                    s += 1;
                }
            }
            let mut g_d = 0.0f32;
            let mut g_o = 0.0f32;
            let mut w = 0;
            #[unroll]
            while w < 8 {
                let dv = *sm_d.add(w);
                if dv > 0.0 {
                    let sc = fast_expf(*sm_m.add(w) - g_m);
                    g_d = fmaf(dv, sc, g_d);
                    g_o = fmaf(*sm_acc.add(w * HEAD_DIM + tid), sc, g_o);
                }
                w += 1;
            }
            if has_bulk {
                let mut s = 0;
                #[unroll]
                while s < KV_SPLIT_K {
                    let sd = *s_d_in.add(base_hr + s);
                    if sd > 0.0 {
                        let sc = fast_expf(*s_m_in.add(base_hr + s) - g_m);
                        g_d = fmaf(sd, sc, g_d);
                        g_o = fmaf(*s_o_in.add((base_hr + s) * HEAD_DIM + tid), sc, g_o);
                    }
                    s += 1;
                }
            }
            let val = div_rn(g_o, g_d);
            let oidx = r * Q_DIM + h * HEAD_DIM + tid;
            if !out.is_null() {
                *out.add(oidx) = val;
            }
            if !out_h.is_null() {
                *out_h.add(oidx) = float2half(val);
            }
        }
    }
}

/// `dspark_swiglu_kernel`: grid ceil(B * 9216 / 256), 256 threads.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn dspark_swiglu(gu: *const f32, out: *mut f32, out_h: *mut u16, b_count: i32) {
    unsafe {
        pdl_wait();
        pdl_trigger();
        let idx = (blockIdx_x() * blockDim_x() + threadIdx_x()) as usize;
        let total = b_count as usize * MTP_INTERMEDIATE;
        if idx >= total {
            return;
        }
        let b = idx / MTP_INTERMEDIATE;
        let j = idx % MTP_INTERMEDIATE;
        let g = *gu.add(b * 2 * MTP_INTERMEDIATE + j);
        let u = *gu.add(b * 2 * MTP_INTERMEDIATE + MTP_INTERMEDIATE + j);
        let v = (g * fast_sigmoid(g)) * u;
        if !out.is_null() {
            *out.add(idx) = v;
        }
        if !out_h.is_null() {
            *out_h.add(idx) = float2half(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Markov head: RNN projection, folded unembed argmax with confidence
// ---------------------------------------------------------------------------

/// `markov_rnn_proj_kernel`: grid 24, 128 threads (8 rows per warp).
/// raw[768] = rnn_w_sp @ [state; w1[prev_tok]] + rnn_bias + c_h_i + c_f.
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn markov_rnn_proj(
    rnn_w_sp: *const u16,
    rnn_bias: *const u16,
    markov_w1: *const u16,
    c_h_i: *const f32,
    c_f: *const f32,
    prev_tok_ptr: *const i32,
    state: *const f32,
    out_raw: *mut f32,
) {
    static mut SP: SharedArray<f32, 512, 16> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let sp = SharedArray::as_raw_mut_ptr(&raw mut SP);
        let tid = threadIdx_x() as usize;
        let prev_tok = *prev_tok_ptr as usize;
        let w1_row = markov_w1.add(prev_tok * MTP_MARKOV_RANK);
        *sp.add(tid) = *state.add(tid);
        *sp.add(128 + tid) = *state.add(128 + tid);
        *sp.add(256 + tid) = bf16_to_f32(*w1_row.add(tid));
        *sp.add(384 + tid) = bf16_to_f32(*w1_row.add(128 + tid));
        sync_threads();

        let warp_i = tid >> 5;
        let lane = tid & 31;
        let base_r = blockIdx_x() as usize * 32 + warp_i * 8;
        let mut rr = 0;
        #[unroll]
        while rr < 8 {
            let r = base_r + rr;
            let wr = rnn_w_sp.add(r * 512) as *const u8;
            let mut acc = 0.0f32;
            let mut iter = 0;
            #[unroll]
            while iter < 2 {
                let v = lane + iter * 32;
                let w8 = ldg_u4(wr.add(v * 16));
                let (w0x, w0y) = bf16x2_to_f32x2(w8[0]);
                let (w1x, w1y) = bf16x2_to_f32x2(w8[1]);
                let (w2x, w2y) = bf16x2_to_f32x2(w8[2]);
                let (w3x, w3y) = bf16x2_to_f32x2(w8[3]);
                let x0 = ld_f4(sp.add(v << 3));
                let x1 = ld_f4(sp.add((v << 3) + 4));
                acc = fmaf(w0x, x0[0], acc);
                acc = fmaf(w0y, x0[1], acc);
                acc = fmaf(w1x, x0[2], acc);
                acc = fmaf(w1y, x0[3], acc);
                acc = fmaf(w2x, x1[0], acc);
                acc = fmaf(w2y, x1[1], acc);
                acc = fmaf(w3x, x1[2], acc);
                acc = fmaf(w3y, x1[3], acc);
                iter += 1;
            }
            let val = warp_sum(acc);
            if lane == 0 {
                *out_raw.add(r) = val + bf16_to_f32(*rnn_bias.add(r)) + *c_h_i.add(r) + *c_f.add(r);
            }
            rr += 1;
        }
    }
}

/// Merge (cm, csum) with another partial (om, os): nvcc's
/// `csum = fma(os, e(om - nm), csum > 0 ? csum * e(cm - nm) : 0)`.
#[inline(always)]
fn merge_lse(cm: &mut f32, csum: &mut f32, om: f32, os: f32) {
    if os > 0.0 {
        let nm = fmaxf(*cm, om);
        let first = if *csum > 0.0 { mul_rn(*csum, fast_expf(*cm - nm)) } else { 0.0 };
        *csum = fmaf(os, fast_expf(om - nm), first);
        *cm = nm;
    }
}

#[inline(always)]
fn better(v: f32, i: i32, bv: f32, bi: i32) -> bool {
    v > bv || (v == bv && i < bi)
}

/// `markov_fold_argmax_stage1_kernel<2>`: grid 340, 256 threads. Updates the
/// RNN state (block 0), then for each draft-vocab row: base logit + fold_b +
/// NVFP4 fold_w @ tanh(lat); per-block argmax and softcapped log-sum-exp.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn markov_fold_argmax_stage1_nvfp4(
    fold_data: *const u8,
    fold_scales: *const u8,
    fold_gscale: f32,
    base_logits: *const f32,
    fold_b: *const f32,
    raw_rnn: *const f32,
    state: *mut f32,
    blk_val: *mut f32,
    blk_idx: *mut i32,
    blk_cm: *mut f32,
    blk_csum: *mut f32,
    v: i32,
) {
    static mut S_LH: SharedArray<u16, 256, 16> = SharedArray::UNINIT;
    static mut S_LF: SharedArray<f32, 256, 16> = SharedArray::UNINIT;
    static mut SV: SharedArray<f32, 16> = SharedArray::UNINIT;
    static mut SCM: SharedArray<f32, 16> = SharedArray::UNINIT;
    static mut SCS: SharedArray<f32, 16> = SharedArray::UNINIT;
    static mut SI: SharedArray<i32, 16> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let s_lh = SharedArray::as_raw_mut_ptr(&raw mut S_LH);
        let s_lf = SharedArray::as_raw_mut_ptr(&raw mut S_LF);
        let sv = SharedArray::as_raw_mut_ptr(&raw mut SV);
        let scm = SharedArray::as_raw_mut_ptr(&raw mut SCM);
        let scs = SharedArray::as_raw_mut_ptr(&raw mut SCS);
        let si = SharedArray::as_raw_mut_ptr(&raw mut SI);
        let tid = threadIdx_x() as usize;
        let lat_v = fast_tanh(*raw_rnn.add(512 + tid));
        *s_lf.add(tid) = lat_v;
        *s_lh.add(tid) = float2half(lat_v);
        if blockIdx_x() == 0 {
            let gate = fast_sigmoid(*raw_rnn.add(tid));
            let cand = fast_tanh(*raw_rnn.add(256 + tid));
            // nvcc: fma(gate, state, (1 - gate) * cand)
            *state.add(tid) = fmaf(gate, *state.add(tid), mul_rn(1.0 - gate, cand));
        }
        sync_threads();

        let r_in_blk = tid >> 4;
        let lane = tid & 15;
        let v = v as usize;
        let mut row = blockIdx_x() as usize * 16 + r_in_blk;
        let tr = gridDim_x() as usize * 16;
        let xh = s_lh.add(lane << 4) as *const [u32; 4];
        let xb0 = *xh;
        let xb1 = *xh.add(1);

        let mut bv = -1e30f32;
        let mut bi = 0i32;
        let mut cm = -1e30f32;
        let mut csum = 0.0f32;
        while row < v {
            let w8 = ldg_u2(fold_data.add(row * 128 + lane * 8));
            let sc = cvt_e4m3_to_float(ldg_u8(fold_scales.add(row * 16 + lane)));
            let hws = half_warp_sum(mul_rn(block16_dot(w8, xb0, xb1), sc));
            if lane == 0 {
                let fb = if fold_b.is_null() { 0.0 } else { *fold_b.add(row) };
                // nvcc: fma(gscale, hws, base + fb)
                let total = fmaf(hws, fold_gscale, *base_logits.add(row) + fb);
                if total > bv {
                    bv = total;
                    bi = row as i32;
                }
                let cap = mul_rn(fast_tanh(total * (1.0 / FINAL_SOFTCAP)), FINAL_SOFTCAP);
                if cap > cm {
                    csum = if csum > 0.0 { fmaf(csum, fast_expf(cm - cap), 1.0) } else { 1.0 };
                    cm = cap;
                } else {
                    csum += fast_expf(cap - cm);
                }
            }
            row += tr;
        }

        if lane == 0 {
            *sv.add(r_in_blk) = bv;
            *si.add(r_in_blk) = bi;
            *scm.add(r_in_blk) = cm;
            *scs.add(r_in_blk) = csum;
        }
        sync_threads();
        if tid == 0 {
            bv = *sv;
            bi = *si;
            cm = *scm;
            csum = *scs;
            let mut w = 1;
            while w < 16 {
                let (ov, oi) = (*sv.add(w), *si.add(w));
                if better(ov, oi, bv, bi) {
                    bv = ov;
                    bi = oi;
                }
                merge_lse(&mut cm, &mut csum, *scm.add(w), *scs.add(w));
                w += 1;
            }
            let bx = blockIdx_x() as usize;
            *blk_val.add(bx) = bv;
            *blk_idx.add(bx) = bi;
            *blk_cm.add(bx) = cm;
            *blk_csum.add(bx) = csum;
        }
    }
}

/// `markov_fold_argmax_stage2_kernel`: grid 1, 256 threads. Final argmax over
/// the 340 block partials (remapped through `draft_vocab_ids` when non-null)
/// and the draft confidence 1 / sum(exp(cap - max)).
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn markov_fold_argmax_stage2(
    blk_val: *const f32,
    blk_idx: *const i32,
    blk_cm: *const f32,
    blk_csum: *const f32,
    draft_vocab_ids: *const i32,
    out_tok: *mut i32,
    out_conf: *mut f32,
) {
    static mut SV: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SCM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SCS: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SI: SharedArray<i32, 8> = SharedArray::UNINIT;
    unsafe {
        pdl_wait();
        pdl_trigger();
        let sv = SharedArray::as_raw_mut_ptr(&raw mut SV);
        let scm = SharedArray::as_raw_mut_ptr(&raw mut SCM);
        let scs = SharedArray::as_raw_mut_ptr(&raw mut SCS);
        let si = SharedArray::as_raw_mut_ptr(&raw mut SI);
        let tid = threadIdx_x() as usize;
        let mut bv = -1e30f32;
        let mut bi = 0i32;
        let mut cm = -1e30f32;
        let mut csum = 0.0f32;
        let mut i = tid;
        while i < MARKOV_BLOCKS {
            let (v, idx) = (*blk_val.add(i), *blk_idx.add(i));
            if better(v, idx, bv, bi) {
                bv = v;
                bi = idx;
            }
            merge_lse(&mut cm, &mut csum, *blk_cm.add(i), *blk_csum.add(i));
            i += 256;
        }
        let mut off = 16u32;
        #[unroll]
        while off > 0 {
            let ov = warp::shuffle_xor_f32_sync(0xffff_ffff, bv, off);
            let oi = warp::shuffle_xor_sync(0xffff_ffff, bi as u32, off) as i32;
            if better(ov, oi, bv, bi) {
                bv = ov;
                bi = oi;
            }
            let om = warp::shuffle_xor_f32_sync(0xffff_ffff, cm, off);
            let os = warp::shuffle_xor_f32_sync(0xffff_ffff, csum, off);
            merge_lse(&mut cm, &mut csum, om, os);
            off >>= 1;
        }
        if (tid & 31) == 0 {
            let w = tid >> 5;
            *sv.add(w) = bv;
            *si.add(w) = bi;
            *scm.add(w) = cm;
            *scs.add(w) = csum;
        }
        sync_threads();
        if tid == 0 {
            bv = *sv;
            bi = *si;
            cm = *scm;
            csum = *scs;
            let mut w = 1;
            while w < 8 {
                let (ov, oi) = (*sv.add(w), *si.add(w));
                if better(ov, oi, bv, bi) {
                    bv = ov;
                    bi = oi;
                }
                merge_lse(&mut cm, &mut csum, *scm.add(w), *scs.add(w));
                w += 1;
            }
            *out_tok = if draft_vocab_ids.is_null() { bi } else { *draft_vocab_ids.add(bi as usize) };
            if !out_conf.is_null() {
                *out_conf = if csum > 0.0 { rcp_rn(csum) } else { 0.0 };
            }
        }
    }
}

/// `copy_f_prev_zero_state_kernel`: grid 9, 256 threads. f_prev =
/// f_ring[(p - 1) & 15] (zero at p = 0) and the RNN state zeroed.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn copy_f_prev_zero_state(f_ring: *const f32, d_pos: *const i32, out_f: *mut f32, state: *mut f32) {
    unsafe {
        pdl_wait();
        pdl_trigger();
        let i = (blockIdx_x() * blockDim_x() + threadIdx_x()) as usize;
        if i < HIDDEN {
            let prev_p = *d_pos - 1;
            *out_f.add(i) = if prev_p >= 0 { *f_ring.add((prev_p & 15) as usize * HIDDEN + i) } else { 0.0 };
        }
        if i < MTP_MARKOV_RANK {
            *state.add(i) = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

/// `spec_verify_kernel`: grid 1, 1 thread. Accepts target predictions while
/// they match the drafts, then the first mismatch or the bonus token;
/// advances `d_pos` and sets the next anchor.
#[kernel]
pub unsafe fn spec_verify(
    target_preds: *const i32,
    draft_tokens: *const i32,
    b_count: i32,
    out_accepted: *mut i32,
    out_num_accepted: *mut i32,
    d_pos: *mut i32,
    d_next_anchor: *mut i32,
) {
    unsafe {
        if threadIdx_x() != 0 {
            return;
        }
        let mut n_acc = 0i32;
        let mut i = 0;
        while i < b_count - 1 {
            let tp = *target_preds.add(i as usize);
            *out_accepted.add(n_acc as usize) = tp;
            n_acc += 1;
            if tp != *draft_tokens.add(i as usize) {
                *out_num_accepted = n_acc;
                *d_pos += n_acc;
                *d_next_anchor = tp;
                return;
            }
            i += 1;
        }
        let bonus = *target_preds.add((b_count - 1) as usize);
        *out_accepted.add(n_acc as usize) = bonus;
        n_acc += 1;
        *out_num_accepted = n_acc;
        *d_pos += n_acc;
        *d_next_anchor = bonus;
    }
}

/// `z_assemble_verify_tokens_kernel`: grid 1, num_drafts + 1 threads.
/// tokens = [anchor, drafts...].
#[kernel]
pub unsafe fn z_assemble_verify_tokens(d_anchor: *const i32, d_drafts: *const i32, d_tokens: *mut i32, num_drafts: i32) {
    unsafe {
        let i = threadIdx_x() as i32;
        if i == 0 {
            *d_tokens = *d_anchor;
        } else if i <= num_drafts {
            *d_tokens.add(i as usize) = *d_drafts.add((i - 1) as usize);
        }
    }
}

// ---------------------------------------------------------------------------
// Load time: fold the Markov head's hidden_out into the draft unembed
// ---------------------------------------------------------------------------

/// `z_compute_dspark_fold_bias_kernel`: grid ceil(V / 8), block (32, 8).
/// fold_b[v] = unembed[v] . hidden_out_b.
#[kernel]
pub unsafe fn z_compute_dspark_fold_bias(unembed: *const u16, hidden_out_b: *const u16, fold_b: *mut f32, v: i32, k: i32) {
    unsafe {
        let vi = (blockIdx_x() * blockDim_y() + threadIdx_y()) as i32;
        if vi >= v {
            return;
        }
        let lane = threadIdx_x() as usize;
        let k = k as usize;
        let row = unembed.add(vi as usize * k);
        let mut acc = 0.0f32;
        let mut kk = lane;
        while kk < k {
            acc = fmaf(bf16_to_f32(*row.add(kk)), bf16_to_f32(*hidden_out_b.add(kk)), acc);
            kk += 32;
        }
        acc = warp_sum(acc);
        if lane == 0 {
            *fold_b.add(vi as usize) = acc;
        }
    }
}

/// `z_compute_dspark_fold_w_kernel`: grid V, 256 threads (R = 256).
/// fold_out[v, r] = bf16(sum_k unembed[v, k] * hidden_out_w[k, r]).
#[kernel]
#[launch_bounds(256)]
pub unsafe fn z_compute_dspark_fold_w(unembed: *const u16, hidden_out_w: *const u16, fold_out: *mut u16, v: i32, k: i32, r_dim: i32) {
    static mut U_ROW: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
    unsafe {
        let u_row = SharedArray::as_raw_mut_ptr(&raw mut U_ROW);
        let vi = blockIdx_x() as i32;
        if vi >= v {
            return;
        }
        let r = threadIdx_x() as usize;
        let k = k as usize;
        let rd = r_dim as usize;
        let u_ptr = unembed.add(vi as usize * k);
        let mut kk = r;
        while kk < k {
            *u_row.add(kk) = bf16_to_f32(*u_ptr.add(kk));
            kk += 256;
        }
        sync_threads();
        let mut acc = 0.0f32;
        let mut kk = 0;
        while kk < k {
            acc = fmaf(*u_row.add(kk), bf16_to_f32(*hidden_out_w.add(kk * rd + r)), acc);
            kk += 1;
        }
        *fold_out.add(vi as usize * rd + r) = f32_to_bf16(acc);
    }
}

/// `z_split_dspark_rnn_kernel`: grid 768, 256 threads. Splits the
/// [768, 512 + 2 * 2304] RNN projection into w_sp, w_h and w_f.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn z_split_dspark_rnn(src: *const u16, w_sp: *mut u16, w_h: *mut u16, w_f: *mut u16) {
    unsafe {
        const K_IN: usize = 2 * MTP_MARKOV_RANK + 2 * HIDDEN;
        let o = blockIdx_x() as usize;
        let row = src.add(o * K_IN);
        let mut c = threadIdx_x() as usize;
        while c < K_IN {
            let v = *row.add(c);
            if c < 512 {
                *w_sp.add(o * 512 + c) = v;
            } else if c < 512 + HIDDEN {
                *w_h.add(o * HIDDEN + (c - 512)) = v;
            } else {
                *w_f.add(o * HIDDEN + (c - 512 - HIDDEN)) = v;
            }
            c += 256;
        }
    }
}
