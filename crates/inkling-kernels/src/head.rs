//! Token in and out: embedding + norm, the exact-mode (BF16) LM head, the
//! argmax/sampling stage, and the per-step position/token update.
//!
//! Ports of squid5090 @ 3ee093c:
//! - `embed_norm_kernel` (kernels_inkling.cu:27) as `embed_norm`;
//! - `lm_head_stage1_kernel<1, 0>` (kernels_gemv.cu:872) as `lm_head_stage1_bf16_b1`;
//! - `lm_head_stage2_argmax_kernel` (kernels_gemv.cu:1176) as `lm_head_stage2_argmax`;
//! - `z_inc_pos_and_copy_tok_kernel` (cuda_abi.cu:211) as `inc_pos_and_copy_tok`.

use crate::common::*;
use crate::{FINAL_SOFTCAP, HIDDEN, LM_HEAD_BLOCKS, fast_tanh, fmaf, ld_dep_f32, pdl_trigger, pdl_wait};
use cuda_device::thread::{blockIdx_x, gridDim_x, sync_threads, threadIdx_x};
use cuda_device::{SharedArray, device, kernel, launch_bounds, ptx_asm, warp};

/// `embed_norm_kernel`: embedding row lookup and RMSNorm. Grid B, 256 threads.
#[kernel]
pub unsafe fn embed_norm(embed: *const u16, norm_w: *const u16, tokens: *const i32, out: *mut f32) {
    static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV_RMS: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv_rms = SharedArray::as_raw_mut_ptr(&raw mut INV_RMS);
        let b = blockIdx_x() as usize;
        let tid = threadIdx_x() as usize;
        let tok = *tokens.add(b);
        let row = embed.add(tok as usize * HIDDEN);
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
            let mut tot = 0.0f32;
            let mut i = 0;
            while i < 8 {
                tot += *sm.add(i);
                i += 1;
            }
            *inv_rms = rsqrtf(div_rn(tot, HIDDEN as f32) + RMS_EPS);
        }
        sync_threads();
        let s = *inv_rms;
        let o = out.add(b * HIDDEN);
        let mut d = tid;
        while d < HIDDEN {
            let mut v = bf16_to_f32(*row.add(d)) * s;
            v = round_bf16(v) * bf16_to_f32(*norm_w.add(d));
            *o.add(d) = round_bf16(v);
            d += 256;
        }
    }
}

const BF_TPR: usize = 32;
const BF_RPB: usize = 256 / BF_TPR; // 8 rows per block

/// `lm_head_stage1_kernel<B, 0>` (BF16 weights). Grid 680, 256 threads.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn lm_head_stage1_bf16_body<const B: usize>(
    s_x16: *mut u16,
    sm_val: *mut f32,
    sm_idx: *mut i32,
    w_data: *const u16,
    x: *const f32,
    out_logits: *mut f32,
    blk_max_val: *mut f32,
    blk_max_idx: *mut i32,
    sample_cfg: *const f32,
    v: i32,
) {
    unsafe {
        let write_logits = !out_logits.is_null() && (sample_cfg.is_null() || *sample_cfg > 0.0);
        let tid = threadIdx_x() as usize;
        let total_x = B * HIDDEN;
        pdl_wait();
        pdl_trigger();
        let mut i = tid;
        while i < total_x {
            *s_x16.add(i) = f32_to_bf16(ld_dep_f32(x.add(i)));
            i += 256;
        }
        sync_threads();

        let r_in_blk = tid >> 5;
        let lane = tid & (BF_TPR - 1);
        let global_r = blockIdx_x() as usize * BF_RPB + r_in_blk;
        let total_r = gridDim_x() as usize * BF_RPB;

        let mut best_val = [-1e30f32; B];
        let mut best_idx = [0i32; B];

        let mut row = global_r;
        while row < v as usize {
            let mut acc = [0.0f32; B];
            let wr = w_data.add(row * HIDDEN) as *const u8;
            const NVEC: usize = HIDDEN >> 3;
            let mut vv = lane;
            while vv < NVEC {
                let w8 = ldg_u4(wr.add(vv * 16));
                let (w0x, w0y) = bf16x2_to_f32x2(w8[0]);
                let (w1x, w1y) = bf16x2_to_f32x2(w8[1]);
                let (w2x, w2y) = bf16x2_to_f32x2(w8[2]);
                let (w3x, w3y) = bf16x2_to_f32x2(w8[3]);
                let koff = vv << 3;
                let mut b = 0;
                #[unroll]
                while b < B {
                    let xs = (s_x16 as *const u32).add((b * HIDDEN + koff) >> 1);
                    let (x0x, x0y) = bf16x2_to_f32x2(*xs);
                    let (x1x, x1y) = bf16x2_to_f32x2(*xs.add(1));
                    let (x2x, x2y) = bf16x2_to_f32x2(*xs.add(2));
                    let (x3x, x3y) = bf16x2_to_f32x2(*xs.add(3));
                    let mut a = acc[b];
                    a = fmaf(w0x, x0x, a);
                    a = fmaf(w0y, x0y, a);
                    a = fmaf(w1x, x1x, a);
                    a = fmaf(w1y, x1y, a);
                    a = fmaf(w2x, x2x, a);
                    a = fmaf(w2y, x2y, a);
                    a = fmaf(w3x, x3x, a);
                    a = fmaf(w3y, x3y, a);
                    acc[b] = a;
                    b += 1;
                }
                vv += 32;
            }

            let mut b = 0;
            #[unroll]
            while b < B {
                let val = warp_sum(acc[b]);
                if lane == 0 {
                    if write_logits {
                        let cap_val = fast_tanh(val * (1.0 / FINAL_SOFTCAP)) * FINAL_SOFTCAP;
                        *out_logits.add(b * v as usize + row) = cap_val;
                        if cap_val > best_val[b] {
                            best_val[b] = cap_val;
                            best_idx[b] = row as i32;
                        }
                    } else if val > best_val[b] {
                        best_val[b] = val;
                        best_idx[b] = row as i32;
                    }
                }
                b += 1;
            }
            row += total_r;
        }

        if lane == 0 {
            let mut b = 0;
            #[unroll]
            while b < B {
                *sm_val.add(b * BF_RPB + r_in_blk) = best_val[b];
                *sm_idx.add(b * BF_RPB + r_in_blk) = best_idx[b];
                b += 1;
            }
        }
        sync_threads();
        if tid < B {
            let b = tid;
            let mut bv = *sm_val.add(b * BF_RPB);
            let mut bi = *sm_idx.add(b * BF_RPB);
            let mut w = 1;
            while w < BF_RPB {
                let cv = *sm_val.add(b * BF_RPB + w);
                let ci = *sm_idx.add(b * BF_RPB + w);
                if cv > bv || (cv == bv && ci < bi) {
                    bv = cv;
                    bi = ci;
                }
                w += 1;
            }
            *blk_max_val.add(b * LM_HEAD_BLOCKS + blockIdx_x() as usize) = bv;
            *blk_max_idx.add(b * LM_HEAD_BLOCKS + blockIdx_x() as usize) = bi;
        }
    }
}

/// `lm_head_stage1_kernel<1, 0>`: exact-mode BF16 LM head, B=1. The
/// `w_scales`, `row_scales` and `global_scale` parameters are unused for BF16
/// and kept only so the parameter list matches squid5090's.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn lm_head_stage1_bf16_b1(
    w_data: *const u16,
    _w_scales: *const u8,
    _row_scales: *const f32,
    _global_scale: f32,
    x: *const f32,
    out_logits: *mut f32,
    blk_max_val: *mut f32,
    blk_max_idx: *mut i32,
    sample_cfg: *const f32,
    v: i32,
) {
    static mut S_X16: SharedArray<u16, { HIDDEN }, 16> = SharedArray::UNINIT;
    static mut SM_VAL: SharedArray<f32, { BF_RPB }> = SharedArray::UNINIT;
    static mut SM_IDX: SharedArray<i32, { BF_RPB }> = SharedArray::UNINIT;
    unsafe {
        lm_head_stage1_bf16_body::<1>(
            SharedArray::as_raw_mut_ptr(&raw mut S_X16),
            SharedArray::as_raw_mut_ptr(&raw mut SM_VAL),
            SharedArray::as_raw_mut_ptr(&raw mut SM_IDX),
            w_data, x, out_logits, blk_max_val, blk_max_idx, sample_cfg, v,
        )
    }
}

/// `SampleConfig` (config.hpp).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SampleConfig {
    pub inv_temp: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub pad: u32,
    pub seed: u64,
}

/// `hash_gumbel(seed, pos, tok)`.
#[inline(always)]
fn hash_gumbel(seed: u64, pos: i32, tok: i32) -> f32 {
    let mut h = seed
        ^ ((pos as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ ((tok as u32 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    let u = ((h >> 41) as f32 + 0.5) * (1.0 / 8_388_608.0);
    -fast_logf(-fast_logf(u))
}

/// Shared-memory `atomicAdd(float*)` (`atom.shared.add.f32`, as nvcc emits).
#[inline(always)]
unsafe fn atomic_add_f32(p: *mut f32, v: f32) {
    let addr = unsafe { cuda_device::shared::cvta_generic_to_shared_u32(p as *const u8) };
    let _old: f32;
    unsafe {
        ptx_asm!("atom.shared.add.f32 %0, [%1], %2;", out("=f") _old, in("r") addr, in("f") v, clobber("memory"));
    }
}

/// Shared-memory `atomicAdd(uint32_t*)`.
#[inline(always)]
unsafe fn atomic_add_u32(p: *mut u32, v: u32) {
    let addr = unsafe { cuda_device::shared::cvta_generic_to_shared_u32(p as *const u8) };
    let _old: u32;
    unsafe {
        ptx_asm!("atom.shared.add.u32 %0, [%1], %2;", out("=r") _old, in("r") addr, in("r") v, clobber("memory"));
    }
}

/// Warp-level (value, index) argmax, ties to the lower index.
#[inline(always)]
fn warp_argmax(mut bv: f32, mut bi: i32) -> (f32, i32) {
    let mut off = 16u32;
    while off > 0 {
        let ov = warp::shuffle_xor_f32_sync(0xffff_ffff, bv, off);
        let oi = warp::shuffle_xor_sync(0xffff_ffff, bi as u32, off) as i32;
        if ov > bv || (ov == bv && oi < bi) {
            bv = ov;
            bi = oi;
        }
        off >>= 1;
    }
    (bv, bi)
}

/// `lm_head_stage2_argmax_kernel`: reduces the 680 block maxima to the greedy
/// token, or samples with top-p/top-k and hash-Gumbel noise when `sample_cfg`
/// asks for temperature > 0. Grid B, 256 threads.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn lm_head_stage2_argmax(
    blk_max_val: *const f32,
    blk_max_idx: *const i32,
    out_logits: *const f32,
    out_tokens: *mut i32,
    sample_cfg: *const SampleConfig,
    d_pos: *const i32,
    v: i32,
    bsz: i32,
) {
    static mut SV: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SI: SharedArray<i32, 8> = SharedArray::UNINIT;
    static mut S_MAX: SharedArray<f32, 1> = SharedArray::UNINIT;
    static mut S_MAX_IDX: SharedArray<i32, 1> = SharedArray::UNINIT;
    static mut HIST_SUM: SharedArray<f32, 256> = SharedArray::UNINIT;
    static mut HIST_CNT: SharedArray<u32, 256> = SharedArray::UNINIT;
    static mut S_TAU: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sv = SharedArray::as_raw_mut_ptr(&raw mut SV);
        let si = SharedArray::as_raw_mut_ptr(&raw mut SI);
        let s_max = SharedArray::as_raw_mut_ptr(&raw mut S_MAX);
        let s_max_idx = SharedArray::as_raw_mut_ptr(&raw mut S_MAX_IDX);
        let hist_sum = SharedArray::as_raw_mut_ptr(&raw mut HIST_SUM);
        let hist_cnt = SharedArray::as_raw_mut_ptr(&raw mut HIST_CNT);
        let s_tau = SharedArray::as_raw_mut_ptr(&raw mut S_TAU);

        pdl_wait();
        pdl_trigger();
        let b = blockIdx_x() as i32;
        if b >= bsz {
            return;
        }
        let b = b as usize;
        let tid = threadIdx_x() as usize;
        let mut bv = -1e30f32;
        let mut bi = 0i32;
        let mut i = tid;
        while i < LM_HEAD_BLOCKS {
            let val = *blk_max_val.add(b * LM_HEAD_BLOCKS + i);
            let idx = *blk_max_idx.add(b * LM_HEAD_BLOCKS + i);
            if val > bv || (val == bv && idx < bi) {
                bv = val;
                bi = idx;
            }
            i += 256;
        }
        (bv, bi) = warp_argmax(bv, bi);
        if (tid & 31) == 0 {
            *sv.add(tid >> 5) = bv;
            *si.add(tid >> 5) = bi;
        }
        sync_threads();

        let do_sample = !sample_cfg.is_null() && (*sample_cfg).inv_temp > 0.0 && !out_logits.is_null();
        if !do_sample {
            if tid == 0 {
                bv = *sv;
                bi = *si;
                let mut w = 1;
                while w < 8 {
                    if *sv.add(w) > bv || (*sv.add(w) == bv && *si.add(w) < bi) {
                        bv = *sv.add(w);
                        bi = *si.add(w);
                    }
                    w += 1;
                }
                *out_tokens.add(b) = bi;
            }
            return;
        }

        if tid == 0 {
            bv = *sv;
            bi = *si;
            let mut w = 1;
            while w < 8 {
                if *sv.add(w) > bv || (*sv.add(w) == bv && *si.add(w) < bi) {
                    bv = *sv.add(w);
                    bi = *si.add(w);
                }
                w += 1;
            }
            *s_max = bv;
            *s_max_idx = bi;
        }
        sync_threads();

        let max_logit = *s_max;
        let greedy_idx = *s_max_idx;
        let cfg = *sample_cfg;
        let inv_temp = cfg.inv_temp;
        let top_p = cfg.top_p;
        let top_k = cfg.top_k;
        let seed = cfg.seed;
        let pos = if d_pos.is_null() { b as i32 } else { *d_pos + b as i32 };
        let logits = out_logits.add(b * v as usize);

        const HIST_RANGE: f32 = 20.0;
        const HIST_SCALE: f32 = 256.0 / HIST_RANGE;

        if (top_p > 0.0 && top_p < 0.9999) || top_k > 0 {
            *hist_sum.add(tid) = 0.0;
            *hist_cnt.add(tid) = 0;
            sync_threads();

            let mut row = tid;
            while row < v as usize {
                let d = (max_logit - *logits.add(row)) * inv_temp;
                if d < HIST_RANGE {
                    let mut bin = (d * HIST_SCALE) as i32;
                    if bin < 0 {
                        bin = 0;
                    } else if bin > 255 {
                        bin = 255;
                    }
                    atomic_add_f32(hist_sum.add(bin as usize), fast_expf(-d));
                    atomic_add_u32(hist_cnt.add(bin as usize), 1);
                }
                row += 256;
            }
            sync_threads();

            if tid == 0 {
                let mut total_z = 0.0f32;
                let mut k = 0;
                while k < 256 {
                    total_z += *hist_sum.add(k);
                    k += 1;
                }
                let mut cum_z = 0.0f32;
                let mut cum_c = 0u32;
                let mut cut_bin = 255i32;
                let target_z = if top_p > 0.0 && top_p < 1.0 { top_p * total_z } else { total_z };
                let mut k = 0;
                while k < 256 {
                    cum_z += *hist_sum.add(k);
                    cum_c += *hist_cnt.add(k);
                    let hit_p = top_p > 0.0 && top_p < 0.9999 && cum_z >= target_z;
                    let hit_k = top_k > 0 && cum_c >= top_k as u32;
                    if hit_p || hit_k {
                        cut_bin = k as i32;
                        break;
                    }
                    k += 1;
                }
                *s_tau = max_logit - div_rn(div_rn((cut_bin + 1) as f32, HIST_SCALE), inv_temp);
            }
            sync_threads();
        } else {
            if tid == 0 {
                *s_tau = max_logit - div_rn(HIST_RANGE, inv_temp);
            }
            sync_threads();
        }

        let tau = *s_tau;
        let mut best_s = -1e30f32;
        let mut best_t = greedy_idx;
        let mut row = tid;
        while row < v as usize {
            let l = *logits.add(row);
            if l >= tau {
                let sc = fmaf(l - max_logit, inv_temp, hash_gumbel(seed, pos, row as i32));
                if sc > best_s {
                    best_s = sc;
                    best_t = row as i32;
                }
            }
            row += 256;
        }
        (best_s, best_t) = warp_argmax(best_s, best_t);
        if (tid & 31) == 0 {
            *sv.add(tid >> 5) = best_s;
            *si.add(tid >> 5) = best_t;
        }
        sync_threads();
        if tid == 0 {
            best_s = *sv;
            best_t = *si;
            let mut w = 1;
            while w < 8 {
                if *sv.add(w) > best_s || (*sv.add(w) == best_s && *si.add(w) < best_t) {
                    best_s = *sv.add(w);
                    best_t = *si.add(w);
                }
                w += 1;
            }
            *out_tokens.add(b) = best_t;
        }
    }
}

/// `z_inc_pos_and_copy_tok_kernel`: advances `d_pos` and feeds the sampled
/// token back as the next input. One thread.
#[kernel]
pub unsafe fn inc_pos_and_copy_tok(d_pos: *mut i32, d_tok_in: *mut i32, d_tok_out: *const i32) {
    if threadIdx_x() == 0 {
        unsafe {
            *d_pos += 1;
            *d_tok_in = *d_tok_out;
        }
    }
}
