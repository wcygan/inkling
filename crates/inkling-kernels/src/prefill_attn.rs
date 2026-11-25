//! Wide prefill (16-256 tokens per step): the K/V short-conv prep kernels of `attn.rs` without their
//! ring writes. With more than 13 tokens in one launch, token b's ring write (slot (p0 + b) & 15) would
//! race with tokens 0-2 reading the slots of positions p0-3..p0-1; here the rings are only read, and
//! `prefill_ring_store` writes the chunk's last 16 positions afterwards. Otherwise identical (same
//! arithmetic, grid (64, n), 128 threads, for tokens b0..b0 + n of the chunk).

use crate::common::*;
use crate::{HIDDEN, fmaf, pdl_trigger, pdl_wait};
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{SharedArray, kernel, launch_bounds, warp};

/// Sum of the four per-warp partials, in squid5090's order.
#[inline(always)]
unsafe fn sum4(sm: *const f32) -> f32 {
    unsafe { *sm + *sm.add(1) + *sm.add(2) + *sm.add(3) }
}

/// The ring-buffer taps of a 4-tap short conv for channel `c` of a
/// `KV_DIM`-wide projection: (t2, t1, t0) = values at p-1, p-2, p-3.
#[inline(always)]
unsafe fn kv_taps(proj: *const f32, ring: *const f32, b: usize, p: i32, c: usize) -> (f32, f32, f32) {
    unsafe {
        let at = |k: usize| -> f32 {
            if b >= k {
                *proj.add((b - k) * KV_DIM + c)
            } else if p >= k as i32 {
                *ring.add((((p - k as i32) & 15) as usize) * KV_DIM + c)
            } else {
                0.0
            }
        };
        (at(1), at(2), at(3))
    }
}

/// `mega_local_sconv_prep_store_kernel`: local-layer Q/R clip, K/V short
/// conv, K clip, V RMS norm and BF16 KV store. Grid (64, B), 128 threads.
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_local_sconv_prep_store(
    qrgs: *mut f32,
    k_proj: *const f32,
    v_proj: *const f32,
    k_sconv_w: *const u16,
    v_sconv_w: *const u16,
    k_ring: *const f32,
    v_ring: *const f32,
    q_clip_scale: *const u16,
    rel_clip_scale: *const u16,
    k_cache: *mut u16,
    v_cache: *mut u16,
    kv_cap: i32,
    d_pos: *const i32,
    b0: i32,
) {
    static mut SM: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut FACTOR: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let factor = SharedArray::as_raw_mut_ptr(&raw mut FACTOR);
        let kind_id = blockIdx_x() as usize;
        let b = b0 as usize + blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        pdl_wait();
        pdl_trigger();
        let pos0 = *d_pos;
        let p = pos0 + b as i32;

        if kind_id < 24 {
            let h = kind_id;
            let qp = qrgs.add(b * QRGS_STRIDE + h * HEAD_DIM);
            let val = *qp.add(tid) * bf16_to_f32(*q_clip_scale.add(h * HEAD_DIM + tid));
            let sq = warp_sum(mul_rn(val, val));
            if (tid & 31) == 0 {
                *sm.add(tid >> 5) = sq;
            }
            sync_threads();
            if tid == 0 {
                *factor = fminf(div_rn(QK_CLIP, sqrtf(sum4(sm)) + 1e-6), 1.0);
            }
            sync_threads();
            *qp.add(tid) = val * *factor;
        } else if kind_id < 48 {
            if tid >= D_REL {
                return;
            }
            let h = kind_id - 24;
            let rp = qrgs.add(b * QRGS_STRIDE + Q_DIM + h * D_REL);
            let val = *rp.add(tid) * bf16_to_f32(*rel_clip_scale.add(h * D_REL + tid));
            let mut sq = mul_rn(val, val);
            let mut off = 8u32;
            #[unroll]
            while off > 0 {
                sq += warp::shuffle_xor_f32_sync(0xffff, sq, off);
                off >>= 1;
            }
            *rp.add(tid) = val * fminf(div_rn(REL_CLIP, sqrtf(sq) + 1e-6), 1.0);
        } else {
            let is_k = kind_id < 56;
            let h = if is_k { kind_id - 48 } else { kind_id - 56 };
            let (proj, ring, sw) = if is_k { (k_proj, k_ring, k_sconv_w) } else { (v_proj, v_ring, v_sconv_w) };
            let c = h * HEAD_DIM + tid;
            let cur = *proj.add(b * KV_DIM + c);
            let (t2, t1, t0) = kv_taps(proj, ring, b, p, c);
            let w0 = bf16_to_f32(*sw.add(c * 4));
            let w1 = bf16_to_f32(*sw.add(c * 4 + 1));
            let w2 = bf16_to_f32(*sw.add(c * 4 + 2));
            let w3 = bf16_to_f32(*sw.add(c * 4 + 3));
            let val = (fmaf(w3, cur, fmaf(w2, t2, fmaf(w1, t1, w0 * t0))) + cur) * INV_SQRT_5;
            let sq = warp_sum(mul_rn(val, val));
            if (tid & 31) == 0 {
                *sm.add(tid >> 5) = sq;
            }
            sync_threads();
            if tid == 0 {
                *factor = if is_k {
                    fminf(div_rn(QK_CLIP, sqrtf(sum4(sm)) + 1e-6), 1.0)
                } else {
                    rsqrtf(div_rn(sum4(sm), HEAD_DIM as f32) + RMS_EPS)
                };
            }
            sync_threads();
            let cache = if is_k { k_cache } else { v_cache };
            *cache.add((p % kv_cap) as usize * KV_DIM + c) = f32_to_bf16(val * *factor);
        }
    }
}

/// `mega_global_sconv_prep_store_kernel<0>`: global-layer Q/R clip with the
/// log-position temperature, dynamic K/V short conv, K clip, V norm and BF16
/// KV store. Grid (64, B), 128 threads. `k_scale`/`v_scale` are unused for BF16.
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_global_sconv_prep_store_bf16kv(
    qrgs: *mut f32,
    k_proj: *const f32,
    v_proj: *const f32,
    in_k: *const f32,
    in_v: *const f32,
    k_sconv_w: *const u16,
    v_sconv_w: *const u16,
    k_sconv_proj: *const u16,
    v_sconv_proj: *const u16,
    k_ring: *const f32,
    v_ring: *const f32,
    q_clip_scale: *const u16,
    rel_clip_scale: *const u16,
    k_cache: *mut u16,
    v_cache: *mut u16,
    fp8_shadow: *mut u16,
    _v_scale: *mut u16,
    kv_cap: i32,
    d_pos: *const i32,
    b0: i32,
) {
    static mut SM: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut S_DT: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut FACTOR: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let s_dt = SharedArray::as_raw_mut_ptr(&raw mut S_DT);
        let factor = SharedArray::as_raw_mut_ptr(&raw mut FACTOR);
        let kind_id = blockIdx_x() as usize;
        let b = b0 as usize + blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        pdl_wait();
        pdl_trigger();
        let p = *d_pos + b as i32;
        let ratio = fmaxf(div_rn((p + 1) as f32, LOG_SCALE_FLOOR as f32), 1.0);
        let tau = fmaf(LOG_SCALE_ALPHA, logf(ratio), 1.0);

        if kind_id < 24 {
            let h = kind_id;
            let qp = qrgs.add(b * QRGS_STRIDE + h * HEAD_DIM);
            let val = *qp.add(tid) * bf16_to_f32(*q_clip_scale.add(h * HEAD_DIM + tid));
            let sq = warp_sum(mul_rn(val, val));
            if lane == 0 {
                *sm.add(warp_i) = sq;
            }
            sync_threads();
            if tid == 0 {
                *factor = fminf(div_rn(QK_CLIP, sqrtf(sum4(sm)) + 1e-6), 1.0) * tau;
            }
            sync_threads();
            *qp.add(tid) = val * *factor;
        } else if kind_id < 48 {
            if tid >= D_REL {
                return;
            }
            let h = kind_id - 24;
            let rp = qrgs.add(b * QRGS_STRIDE + Q_DIM + h * D_REL);
            let val = *rp.add(tid) * bf16_to_f32(*rel_clip_scale.add(h * D_REL + tid));
            let mut sq = mul_rn(val, val);
            let mut off = 8u32;
            #[unroll]
            while off > 0 {
                sq += warp::shuffle_xor_f32_sync(0xffff, sq, off);
                off >>= 1;
            }
            *rp.add(tid) = val * fminf(div_rn(REL_CLIP, sqrtf(sq) + 1e-6), 1.0) * tau;
        } else {
            let is_k = kind_id < 56;
            let h = if is_k { kind_id - 48 } else { kind_id - 56 };
            let (proj, ring, sw, sproj, x_in) = if is_k {
                (k_proj, k_ring, k_sconv_w, k_sconv_proj, in_k)
            } else {
                (v_proj, v_ring, v_sconv_w, v_sconv_proj, in_v)
            };
            // Four warps compute the four dynamic taps for this KV head.
            let wr = sproj.add((h * 4 + warp_i) * HIDDEN) as *const u8;
            let xb = x_in.add(b * HIDDEN);
            let mut dacc = 0.0f32;
            let mut iter = 0;
            #[unroll]
            while iter < 9 {
                let v = lane + iter * 32;
                let w8 = ldg_u4(wr.add(v * 16));
                let (w0x, w0y) = bf16x2_to_f32x2(w8[0]);
                let (w1x, w1y) = bf16x2_to_f32x2(w8[1]);
                let (w2x, w2y) = bf16x2_to_f32x2(w8[2]);
                let (w3x, w3y) = bf16x2_to_f32x2(w8[3]);
                let xf = xb.add(v << 3);
                let x0 = ldg_f4(xf);
                let x1 = ldg_f4(xf.add(4));
                dacc = fmaf(w0x, x0[0], dacc);
                dacc = fmaf(w0y, x0[1], dacc);
                dacc = fmaf(w1x, x0[2], dacc);
                dacc = fmaf(w1y, x0[3], dacc);
                dacc = fmaf(w2x, x1[0], dacc);
                dacc = fmaf(w2y, x1[1], dacc);
                dacc = fmaf(w3x, x1[2], dacc);
                dacc = fmaf(w3y, x1[3], dacc);
                iter += 1;
            }
            dacc = warp_sum(dacc);
            if lane == 0 {
                *s_dt.add(warp_i) = dacc;
            }
            sync_threads();

            let c = h * HEAD_DIM + tid;
            let cur = *proj.add(b * KV_DIM + c);
            let (h1, h2, h3) = kv_taps(proj, ring, b, p, c);
            let a0 = bf16_to_f32(*sw.add(c)) + *s_dt;
            let a1 = bf16_to_f32(*sw.add(KV_DIM + c)) + *s_dt.add(1);
            let a2 = bf16_to_f32(*sw.add(2 * KV_DIM + c)) + *s_dt.add(2);
            let a3 = bf16_to_f32(*sw.add(3 * KV_DIM + c)) + *s_dt.add(3);
            let acc = fmaf(h3, a3, fmaf(h2, a2, fmaf(h1, a1, cur * a0)));
            let val = (acc + cur) * INV_SQRT_9;
            let sq = warp_sum(mul_rn(val, val));
            if lane == 0 {
                *sm.add(warp_i) = sq;
            }
            sync_threads();
            if tid == 0 {
                *factor = if is_k {
                    fminf(div_rn(QK_CLIP, sqrtf(sum4(sm)) + 1e-6), 1.0)
                } else {
                    rsqrtf(div_rn(sum4(sm), HEAD_DIM as f32) + RMS_EPS)
                };
            }
            sync_threads();
            let cache = if is_k { k_cache } else { v_cache };
            let bf_slot = if fp8_shadow.is_null() { p % kv_cap } else { p % GLOBAL_KV_RING as i32 };
            *cache.add(bf_slot as usize * KV_DIM + h * HEAD_DIM + tid) = f32_to_bf16(val * *factor);
            if !fp8_shadow.is_null() {
                crate::attn::store_kv_fp8_shadow(fp8_shadow, kv_cap, (p % kv_cap) as usize, h, tid, is_k, val * *factor, sm);
            }
        }
    }
}

