//! Attention (exact mode, BF16 KV cache, B = 1).
//!
//! Ports of squid5090 @ 3ee093c `cuda/kernels_inkling.cu` and the shared
//! split-K engine in `kernels_common.cuh`:
//! - `mega_local_sconv_prep_store_kernel` (1827) as `mega_local_sconv_prep_store`;
//! - `mega_global_sconv_prep_store_kernel<0>` (1686) as `mega_global_sconv_prep_store_bf16kv`;
//! - `global_attn_bulk_splitk_kernel<0, 1>` (652, with `run_tc_bulk_splitk`) as
//!   `global_attn_bulk_splitk_bf16kv_b1`;
//! - `local_tail_attn_tc_kernel<0, 1>` (686) as `local_tail_attn_tc_bf16kv_b1`;
//! - `attn_merge_sink_gate_kernel` (1005) as `attn_merge_sink_gate`.

use crate::common::*;
use crate::{HIDDEN, fast_sigmoid, fmaf, float2half, pdl_trigger, pdl_wait};
use cuda_device::f16x2::{add_f16x2, fma_f16x2, mul_f16x2};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{SharedArray, device, kernel, launch_bounds, ptx_asm, warp};

// ---------------------------------------------------------------------------
// KV short conv + Q/R/K clip + V norm + KV store
// ---------------------------------------------------------------------------

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
pub unsafe fn mega_local_sconv_prep_store(
    qrgs: *mut f32,
    k_proj: *const f32,
    v_proj: *const f32,
    k_sconv_w: *const u16,
    v_sconv_w: *const u16,
    k_ring: *mut f32,
    v_ring: *mut f32,
    q_clip_scale: *const u16,
    rel_clip_scale: *const u16,
    k_cache: *mut u16,
    v_cache: *mut u16,
    kv_cap: i32,
    d_pos: *const i32,
) {
    static mut SM: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut FACTOR: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let factor = SharedArray::as_raw_mut_ptr(&raw mut FACTOR);
        let kind_id = blockIdx_x() as usize;
        let b = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        // Weights don't depend on the predecessor: load them before the PDL wait.
        let mut pre = [0u16; 4];
        if kind_id < 24 {
            pre[0] = *q_clip_scale.add(kind_id * HEAD_DIM + tid);
        } else if kind_id < 48 {
            if tid < D_REL {
                pre[0] = *rel_clip_scale.add((kind_id - 24) * D_REL + tid);
            }
        } else {
            let sw = if kind_id < 56 { k_sconv_w } else { v_sconv_w };
            let c = (if kind_id < 56 { kind_id - 48 } else { kind_id - 56 }) * HEAD_DIM + tid;
            pre = [*sw.add(c * 4), *sw.add(c * 4 + 1), *sw.add(c * 4 + 2), *sw.add(c * 4 + 3)];
        }
        // Trigger before the wait: the tail kernel can launch during the QKV projection and load its old
        // K/V rows (written by earlier steps) early; it still waits for this grid before reading new data.
        pdl_trigger();
        pdl_wait();
        let pos0 = *d_pos;
        let p = pos0 + b as i32;

        if kind_id < 24 {
            let h = kind_id;
            let qp = qrgs.add(b * QRGS_STRIDE + h * HEAD_DIM);
            let val = *qp.add(tid) * bf16_to_f32(pre[0]);
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
            let val = *rp.add(tid) * bf16_to_f32(pre[0]);
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
            *ring.add(((p & 15) as usize) * KV_DIM + c) = cur;
            let _ = sw;
            let w0 = bf16_to_f32(pre[0]);
            let w1 = bf16_to_f32(pre[1]);
            let w2 = bf16_to_f32(pre[2]);
            let w3 = bf16_to_f32(pre[3]);
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

/// FP8 shadow of a global KV cache (prototype, `INKLING_KV_FP8`), one buffer per visit: K codes
/// `[cap][KV_DIM]` u8, V codes the same, then K scales `[cap][8]` f16 and V scales `[cap][8]` f16
/// (squid5090's `--kv fp8` layout: E4M3 with one half scale per token and KV head).
pub(crate) const FP8_SHADOW_SCALES_PER_TOK: usize = NUM_KV_HEADS;

#[inline(always)]
pub(crate) unsafe fn fp8_shadow_parts(shadow: *const u16, kv_cap: i32) -> (*const u8, *const u8, *const u16, *const u16) {
    unsafe {
        let base = shadow as *const u8;
        let cap = kv_cap as usize;
        let ks = base.add(2 * cap * KV_DIM) as *const u16;
        (base, base.add(cap * KV_DIM), ks, ks.add(cap * FP8_SHADOW_SCALES_PER_TOK))
    }
}

/// `store_kv_head_128<1>` into the FP8 shadow: 128 threads hold the head's 128 values; one CTA-wide
/// amax (reusing `sm[4]`, free after the caller's last barrier), scale = max(amax / 448, 1e-8) kept as f16.
#[device]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn store_kv_fp8_shadow(shadow: *mut u16, kv_cap: i32, slot: usize, h: usize, tid: usize, is_k: bool, x: f32, sm: *mut f32) {
    unsafe {
        let wm = warp_max(fmaxf(x, -x));
        if (tid & 31) == 0 {
            *sm.add(tid >> 5) = wm;
        }
        sync_threads();
        let amax = fmaxf(fmaxf(*sm, *sm.add(1)), fmaxf(*sm.add(2), *sm.add(3)));
        let sc = fmaxf(div_rn(amax, 448.0), 1e-8);
        let (k8, v8, ks, vs) = fp8_shadow_parts(shadow, kv_cap);
        let (codes, scales) = if is_k { (k8, ks) } else { (v8, vs) };
        if tid == 0 {
            *(scales as *mut u16).add(slot * FP8_SHADOW_SCALES_PER_TOK + h) = float2half(sc);
        }
        *(codes as *mut u8).add(slot * KV_DIM + h * HEAD_DIM + tid) = crate::nvfp4::f32_to_e4m3(div_rn(x, sc));
    }
}

/// `mega_global_sconv_prep_store_kernel<0>`: global-layer Q/R clip with the
/// log-position temperature, dynamic K/V short conv, K clip, V norm and BF16
/// KV store. Grid (64, B), 128 threads. `k_scale`/`v_scale` are unused for BF16.
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mega_global_sconv_prep_store_bf16kv(
    qrgs: *mut f32,
    k_proj: *const f32,
    v_proj: *const f32,
    in_k: *const f32,
    in_v: *const f32,
    k_sconv_w: *const u16,
    v_sconv_w: *const u16,
    k_sconv_proj: *const u16,
    v_sconv_proj: *const u16,
    k_ring: *mut f32,
    v_ring: *mut f32,
    q_clip_scale: *const u16,
    rel_clip_scale: *const u16,
    k_cache: *mut u16,
    v_cache: *mut u16,
    fp8_shadow: *mut u16,
    _v_scale: *mut u16,
    kv_cap: i32,
    d_pos: *const i32,
) {
    static mut SM: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut S_DT: SharedArray<f32, 4> = SharedArray::UNINIT;
    static mut FACTOR: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let s_dt = SharedArray::as_raw_mut_ptr(&raw mut S_DT);
        let factor = SharedArray::as_raw_mut_ptr(&raw mut FACTOR);
        let kind_id = blockIdx_x() as usize;
        let b = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        // Weights don't depend on the predecessor: load them before the PDL wait.
        let mut pre = [0u16; 4];
        let mut wpre = [[0u32; 4]; 9];
        if kind_id < 24 {
            pre[0] = *q_clip_scale.add(kind_id * HEAD_DIM + tid);
        } else if kind_id < 48 {
            if tid < D_REL {
                pre[0] = *rel_clip_scale.add((kind_id - 24) * D_REL + tid);
            }
        } else {
            let is_k = kind_id < 56;
            let h = if is_k { kind_id - 48 } else { kind_id - 56 };
            let (sw, sproj) = if is_k { (k_sconv_w, k_sconv_proj) } else { (v_sconv_w, v_sconv_proj) };
            let wr = sproj.add((h * 4 + warp_i) * HIDDEN) as *const u8;
            let mut iter = 0;
            #[unroll]
            while iter < 9 {
                wpre[iter] = ldg_u4(wr.add((lane + iter * 32) * 16));
                iter += 1;
            }
            let c = h * HEAD_DIM + tid;
            pre = [*sw.add(c), *sw.add(KV_DIM + c), *sw.add(2 * KV_DIM + c), *sw.add(3 * KV_DIM + c)];
        }
        pdl_wait();
        pdl_trigger();
        let p = *d_pos + b as i32;
        let ratio = fmaxf(div_rn((p + 1) as f32, LOG_SCALE_FLOOR as f32), 1.0);
        let tau = fmaf(LOG_SCALE_ALPHA, logf(ratio), 1.0);

        if kind_id < 24 {
            let h = kind_id;
            let qp = qrgs.add(b * QRGS_STRIDE + h * HEAD_DIM);
            let val = *qp.add(tid) * bf16_to_f32(pre[0]);
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
            let val = *rp.add(tid) * bf16_to_f32(pre[0]);
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
                let _ = wr;
                let w8 = wpre[iter];
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
            *ring.add(((p & 15) as usize) * KV_DIM + c) = cur;
            let _ = sw;
            let a0 = bf16_to_f32(pre[0]) + *s_dt;
            let a1 = bf16_to_f32(pre[1]) + *s_dt.add(1);
            let a2 = bf16_to_f32(pre[2]) + *s_dt.add(2);
            let a3 = bf16_to_f32(pre[3]) + *s_dt.add(3);
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

// ---------------------------------------------------------------------------
// Tensor-core flash decoding
// ---------------------------------------------------------------------------

/// Two floats to a packed half2 (`__floats2half2_rn(lo, hi)`).
#[inline(always)]
fn pack_h2(lo: f32, hi: f32) -> u32 {
    pack_f16x2(lo, hi)
}

/// `ldmatrix_x4`.
#[inline(always)]
unsafe fn ldmatrix_x4(p: *const u16) -> [u32; 4] {
    let addr = unsafe { cvta_generic_to_shared_u32(p as *const u8) };
    let (r0, r1, r2, r3): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];",
            out("=r") r0,
            out("=r") r1,
            out("=r") r2,
            out("=r") r3,
            in("r") addr,
        );
    }
    [r0, r1, r2, r3]
}

/// `ldmatrix_x2_trans`.
#[inline(always)]
unsafe fn ldmatrix_x2_trans(p: *const u16) -> [u32; 2] {
    let addr = unsafe { cvta_generic_to_shared_u32(p as *const u8) };
    let (r0, r1): (u32, u32);
    unsafe {
        ptx_asm!(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];",
            out("=r") r0,
            out("=r") r1,
            in("r") addr,
        );
    }
    [r0, r1]
}

/// `load_kv_8elems_h2x4<0>`: 8 BF16 KV values at (slot, kvh, sub * 8) as 4 half2.
#[inline(always)]
unsafe fn load_kv8_bf16(cache: *const u16, slot: usize, kvh: usize, sub: usize) -> [u32; 4] {
    unsafe {
        let raw = ldg_u4(cache.add(slot * KV_DIM + kvh * HEAD_DIM + sub * 8) as *const u8);
        let mut h = [0u32; 4];
        let mut i = 0;
        while i < 4 {
            let (x, y) = bf16x2_to_f32x2(raw[i]);
            h[i] = pack_h2(x, y);
            i += 1;
        }
        h
    }
}

#[inline(always)]
unsafe fn st_u4(p: *mut u16, v: [u32; 4]) {
    unsafe { *(p as *mut [u32; 4]) = v }
}

/// 8 raw BF16 values (one 16-byte load) as 4 half2: the conversion half of `load_kv8_bf16`.
#[inline(always)]
fn bf16x8_to_h2x4(raw: [u32; 4]) -> [u32; 4] {
    let mut h = [0u32; 4];
    let mut i = 0;
    while i < 4 {
        let (x, y) = bf16x2_to_f32x2(raw[i]);
        h[i] = pack_h2(x, y);
        i += 1;
    }
    h
}

/// load_kv_8elems_h2x4<1>: 8 E4M3 codes (words 0-1) times the f16 scale (word 2), as 4 half2.
#[inline(always)]
pub(crate) fn e4m3x8_to_h2x4(raw: [u32; 4]) -> [u32; 4] {
    let s = raw[2] & 0xffff;
    let hsc = s | (s << 16);
    [
        mul_f16x2(crate::cvt_e4m3x2_to_half2((raw[0] & 0xffff) as u16), hsc),
        mul_f16x2(crate::cvt_e4m3x2_to_half2((raw[0] >> 16) as u16), hsc),
        mul_f16x2(crate::cvt_e4m3x2_to_half2((raw[1] & 0xffff) as u16), hsc),
        mul_f16x2(crate::cvt_e4m3x2_to_half2((raw[1] >> 16) as u16), hsc),
    ]
}

/// One 32-token bulk tile of K and V as raw BF16 loads held in registers (chunk j of a thread is
/// 16-byte piece `tid + j * threads` of the 512 per tile); rows past the tile are left zero.
#[derive(Clone, Copy)]
pub(crate) struct BulkTile {
    k: [[u32; 4]; 8],
    v: [[u32; 4]; 8],
}

/// Issues every load of one bulk tile at once (no dependent use), so they are all in flight together.
#[device]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn bulk_tile_load<const FP8: bool>(
    k_cache: *const u16,
    v_cache: *const u16,
    ks: *const u16,
    vs: *const u16,
    kv_cap: i32,
    kvh: usize,
    tid: usize,
    threads: usize,
    t0: i32,
    tile_len: usize,
) -> BulkTile {
    let mut r = BulkTile { k: [[0; 4]; 8], v: [[0; 4]; 8] };
    let nch = 512 / threads;
    let mut j = 0;
    #[unroll]
    while j < 8 {
        let idx = tid + j * threads;
        let t_loc = idx >> 4;
        if j < nch && t_loc < tile_len {
            let slot = ((t0 + t_loc as i32) % kv_cap) as usize;
            let off = slot * KV_DIM + kvh * HEAD_DIM + (idx & 15) * 8;
            unsafe {
                if FP8 {
                    // FP8 caches (u8 codes, KV_DIM bytes per slot): 8 E4M3 codes plus the token's f16 scale.
                    let kr = crate::ldg_u2((k_cache as *const u8).add(off));
                    let vr = crate::ldg_u2((v_cache as *const u8).add(off));
                    let si = slot * FP8_SHADOW_SCALES_PER_TOK + kvh;
                    r.k[j] = [kr[0], kr[1], ldg_u16(ks.add(si)) as u32, 0];
                    r.v[j] = [vr[0], vr[1], ldg_u16(vs.add(si)) as u32, 0];
                } else {
                    r.k[j] = ldg_u4(k_cache.add(off) as *const u8);
                    r.v[j] = ldg_u4(v_cache.add(off) as *const u8);
                }
            }
        }
        j += 1;
    }
    r
}

/// Converts a loaded bulk tile to half2 and stores it in the K/V tiles (zeros past the tile end).
#[device]
unsafe fn bulk_tile_store<const FP8: bool>(s_k: *mut u16, s_v: *mut u16, r: &BulkTile, tid: usize, threads: usize, tile_len: usize) {
    let nch = 512 / threads;
    let mut j = 0;
    #[unroll]
    while j < 8 {
        let idx = tid + j * threads;
        let t_loc = idx >> 4;
        if j < nch {
            let off = t_loc * KV_ROW + (idx & 15) * 8;
            let (kh, vh) = if t_loc >= tile_len {
                ([0; 4], [0; 4])
            } else if FP8 {
                (e4m3x8_to_h2x4(r.k[j]), e4m3x8_to_h2x4(r.v[j]))
            } else {
                (bf16x8_to_h2x4(r.k[j]), bf16x8_to_h2x4(r.v[j]))
            };
            unsafe {
                st_u4(s_k.add(off), kh);
                st_u4(s_v.add(off), vh);
            }
        }
        j += 1;
    }
}

/// The Q fragment for the m16n8k16 A operand: rows q0 and q1 (null = zero),
/// pre-scaled by 1/sqrt(HEAD_DIM) and rounded to f16.
#[inline(always)]
unsafe fn load_q_frag(qp0: *const f32, qp1: *const f32, tid4: usize) -> [[u32; 4]; 8] {
    unsafe {
        let mut q = [[0u32; 4]; 8];
        let mut kt = 0;
        while kt < 8 {
            let k_off = kt * 16;
            if !qp0.is_null() {
                q[kt][0] = pack_h2(
                    *qp0.add(k_off + tid4 * 2) * INV_SQRT_HEAD_DIM,
                    *qp0.add(k_off + tid4 * 2 + 1) * INV_SQRT_HEAD_DIM,
                );
                q[kt][2] = pack_h2(
                    *qp0.add(k_off + tid4 * 2 + 8) * INV_SQRT_HEAD_DIM,
                    *qp0.add(k_off + tid4 * 2 + 9) * INV_SQRT_HEAD_DIM,
                );
            }
            if !qp1.is_null() {
                q[kt][1] = pack_h2(
                    *qp1.add(k_off + tid4 * 2) * INV_SQRT_HEAD_DIM,
                    *qp1.add(k_off + tid4 * 2 + 1) * INV_SQRT_HEAD_DIM,
                );
                q[kt][3] = pack_h2(
                    *qp1.add(k_off + tid4 * 2 + 8) * INV_SQRT_HEAD_DIM,
                    *qp1.add(k_off + tid4 * 2 + 9) * INV_SQRT_HEAD_DIM,
                );
            }
            kt += 1;
        }
        q
    }
}

/// Q row pointer for query index `qi` of KV head `kvh` (null past Q_CNT).
#[inline(always)]
unsafe fn q_row(qrgs: *const f32, stride: usize, q_cnt: usize, kvh: usize, qi: usize) -> *const f32 {
    if qi < q_cnt {
        let h = kvh * GQA_RATIO + (qi % GQA_RATIO);
        let b = qi / GQA_RATIO;
        unsafe { qrgs.add(b * stride + h * HEAD_DIM) }
    } else {
        core::ptr::null()
    }
}

/// Row max / row sum across the 4 lanes that share a row.
#[inline(always)]
fn quad_max(v: f32) -> f32 {
    let v = fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 1));
    fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 2))
}

#[inline(always)]
fn quad_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 1);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 2);
    v
}

const KV_ROW: usize = HEAD_DIM + 8; // padded smem row (halves)
pub(crate) const BULK_SMEM: usize = 2 * 32 * KV_ROW * 2; // 17,408 B (s_k + s_v; the reduction aliases it)

/// `run_tc_bulk_splitk<0, 3B, STRIDE, THREADS>`: one split of the global
/// KV prefix, 32-token tiles, two 16-token slices per tile. Q_CNT = 3B query
/// rows; a second 16-row M tile and 128 threads when Q_CNT > 16 (B >= 6).
#[device]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_tc_bulk_splitk<const B: usize, const STRIDE: usize>(
    smem: *mut u8,
    qrgs: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    kv_cap: i32,
    kvh: usize,
    split_slot: usize,
    t_start: i32,
    t_end: i32,
    split_buf: *mut f32,
) {
    unsafe {
        let threads = if GQA_RATIO * B > 16 { 128 } else { 64 };
        let first = if t_start < t_end {
            bulk_tile_load::<false>(k_cache, v_cache, core::ptr::null(), core::ptr::null(), kv_cap, kvh, threadIdx_x() as usize, threads, t_start, core::cmp::min(32, t_end - t_start) as usize)
        } else {
            BulkTile { k: [[0; 4]; 8], v: [[0; 4]; 8] }
        };
        run_tc_bulk_splitk_pre::<B, STRIDE, false>(smem, qrgs, k_cache, v_cache, core::ptr::null(), core::ptr::null(), kv_cap, kvh, split_slot, t_start, t_end, split_buf, first);
    }
}

/// `run_tc_bulk_splitk` with the first tile's loads already issued (`first`), the next tile's loads
/// issued before each tile's math (register double buffering). Same arithmetic in the same order.
#[device]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_tc_bulk_splitk_pre<const B: usize, const STRIDE: usize, const FP8: bool>(
    smem: *mut u8,
    qrgs: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    ks: *const u16,
    vs: *const u16,
    kv_cap: i32,
    kvh: usize,
    split_slot: usize,
    t_start: i32,
    t_end: i32,
    split_buf: *mut f32,
    first: BulkTile,
) {
    unsafe {
        let q_cnt = GQA_RATIO * B;
        let nm = if q_cnt > 16 { 2 } else { 1 };
        let threads = if q_cnt > 16 { 128 } else { 64 };
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let grp = lane >> 2;
        let tid4 = lane & 3;
        let slice = warp_i & 1;
        let m_tile = warp_i >> 1;

        let s_m_out = split_buf.add(SPLIT_M_OFF);
        let s_d_out = split_buf.add(SPLIT_D_OFF);
        let s_o_out = split_buf.add(SPLIT_O_OFF);

        if t_start >= t_end {
            if tid < q_cnt {
                let gqa = tid % GQA_RATIO;
                let b = tid / GQA_RATIO;
                let h = kvh * GQA_RATIO + gqa;
                let idx = (h * 8 + b) * KV_TOTAL_SPLITS + split_slot;
                *s_m_out.add(idx) = -1e30;
                *s_d_out.add(idx) = 0.0;
            }
            return;
        }

        let q0 = m_tile * 16 + grp;
        let q1 = m_tile * 16 + grp + 8;
        let q_reg = load_q_frag(q_row(qrgs, STRIDE, q_cnt, kvh, q0), q_row(qrgs, STRIDE, q_cnt, kvh, q1), tid4);

        let mut m0 = -1e30f32;
        let mut m1 = -1e30f32;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut o = [[0.0f32; 4]; 16];

        let s_k = smem as *mut u16;
        let s_v = smem.add(32 * KV_ROW * 2) as *mut u16;

        let c0 = tid4 * 2;
        let c1 = tid4 * 2 + 1;
        let c4 = tid4 * 2 + 8;
        let c5 = tid4 * 2 + 9;
        let t_base = slice * 16;
        let k_ld_row = t_base + ((lane & 16) >> 1) + (lane & 7);
        let k_ld_col = lane & 8;
        let v_ld_row = t_base + (lane & 15);

        let mut t0 = t_start;
        let mut cur = first;
        // FP8 tiles are half the bytes, so keep two tiles in flight (registers allow it) to cover latency.
        let mut nxt = if FP8 && t_start + 32 < t_end {
            bulk_tile_load::<FP8>(k_cache, v_cache, ks, vs, kv_cap, kvh, tid, threads, t_start + 32, core::cmp::min(32, t_end - t_start - 32) as usize)
        } else {
            first
        };
        while t0 < t_end {
            let tile_len = core::cmp::min(32, t_end - t0) as usize;
            bulk_tile_store::<FP8>(s_k, s_v, &cur, tid, threads, tile_len);
            sync_threads();
            // The next tile's loads fly during this tile's math (FP8: the next two tiles).
            if FP8 {
                cur = nxt;
                let t_far = t0 + 64;
                if t_far < t_end {
                    nxt = bulk_tile_load::<FP8>(k_cache, v_cache, ks, vs, kv_cap, kvh, tid, threads, t_far, core::cmp::min(32, t_end - t_far) as usize);
                }
            } else {
                let t_next = t0 + 32;
                if t_next < t_end {
                    cur = bulk_tile_load::<FP8>(k_cache, v_cache, ks, vs, kv_cap, kvh, tid, threads, t_next, core::cmp::min(32, t_end - t_next) as usize);
                }
            }

            let w_len = core::cmp::min(16, tile_len.saturating_sub(t_base));
            if w_len > 0 {
                let mut sa = [0.0f32; 4];
                let mut sb = [0.0f32; 4];
                let mut kt = 0;
                #[unroll]
                while kt < 8 {
                    let kb = ldmatrix_x4(s_k.add(k_ld_row * KV_ROW + kt * 16 + k_ld_col));
                    sa = mma_m16n8k16_f32_f16(sa, q_reg[kt], [kb[0], kb[1]]);
                    sb = mma_m16n8k16_f32_f16(sb, q_reg[kt], [kb[2], kb[3]]);
                    kt += 1;
                }
                let (mut s0, mut s1, mut s2, mut s3) = (sa[0], sa[1], sa[2], sa[3]);
                let (mut s4, mut s5, mut s6, mut s7) = (sb[0], sb[1], sb[2], sb[3]);

                if w_len < 16 {
                    if c0 >= w_len {
                        s0 = -1e30;
                        s2 = -1e30;
                    }
                    if c1 >= w_len {
                        s1 = -1e30;
                        s3 = -1e30;
                    }
                    if c4 >= w_len {
                        s4 = -1e30;
                        s6 = -1e30;
                    }
                    if c5 >= w_len {
                        s5 = -1e30;
                        s7 = -1e30;
                    }
                }

                let rmax0 = quad_max(fmaxf(fmaxf(s0, s1), fmaxf(s4, s5)));
                let rmax1 = quad_max(fmaxf(fmaxf(s2, s3), fmaxf(s6, s7)));
                let m0_new = fmaxf(m0, rmax0);
                let m1_new = fmaxf(m1, rmax1);
                let alpha0 = if m0 > -1e29 { fast_expf(m0 - m0_new) } else { 0.0 };
                let alpha1 = if m1 > -1e29 { fast_expf(m1 - m1_new) } else { 0.0 };
                let mut nt = 0;
                #[unroll]
                while nt < 16 {
                    o[nt][0] *= alpha0;
                    o[nt][1] *= alpha0;
                    o[nt][2] *= alpha1;
                    o[nt][3] *= alpha1;
                    nt += 1;
                }

                let ps0 = if c0 < w_len { fast_expf(s0 - m0_new) } else { 0.0 };
                let ps1 = if c1 < w_len { fast_expf(s1 - m0_new) } else { 0.0 };
                let ps4 = if c4 < w_len { fast_expf(s4 - m0_new) } else { 0.0 };
                let ps5 = if c5 < w_len { fast_expf(s5 - m0_new) } else { 0.0 };
                let ps2 = if c0 < w_len { fast_expf(s2 - m1_new) } else { 0.0 };
                let ps3 = if c1 < w_len { fast_expf(s3 - m1_new) } else { 0.0 };
                let ps6 = if c4 < w_len { fast_expf(s6 - m1_new) } else { 0.0 };
                let ps7 = if c5 < w_len { fast_expf(s7 - m1_new) } else { 0.0 };

                let rsum0 = quad_sum(ps0 + ps1 + ps4 + ps5);
                let rsum1 = quad_sum(ps2 + ps3 + ps6 + ps7);
                d0 = fmaf(d0, alpha0, rsum0);
                m0 = m0_new;
                d1 = fmaf(d1, alpha1, rsum1);
                m1 = m1_new;

                let pa = [pack_h2(ps0, ps1), pack_h2(ps2, ps3), pack_h2(ps4, ps5), pack_h2(ps6, ps7)];
                let mut nt = 0;
                #[unroll]
                while nt < 16 {
                    let vb = ldmatrix_x2_trans(s_v.add(v_ld_row * KV_ROW + nt * 8));
                    o[nt] = mma_m16n8k16_f32_f16(o[nt], pa, vb);
                    nt += 1;
                }
            }
            sync_threads();
            t0 += 32;
        }

        // Reduce the 2 token slices per M tile (all 16 queries of a tile at once).
        let red_m = smem as *mut f32; // [16][2]
        let red_d = red_m.add(32); // [16][2]
        let red_o = red_m.add(64); // [2][16][128]
        let mut mt = 0;
        while mt < nm {
            if m_tile == mt {
                if tid4 == 0 {
                    *red_m.add(grp * 2 + slice) = m0;
                    *red_d.add(grp * 2 + slice) = d0;
                    *red_m.add((grp + 8) * 2 + slice) = m1;
                    *red_d.add((grp + 8) * 2 + slice) = d1;
                }
                let mut nt = 0;
                #[unroll]
                while nt < 16 {
                    *red_o.add((slice * 16 + grp) * HEAD_DIM + nt * 8 + c0) = o[nt][0];
                    *red_o.add((slice * 16 + grp) * HEAD_DIM + nt * 8 + c1) = o[nt][1];
                    *red_o.add((slice * 16 + grp + 8) * HEAD_DIM + nt * 8 + c0) = o[nt][2];
                    *red_o.add((slice * 16 + grp + 8) * HEAD_DIM + nt * 8 + c1) = o[nt][3];
                    nt += 1;
                }
            }
            sync_threads();

            let mut i = tid;
            while i < 16 * HEAD_DIM {
                let qi = i >> 7;
                let dim = i & 127;
                let q_glob = mt * 16 + qi;
                if q_glob < q_cnt {
                    let gqa = q_glob % GQA_RATIO;
                    let b = q_glob / GQA_RATIO;
                    let h = kvh * GQA_RATIO + gqa;
                    let idx = (h * 8 + b) * KV_TOTAL_SPLITS + split_slot;
                    let gm = fmaxf(*red_m.add(qi * 2), *red_m.add(qi * 2 + 1));
                    let (mut gd, mut go) = (0.0f32, 0.0f32);
                    let mut s = 0;
                    while s < 2 {
                        let dv = *red_d.add(qi * 2 + s);
                        if dv > 0.0 {
                            let sc = fast_expf(*red_m.add(qi * 2 + s) - gm);
                            gd = fmaf(dv, sc, gd);
                            go = fmaf(*red_o.add((s * 16 + qi) * HEAD_DIM + dim), sc, go);
                        }
                        s += 1;
                    }
                    if dim == 0 {
                        *s_m_out.add(idx) = gm;
                        *s_d_out.add(idx) = gd;
                    }
                    *s_o_out.add(idx * HEAD_DIM + dim) = go;
                }
                i += threads;
            }
            if mt + 1 < nm {
                sync_threads();
            }
            mt += 1;
        }
    }
}

/// `global_attn_bulk_splitk_kernel<0, B>` body: attention over the global
/// KV prefix older than the 1,024-token relative window, split 64 ways.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn global_attn_bulk_body<const B: usize, const FP8: bool>(
    smem: *mut u8,
    qrgs: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    ks: *const u16,
    vs: *const u16,
    kv_cap: i32,
    d_pos: *const i32,
    split_buf: *mut f32,
) {
    unsafe {
        // d_pos is fixed for the whole step, and every bulk row is older than this step's KV rows, so the
        // first tile loads before the PDL wait (unless the ring has wrapped onto this step's slots).
        let pos0 = *d_pos;
        let tail_start = pos0 - REL_EXTENT as i32 + 1;
        if tail_start <= 0 {
            pdl_wait();
            pdl_trigger();
            return;
        }
        let split = blockIdx_y() as i32;
        let chunk = (tail_start + KV_SPLIT_K as i32 - 1) / KV_SPLIT_K as i32;
        let t_start = split * chunk;
        let t_end = core::cmp::min(tail_start, t_start + chunk);
        let threads = if GQA_RATIO * B > 16 { 128 } else { 64 };
        let early = t_start < t_end && pos0 + MAX_VERIFY_B as i32 <= kv_cap;
        let first = if early {
            bulk_tile_load::<FP8>(k_cache, v_cache, ks, vs, kv_cap, blockIdx_x() as usize, threadIdx_x() as usize, threads, t_start, core::cmp::min(32, t_end - t_start) as usize)
        } else {
            BulkTile { k: [[0; 4]; 8], v: [[0; 4]; 8] }
        };
        pdl_wait();
        pdl_trigger();
        let first = if t_start < t_end && !early {
            bulk_tile_load::<FP8>(k_cache, v_cache, ks, vs, kv_cap, blockIdx_x() as usize, threadIdx_x() as usize, threads, t_start, core::cmp::min(32, t_end - t_start) as usize)
        } else {
            first
        };
        run_tc_bulk_splitk_pre::<B, QRGS_STRIDE, FP8>(
            smem,
            qrgs,
            k_cache,
            v_cache,
            ks,
            vs,
            kv_cap,
            blockIdx_x() as usize,
            split as usize,
            t_start,
            t_end,
            split_buf,
            first,
        );
    }
}

macro_rules! bulk_kernel {
    ($name:ident, $b:expr, $threads:expr, $doc:literal) => {
        bulk_kernel!($name, $b, $threads, false, $doc);
    };
    ($name:ident, $b:expr, $threads:expr, $fp8:expr, $doc:literal) => {
        #[doc = $doc]
        #[kernel]
        #[launch_bounds($threads)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            qrgs: *const f32,
            k_cache: *const u16,
            v_cache: *const u16,
            k_scale: *const u16,
            v_scale: *const u16,
            kv_cap: i32,
            d_pos: *const i32,
            split_buf: *mut f32,
        ) {
            static mut SMEM: SharedArray<u8, { BULK_SMEM }, 16> = SharedArray::UNINIT;
            unsafe {
                global_attn_bulk_body::<$b, $fp8>(
                    SharedArray::as_raw_mut_ptr(&raw mut SMEM),
                    qrgs,
                    k_cache,
                    v_cache,
                    k_scale,
                    v_scale,
                    kv_cap,
                    d_pos,
                    split_buf,
                )
            }
        }
    };
}

bulk_kernel!(global_attn_bulk_splitk_bf16kv_b1, 1, 64, "`global_attn_bulk_splitk_kernel<0, 1>`: grid (8, 64), 64 threads; the scale pointers are unused for BF16.");
bulk_kernel!(global_attn_bulk_splitk_bf16kv_b2, 2, 64, "`global_attn_bulk_splitk_kernel<0, 2>`: grid (8, 64), 64 threads.");
bulk_kernel!(global_attn_bulk_splitk_bf16kv_b4, 4, 64, "`global_attn_bulk_splitk_kernel<0, 4>`: grid (8, 64), 64 threads.");
bulk_kernel!(global_attn_bulk_splitk_bf16kv_b8, 8, 128, "`global_attn_bulk_splitk_kernel<0, 8>`: grid (8, 64), 128 threads.");
bulk_kernel!(global_attn_bulk_splitk_fp8kv_b1, 1, 64, true, "`global_attn_bulk_splitk_kernel<1, 1>`: FP8 E4M3 KV (u8 codes) with f16 per-token, per-head scales.");
bulk_kernel!(global_attn_bulk_splitk_fp8kv_b2, 2, 64, true, "`global_attn_bulk_splitk_kernel<1, 2>`.");
bulk_kernel!(global_attn_bulk_splitk_fp8kv_b4, 4, 64, true, "`global_attn_bulk_splitk_kernel<1, 4>`.");
bulk_kernel!(global_attn_bulk_splitk_fp8kv_b8, 8, 128, true, "`global_attn_bulk_splitk_kernel<1, 8>`.");

const LOCAL_KV_SMEM: usize = 64 * KV_ROW * 2; // 17,408 B per K or V tile (the reduction aliases both)

/// `local_tail_attn_tc_kernel<0, B>` body: the most recent window (local
/// layers) or the 1,024-token relative tail (global layers), one 64-token
/// tile per CTA (K and V in separate buffers, rows older than pos0 loaded
/// before the PDL wait), with exact relative bias and causal/window masks. Q_CNT = 3B
/// rows; a second M tile and 256 threads when Q_CNT > 16.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn local_tail_body<const B: usize>(
    s_r: *mut u16,
    s_rp: *mut u16,
    smem: *mut u8,
    s_v: *mut u16,
    qrgs: *const f32,
    rel_proj: *const u16,
    k_cache: *const u16,
    v_cache: *const u16,
    kv_cap: i32,
    window: i32,
    d_pos: *const i32,
    split_buf: *mut f32,
) {
    unsafe {
        let s_kv = smem as *mut u16;
        let q_cnt = GQA_RATIO * B;
        let nm = if q_cnt > 16 { 2 } else { 1 };
        let threads = if q_cnt > 16 { 256 } else { 128 };

        let kvh = blockIdx_x() as usize;
        let lsplit = blockIdx_y() as i32;
        let split_slot = if window == 0 { KV_SPLIT_K + lsplit as usize } else { lsplit as usize };
        let tid = threadIdx_x() as usize;

        let pos0 = *d_pos;
        let win_start = if window > 0 {
            core::cmp::max(0, pos0 + 1 - window)
        } else {
            core::cmp::max(0, pos0 - REL_EXTENT as i32 + 1)
        };
        let win_end = pos0 + B as i32;
        let t0 = win_start + lsplit * 64;
        let tile_len = core::cmp::min(64, core::cmp::max(0, win_end - t0));

        let s_m_out = split_buf.add(SPLIT_M_OFF);
        let s_d_out = split_buf.add(SPLIT_D_OFF);
        let s_o_out = split_buf.add(SPLIT_O_OFF);

        if tile_len <= 0 {
            if tid < q_cnt {
                let gqa = tid % GQA_RATIO;
                let b = tid / GQA_RATIO;
                let h = kvh * GQA_RATIO + gqa;
                let idx = (h * 8 + b) * KV_TOTAL_SPLITS + split_slot;
                *s_m_out.add(idx) = -1e30;
                *s_d_out.add(idx) = 0.0;
            }
            pdl_wait();
            return;
        }
        let tile_len = tile_len as usize;
        // Rows below pos0 were stored by earlier steps (the ring holds window + B_MAX - 1 rows), so they
        // load before the PDL wait; rows from pos0 on are this step's and load after it.
        let new_lo = core::cmp::min(tile_len, core::cmp::max(0, pos0 - t0) as usize);

        let warp_i = tid >> 5;
        let lane = tid & 31;
        let grp = lane >> 2;
        let tid4 = lane & 3;
        let slice = warp_i & 3;
        let m_tile = warp_i >> 2;

        // 2. The rel_proj columns this tile can reach, transposed.
        let d_min = core::cmp::max(0, pos0 - (t0 + tile_len as i32 - 1));
        let d_max = core::cmp::min(REL_EXTENT as i32 - 1, pos0 + B as i32 - 1 - t0);
        let d_span = if d_max >= d_min { (d_max - d_min + 1) as usize } else { 0 };
        let mut i = tid;
        while i < d_span * D_REL {
            let di = i >> 4;
            let k = i & 15;
            let dist = d_min as usize + di;
            *s_rp.add(di * D_REL + k) = float2half(bf16_to_f32(ldg_u16(rel_proj.add(k * REL_EXTENT + dist))));
            i += threads;
        }

        // 3. Old K and V rows (and zero padding) before the wait, all loads in flight at once.
        let mut kr = [[0u32; 4]; 8];
        let mut vr = [[0u32; 4]; 8];
        let mut j = 0;
        #[unroll]
        while j < 8 {
            let idx = tid + j * threads;
            let t_loc = idx >> 4;
            if idx < 1024 && t_loc < new_lo {
                let slot = ((t0 + t_loc as i32) % kv_cap) as usize;
                kr[j] = load_kv8_bf16(k_cache, slot, kvh, idx & 15);
                vr[j] = load_kv8_bf16(v_cache, slot, kvh, idx & 15);
            }
            j += 1;
        }
        let mut j = 0;
        #[unroll]
        while j < 8 {
            let idx = tid + j * threads;
            let t_loc = idx >> 4;
            let off = t_loc * KV_ROW + (idx & 15) * 8;
            if idx < 1024 && (t_loc < new_lo || t_loc >= tile_len) {
                st_u4(s_kv.add(off), kr[j]);
                st_u4(s_v.add(off), vr[j]);
            }
            j += 1;
        }

        pdl_wait();
        pdl_trigger();

        // 1b. R rows into shared memory.
        let mut i = tid;
        while i < q_cnt * D_REL {
            let qi = i >> 4;
            let k = i & 15;
            let h = kvh * GQA_RATIO + (qi % GQA_RATIO);
            let b = qi / GQA_RATIO;
            *s_r.add(qi * D_REL + k) = float2half(*qrgs.add(b * QRGS_STRIDE + Q_DIM + h * D_REL + k));
            i += threads;
        }

        // 3b. This step's K and V rows.
        let mut idx = new_lo * 16 + tid;
        while idx < tile_len * 16 {
            let t_loc = idx >> 4;
            let sub = idx & 15;
            let off = t_loc * KV_ROW + sub * 8;
            let slot = ((t0 + t_loc as i32) % kv_cap) as usize;
            st_u4(s_kv.add(off), load_kv8_bf16(k_cache, slot, kvh, sub));
            st_u4(s_v.add(off), load_kv8_bf16(v_cache, slot, kvh, sub));
            idx += threads;
        }

        // 4. Q fragments.
        let q0 = m_tile * 16 + grp;
        let q1 = m_tile * 16 + grp + 8;
        let pos_q0 = if q0 < q_cnt { pos0 + (q0 / GQA_RATIO) as i32 } else { pos0 };
        let pos_q1 = if q1 < q_cnt { pos0 + (q1 / GQA_RATIO) as i32 } else { pos0 };
        let q_reg = load_q_frag(q_row(qrgs, QRGS_STRIDE, q_cnt, kvh, q0), q_row(qrgs, QRGS_STRIDE, q_cnt, kvh, q1), tid4);
        sync_threads();

        let c0 = tid4 * 2;
        let c1 = tid4 * 2 + 1;
        let c4 = tid4 * 2 + 8;
        let c5 = tid4 * 2 + 9;
        let t_base = slice * 16;
        let w_len = core::cmp::min(16, tile_len.saturating_sub(t_base));

        let mut m0 = -1e30f32;
        let mut m1 = -1e30f32;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut o = [[0.0f32; 4]; 16];
        let mut pa = [0u32; 4];

        if w_len > 0 {
            // 5. QK^T.
            let mut sa = [0.0f32; 4];
            let mut sb = [0.0f32; 4];
            let k_ld_row = t_base + ((lane & 16) >> 1) + (lane & 7);
            let k_ld_col = lane & 8;
            let mut kt = 0;
            #[unroll]
            while kt < 8 {
                let kb = ldmatrix_x4(s_kv.add(k_ld_row * KV_ROW + kt * 16 + k_ld_col));
                sa = mma_m16n8k16_f32_f16(sa, q_reg[kt], [kb[0], kb[1]]);
                sb = mma_m16n8k16_f32_f16(sb, q_reg[kt], [kb[2], kb[3]]);
                kt += 1;
            }

            // 6. Relative bias + causal / window mask.
            let apply_rel = |s: f32, col: usize, qi: usize, pos_q: i32| -> f32 {
                if qi >= q_cnt || col >= w_len {
                    return -1e30;
                }
                let t = t0 + (t_base + col) as i32;
                if t > pos_q || (window > 0 && t < pos_q + 1 - window) {
                    return -1e30;
                }
                let dist = pos_q - t;
                if dist < REL_EXTENT as i32 {
                    let rh = s_r.add(qi * D_REL) as *const u32;
                    let ph = s_rp.add((dist - d_min) as usize * D_REL) as *const u32;
                    let mut a0 = mul_f16x2(*rh, *ph);
                    let mut a1 = mul_f16x2(*rh.add(1), *ph.add(1));
                    a0 = fma_f16x2(*rh.add(2), *ph.add(2), a0);
                    a1 = fma_f16x2(*rh.add(3), *ph.add(3), a1);
                    a0 = fma_f16x2(*rh.add(4), *ph.add(4), a0);
                    a1 = fma_f16x2(*rh.add(5), *ph.add(5), a1);
                    a0 = fma_f16x2(*rh.add(6), *ph.add(6), a0);
                    a1 = fma_f16x2(*rh.add(7), *ph.add(7), a1);
                    let f = add_f16x2(a0, a1);
                    s + (crate::h2f_lo(f) + crate::h2f_hi(f))
                } else {
                    s
                }
            };
            let s0 = apply_rel(sa[0], c0, q0, pos_q0);
            let s1 = apply_rel(sa[1], c1, q0, pos_q0);
            let s4 = apply_rel(sb[0], c4, q0, pos_q0);
            let s5 = apply_rel(sb[1], c5, q0, pos_q0);
            let s2 = apply_rel(sa[2], c0, q1, pos_q1);
            let s3 = apply_rel(sa[3], c1, q1, pos_q1);
            let s6 = apply_rel(sb[2], c4, q1, pos_q1);
            let s7 = apply_rel(sb[3], c5, q1, pos_q1);

            // 7. Single-tile softmax.
            let rmax0 = quad_max(fmaxf(fmaxf(s0, s1), fmaxf(s4, s5)));
            let rmax1 = quad_max(fmaxf(fmaxf(s2, s3), fmaxf(s6, s7)));
            let e = |s: f32, m: f32| if s > -1e29 { fast_expf(s - m) } else { 0.0 };
            let ps0 = e(s0, rmax0);
            let ps1 = e(s1, rmax0);
            let ps4 = e(s4, rmax0);
            let ps5 = e(s5, rmax0);
            let ps2 = e(s2, rmax1);
            let ps3 = e(s3, rmax1);
            let ps6 = e(s6, rmax1);
            let ps7 = e(s7, rmax1);
            m0 = rmax0;
            d0 = quad_sum(ps0 + ps1 + ps4 + ps5);
            m1 = rmax1;
            d1 = quad_sum(ps2 + ps3 + ps6 + ps7);
            pa = [pack_h2(ps0, ps1), pack_h2(ps2, ps3), pack_h2(ps4, ps5), pack_h2(ps6, ps7)];
        }

        // 9. P @ V.
        if w_len > 0 {
            let v_ld_row = t_base + (lane & 15);
            let mut nt = 0;
            #[unroll]
            while nt < 16 {
                let vb = ldmatrix_x2_trans(s_v.add(v_ld_row * KV_ROW + nt * 8));
                o[nt] = mma_m16n8k16_f32_f16(o[nt], pa, vb);
                nt += 1;
            }
        }
        sync_threads();

        // 10. Reduce the 4 token slices into split_buf, one M tile (16 queries) per pass; the
        // [4][16][128] partials fill the K and V buffers.
        let red_m = smem as *mut f32; // [16][4]
        let red_d = red_m.add(64); // [16][4]
        let red_o = red_m.add(128); // [4][16][128]
        let mut mt = 0;
        while mt < nm {
            if m_tile == mt {
                if tid4 == 0 {
                    *red_m.add(grp * 4 + slice) = m0;
                    *red_d.add(grp * 4 + slice) = d0;
                    *red_m.add((grp + 8) * 4 + slice) = m1;
                    *red_d.add((grp + 8) * 4 + slice) = d1;
                }
                let mut nt = 0;
                #[unroll]
                while nt < 16 {
                    *red_o.add((slice * 16 + grp) * HEAD_DIM + nt * 8 + c0) = o[nt][0];
                    *red_o.add((slice * 16 + grp) * HEAD_DIM + nt * 8 + c1) = o[nt][1];
                    *red_o.add((slice * 16 + grp + 8) * HEAD_DIM + nt * 8 + c0) = o[nt][2];
                    *red_o.add((slice * 16 + grp + 8) * HEAD_DIM + nt * 8 + c1) = o[nt][3];
                    nt += 1;
                }
            }
            sync_threads();

            let rows = core::cmp::min(16, q_cnt - mt * 16);
            let mut i = tid;
            while i < rows * HEAD_DIM {
                let qi = i >> 7;
                let dim = i & 127;
                let q_glob = mt * 16 + qi;
                let gqa = q_glob % GQA_RATIO;
                let b = q_glob / GQA_RATIO;
                let h = kvh * GQA_RATIO + gqa;
                let idx = (h * 8 + b) * KV_TOTAL_SPLITS + split_slot;
                let gm = fmaxf(
                    fmaxf(*red_m.add(qi * 4), *red_m.add(qi * 4 + 1)),
                    fmaxf(*red_m.add(qi * 4 + 2), *red_m.add(qi * 4 + 3)),
                );
                let (mut gd, mut go) = (0.0f32, 0.0f32);
                let mut s = 0;
                #[unroll]
                while s < 4 {
                    let dv = *red_d.add(qi * 4 + s);
                    if dv > 0.0 {
                        let sc = fast_expf(*red_m.add(qi * 4 + s) - gm);
                        gd = fmaf(dv, sc, gd);
                        go = fmaf(*red_o.add((s * 16 + qi) * HEAD_DIM + dim), sc, go);
                    }
                    s += 1;
                }
                if dim == 0 {
                    *s_m_out.add(idx) = gm;
                    *s_d_out.add(idx) = gd;
                }
                *s_o_out.add(idx * HEAD_DIM + dim) = go;
                i += threads;
            }
            if mt + 1 < nm {
                sync_threads();
            }
            mt += 1;
        }
    }
}

macro_rules! local_tail_kernel {
    ($name:ident, $b:expr, $threads:expr, $doc:literal) => {
        #[doc = $doc]
        #[kernel]
        #[launch_bounds($threads)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            qrgs: *const f32,
            rel_proj: *const u16,
            k_cache: *const u16,
            v_cache: *const u16,
            _k_scale: *const u16,
            _v_scale: *const u16,
            kv_cap: i32,
            window: i32,
            d_pos: *const i32,
            split_buf: *mut f32,
        ) {
            static mut S_R: SharedArray<u16, { 32 * D_REL }, 16> = SharedArray::UNINIT;
            static mut S_RP: SharedArray<u16, { 72 * D_REL }, 16> = SharedArray::UNINIT;
            static mut SMEM: SharedArray<u8, { 2 * LOCAL_KV_SMEM }, 16> = SharedArray::UNINIT;
            unsafe {
                let smem = SharedArray::as_raw_mut_ptr(&raw mut SMEM);
                local_tail_body::<$b>(
                    SharedArray::as_raw_mut_ptr(&raw mut S_R),
                    SharedArray::as_raw_mut_ptr(&raw mut S_RP),
                    smem,
                    smem.add(LOCAL_KV_SMEM) as *mut u16,
                    qrgs,
                    rel_proj,
                    k_cache,
                    v_cache,
                    kv_cap,
                    window,
                    d_pos,
                    split_buf,
                )
            }
        }
    };
}

local_tail_kernel!(local_tail_attn_tc_bf16kv_b1, 1, 128, "`local_tail_attn_tc_kernel<0, 1>`: grid (8, 17), 128 threads.");
local_tail_kernel!(local_tail_attn_tc_bf16kv_b2, 2, 128, "`local_tail_attn_tc_kernel<0, 2>`: grid (8, 17), 128 threads.");
local_tail_kernel!(local_tail_attn_tc_bf16kv_b4, 4, 128, "`local_tail_attn_tc_kernel<0, 4>`: grid (8, 17), 128 threads.");
local_tail_kernel!(local_tail_attn_tc_bf16kv_b8, 8, 256, "`local_tail_attn_tc_kernel<0, 8>`: grid (8, 17), 256 threads.");

/// `__ldg(const __nv_bfloat16*)`.
#[inline(always)]
unsafe fn ldg_u16(p: *const u16) -> u16 {
    let v: u16;
    unsafe {
        ptx_asm!("ld.global.nc.u16 %0, [%1];", out("=h") v, in("l") p, options(register_only));
    }
    v
}

/// `attn_merge_sink_gate_kernel`: merges the split partials with the
/// attention sink, then applies the sigmoid output gate. Grid (24, B),
/// 128 threads.
#[kernel]
#[launch_bounds(128)]
pub unsafe fn attn_merge_sink_gate(
    qrgs: *const f32,
    split_buf: *const f32,
    out_gated: *mut f32,
    out_gated_h: *mut u16,
    window: i32,
    d_pos: *const i32,
) {
    unsafe {
        pdl_wait();
        pdl_trigger();
        let h = blockIdx_x() as usize;
        let b = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;

        let qrgs_b = qrgs.add(b * QRGS_STRIDE);
        let gate_p = qrgs_b.add(Q_DIM + R_DIM + h * HEAD_DIM);
        let sink_logit = *qrgs_b.add(Q_DIM + R_DIM + Q_DIM + h);

        let base_hb = (h * 8 + b) * KV_TOTAL_SPLITS;
        let s_m_in = split_buf.add(SPLIT_M_OFF + base_hb);
        let s_d_in = split_buf.add(SPLIT_D_OFF + base_hb);
        let s_o_in = split_buf.add(SPLIT_O_OFF);

        let (s_begin, s_end) = if window > 0 {
            (0, KV_LOCAL_SPLITS)
        } else {
            let tail_start = *d_pos - REL_EXTENT as i32 + 1;
            (if tail_start > 0 { 0 } else { KV_SPLIT_K }, KV_TOTAL_SPLITS)
        };

        // KV_LOCAL_SPLITS (17) splits per batch, all loads of a batch in flight together: the local
        // window and the short-context global tail take one batch. Same order of operations as before.
        const MB: usize = KV_LOCAL_SPLITS;
        let mut g_m = sink_logit;
        let one_batch = s_end - s_begin <= MB;
        let mut sd = [0.0f32; MB];
        let mut sm = [0.0f32; MB];
        let mut so = [0.0f32; MB];
        if one_batch {
            // One round of loads: m, d and o of every split together.
            let mut j = 0;
            #[unroll]
            while j < MB {
                if s_begin + j < s_end {
                    sd[j] = *s_d_in.add(s_begin + j);
                    sm[j] = *s_m_in.add(s_begin + j);
                    so[j] = *s_o_in.add((base_hb + s_begin + j) * HEAD_DIM + tid);
                }
                j += 1;
            }
            let mut j = 0;
            #[unroll]
            while j < MB {
                if s_begin + j < s_end {
                    g_m = fmaxf(g_m, sm[j]);
                }
                j += 1;
            }
        }
        let mut s = if one_batch { s_end } else { s_begin };
        while s < s_end {
            let mut mv = [-1e30f32; MB];
            let mut j = 0;
            #[unroll]
            while j < MB {
                if s + j < s_end {
                    mv[j] = *s_m_in.add(s + j);
                }
                j += 1;
            }
            let mut j = 0;
            #[unroll]
            while j < MB {
                if s + j < s_end {
                    g_m = fmaxf(g_m, mv[j]);
                }
                j += 1;
            }
            s += MB;
        }
        let mut g_d = fast_expf(sink_logit - g_m);
        let mut g_o = 0.0f32;
        // Empty splits' partials are read and discarded.
        let mut s = s_begin;
        while s < s_end {
            let mut j = 0;
            #[unroll]
            while j < MB {
                if !one_batch && s + j < s_end {
                    sd[j] = *s_d_in.add(s + j);
                    sm[j] = *s_m_in.add(s + j);
                    so[j] = *s_o_in.add((base_hb + s + j) * HEAD_DIM + tid);
                }
                j += 1;
            }
            let mut j = 0;
            #[unroll]
            while j < MB {
                if s + j < s_end && sd[j] > 0.0 {
                    let sc = fast_expf(sm[j] - g_m);
                    g_d = fmaf(sd[j], sc, g_d);
                    g_o = fmaf(so[j], sc, g_o);
                }
                j += 1;
            }
            s += MB;
        }

        let val = div_rn(g_o, g_d) * fast_sigmoid(*gate_p.add(tid));
        let oidx = b * Q_DIM + h * HEAD_DIM + tid;
        if !out_gated.is_null() {
            *out_gated.add(oidx) = val;
        }
        if !out_gated_h.is_null() {
            *out_gated_h.add(oidx) = float2half(val);
        }
    }
}
