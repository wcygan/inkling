//! Depth attention, residual and norm kernels (exact mode).
//!
//! Ports of squid5090 @ 3ee093c `cuda/kernels_inkling.cu`:
//! - `rmsnorm_kernel` (62) as `rmsnorm`;
//! - `fused_depth_norm_kernel` (156) as `fused_depth_norm`;
//! - `extract_block_kernel` (2457) as `extract_block`.

use crate::common::*;
use crate::{HIDDEN, fmaf, float2half};
use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{SharedArray, kernel, launch_bounds};

/// `rmsnorm_kernel`: RMSNorm per group of `group_size` columns. Grid
/// B * width / group_size, 256 threads (128 when group_size < 256).
#[kernel]
pub unsafe fn rmsnorm(x: *const f32, weight: *const u16, out: *mut f32, width: i32, group_size: i32, eps: f32) {
    static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV_RMS: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv_rms = SharedArray::as_raw_mut_ptr(&raw mut INV_RMS);
        let groups_per_row = width / group_size;
        let blk = blockIdx_x() as i32;
        let b = blk / groups_per_row;
        let g = blk % groups_per_row;
        let tid = threadIdx_x() as i32;
        let bdim = blockDim_x() as i32;
        let off = b as usize * width as usize + (g * group_size) as usize;
        let xi = x.add(off);
        let xo = out.add(off);

        let mut sum_sq = 0.0f32;
        let mut d = tid;
        while d < group_size {
            let v = *xi.add(d as usize);
            sum_sq = fmaf(v, v, sum_sq);
            d += bdim;
        }
        sum_sq = warp_sum(sum_sq);
        let nwarps = bdim >> 5;
        if (tid & 31) == 0 {
            *sm.add((tid >> 5) as usize) = sum_sq;
        }
        sync_threads();
        if tid == 0 {
            let mut tot = 0.0f32;
            let mut i = 0;
            while i < nwarps {
                tot += *sm.add(i as usize);
                i += 1;
            }
            *inv_rms = rsqrtf(div_rn(tot, group_size as f32) + eps);
        }
        sync_threads();
        let s = *inv_rms;
        let mut d = tid;
        while d < group_size {
            let mut v = *xi.add(d as usize) * s;
            if !weight.is_null() {
                v *= bf16_to_f32(*weight.add(d as usize));
            }
            *xo.add(d as usize) = v;
            d += bdim;
        }
    }
}

/// `fused_depth_norm_kernel`: partial = rmsnorm(hidden); then for stream s,
/// depth attention over [blocks[0..n_blocks), partial] with query s and an
/// optional RMSNorm. Grid (B, n_out), 256 threads.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_depth_norm(
    hidden: *const f32,
    partial_out: *mut f32,
    blocks: *const f32,
    n_blocks: i32,
    queries: *const f32,
    norms: Bf16Ptr3,
    outs: MutPtr3,
    outs_h: HalfPtr3,
) {
    static mut S_PARTIAL: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
    static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV_RMS: SharedArray<f32, 1> = SharedArray::UNINIT;
    static mut S_VAL: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
    static mut PARTIALS: SharedArray<f32, 64> = SharedArray::UNINIT;
    static mut WEIGHTS: SharedArray<f32, 8> = SharedArray::UNINIT;
    unsafe {
        let s_partial = SharedArray::as_raw_mut_ptr(&raw mut S_PARTIAL);
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv_rms = SharedArray::as_raw_mut_ptr(&raw mut INV_RMS);
        let s_val = SharedArray::as_raw_mut_ptr(&raw mut S_VAL);
        let partials = SharedArray::as_raw_mut_ptr(&raw mut PARTIALS);
        let weights = SharedArray::as_raw_mut_ptr(&raw mut WEIGHTS);

        let b = blockIdx_x() as usize;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;

        let hb = hidden.add(b * HIDDEN);
        let mut sum_sq = 0.0f32;
        let mut d = tid;
        while d < HIDDEN {
            let v = *hb.add(d);
            *s_partial.add(d) = v;
            sum_sq = fmaf(v, v, sum_sq);
            d += 256;
        }
        sum_sq = warp_sum(sum_sq);
        if lane == 0 {
            *sm.add(warp_i) = sum_sq;
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

        let ir = *inv_rms;
        let po = if s == 0 && !partial_out.is_null() { partial_out.add(b * HIDDEN) } else { core::ptr::null_mut() };
        let mut d = tid;
        while d < HIDDEN {
            let pv = *s_partial.add(d) * ir;
            *s_partial.add(d) = pv;
            if !po.is_null() {
                *po.add(d) = pv;
            }
            d += 256;
        }
        sync_threads();

        let oh = if pick3(outs_h.p, s).is_null() { core::ptr::null_mut() } else { pick3(outs_h.p, s).add(b * HIDDEN) };
        if !queries.is_null() {
            let q = queries.add(s * HIDDEN);
            let blk_b = blocks.add(b * 4 * HIDDEN);
            let n_cand = n_blocks + 1;

            let mut c = 0;
            while c < n_cand {
                let src = if c < n_blocks { blk_b.add(c as usize * HIDDEN) } else { s_partial as *const f32 };
                let mut acc = 0.0f32;
                let mut d = tid;
                while d < HIDDEN {
                    acc = fmaf(*q.add(d), *src.add(d), acc);
                    d += 256;
                }
                acc = warp_sum(acc);
                if lane == 0 {
                    *partials.add(warp_i * 8 + c as usize) = acc;
                }
                c += 1;
            }
            sync_threads();
            if tid == 0 {
                let mut logits = [0.0f32; 8];
                let mut mx = -1e30f32;
                let mut c = 0;
                while c < n_cand {
                    let mut dot = 0.0f32;
                    let mut w = 0;
                    while w < 8 {
                        dot += *partials.add(w * 8 + c as usize);
                        w += 1;
                    }
                    logits[c as usize] = mul_rn(dot, INV_SQRT_HIDDEN);
                    mx = fmaxf(mx, logits[c as usize]);
                    c += 1;
                }
                let mut den = 0.0f32;
                let mut c = 0;
                while c < n_cand {
                    let e = expf(logits[c as usize] - mx);
                    *weights.add(c as usize) = e;
                    den += e;
                    c += 1;
                }
                let inv_d = rcp_rn(den);
                let mut c = 0;
                while c < n_cand {
                    *weights.add(c as usize) *= inv_d;
                    c += 1;
                }
            }
            sync_threads();

            let mut n_sq = 0.0f32;
            let mut d = tid;
            while d < HIDDEN {
                let mut v = 0.0f32;
                let mut c = 0;
                while c < n_cand {
                    let src = if c < n_blocks { blk_b.add(c as usize * HIDDEN) } else { s_partial as *const f32 };
                    v = fmaf(*weights.add(c as usize), *src.add(d), v);
                    c += 1;
                }
                *s_val.add(d) = v;
                n_sq = fmaf(v, v, n_sq);
                d += 256;
            }
            let nw = pick3(norms.p, s);
            let ob = pick3(outs.p, s).add(b * HIDDEN);
            if !nw.is_null() {
                n_sq = warp_sum(n_sq);
                if lane == 0 {
                    *sm.add(warp_i) = n_sq;
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
                let s2 = *inv_rms;
                let mut d = tid;
                while d < HIDDEN {
                    let v = *s_val.add(d) * s2 * bf16_to_f32(*nw.add(d));
                    *ob.add(d) = v;
                    if !oh.is_null() {
                        *oh.add(d) = float2half(v);
                    }
                    d += 256;
                }
            } else {
                let mut d = tid;
                while d < HIDDEN {
                    let v = *s_val.add(d);
                    *ob.add(d) = v;
                    if !oh.is_null() {
                        *oh.add(d) = float2half(v);
                    }
                    d += 256;
                }
            }
        } else {
            let nw = pick3(norms.p, s);
            let ob = pick3(outs.p, s).add(b * HIDDEN);
            let mut d = tid;
            while d < HIDDEN {
                let mut v = *s_partial.add(d);
                if !nw.is_null() {
                    v = round_bf16(v) * bf16_to_f32(*nw.add(d));
                }
                *ob.add(d) = v;
                if !oh.is_null() {
                    *oh.add(d) = float2half(v);
                }
                d += 256;
            }
        }
    }
}

/// `extract_block_kernel`: dst[b, :] = blocks[b, slot_idx, :]. Grid
/// ceil(B * HIDDEN / 256), 256 threads.
#[kernel]
pub unsafe fn extract_block(blocks: *const f32, max_blocks: i32, slot_idx: i32, dst: *mut f32, bsz: i32) {
    let i = (blockIdx_x() * blockDim_x() + threadIdx_x()) as i32;
    let total = bsz * HIDDEN as i32;
    if i < total {
        let b = i / HIDDEN as i32;
        let d = i % HIDDEN as i32;
        unsafe {
            *dst.add(i as usize) =
                *blocks.add(((b as usize * max_blocks as usize) + slot_idx as usize) * HIDDEN + d as usize);
        }
    }
}
