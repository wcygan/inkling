//! Tiled FP8 dense projections for the wide prefill (QKV and O), bit-identical to the split-K kernels of
//! `prefill_dense.rs` (`qkvrg_fp8_prefill_mt4`, `linear_fp8_prefill_k3072_mt4`).
//!
//! Contract: `y[t][n] = row_scale[n] * sum_k x[t][k] * w[n][k]`, x f16 [T][K], w E4M3 rows, y f32 [T][N].
//! The split-K kernels give each output SPLITK warp segments of `steps` x 128 columns: per segment an MMA chain
//! from zero over (step s, v, ii), where lane t takes its k pairs from its own 32-column chunk (columns
//! chunk * 32 + 8v + 4ii + {0, 1} and {2, 3}); the segments are then added in order to 0.0 and scaled.
//! These kernels compute a 128-token x 128-row tile per CTA (8 warps of 64 x 32) instead of 32 rows x 64 tokens
//! with K split over warps, so each weight and activation is read from L2 a quarter as often, and replay exactly
//! that arithmetic:
//! - K is staged in half steps (each chunk's 16 columns for v in {2h, 2h + 1}) with cp.async double
//!   buffering, the weights decoded to f16 in shared memory (`cvt.rn.f16x2.e4m3x2`, exact);
//! - every MMA takes the same 16 products in the same k positions, each segment's chain starts from zero
//!   and runs in the same order, and the segment sums are added in order;
//! so the outputs are bit-identical. Tokens past T and rows past N are zero-filled and not written.
//!
//! A plain tile with one MMA chain over all of K was 2.1x faster but not usable: the tensor core's f32
//! accumulation truncates, so long chains lose accuracy, and any change of summation order flipped a fragile
//! long-context greedy step.

use crate::common::*;
use crate::{HIDDEN, cvt_e4m3x2_to_half2};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{DynamicSharedArray, device, kernel, launch_bounds, ptx_asm};

const BM: usize = 128; // tokens per CTA
const BN: usize = 128; // weight rows per CTA
const THREADS: usize = 256;

#[inline(always)]
unsafe fn cp_async16(dst: *mut u8, src: *const u8) {
    let d = unsafe { cvta_generic_to_shared_u32(dst as *const u8) };
    unsafe {
        ptx_asm!("cp.async.cg.shared.global [%0], [%1], 16;", in("r") d, in("l") src, clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_async_commit() {
    unsafe {
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_async_wait1() {
    unsafe {
        ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_async_wait0() {
    unsafe {
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
    }
}

const TQ: usize = QRGS_STRIDE.div_ceil(BN); // 52
const TK: usize = KV_DIM / BN; // 8

const XP: usize = 72; // f16 row pitch (64 + 8)
const XPB: usize = 80; // E4M3 row pitch in bytes (64 + 16)
const XA_BYTES: usize = BM * XP * 2; // 18,432
const XW8_BYTES: usize = BN * XPB; // 10,240
const XSTAGE: usize = XA_BYTES + XW8_BYTES;
/// Dynamic shared memory of the `_tcx` kernels: two stages and the decoded weights (75,776 B).
pub const GEMM_TCX_SMEM: usize = 2 * XSTAGE + BN * XP * 2;

/// Half step `hs` (= 2 * step + h) of the K loop: per chunk t the 16 columns (4 * step + t) * 32 + 16h .. + 16.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn x_load<const K: usize>(sa: *mut u8, sw: *mut u8, x: *const u16, w: *const u8, tok0: usize, m: usize, row0: usize, n: usize, hs: usize, tid: usize) {
    unsafe {
        let mut i = tid;
        while i < BM * 8 {
            let r = i >> 3;
            let t = (i >> 1) & 3;
            let p = i & 1;
            let col = ((hs >> 1) * 4 + t) * 32 + (hs & 1) * 16;
            let dst = sa.add((r * XP + t * 16 + p * 8) * 2);
            if tok0 + r < m {
                cp_async16(dst, x.add((tok0 + r) * K + col + p * 8) as *const u8);
            } else {
                *(dst as *mut [u32; 4]) = [0; 4];
            }
            i += THREADS;
        }
        let mut i = tid;
        while i < BN * 4 {
            let r = i >> 2;
            let t = i & 3;
            let col = ((hs >> 1) * 4 + t) * 32 + (hs & 1) * 16;
            let dst = sw.add(r * XPB + t * 16);
            if row0 + r < n {
                cp_async16(dst, w.add((row0 + r) * K + col));
            } else {
                *(dst as *mut [u32; 4]) = [0; 4];
            }
            i += THREADS;
        }
    }
}

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_tile_x<const K: usize, const STEPS: usize>(smem: *mut u8, w: *const u8, rs: *const f32, x: *const u16, y: *mut f32, n: usize, m: usize, row0: usize, tok0: usize) {
    unsafe {
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let g = lane >> 2;
        let t = lane & 3;
        let wm = warp_i & 1;
        let wn = warp_i >> 1;
        let stage_a = |s: usize| smem.add(s * XSTAGE);
        let stage_w = |s: usize| smem.add(s * XSTAGE + XA_BYTES);
        let wh = smem.add(2 * XSTAGE) as *mut u16;
        let mut acc = [[[0.0f32; 4]; 4]; 4];
        let mut tot = [[[0.0f32; 4]; 4]; 4];
        let nhs = K / 64;
        let seg_hs = 2 * STEPS;

        x_load::<K>(stage_a(0), stage_w(0), x, w, tok0, m, row0, n, 0, tid);
        cp_async_commit();
        let mut hs = 0;
        #[nounroll]
        while hs < nhs {
            let st = hs & 1;
            if hs + 1 < nhs {
                x_load::<K>(stage_a(st ^ 1), stage_w(st ^ 1), x, w, tok0, m, row0, n, hs + 1, tid);
                cp_async_commit();
                cp_async_wait1();
            } else {
                cp_async_wait0();
            }
            sync_threads();
            // Decode: 512 pieces of 16 codes (row, chunk), two per thread.
            let mut i = tid;
            while i < BN * 4 {
                let r = i >> 2;
                let tc = i & 3;
                let q = *(stage_w(st).add(r * XPB + tc * 16) as *const [u32; 4]);
                let mut h = [0u32; 8];
                let mut j = 0;
                while j < 4 {
                    h[2 * j] = cvt_e4m3x2_to_half2((q[j] & 0xffff) as u16);
                    h[2 * j + 1] = cvt_e4m3x2_to_half2((q[j] >> 16) as u16);
                    j += 1;
                }
                let d = wh.add(r * XP + tc * 16) as *mut [u32; 4];
                *d = [h[0], h[1], h[2], h[3]];
                *d.add(1) = [h[4], h[5], h[6], h[7]];
                i += THREADS;
            }
            sync_threads();
            let sa = stage_a(st) as *const u16;
            let sw = wh as *const u16;
            let mut vl = 0;
            #[unroll]
            while vl < 2 {
                let mut ii = 0;
                #[unroll]
                while ii < 2 {
                    let off = t * 16 + vl * 8 + 4 * ii;
                    let mut bf = [[0u32; 2]; 4];
                    let mut nt = 0;
                    #[unroll]
                    while nt < 4 {
                        let rb = wn * 32 + nt * 8 + g;
                        bf[nt] = [*(sw.add(rb * XP + off) as *const u32), *(sw.add(rb * XP + off + 2) as *const u32)];
                        nt += 1;
                    }
                    let mut mt = 0;
                    #[unroll]
                    while mt < 4 {
                        let ra = wm * 64 + mt * 16 + g;
                        let a = [
                            *(sa.add(ra * XP + off) as *const u32),
                            *(sa.add((ra + 8) * XP + off) as *const u32),
                            *(sa.add(ra * XP + off + 2) as *const u32),
                            *(sa.add((ra + 8) * XP + off + 2) as *const u32),
                        ];
                        let mut nt = 0;
                        #[unroll]
                        while nt < 4 {
                            acc[mt][nt] = mma_m16n8k16_f32_f16(acc[mt][nt], a, bf[nt]);
                            nt += 1;
                        }
                        mt += 1;
                    }
                    ii += 1;
                }
                vl += 1;
            }
            if (hs + 1) % seg_hs == 0 {
                // End of a split-K segment: added to the total in segment order, as the split-K reduction does.
                let mut mt = 0;
                #[unroll]
                while mt < 4 {
                    let mut nt = 0;
                    #[unroll]
                    while nt < 4 {
                        let mut e = 0;
                        #[unroll]
                        while e < 4 {
                            tot[mt][nt][e] += acc[mt][nt][e];
                            acc[mt][nt][e] = 0.0;
                            e += 1;
                        }
                        nt += 1;
                    }
                    mt += 1;
                }
            }
            sync_threads();
            hs += 1;
        }

        let mut mt = 0;
        while mt < 4 {
            let mut nt = 0;
            while nt < 4 {
                let tk = tok0 + wm * 64 + mt * 16 + g;
                let rw = row0 + wn * 32 + nt * 8 + 2 * t;
                let mut h = 0;
                while h < 2 {
                    let tok = tk + 8 * h;
                    if tok < m {
                        if rw < n {
                            *y.add(tok * n + rw) = mul_rn(tot[mt][nt][2 * h], *rs.add(rw));
                        }
                        if rw + 1 < n {
                            *y.add(tok * n + rw + 1) = mul_rn(tot[mt][nt][2 * h + 1], *rs.add(rw + 1));
                        }
                    }
                    h += 1;
                }
                nt += 1;
            }
            mt += 1;
        }
    }
}

/// Bit-identical tiled QKV (see `gemm_tile_x`): grid (68, ceil(T / 128)), 256 threads, GEMM_TCX_SMEM dynamic.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn qkvrg_fp8_prefill_tcx(
    wq: *const u8,
    sq: *const f32,
    wk: *const u8,
    sk: *const f32,
    wv: *const u8,
    sv: *const f32,
    xq_h: *const u16,
    xk_h: *const u16,
    xv_h: *const u16,
    y_qrgs: *mut f32,
    y_k: *mut f32,
    y_v: *mut f32,
    m: i32,
) {
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let blk = blockIdx_x() as usize;
        let tok0 = blockIdx_y() as usize * BM;
        let (w, rs, x, y, r0, n) = if blk < TQ {
            (wq, sq, xq_h, y_qrgs, blk * BN, QRGS_STRIDE)
        } else if blk < TQ + TK {
            (wk, sk, xk_h, y_k, (blk - TQ) * BN, KV_DIM)
        } else {
            (wv, sv, xv_h, y_v, (blk - TQ - TK) * BN, KV_DIM)
        };
        // qkvrg_fp8_prefill_mt4: SPLITK 6 over K = 2304, 3 steps of 128 per segment.
        gemm_tile_x::<HIDDEN, 3>(smem, w, rs, x, y, n, m as usize, r0, tok0);
    }
}

/// Bit-identical tiled O projection: grid (18, ceil(T / 128)), 256 threads, GEMM_TCX_SMEM dynamic.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn linear_fp8_prefill_k3072_tcx(w: *const u8, row_scales: *const f32, x_h: *const u16, y: *mut f32, m: i32) {
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let r0 = blockIdx_x() as usize * BN;
        let tok0 = blockIdx_y() as usize * BM;
        // linear_fp8_prefill_k3072_mt4: SPLITK 4 over K = 3072, 6 steps of 128 per segment.
        gemm_tile_x::<3072, 6>(smem, w, row_scales, x_h, y, HIDDEN, m as usize, r0, tok0);
    }
}
