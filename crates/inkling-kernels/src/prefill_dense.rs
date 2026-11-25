//! FP8 dense attention projections for prefill chunks of M = 16..256 tokens, on the same row-scaled E4M3
//! weights as the B ≤ 8 kernels in `fp8.rs` (`pack_bf16_to_fp8_rows`).
//!
//! Contract (the prefill lane's): `y[t][n] = row_scale[n] * sum_k x[t][k] * w[n][k]` for t < T, with x f16
//! (u16 bits) row-major `[T][K]`, y f32 row-major `[T][N]`, and w / row_scale the `Fp8Dense` packing (E4M3
//! rows, one f32 scale per row) that `qkvrg_fused_fp8_mma_b*` / `linear_fp8_mma_b*_k3072` take. T is a runtime
//! i32 (any T ≥ 1; tokens past T read zeros and write nothing). Plain stream-ordered launches; launched with
//! PDL they also work (they wait before the first activation load and trigger after it).
//!
//! | Kernel | Args | Grid | Block |
//! | --- | --- | --- | --- |
//! | `qkvrg_fp8_prefill_mt{1,2,4}` | `(wq, sq, wk, sk, wv, sv, xq, xk, xv, y_qrgs, y_k, y_v, T)`; K = 2304, N = 6552 / 1024 / 1024 | (269, ceil(T / (16·MT))) | 192 |
//! | `linear_fp8_prefill_k3072_mt{1,2,4}` | `(w, s, x, y, T)`; O: K = 3072, N = 2304 | (144, ceil(T / 16)) for MT = 1; (144, ceil(T / 32)) for MT = 2; (72, ceil(T / 64)) for MT = 4 | 256 (MT = 1, 2), 128 (MT = 4) |
//!
//! Pick MT by T: 1 for T ≤ 16, 2 for T ≤ 32, 4 above (64 tokens per CTA). No shared-memory or other launch
//! attributes beyond grid and block.
//!
//! Design. A CTA computes 32 weight rows (16 for O at MT = 1, 2) x 16·MT tokens. Its warps split K (threads =
//! 32·SPLITK; SPLITK = 6 for QKV, 8 / 8 / 4 for O at MT = 1 / 2 / 4); each warp holds its 32 rows' E4M3 codes for one 128-column step decoded exactly to f16 in
//! registers (`cvt.rn.f16x2.e4m3x2`, the B operand of `mma.m16n8k16`, four n8 tiles) and runs every token
//! tile of the CTA against them (the A operand, 16 tokens per m16 tile, f16 loaded straight from global /
//! L1). The k order inside a step is the one the B ≤ 8 kernels use (each lane's 32-column chunk), applied
//! to A and B alike, so products are exact and only the f32 summation order differs. The SPLITK partial
//! tiles are summed in warp order in shared memory (deterministic), then scaled by the row scale.

use crate::common::*;
use crate::{HIDDEN, cvt_e4m3x2_to_half2, ld_stream_u4, pdl_trigger, pdl_wait};
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{SharedArray, device, kernel, launch_bounds};

/// Weight rows per CTA for the QKV kernels (four n8 tiles).
const TILE_N: usize = 32;

/// 32 E4M3 codes of one row at column chunk `c`, decoded to 16 half2 in memory order.
#[inline(always)]
unsafe fn load_decode(row: *const u8, c: usize) -> [u32; 16] {
    let (a, b) = unsafe { (ld_stream_u4(row.add(c * 32)), ld_stream_u4(row.add(c * 32 + 16))) };
    let ws = [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]];
    let mut o = [0u32; 16];
    let mut w = 0;
    while w < 8 {
        o[2 * w] = cvt_e4m3x2_to_half2((ws[w] & 0xffff) as u16);
        o[2 * w + 1] = cvt_e4m3x2_to_half2((ws[w] >> 16) as u16);
        w += 1;
    }
    o
}

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_body<const K: usize, const MT: usize, const SPLITK: usize, const NT: usize>(
    cs: *mut f32,
    w: *const u8,
    rs: *const f32,
    x: *const u16,
    y: *mut f32,
    n: usize,
    m: usize,
    r0: usize,
) {
    unsafe {
        let steps = K / (128 * SPLITK);
        let tile_n = 8 * NT;
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let g = lane >> 2;
        let t = lane & 3;
        let m0 = blockIdx_y() as usize * 16 * MT;
        let mut acc = [[[0.0f32; 4]; NT]; MT];
        // No producer data is read before this, so wait (and trigger) up front; the activation loads below
        // are then plain non-coherent loads the compiler may schedule ahead of the MMAs that precede them.
        pdl_wait();
        pdl_trigger();
        let mut s = 0;
        #[nounroll]
        while s < steps {
            let c = (warp_i * steps + s) * 4 + t;
            let mut wb = [[0u32; 16]; NT];
            let mut j = 0;
            #[unroll]
            while j < NT {
                let row = r0 + 8 * j + g;
                if row < n {
                    wb[j] = load_decode(w.add(row * K), c);
                }
                j += 1;
            }
            let mut v = 0;
            #[unroll]
            while v < 4 {
                // Tokens g and g + 8 of each m16 tile: 8 halves (half2 positions 4v..4v+3 of the chunk).
                let mut xa = [[0u32; 4]; 8]; // 2·MT used (MT ≤ 4)
                let mut r = 0;
                #[unroll]
                while r < 2 * MT {
                    let tok = m0 + 16 * (r >> 1) + g + 8 * (r & 1);
                    if tok < m {
                        xa[r] = ldg_u4(x.add(tok * K + c * 32 + 8 * v) as *const u8);
                    }
                    r += 1;
                }
                let mut ii = 0;
                #[unroll]
                while ii < 2 {
                    let i = 2 * v + ii;
                    let mut mt = 0;
                    #[unroll]
                    while mt < MT {
                        let a = [xa[2 * mt][2 * ii], xa[2 * mt + 1][2 * ii], xa[2 * mt][2 * ii + 1], xa[2 * mt + 1][2 * ii + 1]];
                        let mut j = 0;
                        #[unroll]
                        while j < NT {
                            acc[mt][j] = mma_m16n8k16_f32_f16(acc[mt][j], a, [wb[j][2 * i], wb[j][2 * i + 1]]);
                            j += 1;
                        }
                        mt += 1;
                    }
                    ii += 1;
                }
                v += 1;
            }
            s += 1;
        }

        // Partial tiles: cs[warp][token 16·MT][weight row 32].
        let slab = 16 * MT * tile_n;
        let base = cs.add(warp_i * slab);
        let mut mt = 0;
        while mt < MT {
            let mut j = 0;
            while j < NT {
                let tok = 16 * mt + g;
                let wr = 8 * j + 2 * t;
                *base.add(tok * tile_n + wr) = acc[mt][j][0];
                *base.add(tok * tile_n + wr + 1) = acc[mt][j][1];
                *base.add((tok + 8) * tile_n + wr) = acc[mt][j][2];
                *base.add((tok + 8) * tile_n + wr + 1) = acc[mt][j][3];
                j += 1;
            }
            mt += 1;
        }
        sync_threads();
        let mut k = tid;
        while k < slab {
            let tok = m0 + k / tile_n;
            let row = r0 + k % tile_n;
            if tok < m && row < n {
                let mut sum = 0.0f32;
                let mut wi = 0;
                while wi < SPLITK {
                    sum += *cs.add(wi * slab + k);
                    wi += 1;
                }
                *y.add(tok * n + row) = mul_rn(sum, *rs.add(row));
            }
            k += SPLITK * 32;
        }
    }
}

const T_Q: usize = QRGS_STRIDE.div_ceil(TILE_N); // 205
const T_K: usize = KV_DIM / TILE_N; // 32

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn qkvrg_body<const MT: usize, const SPLITK: usize>(
    cs: *mut f32,
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
    let blk = blockIdx_x() as usize;
    let (w, rs, x, y, r0, n) = if blk < T_Q {
        (wq, sq, xq_h, y_qrgs, blk * TILE_N, QRGS_STRIDE)
    } else if blk < T_Q + T_K {
        (wk, sk, xk_h, y_k, (blk - T_Q) * TILE_N, KV_DIM)
    } else {
        (wv, sv, xv_h, y_v, (blk - T_Q - T_K) * TILE_N, KV_DIM)
    };
    unsafe { gemm_body::<HIDDEN, MT, SPLITK, 4>(cs, w, rs, x, y, n, m as usize, r0) }
}

macro_rules! qkvrg_prefill_kernel {
    ($name:ident, $mt:expr, $sk:expr, $threads:expr) => {
        /// FP8 Q|R|gate|sink + K + V for a prefill chunk: grid (269, ceil(T / (16·MT))), 32·SPLITK threads.
        #[kernel]
        #[launch_bounds($threads)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
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
            static mut CS: SharedArray<f32, { $sk * 16 * $mt * TILE_N }, 16> = SharedArray::UNINIT;
            unsafe {
                qkvrg_body::<$mt, $sk>(SharedArray::as_raw_mut_ptr(&raw mut CS), wq, sq, wk, sk, wv, sv, xq_h, xk_h, xv_h, y_qrgs, y_k, y_v, m)
            }
        }
    };
}

// SPLITK: 18 steps of 128 columns over K = 2304; more warps per CTA where few token tiles run.
qkvrg_prefill_kernel!(qkvrg_fp8_prefill_mt1, 1, 6, 192);
qkvrg_prefill_kernel!(qkvrg_fp8_prefill_mt2, 2, 6, 192);
qkvrg_prefill_kernel!(qkvrg_fp8_prefill_mt4, 4, 6, 192);

macro_rules! linear_prefill_kernel {
    ($name:ident, $mt:expr, $sk:expr, $threads:expr, $nt:expr) => {
        /// FP8 O projection (K = 3072, N = 2304) for a prefill chunk: grid (2304 / (8·NT), ceil(T / (16·MT))), 32·SPLITK threads.
        #[kernel]
        #[launch_bounds($threads)]
        pub unsafe fn $name(w: *const u8, row_scales: *const f32, x_h: *const u16, y: *mut f32, m: i32) {
            static mut CS: SharedArray<f32, { $sk * 16 * $mt * 8 * $nt }, 16> = SharedArray::UNINIT;
            let r0 = blockIdx_x() as usize * 8 * $nt;
            unsafe {
                gemm_body::<3072, $mt, $sk, $nt>(SharedArray::as_raw_mut_ptr(&raw mut CS), w, row_scales, x_h, y, HIDDEN, m as usize, r0)
            }
        }
    };
}

// SPLITK: 24 steps over K = 3072. Small T: 16-row tiles (144 CTAs) and 8 warps, as the B = 8 kernel.
linear_prefill_kernel!(linear_fp8_prefill_k3072_mt1, 1, 8, 256, 2);
linear_prefill_kernel!(linear_fp8_prefill_k3072_mt2, 2, 8, 256, 2);
linear_prefill_kernel!(linear_fp8_prefill_k3072_mt4, 4, 4, 128, 4);
