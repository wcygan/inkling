//! FP8 (E4M3, one f32 scale per output row) dense attention projections, for a quality-passing
//! speculative mode between BF16 dense and NVFP4 dense.
//!
//! - `pack_bf16_to_fp8_rows`: squid5090 `z_quant_bf16_to_fp8_kernel` (cuda_abi.cu:83), grid N, 256
//!   threads: scale = max(amax_row / 448, 1e-12), codes = RN-satfinite E4M3 of x / scale.
//! - `linear_fp8_mma_b{1,2,4,8}_k3072` (O) and `qkvrg_fused_fp8_mma_b{1,2,4,8}` (Q|R|gate|sink, K, V):
//!   the NVFP4 tensor-core kernels' structure (16 rows per CTA, SPLITK warps over K, the batch as the
//!   N = 8 side of `mma.m16n8k16`) with each 32-column chunk decoded exactly from E4M3 to f16
//!   (`cvt.rn.f16x2.e4m3x2`), f32 accumulation in the MMA, and the row scale in an f32 epilogue. This
//!   avoids squid5090's `linear_fp8_h` FP16 accumulation (its bug 6). B = 1 uses the same kernels
//!   (7 of the 8 MMA columns are zero); they are bandwidth-bound.
//!
//! Not in squid5090; checked against an f64 CPU reference in kernel-parity's `fp8` group.

use crate::common::*;
use crate::nvfp4::{fabsf, f32_to_e4m3};
use crate::{HIDDEN, cvt_e4m3x2_to_half2, ld_dep_u4, ld_stream_u4, pdl_trigger};
use cuda_device::thread::{blockDim_x, blockIdx_x, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{SharedArray, device, kernel, launch_bounds};

/// `z_quant_bf16_to_fp8_kernel`: one block per row.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn pack_bf16_to_fp8_rows(src: *const u16, dst_w: *mut u8, dst_row_scales: *mut f32, n: i32, k: i32) {
    static mut SMEM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let smem = SharedArray::as_raw_mut_ptr(&raw mut SMEM);
        let inv = SharedArray::as_raw_mut_ptr(&raw mut INV);
        let r = blockIdx_x() as i32;
        if r >= n {
            return;
        }
        let k = k as usize;
        let tid = threadIdx_x() as usize;
        let bdim = blockDim_x() as usize;
        let row_in = src.add(r as usize * k);
        let mut lmax = 0.0f32;
        let mut c = tid;
        while c < k {
            lmax = fmaxf(lmax, fabsf(bf16_to_f32(*row_in.add(c))));
            c += bdim;
        }
        lmax = warp_max(lmax);
        let lane = tid & 31;
        let warp_i = tid >> 5;
        if lane == 0 {
            *smem.add(warp_i) = lmax;
        }
        sync_threads();
        if warp_i == 0 {
            let v = warp_max(if lane < (bdim >> 5) { *smem.add(lane) } else { 0.0 });
            if lane == 0 {
                let sc = fmaxf(div_rn(v, 448.0), 1e-12);
                *dst_row_scales.add(r as usize) = sc;
                *inv = rcp_rn(sc);
            }
        }
        sync_threads();
        let isc = *inv;
        let row_out = dst_w.add(r as usize * k);
        let mut c = tid;
        while c < k {
            *row_out.add(c) = f32_to_e4m3(bf16_to_f32(*row_in.add(c)) * isc);
            c += bdim;
        }
    }
}

const U: usize = 3;

/// 32 E4M3 codes (one chunk of one row) as 8 words.
type Chunk = [u32; 8];

#[inline(always)]
unsafe fn load_chunk(row: *const u8, c: usize) -> Chunk {
    unsafe {
        let a = ld_stream_u4(row.add(c * 32));
        let b = ld_stream_u4(row.add(c * 32 + 16));
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]]
    }
}

/// Rows g and g + 8 of the tile, U chunks each, zero past `n`.
#[inline(always)]
unsafe fn load_rows(w: *const u8, k: usize, ra: usize, rb: usize, n: usize, c_base: usize) -> ([Chunk; U], [Chunk; U]) {
    let mut a = [[0u32; 8]; U];
    let mut b = [[0u32; 8]; U];
    let mut u = 0;
    while u < U {
        let c = c_base + 4 * u;
        unsafe {
            if ra < n {
                a[u] = load_chunk(w.add(ra * k), c);
            }
            if rb < n {
                b[u] = load_chunk(w.add(rb * k), c);
            }
        }
        u += 1;
    }
    (a, b)
}

/// Exact E4M3 -> f16 of one chunk: 16 half2 in memory order (as the NVFP4 chunk decode).
#[inline(always)]
fn decode(ch: &Chunk) -> [u32; 16] {
    let mut o = [0u32; 16];
    let mut w = 0;
    while w < 8 {
        o[2 * w] = cvt_e4m3x2_to_half2((ch[w] & 0xffff) as u16);
        o[2 * w + 1] = cvt_e4m3x2_to_half2((ch[w] >> 16) as u16);
        w += 1;
    }
    o
}

/// The B operand: 16 half2 of token g at columns (c_base + 4u) * 32, zero past B.
#[inline(always)]
unsafe fn load_x<const B: usize>(x_h: *const u16, k: usize, g: usize, c_base: usize) -> [[u32; 16]; U] {
    let mut xw = [[0u32; 16]; U];
    if g < B {
        let xg = unsafe { x_h.add(g * k) };
        let mut u = 0;
        while u < U {
            let col = (c_base + 4 * u) * 32;
            let mut v = 0;
            while v < 4 {
                let q = unsafe { ld_dep_u4(xg.add(col + 8 * v)) };
                xw[u][4 * v] = q[0];
                xw[u][4 * v + 1] = q[1];
                xw[u][4 * v + 2] = q[2];
                xw[u][4 * v + 3] = q[3];
                v += 1;
            }
            u += 1;
        }
    }
    xw
}

#[device]
fn fp8_mma_chunks(a: &[Chunk; U], b: &[Chunk; U], xw: &[[u32; 16]; U], dep: u32, d: &mut [f32; 4]) {
    let mut u = 0;
    #[unroll]
    while u < U {
        let fa = decode(&a[u]);
        let fb = decode(&b[u]);
        let mut i = 0;
        #[unroll]
        while i < 8 {
            *d = unsafe { mma_m16n8k16_f32_f16(*d, [fa[2 * i], fb[2 * i], fa[2 * i + 1], fb[2 * i + 1]], [xw[u][2 * i] ^ dep, xw[u][2 * i + 1]]) };
            i += 1;
        }
        u += 1;
    }
}

#[inline(always)]
fn dep_of(a: &[Chunk; U], b: &[Chunk; U], x0: u32, zero: u32) -> u32 {
    (a[0][0] ^ a[U - 1][7] ^ b[0][0] ^ b[U - 1][7] ^ x0) & zero
}

#[inline(always)]
unsafe fn store_tile(cs: *mut f32, warp_i: usize, g: usize, t: usize, d: &[f32; 4]) {
    unsafe {
        let base = cs.add(warp_i * 16 * 8);
        *base.add(g * 8 + 2 * t) = d[0];
        *base.add(g * 8 + 2 * t + 1) = d[1];
        *base.add((g + 8) * 8 + 2 * t) = d[2];
        *base.add((g + 8) * 8 + 2 * t + 1) = d[3];
    }
}

/// y[m, row] = (sum of the SPLITK partial tiles) * row_scale[row].
#[inline(always)]
unsafe fn epilogue<const B: usize, const SPLITK: usize>(cs: *const f32, rs: *const f32, y: *mut f32, r0: usize, n: usize, tid: usize) {
    unsafe {
        let mut k = tid;
        while k < 16 * B {
            let r = k / B;
            let m = k % B;
            let row = r0 + r;
            if row < n {
                let mut s = 0.0f32;
                let mut w = 0;
                while w < SPLITK {
                    s += *cs.add((w * 16 + r) * 8 + m);
                    w += 1;
                }
                *y.add(m * n + row) = mul_rn(s, *rs.add(row));
            }
            k += SPLITK * 32;
        }
    }
}

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn linear_body<const B: usize, const K: usize, const SPLITK: usize>(
    cs: *mut f32,
    w: *const u8,
    rs: *const f32,
    x_h: *const u16,
    y: *mut f32,
    n: i32,
    zero: u32,
) {
    unsafe {
        let total_u = K / (128 * SPLITK);
        let nb = total_u / U;
        let n = n as usize;
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let g = lane >> 2;
        let t = lane & 3;
        let r0 = blockIdx_x() as usize * 16;
        let c_warp = warp_i * (4 * total_u) + t;
        let mut d = [0.0f32; 4];
        let mut jb = 0;
        #[nounroll]
        while jb < nb {
            let c_base = c_warp + jb * (4 * U);
            let (a, b) = load_rows(w, K, r0 + g, r0 + g + 8, n, c_base);
            if jb == 0 {
                pdl_wait_fenced(zero);
                pdl_trigger();
            }
            let xw = load_x::<B>(x_h, K, g, c_base);
            let dep = dep_of(&a, &b, xw[0][0], zero);
            fp8_mma_chunks(&a, &b, &xw, dep, &mut d);
            jb += 1;
        }
        store_tile(cs, warp_i, g, t, &d);
        sync_threads();
        epilogue::<B, SPLITK>(cs, rs, y, r0, n, tid);
    }
}

macro_rules! linear_fp8_kernel {
    ($name:ident, $b:expr) => {
        /// FP8 linear, K = 3072 (O): y[B, N] = (x_h[B, K] W^T) * row_scale. Grid ceil(N / 16), 256 threads, PDL.
        #[kernel]
        #[launch_bounds(256)]
        pub unsafe fn $name(w: *const u8, row_scales: *const f32, x_h: *const u16, y: *mut f32, n: i32, zero: u32) {
            static mut CS: SharedArray<f32, { 8 * 16 * 8 }, 16> = SharedArray::UNINIT;
            unsafe { linear_body::<$b, 3072, 8>(SharedArray::as_raw_mut_ptr(&raw mut CS), w, row_scales, x_h, y, n, zero) }
        }
    };
}

linear_fp8_kernel!(linear_fp8_mma_b1_k3072, 1);
linear_fp8_kernel!(linear_fp8_mma_b2_k3072, 2);
linear_fp8_kernel!(linear_fp8_mma_b4_k3072, 4);
linear_fp8_kernel!(linear_fp8_mma_b8_k3072, 8);

const QK_SPLITK: usize = 6;

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn qkvrg_body<const B: usize>(
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
    zero: u32,
) {
    unsafe {
        const N_Q: usize = QRGS_STRIDE;
        const T_Q: usize = N_Q.div_ceil(16); // 410
        const T_K: usize = KV_DIM / 16; // 64
        let blk = blockIdx_x() as usize;
        let (w, rs, x_h, y_out, r0, n_out) = if blk < T_Q {
            (wq, sq, xq_h, y_qrgs, blk * 16, N_Q)
        } else if blk < T_Q + T_K {
            (wk, sk, xk_h, y_k, (blk - T_Q) * 16, KV_DIM)
        } else {
            (wv, sv, xv_h, y_v, (blk - T_Q - T_K) * 16, KV_DIM)
        };
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let g = lane >> 2;
        let t = lane & 3;
        let c_base = warp_i * (4 * U) + t;
        let (a, b) = load_rows(w, HIDDEN, r0 + g, r0 + g + 8, n_out, c_base);
        pdl_wait_fenced(zero);
        pdl_trigger();
        let xw = load_x::<B>(x_h, HIDDEN, g, c_base);
        let dep = dep_of(&a, &b, xw[0][0], zero);
        let mut d = [0.0f32; 4];
        fp8_mma_chunks(&a, &b, &xw, dep, &mut d);
        store_tile(cs, warp_i, g, t, &d);
        sync_threads();
        epilogue::<B, QK_SPLITK>(cs, rs, y_out, r0, n_out, tid);
    }
}

macro_rules! qkvrg_fp8_kernel {
    ($name:ident, $b:expr) => {
        /// FP8 Q|R|gate|sink (410 tiles) + K (64) + V (64) = 538 CTAs of 16 rows, 192 threads, PDL.
        #[kernel]
        #[launch_bounds(192)]
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
            zero: u32,
        ) {
            static mut CS: SharedArray<f32, { QK_SPLITK * 16 * 8 }, 16> = SharedArray::UNINIT;
            unsafe {
                qkvrg_body::<$b>(SharedArray::as_raw_mut_ptr(&raw mut CS), wq, sq, wk, sk, wv, sv, xq_h, xk_h, xv_h, y_qrgs, y_k, y_v, zero)
            }
        }
    };
}

qkvrg_fp8_kernel!(qkvrg_fused_fp8_mma_b1, 1);
qkvrg_fp8_kernel!(qkvrg_fused_fp8_mma_b2, 2);
qkvrg_fp8_kernel!(qkvrg_fused_fp8_mma_b4, 4);
qkvrg_fp8_kernel!(qkvrg_fused_fp8_mma_b8, 8);
