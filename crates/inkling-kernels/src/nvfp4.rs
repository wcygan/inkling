//! NVFP4-mode kernels: B=1 decode GEMVs, the NVFP4 router and fused
//! controller, and the load-time BF16/MXFP8 → NVFP4 packing kernels.
//!
//! Ports of squid5090 @ 3ee093c:
//! - `qkvrg_fused_nvfp4_b1_pdl_kernel` (kernels_gemv.cu:340) as `qkvrg_fused_nvfp4_b1_pdl`;
//! - `linear_nvfp4_b1_pdl_kernel<false, 12>` / `<false, 9>` (kernels_gemv.cu:172) as
//!   `linear_nvfp4_b1_pdl_k3072` / `linear_nvfp4_b1_pdl_k2304`;
//! - `route_logits_nvfp4_parallel_kernel<1>` (kernels_inkling.cu:1254) as `route_logits_nvfp4_parallel_b1`;
//! - `mega_controller_nvfp4_kernel` (kernels_inkling.cu:1439) as `mega_controller_nvfp4`;
//! - cuda_abi.cu packing: `z_bf16_amax_kernel` as `amax_bf16`,
//!   `z_quant_bf16_to_nvfp4_kernel` as `pack_bf16_to_nvfp4`,
//!   `z_mxfp8_expert_amax_kernel` as `mxfp8_expert_amax`,
//!   `z_mxfp8_to_nvfp4_expert_kernel` as `repack_mxfp8_to_nvfp4`.

use crate::common::*;
use crate::{
    HIDDEN, cvt_e2m1x8_to_half2x4, cvt_e4m3_to_float, fast_tanh, fmaf, float2half, h2f_hi, h2f_lo, half_warp_sum,
    ld_dep_u4, pdl_trigger, pdl_wait,
};
use cuda_device::f16x2::{add_f16x2, fma_f16x2, mul_f16x2};
use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, gridDim_x, sync_threads, threadIdx_x};
use cuda_device::{SharedArray, device, kernel, launch_bounds, ptx_asm, warp};

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// `ld_stream_u2`: streaming 64-bit weight load that bypasses L1.
#[inline(always)]
pub(crate) unsafe fn ld_stream_u2(p: *const u8) -> [u32; 2] {
    let (x, y): (u32, u32);
    unsafe {
        ptx_asm!(
            "ld.global.nc.L1::no_allocate.L2::256B.v2.u32 {%0,%1}, [%2];",
            out("=r") x,
            out("=r") y,
            in("l") p,
        );
    }
    [x, y]
}

/// `ld_scale_u8`.
#[inline(always)]
pub(crate) unsafe fn ld_scale_u8(p: *const u8) -> u16 {
    let v: u16;
    unsafe {
        ptx_asm!("ld.global.nc.u8 %0, [%1];", out("=h") v, in("l") p);
    }
    v
}


/// The 16-weight NVFP4 dot product used by every B=1 GEMV: weights (two u32
/// of FP4 codes) times 16 halves (`xb0`, `xb1`), accumulated as half2 and
/// returned as a float (`nvfp4_block16_dot` / the inline loops).
#[inline(always)]
pub(crate) fn block16_dot(w: [u32; 2], xb0: [u32; 4], xb1: [u32; 4]) -> f32 {
    let lo = cvt_e2m1x8_to_half2x4(w[0]);
    let hi = cvt_e2m1x8_to_half2x4(w[1]);
    let mut p0 = mul_f16x2(lo[0], xb0[0]);
    let mut p1 = mul_f16x2(lo[1], xb0[1]);
    p0 = fma_f16x2(lo[2], xb0[2], p0);
    p1 = fma_f16x2(lo[3], xb0[3], p1);
    p0 = fma_f16x2(hi[0], xb1[0], p0);
    p1 = fma_f16x2(hi[1], xb1[1], p1);
    p0 = fma_f16x2(hi[2], xb1[2], p0);
    p1 = fma_f16x2(hi[3], xb1[3], p1);
    let s = add_f16x2(p0, p1);
    h2f_lo(s) + h2f_hi(s)
}

// ---------------------------------------------------------------------------
// B=1 GEMVs (16 threads per row, all weight loads issued before the PDL wait)
// ---------------------------------------------------------------------------

/// `linear_nvfp4_b1_pdl_kernel<ADD_RES, ITERS>` body; K = ITERS * 256.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn linear_nvfp4_b1_body<const ITERS: usize>(
    w_data: *const u8,
    w_scales: *const u8,
    global_scale: f32,
    x_h: *const u16,
    y: *mut f32,
    n: i32,
    k: i32,
    zero: u32,
) {
    unsafe {
        let hw = (threadIdx_x() >> 4) as usize;
        let lane = (threadIdx_x() & 15) as usize;
        let row = blockIdx_x() as usize * 8 + hw;
        if row >= n as usize {
            return;
        }
        let k = k as usize;
        let w_row = w_data.add(row * (k >> 1));
        let s_row = w_scales.add(row * (k >> 4));

        let mut w8 = [[0u32; 2]; ITERS];
        let mut s8 = [0u16; ITERS];
        let mut iter = 0;
        #[unroll]
        while iter < ITERS {
            let blk = lane + iter * 16;
            w8[iter] = ld_stream_u2(w_row.add(blk * 8));
            s8[iter] = ld_scale_u8(s_row.add(blk));
            iter += 1;
        }
        pdl_wait_fenced(zero);
        pdl_trigger();

        let mut acc = 0.0f32;
        let mut iter = 0;
        #[unroll]
        while iter < ITERS {
            let koff = (lane + iter * 16) << 4;
            let xv0 = ld_dep_u4(x_h.add(koff));
            let xv1 = ld_dep_u4(x_h.add(koff + 8));
            let sc = cvt_e4m3_to_float(s8[iter] & 0xff);
            acc = fmaf(block16_dot(w8[iter], xv0, xv1), sc, acc);
            iter += 1;
        }
        let s = half_warp_sum(acc) * global_scale;
        if lane == 0 {
            *y.add(row) = s;
        }
    }
}

/// `linear_nvfp4_b1_pdl_kernel<false, 12>`: y = W x for K = 3072 (O projection).
/// Grid ceil(N / 8), 128 threads, PDL.
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn linear_nvfp4_b1_pdl_k3072(
    w_data: *const u8,
    w_scales: *const u8,
    global_scale: f32,
    x_h: *const u16,
    y: *mut f32,
    n: i32,
    k: i32,
    zero: u32,
) {
    unsafe { linear_nvfp4_b1_body::<12>(w_data, w_scales, global_scale, x_h, y, n, k, zero) }
}

/// `linear_nvfp4_b1_pdl_kernel<false, 9>`: y = W x for K = 2304.
/// Grid ceil(N / 8), 128 threads, PDL.
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn linear_nvfp4_b1_pdl_k2304(
    w_data: *const u8,
    w_scales: *const u8,
    global_scale: f32,
    x_h: *const u16,
    y: *mut f32,
    n: i32,
    k: i32,
    zero: u32,
) {
    unsafe { linear_nvfp4_b1_body::<9>(w_data, w_scales, global_scale, x_h, y, n, k, zero) }
}

/// `qkvrg_fused_nvfp4_b1_pdl_kernel`: QRGS (6552 rows) + K (1024) + V (1024)
/// in one launch. Grid 1075, 128 threads, PDL.
#[kernel]
#[launch_bounds(128)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn qkvrg_fused_nvfp4_b1_pdl(
    wq_d: *const u8,
    wq_s: *const u8,
    q_gs: f32,
    wk_d: *const u8,
    wk_s: *const u8,
    k_gs: f32,
    wv_d: *const u8,
    wv_s: *const u8,
    v_gs: f32,
    xq_h: *const u16,
    xk_h: *const u16,
    xv_h: *const u16,
    y_qrgs: *mut f32,
    y_k: *mut f32,
    y_v: *mut f32,
    zero: u32,
) {
    unsafe {
        const N_Q: usize = QRGS_STRIDE; // 6552
        const N_K: usize = KV_DIM;
        const N_TOT: usize = N_Q + N_K + N_K; // 8600
        let hw = (threadIdx_x() >> 4) as usize;
        let lane = (threadIdx_x() & 15) as usize;
        let g_row = blockIdx_x() as usize * 8 + hw;
        if g_row >= N_TOT {
            return;
        }
        let (w_data, w_scales, gscale, x_h, y_out, row) = if g_row < N_Q {
            (wq_d, wq_s, q_gs, xq_h, y_qrgs, g_row)
        } else if g_row < N_Q + N_K {
            (wk_d, wk_s, k_gs, xk_h, y_k, g_row - N_Q)
        } else {
            (wv_d, wv_s, v_gs, xv_h, y_v, g_row - (N_Q + N_K))
        };

        const NBLK: usize = HIDDEN >> 4; // 144
        let w_row = w_data.add(row * (HIDDEN >> 1));
        let s_row = w_scales.add(row * NBLK);
        let mut w8 = [[0u32; 2]; 9];
        let mut s8 = [0u16; 9];
        let mut iter = 0;
        #[unroll]
        while iter < 9 {
            let blk = lane + iter * 16;
            w8[iter] = ld_stream_u2(w_row.add(blk * 8));
            s8[iter] = ld_scale_u8(s_row.add(blk));
            iter += 1;
        }
        pdl_wait_fenced(zero);
        pdl_trigger();

        let mut acc = 0.0f32;
        let mut iter = 0;
        #[unroll]
        while iter < 9 {
            let koff = (lane + iter * 16) << 4;
            let xv0 = ld_dep_u4(x_h.add(koff));
            let xv1 = ld_dep_u4(x_h.add(koff + 8));
            let sc = cvt_e4m3_to_float(s8[iter] & 0xff);
            acc = fmaf(block16_dot(w8[iter], xv0, xv1), sc, acc);
            iter += 1;
        }
        let s = half_warp_sum(acc) * gscale;
        if lane == 0 {
            *y_out.add(row) = s;
        }
    }
}

/// The same 16-weight dot product as `block16_dot`, with the weights already
/// decoded to half2 (so B tokens share one decode, as in squid5090).
#[inline(always)]
pub(crate) fn block16_dot_decoded(wh: &[u32; 8], xb0: [u32; 4], xb1: [u32; 4]) -> f32 {
    let mut p0 = mul_f16x2(wh[0], xb0[0]);
    let mut p1 = mul_f16x2(wh[1], xb0[1]);
    p0 = fma_f16x2(wh[2], xb0[2], p0);
    p1 = fma_f16x2(wh[3], xb0[3], p1);
    p0 = fma_f16x2(wh[4], xb1[0], p0);
    p1 = fma_f16x2(wh[5], xb1[1], p1);
    p0 = fma_f16x2(wh[6], xb1[2], p0);
    p1 = fma_f16x2(wh[7], xb1[3], p1);
    let s = add_f16x2(p0, p1);
    h2f_lo(s) + h2f_hi(s)
}

/// `route_logits_nvfp4_parallel_kernel<B>` body: 128 expert logits for B
/// f16 MLP inputs, two experts per 32-thread block.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn route_logits_nvfp4_body<const B: usize>(
    m_h: *const u16,
    w_data: *const u8,
    w_scales: *const u8,
    gscale: f32,
    out_logits: *mut f32,
    zero: u32,
) {
    unsafe {
        let sub_e = (threadIdx_x() >> 4) as usize;
        let lane = (threadIdx_x() & 15) as usize;
        let e = blockIdx_x() as usize * 2 + sub_e;
        const W_STRIDE: usize = HIDDEN >> 1;
        const S_STRIDE: usize = HIDDEN >> 4;
        let wr = w_data.add(e * W_STRIDE);
        let sr = w_scales.add(e * S_STRIDE);

        let mut w8 = [[0u32; 2]; 9];
        let mut s8 = [0u16; 9];
        let mut iter = 0;
        #[unroll]
        while iter < 9 {
            let blk = lane + iter * 16;
            w8[iter] = ld_stream_u2(wr.add(blk * 8));
            s8[iter] = ld_scale_u8(sr.add(blk));
            iter += 1;
        }
        pdl_wait_fenced(zero);
        pdl_trigger();

        let mut acc = [0.0f32; B];
        let mut iter = 0;
        #[unroll]
        while iter < 9 {
            let blk = lane + iter * 16;
            let sc = cvt_e4m3_to_float(s8[iter] & 0xff);
            let lo = cvt_e2m1x8_to_half2x4(w8[iter][0]);
            let hi = cvt_e2m1x8_to_half2x4(w8[iter][1]);
            let wh = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
            let koff = blk << 4;
            let mut b = 0;
            #[unroll]
            while b < B {
                let xv0 = ld_dep_u4(m_h.add(b * HIDDEN + koff));
                let xv1 = ld_dep_u4(m_h.add(b * HIDDEN + koff + 8));
                acc[b] = fmaf(block16_dot_decoded(&wh, xv0, xv1), sc, acc[b]);
                b += 1;
            }
            iter += 1;
        }
        let mut b = 0;
        #[unroll]
        while b < B {
            let val = half_warp_sum(acc[b]) * gscale;
            if lane == 0 {
                *out_logits.add(b * NUM_EXPERTS + e) = val;
            }
            b += 1;
        }
    }
}

macro_rules! route_logits_nvfp4_kernel {
    ($name:ident, $b:expr) => {
        /// `route_logits_nvfp4_parallel_kernel<B>`: grid 64, 32 threads, PDL.
        /// `moe_down_zero` is unused (squid5090 ignores it here too).
        #[kernel]
        #[launch_bounds(32)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            m_h: *const u16,
            w_data: *const u8,
            w_scales: *const u8,
            gscale: f32,
            out_logits: *mut f32,
            _moe_down_zero: *mut f32,
            zero: u32,
        ) {
            unsafe { route_logits_nvfp4_body::<$b>(m_h, w_data, w_scales, gscale, out_logits, zero) }
        }
    };
}

route_logits_nvfp4_kernel!(route_logits_nvfp4_parallel_b1, 1);
route_logits_nvfp4_kernel!(route_logits_nvfp4_parallel_b2, 2);
route_logits_nvfp4_kernel!(route_logits_nvfp4_parallel_b4, 4);
route_logits_nvfp4_kernel!(route_logits_nvfp4_parallel_b8, 8);

// ---------------------------------------------------------------------------
// linear_nvfp4_f32_kernel<B, false> (kernels_gemv.cu:649): f32 input
// ---------------------------------------------------------------------------

/// `linear_nvfp4_f32_kernel<B, false>` body: one warp per output row, x
/// converted to half2 on the fly (`cvt.rn.f16x2.f32`).
#[device]
unsafe fn linear_nvfp4_f32_body<const B: usize>(
    w_data: *const u8,
    w_scales: *const u8,
    global_scale: f32,
    x: *const f32,
    y: *mut f32,
    n: i32,
    k: i32,
) {
    unsafe {
        let warp_i = (threadIdx_x() >> 5) as usize;
        let lane = (threadIdx_x() & 31) as usize;
        let row = blockIdx_x() as usize * 4 + warp_i;
        if row >= n as usize {
            return;
        }
        let (n, k) = (n as usize, k as usize);
        let w_row = w_data.add(row * (k >> 1));
        let s_row = w_scales.add(row * (k >> 4));
        let n_blocks16 = k >> 4;
        let mut acc = [0.0f32; B];
        let mut blk = lane;
        while blk < n_blocks16 {
            let w8 = crate::ldg_u2(w_row.add(blk * 8));
            let sc = cvt_e4m3_to_float(crate::ldg_u8(s_row.add(blk)));
            let lo = cvt_e2m1x8_to_half2x4(w8[0]);
            let hi = cvt_e2m1x8_to_half2x4(w8[1]);
            let koff = blk << 4;
            let mut b = 0;
            #[unroll]
            while b < B {
                let xf = x.add(b * k + koff);
                let f0 = ldg_f4(xf);
                let f1 = ldg_f4(xf.add(4));
                let f2 = ldg_f4(xf.add(8));
                let f3 = ldg_f4(xf.add(12));
                let mut p0 = mul_f16x2(lo[0], pack_f16x2(f0[0], f0[1]));
                let mut p1 = mul_f16x2(lo[1], pack_f16x2(f0[2], f0[3]));
                p0 = fma_f16x2(lo[2], pack_f16x2(f1[0], f1[1]), p0);
                p1 = fma_f16x2(lo[3], pack_f16x2(f1[2], f1[3]), p1);
                p0 = fma_f16x2(hi[0], pack_f16x2(f2[0], f2[1]), p0);
                p1 = fma_f16x2(hi[1], pack_f16x2(f2[2], f2[3]), p1);
                p0 = fma_f16x2(hi[2], pack_f16x2(f3[0], f3[1]), p0);
                p1 = fma_f16x2(hi[3], pack_f16x2(f3[2], f3[3]), p1);
                let s = add_f16x2(p0, p1);
                acc[b] = fmaf(h2f_lo(s) + h2f_hi(s), sc, acc[b]);
                b += 1;
            }
            blk += 32;
        }
        let mut b = 0;
        #[unroll]
        while b < B {
            let s = mul_rn(warp_sum(acc[b]), global_scale);
            if lane == 0 {
                *y.add(b * n + row) = s;
            }
            b += 1;
        }
    }
}

macro_rules! linear_nvfp4_f32_kernel {
    ($name:ident, $b:expr) => {
        /// `linear_nvfp4_f32_kernel<B, false>`: y[B, N] = x[B, K] W^T with f32 x.
        /// Grid ceil(N / 4), 128 threads (DSpark commit `w_kv_ctx`).
        #[kernel]
        #[launch_bounds(128)]
        pub unsafe fn $name(w_data: *const u8, w_scales: *const u8, global_scale: f32, x: *const f32, y: *mut f32, n: i32, k: i32) {
            unsafe { linear_nvfp4_f32_body::<$b>(w_data, w_scales, global_scale, x, y, n, k) }
        }
    };
}

linear_nvfp4_f32_kernel!(linear_nvfp4_f32_b1, 1);
linear_nvfp4_f32_kernel!(linear_nvfp4_f32_b2, 2);
linear_nvfp4_f32_kernel!(linear_nvfp4_f32_b4, 4);
linear_nvfp4_f32_kernel!(linear_nvfp4_f32_b8, 8);

// ---------------------------------------------------------------------------
// Fused NVFP4 cross-pass controller
// ---------------------------------------------------------------------------

/// `CtrlNVFP4Ptr3` (kernels_inkling.cu): per stream, the NVFP4 data, scales
/// and global scale of the down, forget and gain projections. 192 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CtrlNvfp4Ptr3 {
    pub down_d: [*const u8; 3],
    pub down_s: [*const u8; 3],
    pub down_gs: [f32; 3],
    pub forget_d: [*const u8; 3],
    pub forget_s: [*const u8; 3],
    pub forget_gs: [f32; 3],
    pub gain_d: [*const u8; 3],
    pub gain_s: [*const u8; 3],
    pub gain_gs: [f32; 3],
}

/// `nvfp4_block16_dot(w8, xh)` with the 16 halves read from shared memory.
#[inline(always)]
pub(crate) unsafe fn block16_dot_smem(w8: [u32; 2], xh: *const u32) -> f32 {
    unsafe {
        block16_dot(
            w8,
            [*xh, *xh.add(1), *xh.add(2), *xh.add(3)],
            [*xh.add(4), *xh.add(5), *xh.add(6), *xh.add(7)],
        )
    }
}

/// `__ldg(const uint2*)`.
#[inline(always)]
unsafe fn ldg_u2(p: *const u8) -> [u32; 2] {
    unsafe { crate::ldg_u2(p) }
}

/// `mega_controller_nvfp4_kernel`: the whole NVFP4 cross-pass controller for
/// one (token, stream) in one 1024-thread block: depth attention over
/// (saved, prior_last), cat = [norm(x), norm(msg)] in f16, down projection,
/// forget/gain projections, merge and the post norm. Grid (B, n_streams).
#[kernel]
#[launch_bounds(1024)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mega_controller_nvfp4(
    xs: MutPtr3,
    saved: *const f32,
    prior_last: *const f32,
    cp_queries: Ptr3,
    cws: CtrlNvfp4Ptr3,
    post_norms: Bf16Ptr3,
    outs: MutPtr3,
    outs_h: HalfPtr3,
) {
    static mut S_CAT: SharedArray<u16, { 2 * HIDDEN }, 16> = SharedArray::UNINIT;
    static mut S_MSG: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
    static mut S_M: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
    static mut S_D: SharedArray<u16, { GATE_RANK }, 16> = SharedArray::UNINIT;
    static mut SM0: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut SM1: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut SMX: SharedArray<f32, 32> = SharedArray::UNINIT;
    static mut SCAL: SharedArray<f32, 4> = SharedArray::UNINIT; // w0, w1, inv_x, inv_m
    unsafe {
        let s_cat = SharedArray::as_raw_mut_ptr(&raw mut S_CAT);
        let s_msg = SharedArray::as_raw_mut_ptr(&raw mut S_MSG);
        let s_m = SharedArray::as_raw_mut_ptr(&raw mut S_M);
        let s_d = SharedArray::as_raw_mut_ptr(&raw mut S_D);
        let sm0 = SharedArray::as_raw_mut_ptr(&raw mut SM0);
        let sm1 = SharedArray::as_raw_mut_ptr(&raw mut SM1);
        let smx = SharedArray::as_raw_mut_ptr(&raw mut SMX);
        let scal = SharedArray::as_raw_mut_ptr(&raw mut SCAL);

        let b = blockIdx_x() as usize;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;

        let q = pick3(cp_queries.p, s);
        let sv = saved.add(b * HIDDEN);
        let pl = prior_last.add(b * HIDDEN);
        let xb = pick3(xs.p, s).add(b * HIDDEN) as *const f32;

        // 1. Depth attention logits and norm(x).
        let (mut d0, mut d1, mut xsq) = (0.0f32, 0.0f32, 0.0f32);
        let mut d = tid;
        while d < HIDDEN {
            let qv = *q.add(d);
            d0 = fmaf(qv, *sv.add(d), d0);
            d1 = fmaf(qv, *pl.add(d), d1);
            let xv = *xb.add(d);
            *s_m.add(d) = xv;
            xsq = fmaf(xv, xv, xsq);
            d += 1024;
        }
        d0 = warp_sum(d0);
        d1 = warp_sum(d1);
        xsq = warp_sum(xsq);
        if lane == 0 {
            *sm0.add(warp_i) = d0;
            *sm1.add(warp_i) = d1;
            *smx.add(warp_i) = xsq;
        }
        sync_threads();
        if warp_i == 0 {
            let t0 = warp_sum(*sm0.add(lane));
            let t1 = warp_sum(*sm1.add(lane));
            let tx = warp_sum(*smx.add(lane));
            if lane == 0 {
                let l0 = mul_rn(t0, INV_SQRT_HIDDEN);
                let l1 = mul_rn(t1, INV_SQRT_HIDDEN);
                let mx = fmaxf(l0, l1);
                let e0 = expf(l0 - mx);
                let e1 = expf(l1 - mx);
                let inv = rcp_rn(e0 + e1);
                *scal = e0 * inv;
                *scal.add(1) = e1 * inv;
                *scal.add(2) = rsqrtf(div_rn(tx, HIDDEN as f32) + RMS_EPS);
            }
        }
        sync_threads();

        let ww0 = *scal;
        let ww1 = *scal.add(1);
        let ix = *scal.add(2);
        let mut msq = 0.0f32;
        let mut d = tid;
        while d < HIDDEN {
            let mv = fmaf(ww0, *sv.add(d), ww1 * *pl.add(d));
            *s_msg.add(d) = mv;
            *s_cat.add(d) = float2half(*s_m.add(d) * ix);
            msq = fmaf(mv, mv, msq);
            d += 1024;
        }
        msq = warp_sum(msq);
        if lane == 0 {
            *sm0.add(warp_i) = msq;
        }
        sync_threads();
        if warp_i == 0 {
            let tm = warp_sum(*sm0.add(lane));
            if lane == 0 {
                *scal.add(3) = rsqrtf(div_rn(tm, HIDDEN as f32) + RMS_EPS);
            }
        }
        sync_threads();
        let im = *scal.add(3);
        let mut d = tid;
        while d < HIDDEN {
            *s_cat.add(HIDDEN + d) = float2half(*s_msg.add(d) * im);
            d += 1024;
        }
        sync_threads();

        // 2. Down projection: 64 rows x 4608, two rows per warp.
        let dd = pick3(cws.down_d, s);
        let ds = pick3(cws.down_s, s);
        let dgs = pick3(cws.down_gs, s);
        const D_NBLK: usize = (2 * HIDDEN) >> 4; // 288
        let mut pass = 0;
        #[unroll]
        while pass < 2 {
            let r = warp_i + pass * 32;
            let wr = dd.add(r * HIDDEN);
            let sr = ds.add(r * D_NBLK);
            let mut acc = 0.0f32;
            let mut iter = 0;
            #[unroll]
            while iter < 9 {
                let blk = lane + iter * 32;
                let w8 = ldg_u2(wr.add(blk * 8));
                let sc = cvt_e4m3_to_float(crate::ldg_u8(sr.add(blk)));
                let xh = (s_cat as *const u32).add(blk << 3);
                acc = fmaf(block16_dot_smem(w8, xh), sc, acc);
                iter += 1;
            }
            let val = warp_sum(acc) * dgs;
            if lane == 0 {
                *s_d.add(r) = float2half(val);
            }
            pass += 1;
        }
        sync_threads();

        // 3. Forget / gain projections (4 threads per row, 256 rows at a time), merge.
        let fd = pick3(cws.forget_d, s);
        let fs = pick3(cws.forget_s, s);
        let fgs = pick3(cws.forget_gs, s);
        let gd = pick3(cws.gain_d, s);
        let gs = pick3(cws.gain_s, s);
        let ggs = pick3(cws.gain_gs, s);
        let sub = tid & 3;
        let row_grp = tid >> 2;
        let dh_sub = (s_d as *const u32).add(sub << 3);

        let mut out_sq = 0.0f32;
        let mut iter = 0;
        #[unroll]
        while iter < 9 {
            let c = row_grp + iter * 256;
            let fw8 = ldg_u2(fd.add(c * 32 + sub * 8));
            let gw8 = ldg_u2(gd.add(c * 32 + sub * 8));
            let fsc = cvt_e4m3_to_float(crate::ldg_u8(fs.add(c * 4 + sub)));
            let gsc = cvt_e4m3_to_float(crate::ldg_u8(gs.add(c * 4 + sub)));
            let mut af = mul_rn(block16_dot_smem(fw8, dh_sub), fsc);
            let mut ag = mul_rn(block16_dot_smem(gw8, dh_sub), gsc);
            af += warp::shuffle_xor_f32_sync(0xffff_ffff, af, 1);
            af += warp::shuffle_xor_f32_sync(0xffff_ffff, af, 2);
            ag += warp::shuffle_xor_f32_sync(0xffff_ffff, ag, 1);
            ag += warp::shuffle_xor_f32_sync(0xffff_ffff, ag, 2);
            if sub == 0 {
                let merged = fmaf(1.0 - fast_tanh(af * fgs), *s_m.add(c), fast_tanh(ag * ggs) * *s_msg.add(c));
                *s_m.add(c) = merged;
                out_sq = fmaf(merged, merged, out_sq);
            }
            iter += 1;
        }
        out_sq = warp_sum(out_sq);
        if lane == 0 {
            *sm0.add(warp_i) = out_sq;
        }
        sync_threads();
        if warp_i == 0 {
            let tot = warp_sum(*sm0.add(lane));
            if lane == 0 {
                *scal.add(2) = rsqrtf(div_rn(tot, HIDDEN as f32) + RMS_EPS);
            }
        }
        sync_threads();
        let ir = *scal.add(2);
        let nw = pick3(post_norms.p, s);
        let ob = pick3(outs.p, s).add(b * HIDDEN);
        let oh_base = pick3(outs_h.p, s);
        let ohb = if oh_base.is_null() { core::ptr::null_mut() } else { oh_base.add(b * HIDDEN) };
        let mut c = tid;
        while c < HIDDEN {
            let v = *s_m.add(c) * ir * bf16_to_f32(*nw.add(c));
            *ob.add(c) = v;
            if !ohb.is_null() {
                *ohb.add(c) = float2half(v);
            }
            c += 1024;
        }
    }
}

// ---------------------------------------------------------------------------
// Load-time packing (cuda_abi.cu)
// ---------------------------------------------------------------------------

/// `atomicMax((int*)p, v)` on global memory.
#[inline(always)]
unsafe fn atomic_max_i32(p: *mut i32, v: i32) {
    let _old: u32;
    unsafe {
        ptx_asm!("atom.global.max.s32 %0, [%1], %2;", out("=r") _old, in("l") p, in("r") v, clobber("memory"));
    }
}

/// `fabsf`.
#[inline(always)]
pub(crate) fn fabsf(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7fff_ffff)
}

/// `z_bf16_amax_kernel`: max |x| over n BF16 values into `*out_amax` (which
/// must start at 0) with an integer atomicMax on the float bits.
/// Grid min(ceil(n / 256), 2048), 256 threads.
#[kernel]
pub unsafe fn amax_bf16(x: *const u16, n: u64, out_amax: *mut f32) {
    static mut SMEM: SharedArray<f32, 32> = SharedArray::UNINIT;
    unsafe {
        let smem = SharedArray::as_raw_mut_ptr(&raw mut SMEM);
        let mut local_max = 0.0f32;
        let stride = gridDim_x() as u64 * blockDim_x() as u64;
        let mut i = blockIdx_x() as u64 * blockDim_x() as u64 + threadIdx_x() as u64;
        while i < n {
            local_max = fmaxf(local_max, fabsf(bf16_to_f32(*x.add(i as usize))));
            i += stride;
        }
        local_max = warp_max(local_max);
        let lane = (threadIdx_x() & 31) as usize;
        let warp_i = (threadIdx_x() >> 5) as usize;
        if lane == 0 {
            *smem.add(warp_i) = local_max;
        }
        sync_threads();
        if warp_i == 0 {
            let mut v = if lane < (blockDim_x() >> 5) as usize { *smem.add(lane) } else { 0.0 };
            v = warp_max(v);
            if lane == 0 {
                atomic_max_i32(out_amax as *mut i32, v.to_bits() as i32);
            }
        }
    }
}

/// `z_quantize_e2m1`: round |v| to the FP4 grid by thresholds, with sign.
#[inline(always)]
fn quantize_e2m1(v: f32) -> u8 {
    let sign: u8 = if v < 0.0 { 8 } else { 0 };
    let a = fabsf(v);
    let code: u8 = if a < 0.25 {
        0
    } else if a < 0.75 {
        1
    } else if a < 1.25 {
        2
    } else if a < 1.75 {
        3
    } else if a < 2.50 {
        4
    } else if a < 3.50 {
        5
    } else if a < 5.00 {
        6
    } else {
        7
    };
    sign | code
}

/// `__nv_fp8_e4m3(float)`: `cvt.rn.satfinite.e4m3x2.f32` with the value in the low byte.
#[inline(always)]
pub(crate) fn f32_to_e4m3(x: f32) -> u8 {
    let r: u16;
    unsafe {
        ptx_asm!(
            "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;",
            out("=h") r,
            in("f") 0.0f32,
            in("f") x,
            options(register_only),
        );
    }
    r as u8
}

/// Quantise 16 values to one NVFP4 block: E4M3 block scale (written to
/// `dst_s`) and 8 bytes of FP4 codes (written to `dst_w`). Shared by both
/// packing kernels.
#[device]
unsafe fn quant_block16(vals: &[f32; 16], bmax: f32, gscale: f32, dst_s: *mut u8, dst_w: *mut u8) {
    unsafe {
        let raw_scale = fminf(div_rn(bmax, 6.0 * gscale), 448.0);
        let s_fp8 = f32_to_e4m3(raw_scale);
        *dst_s = s_fp8;
        let s_eff = cvt_e4m3_to_float(s_fp8 as u16) * gscale;
        let inv_s = if s_eff > 1e-30 { rcp_rn(s_eff) } else { 0.0 };
        let mut i = 0;
        #[unroll]
        while i < 8 {
            *dst_w.add(i) = quantize_e2m1(vals[2 * i] * inv_s) | (quantize_e2m1(vals[2 * i + 1] * inv_s) << 4);
            i += 1;
        }
    }
}

/// `z_quant_bf16_to_nvfp4_kernel`: BF16 [N, K] to NVFP4 codes [N, K/2] and
/// E4M3 block scales [N, K/16] under `global_scale` (amax / 2688). One thread
/// per 16-value block. Grid ceil(N*K/16 / 256), 256 threads.
#[kernel]
pub unsafe fn pack_bf16_to_nvfp4(src: *const u16, dst_w: *mut u8, dst_s: *mut u8, n: i32, k: i32, global_scale: f32) {
    unsafe {
        let total_blocks = n as u64 * (k / 16) as u64;
        let bidx = blockIdx_x() as u64 * blockDim_x() as u64 + threadIdx_x() as u64;
        if bidx >= total_blocks {
            return;
        }
        let row_in = src.add((bidx * 16) as usize);
        let mut vals = [0.0f32; 16];
        let mut bmax = 0.0f32;
        let mut i = 0;
        #[unroll]
        while i < 16 {
            vals[i] = bf16_to_f32(*row_in.add(i));
            bmax = fmaxf(bmax, fabsf(vals[i]));
            i += 1;
        }
        if global_scale > 0.0 {
            quant_block16(&vals, bmax, global_scale, dst_s.add(bidx as usize), dst_w.add((bidx * 8) as usize));
        } else {
            // raw_scale = 0 -> E4M3 zero, s_eff = 0, inv_s = 0: every code is +0.
            *dst_s.add(bidx as usize) = f32_to_e4m3(0.0);
            let mut i = 0;
            while i < 8 {
                *dst_w.add((bidx * 8) as usize + i) = quantize_e2m1(vals[2 * i] * 0.0) | (quantize_e2m1(vals[2 * i + 1] * 0.0) << 4);
                i += 1;
            }
        }
    }
}

/// `z_mxfp8_expert_amax_kernel`: per expert, the NVFP4 global scale
/// max(|e4m3 * ue8m0|) / 2688 (at least 1e-12). Grid n_experts, 256 threads.
#[kernel]
pub unsafe fn mxfp8_expert_amax(src_w: *const u8, src_s: *const u8, exp_amax: *mut f32, rows: i32, cols: i32) {
    static mut SMEM: SharedArray<f32, 8> = SharedArray::UNINIT;
    unsafe {
        let smem = SharedArray::as_raw_mut_ptr(&raw mut SMEM);
        let e = blockIdx_x() as usize;
        let expert_elems = rows as usize * cols as usize;
        let ew = src_w.add(e * expert_elems);
        let es = src_s.add(e * (expert_elems / 32));
        let mut lmax = 0.0f32;
        let mut idx = threadIdx_x() as usize;
        while idx < expert_elems {
            let v = fabsf(cvt_e4m3_to_float(*ew.add(idx) as u16) * cvt_ue8m0_to_float(*es.add(idx >> 5)));
            lmax = fmaxf(lmax, v);
            idx += blockDim_x() as usize;
        }
        lmax = warp_max(lmax);
        let lane = (threadIdx_x() & 31) as usize;
        let warp_i = (threadIdx_x() >> 5) as usize;
        if lane == 0 {
            *smem.add(warp_i) = lmax;
        }
        sync_threads();
        if warp_i == 0 {
            let v = warp_max(if lane < (blockDim_x() >> 5) as usize { *smem.add(lane) } else { 0.0 });
            if lane == 0 {
                *exp_amax.add(e) = fmaxf(div_rn(v, 448.0 * 6.0), 1e-12);
            }
        }
    }
}

/// `z_mxfp8_to_nvfp4_expert_kernel`: re-quantise MXFP8 experts (E4M3 codes,
/// UE8M0 scale per 32) to NVFP4 under each expert's global scale. One thread
/// per 16-value block. Grid ceil(n_experts*rows*cols/16 / 256), 256 threads.
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn repack_mxfp8_to_nvfp4(
    src_w: *const u8,
    src_s: *const u8,
    exp_global_scale: *const f32,
    dst_w: *mut u8,
    dst_s: *mut u8,
    n_experts: i32,
    rows: i32,
    cols: i32,
) {
    unsafe {
        let blocks_per_expert = rows as u64 * (cols / 16) as u64;
        let total_blocks = n_experts as u64 * blocks_per_expert;
        let bidx = blockIdx_x() as u64 * blockDim_x() as u64 + threadIdx_x() as u64;
        if bidx >= total_blocks {
            return;
        }
        let e = (bidx / blocks_per_expert) as usize;
        let gscale = *exp_global_scale.add(e);
        let elem_off = (bidx * 16) as usize;
        let mx_scale = cvt_ue8m0_to_float(*src_s.add(elem_off >> 5));
        let mut vals = [0.0f32; 16];
        let mut bmax = 0.0f32;
        let mut i = 0;
        #[unroll]
        while i < 16 {
            vals[i] = cvt_e4m3_to_float(*src_w.add(elem_off + i) as u16) * mx_scale;
            bmax = fmaxf(bmax, fabsf(vals[i]));
            i += 1;
        }
        quant_block16(&vals, bmax, gscale, dst_s.add(bidx as usize), dst_w.add((bidx * 8) as usize));
    }
}
