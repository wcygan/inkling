//! Dense projections (exact mode, BF16 weights).
//!
//! Port of squid5090 @ 3ee093c `linear_bf16_nosmem_t_kernel<B, ADD_RES>`
//! (`cuda/kernels_gemv.cu:602`) as `linear_bf16_nosmem_t_b1` (y = W x) and
//! `linear_bf16_nosmem_t_b1_add` (y += W x). Grid ceil(N / 4), 128 threads: one warp
//! per output row.

use crate::common::*;
use crate::fmaf;
use cuda_device::thread::{blockIdx_x, threadIdx_x};
use cuda_device::{device, kernel, launch_bounds};

#[device]
unsafe fn linear_bf16_body<const B: usize, const ADD_RES: bool>(
    w_data: *const u16,
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
        let k = k as usize;
        let n = n as usize;
        let w_row = w_data.add(row * k) as *const u8;
        let n_vec8 = k >> 3;

        let mut acc = [0.0f32; B];
        let mut v = lane;
        while v < n_vec8 {
            let w8 = ldg_u4(w_row.add(v * 16));
            let (w0x, w0y) = bf16x2_to_f32x2(w8[0]);
            let (w1x, w1y) = bf16x2_to_f32x2(w8[1]);
            let (w2x, w2y) = bf16x2_to_f32x2(w8[2]);
            let (w3x, w3y) = bf16x2_to_f32x2(w8[3]);
            let koff = v << 3;
            let mut b = 0;
            #[unroll]
            while b < B {
                let xf = x.add(b * k + koff);
                let x0 = ldg_f4(xf);
                let x1 = ldg_f4(xf.add(4));
                let mut a = acc[b];
                a = fmaf(w0x, x0[0], a);
                a = fmaf(w0y, x0[1], a);
                a = fmaf(w1x, x0[2], a);
                a = fmaf(w1y, x0[3], a);
                a = fmaf(w2x, x1[0], a);
                a = fmaf(w2y, x1[1], a);
                a = fmaf(w3x, x1[2], a);
                a = fmaf(w3y, x1[3], a);
                acc[b] = a;
                b += 1;
            }
            v += 32;
        }
        let mut b = 0;
        #[unroll]
        while b < B {
            let s = warp_sum(acc[b]);
            if lane == 0 {
                if ADD_RES {
                    *y.add(b * n + row) += s;
                } else {
                    *y.add(b * n + row) = s;
                }
            }
            b += 1;
        }
    }
}

/// `linear_bf16_nosmem_t_kernel<1, false>`: y[N] = W[N, K] x[K].
#[kernel]
#[launch_bounds(128)]
pub unsafe fn linear_bf16_nosmem_t_b1(w_data: *const u16, x: *const f32, y: *mut f32, n: i32, k: i32) {
    unsafe { linear_bf16_body::<1, false>(w_data, x, y, n, k) }
}

/// `linear_bf16_nosmem_t_kernel<1, true>`: y[N] += W[N, K] x[K].
#[kernel]
#[launch_bounds(128)]
pub unsafe fn linear_bf16_nosmem_t_b1_add(w_data: *const u16, x: *const f32, y: *mut f32, n: i32, k: i32) {
    unsafe { linear_bf16_body::<1, true>(w_data, x, y, n, k) }
}

/// `linear_bf16_nosmem_t_kernel<7, false>`: y[7, N] = x[7, K] W^T (DSpark
/// `rnn_w_h` over the 7 draft rows).
#[kernel]
#[launch_bounds(128)]
pub unsafe fn linear_bf16_nosmem_t_b7(w_data: *const u16, x: *const f32, y: *mut f32, n: i32, k: i32) {
    unsafe {
        crate::pdl_wait();
        crate::pdl_trigger();
        linear_bf16_body::<7, false>(w_data, x, y, n, k)
    }
}

macro_rules! linear_bf16_batch {
    ($name:ident, $b:expr) => {
        /// `linear_bf16_nosmem_t_kernel<B, false>`: y[B, N] = x[B, K] W^T (exact-precision dense
        /// linears in a batched / verify step).
        #[kernel]
        #[launch_bounds(128)]
        pub unsafe fn $name(w_data: *const u16, x: *const f32, y: *mut f32, n: i32, k: i32) {
            unsafe { linear_bf16_body::<$b, false>(w_data, x, y, n, k) }
        }
    };
}

linear_bf16_batch!(linear_bf16_nosmem_t_b2, 2);
linear_bf16_batch!(linear_bf16_nosmem_t_b4, 4);
linear_bf16_batch!(linear_bf16_nosmem_t_b8, 8);
