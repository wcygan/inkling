//! B = 2..8 tensor-core NVFP4 kernels for the speculative verify step: the
//! batch is the N = 8 dimension of `mma.m16n8k16`, 16 weight rows per CTA,
//! and SPLITK warps split K.
//!
//! Ports of squid5090 @ 3ee093c `cuda/kernels_gemv.cu`:
//! - `linear_nvfp4_mma_kernel<B, false, K, SPLITK>` (232) as
//!   `linear_nvfp4_mma_b{2,4,8}_k3072` (SPLITK 8, 256 threads) and `_k2304` (SPLITK 6, 192),
//!   plus the DSpark residual variants `<8, true, 3072, 8>` / `<8, true, 9216, 8>` as
//!   `linear_nvfp4_mma_b8_k3072_add` / `linear_nvfp4_mma_b8_k9216_add`;
//! - `qkvrg_fused_nvfp4_mma_kernel<B>` (421) as `qkvrg_fused_nvfp4_mma_b{2,4,8}`;
//! - `lm_head_nvfp4_mma_stage1_kernel<B>` (1033) as `lm_head_nvfp4_mma_stage1_b{2,4,8}`;
//! - `kernels_dspark.cu` `dspark_base_logits_nvfp4_mma_kernel<7>` (510) as
//!   `dspark_base_logits_nvfp4_mma_k7`.

use crate::common::*;
use crate::{
    FINAL_SOFTCAP, HIDDEN, fmaf, LM_HEAD_BLOCKS, Nvfp4Chunk32, decode_nvfp4_chunk32_f16, fast_tanh, float2half, ld_dep_f32,
    ld_dep_u4, load_nvfp4_chunk32, pdl_trigger,
};
use cuda_device::thread::{blockIdx_x, gridDim_x, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{SharedArray, device, kernel, launch_bounds, warp};

const U: usize = 3;

/// The two rows (g, g + 8) of U chunks, zero when the row is past `n`.
#[inline(always)]
unsafe fn load_rows(w_data: *const u8, w_scales: *const u8, cb: usize, sb: usize, ra: usize, rb: usize, n: usize, c_base: usize) -> ([Nvfp4Chunk32; U], [Nvfp4Chunk32; U]) {
    let zero = Nvfp4Chunk32 { q: [0; 4], s: 0 };
    let mut a = [zero; U];
    let mut b = [zero; U];
    let mut u = 0;
    while u < U {
        let c = c_base + 4 * u;
        unsafe {
            if ra < n {
                a[u] = load_nvfp4_chunk32(w_data.add(ra * cb), w_scales.add(ra * sb), c);
            }
            if rb < n {
                b[u] = load_nvfp4_chunk32(w_data.add(rb * cb), w_scales.add(rb * sb), c);
            }
        }
        u += 1;
    }
    (a, b)
}

/// The B operand: 16 half2 words of token g at columns `(c_base + 4u) * 32`, zero past B.
#[inline(always)]
unsafe fn load_x_global<const B: usize>(x_h: *const u16, k: usize, g: usize, c_base: usize) -> [[u32; 16]; U] {
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

/// U chunks of 8 MMAs; `dep` (always 0 at run time) ties the MMAs to the loads.
#[device]
fn mma_chunks(a: &[Nvfp4Chunk32; U], b: &[Nvfp4Chunk32; U], xw: &[[u32; 16]; U], dep: u32, d: &mut [f32; 4]) {
    let mut u = 0;
    #[unroll]
    while u < U {
        let mut fa = [0u32; 16];
        let mut fb = [0u32; 16];
        decode_nvfp4_chunk32_f16(&a[u], &mut fa);
        decode_nvfp4_chunk32_f16(&b[u], &mut fb);
        let mut i = 0;
        #[unroll]
        while i < 8 {
            *d = unsafe {
                mma_m16n8k16_f32_f16(
                    *d,
                    [fa[2 * i], fb[2 * i], fa[2 * i + 1], fb[2 * i + 1]],
                    [xw[u][2 * i] ^ dep, xw[u][2 * i + 1]],
                )
            };
            i += 1;
        }
        u += 1;
    }
}

#[inline(always)]
fn dep_of(a: &[Nvfp4Chunk32; U], b: &[Nvfp4Chunk32; U], xw: Option<&[[u32; 16]; U]>, zero: u32) -> u32 {
    let mut dep = 0u32;
    let mut u = 0;
    while u < U {
        dep ^= a[u].q[0] ^ a[u].q[3] ^ b[u].q[0] ^ b[u].q[3];
        if let Some(x) = xw {
            dep ^= x[u][0];
        }
        u += 1;
    }
    dep & zero
}

/// Store the warp's 16x8 accumulator tile into `cs[warp]`.
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

// ---------------------------------------------------------------------------
// linear_nvfp4_mma_kernel<B, false, K, SPLITK>
// ---------------------------------------------------------------------------

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn linear_mma_body<const B: usize, const ADD_RES: bool, const K: usize, const SPLITK: usize>(
    cs: *mut f32,
    w_data: *const u8,
    w_scales: *const u8,
    global_scale: f32,
    x_h: *const u16,
    y: *mut f32,
    n: i32,
    zero: u32,
) {
    unsafe {
        let total_u = K / (128 * SPLITK);
        let nb = total_u / U;
        let cb = K >> 1;
        let sb = K >> 4;
        let n = n as usize;
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let g = lane >> 2;
        let t = lane & 3;
        let r0 = blockIdx_x() as usize * 16;
        let (ra, rb) = (r0 + g, r0 + g + 8);
        let c_warp = warp_i * (4 * total_u) + t;

        let mut d = [0.0f32; 4];
        let mut jb = 0;
        #[nounroll]
        while jb < nb {
            let c_base = c_warp + jb * (4 * U);
            let (a, b) = load_rows(w_data, w_scales, cb, sb, ra, rb, n, c_base);
            if jb == 0 {
                pdl_wait_fenced(zero);
                pdl_trigger();
            }
            let xw = load_x_global::<B>(x_h, K, g, c_base);
            let dep = dep_of(&a, &b, Some(&xw), zero);
            mma_chunks(&a, &b, &xw, dep, &mut d);
            jb += 1;
        }

        store_tile(cs, warp_i, g, t, &d);
        sync_threads();

        let mut k = tid;
        while k < 16 * B {
            let r = k / B;
            let m = k % B;
            let row = r0 + r;
            if row < n {
                let mut s = 0.0f32;
                let mut w = 0;
                #[unroll]
                while w < SPLITK {
                    s += *cs.add((w * 16 + r) * 8 + m);
                    w += 1;
                }
                if ADD_RES {
                    // nvcc contracts `s *= gs; y += s` into fma(gs, s, y).
                    let yp = y.add(m * n + row);
                    *yp = fmaf(s, global_scale, *yp);
                } else {
                    *y.add(m * n + row) = mul_rn(s, global_scale);
                }
            }
            k += SPLITK * 32;
        }
    }
}

macro_rules! linear_mma_kernel {
    ($name:ident, $b:expr, $add:expr, $k:expr, $splitk:expr, $threads:expr) => {
        /// `linear_nvfp4_mma_kernel<B, ADD_RES, K, SPLITK>`: y[B, N] (+)= x[B, K] W^T.
        /// Grid ceil(N / 16), SPLITK * 32 threads, PDL.
        #[kernel]
        #[launch_bounds($threads)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            w_data: *const u8,
            w_scales: *const u8,
            global_scale: f32,
            x_h: *const u16,
            y: *mut f32,
            n: i32,
            zero: u32,
        ) {
            static mut CS: SharedArray<f32, { $splitk * 16 * 8 }, 16> = SharedArray::UNINIT;
            unsafe {
                linear_mma_body::<$b, $add, $k, $splitk>(
                    SharedArray::as_raw_mut_ptr(&raw mut CS),
                    w_data,
                    w_scales,
                    global_scale,
                    x_h,
                    y,
                    n,
                    zero,
                )
            }
        }
    };
}

linear_mma_kernel!(linear_nvfp4_mma_b2_k3072, 2, false, 3072, 8, 256);
linear_mma_kernel!(linear_nvfp4_mma_b4_k3072, 4, false, 3072, 8, 256);
linear_mma_kernel!(linear_nvfp4_mma_b8_k3072, 8, false, 3072, 8, 256);
linear_mma_kernel!(linear_nvfp4_mma_b2_k2304, 2, false, 2304, 6, 192);
linear_mma_kernel!(linear_nvfp4_mma_b4_k2304, 4, false, 2304, 6, 192);
linear_mma_kernel!(linear_nvfp4_mma_b8_k2304, 8, false, 2304, 6, 192);
// DSpark drafter (8-row block): w_o and w_down add into the residual.
linear_mma_kernel!(linear_nvfp4_mma_b8_k3072_add, 8, true, 3072, 8, 256);
linear_mma_kernel!(linear_nvfp4_mma_b8_k9216_add, 8, true, 9216, 8, 256);

// ---------------------------------------------------------------------------
// qkvrg_fused_nvfp4_mma_kernel<B>
// ---------------------------------------------------------------------------

const QK_SPLITK: usize = 6;

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn qkvrg_mma_body<const B: usize>(
    cs: *mut f32,
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
        const N_Q: usize = QRGS_STRIDE;
        const T_Q: usize = N_Q.div_ceil(16); // 410
        const T_K: usize = KV_DIM / 16; // 64
        const CB: usize = HIDDEN >> 1;
        const SB: usize = HIDDEN >> 4;
        let blk = blockIdx_x() as usize;
        let (w_data, w_scales, gscale, x_h, y_out, r0, n_out) = if blk < T_Q {
            (wq_d, wq_s, q_gs, xq_h, y_qrgs, blk * 16, N_Q)
        } else if blk < T_Q + T_K {
            (wk_d, wk_s, k_gs, xk_h, y_k, (blk - T_Q) * 16, KV_DIM)
        } else {
            (wv_d, wv_s, v_gs, xv_h, y_v, (blk - T_Q - T_K) * 16, KV_DIM)
        };

        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let g = lane >> 2;
        let t = lane & 3;
        let (ra, rb) = (r0 + g, r0 + g + 8);
        let c_base = warp_i * (4 * U) + t;

        let (a, b) = load_rows(w_data, w_scales, CB, SB, ra, rb, n_out, c_base);
        pdl_wait_fenced(zero);
        pdl_trigger();
        let xw = load_x_global::<B>(x_h, HIDDEN, g, c_base);
        let dep = dep_of(&a, &b, Some(&xw), zero);
        let mut d = [0.0f32; 4];
        mma_chunks(&a, &b, &xw, dep, &mut d);

        store_tile(cs, warp_i, g, t, &d);
        sync_threads();

        let mut k = tid;
        while k < 16 * B {
            let r = k / B;
            let m = k % B;
            let row = r0 + r;
            if row < n_out {
                let mut s = 0.0f32;
                let mut w = 0;
                #[unroll]
                while w < QK_SPLITK {
                    s += *cs.add((w * 16 + r) * 8 + m);
                    w += 1;
                }
                *y_out.add(m * n_out + row) = s * gscale;
            }
            k += QK_SPLITK * 32;
        }
    }
}

macro_rules! qkvrg_mma_kernel {
    ($name:ident, $b:expr) => {
        /// `qkvrg_fused_nvfp4_mma_kernel<B>`: QRGS (410 tiles) + K (64) + V (64)
        /// = 538 CTAs of 16 rows, 192 threads, PDL.
        #[kernel]
        #[launch_bounds(192)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
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
            static mut CS: SharedArray<f32, { QK_SPLITK * 16 * 8 }, 16> = SharedArray::UNINIT;
            unsafe {
                qkvrg_mma_body::<$b>(
                    SharedArray::as_raw_mut_ptr(&raw mut CS),
                    wq_d, wq_s, q_gs, wk_d, wk_s, k_gs, wv_d, wv_s, v_gs, xq_h, xk_h, xv_h, y_qrgs, y_k, y_v, zero,
                )
            }
        }
    };
}

qkvrg_mma_kernel!(qkvrg_fused_nvfp4_mma_b2, 2);
qkvrg_mma_kernel!(qkvrg_fused_nvfp4_mma_b4, 4);
qkvrg_mma_kernel!(qkvrg_fused_nvfp4_mma_b8, 8);

// ---------------------------------------------------------------------------
// lm_head_nvfp4_mma_stage1_kernel<B>
// ---------------------------------------------------------------------------

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn lm_head_mma_body<const B: usize>(
    s_xh: *mut u16,
    cs: *mut f32,
    w_data: *const u8,
    w_scales: *const u8,
    global_scale: f32,
    x: *const f32,
    out_logits: *mut f32,
    blk_max_val: *mut f32,
    blk_max_idx: *mut i32,
    sample_cfg: *const f32,
    v: i32,
    zero: u32,
) {
    unsafe {
        let write_logits = !out_logits.is_null() && (sample_cfg.is_null() || *sample_cfg > 0.0);
        const CB: usize = HIDDEN >> 1;
        const SB: usize = HIDDEN >> 4;
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let g = lane >> 2;
        let t = lane & 3;
        let c_base = warp_i * (4 * U) + t;
        let v = v as usize;

        pdl_wait_fenced(zero);
        pdl_trigger();

        let total_x = B * HIDDEN;
        let mut i = tid;
        while i < total_x {
            *s_xh.add(i) = float2half(ld_dep_f32(x.add(i)));
            i += 192;
        }
        sync_threads();

        let mut xw = [[0u32; 16]; U];
        if g < B {
            let mut u = 0;
            while u < U {
                let col = (c_base + 4 * u) * 32;
                let xp = s_xh.add(g * HIDDEN + col) as *const [u32; 4];
                let mut vv = 0;
                while vv < 4 {
                    let q = *xp.add(vv);
                    xw[u][4 * vv] = q[0];
                    xw[u][4 * vv + 1] = q[1];
                    xw[u][4 * vv + 2] = q[2];
                    xw[u][4 * vv + 3] = q[3];
                    vv += 1;
                }
                u += 1;
            }
        }

        let mut best_val = [-1e30f32; B];
        let mut best_idx = [0i32; B];

        let n_tiles = v.div_ceil(16);
        let mut tile = blockIdx_x() as usize;
        while tile < n_tiles {
            let r0 = tile * 16;
            let (ra, rb) = (r0 + g, r0 + g + 8);
            let (a, b) = load_rows(w_data, w_scales, CB, SB, ra, rb, v, c_base);
            let dep = dep_of(&a, &b, None, zero);
            let mut d = [0.0f32; 4];
            mma_chunks(&a, &b, &xw, dep, &mut d);
            store_tile(cs, warp_i, g, t, &d);
            sync_threads();

            if tid < 16 {
                let row = r0 + tid;
                if row < v {
                    let mut bb = 0;
                    #[unroll]
                    while bb < B {
                        let c = |w: usize| *cs.add((w * 16 + tid) * 8 + bb);
                        let s = c(0) + c(1) + c(2) + c(3) + c(4) + c(5);
                        let val = s * global_scale;
                        if write_logits {
                            let cap_val = fast_tanh(val * (1.0 / FINAL_SOFTCAP)) * FINAL_SOFTCAP;
                            *out_logits.add(bb * v + row) = cap_val;
                            if cap_val > best_val[bb] || (cap_val == best_val[bb] && (row as i32) < best_idx[bb]) {
                                best_val[bb] = cap_val;
                                best_idx[bb] = row as i32;
                            }
                        } else if val > best_val[bb] || (val == best_val[bb] && (row as i32) < best_idx[bb]) {
                            best_val[bb] = val;
                            best_idx[bb] = row as i32;
                        }
                        bb += 1;
                    }
                }
            }
            sync_threads();
            tile += gridDim_x() as usize;
        }

        if tid < 16 {
            let mut bb = 0;
            #[unroll]
            while bb < B {
                let mut bv = best_val[bb];
                let mut bi = best_idx[bb];
                let mut off = 8u32;
                #[unroll]
                while off > 0 {
                    let ov = warp::shuffle_xor_f32_sync(0xffff, bv, off);
                    let oi = warp::shuffle_xor_sync(0xffff, bi as u32, off) as i32;
                    if ov > bv || (ov == bv && oi < bi) {
                        bv = ov;
                        bi = oi;
                    }
                    off >>= 1;
                }
                if tid == 0 {
                    *blk_max_val.add(bb * LM_HEAD_BLOCKS + blockIdx_x() as usize) = bv;
                    *blk_max_idx.add(bb * LM_HEAD_BLOCKS + blockIdx_x() as usize) = bi;
                }
                bb += 1;
            }
        }
    }
}

macro_rules! lm_head_mma_kernel {
    ($name:ident, $b:expr) => {
        /// `lm_head_nvfp4_mma_stage1_kernel<B>`: NVFP4 LM head for B tokens with
        /// softcap and per-block argmax. Grid 680, 192 threads, PDL.
        #[kernel]
        #[launch_bounds(192)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            w_data: *const u8,
            w_scales: *const u8,
            global_scale: f32,
            x: *const f32,
            out_logits: *mut f32,
            blk_max_val: *mut f32,
            blk_max_idx: *mut i32,
            sample_cfg: *const f32,
            v: i32,
            zero: u32,
        ) {
            static mut S_XH: SharedArray<u16, { $b * HIDDEN }, 16> = SharedArray::UNINIT;
            static mut CS: SharedArray<f32, { 6 * 16 * 8 }, 16> = SharedArray::UNINIT;
            unsafe {
                lm_head_mma_body::<$b>(
                    SharedArray::as_raw_mut_ptr(&raw mut S_XH),
                    SharedArray::as_raw_mut_ptr(&raw mut CS),
                    w_data, w_scales, global_scale, x, out_logits, blk_max_val, blk_max_idx, sample_cfg, v, zero,
                )
            }
        }
    };
}

lm_head_mma_kernel!(lm_head_nvfp4_mma_stage1_b2, 2);
lm_head_mma_kernel!(lm_head_nvfp4_mma_stage1_b4, 4);
lm_head_mma_kernel!(lm_head_nvfp4_mma_stage1_b8, 8);

// ---------------------------------------------------------------------------
// dspark_base_logits_nvfp4_mma_kernel<K_CONST> (kernels_dspark.cu:510)
// ---------------------------------------------------------------------------

#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn dspark_base_logits_body<const KC: usize>(
    s_xh: *mut u16,
    cs: *mut f32,
    w_data: *const u8,
    w_scales: *const u8,
    gscale: f32,
    normed_h: *const u16,
    fold_b: *const f32,
    out_base: *mut f32,
    v: i32,
    zero: u32,
) {
    unsafe {
        const CB: usize = HIDDEN >> 1;
        const SB: usize = HIDDEN >> 4;
        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let g = lane >> 2;
        let t = lane & 3;
        let c_base = warp_i * (4 * U) + t;
        let v = v as usize;

        pdl_wait_fenced(zero);
        pdl_trigger();

        let mut i = tid;
        while i < KC * HIDDEN {
            *s_xh.add(i) = *normed_h.add(i);
            i += 192;
        }
        sync_threads();

        let mut xw = [[0u32; 16]; U];
        if g < KC {
            let mut u = 0;
            while u < U {
                let col = (c_base + 4 * u) * 32;
                let xp = s_xh.add(g * HIDDEN + col) as *const [u32; 4];
                let mut vv = 0;
                while vv < 4 {
                    let q = *xp.add(vv);
                    xw[u][4 * vv] = q[0];
                    xw[u][4 * vv + 1] = q[1];
                    xw[u][4 * vv + 2] = q[2];
                    xw[u][4 * vv + 3] = q[3];
                    vv += 1;
                }
                u += 1;
            }
        }

        let n_tiles = v.div_ceil(16);
        let mut tile = blockIdx_x() as usize;
        while tile < n_tiles {
            let r0 = tile * 16;
            let (a, b) = load_rows(w_data, w_scales, CB, SB, r0 + g, r0 + g + 8, v, c_base);
            let dep = dep_of(&a, &b, None, zero);
            let mut d = [0.0f32; 4];
            mma_chunks(&a, &b, &xw, dep, &mut d);
            store_tile(cs, warp_i, g, t, &d);
            sync_threads();

            let mut idx = tid;
            while idx < 16 * KC {
                let r = idx / KC;
                let k = idx % KC;
                let row = r0 + r;
                if row < v {
                    let c = |w: usize| *cs.add((w * 16 + r) * 8 + k);
                    let s = c(0) + c(1) + c(2) + c(3) + c(4) + c(5);
                    let fb = if fold_b.is_null() { 0.0 } else { ld_dep_f32(fold_b.add(row)) };
                    // nvcc: fma(gscale, s, fb)
                    *out_base.add(k * v + row) = fmaf(s, gscale, fb);
                }
                idx += 192;
            }
            sync_threads();
            tile += gridDim_x() as usize;
        }
    }
}

/// `dspark_base_logits_nvfp4_mma_kernel<7>`: base_logits[7, V] = unembed_draft
/// @ normed_h[7] + fold_b. Grid 680, 192 threads, PDL; V tiles of 16 rows.
#[kernel]
#[launch_bounds(192)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dspark_base_logits_nvfp4_mma_k7(
    w_data: *const u8,
    w_scales: *const u8,
    gscale: f32,
    normed_h: *const u16,
    fold_b: *const f32,
    out_base: *mut f32,
    v: i32,
    zero: u32,
) {
    static mut S_XH: SharedArray<u16, { 7 * HIDDEN }, 16> = SharedArray::UNINIT;
    static mut CS: SharedArray<f32, { 6 * 16 * 8 }, 16> = SharedArray::UNINIT;
    unsafe {
        dspark_base_logits_body::<7>(
            SharedArray::as_raw_mut_ptr(&raw mut S_XH),
            SharedArray::as_raw_mut_ptr(&raw mut CS),
            w_data, w_scales, gscale, normed_h, fold_b, out_base, v, zero,
        )
    }
}
