//! Exact-mode cross-pass controllers (BF16 weights), all three attention
//! streams in one launch each.
//!
//! Ports of squid5090 @ 3ee093c `cuda/kernels_inkling.cu`:
//! - `ctrl_prep_cat_batched_kernel` (287) as `ctrl_prep_cat_batched`;
//! - `ctrl_down_batched_kernel<1>` (364) as `ctrl_down_batched_b1`;
//! - `ctrl_up_merge_batched_kernel` (410) as `ctrl_up_merge_batched`;
//! - `batched_rmsnorm_kernel` (476) as `batched_rmsnorm`.

use crate::common::*;
use crate::{HIDDEN, fast_tanh, fmaf, float2half};
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{SharedArray, device, kernel, launch_bounds, warp};

/// `ctrl_prep_cat_batched_kernel`: msg = 2-way depth attention over
/// (saved, prior_last) with query s; cat = [norm(x), norm(msg)].
/// Grid (B, n_streams), 256 threads.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn ctrl_prep_cat_batched(
    xs: MutPtr3,
    saved: *const f32,
    prior_last: *const f32,
    cp_queries: Ptr3,
    out_msg: *mut f32,
    out_cat: *mut f32,
    bsz: i32,
) {
    static mut SM0: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SM1: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut SMX: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut W01X: SharedArray<f32, 3> = SharedArray::UNINIT;
    static mut INV_M: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sm0 = SharedArray::as_raw_mut_ptr(&raw mut SM0);
        let sm1 = SharedArray::as_raw_mut_ptr(&raw mut SM1);
        let smx = SharedArray::as_raw_mut_ptr(&raw mut SMX);
        let w01x = SharedArray::as_raw_mut_ptr(&raw mut W01X);
        let inv_m = SharedArray::as_raw_mut_ptr(&raw mut INV_M);

        let b = blockIdx_x() as usize;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let bsz = bsz as usize;

        let q = pick3(cp_queries.p, s);
        let sv = saved.add(b * HIDDEN);
        let pl = prior_last.add(b * HIDDEN);
        let xb = pick3(xs.p, s).add(b * HIDDEN) as *const f32;

        let (mut d0, mut d1, mut xsq) = (0.0f32, 0.0f32, 0.0f32);
        let mut d = tid;
        while d < HIDDEN {
            let qv = *q.add(d);
            d0 = fmaf(qv, *sv.add(d), d0);
            d1 = fmaf(qv, *pl.add(d), d1);
            let xv = *xb.add(d);
            xsq = fmaf(xv, xv, xsq);
            d += 256;
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

        if tid == 0 {
            let (mut t0, mut t1, mut tx) = (0.0f32, 0.0f32, 0.0f32);
            let mut i = 0;
            while i < 8 {
                t0 += *sm0.add(i);
                t1 += *sm1.add(i);
                tx += *smx.add(i);
                i += 1;
            }
            let l0 = mul_rn(t0, INV_SQRT_HIDDEN);
            let l1 = mul_rn(t1, INV_SQRT_HIDDEN);
            let mx = fmaxf(l0, l1);
            let e0 = expf(l0 - mx);
            let e1 = expf(l1 - mx);
            let inv = rcp_rn(e0 + e1);
            *w01x = e0 * inv;
            *w01x.add(1) = e1 * inv;
            *w01x.add(2) = rsqrtf(div_rn(tx, HIDDEN as f32) + RMS_EPS);
        }
        sync_threads();

        let ww0 = *w01x;
        let ww1 = *w01x.add(1);
        let ix = *w01x.add(2);
        let msg_b = out_msg.add((s * bsz + b) * HIDDEN);
        let cat_b = out_cat.add((s * bsz + b) * (2 * HIDDEN));

        let mut msq = 0.0f32;
        let mut d = tid;
        while d < HIDDEN {
            let mv = fmaf(ww0, *sv.add(d), ww1 * *pl.add(d));
            *msg_b.add(d) = mv;
            *cat_b.add(d) = *xb.add(d) * ix;
            msq = fmaf(mv, mv, msq);
            d += 256;
        }
        msq = warp_sum(msq);
        if lane == 0 {
            *sm0.add(warp_i) = msq;
        }
        sync_threads();
        if tid == 0 {
            let mut tm = 0.0f32;
            let mut i = 0;
            while i < 8 {
                tm += *sm0.add(i);
                i += 1;
            }
            *inv_m = rsqrtf(div_rn(tm, HIDDEN as f32) + RMS_EPS);
        }
        sync_threads();
        let im = *inv_m;
        let mut d = tid;
        while d < HIDDEN {
            *cat_b.add(HIDDEN + d) = *msg_b.add(d) * im;
            d += 256;
        }
    }
}

/// `ctrl_down_batched_kernel<B>`: d[s, b, 64] = down_w[s] @ cat[s, b, 4608].
#[device]
unsafe fn ctrl_down_batched_body<const B: usize>(cws: CtrlPtr3, cat: *const f32, out_d: *mut f32) {
    unsafe {
        let warp_i = (threadIdx_x() >> 5) as usize;
        let lane = (threadIdx_x() & 31) as usize;
        let row = blockIdx_x() as usize * 4 + warp_i;
        let s = blockIdx_y() as usize;
        if row >= GATE_RANK {
            return;
        }
        const K: usize = 2 * HIDDEN;
        let wr = pick3(cws.down, s).add(row * K) as *const u8;

        let mut acc = [0.0f32; B];
        let mut v = lane;
        while v < (K >> 3) {
            let w8 = ldg_u4(wr.add(v * 16));
            let (w0x, w0y) = bf16x2_to_f32x2(w8[0]);
            let (w1x, w1y) = bf16x2_to_f32x2(w8[1]);
            let (w2x, w2y) = bf16x2_to_f32x2(w8[2]);
            let (w3x, w3y) = bf16x2_to_f32x2(w8[3]);
            let koff = v << 3;
            let mut b = 0;
            #[unroll]
            while b < B {
                let xf = cat.add((s * B + b) * K + koff);
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
            let val = warp_sum(acc[b]);
            if lane == 0 {
                *out_d.add((s * B + b) * GATE_RANK + row) = val;
            }
            b += 1;
        }
    }
}

/// `ctrl_down_batched_kernel<1>`. Grid (16, n_streams), 128 threads.
#[kernel]
#[launch_bounds(128)]
pub unsafe fn ctrl_down_batched_b1(cws: CtrlPtr3, cat: *const f32, out_d: *mut f32) {
    unsafe { ctrl_down_batched_body::<1>(cws, cat, out_d) }
}

/// `ctrl_up_merge_batched_kernel`: out = (1 - tanh(forget_w @ d)) * x +
/// tanh(gain_w @ d) * msg. Grid (36, n_streams), 256 threads: 4 threads per
/// output row, 64 rows per block.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn ctrl_up_merge_batched(
    cws: CtrlPtr3,
    xs: MutPtr3,
    msg: *const f32,
    d_in: *const f32,
    outs: MutPtr3,
    bsz: i32,
) {
    static mut S_D: SharedArray<f32, { MAX_VERIFY_B * GATE_RANK }, 16> = SharedArray::UNINIT;
    unsafe {
        let s_d = SharedArray::as_raw_mut_ptr(&raw mut S_D);
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let bsz = bsz as usize;
        let mut b = 0;
        while b < bsz {
            if tid < GATE_RANK {
                *s_d.add(b * GATE_RANK + tid) = *d_in.add((s * bsz + b) * GATE_RANK + tid);
            }
            b += 1;
        }
        sync_threads();

        let row_in_blk = tid >> 2;
        let sub = tid & 3;
        let row = blockIdx_x() as usize * 64 + row_in_blk;
        if row >= HIDDEN {
            return;
        }

        let wf = pick3(cws.forget, s).add(row * GATE_RANK) as *const u8;
        let wg = pick3(cws.gain, s).add(row * GATE_RANK) as *const u8;
        let f0 = ldg_u4(wf.add(sub * 32));
        let f1 = ldg_u4(wf.add(sub * 32 + 16));
        let g0 = ldg_u4(wg.add(sub * 32));
        let g1 = ldg_u4(wg.add(sub * 32 + 16));
        let fw = [f0[0], f0[1], f0[2], f0[3], f1[0], f1[1], f1[2], f1[3]];
        let gw = [g0[0], g0[1], g0[2], g0[3], g1[0], g1[1], g1[2], g1[3]];
        let mut ff = [(0.0f32, 0.0f32); 8];
        let mut gg = [(0.0f32, 0.0f32); 8];
        let mut i = 0;
        #[unroll]
        while i < 8 {
            ff[i] = bf16x2_to_f32x2(fw[i]);
            gg[i] = bf16x2_to_f32x2(gw[i]);
            i += 1;
        }

        let koff = sub * 16;
        let mut b = 0;
        while b < bsz {
            let db = s_d.add(b * GATE_RANK + koff) as *const f32;
            let (mut af, mut ag) = (0.0f32, 0.0f32);
            let mut i = 0;
            #[unroll]
            while i < 8 {
                let d0 = *db.add(2 * i);
                let d1 = *db.add(2 * i + 1);
                af = fmaf(ff[i].0, d0, fmaf(ff[i].1, d1, af));
                ag = fmaf(gg[i].0, d0, fmaf(gg[i].1, d1, ag));
                i += 1;
            }
            af += warp::shuffle_xor_f32_sync(0xffff_ffff, af, 1);
            af += warp::shuffle_xor_f32_sync(0xffff_ffff, af, 2);
            ag += warp::shuffle_xor_f32_sync(0xffff_ffff, ag, 1);
            ag += warp::shuffle_xor_f32_sync(0xffff_ffff, ag, 2);
            if sub == 0 {
                let xv = *pick3(xs.p, s).add(b * HIDDEN + row);
                let mv = *msg.add((s * bsz + b) * HIDDEN + row);
                *pick3(outs.p, s).add(b * HIDDEN + row) = fmaf(1.0 - fast_tanh(af), xv, fast_tanh(ag) * mv);
            }
            b += 1;
        }
    }
}

/// `batched_rmsnorm_kernel`: in-place RMSNorm of n_streams rows with an
/// optional half copy. Grid (B, n_streams), 256 threads.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn batched_rmsnorm(xs: MutPtr3, xs_h: HalfPtr3, norms: Bf16Ptr3) {
    static mut SM: SharedArray<f32, 8> = SharedArray::UNINIT;
    static mut INV_RMS: SharedArray<f32, 1> = SharedArray::UNINIT;
    unsafe {
        let sm = SharedArray::as_raw_mut_ptr(&raw mut SM);
        let inv_rms = SharedArray::as_raw_mut_ptr(&raw mut INV_RMS);
        let b = blockIdx_x() as usize;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let xb = pick3(xs.p, s).add(b * HIDDEN);
        let xhb = if pick3(xs_h.p, s).is_null() { core::ptr::null_mut() } else { pick3(xs_h.p, s).add(b * HIDDEN) };
        let nw = pick3(norms.p, s);
        let mut sq = 0.0f32;
        let mut d = tid;
        while d < HIDDEN {
            let v = *xb.add(d);
            sq = fmaf(v, v, sq);
            d += 256;
        }
        sq = warp_sum(sq);
        if (tid & 31) == 0 {
            *sm.add(tid >> 5) = sq;
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
        let mut d = tid;
        while d < HIDDEN {
            let v = *xb.add(d) * ir * bf16_to_f32(*nw.add(d));
            *xb.add(d) = v;
            if !xhb.is_null() {
                *xhb.add(d) = float2half(v);
            }
            d += 256;
        }
    }
}
