//! `mega_controller_nvfp4_cl8`: the NVFP4 cross-pass controller spread over a cluster of 8 CTAs per
//! (token, stream) instead of one 1024-thread CTA.
//!
//! Same parameters and semantics as `mega_controller_nvfp4`; only the launch changes: grid
//! (8 * B, n_streams), 288 threads, cluster (8, 1, 1) fixed at compile time (`.reqnctapercluster`), so a
//! plain launch works. Per CTA (rank r):
//! 1. Redundantly (identical in every CTA): depth-attention weights, norm(x), msg, norm(msg) and the f16
//!    `cat` = [norm(x), norm(msg)] (27 KB read from L2 per CTA).
//! 2. Down rows 8r..8r+8 (one warp each, K = 4608), exchanged through distributed shared memory.
//! 3. Forget/gain rows 288r..288r+288 (4 threads per row) and the merge.
//! 4. Post RMSNorm: the 8 partial sums of squares are read from every rank in rank order (deterministic),
//!    then each CTA writes its rows.
//!
//! Not bit-identical to `mega_controller_nvfp4` (reduction orders differ); checked against it with a
//! ULP / relative-error tolerance in kernel-parity's `v2` group.
//!
//! Under a PDL launch the constant inputs (query, NVFP4 weights and scales, post norm) are loaded into
//! registers before `griddepcontrol.wait`; under a plain launch the wait is a no-op.
//!
//! Two fused variants (same cluster shape) run the depth-norm / post kernel that feeds a controller
//! inside the controller, so on a loop visit the pair costs one launch instead of two:
//! - `mega_post_next_ctrl_cl8` = `mega_mlp_post_next_depth_norm` (prior visits, no norms) + the
//!   attention controller;
//! - `mega_attn_post_ctrl_cl8` = `mega_attn_post_mlp_in` (prior visits, no norm) + the MLP controller.
//!
//! Every CTA of a cluster computes the post / depth step for its (token, stream) redundantly, with two
//! float4s per thread laid out as the 576-thread kernels lay out one: float4 `tid + 288 j` belongs to
//! that kernel's warp `warp + 9 j`. The per-warp partials, their order and every product are the same,
//! so the depth output is bit-identical to the 576-thread kernel. Only rank 0 of stream 0 writes the
//! side effects (ring, residual / hidden, taps, partial, boundary block, depth output). The controller
//! then reads x from shared memory instead of global memory.

use crate::common::*;
use crate::mega::{F4, block_dots, f4_dot, inv_rms_of, ld4, load_blocks, sconv_apply_f4_slotted, sconv_weights, st4, sumsq_partial};
use crate::nvfp4::{CtrlNvfp4Ptr3, block16_dot_smem};
use crate::{HIDDEN, cvt_e4m3_to_float, fast_tanh, fmaf, float2half, pdl_trigger, pdl_wait};
use cuda_device::cluster;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{SharedArray, cluster_launch, kernel, launch_bounds, warp};

const G: usize = 8; // CTAs per cluster
const THREADS: usize = 288; // 9 warps
const NW: usize = THREADS / 32;
const ROWS: usize = HIDDEN / G; // 288 forget/gain rows per CTA
const DOWN_ROWS: usize = GATE_RANK / G; // 8 down rows per CTA
const D_NBLK: usize = (2 * HIDDEN) >> 4;

/// Shared memory of the controller.
struct Smem {
    s_cat: *mut u16,
    s_x: *mut f32,
    s_msg: *mut f32,
    s_dloc: *mut u16,
    s_d: *mut u16,
    s_merged: *mut f32,
    sm0: *mut f32,
    sm1: *mut f32,
    smx: *mut f32,
    scal: *mut f32, // w0, w1, inv_x, inv_m, part_sq, inv_out
}

macro_rules! cl8_smem {
    () => {{
        static mut S_CAT: SharedArray<u16, { 2 * HIDDEN }, 16> = SharedArray::UNINIT;
        static mut S_X: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
        static mut S_MSG: SharedArray<f32, { HIDDEN }, 16> = SharedArray::UNINIT;
        static mut S_DLOC: SharedArray<u16, { DOWN_ROWS }, 16> = SharedArray::UNINIT;
        static mut S_D: SharedArray<u16, { GATE_RANK }, 16> = SharedArray::UNINIT;
        static mut S_MERGED: SharedArray<f32, { ROWS }, 16> = SharedArray::UNINIT;
        static mut SM0: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut SM1: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut SMX: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut SCAL: SharedArray<f32, 8> = SharedArray::UNINIT;
        Smem {
            s_cat: SharedArray::as_raw_mut_ptr(&raw mut S_CAT),
            s_x: SharedArray::as_raw_mut_ptr(&raw mut S_X),
            s_msg: SharedArray::as_raw_mut_ptr(&raw mut S_MSG),
            s_dloc: SharedArray::as_raw_mut_ptr(&raw mut S_DLOC),
            s_d: SharedArray::as_raw_mut_ptr(&raw mut S_D),
            s_merged: SharedArray::as_raw_mut_ptr(&raw mut S_MERGED),
            sm0: SharedArray::as_raw_mut_ptr(&raw mut SM0),
            sm1: SharedArray::as_raw_mut_ptr(&raw mut SM1),
            smx: SharedArray::as_raw_mut_ptr(&raw mut SMX),
            scal: SharedArray::as_raw_mut_ptr(&raw mut SCAL),
        }
    }};
}

/// Shared memory of the fused post / depth step (the 576-thread kernels' reduction buffers).
macro_rules! post_smem {
    () => {{
        static mut SM_RMS: SharedArray<f32, 18> = SharedArray::UNINIT;
        static mut PARTIALS: SharedArray<f32, { 18 * 8 }> = SharedArray::UNINIT;
        (SharedArray::as_raw_mut_ptr(&raw mut SM_RMS), SharedArray::as_raw_mut_ptr(&raw mut PARTIALS))
    }};
}

/// The controller's constant per-thread inputs.
#[derive(Clone, Copy)]
struct Pre {
    qreg: [f32; HIDDEN / THREADS],
    dw8: [[u32; 2]; 9],
    dsc: [u16; 9],
    fw: [[u32; 2]; 4],
    gw: [[u32; 2]; 4],
    fsb: [u16; 4],
    gsb: [u16; 4],
    nwv: f32,
}

#[inline(always)]
unsafe fn prefetch(cp_queries: &Ptr3, cws: &CtrlNvfp4Ptr3, post_norms: &Bf16Ptr3, rank: usize, s: usize, tid: usize) -> Pre {
    unsafe {
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let q = pick3(cp_queries.p, s);
        let mut pre = Pre {
            qreg: [0.0; HIDDEN / THREADS],
            dw8: [[0; 2]; 9],
            dsc: [0; 9],
            fw: [[0; 2]; 4],
            gw: [[0; 2]; 4],
            fsb: [0; 4],
            gsb: [0; 4],
            nwv: 0.0,
        };
        let mut i = 0;
        while i < HIDDEN / THREADS {
            pre.qreg[i] = *q.add(tid + i * THREADS);
            i += 1;
        }
        if warp_i < DOWN_ROWS {
            let r = rank * DOWN_ROWS + warp_i;
            let wr = pick3(cws.down_d, s).add(r * HIDDEN);
            let sr = pick3(cws.down_s, s).add(r * D_NBLK);
            let mut iter = 0;
            while iter < 9 {
                let blk = lane + iter * 32;
                pre.dw8[iter] = crate::ldg_u2(wr.add(blk * 8));
                pre.dsc[iter] = crate::ldg_u8(sr.add(blk));
                iter += 1;
            }
        }
        let sub = tid & 3;
        let row_grp = tid >> 2;
        let base = rank * ROWS;
        let (fd, fs, gd, gs) = (pick3(cws.forget_d, s), pick3(cws.forget_s, s), pick3(cws.gain_d, s), pick3(cws.gain_s, s));
        let mut pass = 0;
        while pass < 4 {
            let c = base + row_grp + pass * (THREADS / 4);
            pre.fw[pass] = crate::ldg_u2(fd.add(c * 32 + sub * 8));
            pre.gw[pass] = crate::ldg_u2(gd.add(c * 32 + sub * 8));
            pre.fsb[pass] = crate::ldg_u8(fs.add(c * 4 + sub));
            pre.gsb[pass] = crate::ldg_u8(gs.add(c * 4 + sub));
            pass += 1;
        }
        pre.nwv = bf16_to_f32(*pick3(post_norms.p, s).add(base + tid));
        pre
    }
}

/// The controller after `pdl_wait`. `xb` null: x is already in `s_x` (and the caller has synced).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn run(
    m: &Smem,
    pre: &Pre,
    xb: *const f32,
    saved: *const f32,
    prior_last: *const f32,
    cws: &CtrlNvfp4Ptr3,
    outs: &MutPtr3,
    outs_h: &HalfPtr3,
    rank: usize,
    b: usize,
    s: usize,
    tid: usize,
) {
    unsafe {
        let (s_cat, s_x, s_msg, s_dloc, s_d, s_merged) = (m.s_cat, m.s_x, m.s_msg, m.s_dloc, m.s_d, m.s_merged);
        let (sm0, sm1, smx, scal) = (m.sm0, m.sm1, m.smx, m.scal);
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let sub = tid & 3;
        let row_grp = tid >> 2;
        let base = rank * ROWS;
        let sv = saved.add(b * HIDDEN);
        let pl = prior_last.add(b * HIDDEN);

        // 1. Depth-attention logits, norm(x), msg, norm(msg): identical in every CTA.
        let (mut d0, mut d1, mut xsq) = (0.0f32, 0.0f32, 0.0f32);
        let mut i = 0;
        while i < HIDDEN / THREADS {
            let d = tid + i * THREADS;
            let qv = pre.qreg[i];
            d0 = fmaf(qv, *sv.add(d), d0);
            d1 = fmaf(qv, *pl.add(d), d1);
            let xv = if xb.is_null() {
                *s_x.add(d)
            } else {
                let v = *xb.add(d);
                *s_x.add(d) = v;
                v
            };
            xsq = fmaf(xv, xv, xsq);
            i += 1;
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
            let mut w = 0;
            while w < NW {
                t0 += *sm0.add(w);
                t1 += *sm1.add(w);
                tx += *smx.add(w);
                w += 1;
            }
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
        sync_threads();
        let (ww0, ww1, ix) = (*scal, *scal.add(1), *scal.add(2));
        let mut msq = 0.0f32;
        let mut d = tid;
        while d < HIDDEN {
            let mv = fmaf(ww0, *sv.add(d), mul_rn(ww1, *pl.add(d)));
            *s_msg.add(d) = mv;
            *s_cat.add(d) = float2half(*s_x.add(d) * ix);
            msq = fmaf(mv, mv, msq);
            d += THREADS;
        }
        msq = warp_sum(msq);
        if lane == 0 {
            *sm0.add(warp_i) = msq;
        }
        sync_threads();
        if tid == 0 {
            let mut tm = 0.0f32;
            let mut w = 0;
            while w < NW {
                tm += *sm0.add(w);
                w += 1;
            }
            *scal.add(3) = rsqrtf(div_rn(tm, HIDDEN as f32) + RMS_EPS);
        }
        sync_threads();
        let im = *scal.add(3);
        let mut d = tid;
        while d < HIDDEN {
            *s_cat.add(HIDDEN + d) = float2half(*s_msg.add(d) * im);
            d += THREADS;
        }
        sync_threads();

        // 2. Down rows 8*rank .. 8*rank + 8: one warp per row, K = 4608 (288 blocks of 16).
        if warp_i < DOWN_ROWS {
            let dgs = pick3(cws.down_gs, s);
            let mut acc = 0.0f32;
            let mut iter = 0;
            while iter < 9 {
                let blk = lane + iter * 32;
                let sc = cvt_e4m3_to_float(pre.dsc[iter]);
                let xh = (s_cat as *const u32).add(blk << 3);
                acc = fmaf(block16_dot_smem(pre.dw8[iter], xh), sc, acc);
                iter += 1;
            }
            let val = warp_sum(acc) * dgs;
            if lane == 0 {
                *s_dloc.add(warp_i) = float2half(val);
            }
        }
        cluster::cluster_sync();
        if tid < GATE_RANK {
            let src = cluster::map_shared_rank(s_dloc as *const u16, (tid / DOWN_ROWS) as u32);
            *s_d.add(tid) = *src.add(tid % DOWN_ROWS);
        }
        sync_threads();

        // 3. Forget / gain rows 288*rank .. +288 (4 threads per row, 72 rows per pass), merge.
        let fgs = pick3(cws.forget_gs, s);
        let ggs = pick3(cws.gain_gs, s);
        let dh_sub = (s_d as *const u32).add(sub << 3);
        let mut out_sq = 0.0f32;
        let mut pass = 0;
        while pass < 4 {
            let lr = row_grp + pass * (THREADS / 4);
            let c = base + lr;
            let fsc = cvt_e4m3_to_float(pre.fsb[pass]);
            let gsc = cvt_e4m3_to_float(pre.gsb[pass]);
            let mut af = mul_rn(block16_dot_smem(pre.fw[pass], dh_sub), fsc);
            let mut ag = mul_rn(block16_dot_smem(pre.gw[pass], dh_sub), gsc);
            af += warp::shuffle_xor_f32_sync(0xffff_ffff, af, 1);
            af += warp::shuffle_xor_f32_sync(0xffff_ffff, af, 2);
            ag += warp::shuffle_xor_f32_sync(0xffff_ffff, ag, 1);
            ag += warp::shuffle_xor_f32_sync(0xffff_ffff, ag, 2);
            if sub == 0 {
                let merged = fmaf(1.0 - fast_tanh(af * fgs), *s_x.add(c), fast_tanh(ag * ggs) * *s_msg.add(c));
                *s_merged.add(lr) = merged;
                out_sq = fmaf(merged, merged, out_sq);
            }
            pass += 1;
        }
        out_sq = warp_sum(out_sq);
        if lane == 0 {
            *sm0.add(warp_i) = out_sq;
        }
        sync_threads();
        if tid == 0 {
            let mut t = 0.0f32;
            let mut w = 0;
            while w < NW {
                t += *sm0.add(w);
                w += 1;
            }
            *scal.add(4) = t;
        }

        // 4. Post RMSNorm over the whole row: partial sums from every rank, in rank order.
        cluster::cluster_sync();
        if tid == 0 {
            let mut tot = 0.0f32;
            let mut r = 0;
            while r < G {
                tot += *cluster::map_shared_rank(scal.add(4) as *const f32, r as u32);
                r += 1;
            }
            *scal.add(5) = rsqrtf(div_rn(tot, HIDDEN as f32) + RMS_EPS);
        }
        sync_threads();
        let ir = *scal.add(5);
        let ob = pick3(outs.p, s).add(b * HIDDEN);
        let oh_base = pick3(outs_h.p, s);
        let ohb = if oh_base.is_null() { core::ptr::null_mut() } else { oh_base.add(b * HIDDEN) };
        // ROWS == THREADS: one row per thread.
        let c = base + tid;
        let v = *s_merged.add(tid) * ir * pre.nwv;
        *ob.add(c) = v;
        if !ohb.is_null() {
            *ohb.add(c) = float2half(v);
        }
    }
}

#[kernel]
#[cluster_launch(8, 1, 1)]
#[launch_bounds(288)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mega_controller_nvfp4_cl8(
    xs: MutPtr3,
    saved: *const f32,
    prior_last: *const f32,
    cp_queries: Ptr3,
    cws: CtrlNvfp4Ptr3,
    post_norms: Bf16Ptr3,
    outs: MutPtr3,
    outs_h: HalfPtr3,
) {
    unsafe {
        let m = cl8_smem!();
        let rank = cluster::cluster_ctaidX() as usize;
        let b = blockIdx_x() as usize / G;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let pre = prefetch(&cp_queries, &cws, &post_norms, rank, s, tid);
        pdl_wait();
        pdl_trigger();
        let xb = pick3(xs.p, s).add(b * HIDDEN) as *const f32;
        run(&m, &pre, xb, saved, prior_last, &cws, &outs, &outs_h, rank, b, s, tid);
        // No CTA may exit while another still reads its shared memory.
        cluster::cluster_sync();
    }
}

/// `mega_controller_nvfp4_cl8` for the wide prefill: each cluster runs `tb` tokens one after another
/// (tokens `cluster * tb ..`, below `t`) with the NVFP4 weights, query and post norm loaded into registers
/// once, instead of once per token. Per token the code is `run`, so the outputs are bit-identical.
/// Grid (8 * ceil(t / tb), n_streams), 288 threads.
#[kernel]
#[cluster_launch(8, 1, 1)]
#[launch_bounds(288)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_controller_nvfp4_cl8(
    xs: MutPtr3,
    saved: *const f32,
    prior_last: *const f32,
    cp_queries: Ptr3,
    cws: CtrlNvfp4Ptr3,
    post_norms: Bf16Ptr3,
    outs: MutPtr3,
    outs_h: HalfPtr3,
    tb: i32,
    t: i32,
) {
    unsafe {
        let m = cl8_smem!();
        let rank = cluster::cluster_ctaidX() as usize;
        let c0 = (blockIdx_x() as usize / G) * tb as usize;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let pre = prefetch(&cp_queries, &cws, &post_norms, rank, s, tid);
        let mut i = 0;
        while i < tb as usize && c0 + i < t as usize {
            let b = c0 + i;
            let xb = pick3(xs.p, s).add(b * HIDDEN) as *const f32;
            run(&m, &pre, xb, saved, prior_last, &cws, &outs, &outs_h, rank, b, s, tid);
            // Every rank has read this token's partial sums before the next token overwrites them.
            cluster::cluster_sync();
            i += 1;
        }
    }
}

/// The depth attention without a norm (loop visits) for this thread's two float4s, with the block
/// dot products already in `partials`; writes the output into `s_x` (and `out`, when non-null).
/// Same arithmetic and reduction order as `mega::depth_attn_576` without a norm.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn depth_tail_2x(
    partials: *mut f32,
    s_x: *mut f32,
    out: *mut f32,
    r_blk: &[[F4; 4]; 2],
    n_blocks: i32,
    cur: [F4; 2],
    qv: [F4; 2],
    tid: usize,
) {
    unsafe {
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let n_cand = n_blocks + 1;
        let mut j = 0;
        while j < 2 {
            let d = warp_sum(f4_dot(qv[j], cur[j]));
            if lane == 0 {
                *partials.add((warp_i + 9 * j) * 8 + n_blocks as usize) = d;
            }
            j += 1;
        }
        sync_threads();

        let mut logits = [0.0f32; 5];
        let mut mx = -1e30f32;
        let mut c = 0;
        while c < 5 {
            if (c as i32) < n_cand {
                let dot = warp_sum(if lane < 18 { *partials.add(lane * 8 + c) } else { 0.0 });
                logits[c] = mul_rn(dot, INV_SQRT_HIDDEN);
                mx = fmaxf(mx, logits[c]);
            }
            c += 1;
        }
        let mut w = [0.0f32; 5];
        let mut den = 0.0f32;
        let mut c = 0;
        while c < 5 {
            if (c as i32) < n_cand {
                let e = expf(logits[c] - mx);
                w[c] = e;
                den += e;
            }
            c += 1;
        }
        let inv_d = rcp_rn(den);
        let mut wm = 0.0f32;
        let mut c = 0;
        while c < 5 {
            w[c] *= inv_d;
            if c as i32 == n_blocks {
                wm = w[c];
            }
            c += 1;
        }

        let mut j = 0;
        while j < 2 {
            let cj = cur[j];
            let mut v = [wm * cj[0], wm * cj[1], wm * cj[2], wm * cj[3]];
            let mut c = 0;
            while c < 4 {
                if (c as i32) < n_blocks {
                    let wc = w[c];
                    let rb = r_blk[j][c];
                    v[0] = fmaf(wc, rb[0], v[0]);
                    v[1] = fmaf(wc, rb[1], v[1]);
                    v[2] = fmaf(wc, rb[2], v[2]);
                    v[3] = fmaf(wc, rb[3], v[3]);
                }
                c += 1;
            }
            let c4 = (tid + THREADS * j) << 2;
            st4(s_x.add(c4), v);
            if !out.is_null() {
                st4(out.add(c4), v);
            }
            j += 1;
        }
    }
}

/// `mega_mlp_post_next_depth_norm` on a loop visit (no norms) fused with the attention controller.
/// Grid (8B, n_streams), 288 threads, clusters of 8. The depth step uses query stream `s` when
/// `depth_streams` is 3 (global layer: x_s = attn_in[s]) and stream 0 for every s when it is 1 (local
/// layer: x = mlp_in for all three). `depth_out` (mlp_in on a local layer, else null) receives the
/// depth output from rank 0 of stream 0; the controller output goes to `outs` as in the cl8 kernel.
#[kernel]
#[cluster_launch(8, 1, 1)]
#[launch_bounds(288)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mega_post_next_ctrl_cl8(
    moe_down: *const f32,
    prev_mlp_sconv_w: *const u16,
    prev_mlp_ring: *mut f32,
    residual: *const f32,
    hidden_out: *mut f32,
    pre_tap_out: *mut f32,
    partial_out: *mut f32,
    blocks: *const f32,
    n_blocks: i32,
    queries: *const f32,
    depth_streams: i32,
    depth_out: *mut f32,
    saved: *const f32,
    prior_last: *const f32,
    cp_queries: Ptr3,
    cws: CtrlNvfp4Ptr3,
    post_norms: Bf16Ptr3,
    outs: MutPtr3,
    outs_h: HalfPtr3,
    d_pos: *const i32,
) {
    unsafe {
        let m = cl8_smem!();
        let (sm_rms, partials) = post_smem!();
        let rank = cluster::cluster_ctaidX() as usize;
        let b = blockIdx_x() as usize / G;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let ds = if depth_streams == 1 { 0 } else { s };
        let c4s = [tid << 2, (tid + THREADS) << 2];
        let sw = [sconv_weights(prev_mlp_sconv_w, c4s[0]), sconv_weights(prev_mlp_sconv_w, c4s[1])];
        let qv = [ld4(queries.add(ds * HIDDEN + c4s[0])), ld4(queries.add(ds * HIDDEN + c4s[1]))];
        let pre = prefetch(&cp_queries, &cws, &post_norms, rank, s, tid);
        pdl_wait();
        pdl_trigger();
        let p = *d_pos + b as i32;
        let writer = rank == 0 && s == 0;

        let blk_b = blocks.add(b * 4 * HIDDEN);
        let mut r_blk = [[[0.0f32; 4]; 4]; 2];
        let mut hv = [[0.0f32; 4]; 2];
        let mut j = 0;
        while j < 2 {
            let c4 = c4s[j];
            r_blk[j] = load_blocks(blk_b, n_blocks, c4, -1, [0.0f32; 4]);
            let ff = sconv_apply_f4_slotted(moe_down, false, sw[j], prev_mlp_ring, b, p, c4, writer);
            let res = ld4(residual.add(b * HIDDEN + c4));
            hv[j] = [res[0] + ff[0], res[1] + ff[1], res[2] + ff[2], res[3] + ff[3]];
            if writer {
                if !hidden_out.is_null() {
                    st4(hidden_out.add(b * HIDDEN + c4), hv[j]);
                }
                if !pre_tap_out.is_null() {
                    st4(pre_tap_out.add(b * HIDDEN + c4), hv[j]);
                }
            }
            j += 1;
        }
        let mut j = 0;
        while j < 2 {
            sumsq_partial(sm_rms, hv[j], lane, warp_i + 9 * j);
            block_dots(partials, &r_blk[j], n_blocks, qv[j], lane, warp_i + 9 * j);
            j += 1;
        }
        sync_threads();
        let ir = inv_rms_of(sm_rms, lane);
        let mut r_part = [[0.0f32; 4]; 2];
        let mut j = 0;
        while j < 2 {
            let h = hv[j];
            r_part[j] = [h[0] * ir, h[1] * ir, h[2] * ir, h[3] * ir];
            if writer && !partial_out.is_null() {
                st4(partial_out.add(b * HIDDEN + c4s[j]), r_part[j]);
            }
            j += 1;
        }
        let dout = if writer && !depth_out.is_null() { depth_out.add(b * HIDDEN) } else { core::ptr::null_mut() };
        depth_tail_2x(partials, m.s_x, dout, &r_blk, n_blocks, r_part, qv, tid);
        sync_threads();

        run(&m, &pre, core::ptr::null(), saved, prior_last, &cws, &outs, &outs_h, rank, b, s, tid);
        cluster::cluster_sync();
    }
}

/// `mega_attn_post_mlp_in` on a loop visit (no MLP norm) fused with the MLP controller.
/// Grid (8B, 1), 288 threads, clusters of 8. `mlp_out` (mlp_in) receives the depth output from rank 0.
#[kernel]
#[cluster_launch(8, 1, 1)]
#[launch_bounds(288)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mega_attn_post_ctrl_cl8(
    o_proj: *const f32,
    attn_sconv_w: *const u16,
    attn_ring: *mut f32,
    hidden: *const f32,
    partial_prev: *const f32,
    blocks: *mut f32,
    block_count_before: i32,
    is_boundary: bool,
    residual_out: *mut f32,
    mlp_query: *const f32,
    mlp_out: *mut f32,
    saved: *const f32,
    prior_last: *const f32,
    cp_queries: Ptr3,
    cws: CtrlNvfp4Ptr3,
    post_norms: Bf16Ptr3,
    outs: MutPtr3,
    outs_h: HalfPtr3,
    d_pos: *const i32,
) {
    unsafe {
        let m = cl8_smem!();
        let (sm_rms, partials) = post_smem!();
        let rank = cluster::cluster_ctaidX() as usize;
        let b = blockIdx_x() as usize / G;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let c4s = [tid << 2, (tid + THREADS) << 2];
        let sw = [sconv_weights(attn_sconv_w, c4s[0]), sconv_weights(attn_sconv_w, c4s[1])];
        let qv = [ld4(mlp_query.add(c4s[0])), ld4(mlp_query.add(c4s[1]))];
        let pre = prefetch(&cp_queries, &cws, &post_norms, rank, s, tid);
        pdl_wait();
        pdl_trigger();
        let p = *d_pos + b as i32;
        let writer = rank == 0 && s == 0;
        let n_blocks = block_count_before + if is_boundary { 1 } else { 0 };

        let blk_b = blocks.add(b * 4 * HIDDEN);
        let mut r_blk = [[[0.0f32; 4]; 4]; 2];
        let mut rv = [[0.0f32; 4]; 2];
        let mut j = 0;
        while j < 2 {
            let c4 = c4s[j];
            let pp = if is_boundary { ld4(partial_prev.add(b * HIDDEN + c4)) } else { [0.0f32; 4] };
            r_blk[j] = load_blocks(blk_b, block_count_before, c4, if is_boundary { block_count_before } else { -1 }, pp);
            let ao = sconv_apply_f4_slotted(o_proj, false, sw[j], attn_ring, b, p, c4, writer);
            rv[j] = if is_boundary {
                if writer {
                    st4(blk_b.add(block_count_before as usize * HIDDEN + c4), pp);
                }
                ao
            } else {
                let hv = ld4(hidden.add(b * HIDDEN + c4));
                [hv[0] + ao[0], hv[1] + ao[1], hv[2] + ao[2], hv[3] + ao[3]]
            };
            if writer {
                st4(residual_out.add(b * HIDDEN + c4), rv[j]);
            }
            j += 1;
        }
        let mut j = 0;
        while j < 2 {
            sumsq_partial(sm_rms, rv[j], lane, warp_i + 9 * j);
            block_dots(partials, &r_blk[j], n_blocks, qv[j], lane, warp_i + 9 * j);
            j += 1;
        }
        sync_threads();
        let ir = inv_rms_of(sm_rms, lane);
        let mut r_mp = [[0.0f32; 4]; 2];
        let mut j = 0;
        while j < 2 {
            let r = rv[j];
            r_mp[j] = [r[0] * ir, r[1] * ir, r[2] * ir, r[3] * ir];
            j += 1;
        }
        let dout = if writer && !mlp_out.is_null() { mlp_out.add(b * HIDDEN) } else { core::ptr::null_mut() };
        depth_tail_2x(partials, m.s_x, dout, &r_blk, n_blocks, r_mp, qv, tid);
        sync_threads();

        run(&m, &pre, core::ptr::null(), saved, prior_last, &cws, &outs, &outs_h, rank, b, s, tid);
        cluster::cluster_sync();
    }
}
