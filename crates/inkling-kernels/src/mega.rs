//! Fused residual kernels (576 threads, one float4 of channels per thread).
//!
//! Ports of squid5090 @ 3ee093c `cuda/kernels_inkling.cu`, with the runtime
//! ablation hook (`apply_ablation_576`) and its three `abl_*` parameters left
//! out. With null `abl_r` the hook is the identity, so outputs are unchanged:
//! - `mega_attn_post_mlp_in_kernel` (1945) as `mega_attn_post_mlp_in`;
//! - `mega_mlp_post_res_kernel` (2077) as `mega_mlp_post_res`;
//! - `mega_mlp_post_next_depth_norm_kernel` (2310) as `mega_mlp_post_next_depth_norm`.

use crate::common::*;
use crate::{HIDDEN, TOP_K, fmaf, float2half, pdl_trigger, pdl_wait};
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{SharedArray, device, kernel, launch_bounds};

pub(crate) type F4 = [f32; 4];

/// A float4 with CUDA's 16-byte alignment, so loads and stores vectorise.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct F4A(F4);

#[inline(always)]
pub(crate) unsafe fn ld4(p: *const f32) -> F4 {
    unsafe { (*(p as *const F4A)).0 }
}

#[inline(always)]
pub(crate) unsafe fn st4(p: *mut f32, v: F4) {
    unsafe { *(p as *mut F4A) = F4A(v) }
}

/// `f4_dot`.
#[inline(always)]
pub(crate) fn f4_dot(a: F4, b: F4) -> f32 {
    fmaf(a[0], b[0], fmaf(a[1], b[1], fmaf(a[2], b[2], a[3] * b[3])))
}

/// `store_f4_and_h4`.
#[inline(always)]
pub(crate) unsafe fn store_f4_and_h4(v: F4, dst_f: *mut f32, dst_h: *mut u16) {
    unsafe {
        st4(dst_f, v);
        if !dst_h.is_null() {
            *(dst_h as *mut [u32; 2]) = [pack_f16x2(v[0], v[1]), pack_f16x2(v[2], v[3])];
        }
    }
}

/// `load_moe_f4`: one row, or the sum of its TOP_K slots in fixed order.
#[device]
pub(crate) unsafe fn load_moe_f4(src: *const f32, b: usize, c4: usize, is_slotted: bool) -> F4 {
    unsafe {
        if !is_slotted {
            return ld4(src.add(b * HIDDEN + c4));
        }
        let base = src.add(b * TOP_K * HIDDEN + c4);
        let mut acc = ld4(base);
        let mut s = 1;
        #[unroll]
        while s < TOP_K {
            let v = ld4(base.add(s * HIDDEN));
            acc[0] += v[0];
            acc[1] += v[1];
            acc[2] += v[2];
            acc[3] += v[3];
            s += 1;
        }
        acc
    }
}


/// The four bf16x2-packed tap weights of one float4 of channels (`sconv_w` is
/// constant, so this load may be issued before `pdl_wait`).
#[inline(always)]
pub(crate) unsafe fn sconv_weights(sconv_w: *const u16, c4: usize) -> [u32; 8] {
    unsafe {
        let sw = sconv_w.add(c4 * 4) as *const u8;
        let s0 = ldg_u4(sw);
        let s1 = ldg_u4(sw.add(16));
        [s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]]
    }
}

/// The 4-tap short convolution, given the preloaded weights, the three taps
/// and the current value.
#[inline(always)]
fn sconv_taps(sw: [u32; 8], t0: F4, t1: F4, t2: F4, cur: F4) -> F4 {
    let w0a = bf16x2_to_f32x2(sw[0]);
    let w0b = bf16x2_to_f32x2(sw[1]);
    let w1a = bf16x2_to_f32x2(sw[2]);
    let w1b = bf16x2_to_f32x2(sw[3]);
    let w2a = bf16x2_to_f32x2(sw[4]);
    let w2b = bf16x2_to_f32x2(sw[5]);
    let w3a = bf16x2_to_f32x2(sw[6]);
    let w3b = bf16x2_to_f32x2(sw[7]);
    let ws = [(w0a, w0b), (w1a, w1b), (w2a, w2b), (w3a, w3b)];
    let mut out = [0.0f32; 4];
    let mut i = 0;
    while i < 4 {
        let (wa, wb) = ws[i];
        // A separate `mul.rn`: nvcc never fuses this product into the
        // caller's residual add, and LLVM otherwise would.
        out[i] = mul_rn(fmaf(wa.0, t0[i], fmaf(wa.1, t1[i], fmaf(wb.0, t2[i], wb.1 * cur[i]))) + cur[i], INV_SQRT_5);
        i += 1;
    }
    out
}

/// `sconv_apply_f4_slotted` (and `sconv_apply_f4` when `is_slotted` is false),
/// with the tap weights preloaded.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn sconv_apply_f4_slotted(
    src: *const f32,
    is_slotted: bool,
    sw: [u32; 8],
    ring: *mut f32,
    b: usize,
    p: i32,
    c4: usize,
    write_ring: bool,
) -> F4 {
    unsafe {
        let z4 = [0.0f32; 4];
        let ring_at = |k: i32| ld4(ring.add((((p - k) & 15) as usize) * HIDDEN + c4));
        let cur = load_moe_f4(src, b, c4, is_slotted);
        let t2 = if b >= 1 { load_moe_f4(src, b - 1, c4, is_slotted) } else if p >= 1 { ring_at(1) } else { z4 };
        let t1 = if b >= 2 { load_moe_f4(src, b - 2, c4, is_slotted) } else if p >= 2 { ring_at(2) } else { z4 };
        let t0 = if b >= 3 { load_moe_f4(src, b - 3, c4, is_slotted) } else if p >= 3 { ring_at(3) } else { z4 };
        if write_ring {
            st4(ring.add(((p & 15) as usize) * HIDDEN + c4), cur);
        }
        sconv_taps(sw, t0, t1, t2, cur)
    }
}

/// Block-wide sum of one value per warp over the 18 warps: `warp_sum` of the
/// per-warp partials. Every lane of every warp gets the same bits (the xor
/// butterfly is symmetric), so each warp computes it instead of warp 0
/// broadcasting it through shared memory after a second barrier.
#[inline(always)]
pub(crate) unsafe fn sum18(sm: *const f32, lane: usize) -> f32 {
    unsafe { warp_sum(if lane < 18 { *sm.add(lane) } else { 0.0 }) }
}

/// `__ldg(const uint2*)`.
#[inline(always)]
pub(crate) unsafe fn ld_nc_u2(p: *const u8) -> [u32; 2] {
    unsafe { crate::ldg_u2(p) }
}

/// The constant per-thread inputs of the depth attention: query and norm
/// weight (zero when `norm_w` is null). Loaded before `pdl_wait`.
#[inline(always)]
pub(crate) unsafe fn depth_consts(query: *const f32, norm_w: *const u16, c4: usize) -> (F4, [u32; 2]) {
    unsafe {
        let qv = ld4(query.add(c4));
        let nw = if norm_w.is_null() { [0u32; 2] } else { ld_nc_u2(norm_w.add(c4) as *const u8) };
        (qv, nw)
    }
}

/// Loads `blocks[0..n_blocks)` of this float4 into registers (unrolled, so
/// they never go to local memory), with `sub` at index `sub_at` (-1: none).
#[inline(always)]
pub(crate) unsafe fn load_blocks(blk_b: *const f32, n_load: i32, c4: usize, sub_at: i32, sub: F4) -> [F4; 4] {
    unsafe {
        let mut r = [[0.0f32; 4]; 4];
        let mut c = 0;
        while c < 4 {
            if (c as i32) < n_load {
                r[c] = ld4(blk_b.add(c * HIDDEN + c4));
            } else if c as i32 == sub_at {
                r[c] = sub;
            }
            c += 1;
        }
        r
    }
}

/// Writes the per-warp dot products of the query with the depth blocks into
/// `partials[warp_i * 8 + c]`. They do not depend on the RMS, so they are
/// reduced in the same barrier as the RMS partial.
#[inline(always)]
pub(crate) unsafe fn block_dots(partials: *mut f32, r_blk: &[F4; 4], n_blocks: i32, qv: F4, lane: usize, warp_i: usize) {
    unsafe {
        let mut c = 0;
        while c < 4 {
            if (c as i32) < n_blocks {
                let d = warp_sum(f4_dot(qv, r_blk[c]));
                if lane == 0 {
                    *partials.add(warp_i * 8 + c) = d;
                }
            }
            c += 1;
        }
    }
}

/// Per-warp sum of squares of `v` into `sm[warp_i]` (before a barrier).
#[inline(always)]
pub(crate) unsafe fn sumsq_partial(sm: *mut f32, v: F4, lane: usize, warp_i: usize) {
    unsafe {
        let sum_sq = warp_sum(f4_dot(v, v));
        if lane == 0 {
            *sm.add(warp_i) = sum_sq;
        }
    }
}

/// 1/rms from the 18 per-warp partials in `sm` (after a barrier).
#[inline(always)]
pub(crate) unsafe fn inv_rms_of(sm: *const f32, lane: usize) -> f32 {
    unsafe { rsqrtf(div_rn(sum18(sm, lane), HIDDEN as f32) + RMS_EPS) }
}

/// The depth attention + optional RMSNorm tail shared by
/// `mega_attn_post_mlp_in` and `mega_mlp_post_next_depth_norm`. On entry the
/// block dot products are already in `partials` (before the caller's first
/// barrier); candidates are `r_blk[0..n_blocks)` and `cur`. Every warp
/// computes the softmax and the output RMS itself, so the tail takes two
/// barriers instead of four, with the original arithmetic and order.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn depth_attn_576(
    partials: *mut f32,
    sm_norm: *mut f32,
    r_blk: &[F4; 4],
    n_blocks: i32,
    cur: F4,
    qv: F4,
    has_norm: bool,
    nw: [u32; 2],
    lane: usize,
    warp_i: usize,
) -> F4 {
    unsafe {
        let n_cand = n_blocks + 1;
        {
            let d = warp_sum(f4_dot(qv, cur));
            if lane == 0 {
                *partials.add(warp_i * 8 + n_blocks as usize) = d;
            }
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

        let mut v = [wm * cur[0], wm * cur[1], wm * cur[2], wm * cur[3]];
        let mut c = 0;
        while c < 4 {
            if (c as i32) < n_blocks {
                let wc = w[c];
                let rb = r_blk[c];
                v[0] = fmaf(wc, rb[0], v[0]);
                v[1] = fmaf(wc, rb[1], v[1]);
                v[2] = fmaf(wc, rb[2], v[2]);
                v[3] = fmaf(wc, rb[3], v[3]);
            }
            c += 1;
        }

        if has_norm {
            sumsq_partial(sm_norm, v, lane, warp_i);
            sync_threads();
            let s2 = inv_rms_of(sm_norm, lane);
            let (n0, n1) = bf16x2_to_f32x2(nw[0]);
            let (n2, n3) = bf16x2_to_f32x2(nw[1]);
            v[0] = v[0] * s2 * n0;
            v[1] = v[1] * s2 * n1;
            v[2] = v[2] * s2 * n2;
            v[3] = v[3] * s2 * n3;
        }
        v
    }
}

/// `mega_attn_post_mlp_in_kernel`: attention short conv, residual (or block
/// boundary), RMSNorm, then the MLP-input depth attention and norm.
/// Grid B, 576 threads.
#[kernel]
#[launch_bounds(576)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mega_attn_post_mlp_in(
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
    mlp_norm_w: *const u16,
    mlp_out: *mut f32,
    mlp_out_h: *mut u16,
    d_pos: *const i32,
) {
    static mut SM_RMS: SharedArray<f32, 18> = SharedArray::UNINIT;
    static mut SM_NORM: SharedArray<f32, 18> = SharedArray::UNINIT;
    static mut PARTIALS: SharedArray<f32, { 18 * 8 }> = SharedArray::UNINIT;
    unsafe {
        let sm_rms = SharedArray::as_raw_mut_ptr(&raw mut SM_RMS);
        let sm_norm = SharedArray::as_raw_mut_ptr(&raw mut SM_NORM);
        let partials = SharedArray::as_raw_mut_ptr(&raw mut PARTIALS);

        let b = blockIdx_x() as usize;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let c4 = tid << 2;
        let sw = sconv_weights(attn_sconv_w, c4);
        let (qv, nw) = depth_consts(mlp_query, mlp_norm_w, c4);
        pdl_wait();
        pdl_trigger();
        let p = *d_pos + b as i32;
        let n_blocks = block_count_before + if is_boundary { 1 } else { 0 };

        let blk_b = blocks.add(b * 4 * HIDDEN);
        let pp = if is_boundary { ld4(partial_prev.add(b * HIDDEN + c4)) } else { [0.0f32; 4] };
        let r_blk = load_blocks(blk_b, block_count_before, c4, if is_boundary { block_count_before } else { -1 }, pp);
        let ao = sconv_apply_f4_slotted(o_proj, false, sw, attn_ring, b, p, c4, true);
        let rv = if is_boundary {
            st4(blk_b.add(block_count_before as usize * HIDDEN + c4), pp);
            ao
        } else {
            let hv = ld4(hidden.add(b * HIDDEN + c4));
            [hv[0] + ao[0], hv[1] + ao[1], hv[2] + ao[2], hv[3] + ao[3]]
        };
        st4(residual_out.add(b * HIDDEN + c4), rv);

        sumsq_partial(sm_rms, rv, lane, warp_i);
        block_dots(partials, &r_blk, n_blocks, qv, lane, warp_i);
        sync_threads();
        let ir = inv_rms_of(sm_rms, lane);
        let r_mp = [rv[0] * ir, rv[1] * ir, rv[2] * ir, rv[3] * ir];

        let has_norm = !mlp_norm_w.is_null();
        let v = depth_attn_576(partials, sm_norm, &r_blk, n_blocks, r_mp, qv, has_norm, nw, lane, warp_i);

        let mo = mlp_out.add(b * HIDDEN + c4);
        if has_norm {
            let moh = if mlp_out_h.is_null() { core::ptr::null_mut() } else { mlp_out_h.add(b * HIDDEN + c4) };
            store_f4_and_h4(v, mo, moh);
        } else {
            st4(mo, v);
        }
    }
}

/// `mega_mlp_post_res_kernel`: MLP short conv over the expert output (or the
/// fixed-order sum of its 8 slots) plus the residual. Grid B, 576 threads.
#[kernel]
#[launch_bounds(576)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mega_mlp_post_res(
    moe_down_or_slots: *const f32,
    moe_is_slotted: bool,
    mlp_sconv_w: *const u16,
    mlp_ring: *mut f32,
    residual: *const f32,
    hidden_out: *mut f32,
    d_pos: *const i32,
) {
    unsafe {
        let b = blockIdx_x() as usize;
        let tid = threadIdx_x() as usize;
        let c4 = tid << 2;
        let sw = sconv_weights(mlp_sconv_w, c4);
        pdl_wait();
        pdl_trigger();
        let p = *d_pos + b as i32;
        let ff = sconv_apply_f4_slotted(moe_down_or_slots, moe_is_slotted, sw, mlp_ring, b, p, c4, true);
        let res = ld4(residual.add(b * HIDDEN + c4));
        st4(hidden_out.add(b * HIDDEN + c4), [res[0] + ff[0], res[1] + ff[1], res[2] + ff[2], res[3] + ff[3]]);
    }
}

/// `mega_mlp_post_next_depth_norm_kernel`: the previous visit's MLP post
/// (short conv + residual, with taps), then the next visit's attention-input
/// depth attention and norm for stream s. Grid (B, n_out), 576 threads.
#[kernel]
#[launch_bounds(576)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mega_mlp_post_next_depth_norm(
    moe_down_or_slots: *const f32,
    moe_is_slotted: bool,
    prev_mlp_sconv_w: *const u16,
    prev_mlp_ring: *mut f32,
    residual: *const f32,
    hidden_out: *mut f32,
    pre_tap_out: *mut f32,
    post_tap_out: *mut f32,
    partial_out: *mut f32,
    blocks: *const f32,
    n_blocks: i32,
    queries: *const f32,
    norms: Bf16Ptr3,
    outs: MutPtr3,
    outs_h: HalfPtr3,
    d_pos: *const i32,
) {
    static mut SM_RMS: SharedArray<f32, 18> = SharedArray::UNINIT;
    static mut SM_NORM: SharedArray<f32, 18> = SharedArray::UNINIT;
    static mut PARTIALS: SharedArray<f32, { 18 * 8 }> = SharedArray::UNINIT;
    unsafe {
        let sm_rms = SharedArray::as_raw_mut_ptr(&raw mut SM_RMS);
        let sm_norm = SharedArray::as_raw_mut_ptr(&raw mut SM_NORM);
        let partials = SharedArray::as_raw_mut_ptr(&raw mut PARTIALS);

        let b = blockIdx_x() as usize;
        let s = blockIdx_y() as usize;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let c4 = tid << 2;
        let sw = sconv_weights(prev_mlp_sconv_w, c4);
        let nwp = pick3(norms.p, s);
        let (qv, nw) = depth_consts(queries.add(s * HIDDEN), nwp, c4);
        pdl_wait();
        pdl_trigger();
        let p = *d_pos + b as i32;

        let blk_b = blocks.add(b * 4 * HIDDEN);
        let r_blk = load_blocks(blk_b, n_blocks, c4, -1, [0.0f32; 4]);
        let ff = sconv_apply_f4_slotted(moe_down_or_slots, moe_is_slotted, sw, prev_mlp_ring, b, p, c4, s == 0);
        let res = ld4(residual.add(b * HIDDEN + c4));
        let hv = [res[0] + ff[0], res[1] + ff[1], res[2] + ff[2], res[3] + ff[3]];
        if s == 0 {
            if !hidden_out.is_null() {
                st4(hidden_out.add(b * HIDDEN + c4), hv);
            }
            if !pre_tap_out.is_null() {
                st4(pre_tap_out.add(b * HIDDEN + c4), hv);
            }
        }

        sumsq_partial(sm_rms, hv, lane, warp_i);
        block_dots(partials, &r_blk, n_blocks, qv, lane, warp_i);
        sync_threads();
        let ir = inv_rms_of(sm_rms, lane);
        let r_part = [hv[0] * ir, hv[1] * ir, hv[2] * ir, hv[3] * ir];
        if s == 0 && !partial_out.is_null() {
            st4(partial_out.add(b * HIDDEN + c4), r_part);
        }

        let v = depth_attn_576(partials, sm_norm, &r_blk, n_blocks, r_part, qv, !nwp.is_null(), nw, lane, warp_i);

        let ob = pick3(outs.p, s).add(b * HIDDEN + c4);
        let oh = if pick3(outs_h.p, s).is_null() { core::ptr::null_mut() } else { pick3(outs_h.p, s).add(b * HIDDEN + c4) };
        store_f4_and_h4(v, ob, oh);
        if s == 0 && !post_tap_out.is_null() {
            st4(post_tap_out.add(b * HIDDEN + c4), v);
        }
    }
}
