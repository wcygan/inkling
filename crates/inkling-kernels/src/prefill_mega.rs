//! Wide prefill (16-256 tokens per step): the residual mega kernels of `mega.rs` without their ring
//! writes (see `prefill_attn.rs`), plus the small helpers of the wide path. The MoE output may be
//! slotted ([T][TOP_K][HIDDEN], summed in slot order). Same arithmetic as the originals.

use crate::common::*;
use crate::mega::*;
use crate::{HIDDEN, pdl_trigger, pdl_wait};
use cuda_device::thread::{blockDim_x, blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{SharedArray, kernel, launch_bounds};

#[kernel]
#[launch_bounds(576)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_attn_post_mlp_in(
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
        let ao = sconv_apply_f4_slotted(o_proj, false, sw, attn_ring, b, p, c4, false);
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

#[kernel]
#[launch_bounds(576)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_mlp_post_res(
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
        let ff = sconv_apply_f4_slotted(moe_down_or_slots, moe_is_slotted, sw, mlp_ring, b, p, c4, false);
        let res = ld4(residual.add(b * HIDDEN + c4));
        st4(hidden_out.add(b * HIDDEN + c4), [res[0] + ff[0], res[1] + ff[1], res[2] + ff[2], res[3] + ff[3]]);
    }
}

/// `mega_mlp_post_next_depth_norm_kernel`: the previous visit's MLP post
/// (short conv + residual, with taps), then the next visit's attention-input

#[kernel]
#[launch_bounds(576)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_mlp_post_next_depth_norm(
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
        let ff = sconv_apply_f4_slotted(moe_down_or_slots, moe_is_slotted, sw, prev_mlp_ring, b, p, c4, false);
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

/// Writes the short-conv ring slots of the chunk's last `min(t, 16)` positions: ring[(p0 + b) & 15] =
/// row b of `src` (the fixed-order sum of its TOP_K slots when `slotted`). `width` is HIDDEN or KV_DIM.
/// Grid min(t, 16), 256 threads.
#[kernel]
pub unsafe fn prefill_ring_store(src: *const f32, slotted: bool, width: i32, ring: *mut f32, t: i32, d_pos: *const i32) {
    unsafe {
        let n = if t < 16 { t } else { 16 };
        let b = (t - n) as usize + blockIdx_x() as usize;
        let p = *d_pos + b as i32;
        let w = width as usize;
        let dst = ring.add(((p & 15) as usize) * w);
        let mut c4 = threadIdx_x() as usize * 4;
        while c4 < w {
            let v = if slotted { load_moe_f4(src, b, c4, true) } else { ld4(src.add(b * w + c4)) };
            st4(dst.add(c4), v);
            c4 += 1024;
        }
    }
}

/// dst[i] = bf16(src[i]) for i < n (n a multiple of 4). Grid ceil(n / 1024), 256 threads.
#[kernel]
pub unsafe fn prefill_f32_to_bf16(src: *const f32, dst: *mut u16, n: i32) {
    unsafe {
        let i = (blockIdx_x() as usize * 256 + threadIdx_x() as usize) * 4;
        if i < n as usize {
            let v = ld4(src.add(i));
            *(dst.add(i) as *mut [u16; 4]) = [f32_to_bf16(v[0]), f32_to_bf16(v[1]), f32_to_bf16(v[2]), f32_to_bf16(v[3])];
        }
    }
}

/// out[j] = *d_pos + 8 j for j < n: the start positions of the chunk's 8-token attention sub-steps.
/// Grid 1, 256 threads.
#[kernel]
pub unsafe fn prefill_sub_pos(d_pos: *const i32, out: *mut i32, n: i32) {
    unsafe {
        let mut j = threadIdx_x() as i32;
        let p = *d_pos;
        while j < n {
            *out.add(j as usize) = p + 8 * j;
            j += blockDim_x() as i32;
        }
    }
}
