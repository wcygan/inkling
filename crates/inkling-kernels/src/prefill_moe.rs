//! Wide prefill MoE (16-256 tokens per step, NVFP4 experts): top-8 selection per token, grouping of the
//! (token, slot) pairs by expert, and grouped tensor-core gate/up and down projections that read each
//! expert's weights once per step instead of once per 8 tokens.
//!
//! Pairs are numbered `i = token * TOP_K + slot`. `prefill_moe_group` writes `offs[e]..offs[e + 1]`, the
//! range of expert e's pairs in `pairs` (the order inside one expert is arbitrary: every pair is computed
//! on its own, so the results do not depend on it). The activation `act` is stored in that sorted order
//! ([P][INTERMEDIATE] f16); the down projection writes `slots[i]` ([T * TOP_K][HIDDEN] f32, already
//! times the route weight), which the wide MLP-post kernels sum in slot order.
//!
//! The MMA layout is `moe_up_tc_nvfp4`'s: weight rows are the M = 16 side (rows g and g + 8 of the tile),
//! the expert's pairs are the N = 8 columns, and each warp owns a 128-wide K slice (chunk c = 4 warp + t
//! of 32 values, consistently permuted in A and B).

use crate::common::*;
use crate::{
    GATE_SOFTCAP, HIDDEN, INTERMEDIATE, MOE_TILE, TOP_K, UP_SOFTCAP, cvt_e2m1x8_to_half2x4, cvt_e4m3x2_to_half2,
    fast_sigmoid, fast_tanh, float2half, high2half2, ld_scale_u16, ld_stream_u4, low2half2,
};
use cuda_device::f16x2::mul_f16x2;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{SharedArray, device, kernel, launch_bounds, ptx_asm, warp};

const UP_K: usize = HIDDEN;
const UP_R: usize = 2 * INTERMEDIATE;
const UP_CB: usize = UP_K >> 1;
const UP_SB: usize = UP_K >> 4;
const UP_WARPS: usize = UP_K / 128; // 18
const DN_K: usize = INTERMEDIATE;
const DN_CB: usize = DN_K >> 1;
const DN_SB: usize = DN_K >> 4;
const DN_WARPS: usize = DN_K / 128; // 9
const OUT_STRIDE: usize = 9;

/// `__ldg(const float*)`.
#[inline(always)]
unsafe fn ldg_f32(p: *const f32) -> f32 {
    let v: f32;
    unsafe {
        ptx_asm!("ld.global.nc.f32 %0, [%1];", out("=f") v, in("l") p, options(register_only));
    }
    v
}

/// Shared-memory `atomicAdd(uint32_t*)`, returning the old value.
#[inline(always)]
unsafe fn atomic_add_u32(p: *mut u32, v: u32) -> u32 {
    let addr = unsafe { cuda_device::shared::cvta_generic_to_shared_u32(p as *const u8) };
    let old: u32;
    unsafe {
        ptx_asm!("atom.shared.add.u32 %0, [%1], %2;", out("=r") old, in("r") addr, in("r") v, clobber("memory"));
    }
    old
}

/// One 32-weight NVFP4 chunk (16 code bytes, two E4M3 scales) decoded to 16 f16 pairs, as
/// `moe.rs`'s `load_chunk::<0>` + `decode_nvfp4`.
#[device]
unsafe fn load_decode(codes: *const u8, scales: *const u8, c: usize, o: &mut [u32; 16]) {
    unsafe {
        let q = ld_stream_u4(codes.add(c * 16));
        let sc = cvt_e4m3x2_to_half2(ld_scale_u16(scales.add(c * 2)));
        let slo = low2half2(sc);
        let shi = high2half2(sc);
        let mut w = 0;
        #[unroll]
        while w < 4 {
            let h = cvt_e2m1x8_to_half2x4(q[w]);
            let sm = if w < 2 { slo } else { shi };
            o[4 * w] = mul_f16x2(h[0], sm);
            o[4 * w + 1] = mul_f16x2(h[1], sm);
            o[4 * w + 2] = mul_f16x2(h[2], sm);
            o[4 * w + 3] = mul_f16x2(h[3], sm);
            w += 1;
        }
    }
}

/// 32 f16 activations of chunk c of `row` (zeros for a null row).
#[device]
unsafe fn load_x(row: *const u16, c: usize) -> [u32; 16] {
    let mut x = [0u32; 16];
    if !row.is_null() {
        unsafe {
            let xp = row.add(c * 32);
            let mut v = 0;
            #[unroll]
            while v < 4 {
                let q = ldg_u4(xp.add(v * 8) as *const u8);
                x[4 * v] = q[0];
                x[4 * v + 1] = q[1];
                x[4 * v + 2] = q[2];
                x[4 * v + 3] = q[3];
                v += 1;
            }
        }
    }
    x
}

#[device]
fn mma8(a: &[u32; 16], b: &[u32; 16], x: &[u32; 16]) -> [f32; 4] {
    let mut d = [0.0f32; 4];
    let mut i = 0;
    #[unroll]
    while i < 8 {
        d = unsafe { mma_m16n8k16_f32_f16(d, [a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1]], [x[2 * i], x[2 * i + 1]]) };
        i += 1;
    }
    d
}

/// Top-8 experts per token (`route_select_and_group`'s arithmetic): sigmoid scores, selection by score +
/// bias (ties to the lower expert), weights normalised to ROUTE_SCALE. One warp per token; grid
/// ceil(t / 8), 256 threads.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn prefill_route_select(logits: *const f32, router_bias: *const f32, out_ids: *mut i32, out_weights: *mut f32, t: i32) {
    unsafe {
        let b = blockIdx_x() as usize * 8 + (threadIdx_x() >> 5) as usize;
        let lane = (threadIdx_x() & 31) as usize;
        if b >= t as usize {
            return;
        }
        let mut sc = [0.0f32; 4];
        let mut bi = [0.0f32; 4];
        let mut i = 0;
        #[unroll]
        while i < 4 {
            let e = lane + 32 * i;
            let raw = fast_sigmoid(*logits.add(b * NUM_EXPERTS + e));
            sc[i] = raw;
            bi[i] = raw + ldg_f32(router_bias.add(e));
            i += 1;
        }
        let mut winners = [0i32; TOP_K];
        let mut chosen = [0.0f32; TOP_K];
        let mut k = 0;
        #[unroll]
        while k < TOP_K {
            let mut best = -1e30f32;
            let mut best_e = NUM_EXPERTS as i32;
            let mut i = 0;
            #[unroll]
            while i < 4 {
                let e = (lane + 32 * i) as i32;
                if bi[i] > best || (bi[i] == best && e < best_e) {
                    best = bi[i];
                    best_e = e;
                }
                i += 1;
            }
            let top = warp_max(best);
            let mut cand = if best == top { best_e } else { NUM_EXPERTS as i32 };
            let mut off = 16u32;
            while off > 0 {
                let o = warp::shuffle_xor_sync(0xffff_ffff, cand as u32, off) as i32;
                cand = if o < cand { o } else { cand };
                off >>= 1;
            }
            let mut sel = 0.0f32;
            let mut i = 0;
            #[unroll]
            while i < 4 {
                if (lane + 32 * i) as i32 == cand {
                    sel = sc[i];
                    bi[i] = -1e30;
                }
                i += 1;
            }
            chosen[k] = warp_sum(sel);
            winners[k] = cand;
            k += 1;
        }
        if lane == 0 {
            let mut total = 0.0f32;
            let mut k = 0;
            #[unroll]
            while k < TOP_K {
                total += chosen[k];
                k += 1;
            }
            let inv = div_rn(ROUTE_SCALE, total + 1e-20);
            let mut k = 0;
            #[unroll]
            while k < TOP_K {
                *out_ids.add(b * TOP_K + k) = winners[k];
                *out_weights.add(b * TOP_K + k) = chosen[k] * inv;
                k += 1;
            }
        }
    }
}

/// Groups the `npairs` (token, slot) pairs by expert: `offs[0..=128]` and `pairs[offs[e]..offs[e + 1])`.
/// Grid 1, 1024 threads.
#[kernel]
#[launch_bounds(1024)]
pub unsafe fn prefill_moe_group(ids: *const i32, npairs: i32, offs: *mut i32, pairs: *mut i32) {
    static mut CNT: SharedArray<u32, { NUM_EXPERTS }> = SharedArray::UNINIT;
    static mut OFF: SharedArray<u32, { NUM_EXPERTS }> = SharedArray::UNINIT;
    unsafe {
        let cnt = SharedArray::as_raw_mut_ptr(&raw mut CNT);
        let off = SharedArray::as_raw_mut_ptr(&raw mut OFF);
        let tid = threadIdx_x() as usize;
        let n = npairs as usize;
        if tid < NUM_EXPERTS {
            *cnt.add(tid) = 0;
        }
        sync_threads();
        let mut i = tid;
        while i < n {
            atomic_add_u32(cnt.add(*ids.add(i) as usize), 1);
            i += 1024;
        }
        sync_threads();
        if tid == 0 {
            let mut acc = 0u32;
            let mut e = 0;
            while e < NUM_EXPERTS {
                *offs.add(e) = acc as i32;
                *off.add(e) = acc;
                acc += *cnt.add(e);
                e += 1;
            }
            *offs.add(NUM_EXPERTS) = acc as i32;
        }
        sync_threads();
        let mut i = tid;
        while i < n {
            let e = *ids.add(i) as usize;
            let at = atomic_add_u32(off.add(e), 1);
            *pairs.add(at as usize) = i as i32;
            i += 1024;
        }
    }
}

/// Grouped NVFP4 gate/up projection with the softcapped SwiGLU: act[q] for every sorted pair q of
/// expert `blockIdx.y`, intermediate columns of row tile `blockIdx.x`. Grid (144, 128), 576 threads.
#[kernel]
#[launch_bounds(576)]
pub unsafe fn prefill_moe_up_nvfp4(
    x_h: *const u16,
    offs: *const i32,
    pairs: *const i32,
    w: *const u8,
    ws: *const u8,
    wfactor: *const f32,
    act: *mut u16,
) {
    static mut RED: SharedArray<f32, { UP_WARPS * 32 * 4 }, 16> = SharedArray::UNINIT;
    static mut OUT: SharedArray<f32, { MOE_TILE * OUT_STRIDE }> = SharedArray::UNINIT;
    unsafe {
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let out = SharedArray::as_raw_mut_ptr(&raw mut OUT);
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let tile = blockIdx_x() as usize;
        let e = blockIdx_y() as usize;
        let o0 = *offs.add(e) as usize;
        let n = *offs.add(e + 1) as usize - o0;
        if n == 0 {
            return;
        }
        let row = e * UP_R + tile * MOE_TILE + g;
        let c = 4 * warp_i + t;
        let mut a = [0u32; 16];
        let mut b = [0u32; 16];
        load_decode(w.add(row * UP_CB), ws.add(row * UP_SB), c, &mut a);
        load_decode(w.add((row + 8) * UP_CB), ws.add((row + 8) * UP_SB), c, &mut b);
        let es = *wfactor.add(e);

        let mut base = 0usize;
        #[nounroll]
        while base < n {
            let xrow = if base + g < n { x_h.add((*pairs.add(o0 + base + g) as usize / TOP_K) * HIDDEN) } else { core::ptr::null() };
            let x = load_x(xrow, c);
            let d = mma8(&a, &b, &x);
            let mut v = 0;
            #[unroll]
            while v < 4 {
                *red.add((warp_i * 32 + lane) * 4 + v) = d[v];
                v += 1;
            }
            sync_threads();
            if tid < 32 {
                let mut s = [0.0f32; 4];
                let mut ww = 0;
                #[unroll]
                while ww < UP_WARPS {
                    let mut v = 0;
                    #[unroll]
                    while v < 4 {
                        s[v] += *red.add((ww * 32 + lane) * 4 + v);
                        v += 1;
                    }
                    ww += 1;
                }
                *out.add(g * OUT_STRIDE + 2 * t) = s[0] * es;
                *out.add(g * OUT_STRIDE + 2 * t + 1) = s[1] * es;
                *out.add((g + 8) * OUT_STRIDE + 2 * t) = s[2] * es;
                *out.add((g + 8) * OUT_STRIDE + 2 * t + 1) = s[3] * es;
            }
            sync_threads();
            if tid < 64 {
                let p = tid & 7;
                let nn = tid >> 3;
                if base + nn < n {
                    let gate = *out.add(2 * p * OUT_STRIDE + nn);
                    let up = *out.add((2 * p + 1) * OUT_STRIDE + nn);
                    let g_cap = fast_tanh(gate * (1.0 / GATE_SOFTCAP)) * GATE_SOFTCAP;
                    let u_cap = fast_tanh(up * (1.0 / UP_SOFTCAP)) * UP_SOFTCAP;
                    *act.add((o0 + base + nn) * INTERMEDIATE + tile * (MOE_TILE / 2) + p) =
                        float2half((g_cap * fast_sigmoid(gate)) * u_cap);
                }
            }
            base += 8;
        }
    }
}

/// Grouped NVFP4 down projection: slots[i][tile rows] = route_w[i] * (W2_e act[q]) for every sorted pair
/// q = (pair i) of expert `blockIdx.y`. Grid (144, 128), 288 threads.
#[kernel]
#[launch_bounds(288)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_moe_down_nvfp4(
    act: *const u16,
    offs: *const i32,
    pairs: *const i32,
    route_w: *const f32,
    w: *const u8,
    ws: *const u8,
    wfactor: *const f32,
    slots: *mut f32,
) {
    static mut RED: SharedArray<f32, { DN_WARPS * 32 * 4 }, 16> = SharedArray::UNINIT;
    static mut OUT: SharedArray<f32, { MOE_TILE * OUT_STRIDE }> = SharedArray::UNINIT;
    unsafe {
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let out = SharedArray::as_raw_mut_ptr(&raw mut OUT);
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let tile = blockIdx_x() as usize;
        let e = blockIdx_y() as usize;
        let o0 = *offs.add(e) as usize;
        let n = *offs.add(e + 1) as usize - o0;
        if n == 0 {
            return;
        }
        let row = e * HIDDEN + tile * MOE_TILE + g;
        let c = 4 * warp_i + t;
        let mut a = [0u32; 16];
        let mut b = [0u32; 16];
        load_decode(w.add(row * DN_CB), ws.add(row * DN_SB), c, &mut a);
        load_decode(w.add((row + 8) * DN_CB), ws.add((row + 8) * DN_SB), c, &mut b);
        let es = *wfactor.add(e);

        let mut base = 0usize;
        #[nounroll]
        while base < n {
            let xrow = if base + g < n { act.add((o0 + base + g) * INTERMEDIATE) } else { core::ptr::null() };
            let x = load_x(xrow, c);
            let d = mma8(&a, &b, &x);
            let mut v = 0;
            #[unroll]
            while v < 4 {
                *red.add((warp_i * 32 + lane) * 4 + v) = d[v];
                v += 1;
            }
            sync_threads();
            if tid < 32 {
                let mut s = [0.0f32; 4];
                let mut ww = 0;
                #[unroll]
                while ww < DN_WARPS {
                    let mut v = 0;
                    #[unroll]
                    while v < 4 {
                        s[v] += *red.add((ww * 32 + lane) * 4 + v);
                        v += 1;
                    }
                    ww += 1;
                }
                *out.add(g * OUT_STRIDE + 2 * t) = s[0] * es;
                *out.add(g * OUT_STRIDE + 2 * t + 1) = s[1] * es;
                *out.add((g + 8) * OUT_STRIDE + 2 * t) = s[2] * es;
                *out.add((g + 8) * OUT_STRIDE + 2 * t + 1) = s[3] * es;
            }
            sync_threads();
            if tid < 128 {
                let r = tid & 15;
                let nn = tid >> 4;
                if base + nn < n {
                    let i = *pairs.add(o0 + base + nn) as usize;
                    *slots.add(i * HIDDEN + tile * MOE_TILE + r) = *out.add(r * OUT_STRIDE + nn) * *route_w.add(i);
                }
            }
            base += 8;
        }
    }
}

/// The decode path's NVFP4 router (`route_logits_nvfp4_parallel_b8` arithmetic) over the whole chunk:
/// block (x, y) computes experts 2x, 2x + 1 for tokens 8y..8y + 8. Grid (64, t / 8), 32 threads.
#[kernel]
#[launch_bounds(32)]
pub unsafe fn prefill_route_logits_nvfp4(m_h: *const u16, w_data: *const u8, w_scales: *const u8, gscale: f32, out_logits: *mut f32) {
    unsafe {
        const B: usize = 8;
        let sub_e = (threadIdx_x() >> 4) as usize;
        let lane = (threadIdx_x() & 15) as usize;
        let e = blockIdx_x() as usize * 2 + sub_e;
        let tok0 = blockIdx_y() as usize * B;
        let m_h = m_h.add(tok0 * HIDDEN);
        let out_logits = out_logits.add(tok0 * NUM_EXPERTS);
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
            w8[iter] = crate::nvfp4::ld_stream_u2(wr.add(blk * 8));
            s8[iter] = crate::nvfp4::ld_scale_u8(sr.add(blk));
            iter += 1;
        }
        let mut acc = [0.0f32; B];
        let mut iter = 0;
        #[unroll]
        while iter < 9 {
            let blk = lane + iter * 16;
            let sc = crate::cvt_e4m3_to_float(s8[iter] & 0xff);
            let lo = cvt_e2m1x8_to_half2x4(w8[iter][0]);
            let hi = cvt_e2m1x8_to_half2x4(w8[iter][1]);
            let wh = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
            let koff = blk << 4;
            let mut b = 0;
            #[unroll]
            while b < B {
                let xv0 = ldg_u4(m_h.add(b * HIDDEN + koff) as *const u8);
                let xv1 = ldg_u4(m_h.add(b * HIDDEN + koff + 8) as *const u8);
                acc[b] = crate::fmaf(crate::nvfp4::block16_dot_decoded(&wh, xv0, xv1), sc, acc[b]);
                b += 1;
            }
            iter += 1;
        }
        let mut b = 0;
        #[unroll]
        while b < B {
            let val = crate::half_warp_sum(acc[b]) * gscale;
            if lane == 0 {
                *out_logits.add(b * NUM_EXPERTS + e) = val;
            }
            b += 1;
        }
    }
}

/// The raw NVFP4 chunk `load_decode` reads: 16 code bytes and the two E4M3 scales (as u16).
#[inline(always)]
unsafe fn load_codes(codes: *const u8, scales: *const u8, c: usize) -> ([u32; 4], u16) {
    unsafe { (ld_stream_u4(codes.add(c * 16)), ld_scale_u16(scales.add(c * 2))) }
}

/// `load_decode`'s decode of a chunk loaded by `load_codes` (the same f16 values).
#[device]
fn decode_codes(q: &[u32; 4], s: u16) -> [u32; 16] {
    let sc = cvt_e4m3x2_to_half2(s);
    let slo = low2half2(sc);
    let shi = high2half2(sc);
    let mut o = [0u32; 16];
    let mut w = 0;
    #[unroll]
    while w < 4 {
        let h = cvt_e2m1x8_to_half2x4(q[w]);
        let sm = if w < 2 { slo } else { shi };
        o[4 * w] = mul_f16x2(h[0], sm);
        o[4 * w + 1] = mul_f16x2(h[1], sm);
        o[4 * w + 2] = mul_f16x2(h[2], sm);
        o[4 * w + 3] = mul_f16x2(h[3], sm);
        w += 1;
    }
    o
}

/// `prefill_moe_up_nvfp4` with two 16-row tiles per CTA (tiles 2x and 2x + 1) that share each group's
/// activation load, halving the L2 re-reads of the gathered rows. Both tiles' weights stay as NVFP4 codes in
/// registers and are decoded per group (the same f16 values as `load_decode`), so two CTAs still fit per SM.
/// Every row is computed exactly as in `prefill_moe_up_nvfp4` (same K slices, same warp-order reduction):
/// bit-identical. Grid (72, 128), 576 threads.
#[kernel]
#[launch_bounds(576, 2)]
pub unsafe fn prefill_moe_up_nvfp4_r2c(
    x_h: *const u16,
    offs: *const i32,
    pairs: *const i32,
    w: *const u8,
    ws: *const u8,
    wfactor: *const f32,
    act: *mut u16,
) {
    static mut RED: SharedArray<f32, { 2 * UP_WARPS * 32 * 4 }, 16> = SharedArray::UNINIT;
    static mut OUT: SharedArray<f32, { 2 * MOE_TILE * OUT_STRIDE }> = SharedArray::UNINIT;
    unsafe {
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let out = SharedArray::as_raw_mut_ptr(&raw mut OUT);
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let e = blockIdx_y() as usize;
        let o0 = *offs.add(e) as usize;
        let n = *offs.add(e + 1) as usize - o0;
        if n == 0 {
            return;
        }
        let c = 4 * warp_i + t;
        let row0 = e * UP_R + 2 * blockIdx_x() as usize * MOE_TILE + g;
        let row1 = row0 + MOE_TILE;
        // Both tiles' weights stay as NVFP4 codes in registers and are decoded per group (same f16 values).
        let (qa0, sa0) = load_codes(w.add(row0 * UP_CB), ws.add(row0 * UP_SB), c);
        let (qb0, sb0) = load_codes(w.add((row0 + 8) * UP_CB), ws.add((row0 + 8) * UP_SB), c);
        let (qa1, sa1) = load_codes(w.add(row1 * UP_CB), ws.add(row1 * UP_SB), c);
        let (qb1, sb1) = load_codes(w.add((row1 + 8) * UP_CB), ws.add((row1 + 8) * UP_SB), c);
        let es = *wfactor.add(e);

        let mut base = 0usize;
        #[nounroll]
        while base < n {
            let xrow = if base + g < n { x_h.add((*pairs.add(o0 + base + g) as usize / TOP_K) * HIDDEN) } else { core::ptr::null() };
            let x = load_x(xrow, c);
            let d0 = mma8(&decode_codes(&qa0, sa0), &decode_codes(&qb0, sb0), &x);
            let d1 = mma8(&decode_codes(&qa1, sa1), &decode_codes(&qb1, sb1), &x);
            let mut v = 0;
            #[unroll]
            while v < 4 {
                *red.add((warp_i * 32 + lane) * 4 + v) = d0[v];
                *red.add(((UP_WARPS + warp_i) * 32 + lane) * 4 + v) = d1[v];
                v += 1;
            }
            sync_threads();
            if tid < 64 {
                let tt = tid >> 5;
                let ob = out.add(tt * MOE_TILE * OUT_STRIDE);
                let mut s = [0.0f32; 4];
                let mut ww = 0;
                #[unroll]
                while ww < UP_WARPS {
                    let mut v = 0;
                    #[unroll]
                    while v < 4 {
                        s[v] += *red.add(((tt * UP_WARPS + ww) * 32 + lane) * 4 + v);
                        v += 1;
                    }
                    ww += 1;
                }
                *ob.add(g * OUT_STRIDE + 2 * t) = s[0] * es;
                *ob.add(g * OUT_STRIDE + 2 * t + 1) = s[1] * es;
                *ob.add((g + 8) * OUT_STRIDE + 2 * t) = s[2] * es;
                *ob.add((g + 8) * OUT_STRIDE + 2 * t + 1) = s[3] * es;
            }
            sync_threads();
            if tid < 128 {
                let tt = tid >> 6;
                let p = tid & 7;
                let nn = (tid >> 3) & 7;
                if base + nn < n {
                    let ob = out.add(tt * MOE_TILE * OUT_STRIDE);
                    let gate = *ob.add(2 * p * OUT_STRIDE + nn);
                    let up = *ob.add((2 * p + 1) * OUT_STRIDE + nn);
                    let g_cap = fast_tanh(gate * (1.0 / GATE_SOFTCAP)) * GATE_SOFTCAP;
                    let u_cap = fast_tanh(up * (1.0 / UP_SOFTCAP)) * UP_SOFTCAP;
                    let tile = 2 * blockIdx_x() as usize + tt;
                    *act.add((o0 + base + nn) * INTERMEDIATE + tile * (MOE_TILE / 2) + p) = float2half((g_cap * fast_sigmoid(gate)) * u_cap);
                }
            }
            base += 8;
        }
    }
}

/// `prefill_moe_down_nvfp4` with two 16-row tiles per CTA (see `prefill_moe_up_nvfp4_r2c`); bit-identical.
/// Grid (72, 128), 288 threads.
#[kernel]
#[launch_bounds(288, 3)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_moe_down_nvfp4_r2c(
    act: *const u16,
    offs: *const i32,
    pairs: *const i32,
    route_w: *const f32,
    w: *const u8,
    ws: *const u8,
    wfactor: *const f32,
    slots: *mut f32,
) {
    static mut RED: SharedArray<f32, { 2 * DN_WARPS * 32 * 4 }, 16> = SharedArray::UNINIT;
    static mut OUT: SharedArray<f32, { 2 * MOE_TILE * OUT_STRIDE }> = SharedArray::UNINIT;
    unsafe {
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let out = SharedArray::as_raw_mut_ptr(&raw mut OUT);
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let e = blockIdx_y() as usize;
        let o0 = *offs.add(e) as usize;
        let n = *offs.add(e + 1) as usize - o0;
        if n == 0 {
            return;
        }
        let c = 4 * warp_i + t;
        let row0 = e * HIDDEN + 2 * blockIdx_x() as usize * MOE_TILE + g;
        let row1 = row0 + MOE_TILE;
        // Both tiles' weights stay as NVFP4 codes in registers and are decoded per group (same f16 values).
        let (qa0, sa0) = load_codes(w.add(row0 * DN_CB), ws.add(row0 * DN_SB), c);
        let (qb0, sb0) = load_codes(w.add((row0 + 8) * DN_CB), ws.add((row0 + 8) * DN_SB), c);
        let (qa1, sa1) = load_codes(w.add(row1 * DN_CB), ws.add(row1 * DN_SB), c);
        let (qb1, sb1) = load_codes(w.add((row1 + 8) * DN_CB), ws.add((row1 + 8) * DN_SB), c);
        let es = *wfactor.add(e);

        let mut base = 0usize;
        #[nounroll]
        while base < n {
            let xrow = if base + g < n { act.add((o0 + base + g) * INTERMEDIATE) } else { core::ptr::null() };
            let x = load_x(xrow, c);
            let d0 = mma8(&decode_codes(&qa0, sa0), &decode_codes(&qb0, sb0), &x);
            let d1 = mma8(&decode_codes(&qa1, sa1), &decode_codes(&qb1, sb1), &x);
            let mut v = 0;
            #[unroll]
            while v < 4 {
                *red.add((warp_i * 32 + lane) * 4 + v) = d0[v];
                *red.add(((DN_WARPS + warp_i) * 32 + lane) * 4 + v) = d1[v];
                v += 1;
            }
            sync_threads();
            if tid < 64 {
                let tt = tid >> 5;
                let ob = out.add(tt * MOE_TILE * OUT_STRIDE);
                let mut s = [0.0f32; 4];
                let mut ww = 0;
                #[unroll]
                while ww < DN_WARPS {
                    let mut v = 0;
                    #[unroll]
                    while v < 4 {
                        s[v] += *red.add(((tt * DN_WARPS + ww) * 32 + lane) * 4 + v);
                        v += 1;
                    }
                    ww += 1;
                }
                *ob.add(g * OUT_STRIDE + 2 * t) = s[0] * es;
                *ob.add(g * OUT_STRIDE + 2 * t + 1) = s[1] * es;
                *ob.add((g + 8) * OUT_STRIDE + 2 * t) = s[2] * es;
                *ob.add((g + 8) * OUT_STRIDE + 2 * t + 1) = s[3] * es;
            }
            sync_threads();
            if tid < 256 {
                let tt = tid >> 7;
                let r = tid & 15;
                let nn = (tid >> 4) & 7;
                if base + nn < n {
                    let ob = out.add(tt * MOE_TILE * OUT_STRIDE);
                    let i = *pairs.add(o0 + base + nn) as usize;
                    let tile = 2 * blockIdx_x() as usize + tt;
                    *slots.add(i * HIDDEN + tile * MOE_TILE + r) = *ob.add(r * OUT_STRIDE + nn) * *route_w.add(i);
                }
            }
            base += 8;
        }
    }
}
