//! Router and expert kernels (exact mode).
//!
//! Ports of squid5090 @ 3ee093c:
//! - `route_logits_parallel_kernel<1>` (kernels_inkling.cu:1201) as `route_logits_parallel_b1`;
//! - `route_select_and_group_kernel<1>` (kernels_inkling.cu:1319) as
//!   `route_select_and_group_b1`. At B=1 squid5090 builds no ExpertGroups,
//!   so the `g` parameter is unused; it stays for the parameter list;
//! - `moe_up_tc_kernel<1>` (MXFP8, kernels_gemv.cu:1438) as `moe_up_tc_mxfp8`;
//!   the NVFP4 variant is K0's `moe_up_tc_nvfp4`;
//! - `moe_down_tc_kernel<0>` / `<1>` (kernels_gemv.cu:1519) as
//!   `moe_down_tc_nvfp4` / `moe_down_tc_mxfp8`. Fix for SQ bug 4: the route IDs
//!   are read after `griddepcontrol.wait`, so the weight prefetch no longer
//!   overlaps the predecessor. Outputs are unchanged.

use crate::common::*;
use crate::{
    GATE_SOFTCAP, HIDDEN, INTERMEDIATE, MOE_MAXM, MOE_TILE, TOP_K, UP_SOFTCAP, cvt_e2m1x8_to_half2x4,
    cvt_e4m3x2_to_half2, fast_sigmoid, fast_tanh, float2half, fmaf, h2f_hi, h2f_lo, high2half2, ld_dep_f32, ld_dep_i32,
    ld_dep_u4, ld_scale_u16, ld_stream_u4, low2half2, pdl_trigger, pdl_wait,
};
use cuda_device::f16x2::mul_f16x2;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::{mma_m16n8k16_f32_bf16, mma_m16n8k16_f32_f16};
use cuda_device::{SharedArray, cluster, device, kernel, launch_bounds, ptx_asm, warp};

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// `route_logits_parallel_kernel<1>`: logits[e] = router_w[e] . m. Grid 64,
/// 64 threads (one warp per expert). `moe_down_zero` is only touched for B > 1.
#[kernel]
#[launch_bounds(64)]
pub unsafe fn route_logits_parallel_b1(
    m: *const f32,
    router_w: *const u16,
    out_logits: *mut f32,
    _moe_down_zero: *mut f32,
) {
    unsafe {
        let warp_i = (threadIdx_x() >> 5) as usize;
        let lane = (threadIdx_x() & 31) as usize;
        let e = blockIdx_x() as usize * 2 + warp_i;
        if e < NUM_EXPERTS {
            let wr = router_w.add(e * HIDDEN) as *const u8;
            let mut acc = 0.0f32;
            let mut v = lane;
            while v < (HIDDEN >> 3) {
                let w8 = ldg_u4(wr.add(v * 16));
                let (w0x, w0y) = bf16x2_to_f32x2(w8[0]);
                let (w1x, w1y) = bf16x2_to_f32x2(w8[1]);
                let (w2x, w2y) = bf16x2_to_f32x2(w8[2]);
                let (w3x, w3y) = bf16x2_to_f32x2(w8[3]);
                let xf = m.add(v << 3);
                let x0 = ldg_f4(xf);
                let x1 = ldg_f4(xf.add(4));
                acc = fmaf(w0x, x0[0], acc);
                acc = fmaf(w0y, x0[1], acc);
                acc = fmaf(w1x, x0[2], acc);
                acc = fmaf(w1y, x0[3], acc);
                acc = fmaf(w2x, x1[0], acc);
                acc = fmaf(w2y, x1[1], acc);
                acc = fmaf(w3x, x1[2], acc);
                acc = fmaf(w3y, x1[3], acc);
                v += 32;
            }
            let val = warp_sum(acc);
            if lane == 0 {
                *out_logits.add(e) = val;
            }
        }
    }
}


/// `route_select_and_group_kernel<B>` body: per token (one warp each),
/// sigmoid scores, top-8 by score + bias, weights normalised to sum to
/// ROUTE_SCALE. The ExpertGroups build squid5090 does for B > 1 when `g` is
/// non-null is not ported: the engine passes null, and then squid5090 skips it.
#[device]
unsafe fn route_select_body(logits: *const f32, router_bias: *const f32, out_ids: *mut i32, out_weights: *mut f32, zero: u32) {
    unsafe {
        let b = (threadIdx_x() >> 5) as usize;
        let lane = (threadIdx_x() & 31) as usize;

        let mut rb = [0.0f32; 4];
        let mut i = 0;
        #[unroll]
        while i < 4 {
            rb[i] = ldg_f32(router_bias.add(lane + 32 * i));
            i += 1;
        }
        pdl_wait_fenced(zero);
        pdl_trigger();

        let mut sc = [0.0f32; 4];
        let mut bi = [0.0f32; 4];
        let mut i = 0;
        #[unroll]
        while i < 4 {
            let e = lane + 32 * i;
            let raw = fast_sigmoid(ld_dep_f32(logits.add(b * NUM_EXPERTS + e)));
            sc[i] = raw;
            bi[i] = raw + rb[i];
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
                let w = chosen[k] * inv;
                *out_ids.add(b * TOP_K + k) = winners[k];
                *out_weights.add(b * TOP_K + k) = w;
                k += 1;
            }
        }
    }
}

macro_rules! route_select_kernel {
    ($name:ident, $threads:expr) => {
        /// `route_select_and_group_kernel<B>`: grid 1, B * 32 threads, PDL. `g` must be null.
        #[kernel]
        #[launch_bounds($threads)]
        pub unsafe fn $name(
            logits: *const f32,
            router_bias: *const f32,
            out_ids: *mut i32,
            out_weights: *mut f32,
            _g: *mut u8,
            zero: u32,
        ) {
            unsafe { route_select_body(logits, router_bias, out_ids, out_weights, zero) }
        }
    };
}

route_select_kernel!(route_select_and_group_b1, 32);
route_select_kernel!(route_select_and_group_b2, 64);
route_select_kernel!(route_select_and_group_b4, 128);
route_select_kernel!(route_select_and_group_b8, 256);

/// `__ldg(const float*)`.
#[inline(always)]
unsafe fn ldg_f32(p: *const f32) -> f32 {
    let v: f32;
    unsafe {
        ptx_asm!("ld.global.nc.f32 %0, [%1];", out("=f") v, in("l") p, options(register_only));
    }
    v
}

// ---------------------------------------------------------------------------
// Expert MMA fragments (NVFP4 = FMT 0, MXFP8 = FMT 1)
// ---------------------------------------------------------------------------

const MOE_UP_WARPS: usize = HIDDEN / 128; // 18
const MOE_DOWN_GROUPS: usize = INTERMEDIATE / 128; // 9
const MOE_DOWN_PAR: usize = 2;
const MOE_DOWN_WARPS: usize = MOE_DOWN_GROUPS * MOE_DOWN_PAR; // 18

/// One 32-weight chunk: NVFP4 uses `q[0..4]` and a 16-bit pair of E4M3
/// scales; MXFP8 uses `q[0..8]` and one UE8M0 scale in the low byte.
#[derive(Clone, Copy)]
struct Chunk {
    q: [u32; 8],
    s: u32,
}

/// `MoeFrag<FMT>`: rows g and g+8 of a 16-row tile, plus the x fragment.
#[derive(Clone, Copy)]
struct Frag {
    r0: Chunk,
    r1: Chunk,
    x: [u32; 16],
}

/// `ld_scale_u8`.
#[inline(always)]
unsafe fn ld_scale_u8(p: *const u8) -> u16 {
    let v: u16;
    unsafe {
        ptx_asm!("ld.global.nc.u8 %0, [%1];", out("=h") v, in("l") p);
    }
    v & 0xff
}

/// `load_nvfp4_chunk32` / `load_mxfp8_chunk32`.
#[inline(always)]
unsafe fn load_chunk<const FMT: u32>(codes: *const u8, scales: *const u8, c: usize) -> Chunk {
    unsafe {
        if FMT == 0 {
            let q = ld_stream_u4(codes.add(c * 16));
            Chunk { q: [q[0], q[1], q[2], q[3], 0, 0, 0, 0], s: ld_scale_u16(scales.add(c * 2)) as u32 }
        } else {
            let q0 = ld_stream_u4(codes.add(c * 32));
            let q1 = ld_stream_u4(codes.add(c * 32 + 16));
            Chunk { q: [q0[0], q0[1], q0[2], q0[3], q1[0], q1[1], q1[2], q1[3]], s: ld_scale_u8(scales.add(c)) as u32 }
        }
    }
}

/// `moe_load_w<FMT, K>`.
#[inline(always)]
unsafe fn moe_load_w<const FMT: u32, const K: usize>(f: &mut Frag, codes: *const u8, scales: *const u8, c: usize) {
    let (cb, sb) = if FMT == 0 { (K >> 1, K >> 4) } else { (K, K >> 5) };
    unsafe {
        f.r0 = load_chunk::<FMT>(codes, scales, c);
        f.r1 = load_chunk::<FMT>(codes.add(8 * cb), scales.add(8 * sb), c);
    }
}

/// `moe_load_x<FMT>`: the activation row as f16 pairs (NVFP4) or BF16 pairs
/// (MXFP8); zeros when the row is off.
#[device]
unsafe fn moe_load_x<const FMT: u32>(f: &mut Frag, xrow: *const u16, c: usize) {
    unsafe {
        if !xrow.is_null() {
            let xp = xrow.add(c * 32);
            let mut v = 0;
            #[unroll]
            while v < 4 {
                let q = ld_dep_u4(xp.add(v * 8));
                if FMT == 0 {
                    f.x[4 * v] = q[0];
                    f.x[4 * v + 1] = q[1];
                    f.x[4 * v + 2] = q[2];
                    f.x[4 * v + 3] = q[3];
                } else {
                    let mut j = 0;
                    #[unroll]
                    while j < 4 {
                        f.x[4 * v + j] = pack_bf16x2(h2f_lo(q[j]), h2f_hi(q[j]));
                        j += 1;
                    }
                }
                v += 1;
            }
        } else {
            f.x = [0u32; 16];
        }
    }
}

/// `decode_nvfp4_chunk32_f16`.
#[device]
fn decode_nvfp4(ch: &Chunk, o: &mut [u32; 16]) {
    let sc = cvt_e4m3x2_to_half2(ch.s as u16);
    let slo = low2half2(sc);
    let shi = high2half2(sc);
    let mut w = 0;
    #[unroll]
    while w < 4 {
        let h = cvt_e2m1x8_to_half2x4(ch.q[w]);
        let sm = if w < 2 { slo } else { shi };
        o[4 * w] = mul_f16x2(h[0], sm);
        o[4 * w + 1] = mul_f16x2(h[1], sm);
        o[4 * w + 2] = mul_f16x2(h[2], sm);
        o[4 * w + 3] = mul_f16x2(h[3], sm);
        w += 1;
    }
}

/// `moe_mma_frag<FMT>`: 8 m16n8k16 MMAs over the 128-wide K slice.
#[device]
fn moe_mma_frag<const FMT: u32>(f: &Frag, d: &mut [f32; 4]) {
    let mut a = [0u32; 16];
    let mut b = [0u32; 16];
    if FMT == 0 {
        decode_nvfp4(&f.r0, &mut a);
        decode_nvfp4(&f.r1, &mut b);
    } else {
        decode_mxfp8(&f.r0, &mut a);
        decode_mxfp8(&f.r1, &mut b);
    }
    let mut i = 0;
    #[unroll]
    while i < 8 {
        let af = [a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1]];
        let bf = [f.x[2 * i], f.x[2 * i + 1]];
        *d = unsafe {
            if FMT == 0 { mma_m16n8k16_f32_f16(*d, af, bf) } else { mma_m16n8k16_f32_bf16(*d, af, bf) }
        };
        i += 1;
    }
}

/// `decode_mxfp8_chunk32_bf16`: 32 E4M3 codes times the UE8M0 block scale,
/// as 16 BF16 pairs.
#[device]
fn decode_mxfp8(ch: &Chunk, o: &mut [u32; 16]) {
    let sc = cvt_ue8m0_to_float(ch.s as u8);
    let mut w = 0;
    #[unroll]
    while w < 8 {
        let mut hh = 0;
        #[unroll]
        while hh < 2 {
            let h2 = cvt_e4m3x2_to_half2((ch.q[w] >> (16 * hh)) as u16);
            o[2 * w + hh] = pack_bf16x2(h2f_lo(h2) * sc, h2f_hi(h2) * sc);
            hh += 1;
        }
        w += 1;
    }
}

/// `moe_slot_mask(e, lo, hi)`.
#[inline(always)]
fn moe_slot_mask(e: i32, lo: i32, hi: i32) -> u64 {
    let m0 = warp::ballot_sync(0xffff_ffff, lo == e);
    let m1 = warp::ballot_sync(0xffff_ffff, hi == e);
    (m0 as u64) | ((m1 as u64) << 32)
}

// ---------------------------------------------------------------------------
// moe_up_tc_kernel<1> (MXFP8)
// ---------------------------------------------------------------------------

const UP_K: usize = HIDDEN;
const UP_R: usize = 2 * INTERMEDIATE;
const TILE_OUT_STRIDE: usize = MOE_MAXM + 1;

/// `moe_up_tc_kernel<1>`: MXFP8 expert gate/up projection (layers 20-23).
/// Grid (144, M * TOP_K), 576 threads.
#[kernel]
#[launch_bounds(576)]
pub unsafe fn moe_up_tc_mxfp8(
    x_h: *const u16,
    idx: *const i32,
    w: *const u8,
    ws: *const u8,
    _wfactor: *const f32,
    m: i32,
    out_act_h: *mut u16,
) {
    static mut RED: SharedArray<f32, { MOE_UP_WARPS * 32 * 4 }, 16> = SharedArray::UNINIT;
    static mut TILE_OUT: SharedArray<f32, { MOE_TILE * TILE_OUT_STRIDE }> = SharedArray::UNINIT;
    unsafe {
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
        let tile_out = SharedArray::as_raw_mut_ptr(&raw mut TILE_OUT);
        const CB: usize = UP_K;
        const SB: usize = UP_K >> 5;

        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let tile = blockIdx_x() as usize;
        let slot = blockIdx_y() as usize;
        let nslots = m as usize * TOP_K;

        pdl_wait();
        pdl_trigger();

        let lo = if lane < nslots { ld_dep_i32(idx.add(lane)) } else { -1 };
        let hi = if lane + 32 < nslots { ld_dep_i32(idx.add(lane + 32)) } else { -1 };
        let e = ld_dep_i32(idx.add(slot));
        let mask = moe_slot_mask(e, lo, hi);
        if mask.trailing_zeros() as usize != slot {
            return;
        }

        let row = e as usize * UP_R + tile * MOE_TILE + g;
        let on = (g < m as usize) && ((mask >> (8 * g)) & 0xff) != 0;
        let mut f = Frag { r0: Chunk { q: [0; 8], s: 0 }, r1: Chunk { q: [0; 8], s: 0 }, x: [0; 16] };
        moe_load_w::<1, UP_K>(&mut f, w.add(row * CB), ws.add(row * SB), 4 * warp_i + t);
        moe_load_x::<1>(&mut f, if on { x_h.add(g * UP_K) } else { core::ptr::null() }, 4 * warp_i + t);
        let mut d = [0.0f32; 4];
        moe_mma_frag::<1>(&f, &mut d);

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
            while ww < MOE_UP_WARPS {
                let mut v = 0;
                #[unroll]
                while v < 4 {
                    s[v] += *red.add((ww * 32 + lane) * 4 + v);
                    v += 1;
                }
                ww += 1;
            }
            *tile_out.add(g * TILE_OUT_STRIDE + 2 * t) = s[0];
            *tile_out.add(g * TILE_OUT_STRIDE + 2 * t + 1) = s[1];
            *tile_out.add((g + 8) * TILE_OUT_STRIDE + 2 * t) = s[2];
            *tile_out.add((g + 8) * TILE_OUT_STRIDE + 2 * t + 1) = s[3];
        }
        sync_threads();

        if tid < (MOE_TILE / 2) * MOE_MAXM {
            let p = tid & 7;
            let n = tid >> 3;
            let mut kmask: u32 = if n < m as usize { ((mask >> (8 * n)) as u32) & 0xff } else { 0 };
            if kmask != 0 {
                let gate = *tile_out.add(2 * p * TILE_OUT_STRIDE + n);
                let up = *tile_out.add((2 * p + 1) * TILE_OUT_STRIDE + n);
                let g_cap = fast_tanh(gate * (1.0 / GATE_SOFTCAP)) * GATE_SOFTCAP;
                let u_cap = fast_tanh(up * (1.0 / UP_SOFTCAP)) * UP_SOFTCAP;
                let ah = float2half((g_cap * fast_sigmoid(gate)) * u_cap);
                while kmask != 0 {
                    let k = kmask.trailing_zeros() as usize;
                    kmask &= kmask - 1;
                    *out_act_h.add((n * TOP_K + k) * INTERMEDIATE + tile * (MOE_TILE / 2) + p) = ah;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// moe_down_tc_kernel<FMT>
// ---------------------------------------------------------------------------

const DN_K: usize = INTERMEDIATE;
const DN_R: usize = HIDDEN;
const NSLOT_MAX: usize = TOP_K * MOE_MAXM; // 64

/// `moe_down_tc_kernel<FMT>`: deterministic fixed-order expert reduction.
/// Grid HIDDEN / 16 = 144, 576 threads.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn moe_down_body<const FMT: u32, const CL: usize>(
    part: *mut f32,
    ybuf: *mut f32,
    uexp: *mut i32,
    umask: *mut u64,
    slot_u: *mut u8,
    nuniq: *mut i32,
    act_h: *const u16,
    idx: *const i32,
    wts: *const f32,
    w: *const u8,
    ws: *const u8,
    wfactor: *const f32,
    m: i32,
    out_down: *mut f32,
) {
    unsafe {
        let (cb, sb) = if FMT == 0 { (DN_K >> 1, DN_K >> 4) } else { (DN_K, DN_K >> 5) };
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        // CL > 1: a cluster of CL CTAs shares one tile; rank r computes the unique experts u = r mod CL
        // and the final weighted sum reads each expert's y from its owner's shared memory.
        let rank = if CL > 1 { cluster::cluster_ctaidX() as usize } else { 0 };
        let tile = blockIdx_x() as usize / CL;
        let m = m as usize;
        let nslots = m * TOP_K;

        // The route IDs are read before `griddepcontrol.wait`, as squid5090 does: the router finished
        // before `moe_up` (this kernel's PDL predecessor) passed its own wait and triggered us, so the
        // IDs are complete and visible here. This lets the round-0 weight loads overlap `moe_up`'s tail.
        // It relies on `moe_up` always being the direct predecessor (forward.rs). `act_h` is `moe_up`'s
        // output and is read only after the wait.
        if warp_i == 0 {
            let lo = if lane < nslots { ld_dep_i32(idx.add(lane)) } else { -1 };
            let hi = if lane + 32 < nslots { ld_dep_i32(idx.add(lane + 32)) } else { -1 };
            let mut remaining: u64 = if nslots == 64 { !0u64 } else { (1u64 << nslots) - 1 };
            let mut u = 0usize;
            while remaining != 0 {
                let s = remaining.trailing_zeros() as usize;
                let mine = if s < 32 { lo } else { hi };
                let e = warp::shuffle_sync(0xffff_ffff, mine as u32, (s & 31) as u32) as i32;
                let mask = moe_slot_mask(e, lo, hi);
                remaining &= !mask;
                if lane == 0 {
                    *uexp.add(u) = e;
                    *umask.add(u) = mask;
                }
                if (mask >> lane) & 1 != 0 {
                    *slot_u.add(lane) = u as u8;
                }
                if (mask >> (lane + 32)) & 1 != 0 {
                    *slot_u.add(lane + 32) = u as u8;
                }
                u += 1;
            }
            if lane == 0 {
                *nuniq = u as i32;
            }
        }
        sync_threads();

        let uu = *nuniq as usize;
        // Unique experts this CTA computes: u = li * CL + rank for li < nl.
        let nl = if uu > rank { (uu - rank).div_ceil(CL) } else { 0 };
        let gu = |li: usize| li * CL + rank;
        let p = warp_i / MOE_DOWN_GROUPS;
        let c = 4 * (warp_i % MOE_DOWN_GROUPS) + t;

        let xrow = |u: usize| -> *const u16 {
            let kmask: u32 = if g < m { ((*umask.add(u) >> (8 * g)) as u32) & 0xff } else { 0 };
            if kmask != 0 { act_h.add((g * TOP_K + kmask.trailing_zeros() as usize) * DN_K) } else { core::ptr::null() }
        };
        let load_w_u = |u: usize, f: &mut Frag| {
            let e = *uexp.add(u) as usize;
            let row = e * DN_R + tile * MOE_TILE + g;
            moe_load_w::<FMT, DN_K>(f, w.add(row * cb), ws.add(row * sb), c);
        };

        let mut nxt = Frag { r0: Chunk { q: [0; 8], s: 0 }, r1: Chunk { q: [0; 8], s: 0 }, x: [0; 16] };
        if p < nl {
            load_w_u(gu(p), &mut nxt);
        }
        pdl_wait();
        // Let the next kernel launch now and prefetch its constants; it reads out_down only after its
        // own griddepcontrol.wait, which waits for this whole grid to complete.
        pdl_trigger();
        if p < nl {
            moe_load_x::<FMT>(&mut nxt, xrow(gu(p)), c);
        }

        let mut base = 0usize;
        #[nounroll]
        while base < nl {
            let cur = nxt;
            let li = base + p;
            if li + MOE_DOWN_PAR < nl {
                load_w_u(gu(li + MOE_DOWN_PAR), &mut nxt);
                moe_load_x::<FMT>(&mut nxt, xrow(gu(li + MOE_DOWN_PAR)), c);
            }
            let mut d = [0.0f32; 4];
            if li < nl {
                moe_mma_frag::<FMT>(&cur, &mut d);
            }
            let mut v = 0;
            #[unroll]
            while v < 4 {
                *part.add((warp_i * 32 + lane) * 4 + v) = d[v];
                v += 1;
            }
            sync_threads();

            if tid < MOE_DOWN_PAR * 32 && base + (tid >> 5) < nl {
                let pp = tid >> 5;
                let ug = gu(base + pp);
                let mut s = [0.0f32; 4];
                let mut j = 0;
                #[unroll]
                while j < MOE_DOWN_GROUPS {
                    let mut v = 0;
                    #[unroll]
                    while v < 4 {
                        s[v] += *part.add(((pp * MOE_DOWN_GROUPS + j) * 32 + lane) * 4 + v);
                        v += 1;
                    }
                    j += 1;
                }
                let es = if FMT == 0 { *wfactor.add(*uexp.add(ug) as usize) } else { 1.0 };
                let y = ybuf.add(ug * MOE_TILE * MOE_MAXM);
                if FMT == 0 {
                    *y.add(g * MOE_MAXM + 2 * t) = s[0] * es;
                    *y.add(g * MOE_MAXM + 2 * t + 1) = s[1] * es;
                    *y.add((g + 8) * MOE_MAXM + 2 * t) = s[2] * es;
                    *y.add((g + 8) * MOE_MAXM + 2 * t + 1) = s[3] * es;
                } else {
                    *y.add(g * MOE_MAXM + 2 * t) = s[0];
                    *y.add(g * MOE_MAXM + 2 * t + 1) = s[1];
                    *y.add((g + 8) * MOE_MAXM + 2 * t) = s[2];
                    *y.add((g + 8) * MOE_MAXM + 2 * t + 1) = s[3];
                }
            }
            sync_threads();
            base += MOE_DOWN_PAR;
        }
        if CL > 1 {
            cluster::cluster_sync();
        }

        if tid < MOE_TILE * m && rank == 0 {
            let r = tid & (MOE_TILE - 1);
            let n = tid / MOE_TILE;
            let mut s = 0.0f32;
            let mut k = 0;
            #[unroll]
            while k < TOP_K {
                let sl = n * TOP_K + k;
                let su = *slot_u.add(sl) as usize;
                let yp = ybuf.add((su * MOE_TILE + r) * MOE_MAXM + n);
                let yv = if CL > 1 { *cluster::map_shared_rank(yp as *const f32, (su % CL) as u32) } else { *yp };
                s = fmaf(ld_dep_f32(wts.add(sl)), yv, s);
                k += 1;
            }
            *out_down.add(n * HIDDEN + tile * MOE_TILE + r) = s;
        }
        if CL > 1 {
            // No CTA may exit while another still reads its shared memory.
            cluster::cluster_sync();
        }
    }
}

macro_rules! moe_down_kernel {
    ($name:ident, $fmt:expr, $doc:literal) => {
        moe_down_kernel!(@body $name, $fmt, 1, $doc, );
    };
    // `cluster N`: N-CTA clusters sharing one tile (grid 144 * N). Measured slower than CL = 1 on the
    // RTX 5090 (1.12x at N = 2, 1.4-1.6x at N = 4, bit-identical), so no instance is built.
    ($name:ident, $fmt:expr, cluster $cl:literal, $doc:literal) => {
        moe_down_kernel!(@body $name, $fmt, $cl, $doc, #[cuda_device::cluster_launch($cl, 1, 1)]);
    };
    (@body $name:ident, $fmt:expr, $cl:literal, $doc:literal, $(#[$attr:meta])?) => {
        #[doc = $doc]
        #[kernel]
        $(#[$attr])?
        #[launch_bounds(576)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            act_h: *const u16,
            idx: *const i32,
            wts: *const f32,
            w: *const u8,
            ws: *const u8,
            wfactor: *const f32,
            m: i32,
            out_down: *mut f32,
        ) {
            static mut PART: SharedArray<f32, { MOE_DOWN_WARPS * 32 * 4 }, 16> = SharedArray::UNINIT;
            static mut YBUF: SharedArray<f32, { NSLOT_MAX * MOE_TILE * MOE_MAXM }, 16> = SharedArray::UNINIT;
            static mut UEXP: SharedArray<i32, { NSLOT_MAX }> = SharedArray::UNINIT;
            static mut UMASK: SharedArray<u64, { NSLOT_MAX }> = SharedArray::UNINIT;
            static mut SLOT_U: SharedArray<u8, { NSLOT_MAX }> = SharedArray::UNINIT;
            static mut NUNIQ: SharedArray<i32, 1> = SharedArray::UNINIT;
            unsafe {
                moe_down_body::<$fmt, $cl>(
                    SharedArray::as_raw_mut_ptr(&raw mut PART),
                    SharedArray::as_raw_mut_ptr(&raw mut YBUF),
                    SharedArray::as_raw_mut_ptr(&raw mut UEXP),
                    SharedArray::as_raw_mut_ptr(&raw mut UMASK),
                    SharedArray::as_raw_mut_ptr(&raw mut SLOT_U),
                    SharedArray::as_raw_mut_ptr(&raw mut NUNIQ),
                    act_h, idx, wts, w, ws, wfactor, m, out_down,
                )
            }
        }
    };
}

moe_down_kernel!(moe_down_tc_nvfp4, 0, "`moe_down_tc_kernel<0>` (NVFP4, layers 0-19).");
moe_down_kernel!(moe_down_tc_mxfp8, 1, "`moe_down_tc_kernel<1>` (MXFP8, layers 20-23).");

