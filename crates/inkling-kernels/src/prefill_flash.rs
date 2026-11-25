//! Wide prefill attention: one flash-attention pass per (KV head, 16-token query block) over the whole
//! causal range of the block, with the relative bias, the local window, the attention sink and the
//! sigmoid output gate fused in (what `local_tail_attn_tc` + `global_attn_bulk_splitk` + `attn_merge_sink_gate`
//! do for B <= 8). The chunk's K/V rows must already be in the cache.
//!
//! The math follows the B <= 8 kernels: Q pre-scaled by 1/sqrt(128) and rounded to f16, K/V rounded from
//! BF16 to f16, m16n8k16 f16 MMAs with f32 accumulation, `fast_expf` online softmax, P rounded to f16, the
//! relative bias as f16x2 dot products for distances below REL_EXTENT. The summation order over keys
//! differs (one online pass instead of 64-key splits merged afterwards).
//!
//! Grid (8, T / 16), 192 threads: 6 warps = 3 M tiles of 16 query rows (48 rows = 16 tokens x 3 GQA heads,
//! token-major) x 2 key slices of 32 per 64-key tile; the slices are merged at the end.

use crate::common::*;
use crate::{fast_sigmoid, fmaf, float2half};
use cuda_device::f16x2::{add_f16x2, fma_f16x2, mul_f16x2};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{SharedArray, kernel, launch_bounds, ptx_asm, warp};

const KV_ROW: usize = HEAD_DIM + 8;
const FT: usize = 64;
const QB: usize = 16;
const QROWS: usize = QB * GQA_RATIO; // 48
const THREADS: usize = 192;

#[inline(always)]
fn pack_h2(lo: f32, hi: f32) -> u32 {
    pack_f16x2(lo, hi)
}

#[inline(always)]
unsafe fn ldmatrix_x4(p: *const u16) -> [u32; 4] {
    let addr = unsafe { cvta_generic_to_shared_u32(p as *const u8) };
    let (r0, r1, r2, r3): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];",
            out("=r") r0,
            out("=r") r1,
            out("=r") r2,
            out("=r") r3,
            in("r") addr,
        );
    }
    [r0, r1, r2, r3]
}

#[inline(always)]
unsafe fn ldmatrix_x2_trans(p: *const u16) -> [u32; 2] {
    let addr = unsafe { cvta_generic_to_shared_u32(p as *const u8) };
    let (r0, r1): (u32, u32);
    unsafe {
        ptx_asm!(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];",
            out("=r") r0,
            out("=r") r1,
            in("r") addr,
        );
    }
    [r0, r1]
}

#[inline(always)]
unsafe fn load_kv8_bf16(cache: *const u16, slot: usize, kvh: usize, sub: usize) -> [u32; 4] {
    unsafe {
        let raw = ldg_u4(cache.add(slot * KV_DIM + kvh * HEAD_DIM + sub * 8) as *const u8);
        let mut h = [0u32; 4];
        let mut i = 0;
        while i < 4 {
            let (x, y) = bf16x2_to_f32x2(raw[i]);
            h[i] = pack_h2(x, y);
            i += 1;
        }
        h
    }
}

#[inline(always)]
unsafe fn st_u4(p: *mut u16, v: [u32; 4]) {
    unsafe { *(p as *mut [u32; 4]) = v }
}

#[inline(always)]
unsafe fn ldg_u16(p: *const u16) -> u16 {
    let v: u16;
    unsafe {
        ptx_asm!("ld.global.nc.u16 %0, [%1];", out("=h") v, in("l") p, options(register_only));
    }
    v
}

/// Q fragments of rows q0 / q1 (m16n8k16 A operand), pre-scaled by 1/sqrt(HEAD_DIM), rounded to f16.
#[inline(always)]
unsafe fn load_q_frag(qp0: *const f32, qp1: *const f32, tid4: usize) -> [[u32; 4]; 8] {
    unsafe {
        let mut q = [[0u32; 4]; 8];
        let mut kt = 0;
        while kt < 8 {
            let k_off = kt * 16;
            q[kt][0] = pack_h2(*qp0.add(k_off + tid4 * 2) * INV_SQRT_HEAD_DIM, *qp0.add(k_off + tid4 * 2 + 1) * INV_SQRT_HEAD_DIM);
            q[kt][2] = pack_h2(*qp0.add(k_off + tid4 * 2 + 8) * INV_SQRT_HEAD_DIM, *qp0.add(k_off + tid4 * 2 + 9) * INV_SQRT_HEAD_DIM);
            q[kt][1] = pack_h2(*qp1.add(k_off + tid4 * 2) * INV_SQRT_HEAD_DIM, *qp1.add(k_off + tid4 * 2 + 1) * INV_SQRT_HEAD_DIM);
            q[kt][3] = pack_h2(*qp1.add(k_off + tid4 * 2 + 8) * INV_SQRT_HEAD_DIM, *qp1.add(k_off + tid4 * 2 + 9) * INV_SQRT_HEAD_DIM);
            kt += 1;
        }
        q
    }
}

#[inline(always)]
fn quad_max(v: f32) -> f32 {
    let v = fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 1));
    fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 2))
}

#[inline(always)]
fn quad_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 1);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 2);
    v
}

/// Score with the relative bias, or -1e30 when the key is masked (causal, window, past the tile).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn biased(s: f32, t: i32, valid: bool, pos_q: i32, window: i32, r_row: *const u16, s_rp: *const u16, d_min: i32) -> f32 {
    unsafe {
        if !valid || t > pos_q || (window > 0 && t < pos_q + 1 - window) {
            return -1e30;
        }
        let dist = pos_q - t;
        if dist < REL_EXTENT as i32 {
            let rh = r_row as *const u32;
            let ph = s_rp.add((dist - d_min) as usize * D_REL) as *const u32;
            let mut a0 = mul_f16x2(*rh, *ph);
            let mut a1 = mul_f16x2(*rh.add(1), *ph.add(1));
            a0 = fma_f16x2(*rh.add(2), *ph.add(2), a0);
            a1 = fma_f16x2(*rh.add(3), *ph.add(3), a1);
            a0 = fma_f16x2(*rh.add(4), *ph.add(4), a0);
            a1 = fma_f16x2(*rh.add(5), *ph.add(5), a1);
            a0 = fma_f16x2(*rh.add(6), *ph.add(6), a0);
            a1 = fma_f16x2(*rh.add(7), *ph.add(7), a1);
            let f = add_f16x2(a0, a1);
            s + (crate::h2f_lo(f) + crate::h2f_hi(f))
        } else {
            s
        }
    }
}

#[inline(always)]
fn ex(s: f32, m: f32) -> f32 {
    if s > -1e29 { fast_expf(s - m) } else { 0.0 }
}

#[kernel]
#[launch_bounds(192)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_attn_flash(
    qrgs: *const f32,
    rel_proj: *const u16,
    k_cache: *const u16,
    v_cache: *const u16,
    kv_cap: i32,
    window: i32,
    d_pos: *const i32,
    out_gated: *mut f32,
    out_gated_h: *mut u16,
    tok_off: i32,
    fp8: *const u16,
    fp8_cap: i32,
) {
    static mut S_K: SharedArray<u16, { FT * KV_ROW }, 16> = SharedArray::UNINIT;
    static mut S_V: SharedArray<u16, { FT * KV_ROW }, 16> = SharedArray::UNINIT;
    static mut S_R: SharedArray<u16, { QROWS * D_REL }, 16> = SharedArray::UNINIT;
    static mut S_RP: SharedArray<u16, { (FT + QB) * D_REL }, 16> = SharedArray::UNINIT;
    static mut RED: SharedArray<f32, { 4 * 16 * 2 }, 16> = SharedArray::UNINIT;
    unsafe {
        let s_k = SharedArray::as_raw_mut_ptr(&raw mut S_K);
        let s_v = SharedArray::as_raw_mut_ptr(&raw mut S_V);
        let s_r = SharedArray::as_raw_mut_ptr(&raw mut S_R);
        let s_rp = SharedArray::as_raw_mut_ptr(&raw mut S_RP);
        let red = SharedArray::as_raw_mut_ptr(&raw mut RED);

        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let grp = lane >> 2;
        let tid4 = lane & 3;
        let slice = warp_i & 1;
        let m_tile = warp_i >> 1;
        let kvh = blockIdx_x() as usize;
        let tok0 = tok_off as usize + blockIdx_y() as usize * QB;
        let pos0 = *d_pos;
        let p_first = pos0 + tok0 as i32;
        let p_last = p_first + QB as i32 - 1;

        let mut i = tid;
        while i < QROWS * D_REL {
            let qi = i >> 4;
            let k = i & 15;
            let h = kvh * GQA_RATIO + qi % GQA_RATIO;
            let b = tok0 + qi / GQA_RATIO;
            *s_r.add(qi * D_REL + k) = float2half(*qrgs.add(b * QRGS_STRIDE + Q_DIM + h * D_REL + k));
            i += THREADS;
        }

        let q0 = m_tile * 16 + grp;
        let q1 = q0 + 8;
        let qp = |qi: usize| qrgs.add((tok0 + qi / GQA_RATIO) * QRGS_STRIDE + (kvh * GQA_RATIO + qi % GQA_RATIO) * HEAD_DIM);
        let q_reg = load_q_frag(qp(q0), qp(q1), tid4);
        let pos_q0 = p_first + (q0 / GQA_RATIO) as i32;
        let pos_q1 = p_first + (q1 / GQA_RATIO) as i32;
        let r0 = s_r.add(q0 * D_REL) as *const u16;
        let r1 = s_r.add(q1 * D_REL) as *const u16;

        let c0 = tid4 * 2;
        let c1 = c0 + 1;
        let c4 = c0 + 8;
        let c5 = c0 + 9;
        let start = if window > 0 { core::cmp::max(0, p_first + 1 - window) } else { 0 };
        let end = p_last + 1;

        let mut m0 = -1e30f32;
        let mut m1 = -1e30f32;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut o = [[0.0f32; 4]; 16];

        let mut t0 = start;
        #[nounroll]
        while t0 < end {
            let tile_len = core::cmp::min(FT as i32, end - t0) as usize;
            sync_threads();
            let mut idx = tid;
            while idx < FT * 16 {
                let t_loc = idx >> 4;
                let sub = idx & 15;
                let off = t_loc * KV_ROW + sub * 8;
                let t = t0 + t_loc as i32;
                if t_loc < tile_len && !fp8.is_null() && t < p_first - (REL_EXTENT as i32 - 1) {
                    // FP8 global KV: rows older than the first query's relative window come from the FP8 store
                    // (the BF16 cache is then a ring of GLOBAL_KV_RING rows).
                    let (k8, v8, ks, vs) = crate::attn::fp8_shadow_parts(fp8, fp8_cap);
                    let slot = (t % fp8_cap) as usize;
                    let o8 = slot * KV_DIM + kvh * HEAD_DIM + sub * 8;
                    let si = slot * NUM_KV_HEADS + kvh;
                    let kr = crate::ldg_u2(k8.add(o8));
                    let vr = crate::ldg_u2(v8.add(o8));
                    st_u4(s_k.add(off), crate::attn::e4m3x8_to_h2x4([kr[0], kr[1], *ks.add(si) as u32, 0]));
                    st_u4(s_v.add(off), crate::attn::e4m3x8_to_h2x4([vr[0], vr[1], *vs.add(si) as u32, 0]));
                } else if t_loc < tile_len {
                    let slot = (t % kv_cap) as usize;
                    st_u4(s_k.add(off), load_kv8_bf16(k_cache, slot, kvh, sub));
                    st_u4(s_v.add(off), load_kv8_bf16(v_cache, slot, kvh, sub));
                } else {
                    st_u4(s_k.add(off), [0; 4]);
                    st_u4(s_v.add(off), [0; 4]);
                }
                idx += THREADS;
            }
            let d_min = core::cmp::max(0, p_first - (t0 + tile_len as i32 - 1));
            let d_max = core::cmp::min(REL_EXTENT as i32 - 1, p_last - t0);
            let d_span = if d_max >= d_min { (d_max - d_min + 1) as usize } else { 0 };
            let mut i = tid;
            while i < d_span * D_REL {
                let di = i >> 4;
                let k = i & 15;
                *s_rp.add(di * D_REL + k) = float2half(bf16_to_f32(ldg_u16(rel_proj.add(k * REL_EXTENT + d_min as usize + di))));
                i += THREADS;
            }
            sync_threads();

            let mut sb = 0;
            #[unroll]
            while sb < 2 {
                let t_base = slice * 32 + sb * 16;
                let w_len = core::cmp::min(16, tile_len.saturating_sub(t_base));
                if w_len > 0 {
                    let mut sa = [0.0f32; 4];
                    let mut sbv = [0.0f32; 4];
                    let k_ld_row = t_base + ((lane & 16) >> 1) + (lane & 7);
                    let k_ld_col = lane & 8;
                    let mut kt = 0;
                    #[unroll]
                    while kt < 8 {
                        let kb = ldmatrix_x4(s_k.add(k_ld_row * KV_ROW + kt * 16 + k_ld_col));
                        sa = mma_m16n8k16_f32_f16(sa, q_reg[kt], [kb[0], kb[1]]);
                        sbv = mma_m16n8k16_f32_f16(sbv, q_reg[kt], [kb[2], kb[3]]);
                        kt += 1;
                    }
                    let tb = t0 + t_base as i32;
                    let s0 = biased(sa[0], tb + c0 as i32, c0 < w_len, pos_q0, window, r0, s_rp, d_min);
                    let s1 = biased(sa[1], tb + c1 as i32, c1 < w_len, pos_q0, window, r0, s_rp, d_min);
                    let s4 = biased(sbv[0], tb + c4 as i32, c4 < w_len, pos_q0, window, r0, s_rp, d_min);
                    let s5 = biased(sbv[1], tb + c5 as i32, c5 < w_len, pos_q0, window, r0, s_rp, d_min);
                    let s2 = biased(sa[2], tb + c0 as i32, c0 < w_len, pos_q1, window, r1, s_rp, d_min);
                    let s3 = biased(sa[3], tb + c1 as i32, c1 < w_len, pos_q1, window, r1, s_rp, d_min);
                    let s6 = biased(sbv[2], tb + c4 as i32, c4 < w_len, pos_q1, window, r1, s_rp, d_min);
                    let s7 = biased(sbv[3], tb + c5 as i32, c5 < w_len, pos_q1, window, r1, s_rp, d_min);

                    let rmax0 = quad_max(fmaxf(fmaxf(s0, s1), fmaxf(s4, s5)));
                    let rmax1 = quad_max(fmaxf(fmaxf(s2, s3), fmaxf(s6, s7)));
                    let m0_new = fmaxf(m0, rmax0);
                    let m1_new = fmaxf(m1, rmax1);
                    let alpha0 = if m0 > -1e29 { fast_expf(m0 - m0_new) } else { 0.0 };
                    let alpha1 = if m1 > -1e29 { fast_expf(m1 - m1_new) } else { 0.0 };
                    let mut nt = 0;
                    #[unroll]
                    while nt < 16 {
                        o[nt][0] *= alpha0;
                        o[nt][1] *= alpha0;
                        o[nt][2] *= alpha1;
                        o[nt][3] *= alpha1;
                        nt += 1;
                    }
                    let ps0 = ex(s0, m0_new);
                    let ps1 = ex(s1, m0_new);
                    let ps4 = ex(s4, m0_new);
                    let ps5 = ex(s5, m0_new);
                    let ps2 = ex(s2, m1_new);
                    let ps3 = ex(s3, m1_new);
                    let ps6 = ex(s6, m1_new);
                    let ps7 = ex(s7, m1_new);
                    d0 = fmaf(d0, alpha0, quad_sum(ps0 + ps1 + ps4 + ps5));
                    d1 = fmaf(d1, alpha1, quad_sum(ps2 + ps3 + ps6 + ps7));
                    m0 = m0_new;
                    m1 = m1_new;
                    let pa = [pack_h2(ps0, ps1), pack_h2(ps2, ps3), pack_h2(ps4, ps5), pack_h2(ps6, ps7)];
                    let v_ld_row = t_base + (lane & 15);
                    let mut nt = 0;
                    #[unroll]
                    while nt < 16 {
                        let vb = ldmatrix_x2_trans(s_v.add(v_ld_row * KV_ROW + nt * 8));
                        o[nt] = mma_m16n8k16_f32_f16(o[nt], pa, vb);
                        nt += 1;
                    }
                }
                sb += 1;
            }
            t0 += FT as i32;
        }
        sync_threads();

        // Merge the two key slices of each M tile with the sink, then apply the gate.
        let red_m = red; // [16][2]
        let red_d = red.add(32); // [16][2]
        let red_o = s_k as *mut f32; // [2][16][128] (16 KB of the K tile)
        let mut mt = 0;
        while mt < 3 {
            if m_tile == mt {
                if tid4 == 0 {
                    *red_m.add(grp * 2 + slice) = m0;
                    *red_d.add(grp * 2 + slice) = d0;
                    *red_m.add((grp + 8) * 2 + slice) = m1;
                    *red_d.add((grp + 8) * 2 + slice) = d1;
                }
                let mut nt = 0;
                #[unroll]
                while nt < 16 {
                    *red_o.add((slice * 16 + grp) * HEAD_DIM + nt * 8 + c0) = o[nt][0];
                    *red_o.add((slice * 16 + grp) * HEAD_DIM + nt * 8 + c1) = o[nt][1];
                    *red_o.add((slice * 16 + grp + 8) * HEAD_DIM + nt * 8 + c0) = o[nt][2];
                    *red_o.add((slice * 16 + grp + 8) * HEAD_DIM + nt * 8 + c1) = o[nt][3];
                    nt += 1;
                }
            }
            sync_threads();
            let mut i = tid;
            while i < 16 * HEAD_DIM {
                let qi = i >> 7;
                let dim = i & 127;
                let qg = mt * 16 + qi;
                let b = tok0 + qg / GQA_RATIO;
                let h = kvh * GQA_RATIO + qg % GQA_RATIO;
                let qrow = qrgs.add(b * QRGS_STRIDE);
                let sink = *qrow.add(Q_DIM + R_DIM + Q_DIM + h);
                let gm = fmaxf(sink, fmaxf(*red_m.add(qi * 2), *red_m.add(qi * 2 + 1)));
                let mut gd = fast_expf(sink - gm);
                let mut go = 0.0f32;
                let mut s = 0;
                while s < 2 {
                    let dv = *red_d.add(qi * 2 + s);
                    if dv > 0.0 {
                        let sc = fast_expf(*red_m.add(qi * 2 + s) - gm);
                        gd = fmaf(dv, sc, gd);
                        go = fmaf(*red_o.add((s * 16 + qi) * HEAD_DIM + dim), sc, go);
                    }
                    s += 1;
                }
                let gate = *qrow.add(Q_DIM + R_DIM + h * HEAD_DIM + dim);
                let val = div_rn(go, gd) * fast_sigmoid(gate);
                let oi = b * Q_DIM + h * HEAD_DIM + dim;
                if !out_gated.is_null() {
                    *out_gated.add(oi) = val;
                }
                if !out_gated_h.is_null() {
                    *out_gated_h.add(oi) = float2half(val);
                }
                i += THREADS;
            }
            sync_threads();
            mt += 1;
        }
    }
}
