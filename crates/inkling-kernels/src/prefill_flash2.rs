//! Wide prefill attention v2, for long context. Same math as `prefill_flash.rs` (f16 Q pre-scaled by
//! 1/sqrt(128), K/V from the BF16 cache as f16, m16n8k16 f16 MMAs with f32 accumulation, `fast_expf`
//! online softmax, P rounded to f16, f16x2 relative bias for distances below REL_EXTENT, sink and gate),
//! organised for long key ranges:
//! - 32 query tokens per CTA (96 rows = 6 warps x one m16 tile each, token-major as before), so each
//!   K/V tile serves twice the rows of v1, and no cross-warp merge of key slices;
//! - 32-key tiles in two shared-memory stages filled with `cp.async` (raw BF16), converted to f16 in
//!   place by the thread that loaded each chunk, so the next tile's loads overlap this tile's MMAs;
//! - the relative-bias table read from a per-layer f16 copy in global memory, `[distance][16]` (L1-cached),
//!   so the CTA's shared memory is the two K/V stages and the R rows (37 KB; two CTAs per SM);
//! - masks and the relative bias only on tiles that need them (the causal diagonal, the local window
//!   edge, distances below 1024); interior tiles run MMAs and the softmax only;
//! - optional key splits (grid.z = `splits`) for long rows: each split writes its partial (m, d, o) and
//!   `prefill_attn_merge2` combines them with the sink and gate. With `splits` = 1 the kernel writes the
//!   gated output itself.
//!
//! With FP8 global KV (`fp8` non-null), rows older than the first query's relative window are read from the
//! FP8 store and dequantized in the staging pass (as v1 does); `kv_cap` is then the BF16 ring's slot count.
//!
//! Grid (8, n / 32, splits), 192 threads, FLASH2_SMEM bytes of dynamic shared memory. Query tokens
//! `tok_off + 32 blockIdx.y + i` at positions `*d_pos + tok_off + 32 blockIdx.y + i`.

use crate::common::*;
use crate::{fast_sigmoid, fmaf, float2half};
use cuda_device::f16x2::{add_f16x2, fma_f16x2, mul_f16x2};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockIdx_x, blockIdx_y, blockIdx_z, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{DynamicSharedArray, kernel, launch_bounds, ptx_asm, warp};

const KV_ROW: usize = HEAD_DIM + 8; // padded f16 row (272 B)
const FT: usize = 32; // keys per tile
const QB: usize = 32; // query tokens per CTA
const QROWS: usize = QB * GQA_RATIO; // 96
const THREADS: usize = 192;
const TILE_BYTES: usize = FT * KV_ROW * 2; // 8,704 B: one K or V tile
const STAGE_BYTES: usize = 2 * TILE_BYTES;
const R_OFF: usize = 2 * STAGE_BYTES; // 34,816
/// Dynamic shared memory per CTA: two K/V stages and the R rows (37,888 B).
pub const FLASH2_SMEM: usize = R_OFF + QROWS * D_REL * 2;

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
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];", out("=r") r0, out("=r") r1, in("r") addr);
    }
    [r0, r1]
}

/// `cp.async.cg` of 16 bytes global -> shared.
#[inline(always)]
unsafe fn cp_async16(dst: *mut u16, src: *const u16) {
    let d = unsafe { cvta_generic_to_shared_u32(dst as *const u8) };
    unsafe {
        ptx_asm!("cp.async.cg.shared.global [%0], [%1], 16;", in("r") d, in("l") src, clobber("memory"));
    }
}

/// `cp.async.ca` of 8 bytes global -> shared (FP8 codes of 8 K or V values).
#[inline(always)]
unsafe fn cp_async8(dst: *mut u16, src: *const u8) {
    let d = unsafe { cvta_generic_to_shared_u32(dst as *const u8) };
    unsafe {
        ptx_asm!("cp.async.ca.shared.global [%0], [%1], 8;", in("r") d, in("l") src, clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_async_commit() {
    unsafe {
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_async_wait1() {
    unsafe {
        ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_async_wait0() {
    unsafe {
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
    }
}

#[inline(always)]
unsafe fn st_u4(p: *mut u16, v: [u32; 4]) {
    unsafe { *(p as *mut [u32; 4]) = v }
}

#[inline(always)]
unsafe fn ld_u4(p: *const u16) -> [u32; 4] {
    unsafe { *(p as *const [u32; 4]) }
}

/// 8 BF16 values in shared memory rewritten as f16 (exact for the clipped / normed K and V rows).
#[inline(always)]
unsafe fn bf16_to_f16_inplace(p: *mut u16) {
    unsafe {
        let raw = ld_u4(p);
        let mut h = [0u32; 4];
        let mut i = 0;
        while i < 4 {
            let (x, y) = bf16x2_to_f32x2(raw[i]);
            h[i] = pack_h2(x, y);
            i += 1;
        }
        st_u4(p, h);
    }
}

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

/// Score with the relative bias, or -1e30 when the key is masked (as `prefill_flash::biased`, with the
/// f16 rel table indexed by distance).
#[inline(always)]
unsafe fn biased(s: f32, t: i32, valid: bool, pos_q: i32, window: i32, r_row: *const u16, rel16: *const u16) -> f32 {
    unsafe {
        if !valid || t > pos_q || (window > 0 && t < pos_q + 1 - window) {
            return -1e30;
        }
        let dist = pos_q - t;
        if dist < REL_EXTENT as i32 {
            let rh = r_row as *const u32;
            let ph = rel16.add(dist as usize * D_REL) as *const u32;
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

/// FP8 global KV (`fp8` non-null): rows older than `fp8_before` come from the FP8 store (E4M3 codes and
/// an f16 scale per (token, KV head)); newer rows from the BF16 ring (`kv_cap` = its slots).
#[derive(Clone, Copy)]
struct Fp8Kv {
    k8: *const u8,
    v8: *const u8,
    ks: *const u16,
    vs: *const u16,
    cap: i32,
    before: i32,
}

/// Issues the `cp.async` loads of key tile [t0, t0 + FT) into stage buffers `sk` / `sv` (rows past
/// `t_end` are zero-filled with plain stores). FP8 rows land as 8 code bytes in the low half of the
/// 16-byte slot.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn load_tile(sk: *mut u16, sv: *mut u16, k_cache: *const u16, v_cache: *const u16, kv_cap: i32, kvh: usize, t0: i32, t_end: i32, tid: usize, f8: &Fp8Kv) {
    unsafe {
        let mut idx = tid;
        while idx < FT * 16 {
            let t_loc = idx >> 4;
            let sub = idx & 15;
            let off = t_loc * KV_ROW + sub * 8;
            let t = t0 + t_loc as i32;
            if t < t_end {
                if t < f8.before {
                    let o8 = (t % f8.cap) as usize * KV_DIM + kvh * HEAD_DIM + sub * 8;
                    cp_async8(sk.add(off), f8.k8.add(o8));
                    cp_async8(sv.add(off), f8.v8.add(o8));
                } else {
                    let slot = (t % kv_cap) as usize;
                    let g = slot * KV_DIM + kvh * HEAD_DIM + sub * 8;
                    cp_async16(sk.add(off), k_cache.add(g));
                    cp_async16(sv.add(off), v_cache.add(g));
                }
            } else {
                st_u4(sk.add(off), [0; 4]);
                st_u4(sv.add(off), [0; 4]);
            }
            idx += THREADS;
        }
    }
}

/// Converts this thread's chunks of a landed tile to f16 (only rows below `t_end`; the rest are zeros):
/// BF16 rows exactly, FP8 rows dequantized with their per-(token, KV head) scale.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn convert_tile(sk: *mut u16, sv: *mut u16, t0: i32, t_end: i32, tid: usize, kvh: usize, f8: &Fp8Kv) {
    unsafe {
        let mut idx = tid;
        while idx < FT * 16 {
            let t_loc = idx >> 4;
            let off = t_loc * KV_ROW + (idx & 15) * 8;
            let t = t0 + t_loc as i32;
            if t < t_end {
                if t < f8.before {
                    let si = (t % f8.cap) as usize * NUM_KV_HEADS + kvh;
                    let kc = *(sk.add(off) as *const [u32; 2]);
                    let vc = *(sv.add(off) as *const [u32; 2]);
                    st_u4(sk.add(off), crate::attn::e4m3x8_to_h2x4([kc[0], kc[1], *f8.ks.add(si) as u32, 0]));
                    st_u4(sv.add(off), crate::attn::e4m3x8_to_h2x4([vc[0], vc[1], *f8.vs.add(si) as u32, 0]));
                } else {
                    bf16_to_f16_inplace(sk.add(off));
                    bf16_to_f16_inplace(sv.add(off));
                }
            }
            idx += THREADS;
        }
    }
}

#[kernel]
#[launch_bounds(192)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_attn_flash2(
    qrgs: *const f32,
    rel16: *const u16,
    k_cache: *const u16,
    v_cache: *const u16,
    kv_cap: i32,
    window: i32,
    d_pos: *const i32,
    out_gated: *mut f32,
    out_gated_h: *mut u16,
    tok_off: i32,
    splits: i32,
    part: *mut f32,
    fp8: *const u16,
    fp8_cap: i32,
) {
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let stage_k = |i: usize| smem.add(i * STAGE_BYTES) as *mut u16;
        let stage_v = |i: usize| smem.add(i * STAGE_BYTES + TILE_BYTES) as *mut u16;
        let s_rel = rel16;
        let s_r = smem.add(R_OFF) as *mut u16;

        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let grp = lane >> 2;
        let tid4 = lane & 3;
        let kvh = blockIdx_x() as usize;
        let tok0 = tok_off as usize + blockIdx_y() as usize * QB;
        let split = blockIdx_z() as i32;
        let pos0 = *d_pos;
        let p_first = pos0 + tok0 as i32;
        let p_last = p_first + QB as i32 - 1;

        // This split's key range, 32-aligned.
        let k_start = if window > 0 { core::cmp::max(0, p_first + 1 - window) } else { 0 };
        let k_end = p_last + 1;
        let span = k_end - k_start;
        let per = ((span + splits - 1) / splits + FT as i32 - 1) / FT as i32 * FT as i32;
        let lo = k_start + split * per;
        let hi = core::cmp::min(k_end, lo + per);

        let q0 = warp_i * 16 + grp;
        let q1 = q0 + 8;
        let row_bh = |qi: usize| -> (usize, usize) { (tok0 + qi / GQA_RATIO, kvh * GQA_RATIO + qi % GQA_RATIO) };

        if lo >= hi {
            // Empty split: an empty partial for each row (one writer per row).
            if splits > 1 && tid < QROWS {
                let (b, h) = row_bh(tid);
                let pi = (b * NUM_HEADS + h) * splits as usize + split as usize;
                *part.add(2 * pi) = -1e30;
                *part.add(2 * pi + 1) = 0.0;
            }
            return;
        }

        // FP8 global KV: rows older than the first query's relative window come from the FP8 store.
        let f8 = if fp8.is_null() {
            Fp8Kv { k8: core::ptr::null(), v8: core::ptr::null(), ks: core::ptr::null(), vs: core::ptr::null(), cap: 1, before: i32::MIN }
        } else {
            let (k8, v8, ks, vs) = crate::attn::fp8_shadow_parts(fp8, fp8_cap);
            Fp8Kv { k8, v8, ks, vs, cap: fp8_cap, before: p_first - (REL_EXTENT as i32 - 1) }
        };
        // First tile in flight while the tables load.
        load_tile(stage_k(0), stage_v(0), k_cache, v_cache, kv_cap, kvh, lo, hi, tid, &f8);
        cp_async_commit();

        let mut i = tid;
        while i < QROWS * D_REL {
            let qi = i >> 4;
            let k = i & 15;
            let (b, h) = row_bh(qi);
            *s_r.add(i) = float2half(*qrgs.add(b * QRGS_STRIDE + Q_DIM + h * D_REL + k));
            i += THREADS;
        }

        let qp = |qi: usize| {
            let (b, h) = row_bh(qi);
            qrgs.add(b * QRGS_STRIDE + h * HEAD_DIM)
        };
        let q_reg = load_q_frag(qp(q0), qp(q1), tid4);
        let pos_q0 = p_first + (q0 / GQA_RATIO) as i32;
        let pos_q1 = p_first + (q1 / GQA_RATIO) as i32;
        let r0 = s_r.add(q0 * D_REL) as *const u16;
        let r1 = s_r.add(q1 * D_REL) as *const u16;
        let c0 = tid4 * 2;

        let mut m0 = -1e30f32;
        let mut m1 = -1e30f32;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut o = [[0.0f32; 4]; 16];

        let n_tiles = (hi - lo + FT as i32 - 1) / FT as i32;
        let mut it = 0i32;
        #[nounroll]
        while it < n_tiles {
            let st = (it & 1) as usize;
            let t0 = lo + it * FT as i32;
            // Next tile's loads into the other stage (its previous contents were consumed before the
            // barrier that ended the last iteration).
            if it + 1 < n_tiles {
                load_tile(stage_k(st ^ 1), stage_v(st ^ 1), k_cache, v_cache, kv_cap, kvh, t0 + FT as i32, hi, tid, &f8);
                cp_async_commit();
                cp_async_wait1();
            } else {
                cp_async_wait0();
            }
            convert_tile(stage_k(st), stage_v(st), t0, hi, tid, kvh, &f8);
            sync_threads();

            let sk = stage_k(st) as *const u16;
            let sv = stage_v(st) as *const u16;
            // S = Q K^T for 32 keys: two 16-key halves (x4 ldmatrix gives two n8 tiles).
            let mut sc = [[0.0f32; 4]; 4];
            let mut hlf = 0;
            #[unroll]
            while hlf < 2 {
                let k_ld_row = hlf * 16 + ((lane & 16) >> 1) + (lane & 7);
                let k_ld_col = lane & 8;
                let mut kt = 0;
                #[unroll]
                while kt < 8 {
                    let kb = ldmatrix_x4(sk.add(k_ld_row * KV_ROW + kt * 16 + k_ld_col));
                    sc[2 * hlf] = mma_m16n8k16_f32_f16(sc[2 * hlf], q_reg[kt], [kb[0], kb[1]]);
                    sc[2 * hlf + 1] = mma_m16n8k16_f32_f16(sc[2 * hlf + 1], q_reg[kt], [kb[2], kb[3]]);
                    kt += 1;
                }
                hlf += 1;
            }

            // Masks and bias only where a (query, key) pair can be masked or within REL_EXTENT.
            let t_last = t0 + FT as i32 - 1;
            let special = t_last > p_first || t_last >= hi || (window > 0 && t0 < p_last + 1 - window) || p_first - t_last < REL_EXTENT as i32;
            if special {
                let mut n = 0;
                #[unroll]
                while n < 4 {
                    let c = n * 8 + c0;
                    let ta = t0 + c as i32;
                    let tb = ta + 1;
                    sc[n][0] = biased(sc[n][0], ta, ta < hi, pos_q0, window, r0, s_rel);
                    sc[n][1] = biased(sc[n][1], tb, tb < hi, pos_q0, window, r0, s_rel);
                    sc[n][2] = biased(sc[n][2], ta, ta < hi, pos_q1, window, r1, s_rel);
                    sc[n][3] = biased(sc[n][3], tb, tb < hi, pos_q1, window, r1, s_rel);
                    n += 1;
                }
            }

            let rmax0 = quad_max(fmaxf(fmaxf(fmaxf(sc[0][0], sc[0][1]), fmaxf(sc[1][0], sc[1][1])), fmaxf(fmaxf(sc[2][0], sc[2][1]), fmaxf(sc[3][0], sc[3][1]))));
            let rmax1 = quad_max(fmaxf(fmaxf(fmaxf(sc[0][2], sc[0][3]), fmaxf(sc[1][2], sc[1][3])), fmaxf(fmaxf(sc[2][2], sc[2][3]), fmaxf(sc[3][2], sc[3][3]))));
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
            let mut ps = [[0.0f32; 4]; 4];
            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;
            let mut n = 0;
            #[unroll]
            while n < 4 {
                ps[n][0] = ex(sc[n][0], m0_new);
                ps[n][1] = ex(sc[n][1], m0_new);
                ps[n][2] = ex(sc[n][2], m1_new);
                ps[n][3] = ex(sc[n][3], m1_new);
                sum0 += ps[n][0] + ps[n][1];
                sum1 += ps[n][2] + ps[n][3];
                n += 1;
            }
            d0 = fmaf(d0, alpha0, quad_sum(sum0));
            d1 = fmaf(d1, alpha1, quad_sum(sum1));
            m0 = m0_new;
            m1 = m1_new;

            // O += P V, two 16-key k steps.
            let mut kk = 0;
            #[unroll]
            while kk < 2 {
                let pa = [
                    pack_h2(ps[2 * kk][0], ps[2 * kk][1]),
                    pack_h2(ps[2 * kk][2], ps[2 * kk][3]),
                    pack_h2(ps[2 * kk + 1][0], ps[2 * kk + 1][1]),
                    pack_h2(ps[2 * kk + 1][2], ps[2 * kk + 1][3]),
                ];
                let v_ld_row = kk * 16 + (lane & 15);
                let mut nt = 0;
                #[unroll]
                while nt < 16 {
                    let vb = ldmatrix_x2_trans(sv.add(v_ld_row * KV_ROW + nt * 8));
                    o[nt] = mma_m16n8k16_f32_f16(o[nt], pa, vb);
                    nt += 1;
                }
                kk += 1;
            }
            sync_threads();
            it += 1;
        }

        // Output: rows q0 (o[..][0..2]) and q1 (o[..][2..4]); columns nt * 8 + c0, + 1.
        let mut r = 0;
        #[unroll]
        while r < 2 {
            let qi = if r == 0 { q0 } else { q1 };
            let (m, d) = if r == 0 { (m0, d0) } else { (m1, d1) };
            let (b, h) = row_bh(qi);
            if splits > 1 {
                let pi = (b * NUM_HEADS + h) * splits as usize + split as usize;
                if tid4 == 0 {
                    *part.add(2 * pi) = m;
                    *part.add(2 * pi + 1) = d;
                }
                let base = part.add(part_o_off(b, h, split as usize, splits as usize));
                let mut nt = 0;
                #[unroll]
                while nt < 16 {
                    *base.add(nt * 8 + c0) = o[nt][2 * r];
                    *base.add(nt * 8 + c0 + 1) = o[nt][2 * r + 1];
                    nt += 1;
                }
            } else {
                let qrow = qrgs.add(b * QRGS_STRIDE);
                let sink = *qrow.add(Q_DIM + R_DIM + Q_DIM + h);
                let gm = fmaxf(sink, m);
                let mut gd = fast_expf(sink - gm);
                let sc = if d > 0.0 { fast_expf(m - gm) } else { 0.0 };
                gd = fmaf(d, sc, gd);
                let mut nt = 0;
                #[unroll]
                while nt < 16 {
                    let mut j = 0;
                    #[unroll]
                    while j < 2 {
                        let dim = nt * 8 + c0 + j;
                        let go = o[nt][2 * r + j] * sc;
                        let val = div_rn(go, gd) * fast_sigmoid(*qrow.add(Q_DIM + R_DIM + h * HEAD_DIM + dim));
                        let oi = b * Q_DIM + h * HEAD_DIM + dim;
                        if !out_gated.is_null() {
                            *out_gated.add(oi) = val;
                        }
                        if !out_gated_h.is_null() {
                            *out_gated_h.add(oi) = float2half(val);
                        }
                        j += 1;
                    }
                    nt += 1;
                }
            }
            r += 1;
        }
    }
}

/// The partial buffer: [T * 24 * splits] (m, d) pairs, then [T * 24 * splits][128] o rows, for T up to
/// PART_T_MAX tokens.
pub const PART_T_MAX: usize = 1024;
const PART_O_BASE: usize = 2 * PART_T_MAX * NUM_HEADS * 16;

#[inline(always)]
fn part_o_off(b: usize, h: usize, split: usize, splits: usize) -> usize {
    PART_O_BASE + ((b * NUM_HEADS + h) * splits + split) * HEAD_DIM
}

/// Combines the key splits of `prefill_attn_flash2` with the sink, then applies the gate: tokens
/// `tok_off + blockIdx.x`, head `blockIdx.y`. Grid (n, 24), 128 threads.
#[kernel]
#[launch_bounds(128)]
pub unsafe fn prefill_attn_merge2(qrgs: *const f32, part: *const f32, splits: i32, out_gated: *mut f32, out_gated_h: *mut u16, tok_off: i32) {
    unsafe {
        let b = tok_off as usize + blockIdx_x() as usize;
        let h = blockIdx_y() as usize;
        let dim = threadIdx_x() as usize;
        let ns = splits as usize;
        let qrow = qrgs.add(b * QRGS_STRIDE);
        let sink = *qrow.add(Q_DIM + R_DIM + Q_DIM + h);
        let pi0 = (b * NUM_HEADS + h) * ns;
        let mut gm = sink;
        let mut s = 0;
        while s < ns {
            gm = fmaxf(gm, *part.add(2 * (pi0 + s)));
            s += 1;
        }
        let mut gd = fast_expf(sink - gm);
        let mut go = 0.0f32;
        let mut s = 0;
        while s < ns {
            let dv = *part.add(2 * (pi0 + s) + 1);
            if dv > 0.0 {
                let sc = fast_expf(*part.add(2 * (pi0 + s)) - gm);
                gd = fmaf(dv, sc, gd);
                go = fmaf(*part.add(part_o_off(b, h, s, ns) + dim), sc, go);
            }
            s += 1;
        }
        let val = div_rn(go, gd) * fast_sigmoid(*qrow.add(Q_DIM + R_DIM + h * HEAD_DIM + dim));
        let oi = b * Q_DIM + h * HEAD_DIM + dim;
        if !out_gated.is_null() {
            *out_gated.add(oi) = val;
        }
        if !out_gated_h.is_null() {
            *out_gated_h.add(oi) = float2half(val);
        }
    }
}
