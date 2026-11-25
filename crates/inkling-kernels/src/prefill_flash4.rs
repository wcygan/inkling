//! Wide prefill attention v1x2: `prefill_attn_flash3` with one CTA per two 16-token query blocks (A, B = A + 16).
//! Each K/V window of 64 + offB rows (offB = B's window start - A's: 16 on local visits, 0 on global ones) is
//! staged once and read by both blocks, each at its own v1 tile boundaries, so every block keeps v1's tiles,
//! slices, per-16-key online softmax and merge order. Rows a block reads past its own range are masked
//! (p = 0, exact). The FP8 threshold differs by 16 rows between the blocks: rows in [before_A, before_B) are
//! BF16 for A and FP8 for B, so B reads them from a 16-row band dequantized once at the start.
//! Results are bit-identical to `prefill_attn_flash` / `prefill_attn_flash3`. Grid (8, ceil(n / 32)),
//! 384 threads (warps 0-5 block A, 6-11 block B), FLASH4_SMEM bytes of dynamic shared memory.

use crate::common::*;
use crate::{fast_sigmoid, fmaf, float2half};
use cuda_device::f16x2::{add_f16x2, fma_f16x2, mul_f16x2};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{DynamicSharedArray, SharedArray, kernel, launch_bounds, ptx_asm, warp};

const KV_ROW: usize = HEAD_DIM + 8;
const FT: usize = 64;
const QB: usize = 16;
const QROWS: usize = QB * GQA_RATIO; // 48
const THREADS: usize = 384;
const BT: usize = 192; // threads per query block
/// Staged rows per window: block A's 64-key tile plus up to 16 rows of block B's.
const WIN: usize = FT + 16;
const TILE_BYTES: usize = WIN * KV_ROW * 2; // 21,760 B: one K or V window
const STAGE_BYTES: usize = 2 * TILE_BYTES;
/// Per stage, the FP8 rows' scales: [80 rows][K pair, V pair] u16 (the KV-head pair holding this head).
const SC_BYTES: usize = WIN * 4 * 2;
/// Block B's FP8 band: 16 rows of K, then 16 of V, dequantized to f16.
const BAND_ROWS: usize = 16;
const BAND_BYTES: usize = 2 * BAND_ROWS * KV_ROW * 2;
/// Dynamic shared memory per CTA: two K/V stages, their scale rows and the band (97,024 B).
pub const FLASH4_SMEM: usize = 2 * STAGE_BYTES + 2 * SC_BYTES + BAND_BYTES;

/// `cp.async.ca` of 4 bytes global -> shared (two f16 scales).
#[inline(always)]
unsafe fn cp_async4(dst: *mut u16, src: *const u16) {
    let d = unsafe { cvta_generic_to_shared_u32(dst as *const u8) };
    unsafe {
        ptx_asm!("cp.async.ca.shared.global [%0], [%1], 4;", in("r") d, in("l") src, clobber("memory"));
    }
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

/// FP8 global KV: rows older than `before` come from the FP8 store (E4M3 codes, an f16 scale per (token,
/// KV head)); newer rows from the BF16 cache / ring (`kv_cap` slots).
#[derive(Clone, Copy)]
struct Fp8Kv {
    k8: *const u8,
    v8: *const u8,
    ks: *const u16,
    vs: *const u16,
    cap: i32,
    before: i32,
}

/// Issues the loads of window rows [t0, t0 + tile_len) of `rows` (rows past it zero-filled), as v1 stages them.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn load_tile(sk: *mut u16, sv: *mut u16, ssc: *mut u16, k_cache: *const u16, v_cache: *const u16, kv_cap: i32, kvh: usize, t0: i32, tile_len: usize, rows: usize, tid: usize, f8: &Fp8Kv) {
    unsafe {
        let mut idx = tid;
        while idx < rows * 16 {
            let t_loc = idx >> 4;
            let sub = idx & 15;
            let off = t_loc * KV_ROW + sub * 8;
            let t = t0 + t_loc as i32;
            if t_loc < tile_len && t < f8.before {
                let slot = (t % f8.cap) as usize;
                let o8 = slot * KV_DIM + kvh * HEAD_DIM + sub * 8;
                cp_async8(sk.add(off), f8.k8.add(o8));
                cp_async8(sv.add(off), f8.v8.add(o8));
                if sub == 0 {
                    // The row's scales travel with its codes (no dependent global load in the convert pass).
                    let si = slot * NUM_KV_HEADS + (kvh & !1);
                    cp_async4(ssc.add(t_loc * 4), f8.ks.add(si));
                    cp_async4(ssc.add(t_loc * 4 + 2), f8.vs.add(si));
                }
            } else if t_loc < tile_len {
                let g = (t % kv_cap) as usize * KV_DIM + kvh * HEAD_DIM + sub * 8;
                cp_async16(sk.add(off), k_cache.add(g));
                cp_async16(sv.add(off), v_cache.add(g));
            } else {
                st_u4(sk.add(off), [0; 4]);
                st_u4(sv.add(off), [0; 4]);
            }
            idx += THREADS;
        }
    }
}

/// Converts this thread's landed chunks to f16: FP8 rows with `e4m3x8_to_h2x4` and their scale, BF16 rows
/// as `load_kv8_bf16` does (exactly v1's staged values).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn convert_tile(sk: *mut u16, sv: *mut u16, ssc: *const u16, t0: i32, tile_len: usize, rows: usize, tid: usize, kvh: usize, f8: &Fp8Kv) {
    unsafe {
        let mut idx = tid;
        while idx < rows * 16 {
            let t_loc = idx >> 4;
            let off = t_loc * KV_ROW + (idx & 15) * 8;
            let t = t0 + t_loc as i32;
            if t_loc < tile_len {
                if t < f8.before {
                    let kc = *(sk.add(off) as *const [u32; 2]);
                    let vc = *(sv.add(off) as *const [u32; 2]);
                    let ksc = *ssc.add(t_loc * 4 + (kvh & 1)) as u32;
                    let vsc = *ssc.add(t_loc * 4 + 2 + (kvh & 1)) as u32;
                    st_u4(sk.add(off), crate::attn::e4m3x8_to_h2x4([kc[0], kc[1], ksc, 0]));
                    st_u4(sv.add(off), crate::attn::e4m3x8_to_h2x4([vc[0], vc[1], vsc, 0]));
                } else {
                    let rk = *(sk.add(off) as *const [u32; 4]);
                    let rv = *(sv.add(off) as *const [u32; 4]);
                    let mut hk = [0u32; 4];
                    let mut hv = [0u32; 4];
                    let mut i = 0;
                    while i < 4 {
                        let (x, y) = bf16x2_to_f32x2(rk[i]);
                        hk[i] = pack_h2(x, y);
                        let (x, y) = bf16x2_to_f32x2(rv[i]);
                        hv[i] = pack_h2(x, y);
                        i += 1;
                    }
                    st_u4(sk.add(off), hk);
                    st_u4(sv.add(off), hv);
                }
            }
            idx += THREADS;
        }
    }
}

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
unsafe fn st_u4(p: *mut u16, v: [u32; 4]) {
    unsafe { *(p as *mut [u32; 4]) = v }
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

/// Score with the relative bias, or -1e30 when the key is masked (causal, window, past the tile).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
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

#[kernel]
#[launch_bounds(384)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_attn_flash4(
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
    fp8: *const u16,
    fp8_cap: i32,
    n_q: i32,
) {
    static mut S_R: SharedArray<u16, { 2 * QROWS * D_REL }, 16> = SharedArray::UNINIT;
    unsafe {
        let smem = DynamicSharedArray::<u8, 16>::get();
        let stage_k = |i: usize| smem.add(i * STAGE_BYTES) as *mut u16;
        let stage_v = |i: usize| smem.add(i * STAGE_BYTES + TILE_BYTES) as *mut u16;
        let stage_sc = |i: usize| smem.add(2 * STAGE_BYTES + i * SC_BYTES) as *mut u16;
        let band_k = smem.add(2 * STAGE_BYTES + 2 * SC_BYTES) as *mut u16;
        let band_v = band_k.add(BAND_ROWS * KV_ROW);

        let tid = threadIdx_x() as usize;
        let warp_i = tid >> 5;
        let lane = tid & 31;
        let grp = lane >> 2;
        let tid4 = lane & 3;
        let blk = warp_i / 6;
        let wl = warp_i - blk * 6;
        let btid = tid - blk * BT;
        let slice = wl & 1;
        let m_tile = wl >> 1;
        let kvh = blockIdx_x() as usize;
        let pair0 = blockIdx_y() as usize * 2 * QB;
        let has_b = ((pair0 + QB) as i32) < n_q;
        let active = blk == 0 || has_b;
        let s_r = (SharedArray::as_raw_mut_ptr(&raw mut S_R) as *mut u16).add(blk * QROWS * D_REL);
        // The merge's (m, d) rows alias the band, which is dead after the key loop.
        let red = (band_k as *mut f32).add(blk * 64);
        let pos0 = *d_pos;
        let tok_a = tok_off as usize + pair0;
        let p_a = pos0 + tok_a as i32;
        let tok0 = if active { tok_a + blk * QB } else { tok_a };
        let p_first = pos0 + tok0 as i32;
        let p_last = p_first + QB as i32 - 1;

        if active {
            let mut i = btid;
            while i < QROWS * D_REL {
                let qi = i >> 4;
                let k = i & 15;
                let h = kvh * GQA_RATIO + qi % GQA_RATIO;
                let b = tok0 + qi / GQA_RATIO;
                *s_r.add(qi * D_REL + k) = float2half(*qrgs.add(b * QRGS_STRIDE + Q_DIM + h * D_REL + k));
                i += BT;
            }
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
        // Block A's range starts the window; block B's starts `off_b` rows into it.
        let start_a = if window > 0 { core::cmp::max(0, p_a + 1 - window) } else { 0 };
        let end_a = p_a + QB as i32;
        let start_b = if window > 0 { core::cmp::max(0, p_a + QB as i32 + 1 - window) } else { 0 };
        let end_b = end_a + QB as i32;
        let n_a = (end_a - start_a + FT as i32 - 1) / FT as i32;
        let n_b = (end_b - start_b + FT as i32 - 1) / FT as i32;
        let off_b = if has_b { (start_b - start_a) as usize } else { 0 };
        let rows = FT + off_b;
        let end_ld = if has_b { end_b } else { end_a };
        let n_it = if has_b { core::cmp::max(n_a, n_b) } else { n_a };
        let (start, end, n_mine, off) = if blk == 0 { (start_a, end_a, n_a, 0usize) } else { (start_b, end_b, if has_b { n_b } else { 0 }, off_b) };

        let mut m0 = -1e30f32;
        let mut m1 = -1e30f32;
        let mut d0 = 0.0f32;
        let mut d1 = 0.0f32;
        let mut o = [[0.0f32; 4]; 16];

        let f8 = if fp8.is_null() {
            Fp8Kv { k8: core::ptr::null(), v8: core::ptr::null(), ks: core::ptr::null(), vs: core::ptr::null(), cap: 1, before: i32::MIN }
        } else {
            let (k8, v8, ks, vs) = crate::attn::fp8_shadow_parts(fp8, fp8_cap);
            Fp8Kv { k8, v8, ks, vs, cap: fp8_cap, before: p_a - (REL_EXTENT as i32 - 1) }
        };
        // Block B's band: rows [before_A, before_A + 16) as FP8 (block A reads them as BF16 from the window).
        let (band_lo, band_hi) = if !fp8.is_null() && blk == 1 && has_b { (f8.before, f8.before + BAND_ROWS as i32) } else { (i32::MIN, i32::MIN) };
        if !fp8.is_null() && has_b {
            let mut idx = tid;
            while idx < BAND_ROWS * 16 {
                let r = idx >> 4;
                let sub = idx & 15;
                let t = f8.before + r as i32;
                if t >= 0 && t < end_b {
                    let slot = (t % f8.cap) as usize;
                    let o8 = slot * KV_DIM + kvh * HEAD_DIM + sub * 8;
                    let kc = *(f8.k8.add(o8) as *const [u32; 2]);
                    let vc = *(f8.v8.add(o8) as *const [u32; 2]);
                    let si = slot * NUM_KV_HEADS + kvh;
                    let ksc = *f8.ks.add(si) as u32;
                    let vsc = *f8.vs.add(si) as u32;
                    st_u4(band_k.add(r * KV_ROW + sub * 8), crate::attn::e4m3x8_to_h2x4([kc[0], kc[1], ksc, 0]));
                    st_u4(band_v.add(r * KV_ROW + sub * 8), crate::attn::e4m3x8_to_h2x4([vc[0], vc[1], vsc, 0]));
                }
                idx += THREADS;
            }
        }

        if n_it > 0 {
            load_tile(stage_k(0), stage_v(0), stage_sc(0), k_cache, v_cache, kv_cap, kvh, start_a, core::cmp::min(rows as i32, end_ld - start_a) as usize, rows, tid, &f8);
            cp_async_commit();
        }
        let mut it = 0i32;
        #[nounroll]
        while it < n_it {
            let st = (it & 1) as usize;
            let base = start_a + it * FT as i32;
            if it + 1 < n_it {
                let b1 = base + FT as i32;
                load_tile(stage_k(st ^ 1), stage_v(st ^ 1), stage_sc(st ^ 1), k_cache, v_cache, kv_cap, kvh, b1, core::cmp::max(0, core::cmp::min(rows as i32, end_ld - b1)) as usize, rows, tid, &f8);
                cp_async_commit();
                cp_async_wait1();
            } else {
                cp_async_wait0();
            }
            if !fp8.is_null() {
                // The FP8 scales of a row were loaded by its sub-0 thread.
                sync_threads();
            }
            convert_tile(stage_k(st), stage_v(st), stage_sc(st), base, core::cmp::max(0, core::cmp::min(rows as i32, end_ld - base)) as usize, rows, tid, kvh, &f8);
            sync_threads();
            if it < n_mine {
                let s_k = stage_k(st).add(off * KV_ROW) as *const u16;
                let s_v = stage_v(st).add(off * KV_ROW) as *const u16;
                let t0 = start + it * FT as i32;
                let tile_len = core::cmp::min(FT as i32, end - t0) as usize;
                // Tiles where no pair is masked or within REL_EXTENT: `biased` returns the score unchanged.
                let t_last = t0 + tile_len as i32 - 1;
                let special = tile_len < FT || t_last > p_first || (window > 0 && t0 < p_last + 1 - window) || p_first - t_last < REL_EXTENT as i32;

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
                        let tk = t0 + k_ld_row as i32;
                        let k_row = if tk >= band_lo && tk < band_hi { band_k.add((tk - band_lo) as usize * KV_ROW) as *const u16 } else { s_k.add(k_ld_row * KV_ROW) };
                        let mut kt = 0;
                        #[unroll]
                        while kt < 8 {
                            let kb = ldmatrix_x4(k_row.add(kt * 16 + k_ld_col));
                            sa = mma_m16n8k16_f32_f16(sa, q_reg[kt], [kb[0], kb[1]]);
                            sbv = mma_m16n8k16_f32_f16(sbv, q_reg[kt], [kb[2], kb[3]]);
                            kt += 1;
                        }
                        let tb = t0 + t_base as i32;
                        let (mut s0, mut s1, mut s4, mut s5) = (sa[0], sa[1], sbv[0], sbv[1]);
                        let (mut s2, mut s3, mut s6, mut s7) = (sa[2], sa[3], sbv[2], sbv[3]);
                        if special {
                            s0 = biased(s0, tb + c0 as i32, c0 < w_len, pos_q0, window, r0, rel16);
                            s1 = biased(s1, tb + c1 as i32, c1 < w_len, pos_q0, window, r0, rel16);
                            s4 = biased(s4, tb + c4 as i32, c4 < w_len, pos_q0, window, r0, rel16);
                            s5 = biased(s5, tb + c5 as i32, c5 < w_len, pos_q0, window, r0, rel16);
                            s2 = biased(s2, tb + c0 as i32, c0 < w_len, pos_q1, window, r1, rel16);
                            s3 = biased(s3, tb + c1 as i32, c1 < w_len, pos_q1, window, r1, rel16);
                            s6 = biased(s6, tb + c4 as i32, c4 < w_len, pos_q1, window, r1, rel16);
                            s7 = biased(s7, tb + c5 as i32, c5 < w_len, pos_q1, window, r1, rel16);
                        }

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
                        let tv = t0 + v_ld_row as i32;
                        let v_row = if tv >= band_lo && tv < band_hi { band_v.add((tv - band_lo) as usize * KV_ROW) as *const u16 } else { s_v.add(v_ld_row * KV_ROW) };
                        let mut nt = 0;
                        #[unroll]
                        while nt < 16 {
                            let vb = ldmatrix_x2_trans(v_row.add(nt * 8));
                            o[nt] = mma_m16n8k16_f32_f16(o[nt], pa, vb);
                            nt += 1;
                        }
                    }
                    sb += 1;
                }
            }
            sync_threads();
            it += 1;
        }
        sync_threads();

        // Per block: merge the two key slices of each M tile with the sink, then apply the gate.
        let red_m = red; // [16][2]
        let red_d = red.add(32); // [16][2]
        let red_o = (stage_k(0) as *mut f32).add(blk * 2 * 16 * HEAD_DIM); // [2][16][128] per block
        let mut mt = 0;
        while mt < 3 {
            if active && m_tile == mt {
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
            if active {
                let mut i = btid;
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
                    i += BT;
                }
            }
            sync_threads();
            mt += 1;
        }
    }
}
