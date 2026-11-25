//! Wide prefill MoE on Blackwell's block-scaled FP4 tensor cores (`INKLING_WIDE_MOE=bs`): the grouped expert
//! GEMMs with the NVFP4 weights fed to `mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` as stored
//! (E2M1 codes and one E4M3 scale per 16 values), with no decode to f16. The design follows squid5090's
//! `p_wgemm_blkscale_kernel` (cuda/kernels_prefill.cu, Apache-2.0).
//!
//! The instruction needs both operands in NVFP4, so the activations are quantized first
//! (`prefill_quant_f16_to_nvfp4`: per 16 values, an E4M3 scale of amax / 6 and E2M1 codes). That is a real loss
//! of precision on the activations (4 bits each), unlike every other prefill kernel, which is why this path is
//! opt-in: judge it with `inkling kld`.
//!
//! Tiles are `prefill_moe_tiles`' (64 sorted pairs of one expert). Grid (rows / 128, max tiles), 256 threads:
//! each warp owns 16 weight rows (the N side, two m16n8 tiles) against the tile's 64 pairs (the M side, four
//! tiles). K runs in stages of 128 values (64 code bytes and 8 scale bytes per row), double-buffered in shared
//! memory with `cp.async`; each stage is two k64 MMAs per (M, N) tile.
//!
//! Fragment layout of the instruction, for lane = 4 g + t: A registers hold rows g and g + 8, code bytes
//! 4 t .. 4 t + 4 of the first and of the second 16 bytes of the 32-byte K block; B registers hold weight row g
//! (the column), the same bytes; the A scale register holds the block's four scales of row g (t even) or g + 8
//! (t odd), the B scale register those of weight row g. C/D: [2 half + c] = row g + 8 half, column 2 t + c.

use crate::common::*;
use crate::nvfp4::f32_to_e4m3;
use crate::{GATE_SOFTCAP, HIDDEN, INTERMEDIATE, TOP_K, UP_SOFTCAP, cvt_e4m3_to_float, fast_sigmoid, fast_tanh, float2half, h2f_hi, h2f_lo};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::{DynamicSharedArray, device, kernel, launch_bounds, ptx_asm};

const TP: usize = 64; // pairs per tile
const TW: usize = 128; // weight rows per CTA
const BKB: usize = 64; // code bytes per K stage (128 values)
const STRIDE: usize = 80; // shared-memory pitch of a code row (64 + 16: no bank conflicts between rows)
const SCB: usize = 8; // scale bytes per K stage
const A_CODE: usize = TP * STRIDE;
const A_SC: usize = TP * SCB;
const W_CODE: usize = TW * STRIDE;
const W_SC: usize = TW * SCB;
const STAGE: usize = A_CODE + A_SC + W_CODE + W_SC; // 16,896 B
/// Dynamic shared memory per CTA: two K stages.
pub const MOE_BS_SMEM: usize = 2 * STAGE;
const THREADS: usize = 256;

#[inline(always)]
unsafe fn cp_async16(dst: *mut u8, src: *const u8) {
    let d = unsafe { cvta_generic_to_shared_u32(dst as *const u8) };
    unsafe {
        ptx_asm!("cp.async.cg.shared.global [%0], [%1], 16;", in("r") d, in("l") src, clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_async8(dst: *mut u8, src: *const u8) {
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

/// Two f32 to one byte of E2M1 codes (`lo` in the low nibble), round to nearest, saturating.
#[inline(always)]
fn f32x2_to_e2m1x2(lo: f32, hi: f32) -> u32 {
    let r: u16;
    unsafe {
        ptx_asm!(
            "{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %2, %1; mov.b16 %0, {b, b}; }",
            out("=h") r,
            in("f") lo,
            in("f") hi,
            options(register_only),
        );
    }
    (r & 0xff) as u32
}

/// D = A B + C over one warp: A 16 x 64 and B 64 x 8 in E2M1 with E4M3 scales per 16 values, f32 accumulation.
#[inline(always)]
unsafe fn mma_nvfp4(acc: &mut [f32; 4], a: [u32; 4], b0: u32, b1: u32, sa: u32, sb: u32) {
    let [mut c0, mut c1, mut c2, mut c3] = *acc;
    unsafe {
        ptx_asm!(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};",
            inout("+f") c0,
            inout("+f") c1,
            inout("+f") c2,
            inout("+f") c3,
            in("r") a[0],
            in("r") a[1],
            in("r") a[2],
            in("r") a[3],
            in("r") b0,
            in("r") b1,
            in("r") sa,
            in("r") sb,
            options(register_only),
        );
    }
    *acc = [c0, c1, c2, c3];
}

/// Quantizes f16 rows to NVFP4: block `i` is values 16 i .. 16 i + 16 of `x_h`; its 8 code bytes go to
/// `out_code[8 i ..]` (value 2 j in the low nibble of byte j) and its E4M3 scale to `out_scale[i]`. The scale is
/// amax / 6 (at least 0.002, the smallest E4M3 values round a zero block's scale to zero otherwise).
/// Grid ceil(n_blocks / 256), 256 threads.
#[kernel]
#[launch_bounds(256)]
pub unsafe fn prefill_quant_f16_to_nvfp4(x_h: *const u16, out_code: *mut u8, out_scale: *mut u8, n_blocks: i32) {
    unsafe {
        let i = blockIdx_x() as usize * 256 + threadIdx_x() as usize;
        if i >= n_blocks as usize {
            return;
        }
        let src = x_h.add(i * 16) as *const u8;
        let q0 = ldg_u4(src);
        let q1 = ldg_u4(src.add(16));
        let mut f = [0.0f32; 16];
        let mut amax = 0.0f32;
        let mut j = 0;
        #[unroll]
        while j < 4 {
            f[2 * j] = h2f_lo(q0[j]);
            f[2 * j + 1] = h2f_hi(q0[j]);
            f[8 + 2 * j] = h2f_lo(q1[j]);
            f[8 + 2 * j + 1] = h2f_hi(q1[j]);
            j += 1;
        }
        let mut j = 0;
        #[unroll]
        while j < 16 {
            amax = fmaxf(amax, fmaxf(f[j], -f[j]));
            j += 1;
        }
        let sc8 = f32_to_e4m3(fminf(fmaxf(amax * (1.0 / 6.0), 0.002), 448.0));
        let inv = div_rn(1.0, cvt_e4m3_to_float(sc8 as u16));
        let mut w = [0u32; 2];
        let mut j = 0;
        #[unroll]
        while j < 8 {
            w[j >> 2] |= f32x2_to_e2m1x2(f[2 * j] * inv, f[2 * j + 1] * inv) << (8 * (j & 3));
            j += 1;
        }
        *(out_code.add(i * 8) as *mut [u32; 2]) = w;
        *out_scale.add(i) = sc8;
    }
}

/// Queues K stage `ks` into `st`: the tile's activation codes and scales and this CTA's weight codes and scales.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn load_stage(st: *mut u8, ks: usize, a_code: *const u8, a_sc: *const u8, w_code: *const u8, w_sc: *const u8, cb: usize, sb: usize, tid: usize) {
    unsafe {
        let (sm_ac, sm_as) = (st, st.add(A_CODE));
        let (sm_wc, sm_ws) = (st.add(A_CODE + A_SC), st.add(A_CODE + A_SC + W_CODE));
        // Activation codes: 64 rows x four 16-byte chunks, one per thread. Rows past the tile's pairs are zero.
        let (ar, ach) = (tid >> 2, tid & 3);
        let dst = sm_ac.add(ar * STRIDE + ach * 16);
        if a_code.is_null() {
            *(dst as *mut [u32; 4]) = [0; 4];
        } else {
            cp_async16(dst, a_code.add(ks * BKB + ach * 16));
        }
        if tid < TP {
            let dst = sm_as.add(tid * SCB);
            if a_sc.is_null() {
                *(dst as *mut [u32; 2]) = [0; 2];
            } else {
                cp_async8(dst, a_sc.add(ks * SCB));
            }
        }
        // Weight codes: 128 rows x four chunks, two per thread.
        let mut i = 0;
        #[unroll]
        while i < 2 {
            let id = tid + i * THREADS;
            let (r, ch) = (id >> 2, id & 3);
            cp_async16(sm_wc.add(r * STRIDE + ch * 16), w_code.add(r * cb + ks * BKB + ch * 16));
            i += 1;
        }
        if tid < TW {
            cp_async8(sm_ws.add(tid * SCB), w_sc.add(tid * sb + ks * SCB));
        }
        cp_async_commit();
    }
}

/// The tiled GEMM. UP: activations are the NVFP4 rows of the tokens (`a_codes` [T][HIDDEN / 2]), w = w13, out =
/// act (sorted pairs). Down: activations are the NVFP4 rows of the sorted pairs ([P][INTERMEDIATE / 2]), w = w2,
/// out = slots[pair id] (times the route weight).
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn moe_bs_body<const UP: bool>(
    smem: *mut u8,
    a_codes: *const u8,
    a_scales: *const u8,
    offs: *const i32,
    pairs: *const i32,
    tiles: *const i32,
    tiles_n: *const i32,
    route_w: *const f32,
    w: *const u8,
    ws: *const u8,
    wfactor: *const f32,
    out_act: *mut u16,
    out_slots: *mut f32,
) {
    unsafe {
        let k_dim = if UP { HIDDEN } else { INTERMEDIATE };
        let rows_per_e = if UP { 2 * INTERMEDIATE } else { HIDDEN };
        let tile = blockIdx_y() as usize;
        if tile as i32 >= *tiles_n {
            return;
        }
        let e = *tiles.add(2 * tile) as usize;
        let p0 = *tiles.add(2 * tile + 1) as usize;
        let n_valid = core::cmp::min(TP, *offs.add(e + 1) as usize - p0);
        let n0 = blockIdx_x() as usize * TW;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let g = lane >> 2;
        let t = lane & 3;
        let cb = k_dim >> 1;
        let sb = k_dim >> 4;

        // Row of the activation buffers for tile row r.
        let a_row = |r: usize| -> usize { if UP { *pairs.add(p0 + r) as usize / TOP_K } else { p0 + r } };
        let ar = tid >> 2;
        let a_code: *const u8 = if ar < n_valid { a_codes.add(a_row(ar) * cb) } else { core::ptr::null() };
        let a_sc: *const u8 = if tid < n_valid { a_scales.add(a_row(tid) * sb) } else { core::ptr::null() };
        let w_code = w.add((e * rows_per_e + n0) * cb);
        let w_sc = ws.add((e * rows_per_e + n0) * sb);

        let n_stage = k_dim / (2 * BKB);
        let m_tiles = n_valid.div_ceil(16);
        let mut acc = [[[0.0f32; 4]; 2]; 4];
        load_stage(smem, 0, a_code, a_sc, w_code, w_sc, cb, sb, tid);
        let mut ks = 0usize;
        #[nounroll]
        while ks < n_stage {
            if ks + 1 < n_stage {
                load_stage(smem.add(((ks + 1) & 1) * STAGE), ks + 1, a_code, a_sc, w_code, w_sc, cb, sb, tid);
                cp_async_wait1();
            } else {
                cp_async_wait0();
            }
            sync_threads();
            let st = smem.add((ks & 1) * STAGE) as *const u8;
            let (sm_ac, sm_as) = (st, st.add(A_CODE));
            let (sm_wc, sm_ws) = (st.add(A_CODE + A_SC), st.add(A_CODE + A_SC + W_CODE));
            let mut kk = 0;
            #[unroll]
            while kk < 2 {
                let k_off = kk * 32 + t * 4;
                let mut b0 = [0u32; 2];
                let mut b1 = [0u32; 2];
                let mut sbv = [0u32; 2];
                let mut nt = 0;
                #[unroll]
                while nt < 2 {
                    let wrow = warp_i * 16 + nt * 8 + g;
                    b0[nt] = *(sm_wc.add(wrow * STRIDE + k_off) as *const u32);
                    b1[nt] = *(sm_wc.add(wrow * STRIDE + k_off + 16) as *const u32);
                    sbv[nt] = *(sm_ws.add(wrow * SCB + kk * 4) as *const u32);
                    nt += 1;
                }
                let mut mt = 0;
                #[unroll]
                while mt < 4 {
                    if mt < m_tiles {
                        let r0 = mt * 16 + g;
                        let r1 = r0 + 8;
                        let a = [
                            *(sm_ac.add(r0 * STRIDE + k_off) as *const u32),
                            *(sm_ac.add(r1 * STRIDE + k_off) as *const u32),
                            *(sm_ac.add(r0 * STRIDE + k_off + 16) as *const u32),
                            *(sm_ac.add(r1 * STRIDE + k_off + 16) as *const u32),
                        ];
                        let sa = *(sm_as.add((r0 + (t & 1) * 8) * SCB + kk * 4) as *const u32);
                        mma_nvfp4(&mut acc[mt][0], a, b0[0], b1[0], sa, sbv[0]);
                        mma_nvfp4(&mut acc[mt][1], a, b0[1], b1[1], sa, sbv[1]);
                    }
                    mt += 1;
                }
                kk += 1;
            }
            sync_threads();
            ks += 1;
        }

        let es = *wfactor.add(e);
        let mut mt = 0;
        #[unroll]
        while mt < 4 {
            let mut half = 0;
            #[unroll]
            while half < 2 {
                let row = mt * 16 + g + half * 8;
                if row < n_valid {
                    let mut nt = 0;
                    #[unroll]
                    while nt < 2 {
                        let cl = n0 + warp_i * 16 + nt * 8 + t * 2;
                        let c0 = acc[mt][nt][2 * half] * es;
                        let c1 = acc[mt][nt][2 * half + 1] * es;
                        if UP {
                            // Rows 2 p (gate) and 2 p + 1 (up) of the expert give intermediate column p.
                            let g_cap = fast_tanh(c0 * (1.0 / GATE_SOFTCAP)) * GATE_SOFTCAP;
                            let u_cap = fast_tanh(c1 * (1.0 / UP_SOFTCAP)) * UP_SOFTCAP;
                            *out_act.add((p0 + row) * INTERMEDIATE + cl / 2) = float2half((g_cap * fast_sigmoid(c0)) * u_cap);
                        } else {
                            let pid = *pairs.add(p0 + row) as usize;
                            let rw = *route_w.add(pid);
                            *out_slots.add(pid * HIDDEN + cl) = c0 * rw;
                            *out_slots.add(pid * HIDDEN + cl + 1) = c1 * rw;
                        }
                        nt += 1;
                    }
                }
                half += 1;
            }
            mt += 1;
        }
    }
}

/// Gate/up: grid (2 I / 128 = 18, max tiles), 256 threads, MOE_BS_SMEM bytes of dynamic shared memory.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_moe_up_bs(
    a_codes: *const u8,
    a_scales: *const u8,
    offs: *const i32,
    pairs: *const i32,
    tiles: *const i32,
    tiles_n: *const i32,
    w: *const u8,
    ws: *const u8,
    wfactor: *const f32,
    act: *mut u16,
) {
    unsafe {
        moe_bs_body::<true>(
            DynamicSharedArray::<u8, 16>::get(),
            a_codes,
            a_scales,
            offs,
            pairs,
            tiles,
            tiles_n,
            core::ptr::null(),
            w,
            ws,
            wfactor,
            act,
            core::ptr::null_mut(),
        )
    }
}

/// Down: grid (HIDDEN / 128 = 18, max tiles), 256 threads, MOE_BS_SMEM bytes of dynamic shared memory.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_moe_down_bs(
    a_codes: *const u8,
    a_scales: *const u8,
    offs: *const i32,
    pairs: *const i32,
    tiles: *const i32,
    tiles_n: *const i32,
    route_w: *const f32,
    w: *const u8,
    ws: *const u8,
    wfactor: *const f32,
    slots: *mut f32,
) {
    unsafe {
        moe_bs_body::<false>(
            DynamicSharedArray::<u8, 16>::get(),
            a_codes,
            a_scales,
            offs,
            pairs,
            tiles,
            tiles_n,
            route_w,
            w,
            ws,
            wfactor,
            core::ptr::null_mut(),
            slots,
        )
    }
}
