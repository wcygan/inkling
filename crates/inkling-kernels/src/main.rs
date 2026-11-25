//! cuda-oxide ports of squid5090's decode kernels.
//!
//! This file holds the K0 spike kernels and the primitives they share; the
//! exact-mode B=1 kernels live in the modules declared at the bottom.
//!
//! Each kernel is a line-for-line port of tensorfish/squid5090 @ 3ee093c,
//! `cuda/kernels_gemv.cu`:
//! - `lm_head_stage1_kernel<B, 2>` (NVFP4, line 871) as `lm_head_stage1_nvfp4_b{1,2,4,8}`;
//! - `moe_up_tc_kernel<0>` (NVFP4, line 1438) as `moe_up_tc_nvfp4`.
//!
//! Kernel parameters keep the CUDA C++ order and types, so one host harness
//! launches both builds with the same argument list. Helpers that the
//! cuda-oxide catalog provides (f16x2 arithmetic, the f32-accumulate f16 MMA,
//! shuffles, ballots, PDL, `tanh.approx`, `ex2.approx`, E4M3 decode) use the
//! catalog. Instructions it lacks (`cvt.rn.f16x2.e2m1x2`, cache-hinted and
//! dependency loads, `cvt.rn.f16.f32`, `rcp.rn.f32`) use `ptx_asm!` with the
//! exact PTX squid5090's inline assembly or nvcc emits.

use cuda_device::convert::cvt_rn_f16x2_e4m3x2;
use cuda_device::f16x2::{add_f16x2, fma_f16x2, mul_f16x2};
use cuda_device::float::{ex2_approx_f32, tanh_approx_f32};
use cuda_device::grid::dependency;
use cuda_device::thread::{blockIdx_x, blockIdx_y, gridDim_x, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{SharedArray, device, kernel, launch_bounds, ptx_asm, warp};
use cuda_host::cuda_module;

// squid5090 cuda/config.hpp and kernels_gemv.cu constants.
pub const HIDDEN: usize = 2304;
pub const INTERMEDIATE: usize = 1152;
pub const TOP_K: usize = 8;
pub const LM_HEAD_BLOCKS: usize = 680;
pub const FINAL_SOFTCAP: f32 = 30.0;
pub const GATE_SOFTCAP: f32 = 4.0;
pub const UP_SOFTCAP: f32 = 25.0;
pub const MOE_TILE: usize = 16;
pub const MOE_MAXM: usize = 8;
pub const MOE_UP_WARPS: usize = HIDDEN / 128; // 18 warps (576 threads)

// ---------------------------------------------------------------------------
// kernels_common.cuh primitives
// ---------------------------------------------------------------------------

/// `pdl_wait()`: `griddepcontrol.wait`.
#[inline(always)]
pub(crate) fn pdl_wait() {
    unsafe { dependency::wait() }
}

/// `pdl_trigger()`: `griddepcontrol.launch_dependents`.
#[inline(always)]
pub(crate) fn pdl_trigger() {
    unsafe { dependency::trigger_dependents() }
}

/// `ld_stream_u4`: streaming 128-bit weight load that bypasses L1.
#[inline(always)]
pub(crate) unsafe fn ld_stream_u4(p: *const u8) -> [u32; 4] {
    let (x, y, z, w): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];",
            out("=r") x,
            out("=r") y,
            out("=r") z,
            out("=r") w,
            in("l") p,
        );
    }
    [x, y, z, w]
}

/// `ld_scale_u16`: read-only 16-bit scale load.
#[inline(always)]
pub(crate) unsafe fn ld_scale_u16(p: *const u8) -> u16 {
    let v: u16;
    unsafe {
        ptx_asm!("ld.global.nc.u16 %0, [%1];", out("=h") v, in("l") p);
    }
    v
}

/// `ld_dep_u4`: coherent, non-hoistable 128-bit load of PDL-produced data.
#[inline(always)]
pub(crate) unsafe fn ld_dep_u4(p: *const u16) -> [u32; 4] {
    let (x, y, z, w): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];",
            out("=r") x,
            out("=r") y,
            out("=r") z,
            out("=r") w,
            in("l") p,
            clobber("memory"),
        );
    }
    [x, y, z, w]
}

/// `ld_dep_i32`.
#[inline(always)]
pub(crate) unsafe fn ld_dep_i32(p: *const i32) -> i32 {
    let v: u32;
    unsafe {
        ptx_asm!("ld.global.s32 %0, [%1];", out("=r") v, in("l") p, clobber("memory"));
    }
    v as i32
}

/// `ld_dep_f32`.
#[inline(always)]
pub(crate) unsafe fn ld_dep_f32(p: *const f32) -> f32 {
    let v: f32;
    unsafe {
        ptx_asm!("ld.global.f32 %0, [%1];", out("=f") v, in("l") p, clobber("memory"));
    }
    v
}

/// `__ldg(const uint2*)`: `ld.global.nc.v2.u32`.
#[inline(always)]
pub(crate) unsafe fn ldg_u2(p: *const u8) -> [u32; 2] {
    let (x, y): (u32, u32);
    unsafe {
        ptx_asm!(
            "ld.global.nc.v2.u32 {%0,%1}, [%2];",
            out("=r") x,
            out("=r") y,
            in("l") p,
            options(register_only),
        );
    }
    [x, y]
}

/// `__ldg(const uint8_t*)`: `ld.global.nc.u8`.
#[inline(always)]
pub(crate) unsafe fn ldg_u8(p: *const u8) -> u16 {
    let v: u16;
    unsafe {
        ptx_asm!("ld.global.nc.u8 %0, [%1];", out("=h") v, in("l") p, options(register_only));
    }
    v
}

/// `cvt_e2m1x8_to_half2x4`: 8 FP4 weights (one u32) to four half2 words with
/// four `cvt.rn.f16x2.e2m1x2`. Each `.b8` operand is split from the `.b32`
/// with `mov.b32 {b0,b1,b2,b3}`.
#[inline(always)]
pub(crate) fn cvt_e2m1x8_to_half2x4(w: u32) -> [u32; 4] {
    let (r0, r1, r2, r3): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "{ .reg .b8 b0, b1, b2, b3;
  mov.b32 {b0, b1, b2, b3}, %4;
  cvt.rn.f16x2.e2m1x2 %0, b0;
  cvt.rn.f16x2.e2m1x2 %1, b1;
  cvt.rn.f16x2.e2m1x2 %2, b2;
  cvt.rn.f16x2.e2m1x2 %3, b3; }",
            out("=r") r0,
            out("=r") r1,
            out("=r") r2,
            out("=r") r3,
            in("r") w,
            options(register_only),
        );
    }
    [r0, r1, r2, r3]
}

/// `cvt_e4m3x2_to_half2(uint16_t)`.
#[inline(always)]
pub(crate) fn cvt_e4m3x2_to_half2(packed: u16) -> u32 {
    cvt_rn_f16x2_e4m3x2(packed)
}

/// `cvt_e4m3_to_float(uint8_t)`: E4M3 to half (exact), then to float (exact).
#[inline(always)]
pub(crate) fn cvt_e4m3_to_float(code: u16) -> f32 {
    h2f_lo(cvt_rn_f16x2_e4m3x2(code))
}

/// Low half of a half2 word as float (`cvt.f32.f16`).
#[inline(always)]
pub(crate) fn h2f_lo(h2: u32) -> f32 {
    let lo = h2 as u16;
    let f: f32;
    unsafe {
        ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") f, in("h") lo, options(register_only));
    }
    f
}

/// High half of a half2 word as float.
#[inline(always)]
pub(crate) fn h2f_hi(h2: u32) -> f32 {
    let hi = (h2 >> 16) as u16;
    let f: f32;
    unsafe {
        ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") f, in("h") hi, options(register_only));
    }
    f
}

/// `__float2half`: `cvt.rn.f16.f32`.
#[inline(always)]
pub(crate) fn float2half(x: f32) -> u16 {
    let h: u16;
    unsafe {
        ptx_asm!("cvt.rn.f16.f32 %0, %1;", out("=h") h, in("f") x, options(register_only));
    }
    h
}

/// `fmaf`: `fma.rn.f32`.
#[inline(always)]
pub(crate) fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    cuda_device::float::fma_rn_f32(a, b, c)
}

/// `fast_tanh`: `tanh.approx.f32`.
#[inline(always)]
pub(crate) fn fast_tanh(x: f32) -> f32 {
    tanh_approx_f32(x)
}

/// `fast_sigmoid`: `1 / (1 + __expf(-x))`. nvcc lowers `__expf(-x)` to
/// `ex2.approx.f32(x * -log2(e))` and the reciprocal to `rcp.rn.f32`.
#[inline(always)]
pub(crate) fn fast_sigmoid(x: f32) -> f32 {
    let e = ex2_approx_f32(x * -core::f32::consts::LOG2_E);
    let d = e + 1.0;
    let r: f32;
    unsafe {
        ptx_asm!("rcp.rn.f32 %0, %1;", out("=f") r, in("f") d, options(register_only));
    }
    r
}

/// `half_warp_sum`.
#[inline(always)]
pub(crate) fn half_warp_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 8);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 4);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 2);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 1);
    v
}

/// `__low2half2` / `__high2half2`.
#[inline(always)]
pub(crate) fn low2half2(h2: u32) -> u32 {
    (h2 & 0xffff) | (h2 << 16)
}

#[inline(always)]
pub(crate) fn high2half2(h2: u32) -> u32 {
    (h2 >> 16) | (h2 & 0xffff_0000)
}

// ---------------------------------------------------------------------------
// kernels_gemv.cu NVFP4 chunk helpers
// ---------------------------------------------------------------------------

/// `Nvfp4Chunk32`: 32 FP4 codes (16 bytes) and two E4M3 scales.
#[derive(Clone, Copy)]
pub(crate) struct Nvfp4Chunk32 {
    pub(crate) q: [u32; 4],
    pub(crate) s: u16,
}

/// `load_nvfp4_chunk32`.
#[inline(always)]
pub(crate) unsafe fn load_nvfp4_chunk32(codes: *const u8, scales: *const u8, c: usize) -> Nvfp4Chunk32 {
    unsafe {
        Nvfp4Chunk32 {
            q: ld_stream_u4(codes.add(c * 16)),
            s: ld_scale_u16(scales.add(c * 2)),
        }
    }
}

/// `decode_nvfp4_chunk32_f16`: exact FP16 decode of 32 NVFP4 weights times
/// their two E4M3 block scales into 16 half2 words.
#[device]
pub(crate) fn decode_nvfp4_chunk32_f16(ch: &Nvfp4Chunk32, o: &mut [u32; 16]) {
    let sc = cvt_e4m3x2_to_half2(ch.s);
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

// ---------------------------------------------------------------------------
// lm_head_stage1_kernel<B_CONST, KIND = 2> (NVFP4)
// ---------------------------------------------------------------------------

pub(crate) const LM_TPR: usize = 16;
pub(crate) const LM_RPB: usize = 256 / LM_TPR; // 16 rows per block
pub(crate) const LM_MAX_B: usize = 8;

/// Body of `lm_head_stage1_kernel<B, 2>`. The kernel wrappers own the shared
/// arrays (their sizes depend on B) and pass them in.
#[device]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn lm_head_stage1_body<const B: usize>(
    s_x16: *mut u16,
    sm_val: *mut f32,
    sm_idx: *mut i32,
    w_data: *const u8,
    w_scales: *const u8,
    global_scale: f32,
    x: *const f32,
    out_logits: *mut f32,
    blk_max_val: *mut f32,
    blk_max_idx: *mut i32,
    sample_cfg: *const f32,
    v: i32,
) {
    unsafe {
        // sample_cfg->inv_temp is the first field of SampleConfig.
        let write_logits = !out_logits.is_null() && (sample_cfg.is_null() || *sample_cfg > 0.0);
        let tid = threadIdx_x() as usize;
        let total_x = B * HIDDEN;
        pdl_wait();
        pdl_trigger();
        let mut i = tid;
        while i < total_x {
            *s_x16.add(i) = float2half(ld_dep_f32(x.add(i)));
            i += 256;
        }
        sync_threads();
        let s_xh2 = s_x16 as *const u32;

        let r_in_blk = tid >> 4;
        let lane = tid & (LM_TPR - 1);
        let global_r = blockIdx_x() as usize * LM_RPB + r_in_blk;
        let total_r = gridDim_x() as usize * LM_RPB;

        let mut best_val = [-1e30f32; B];
        let mut best_idx = [0i32; B];

        let mut row = global_r;
        while row < v as usize {
            let mut acc = [0.0f32; B];

            let wr = w_data.add(row * (HIDDEN >> 1));
            let sr = w_scales.add(row * (HIDDEN >> 4));
            let mut iter = 0;
            #[unroll]
            while iter < 9 {
                let blk = lane + iter * 16;
                let w8 = ldg_u2(wr.add(blk * 8));
                let sc = cvt_e4m3_to_float(ldg_u8(sr.add(blk)));
                let lo = cvt_e2m1x8_to_half2x4(w8[0]);
                let hi = cvt_e2m1x8_to_half2x4(w8[1]);
                let wh = [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]];
                let koff = blk << 4;
                let mut b = 0;
                #[unroll]
                while b < B {
                    let xb = s_xh2.add((b * HIDDEN + koff) >> 1);
                    let mut p0 = mul_f16x2(wh[0], *xb);
                    let mut p1 = mul_f16x2(wh[1], *xb.add(1));
                    p0 = fma_f16x2(wh[2], *xb.add(2), p0);
                    p1 = fma_f16x2(wh[3], *xb.add(3), p1);
                    p0 = fma_f16x2(wh[4], *xb.add(4), p0);
                    p1 = fma_f16x2(wh[5], *xb.add(5), p1);
                    p0 = fma_f16x2(wh[6], *xb.add(6), p0);
                    p1 = fma_f16x2(wh[7], *xb.add(7), p1);
                    let s = add_f16x2(p0, p1);
                    acc[b] = fmaf(h2f_lo(s) + h2f_hi(s), sc, acc[b]);
                    b += 1;
                }
                iter += 1;
            }

            let scale = global_scale;
            let mut b = 0;
            #[unroll]
            while b < B {
                let val = half_warp_sum(acc[b]) * scale;
                if lane == 0 {
                    if write_logits {
                        let cap_val = fast_tanh(val * (1.0 / FINAL_SOFTCAP)) * FINAL_SOFTCAP;
                        *out_logits.add(b * v as usize + row) = cap_val;
                        if cap_val > best_val[b] {
                            best_val[b] = cap_val;
                            best_idx[b] = row as i32;
                        }
                    } else if val > best_val[b] {
                        best_val[b] = val;
                        best_idx[b] = row as i32;
                    }
                }
                b += 1;
            }
            row += total_r;
        }

        if lane == 0 {
            let mut b = 0;
            #[unroll]
            while b < B {
                *sm_val.add(b * LM_RPB + r_in_blk) = best_val[b];
                *sm_idx.add(b * LM_RPB + r_in_blk) = best_idx[b];
                b += 1;
            }
        }
        sync_threads();
        if tid < B {
            let b = tid;
            let mut bv = *sm_val.add(b * LM_RPB);
            let mut bi = *sm_idx.add(b * LM_RPB);
            let mut w = 1;
            while w < LM_RPB {
                let cv = *sm_val.add(b * LM_RPB + w);
                let ci = *sm_idx.add(b * LM_RPB + w);
                if cv > bv || (cv == bv && ci < bi) {
                    bv = cv;
                    bi = ci;
                }
                w += 1;
            }
            *blk_max_val.add(b * LM_HEAD_BLOCKS + blockIdx_x() as usize) = bv;
            *blk_max_idx.add(b * LM_HEAD_BLOCKS + blockIdx_x() as usize) = bi;
        }
    }
}

// ---------------------------------------------------------------------------
// moe_up_tc_kernel<FMT = 0> (NVFP4)
// ---------------------------------------------------------------------------

pub(crate) const MOE_K: usize = HIDDEN;
pub(crate) const MOE_R: usize = 2 * INTERMEDIATE;
pub(crate) const MOE_CB: usize = MOE_K >> 1;
pub(crate) const MOE_SB: usize = MOE_K >> 4;
pub(crate) const MOE_TILE_OUT_STRIDE: usize = MOE_MAXM + 1;

/// `MoeFrag<0>`.
pub(crate) struct MoeFrag {
    pub(crate) r0: Nvfp4Chunk32,
    pub(crate) r1: Nvfp4Chunk32,
    pub(crate) x: [u32; 16],
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `lm_head_stage1_kernel<1, 2>`.
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn lm_head_stage1_nvfp4_b1(
        w_data: *const u8,
        w_scales: *const u8,
        _row_scales: *const f32,
        global_scale: f32,
        x: *const f32,
        out_logits: *mut f32,
        blk_max_val: *mut f32,
        blk_max_idx: *mut i32,
        sample_cfg: *const f32,
        v: i32,
    ) {
        static mut S_X16: SharedArray<u16, { HIDDEN }, 16> = SharedArray::UNINIT;
        static mut SM_VAL: SharedArray<f32, { LM_RPB }> = SharedArray::UNINIT;
        static mut SM_IDX: SharedArray<i32, { LM_RPB }> = SharedArray::UNINIT;
        unsafe {
            lm_head_stage1_body::<1>(
                SharedArray::as_raw_mut_ptr(&raw mut S_X16),
                SharedArray::as_raw_mut_ptr(&raw mut SM_VAL),
                SharedArray::as_raw_mut_ptr(&raw mut SM_IDX),
                w_data, w_scales, global_scale, x, out_logits, blk_max_val, blk_max_idx,
                sample_cfg, v,
            )
        }
    }

    /// `lm_head_stage1_kernel<2, 2>`.
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn lm_head_stage1_nvfp4_b2(
        w_data: *const u8,
        w_scales: *const u8,
        _row_scales: *const f32,
        global_scale: f32,
        x: *const f32,
        out_logits: *mut f32,
        blk_max_val: *mut f32,
        blk_max_idx: *mut i32,
        sample_cfg: *const f32,
        v: i32,
    ) {
        static mut S_X16: SharedArray<u16, { 2 * HIDDEN }, 16> = SharedArray::UNINIT;
        static mut SM_VAL: SharedArray<f32, { 2 * LM_RPB }> = SharedArray::UNINIT;
        static mut SM_IDX: SharedArray<i32, { 2 * LM_RPB }> = SharedArray::UNINIT;
        unsafe {
            lm_head_stage1_body::<2>(
                SharedArray::as_raw_mut_ptr(&raw mut S_X16),
                SharedArray::as_raw_mut_ptr(&raw mut SM_VAL),
                SharedArray::as_raw_mut_ptr(&raw mut SM_IDX),
                w_data, w_scales, global_scale, x, out_logits, blk_max_val, blk_max_idx,
                sample_cfg, v,
            )
        }
    }

    /// `lm_head_stage1_kernel<4, 2>`.
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn lm_head_stage1_nvfp4_b4(
        w_data: *const u8,
        w_scales: *const u8,
        _row_scales: *const f32,
        global_scale: f32,
        x: *const f32,
        out_logits: *mut f32,
        blk_max_val: *mut f32,
        blk_max_idx: *mut i32,
        sample_cfg: *const f32,
        v: i32,
    ) {
        static mut S_X16: SharedArray<u16, { 4 * HIDDEN }, 16> = SharedArray::UNINIT;
        static mut SM_VAL: SharedArray<f32, { 4 * LM_RPB }> = SharedArray::UNINIT;
        static mut SM_IDX: SharedArray<i32, { 4 * LM_RPB }> = SharedArray::UNINIT;
        unsafe {
            lm_head_stage1_body::<4>(
                SharedArray::as_raw_mut_ptr(&raw mut S_X16),
                SharedArray::as_raw_mut_ptr(&raw mut SM_VAL),
                SharedArray::as_raw_mut_ptr(&raw mut SM_IDX),
                w_data, w_scales, global_scale, x, out_logits, blk_max_val, blk_max_idx,
                sample_cfg, v,
            )
        }
    }

    /// `lm_head_stage1_kernel<8, 2>`.
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn lm_head_stage1_nvfp4_b8(
        w_data: *const u8,
        w_scales: *const u8,
        _row_scales: *const f32,
        global_scale: f32,
        x: *const f32,
        out_logits: *mut f32,
        blk_max_val: *mut f32,
        blk_max_idx: *mut i32,
        sample_cfg: *const f32,
        v: i32,
    ) {
        static mut S_X16: SharedArray<u16, { LM_MAX_B * HIDDEN }, 16> = SharedArray::UNINIT;
        static mut SM_VAL: SharedArray<f32, { LM_MAX_B * LM_RPB }> = SharedArray::UNINIT;
        static mut SM_IDX: SharedArray<i32, { LM_MAX_B * LM_RPB }> = SharedArray::UNINIT;
        unsafe {
            lm_head_stage1_body::<8>(
                SharedArray::as_raw_mut_ptr(&raw mut S_X16),
                SharedArray::as_raw_mut_ptr(&raw mut SM_VAL),
                SharedArray::as_raw_mut_ptr(&raw mut SM_IDX),
                w_data, w_scales, global_scale, x, out_logits, blk_max_val, blk_max_idx,
                sample_cfg, v,
            )
        }
    }

    /// `moe_up_tc_kernel<0>`: tensor-core NVFP4 expert gate/up projection with
    /// in-kernel expert dedup and softcapped SwiGLU. Grid (144, M * TOP_K),
    /// 576 threads.
    #[kernel]
    #[launch_bounds(576)]
    pub unsafe fn moe_up_tc_nvfp4(
        x_h: *const u16,
        idx: *const i32,
        w: *const u8,
        ws: *const u8,
        wfactor: *const f32,
        m: i32,
        out_act_h: *mut u16,
    ) {
        static mut RED: SharedArray<f32, { MOE_UP_WARPS * 32 * 4 }, 16> = SharedArray::UNINIT;
        static mut TILE_OUT: SharedArray<f32, { MOE_TILE * MOE_TILE_OUT_STRIDE }> =
            SharedArray::UNINIT;
        unsafe {
            let red = SharedArray::as_raw_mut_ptr(&raw mut RED);
            let tile_out = SharedArray::as_raw_mut_ptr(&raw mut TILE_OUT);

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
            // moe_slot_mask
            let m0 = warp::ballot_sync(0xffff_ffff, lo == e);
            let m1 = warp::ballot_sync(0xffff_ffff, hi == e);
            let mask = (m0 as u64) | ((m1 as u64) << 32);
            // __ffsll(mask) - 1 != slot; mask always holds this slot's bit.
            if mask.trailing_zeros() as usize != slot {
                return;
            }

            let row = e as usize * MOE_R + tile * MOE_TILE + g;
            let on = (g < m as usize) && ((mask >> (8 * g)) & 0xff) != 0;
            let c = 4 * warp_i + t;
            // moe_load_w<0, K>
            let codes = w.add(row * MOE_CB);
            let scales = ws.add(row * MOE_SB);
            let mut f = MoeFrag {
                r0: load_nvfp4_chunk32(codes, scales, c),
                r1: load_nvfp4_chunk32(codes.add(8 * MOE_CB), scales.add(8 * MOE_SB), c),
                x: [0u32; 16],
            };
            // moe_load_x<0>
            if on {
                let xp = x_h.add(g * MOE_K + c * 32);
                let mut vv = 0;
                #[unroll]
                while vv < 4 {
                    let q = ld_dep_u4(xp.add(vv * 8));
                    f.x[4 * vv] = q[0];
                    f.x[4 * vv + 1] = q[1];
                    f.x[4 * vv + 2] = q[2];
                    f.x[4 * vv + 3] = q[3];
                    vv += 1;
                }
            }
            // moe_mma_frag<0>
            let mut d = [0.0f32; 4];
            let mut a = [0u32; 16];
            let mut b = [0u32; 16];
            decode_nvfp4_chunk32_f16(&f.r0, &mut a);
            decode_nvfp4_chunk32_f16(&f.r1, &mut b);
            let mut i = 0;
            #[unroll]
            while i < 8 {
                d = mma_m16n8k16_f32_f16(
                    d,
                    [a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1]],
                    [f.x[2 * i], f.x[2 * i + 1]],
                );
                i += 1;
            }

            let mut vv = 0;
            #[unroll]
            while vv < 4 {
                *red.add((warp_i * 32 + lane) * 4 + vv) = d[vv];
                vv += 1;
            }
            sync_threads();

            if tid < 32 {
                let mut s = [0.0f32; 4];
                let mut ww = 0;
                #[unroll]
                while ww < MOE_UP_WARPS {
                    let mut vv = 0;
                    #[unroll]
                    while vv < 4 {
                        s[vv] += *red.add((ww * 32 + lane) * 4 + vv);
                        vv += 1;
                    }
                    ww += 1;
                }
                let es = *wfactor.add(e as usize);
                *tile_out.add(g * MOE_TILE_OUT_STRIDE + 2 * t) = s[0] * es;
                *tile_out.add(g * MOE_TILE_OUT_STRIDE + 2 * t + 1) = s[1] * es;
                *tile_out.add((g + 8) * MOE_TILE_OUT_STRIDE + 2 * t) = s[2] * es;
                *tile_out.add((g + 8) * MOE_TILE_OUT_STRIDE + 2 * t + 1) = s[3] * es;
            }
            sync_threads();

            if tid < (MOE_TILE / 2) * MOE_MAXM {
                let p = tid & 7;
                let n = tid >> 3;
                let mut kmask: u32 = if n < m as usize { ((mask >> (8 * n)) as u32) & 0xff } else { 0 };
                if kmask != 0 {
                    let gate = *tile_out.add(2 * p * MOE_TILE_OUT_STRIDE + n);
                    let up = *tile_out.add((2 * p + 1) * MOE_TILE_OUT_STRIDE + n);
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
}

mod attn;
mod common;
mod head;
mod depth;
mod fp8;
mod prefill_dense;
mod dspark;
mod ctrl;
mod ctrl2;
mod linear;
mod mega;
mod mma;
mod moe;
mod nvfp4;
mod router;
mod prefill_attn;
mod prefill_flash;
mod prefill_flash2;
mod prefill_flash3;
mod prefill_gemm;
mod prefill_flash4;
mod prefill_flash5;
mod prefill_mega;
mod prefill_moe;
mod prefill_moe_mma;
mod prefill_moe_bs;
mod prefill_moe_wg;

fn main() {}
