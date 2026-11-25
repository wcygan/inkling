//! Primitives shared by the exact-mode kernels: the rest of squid5090's
//! `kernels_common.cuh` and `config.hpp`, written so that each lowers to the
//! same PTX instruction nvcc emits for the CUDA C++ original (no fast-math).

#![allow(dead_code)]

use cuda_device::{device, ptx_asm};
use cuda_device::warp;

pub(crate) const NUM_HEADS: usize = 24;
pub(crate) const NUM_KV_HEADS: usize = 8;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const Q_DIM: usize = NUM_HEADS * HEAD_DIM; // 3072
pub(crate) const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM; // 1024
pub(crate) const GQA_RATIO: usize = NUM_HEADS / NUM_KV_HEADS; // 3
pub(crate) const D_REL: usize = 16;
pub(crate) const R_DIM: usize = NUM_HEADS * D_REL; // 384
pub(crate) const REL_EXTENT: usize = 1024;
pub(crate) const SWA_WINDOW: usize = 1024;
pub(crate) const NUM_EXPERTS: usize = 128;
pub(crate) const GATE_RANK: usize = 64;
pub(crate) const MAX_VERIFY_B: usize = 8;
pub(crate) const RMS_EPS: f32 = 1e-6;
pub(crate) const QK_CLIP: f32 = 22.627416997969522;
pub(crate) const REL_CLIP: f32 = 8.0;
pub(crate) const LOG_SCALE_FLOOR: i32 = 8192;
pub(crate) const LOG_SCALE_ALPHA: f32 = 0.1;
pub(crate) const ROUTE_SCALE: f32 = 8.0;
pub(crate) const INV_SQRT_HEAD_DIM: f32 = 0.08838834764831845;
pub(crate) const INV_SQRT_HIDDEN: f32 = 0.020833333333333332;
pub(crate) const INV_SQRT_5: f32 = 0.4472135954999579;
pub(crate) const INV_SQRT_9: f32 = 0.3333333333333333;
pub(crate) const KV_SPLIT_K: usize = 64;
pub(crate) const KV_LOCAL_SPLITS: usize = 17;
/// With FP8 global KV, the BF16 global cache is a ring of this many slots (the tail's 1,024 + chunk rows).
pub(crate) const GLOBAL_KV_RING: usize = 2304;
pub(crate) const KV_TOTAL_SPLITS: usize = KV_SPLIT_K + KV_LOCAL_SPLITS; // 81
/// QRGS row: Q (3072) | R (384) | gate G (3072) | sink S (24).
pub(crate) const QRGS_STRIDE: usize = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;

// Split-K attention partial buffer offsets (kernels_common.cuh).
pub(crate) const SPLIT_MD_ELEMS: usize = NUM_HEADS * 8 * KV_TOTAL_SPLITS;
pub(crate) const SPLIT_M_OFF: usize = 0;
pub(crate) const SPLIT_D_OFF: usize = SPLIT_MD_ELEMS;
pub(crate) const SPLIT_O_OFF: usize = 2 * SPLIT_MD_ELEMS;

/// `struct Ptr3 { const float* p[3]; }`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Ptr3 {
    pub p: [*const f32; 3],
}

/// `struct MutPtr3 { float* p[3]; }`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MutPtr3 {
    pub p: [*mut f32; 3],
}

/// `struct HalfPtr3 { __half* p[3]; }`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HalfPtr3 {
    pub p: [*mut u16; 3],
}

/// `struct BF16Ptr3 { const __nv_bfloat16* p[3]; }`
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Bf16Ptr3 {
    pub p: [*const u16; 3],
}

/// `struct CtrlPtr3 { down[3], forget[3], gain[3]; }` (BF16 weights).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CtrlPtr3 {
    pub down: [*const u16; 3],
    pub forget: [*const u16; 3],
    pub gain: [*const u16; 3],
}

/// Element `s` (0..3) of a by-value pointer array, chosen with branches so
/// the kernel-parameter array is read with constant offsets instead of being
/// copied to the stack for a dynamic index.
#[inline(always)]
pub(crate) fn pick3<T: Copy>(a: [T; 3], s: usize) -> T {
    if s == 0 {
        a[0]
    } else if s == 1 {
        a[1]
    } else {
        a[2]
    }
}

unsafe extern "C" {
    fn __nv_expf(x: f32) -> f32;
    fn __nv_logf(x: f32) -> f32;
}

/// `expf` without fast-math: libdevice `__nv_expf`, as nvcc inlines it.
#[inline(always)]
pub(crate) fn expf(x: f32) -> f32 {
    unsafe { __nv_expf(x) }
}

/// `logf` without fast-math: libdevice `__nv_logf`.
#[inline(always)]
pub(crate) fn logf(x: f32) -> f32 {
    unsafe { __nv_logf(x) }
}

/// `__expf`: `ex2.approx.f32(x * log2e)`.
#[inline(always)]
pub(crate) fn fast_expf(x: f32) -> f32 {
    let t = x * core::f32::consts::LOG2_E;
    let r: f32;
    unsafe {
        ptx_asm!("ex2.approx.f32 %0, %1;", out("=f") r, in("f") t, options(register_only));
    }
    r
}

/// `__logf`: `lg2.approx.f32(x) * ln2`.
#[inline(always)]
pub(crate) fn fast_logf(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("lg2.approx.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
    }
    r * core::f32::consts::LN_2
}

/// `rsqrtf`: `rsqrt.approx.f32`.
#[inline(always)]
pub(crate) fn rsqrtf(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("rsqrt.approx.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
    }
    r
}

/// `sqrtf` without fast-math: `sqrt.rn.f32`.
#[inline(always)]
pub(crate) fn sqrtf(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("sqrt.rn.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
    }
    r
}

/// IEEE division `a / b` (`div.rn.f32`).
#[inline(always)]
pub(crate) fn div_rn(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("div.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only));
    }
    r
}

/// `a * b` that is never contracted into a neighbouring add (`mul.rn.f32`).
#[inline(always)]
pub(crate) fn mul_rn(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("mul.rn.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only));
    }
    r
}

/// `1.0f / x` (nvcc emits `rcp.rn.f32`).
#[inline(always)]
pub(crate) fn rcp_rn(x: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("rcp.rn.f32 %0, %1;", out("=f") r, in("f") x, options(register_only));
    }
    r
}

/// `fmaxf` (`max.f32`).
#[inline(always)]
pub(crate) fn fmaxf(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("max.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only));
    }
    r
}

/// `fminf` (`min.f32`).
#[inline(always)]
pub(crate) fn fminf(a: f32, b: f32) -> f32 {
    let r: f32;
    unsafe {
        ptx_asm!("min.f32 %0, %1, %2;", out("=f") r, in("f") a, in("f") b, options(register_only));
    }
    r
}

/// `__bfloat162float` (exact).
#[inline(always)]
pub(crate) fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `__float2bfloat16`: `cvt.rn.bf16.f32`.
#[inline(always)]
pub(crate) fn f32_to_bf16(x: f32) -> u16 {
    let h: u16;
    unsafe {
        ptx_asm!("cvt.rn.bf16.f32 %0, %1;", out("=h") h, in("f") x, options(register_only));
    }
    h
}

/// `round_bf16`: round to BF16 and back.
#[inline(always)]
pub(crate) fn round_bf16(x: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(x))
}

/// `__half2float`: `cvt.f32.f16`.
#[inline(always)]
pub(crate) fn half_to_f32(h: u16) -> f32 {
    let f: f32;
    unsafe {
        ptx_asm!("cvt.f32.f16 %0, %1;", out("=f") f, in("h") h, options(register_only));
    }
    f
}

/// `pack_bf16x2` / `__floats2bfloat162_rn(lo, hi)`: one `cvt.rn.bf16x2.f32`
/// (the first source operand goes to the high half).
#[inline(always)]
pub(crate) fn pack_bf16x2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe {
        ptx_asm!("cvt.rn.bf16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo, options(register_only));
    }
    r
}

/// `__floats2half2_rn(lo, hi)`: one `cvt.rn.f16x2.f32`.
#[inline(always)]
pub(crate) fn pack_f16x2(lo: f32, hi: f32) -> u32 {
    let r: u32;
    unsafe {
        ptx_asm!("cvt.rn.f16x2.f32 %0, %1, %2;", out("=r") r, in("f") hi, in("f") lo, options(register_only));
    }
    r
}

/// `cvt_ue8m0_to_float`.
#[inline(always)]
pub(crate) fn cvt_ue8m0_to_float(e: u8) -> f32 {
    let bits: u32 = if e == 0 {
        0x0040_0000
    } else if e == 255 {
        0x7fc0_0000
    } else {
        (e as u32) << 23
    };
    f32::from_bits(bits)
}

/// `warp_sum`.
#[inline(always)]
pub(crate) fn warp_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 16);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 8);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 4);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 2);
    v += warp::shuffle_xor_f32_sync(0xffff_ffff, v, 1);
    v
}

/// `warp_max`.
#[inline(always)]
pub(crate) fn warp_max(mut v: f32) -> f32 {
    v = fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 16));
    v = fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 8));
    v = fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 4));
    v = fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 2));
    v = fmaxf(v, warp::shuffle_xor_f32_sync(0xffff_ffff, v, 1));
    v
}

/// `__ldg(const uint4*)`: `ld.global.nc.v4.u32`.
#[inline(always)]
pub(crate) unsafe fn ldg_u4(p: *const u8) -> [u32; 4] {
    let (x, y, z, w): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];",
            out("=r") x,
            out("=r") y,
            out("=r") z,
            out("=r") w,
            in("l") p,
            options(register_only),
        );
    }
    [x, y, z, w]
}

/// `__ldg(const float4*)`: `ld.global.nc.v4.f32`.
#[inline(always)]
pub(crate) unsafe fn ldg_f4(p: *const f32) -> [f32; 4] {
    let (x, y, z, w): (f32, f32, f32, f32);
    unsafe {
        ptx_asm!(
            "ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];",
            out("=f") x,
            out("=f") y,
            out("=f") z,
            out("=f") w,
            in("l") p,
            options(register_only),
        );
    }
    [x, y, z, w]
}

/// Two BF16 values of a packed word as floats (`__bfloat1622float2`).
#[inline(always)]
pub(crate) fn bf16x2_to_f32x2(w: u32) -> (f32, f32) {
    (f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000))
}

/// `pdl_wait_fenced(zero)`: a loop the compiler can't prove runs once, so no
/// dependent load moves above the PDL wait.
#[device]
pub(crate) fn pdl_wait_fenced(zero: u32) {
    let mut i = 0u32;
    #[nounroll]
    while i <= zero {
        crate::pdl_wait();
        i += 1;
    }
}
