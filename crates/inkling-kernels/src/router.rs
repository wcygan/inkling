//! Router top-8 select with `redux.sync` (small-ops lane). The original `route_select_and_group_b*`
//! in `moe.rs` stays as the parity oracle.
//!
//! - `route_select_fast_b{1,2,4,8}`: `route_select_and_group_kernel<B>` with each of the eight
//!   arg-max rounds' three 5-step shuffle reductions replaced by two `redux.sync` and one shuffle.
//!   Ids and weights are bit-identical (parity includes exact ties).

use crate::common::*;
use crate::{TOP_K, fast_sigmoid, ld_dep_f32, pdl_trigger};
use cuda_device::thread::threadIdx_x;
use cuda_device::{device, kernel, launch_bounds, ptx_asm, warp};

/// `redux.sync.max.u32` / `redux.sync.min.u32` over the full warp.
#[inline(always)]
fn redux_max_u32(v: u32) -> u32 {
    let r: u32;
    unsafe {
        ptx_asm!("redux.sync.max.u32 %0, %1, 0xffffffff;", out("=r") r, in("r") v);
    }
    r
}

#[inline(always)]
fn redux_min_u32(v: u32) -> u32 {
    let r: u32;
    unsafe {
        ptx_asm!("redux.sync.min.u32 %0, %1, 0xffffffff;", out("=r") r, in("r") v);
    }
    r
}

/// Order-preserving map of a (non-NaN) f32 onto u32, and back.
#[inline(always)]
fn f32_key(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

#[inline(always)]
fn key_f32(k: u32) -> f32 {
    f32::from_bits(if k & 0x8000_0000 != 0 { k & 0x7fff_ffff } else { !k })
}

/// The same top-8 select as `moe::route_select_body`, with each round's three 5-step shuffle
/// reductions replaced by two `redux.sync` and one shuffle:
/// - the arg-max value: max over order-preserving keys (the scores are finite, so this equals
///   `warp_max`'s fmaxf tree; ties between -0 and +0 still compare equal in `best == top`);
/// - the arg-max index: min over candidate indices (exact for integers);
/// - the chosen score: read from the owning lane. `warp_sum` of one non-zero score and 31 zeros
///   returns that score exactly, so the weights are bit-identical.
#[device]
unsafe fn route_select_warp_fast(logits: *const f32, rb: [f32; 4], out_ids: *mut i32, out_weights: *mut f32) {
    unsafe {
        let b = (threadIdx_x() >> 5) as usize;
        let lane = (threadIdx_x() & 31) as usize;
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
            let top = key_f32(redux_max_u32(f32_key(best)));
            let cand = redux_min_u32(if best == top { best_e as u32 } else { NUM_EXPERTS as u32 }) as i32;
            let slot = (cand >> 5) as usize;
            let mut mine = 0.0f32;
            let mut i = 0;
            #[unroll]
            while i < 4 {
                if i == slot {
                    mine = sc[i];
                }
                if (lane + 32 * i) as i32 == cand {
                    bi[i] = -1e30;
                }
                i += 1;
            }
            chosen[k] = warp::shuffle_f32_sync(0xffff_ffff, mine, (cand & 31) as u32);
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

macro_rules! route_select_fast_kernel {
    ($name:ident, $threads:expr) => {
        /// `route_select_and_group_b{B}` with the redux select: grid 1, 32 * B threads, PDL.
        /// Same parameters (`g` must be null) and bit-identical outputs.
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
            unsafe {
                let lane = (threadIdx_x() & 31) as usize;
                let mut rb = [0.0f32; 4];
                let mut i = 0;
                #[unroll]
                while i < 4 {
                    rb[i] = ld_dep_f32(router_bias.add(lane + 32 * i));
                    i += 1;
                }
                pdl_wait_fenced(zero);
                pdl_trigger();
                route_select_warp_fast(logits, rb, out_ids, out_weights);
            }
        }
    };
}

route_select_fast_kernel!(route_select_fast_b1, 32);
route_select_fast_kernel!(route_select_fast_b2, 64);
route_select_fast_kernel!(route_select_fast_b4, 128);
route_select_fast_kernel!(route_select_fast_b8, 256);
