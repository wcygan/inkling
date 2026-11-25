//! Wide prefill MoE, tiled: grouped NVFP4 x f16 GEMMs over 128 weight rows x 64 pairs per CTA, so each
//! weight chunk is decoded once per 64 pairs and each activation chunk is reused by 128 rows (the
//! `prefill_moe.rs` kernels reuse activations across only 16 rows, which costs L2 bandwidth at large T).
//!
//! `prefill_moe_tiles` lists the (expert, first pair) tiles of 64 sorted pairs after `prefill_moe_group`.
//! Grid (rows / 128, max tiles), 256 threads (8 warps = 4 along M x 2 along N, 32 x 32 per warp). Per
//! 64-wide K chunk the weights are decoded to f16 in shared memory, the pairs' activation rows gathered,
//! then 4 k16 steps of m16n8k16 MMAs (f32 accumulation). The epilogue stages the 128 x 64 tile in shared
//! memory: gate/up rows are interleaved (row 2p gate, 2p + 1 up), so the SwiGLU needs both rows.

use crate::common::*;
use crate::{
    GATE_SOFTCAP, HIDDEN, INTERMEDIATE, TOP_K, UP_SOFTCAP, cvt_e2m1x8_to_half2x4, cvt_e4m3x2_to_half2, fast_sigmoid,
    fast_tanh, float2half, high2half2, ld_scale_u16, ld_stream_u4, low2half2,
};
use cuda_device::f16x2::mul_f16x2;
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{DynamicSharedArray, device, kernel, launch_bounds, ptx_asm};

const TM: usize = 128;
const TN: usize = 64;
const KC: usize = 128;
const AST: usize = KC; // f16 row of the A / B tiles (16-byte chunks XOR-swizzled by row, see `sw`)
const CST: usize = TN + 1; // f32 row of the staged C tile
const STAGE: usize = (TM + TN) * AST * 2; // 49,152 B: one A + B stage (two fit the 99 KB per-block limit)
/// Dynamic shared memory per CTA: two stages (the C tile, 33,280 B, reuses them).
pub const MOE_MMA_SMEM: usize = 2 * STAGE;
const THREADS: usize = 256;

#[inline(always)]
unsafe fn ldsm_x4(p: *const u16) -> [u32; 4] {
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
unsafe fn ldsm_x2(p: *const u16) -> [u32; 2] {
    let addr = unsafe { cvta_generic_to_shared_u32(p as *const u8) };
    let (r0, r1): (u32, u32);
    unsafe {
        ptx_asm!("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];", out("=r") r0, out("=r") r1, in("r") addr);
    }
    [r0, r1]
}

/// Offset (in f16) of 16-byte chunk `chunk` of tile row `row`: chunks are XOR-swizzled by `row & 7`, so the
/// eight rows an `ldmatrix` phase reads hit eight different bank groups.
#[inline(always)]
fn sw(row: usize, chunk: usize) -> usize {
    row * AST + ((chunk ^ (row & 7)) << 3)
}

#[inline(always)]
unsafe fn st_u4(p: *mut u16, v: [u32; 4]) {
    unsafe { *(p as *mut [u32; 4]) = v }
}

/// Decodes 32 NVFP4 weights (16 code bytes `q`, two E4M3 scales `sc`) into `dst` as f16.
#[device]
unsafe fn decode32(q: [u32; 4], scales: u16, tile: *mut u16, row: usize, chunk0: usize) {
    unsafe {
        let sc = cvt_e4m3x2_to_half2(scales);
        let slo = low2half2(sc);
        let shi = high2half2(sc);
        let mut w = 0;
        #[unroll]
        while w < 4 {
            let h = cvt_e2m1x8_to_half2x4(q[w]);
            let sm = if w < 2 { slo } else { shi };
            st_u4(tile.add(sw(row, chunk0 + w)), [mul_f16x2(h[0], sm), mul_f16x2(h[1], sm), mul_f16x2(h[2], sm), mul_f16x2(h[3], sm)]);
            w += 1;
        }
    }
}

/// Builds the tile list: tiles[2 i] = expert, tiles[2 i + 1] = first sorted pair, for every 64 pairs of
/// every expert; tiles_n[0] = count. Grid 1, 32 threads.
#[kernel]
pub unsafe fn prefill_moe_tiles(offs: *const i32, tiles: *mut i32, tiles_n: *mut i32) {
    unsafe {
        if threadIdx_x() == 0 {
            let mut n = 0usize;
            let mut e = 0;
            while e < NUM_EXPERTS {
                let mut p = *offs.add(e);
                let end = *offs.add(e + 1);
                while p < end {
                    *tiles.add(2 * n) = e as i32;
                    *tiles.add(2 * n + 1) = p;
                    n += 1;
                    p += TN as i32;
                }
                e += 1;
            }
            *tiles_n = n as i32;
        }
    }
}

/// The tiled GEMM. UP: x = mlp_normed_h rows of the pairs' tokens, K = HIDDEN, w = w13 (2 I rows per
/// expert), out = act (sorted pairs). Down: x = act rows, K = INTERMEDIATE, w = w2 (HIDDEN rows), out =
/// slots[pair id] (times the route weight).
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn moe_mma_body<const UP: bool>(
    smem: *mut u8,
    x: *const u16,
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
        let n_valid = core::cmp::min(TN, *offs.add(e + 1) as usize - p0);
        let m0 = blockIdx_x() as usize * TM;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;
        let wm = warp_i & 3;
        let wn = warp_i >> 2;
        let s_a = |i: usize| smem.add(i * STAGE) as *mut u16;
        let s_b = |i: usize| smem.add(i * STAGE + TM * AST * 2) as *mut u16;

        let cb = k_dim >> 1;
        let sb = k_dim >> 4;
        // This thread's activation row (one of the 64) and its four 16-byte pieces of each K chunk.
        let b_row = tid >> 2;
        let b_sub = (tid & 3) * 4;
        let xr: *const u16 = if b_row < n_valid {
            if UP {
                x.add((*pairs.add(p0 + b_row) as usize / TOP_K) * HIDDEN)
            } else {
                x.add((p0 + b_row) * INTERMEDIATE)
            }
        } else {
            core::ptr::null()
        };
        // This thread's weight half-row (row tid / 2, 64 of the 128 K values = two 32-value chunks).
        let a_row = tid >> 1;
        let a_half = tid & 1;
        let wrow = e * rows_per_e + m0 + a_row;
        let wcodes = w.add(wrow * cb + a_half * 32);
        let wscales = ws.add(wrow * sb + a_half * 4);

        // Two smem stages; chunk k + 2's global loads are in flight in registers while chunk k is
        // multiplied and chunk k + 1 is decoded into the other stage. One barrier per chunk.
        let mut wq0 = [0u32; 4];
        let mut wq1 = [0u32; 4];
        let mut ws0 = 0u16;
        let mut ws1 = 0u16;
        let mut xv = [[0u32; 4]; 4];
        let n_it = k_dim / KC;
        // ld(k0)
        wq0 = ld_stream_u4(wcodes);
        wq1 = ld_stream_u4(wcodes.add(16));
        ws0 = ld_scale_u16(wscales);
        ws1 = ld_scale_u16(wscales.add(2));
        if !xr.is_null() {
            let mut j = 0;
            #[unroll]
            while j < 4 {
                xv[j] = ldg_u4(xr.add((b_sub + j) * 8) as *const u8);
                j += 1;
            }
        }
        let mut acc = [[[0.0f32; 4]; 4]; 2];
        let mut it = 0usize;
        #[nounroll]
        while it <= n_it {
            // Store chunk `it` (in registers) into stage it & 1, then load chunk it + 1.
            if it < n_it {
                let st = it & 1;
                decode32(wq0, ws0, s_a(st), a_row, a_half * 8);
                decode32(wq1, ws1, s_a(st), a_row, a_half * 8 + 4);
                let mut j = 0;
                #[unroll]
                while j < 4 {
                    st_u4(s_b(st).add(sw(b_row, b_sub + j)), xv[j]);
                    j += 1;
                }
                if it + 1 < n_it {
                    let k1 = (it + 1) * KC;
                    wq0 = ld_stream_u4(wcodes.add(k1 >> 1));
                    wq1 = ld_stream_u4(wcodes.add((k1 >> 1) + 16));
                    ws0 = ld_scale_u16(wscales.add(k1 >> 4));
                    ws1 = ld_scale_u16(wscales.add((k1 >> 4) + 2));
                    if !xr.is_null() {
                        let mut j = 0;
                        #[unroll]
                        while j < 4 {
                            xv[j] = ldg_u4(xr.add(k1 + (b_sub + j) * 8) as *const u8);
                            j += 1;
                        }
                    }
                }
            }
            // Multiply chunk it - 1 (stored in the previous iteration, before the last barrier).
            if it > 0 {
                let cs = (it - 1) & 1;
                let sa = s_a(cs);
                let sbb = s_b(cs);
                let mut ks = 0;
                #[unroll]
                while ks < KC / 16 {
                    let a0 = ldsm_x4(sa.add(sw(wm * 32 + (lane & 15), ks * 2 + (lane >> 4))));
                    let a1 = ldsm_x4(sa.add(sw(wm * 32 + 16 + (lane & 15), ks * 2 + (lane >> 4))));
                    let mut ni = 0;
                    #[unroll]
                    while ni < 4 {
                        let bf = ldsm_x2(sbb.add(sw(wn * 32 + ni * 8 + (lane & 7), ks * 2 + ((lane >> 3) & 1))));
                        acc[0][ni] = mma_m16n8k16_f32_f16(acc[0][ni], a0, bf);
                        acc[1][ni] = mma_m16n8k16_f32_f16(acc[1][ni], a1, bf);
                        ni += 1;
                    }
                    ks += 1;
                }
            }
            sync_threads();
            it += 1;
        }

        // Stage C (rows x pairs) in shared memory, scaled by the expert's global factor.
        let es = *wfactor.add(e);
        let cs = smem as *mut f32;
        let g = lane >> 2;
        let t = lane & 3;
        let mut mi = 0;
        #[unroll]
        while mi < 2 {
            let mut ni = 0;
            #[unroll]
            while ni < 4 {
                let r = wm * 32 + mi * 16 + g;
                let c = wn * 32 + ni * 8 + 2 * t;
                *cs.add(r * CST + c) = acc[mi][ni][0] * es;
                *cs.add(r * CST + c + 1) = acc[mi][ni][1] * es;
                *cs.add((r + 8) * CST + c) = acc[mi][ni][2] * es;
                *cs.add((r + 8) * CST + c + 1) = acc[mi][ni][3] * es;
                ni += 1;
            }
            mi += 1;
        }
        sync_threads();

        if UP {
            // 64 intermediate columns x 64 pairs.
            let mut i = tid;
            while i < (TM / 2) * TN {
                let p = i & 63;
                let j = i >> 6;
                if j < n_valid {
                    let gate = *cs.add(2 * p * CST + j);
                    let up = *cs.add((2 * p + 1) * CST + j);
                    let g_cap = fast_tanh(gate * (1.0 / GATE_SOFTCAP)) * GATE_SOFTCAP;
                    let u_cap = fast_tanh(up * (1.0 / UP_SOFTCAP)) * UP_SOFTCAP;
                    *out_act.add((p0 + j) * INTERMEDIATE + m0 / 2 + p) = float2half((g_cap * fast_sigmoid(gate)) * u_cap);
                }
                i += THREADS;
            }
        } else {
            let mut i = tid;
            while i < TM * TN {
                let r = i & 127;
                let j = i >> 7;
                if j < n_valid {
                    let pid = *pairs.add(p0 + j) as usize;
                    *out_slots.add(pid * HIDDEN + m0 + r) = *cs.add(r * CST + j) * *route_w.add(pid);
                }
                i += THREADS;
            }
        }
    }
}

/// Gate/up: grid (2 I / 128 = 18, max tiles), 256 threads, MOE_MMA_SMEM bytes of dynamic shared memory.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_moe_up_mma(
    x_h: *const u16,
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
        moe_mma_body::<true>(
            DynamicSharedArray::<u8, 16>::get(),
            x_h,
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

/// Down: grid (HIDDEN / 128 = 18, max tiles), 256 threads, MOE_MMA_SMEM bytes of dynamic shared memory.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_moe_down_mma(
    act: *const u16,
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
        moe_mma_body::<false>(
            DynamicSharedArray::<u8, 16>::get(),
            act,
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
