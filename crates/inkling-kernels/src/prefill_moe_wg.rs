//! Wide prefill MoE, a port of ingine's `wgemm_kernel` (native/src/kernels/prefill/wgemm.cu, Apache-2.0) in its
//! NVFP4, 128-row-tile configuration (`INKLING_WIDE_MOE=wg`, chunks above 512 tokens).
//!
//! A CTA computes up to 128 activation rows (sorted pairs of one expert) x 128 weight rows, K in steps of 64,
//! m16n8k16 MMAs with f32 accumulation. The 8 warps split the weight rows (16 each) and every warp covers all of
//! the tile's activation rows, so each weight is decoded once per CTA, straight from the staged checkpoint codes
//! into the MMA B fragments in registers (no decoded weight tile in shared memory). Activation rows and weight
//! codes stream in through 16-byte `cp.async`: activations in 3 stages of 64 values, codes in 3 slots of 64-byte
//! row pieces (128 values), the tile's block scales once. One barrier per step; step ks + 1's B fragments are
//! decoded under step ks's MMAs. 16-row blocks past the tile's pair count skip their MMAs.
//!
//! Where this differs from ingine, to fit this engine's buffers:
//! - activations are f16 here (ingine: BF16), so the MMA is the f16 one and weights are decoded to f16:
//!   E2M1 x E4M3 is exact in f16, and the expert's factor multiplies the f32 sums instead of each weight;
//! - the epilogues are this engine's: f16 `act` through `fast_tanh` / `fast_sigmoid`, and f32 slots already
//!   multiplied by the route weight (ingine: BF16 rows and a separate combine);
//! - tiles come from `prefill_moe_tiles_n` over this engine's `offs` / `pairs` lists.
//! Activations and weights keep their full precision, as in the default kernels; only the order of the f32 sums
//! differs from them.

use crate::common::*;
use crate::{GATE_SOFTCAP, HIDDEN, INTERMEDIATE, TOP_K, UP_SOFTCAP, cvt_e4m3x2_to_half2, fast_sigmoid, fast_tanh, float2half, h2f_hi, h2f_lo};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::thread::{blockIdx_x, blockIdx_y, sync_threads, threadIdx_x};
use cuda_device::wmma::mma_m16n8k16_f32_f16;
use cuda_device::{DynamicSharedArray, device, kernel, launch_bounds, ptx_asm};

const TN: usize = 128; // weight rows per CTA
const TM: usize = 128; // activation rows (pairs) per tile
const MT_MAX: usize = TM / 16;
const BK: usize = 64; // K values per step
const ASTAGES: usize = 3; // activation stages
const WSLOTS: usize = 3; // weight-code slots (chunks of 128 values = 2 steps)
const NT: usize = 256;
const KCH: usize = BK / 8; // 16-byte chunks per activation tile row
const WCODE: usize = 64; // code bytes per row per chunk
const WSTRIDE: usize = WCODE + 16; // padded: fragment loads on distinct banks
const W_BYTES: usize = TN * WSTRIDE;
const A_BYTES: usize = TM * BK * 2;
const OFF_W: usize = ASTAGES * A_BYTES;
const OFF_SC: usize = OFF_W + WSLOTS * W_BYTES;
/// Dynamic shared memory per CTA (gate/up, the larger): activation stages, code slots, block scales, the E2M1 table.
pub const MOE_WG_SMEM: usize = OFF_SC + TN * (HIDDEN / 16) + 64;

#[inline(always)]
unsafe fn cp_async16_n(dst: *mut u8, src: *const u8, src_bytes: u32) {
    let d = unsafe { cvta_generic_to_shared_u32(dst as *const u8) };
    unsafe {
        ptx_asm!("cp.async.cg.shared.global [%0], [%1], 16, %2;", in("r") d, in("l") src, in("r") src_bytes, clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_async4(dst: *mut u8, src: *const u8) {
    let d = unsafe { cvta_generic_to_shared_u32(dst as *const u8) };
    unsafe {
        ptx_asm!("cp.async.ca.shared.global [%0], [%1], 4;", in("r") d, in("l") src, clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_commit() {
    unsafe {
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_wait1() {
    unsafe {
        ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
    }
}

#[inline(always)]
unsafe fn cp_wait0() {
    unsafe {
        ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
    }
}

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

/// Offset (in f16) of 16-byte chunk `chunk` of activation tile row `row` (XOR-swizzled by `row & 7`).
#[inline(always)]
fn swz(row: usize, chunk: usize) -> usize {
    row * (KCH * 8) + ((chunk ^ (row & 7)) << 3)
}

/// Tile list for tiles of `tr` sorted pairs: tiles[2 i] = expert, tiles[2 i + 1] = first sorted pair;
/// tiles_n[0] = count. Grid 1, 32 threads.
#[kernel]
pub unsafe fn prefill_moe_tiles_n(offs: *const i32, tiles: *mut i32, tiles_n: *mut i32, tr: i32) {
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
                    p += tr;
                }
                e += 1;
            }
            *tiles_n = n as i32;
        }
    }
}

/// The four block scales of step `ks` (64 values) for this thread's two weight rows.
#[device]
unsafe fn scales_of(sc_smem: *const u8, scale_row: usize, ks: usize, warp_i: usize, lane: usize) -> [[f32; 4]; 2] {
    unsafe {
        let mut sc = [[0.0f32; 4]; 2];
        let mut nt = 0;
        #[unroll]
        while nt < 2 {
            let row = warp_i * 16 + nt * 8 + (lane >> 2);
            let s4 = *(sc_smem.add(row * scale_row + ks * 4) as *const u32);
            let lo = cvt_e4m3x2_to_half2(s4 as u16);
            let hi = cvt_e4m3x2_to_half2((s4 >> 16) as u16);
            sc[nt] = [h2f_lo(lo), h2f_hi(lo), h2f_lo(hi), h2f_hi(hi)];
            nt += 1;
        }
        sc
    }
}

/// B fragments of step `ks`, k16 group `kk`, decoded from the staged codes of its weight chunk: values 2 c,
/// 2 c + 1 (byte c) and 8 + 2 c, 8 + 2 c + 1 (byte 4 + c) of the group, for this thread's two weight rows.
#[device]
unsafe fn decode(w_smem: *const u8, lut: *const f32, ks: usize, kk: usize, sc: &[[f32; 4]; 2], warp_i: usize, lane: usize) -> [[u32; 2]; 2] {
    unsafe {
        let ws = w_smem.add(((ks / 2) % WSLOTS) * W_BYTES);
        let kc = (ks % 2) * BK + kk * 16;
        let c = lane & 3;
        let mut b = [[0u32; 2]; 2];
        let mut nt = 0;
        #[unroll]
        while nt < 2 {
            let row = warp_i * 16 + nt * 8 + (lane >> 2);
            let s = sc[nt][kk];
            let wd = *(ws.add(row * WSTRIDE + kc / 2) as *const [u32; 2]);
            let b0 = ((wd[0] >> (8 * c)) & 0xff) as usize;
            let b1 = ((wd[1] >> (8 * c)) & 0xff) as usize;
            b[nt][0] = pack_f16x2(*lut.add(b0 & 0xf) * s, *lut.add(b0 >> 4) * s);
            b[nt][1] = pack_f16x2(*lut.add(b1 & 0xf) * s, *lut.add(b1 >> 4) * s);
            nt += 1;
        }
        b
    }
}

/// One activation step of MMAs for the first MT 16-row blocks: per k16 group the A fragments by ldmatrix, the
/// MMAs, then the next step's B decoding, which runs under the issued MMAs. MT is a constant so that inactive
/// row blocks issue no MMAs at all.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn mma_step<const MT: usize>(
    acc: &mut [[[f32; 4]; 2]; MT_MAX],
    a_stage: *const u16,
    bcur: &[[[u32; 2]; 2]; 4],
    bnext: &mut [[[u32; 2]; 2]; 4],
    more: bool,
    w_smem: *const u8,
    lut: *const f32,
    ks: usize,
    scn: &[[f32; 4]; 2],
    warp_i: usize,
    lane: usize,
) {
    unsafe {
        let mut kk = 0;
        #[unroll]
        while kk < BK / 16 {
            let mut a = [[0u32; 4]; MT];
            let mut mt = 0;
            #[unroll]
            while mt < MT {
                a[mt] = ldsm_x4(a_stage.add(swz(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
                mt += 1;
            }
            let mut mt = 0;
            #[unroll]
            while mt < MT {
                acc[mt][0] = mma_m16n8k16_f32_f16(acc[mt][0], a[mt], bcur[kk][0]);
                acc[mt][1] = mma_m16n8k16_f32_f16(acc[mt][1], a[mt], bcur[kk][1]);
                mt += 1;
            }
            if more {
                bnext[kk] = decode(w_smem, lut, ks + 1, kk, scn, warp_i, lane);
            }
            kk += 1;
        }
    }
}

/// UP: A = `x` rows of the pairs' tokens (f16 [T][HIDDEN]), W = w13 (2 I rows per expert), out = act (sorted
/// pairs). Down: A = act rows (K = INTERMEDIATE), W = w2 (HIDDEN rows), out = slots[pair id] times the route weight.
#[device]
#[allow(clippy::too_many_arguments)]
unsafe fn moe_wg_body<const UP: bool>(
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
        let n_steps = k_dim / BK;
        let n_chunks = k_dim / 128;
        let code_row = k_dim >> 1;
        let scale_row = k_dim >> 4;
        let a_smem = smem;
        let w_smem = smem.add(OFF_W);
        let sc_smem = smem.add(OFF_SC);
        let lut = smem.add(OFF_SC + TN * scale_row) as *mut f32;
        let tid = threadIdx_x() as usize;
        let lane = tid & 31;
        let warp_i = tid >> 5;

        let tile = blockIdx_y() as usize;
        if tile as i32 >= *tiles_n {
            return;
        }
        let e = *tiles.add(2 * tile) as usize;
        let pb = *tiles.add(2 * tile + 1) as usize;
        let n = core::cmp::min(TM, *offs.add(e + 1) as usize - pb);
        let n0 = blockIdx_x() as usize * TN;
        let code_tile = w.add((e * rows_per_e + n0) * code_row);
        let scale_tile = ws.add((e * rows_per_e + n0) * scale_row);
        if tid < 16 {
            let m = tid & 7;
            let mag = if m == 0 {
                0.0f32
            } else if m == 1 {
                0.5
            } else if m == 2 {
                1.0
            } else if m == 3 {
                1.5
            } else if m == 4 {
                2.0
            } else if m == 5 {
                3.0
            } else if m == 6 {
                4.0
            } else {
                6.0
            };
            *lut.add(tid) = if tid < 8 { mag } else { -mag };
        }

        // Activation rows of this thread's cp.async chunks: rows tid / 8 + 32 i, chunk tid % 8.
        let ach = tid & 7;
        let mut arow = [core::ptr::null::<u16>(); TM / 32];
        let mut i = 0;
        #[unroll]
        while i < TM / 32 {
            let r = (tid >> 3) + 32 * i;
            let rr = if r < n { r } else { 0 };
            arow[i] = if UP { x.add((*pairs.add(pb + rr) as usize / TOP_K) * k_dim) } else { x.add((pb + rr) * k_dim) };
            i += 1;
        }

        macro_rules! load_a {
            ($ks:expr) => {{
                let ks: usize = $ks;
                let a_st = a_smem.add((ks % ASTAGES) * A_BYTES) as *mut u16;
                let mut i = 0;
                while i < TM / 32 {
                    let r = (tid >> 3) + 32 * i;
                    cp_async16_n(a_st.add(swz(r, ach)) as *mut u8, arow[i].add(ks * BK + ach * 8) as *const u8, if r < n { 16 } else { 0 });
                    i += 1;
                }
            }};
        }
        macro_rules! load_w {
            ($chunk:expr) => {{
                let chunk: usize = $chunk;
                let slot = w_smem.add((chunk % WSLOTS) * W_BYTES);
                let mut i = 0;
                while i < 2 {
                    let id = tid + i * NT;
                    let (row, ch) = (id >> 2, id & 3);
                    cp_async16_n(slot.add(row * WSTRIDE + ch * 16), code_tile.add(row * code_row + chunk * WCODE + ch * 16), 16);
                    i += 1;
                }
            }};
        }
        // Group j: activation step j, plus whatever weight chunk was issued since the previous commit.
        macro_rules! issue {
            ($j:expr) => {{
                let j: usize = $j;
                if j < n_steps {
                    load_a!(j);
                }
                cp_commit();
            }};
        }

        // The tile's block scales, with the first activation group.
        let mut id = tid;
        while id < TN * scale_row / 4 {
            cp_async4(sc_smem.add(id * 4), scale_tile.add(id * 4));
            id += NT;
        }
        let mut j = 0;
        #[unroll]
        while j < WSLOTS - 1 {
            if j < n_chunks {
                load_w!(j);
            }
            j += 1;
        }
        let mut j = 0;
        #[unroll]
        while j < ASTAGES - 1 {
            issue!(j);
            j += 1;
        }

        let m_act = n.div_ceil(16);
        let mut acc = [[[0.0f32; 4]; 2]; MT_MAX];
        // Software pipeline: step ks's MMAs run on fragments decoded during step ks - 1.
        let mut bcur = [[[0u32; 2]; 2]; 4];
        let mut bnext = [[[0u32; 2]; 2]; 4];
        let mut ks = 0usize;
        #[nounroll]
        while ks < n_steps {
            cp_wait1();
            sync_threads();
            let chunk = ks / 2;
            // First step of a chunk: the chunk two ahead goes into the slot the previous chunk used.
            if ks % 2 == 0 && chunk + WSLOTS - 1 < n_chunks {
                load_w!(chunk + WSLOTS - 1);
            }
            issue!(ks + ASTAGES - 1);
            if ks == 0 {
                let sc0 = scales_of(sc_smem, scale_row, 0, warp_i, lane);
                let mut kk = 0;
                #[unroll]
                while kk < 4 {
                    bcur[kk] = decode(w_smem, lut, 0, kk, &sc0, warp_i, lane);
                    kk += 1;
                }
            }
            let a_st = a_smem.add((ks % ASTAGES) * A_BYTES) as *const u16;
            let more = ks + 1 < n_steps;
            let scn = if more { scales_of(sc_smem, scale_row, ks + 1, warp_i, lane) } else { [[0.0f32; 4]; 2] };
            if m_act == 1 {
                mma_step::<1>(&mut acc, a_st, &bcur, &mut bnext, more, w_smem, lut, ks, &scn, warp_i, lane);
            } else if m_act == 2 {
                mma_step::<2>(&mut acc, a_st, &bcur, &mut bnext, more, w_smem, lut, ks, &scn, warp_i, lane);
            } else if m_act == 3 {
                mma_step::<3>(&mut acc, a_st, &bcur, &mut bnext, more, w_smem, lut, ks, &scn, warp_i, lane);
            } else if m_act == 4 {
                mma_step::<4>(&mut acc, a_st, &bcur, &mut bnext, more, w_smem, lut, ks, &scn, warp_i, lane);
            } else if m_act == 5 {
                mma_step::<5>(&mut acc, a_st, &bcur, &mut bnext, more, w_smem, lut, ks, &scn, warp_i, lane);
            } else if m_act == 6 {
                mma_step::<6>(&mut acc, a_st, &bcur, &mut bnext, more, w_smem, lut, ks, &scn, warp_i, lane);
            } else if m_act == 7 {
                mma_step::<7>(&mut acc, a_st, &bcur, &mut bnext, more, w_smem, lut, ks, &scn, warp_i, lane);
            } else {
                mma_step::<8>(&mut acc, a_st, &bcur, &mut bnext, more, w_smem, lut, ks, &scn, warp_i, lane);
            }
            bcur = bnext;
            ks += 1;
        }
        cp_wait0();

        let es = *wfactor.add(e);
        let g = lane >> 2;
        let c = lane & 3;
        let mut mt = 0;
        #[unroll]
        while mt < MT_MAX {
            let mut half = 0;
            #[unroll]
            while half < 2 {
                let row = mt * 16 + g + half * 8;
                if mt < m_act && row < n {
                    let mut nt = 0;
                    #[unroll]
                    while nt < 2 {
                        let cl = n0 + warp_i * 16 + nt * 8 + c * 2;
                        let c0 = acc[mt][nt][2 * half] * es;
                        let c1 = acc[mt][nt][2 * half + 1] * es;
                        if UP {
                            // Rows 2 p (gate) and 2 p + 1 (up) of the expert give intermediate column p.
                            let g_cap = fast_tanh(c0 * (1.0 / GATE_SOFTCAP)) * GATE_SOFTCAP;
                            let u_cap = fast_tanh(c1 * (1.0 / UP_SOFTCAP)) * UP_SOFTCAP;
                            *out_act.add((pb + row) * INTERMEDIATE + cl / 2) = float2half((g_cap * fast_sigmoid(c0)) * u_cap);
                        } else {
                            let pid = *pairs.add(pb + row) as usize;
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

/// Gate/up: grid (2 I / 128 = 18, max tiles), 256 threads, MOE_WG_SMEM bytes of dynamic shared memory.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_moe_up_wg(
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
        moe_wg_body::<true>(
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

/// Down: grid (HIDDEN / 128 = 18, max tiles), 256 threads, MOE_WG_SMEM bytes of dynamic shared memory.
#[kernel]
#[launch_bounds(256)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_moe_down_wg(
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
        moe_wg_body::<false>(
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
