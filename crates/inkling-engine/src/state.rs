//! Decode state in one device arena, laid out as squid5090's `engine.zig` `init` does for B ≤ 8.
//! Global KV caches hold `max_ctx + 8` positions so that a
//! width-8 verify near the context limit cannot wrap onto position 0 (squid5090 bug 2).

use inkling_model::{SCHEDULE, dims};
use inkling_runtime::{Arena, ArenaPlan, DevSlice, Result, Slot, Stream};

/// Largest batch (verify width) the buffers are sized for. The rings index `p & 15`, so B ≤ 13.
pub const B_MAX: usize = 8;
const _: () = assert!(B_MAX + 3 < 16, "ring slots p & 15 must not alias within one step");

pub const RING_SLOTS: usize = 16;
/// With FP8 global KV (INKLING_KV_FP8), each global visit's BF16 cache is a ring of this many rows: the
/// relative tail (1,024 + B) of decode, or the previous 1,023 rows plus a whole wide-prefill chunk (<= 1,024).
/// Must equal the kernels' GLOBAL_KV_RING.
pub const GLOBAL_KV_RING: usize = 2304;
/// The smallest local KV cache (a 256-token slice); `local_kv_cap()` is the size in use.
pub const LOCAL_KV_CAP: usize = dims::WINDOW + 256; // 1280

/// Tokens the wide prefill stores and attends per local-attention launch (INKLING_LOCAL_SLICE: 256, 512 or
/// 1024; default 512). The local KV caches hold WINDOW + this many positions, so a larger slice costs
/// 40 visits x 2 x 2 KiB per extra position, in the state and in every saved checkpoint: 512 adds 42 MB to
/// each over 256. With 512 the launch has 256 CTAs, which `prefill_attn_flash5` runs two per SM; measured
/// 3 to 4% faster prefill than 256 (128 one-per-SM CTAs on 170 SMs), and 1024 was no faster than 512.
pub fn local_slice() -> usize {
    match std::env::var("INKLING_LOCAL_SLICE").ok().and_then(|v| v.parse().ok()) {
        Some(256) => 256,
        Some(1024) => 1024,
        _ => 512,
    }
}

/// Positions a local visit's KV cache holds.
pub fn local_kv_cap() -> usize {
    dims::WINDOW + local_slice()
}
/// With FP8 global KV (INKLING_KV_FP8), each global visit's BF16 cache is a ring of this many rows: the
/// relative tail (1,024 + B) of decode, or the previous 1,023 rows plus a whole wide-prefill chunk (<= 1,024).
/// Must equal the kernels' GLOBAL_KV_RING. const GLOBAL_KV_RING: usize = 2304;
pub const SPLIT_SLOTS: usize = 81; // 64 bulk + 17 tail
pub const SPLIT_BUF_FLOATS: usize = 2_050_000;
pub const LM_HEAD_BLOCKS: usize = 680;
pub const SAMPLE_CONFIG_BYTES: usize = 24;
pub const BLOCK_SLOTS: usize = 4;

/// Bytes of one visit's FP8 shadow: K and V codes plus K and V f16 scales per token and KV head.
pub const fn kv_fp8_bytes(kv_cap: usize) -> usize {
    2 * kv_cap * dims::KV_DIM + 2 * kv_cap * dims::KV_HEADS * 2
}

pub struct VisitState {
    /// Positions the visit's KV store holds (global: max_ctx + B_MAX; local: `local_kv_cap()`).
    pub kv_cap: usize,
    /// Slots of the BF16 k_cache/v_cache: kv_cap, or GLOBAL_KV_RING for a global visit with FP8 KV.
    pub ring_cap: usize,
    pub k_cache: Slot,
    pub v_cache: Slot,
    pub k_ring: Slot,
    pub v_ring: Slot,
    pub attn_ring: Slot,
    pub mlp_ring: Slot,
    /// FP8 global KV store (INKLING_KV_FP8=1), all kv_cap positions: K codes, V codes (cap x KV_DIM bytes
    /// each), then K and V f16 scales (cap x 8 each). The bulk and the wide prefill read old rows from it.
    pub kv_fp8: Option<Slot>,
}

macro_rules! state_slots {
    ($($name:ident: $ty:ty = $count:expr),* $(,)?) => {
        pub struct Slots { $(pub $name: Slot,)* }
        fn plan_slots(p: &mut ArenaPlan) -> Slots {
            Slots { $($name: p.reserve_of::<$ty>(stringify!($name), $count),)* }
        }
    };
}

const H: usize = dims::HIDDEN;
state_slots! {
    hidden: f32 = B_MAX * H,
    prelude: f32 = B_MAX * H,
    partial: f32 = B_MAX * H,
    saved: f32 = B_MAX * H,
    prior_last: f32 = B_MAX * H,
    residual: f32 = B_MAX * H,
    o_proj: f32 = B_MAX * H,
    mlp_in: f32 = B_MAX * H,
    mlp_normed: f32 = B_MAX * H,
    mlp_normed_h: u16 = B_MAX * H,
    blocks: f32 = B_MAX * BLOCK_SLOTS * H,
    attn_in0: f32 = B_MAX * H,
    attn_in1: f32 = B_MAX * H,
    attn_in2: f32 = B_MAX * H,
    attn_in_h0: u16 = B_MAX * H,
    attn_in_h1: u16 = B_MAX * H,
    attn_in_h2: u16 = B_MAX * H,
    qrgs: f32 = B_MAX * dims::QRGS_ROWS,
    k_proj: f32 = B_MAX * dims::KV_DIM,
    v_proj: f32 = B_MAX * dims::KV_DIM,
    attn_gated: f32 = B_MAX * dims::Q_DIM,
    attn_gated_h: u16 = B_MAX * dims::Q_DIM,
    split_buf: f32 = SPLIT_BUF_FLOATS,
    ctrl_msg: f32 = 3 * B_MAX * H,
    ctrl_cat: f32 = 3 * B_MAX * 2 * H,
    ctrl_d: f32 = 3 * B_MAX * dims::CONTROLLER_RANK,
    router_logits: f32 = B_MAX * dims::EXPERTS,
    route_ids: i32 = B_MAX * dims::TOP_K,
    route_w: f32 = B_MAX * dims::TOP_K,
    moe_act_h: u16 = B_MAX * dims::TOP_K * dims::INTERMEDIATE,
    moe_down: f32 = B_MAX * H,
    lm_blk_val: f32 = B_MAX * 1024,
    lm_blk_idx: i32 = B_MAX * 1024,
    lm_logits: f32 = B_MAX * dims::VOCAB,
    sample_cfg: u8 = SAMPLE_CONFIG_BYTES,
    d_pos: i32 = 1,
    d_tokens: i32 = B_MAX,
    d_next_tokens: i32 = B_MAX,
}

pub struct State {
    pub arena: Arena,
    pub s: Slots,
    pub visits: Vec<VisitState>,
    pub max_ctx: usize,
}

impl State {
    /// Plans and allocates the state for contexts of up to `max_ctx` positions.
    pub fn new(max_ctx: usize) -> Result<State> {
        assert!(max_ctx >= LOCAL_KV_CAP, "max_ctx {max_ctx} below the local window");
        let mut p = ArenaPlan::new();
        let s = plan_slots(&mut p);
        let fp8 = std::env::var_os("INKLING_KV_FP8").is_some_and(|v| v != "0");
        let visits = SCHEDULE
            .iter()
            .map(|v| {
                let kv_cap = if v.global { max_ctx + B_MAX } else { local_kv_cap() };
                let i = v.index;
                let ring_cap = if fp8 && v.global { GLOBAL_KV_RING } else { kv_cap };
                VisitState {
                    kv_cap,
                    ring_cap,
                    k_cache: p.reserve_of::<u16>(format!("v{i}.k_cache"), ring_cap * dims::KV_DIM),
                    v_cache: p.reserve_of::<u16>(format!("v{i}.v_cache"), ring_cap * dims::KV_DIM),
                    k_ring: p.reserve_of::<f32>(format!("v{i}.k_ring"), RING_SLOTS * dims::KV_DIM),
                    v_ring: p.reserve_of::<f32>(format!("v{i}.v_ring"), RING_SLOTS * dims::KV_DIM),
                    attn_ring: p.reserve_of::<f32>(format!("v{i}.attn_ring"), RING_SLOTS * H),
                    mlp_ring: p.reserve_of::<f32>(format!("v{i}.mlp_ring"), RING_SLOTS * H),
                    kv_fp8: (fp8 && v.global).then(|| p.reserve_of::<u8>(format!("v{i}.kv_fp8"), kv_fp8_bytes(kv_cap))),
                }
            })
            .collect();
        Ok(State { arena: p.allocate()?, s, visits, max_ctx })
    }

    pub fn get(&self, slot: Slot) -> DevSlice {
        self.arena.get(slot)
    }

    /// Buffers that hold positions modulo a ring (the short-conv rings of every visit and the local
    /// visits' KV caches). A snapshot of these plus the global KV rows below `p` is the full sequence
    /// state at `p`: global caches are indexed by absolute position, so later steps never touch them.
    pub fn rolling_slices(&self) -> Vec<DevSlice> {
        let mut out = Vec::new();
        for v in &self.visits {
            for slot in [v.k_ring, v.v_ring, v.attn_ring, v.mlp_ring] {
                out.push(self.get(slot));
            }
            // Ring-indexed BF16 caches: the local windows, and the global rings when KV is FP8.
            if v.kv_cap == local_kv_cap() || v.kv_fp8.is_some() {
                out.push(self.get(v.k_cache));
                out.push(self.get(v.v_cache));
            }
        }
        out
    }

    /// Starts a new sequence: zeroes every short-conv ring, the position and the sampling config
    /// (greedy). KV caches are not cleared; attention never reads positions at or beyond `d_pos`.
    pub fn reset(&self, stream: &Stream) -> Result<()> {
        for v in &self.visits {
            for slot in [v.k_ring, v.v_ring, v.attn_ring, v.mlp_ring] {
                self.get(slot).fill_u8(0, stream)?;
            }
        }
        self.get(self.s.d_pos).fill_u8(0, stream)?;
        self.get(self.s.sample_cfg).fill_u8(0, stream)?;
        stream.synchronize()
    }

    /// Sets the device position (the absolute position of the next token) and the input token.
    pub fn set_position_and_token(&self, pos: i32, token: i32) -> Result<()> {
        self.get(self.s.d_pos).upload(&[pos])?;
        self.get(self.s.d_tokens).upload(&[token])
    }

    pub fn total_bytes(&self) -> usize {
        self.arena.total_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_and_global_caches_are_sized_from_the_schedule() {
        // Planning only (no allocation): rebuild the plan the way State::new does.
        let mut p = ArenaPlan::new();
        let _ = plan_slots(&mut p);
        let before = p.total_bytes();
        let ctx = 32768;
        let mut kv = 0usize;
        for v in SCHEDULE.iter() {
            let cap = if v.global { ctx + B_MAX } else { LOCAL_KV_CAP };
            kv += 2 * cap * dims::KV_DIM * 2;
        }
        // 40 local visits x 2 x 1280 x 1024 x 2 B + 8 global visits x 2 x 32776 x 1024 x 2 B
        assert_eq!(kv, 40 * 2 * 1280 * 1024 * 2 + 8 * 2 * 32776 * 1024 * 2);
        assert!(before > 0);
    }
}
