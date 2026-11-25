//! The 48-visit schedule: prelude layers 0–5, core layers 6–17 run three times, coda layers 18–23.

use crate::config::{dims, is_global_layer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    pub index: usize,
    pub layer: usize,
    /// Loop pass: 0 outside the core, 0–2 inside it.
    pub pass: usize,
    pub global: bool,
    /// The AttnRes block slot this visit opens (layers 0, 6, 12, 18 open slots 0–3).
    pub block_slot: Option<usize>,
    /// The visit restarts the loop (layer 6 on passes 1 and 2): the hidden state is reset to the
    /// prelude output and the cross-pass state (saved block, prior pass's last state) is captured.
    pub loop_restart: bool,
    /// Cross-pass controllers run on this visit (core layers on passes 1 and 2).
    pub controllers: bool,
    /// The MLP post-process must be applied at the end of this visit instead of being deferred into
    /// the next visit's depth-norm (the last core visit of passes 0 and 1, before a loop restart).
    pub flush_mlp: bool,
}

const fn build() -> [Visit; dims::VISITS] {
    let mut out = [Visit {
        index: 0,
        layer: 0,
        pass: 0,
        global: false,
        block_slot: None,
        loop_restart: false,
        controllers: false,
        flush_mlp: false,
    }; dims::VISITS];
    let mut i = 0;
    while i < dims::VISITS {
        let core_len = dims::LOOP_END - dims::LOOP_START;
        let (layer, pass) = if i < dims::LOOP_START {
            (i, 0)
        } else if i < dims::LOOP_START + core_len * dims::LOOP_PASSES {
            let k = i - dims::LOOP_START;
            (dims::LOOP_START + k % core_len, k / core_len)
        } else {
            (i - core_len * (dims::LOOP_PASSES - 1), 0)
        };
        let in_core = layer >= dims::LOOP_START && layer < dims::LOOP_END;
        out[i] = Visit {
            index: i,
            layer,
            pass,
            global: is_global_layer(layer),
            // Every boundary opens a slot on every pass: the block count resets to 1 at each loop
            // restart, so layer 6 always writes slot 1 and layer 12 slot 2 (squid5090 engine.zig:1942, 2220).
            block_slot: if layer % dims::BLOCK_LAYERS == 0 {
                Some(layer / dims::BLOCK_LAYERS)
            } else {
                None
            },
            loop_restart: layer == dims::LOOP_START && pass > 0,
            controllers: in_core && pass > 0,
            flush_mlp: layer == dims::LOOP_END - 1 && pass + 1 < dims::LOOP_PASSES,
        };
        i += 1;
    }
    out
}

pub const SCHEDULE: [Visit; dims::VISITS] = build();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_squid5090_and_hf_loop_structure() {
        let layers: Vec<usize> = SCHEDULE.iter().map(|v| v.layer).collect();
        let mut want: Vec<usize> = (0..6).collect();
        for _ in 0..3 {
            want.extend(6..18);
        }
        want.extend(18..24);
        assert_eq!(layers, want);

        let global: Vec<usize> = SCHEDULE.iter().filter(|v| v.global).map(|v| v.index).collect();
        assert_eq!(global, [5, 11, 17, 23, 29, 35, 41, 47]);

        let flush: Vec<usize> = SCHEDULE.iter().filter(|v| v.flush_mlp).map(|v| v.index).collect();
        assert_eq!(flush, [17, 29]);

        let restart: Vec<usize> = SCHEDULE.iter().filter(|v| v.loop_restart).map(|v| v.index).collect();
        assert_eq!(restart, [18, 30]);

        let controllers = SCHEDULE.iter().filter(|v| v.controllers).count();
        assert_eq!(controllers, 24);
        assert!(SCHEDULE[18..42].iter().all(|v| v.controllers));

        let slots: Vec<(usize, usize)> =
            SCHEDULE.iter().filter_map(|v| v.block_slot.map(|s| (v.index, s))).collect();
        assert_eq!(slots, [(0, 0), (6, 1), (12, 2), (18, 1), (24, 2), (30, 1), (36, 2), (42, 3)]);
    }
}
