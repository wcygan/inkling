//! Prompt-lookup drafting: squid5090's `PromptLookupIndex` (src/engine.zig:38-110), host only.
//!
//! Keeps the last 8,192 tokens (prompt + generated) in a ring and a 16,384-slot hash of every 4-gram
//! with its last two end positions. When the last 16 tokens also occurred earlier, the 7 tokens that
//! followed that occurrence are proposed as drafts for the B = 8 verify, skipping the DSpark draft.
//! `min_match_len` is 16 (squid5090 found 6 triggered on false matches).

const HIST_CAP: usize = 8_192;
const HIST_MASK: usize = HIST_CAP - 1;
const TABLE_CAP: usize = 16_384;
const TABLE_MASK: usize = TABLE_CAP - 1;
pub const MIN_MATCH_LEN: u32 = 16;
pub const DRAFTS: usize = 7;

pub struct PromptLookupIndex {
    hist: Vec<i32>,
    count: u32,
    gen_: u16,
    table_hash: Vec<u64>,
    table_pos0: Vec<u32>,
    table_pos1: Vec<u32>,
    table_gen: Vec<u16>,
}

fn hash4(a: i32, b: i32, c: i32, d: i32) -> u64 {
    let mut h = (a as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = (h ^ b as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ c as i64 as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    h = (h ^ d as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^ (h >> 31)
}

impl Default for PromptLookupIndex {
    fn default() -> Self {
        PromptLookupIndex {
            hist: vec![0; HIST_CAP],
            count: 0,
            gen_: 0,
            table_hash: vec![0; TABLE_CAP],
            table_pos0: vec![0; TABLE_CAP],
            table_pos1: vec![0; TABLE_CAP],
            table_gen: vec![0; TABLE_CAP],
        }
    }
}

impl PromptLookupIndex {
    pub fn reset(&mut self) {
        self.count = 0;
        self.gen_ = self.gen_.wrapping_add(1);
        if self.gen_ == 0 {
            self.table_gen.fill(0);
            self.gen_ = 1;
        }
    }

    fn at(&self, abs: u32) -> i32 {
        self.hist[abs as usize & HIST_MASK]
    }

    pub fn push(&mut self, tok: i32) {
        let idx = self.count;
        self.hist[idx as usize & HIST_MASK] = tok;
        self.count = idx + 1;
        if self.count >= 4 {
            let n = self.count;
            let h = hash4(self.at(n - 4), self.at(n - 3), self.at(n - 2), self.at(n - 1));
            let slot = (h as usize) & TABLE_MASK;
            if self.table_gen[slot] == self.gen_ && self.table_hash[slot] == h {
                self.table_pos1[slot] = self.table_pos0[slot];
                self.table_pos0[slot] = n;
            } else {
                self.table_hash[slot] = h;
                self.table_pos0[slot] = n;
                self.table_pos1[slot] = 0;
                self.table_gen[slot] = self.gen_;
            }
        }
    }

    /// The 7 tokens after the latest earlier occurrence of the last 16 tokens, if there is one.
    pub fn find(&self) -> Option<[i32; DRAFTS]> {
        let n = self.count;
        if n < MIN_MATCH_LEN + DRAFTS as u32 + 1 {
            return None;
        }
        let h = hash4(self.at(n - 4), self.at(n - 3), self.at(n - 2), self.at(n - 1));
        let slot = (h as usize) & TABLE_MASK;
        if self.table_gen[slot] != self.gen_ || self.table_hash[slot] != h {
            return None;
        }
        let cand = if self.table_pos0[slot] == n { self.table_pos1[slot] } else { self.table_pos0[slot] };
        if cand < MIN_MATCH_LEN || cand + DRAFTS as u32 >= n {
            return None;
        }
        let win_lo = n.saturating_sub(HIST_CAP as u32);
        if cand - MIN_MATCH_LEN < win_lo {
            return None;
        }
        if (1..=MIN_MATCH_LEN).any(|k| self.at(cand - k) != self.at(n - k)) {
            return None;
        }
        let mut out = [0i32; DRAFTS];
        for (j, o) in out.iter_mut().enumerate() {
            *o = self.at(cand + j as u32);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_continuation_of_a_repeated_span() {
        let mut ix = PromptLookupIndex::default();
        ix.reset();
        let span: Vec<i32> = (100..140).collect();
        for &t in &span {
            ix.push(t);
        }
        for t in 0..5 {
            ix.push(t);
        }
        // Repeat the first 20 tokens of the span: the last 16 match, the next 7 are span[20..27].
        for &t in &span[..20] {
            ix.push(t);
        }
        assert_eq!(ix.find(), Some([120, 121, 122, 123, 124, 125, 126]));
        // A fresh token breaks the match.
        ix.push(9999);
        assert_eq!(ix.find(), None);
    }
}
