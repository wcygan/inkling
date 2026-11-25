//! Host NVFP4 packing with error-minimising block scales.
//!
//! Same format as squid5090's `z_quant_bf16_to_nvfp4_kernel`: codes `[rows][k/2]` (low nibble = even
//! element), one E4M3 block scale per 16 values `[rows][k/16]`, and a per-tensor global scale `gs`
//! (dequantised weight = e2m1 · e4m3 · gs). squid5090 picks each block scale as `blockmax / (6 · gs)`
//! rounded to E4M3. Here each block instead tries the E4M3 values around that choice (smaller ones clip
//! the block's largest value but represent the rest more finely) and keeps the one with the least squared
//! reconstruction error. Kernels are unchanged.

/// The eight non-negative E2M1 magnitudes, by 3-bit code.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e4m3_value(code: u8) -> f32 {
    let e = (code >> 3) & 0xf;
    let m = (code & 7) as f32;
    let mag = if e == 0 { m / 8.0 * 2f32.powi(-6) } else { (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7) };
    if code & 0x80 != 0 { -mag } else { mag }
}

/// Positive finite E4M3 values with their codes, ascending (codes 0x01..=0x7E).
fn e4m3_table() -> Vec<(f32, u8)> {
    (1u8..=0x7e).map(|c| (e4m3_value(c), c)).collect()
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Nearest E2M1 code (sign in bit 3) for `x`, already divided by the block's effective scale.
fn e2m1_code(x: f32) -> (u8, f32) {
    let a = x.abs().min(6.0);
    // Nearest of the eight magnitudes (ties toward the larger, as the squid5090 kernel's thresholds do).
    let mut best = 0usize;
    for (i, &v) in E2M1.iter().enumerate() {
        if (a - v).abs() <= (a - E2M1[best]).abs() {
            best = i;
        }
    }
    let sign = if x < 0.0 && best != 0 { 8 } else { 0 };
    (sign | best as u8, if x < 0.0 { -E2M1[best] } else { E2M1[best] })
}

/// Packs one block of 16 values. Returns the E4M3 scale code and writes 8 code bytes.
fn pack_block(x: &[f32; 16], gs: f32, table: &[(f32, u8)], search: usize, out: &mut [u8; 8]) -> u8 {
    let bmax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    if bmax == 0.0 {
        *out = [0; 8];
        return 0;
    }
    // squid5090's choice: blockmax / (6 gs), clamped to 448, rounded to the nearest E4M3 value.
    let target = (bmax / (6.0 * gs)).min(448.0);
    let mut idx = table.partition_point(|&(v, _)| v < target);
    if idx == table.len() || (idx > 0 && (target - table[idx - 1].0) <= (table[idx].0 - target)) {
        idx = idx.saturating_sub(1).min(table.len() - 1);
    }
    let lo = idx.saturating_sub(search);
    let hi = (idx + 2).min(table.len());
    let (mut best_err, mut best_c) = (f32::INFINITY, table[idx].1);
    for &(s, c) in &table[lo..hi] {
        let eff = s * gs;
        let err: f32 = x.iter().map(|&v| {
            let (_, q) = e2m1_code(v / eff);
            let d = v - q * eff;
            d * d
        }).sum();
        // Prefer squid5090's own choice on ties so the method only changes blocks it improves.
        if err < best_err || (err == best_err && c == table[idx].1) {
            best_err = err;
            best_c = c;
        }
    }
    let eff = e4m3_value(best_c) * gs;
    for j in 0..8 {
        let (lo_c, _) = e2m1_code(x[2 * j] / eff);
        let (hi_c, _) = e2m1_code(x[2 * j + 1] / eff);
        out[j] = lo_c | (hi_c << 4);
    }
    best_c
}

/// Packs a BF16 matrix `[rows][k]` with global scale `gs`. `search` is how many E4M3 steps below the
/// max-based scale to try (one step above is always tried). Uses all cores.
pub fn pack(src: &[u16], rows: usize, k: usize, gs: f32, search: usize) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(src.len(), rows * k);
    assert!(k.is_multiple_of(16));
    let table = e4m3_table();
    let mut codes = vec![0u8; rows * k / 2];
    let mut scales = vec![0u8; rows * k / 16];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let rows_per = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for ((src, codes), scales) in src
            .chunks(rows_per * k)
            .zip(codes.chunks_mut(rows_per * k / 2))
            .zip(scales.chunks_mut(rows_per * k / 16))
        {
            let table = &table;
            scope.spawn(move || {
                for (xs, (out, s)) in src.as_chunks::<16>().0.iter().zip(codes.as_chunks_mut::<8>().0.iter_mut().zip(scales.iter_mut())) {
                    let x = xs.map(bf16_to_f32);
                    *s = pack_block(&x, gs, table, search, out);
                }
            });
        }
    });
    (codes, scales)
}

/// Dequantises packed NVFP4 back to f32 (for tests and error reports).
pub fn unpack(codes: &[u8], scales: &[u8], gs: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(codes.len() * 2);
    for (b, &s) in scales.iter().enumerate() {
        let eff = e4m3_value(s) * gs;
        for &byte in &codes[8 * b..8 * b + 8] {
            for nib in [byte & 0xf, byte >> 4] {
                let v = E2M1[(nib & 7) as usize];
                out.push(if nib & 8 != 0 { -v } else { v } * eff);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_bf16(x: f32) -> u16 {
        (x.to_bits() >> 16) as u16
    }

    #[test]
    fn e4m3_decodes_known_values() {
        assert_eq!(e4m3_value(0x38), 1.0);
        assert_eq!(e4m3_value(0x7e), 448.0);
        assert_eq!(e4m3_value(0x01), 2f32.powi(-9));
    }

    #[test]
    fn search_never_increases_error() {
        // A heavy-tailed block: one outlier, many small values.
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut src = Vec::new();
        for _ in 0..(64 * 256) {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            src.push(to_bf16(u * u * u * 8.0));
        }
        let amax = src.iter().map(|&b| bf16_to_f32(b).abs()).fold(0f32, f32::max);
        let gs = amax / 2688.0;
        let sse = |search| {
            let (c, sc) = pack(&src, 64, 256, gs, search);
            let back = unpack(&c, &sc, gs);
            src.iter().zip(&back).map(|(&a, &b)| (bf16_to_f32(a) - b).powi(2)).sum::<f32>()
        };
        let (base, searched) = (sse(0), sse(4));
        assert!(searched <= base, "searched {searched} > base {base}");
        assert!(searched < 0.97 * base, "expected a real improvement: {searched} vs {base}");
    }
}
