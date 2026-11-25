//! FP8 dense attention (part `dense_fp8`): QRGS, K|V and O re-quantised at load from BF16 to E4M3 with
//! one f32 scale per output row (`amax_row / 448`, squid5090's FP8 `pack_linear`), run by the tensor-core
//! FP8 kernels with f32 accumulation (`crates/inkling-kernels/src/fp8.rs`).

use inkling_model::dims;
use inkling_model::weights::Weights;
use inkling_runtime::{Arena, ArenaPlan, Args, DevSlice, LaunchConfig, Module, Slot, Stream};

const H: usize = dims::HIDDEN;

/// An FP8 matrix: E4M3 codes `[rows, k]` and one f32 scale per row.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Mat {
    pub data: DevSlice,
    pub scales: DevSlice,
    pub rows: usize,
    pub k: usize,
}

impl Fp8Mat {
    /// Rows `[r0, r0 + n)`.
    pub fn rows(&self, r0: usize, n: usize) -> Fp8Mat {
        Fp8Mat { data: self.data.sub(r0 * self.k, n * self.k), scales: self.scales.sub(r0 * 4, n * 4), rows: n, k: self.k }
    }
}

pub struct Fp8Layer {
    pub qrgs: Fp8Mat,
    pub kv: Fp8Mat,
    pub wo: Fp8Mat,
}

pub struct Fp8Dense {
    pub arena: Arena,
    pub layers: Vec<Fp8Layer>,
    pub pack_seconds: f64,
}

impl Fp8Dense {
    pub fn pack(module: &Module, w: &Weights, stream: &Stream) -> Result<Fp8Dense, String> {
        let t = std::time::Instant::now();
        let pack = module.function("pack_bf16_to_fp8_rows").ok_or("kernel pack_bf16_to_fp8_rows missing from the cubin")?;
        let mut plan = ArenaPlan::new();
        let shapes = [("attn.qrgs", dims::QRGS_ROWS, H), ("attn.kv", 2 * dims::KV_DIM, H), ("attn.wo_ud.weight", H, dims::Q_DIM)];
        let slots: Vec<Vec<(Slot, Slot)>> = (0..dims::LAYERS)
            .map(|l| {
                shapes
                    .iter()
                    .map(|&(n, rows, k)| {
                        (plan.reserve(format!("fp8.{l}.{n}"), rows * k), plan.reserve_of::<f32>(format!("fp8.{l}.{n}.scales"), rows))
                    })
                    .collect()
            })
            .collect();
        let arena = plan.allocate().map_err(|e| e.to_string())?;
        let mut layers = Vec::new();
        for (l, sl) in slots.iter().enumerate() {
            let mut mats = Vec::new();
            for (i, &(name, rows, k)) in shapes.iter().enumerate() {
                let src = w.get(&format!("layers.{l}.{name}"));
                assert_eq!(src.bytes, rows * k * 2, "{name}: BF16 source size");
                let (data, scales) = (arena.get(sl[i].0), arena.get(sl[i].1));
                let mut a = Args::new();
                a.push(src).push(data).push(scales).push(rows as i32).push(k as i32);
                unsafe { pack.launch(&LaunchConfig::new(rows as u32, 256u32), stream, &mut a) }.map_err(|e| e.to_string())?;
                mats.push(Fp8Mat { data, scales, rows, k });
            }
            layers.push(Fp8Layer { qrgs: mats[0], kv: mats[1], wo: mats[2] });
        }
        stream.synchronize().map_err(|e| e.to_string())?;
        Ok(Fp8Dense { arena, layers, pack_seconds: t.elapsed().as_secs_f64() })
    }
}
