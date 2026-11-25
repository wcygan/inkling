//! nvfp4 mode: load-time NVFP4 packing of the dense weights, as squid5090 does in `--mode nvfp4`.
//!
//! - Dense attention (QRGS as one tensor, K|V as one tensor, O), the router, the controllers and the
//!   200,058-row LM head are re-quantised from BF16: one global scale per tensor
//!   `max(amax / 2688, 1e-12)` and an E4M3 scale per 16 values.
//! - Experts in layers 20–23 are converted from MXFP8 to NVFP4 with a per-expert global scale, so every
//!   layer runs the NVFP4 MoE kernels.
//! - The BF16 originals stay loaded (exact mode remains available for comparison).

use inkling_model::dims;
use inkling_model::weights::Weights;
use inkling_runtime::{Arena, ArenaPlan, Args, DevSlice, DeviceBuffer, Kernel, LaunchConfig, Module, Slot, Stream};

const H: usize = dims::HIDDEN;

/// An NVFP4 matrix: codes `[rows, K/2]`, E4M3 scales `[rows, K/16]` and one global scale.
#[derive(Clone, Copy, Debug)]
pub struct Packed {
    pub data: DevSlice,
    pub scales: DevSlice,
    pub gs: f32,
    pub rows: usize,
    pub k: usize,
}

impl Packed {
    /// Rows `[r0, r0 + n)` of this matrix (same global scale).
    pub fn rows(&self, r0: usize, n: usize) -> Packed {
        Packed {
            data: self.data.sub(r0 * self.k / 2, n * self.k / 2),
            scales: self.scales.sub(r0 * self.k / 16, n * self.k / 16),
            gs: self.gs,
            rows: n,
            k: self.k,
        }
    }
}

/// NVFP4 experts for one layer: codes, E4M3 scales and per-expert global scales.
#[derive(Clone, Copy, Debug)]
pub struct Experts {
    pub w13: DevSlice,
    pub w13_scales: DevSlice,
    pub w13_factor: DevSlice,
    pub w2: DevSlice,
    pub w2_scales: DevSlice,
    pub w2_factor: DevSlice,
}

pub struct LayerQ {
    pub qrgs: Packed,
    pub kv: Packed,
    pub wo: Packed,
    pub router: Packed,
    /// `[attn, k, v, mlp]` x `[down, forget, gain]`, layers 6–17.
    pub ctrl: Option<[[Packed; 3]; 4]>,
    pub experts: Experts,
}

pub struct Nvfp4Weights {
    pub arena: Arena,
    pub layers: Vec<LayerQ>,
    pub unembed: Packed,
    pub pack_seconds: f64,
}

struct PackKernels {
    amax: Kernel,
    pack: Kernel,
    mx_amax: Kernel,
    mx_repack: Kernel,
}

enum Planned {
    Dense { src: DevSlice, rows: usize, k: usize, data: Slot, scales: Slot },
    Mxfp8 { src_w: DevSlice, src_s: DevSlice, rows: usize, cols: usize, data: Slot, scales: Slot, factor: Slot },
}

fn dense(plan: &mut ArenaPlan, jobs: &mut Vec<Planned>, name: String, src: DevSlice, rows: usize, kdim: usize) -> usize {
    assert_eq!(src.bytes, rows * kdim * 2, "{name}: BF16 source size");
    let data = plan.reserve(format!("{name}.nvfp4"), rows * kdim / 2);
    let scales = plan.reserve(format!("{name}.scales"), rows * kdim / 16);
    jobs.push(Planned::Dense { src, rows, k: kdim, data, scales });
    jobs.len() - 1
}

fn launch(k: Kernel, grid: u32, block: u32, stream: &Stream, mut args: Args) -> Result<(), String> {
    unsafe { k.launch(&LaunchConfig::new(grid, block), stream, &mut args) }.map_err(|e| e.to_string())
}

/// Packs one BF16 `[rows, k]` matrix into `data` / `scales` (same scheme as the dense packs above) and
/// returns its global scale. Synchronizes `stream` once, to read the amax.
pub(crate) fn pack_dense(module: &Module, src: DevSlice, rows: usize, k: usize, data: DevSlice, scales: DevSlice, stream: &Stream) -> Result<f32, String> {
    assert_eq!(src.bytes, rows * k * 2, "pack_dense: BF16 source size");
    let f = |n: &str| module.function(n).ok_or_else(|| format!("kernel {n} missing from the cubin"));
    let amax_buf = DeviceBuffer::alloc(4)?;
    amax_buf.slice().fill_u8(0, stream)?;
    let n = (rows * k) as u64;
    let mut a = Args::new();
    a.push(src).push(n).push(amax_buf.slice());
    launch(f("amax_bf16")?, n.div_ceil(256).min(2048) as u32, 256, stream, a)?;
    stream.synchronize()?;
    let amax = amax_buf.slice().download::<f32>(1)?[0];
    if !amax.is_finite() {
        return Err(format!("pack_dense: amax {amax}"));
    }
    let gs = (amax / 2688.0).max(1e-12);
    let mut a = Args::new();
    a.push(src).push(data).push(scales).push(rows as i32).push(k as i32).push(gs);
    launch(f("pack_bf16_to_nvfp4")?, ((rows * k / 16).div_ceil(256)) as u32, 256, stream, a)?;
    Ok(gs)
}

impl Nvfp4Weights {
    pub fn pack(module: &Module, w: &Weights, stream: &Stream) -> Result<Nvfp4Weights, String> {
        let t = std::time::Instant::now();
        let f = |n: &str| module.function(n).ok_or_else(|| format!("kernel {n} missing from the cubin"));
        let k = PackKernels {
            amax: f("amax_bf16")?,
            pack: f("pack_bf16_to_nvfp4")?,
            mx_amax: f("mxfp8_expert_amax")?,
            mx_repack: f("repack_mxfp8_to_nvfp4")?,
        };
        let mut plan = ArenaPlan::new();
        let mut jobs: Vec<Planned> = Vec::new();
        struct LayerJobs {
            qrgs: usize,
            kv: usize,
            wo: usize,
            router: usize,
            ctrl: Option<[[usize; 3]; 4]>,
            mx: Option<[usize; 2]>,
        }
        let mut layer_jobs = Vec::new();
        for l in 0..dims::LAYERS {
            let g = |s: &str| w.get(&format!("layers.{l}.{s}"));
            let qrgs = dense(&mut plan, &mut jobs, format!("layers.{l}.qrgs"), g("attn.qrgs"), dims::QRGS_ROWS, H);
            let kv = dense(&mut plan, &mut jobs, format!("layers.{l}.kv"), g("attn.kv"), 2 * dims::KV_DIM, H);
            let wo = dense(&mut plan, &mut jobs, format!("layers.{l}.wo"), g("attn.wo_ud.weight"), H, dims::Q_DIM);
            let router = dense(&mut plan, &mut jobs, format!("layers.{l}.router"), g("mlp.gate.weight"), dims::EXPERTS, H);
            let ctrl = (dims::LOOP_START..dims::LOOP_END).contains(&l).then(|| {
                ["attn", "k", "v", "mlp"].map(|c| {
                    [
                        dense(&mut plan, &mut jobs, format!("layers.{l}.ctrl.{c}.down"), g(&format!("controllers.{c}.down_weight")), dims::CONTROLLER_RANK, 2 * H),
                        dense(&mut plan, &mut jobs, format!("layers.{l}.ctrl.{c}.forget"), g(&format!("controllers.{c}.forget_weight")), H, dims::CONTROLLER_RANK),
                        dense(&mut plan, &mut jobs, format!("layers.{l}.ctrl.{c}.gain"), g(&format!("controllers.{c}.gain_weight")), H, dims::CONTROLLER_RANK),
                    ]
                })
            });
            let mx = (l >= dims::NVFP4_EXPERT_LAYERS).then(|| {
                [("w13", 2 * dims::INTERMEDIATE, H), ("w2", H, dims::INTERMEDIATE)].map(|(m, rows, cols)| {
                    let n = dims::EXPERTS * rows * cols;
                    let data = plan.reserve(format!("layers.{l}.{m}.nvfp4"), n / 2);
                    let scales = plan.reserve(format!("layers.{l}.{m}.scales"), n / 16);
                    let factor = plan.reserve_of::<f32>(format!("layers.{l}.{m}.expert_scale"), dims::EXPERTS);
                    jobs.push(Planned::Mxfp8 {
                        src_w: g(&format!("mlp.experts.{m}_weight")),
                        src_s: g(&format!("mlp.experts.{m}_weight_scale")),
                        rows,
                        cols,
                        data,
                        scales,
                        factor,
                    });
                    jobs.len() - 1
                })
            });
            layer_jobs.push(LayerJobs { qrgs, kv, wo, router, ctrl, mx });
        }
        let unembed_src = w.get("unembed.weight").sub(0, dims::VOCAB * H * 2);
        let unembed = dense(&mut plan, &mut jobs, "unembed".into(), unembed_src, dims::VOCAB, H);
        let arena = plan.allocate()?;

        // Run the packing jobs. Dense tensors need their amax on the host for the global scale.
        let amax_buf = DeviceBuffer::alloc(4)?;
        let mut gs = vec![0f32; jobs.len()];
        // INKLING_NVFP4_MSE=<steps>: host repack of dense tensors choosing each block's E4M3 scale by
        // least squared error among `steps` values below squid5090's max-based choice (and one above).
        let mse_search: Option<usize> = std::env::var("INKLING_NVFP4_MSE").ok().and_then(|v| v.parse().ok());
        for (i, job) in jobs.iter().enumerate() {
            match *job {
                Planned::Dense { src, rows, k: kdim, data, scales } => {
                    let n = (rows * kdim) as u64;
                    amax_buf.slice().fill_u8(0, stream)?;
                    let mut a = Args::new();
                    a.push(src).push(n).push(amax_buf.slice());
                    launch(k.amax, n.div_ceil(256).min(2048) as u32, 256, stream, a)?;
                    stream.synchronize()?;
                    let amax = amax_buf.slice().download::<f32>(1)?[0];
                    if !amax.is_finite() {
                        return Err(format!("job {i}: amax {amax}"));
                    }
                    gs[i] = (amax / 2688.0).max(1e-12);
                    let mut a = Args::new();
                    a.push(src)
                        .push(arena.get(data))
                        .push(arena.get(scales))
                        .push(rows as i32)
                        .push(kdim as i32)
                        .push(gs[i]);
                    launch(k.pack, ((rows * kdim / 16).div_ceil(256)) as u32, 256, stream, a)?;
                    if let Some(search) = mse_search {
                        // Repack on the host with error-minimising block scales (same format and gs).
                        stream.synchronize()?;
                        let bf16: Vec<u16> = src.download(rows * kdim)?;
                        let (codes, block_scales) = crate::mse::pack(&bf16, rows, kdim, gs[i], search);
                        arena.get(data).upload(&codes)?;
                        arena.get(scales).upload(&block_scales)?;
                    }
                }
                Planned::Mxfp8 { src_w, src_s, rows, cols, data, scales, factor } => {
                    let mut a = Args::new();
                    a.push(src_w).push(src_s).push(arena.get(factor)).push(rows as i32).push(cols as i32);
                    // One block per expert (squid5090 cuda_abi.cu:498, `<<<n_experts, 256>>>`).
                    launch(k.mx_amax, dims::EXPERTS as u32, 256, stream, a)?;
                    let mut a = Args::new();
                    a.push(src_w)
                        .push(src_s)
                        .push(arena.get(factor))
                        .push(arena.get(data))
                        .push(arena.get(scales))
                        .push(dims::EXPERTS as i32)
                        .push(rows as i32)
                        .push(cols as i32);
                    launch(k.mx_repack, ((dims::EXPERTS * rows * cols / 16).div_ceil(256)) as u32, 256, stream, a)?;
                }
            }
        }
        stream.synchronize()?;

        let packed = |i: usize| -> Packed {
            let Planned::Dense { rows, k: kdim, data, scales, .. } = jobs[i] else { unreachable!() };
            Packed { data: arena.get(data), scales: arena.get(scales), gs: gs[i], rows, k: kdim }
        };
        let layers = layer_jobs
            .iter()
            .enumerate()
            .map(|(l, j)| {
                let g = |s: &str| w.get(&format!("layers.{l}.{s}"));
                let experts = match j.mx {
                    None => Experts {
                        w13: g("mlp.experts.w13_weight"),
                        w13_scales: g("mlp.experts.w13_weight_scale"),
                        w13_factor: g("mlp.experts.w13_global_scale"),
                        w2: g("mlp.experts.w2_weight"),
                        w2_scales: g("mlp.experts.w2_weight_scale"),
                        w2_factor: g("mlp.experts.w2_global_scale"),
                    },
                    Some([a, b]) => {
                        let parts = |i: usize| {
                            let Planned::Mxfp8 { data, scales, factor, .. } = jobs[i] else { unreachable!() };
                            (arena.get(data), arena.get(scales), arena.get(factor))
                        };
                        let ((w13, w13_scales, w13_factor), (w2, w2_scales, w2_factor)) = (parts(a), parts(b));
                        Experts { w13, w13_scales, w13_factor, w2, w2_scales, w2_factor }
                    }
                };
                LayerQ {
                    qrgs: packed(j.qrgs),
                    kv: packed(j.kv),
                    wo: packed(j.wo),
                    router: packed(j.router),
                    ctrl: j.ctrl.map(|c| c.map(|m| m.map(packed))),
                    experts,
                }
            })
            .collect();
        let unembed = packed(unembed);
        Ok(Nvfp4Weights { arena, layers, unembed, pack_seconds: t.elapsed().as_secs_f64() })
    }
}
