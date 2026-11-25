//! Engine setup shared by `parity`, `generate` and `bench`.

use inkling_engine::{Forward, State};
use inkling_model::loader::LoadOptions;
use inkling_model::weights::{self, Weights};
use inkling_model::{Config, dims};
use inkling_runtime::{Gpu, Module, Stream};
use std::path::{Path, PathBuf};

/// A pinned snapshot (see kernels/SNAPSHOT.txt), so kernel-parity rebuilds cannot race engine runs.
pub const DEFAULT_CUBIN: &str = "kernels/inkling_kernels.cubin";

/// Fields drop in declaration order, so `gpu` comes last: its drop releases the primary context,
/// and the stream, weight buffers and module must be destroyed while that context is alive (the
/// other order crashed at process exit in nvcuda64.dll, 0xC0000005). Objects built from an Engine
/// (State, Forward, graphs, Dspark) are locals declared after it, so they drop before it.
pub struct Engine {
    pub stream: Stream,
    pub weights: Weights,
    pub module: Module,
    pub load_seconds: f64,
    pub gpu: Gpu,
}

pub fn cubin_path(arg: Option<&PathBuf>) -> PathBuf {
    arg.cloned().unwrap_or_else(|| {
        // Relative to the inkling-2-mini project root (the binary lives in target/release).
        let exe = std::env::current_exe().unwrap_or_default();
        let root = exe.ancestors().nth(3).map(Path::to_path_buf).unwrap_or_default();
        root.join(DEFAULT_CUBIN)
    })
}

impl Engine {
    pub fn open(model_dir: &Path, cubin: &Path) -> Result<Engine, String> {
        Self::open_split(model_dir, cubin, |_| false)
    }

    /// As `open`, loading the weights for which `transient(name)` holds into a second arena that
    /// `weights.free_transient()` releases once the quantized copies and graphs are built.
    pub fn open_split(model_dir: &Path, cubin: &Path, transient: impl Fn(&str) -> bool) -> Result<Engine, String> {
        Config::load(model_dir)?;
        let gpu = Gpu::open().map_err(|e| e.to_string())?;
        let stream = Stream::new().map_err(|e| e.to_string())?;
        let image = std::fs::read(cubin).map_err(|e| format!("{}: {e}", cubin.display()))?;
        let module = Module::from_cubin(&image, cubin.display().to_string()).map_err(|e| e.to_string())?;
        let t = std::time::Instant::now();
        let weights = weights::load_exact_split(model_dir, &stream, LoadOptions::default(), transient)?;
        Ok(Engine { gpu, stream, weights, module, load_seconds: t.elapsed().as_secs_f64() })
    }
}

/// Runs steps eagerly. `step(token, write_logits)` feeds `token` at the current device position.
pub struct Runner<'a> {
    pub fwd: Forward<'a>,
    pub state: &'a State,
    pub stream: &'a Stream,
}

impl Runner<'_> {
    pub fn reset(&self) -> Result<(), String> {
        self.state.reset(self.stream).map_err(|e| e.to_string())
    }

    /// One token step at the current position; returns the greedy next token.
    pub fn step(&mut self, token: u32, write_logits: bool) -> Result<u32, String> {
        self.state.get(self.state.s.d_tokens).upload(&[token as i32]).map_err(|e| e.to_string())?;
        self.fwd.write_logits = write_logits;
        self.fwd.record_step(self.stream)?;
        self.stream.synchronize().map_err(|e| e.to_string())?;
        let next: Vec<i32> = self.state.get(self.state.s.d_next_tokens).download(1).map_err(|e| e.to_string())?;
        Ok(next[0] as u32)
    }

    /// Errors when positions `[0, end)` do not fit the state (`--max-ctx`). Global KV caches index
    /// `p % kv_cap` with no bounds check, so running past them would silently overwrite position 0 onward.
    pub fn ensure_room(&self, end: usize) -> Result<(), String> {
        if end > self.state.max_ctx {
            return Err(format!(
                "context overflow: {end} positions needed but the state holds {} (raise --max-ctx)",
                self.state.max_ctx
            ));
        }
        Ok(())
    }

    /// Prefill by graph replay: each prompt token is copied into `d_tokens` with a stream-ordered async copy
    /// from pinned memory, then the captured step graph runs. One host sync at the end. Returns the
    /// greedy token after the last prompt token.
    pub fn prefill_graph(&self, graph: &inkling_runtime::GraphExec, prompt: &[u32]) -> Result<u32, String> {
        self.ensure_room(prompt.len())?;
        let mut host = inkling_runtime::PinnedBuffer::alloc(prompt.len() * 4)?;
        for (dst, &t) in host.as_mut_slice().as_chunks_mut::<4>().0.iter_mut().zip(prompt) {
            *dst = (t as i32).to_ne_bytes();
        }
        let d_tok = self.state.get(self.state.s.d_tokens).sub(0, 4);
        for i in 0..prompt.len() {
            host.copy_range_to_device_async(4 * i, &d_tok, self.stream)?;
            graph.launch(self.stream)?;
        }
        self.stream.synchronize()?;
        Ok(self.state.get(self.state.s.d_next_tokens).download::<i32>(1)?[0] as u32)
    }

    /// Chunked prefill: prompt tokens except the last go through the B = 8/4/2 graphs (positions written to
    /// `d_pos` before each chunk), the last through the B = 1 graph, which advances `d_pos` and feeds the
    /// next token into `d_tokens[0]` for decode. `graphs` holds the B = 8, 4, 2 step graphs. All copies
    /// are stream-ordered from pinned memory; one host sync at the end.
    pub fn prefill_chunked(
        &self,
        graphs: &[(i32, inkling_runtime::GraphExec)],
        g1: &inkling_runtime::GraphExec,
        prompt: &[u32],
    ) -> Result<u32, String> {
        self.ensure_room(prompt.len())?;
        let n = prompt.len();
        // Layout: tokens, then one position per chunk (worst case one per token).
        let mut host = inkling_runtime::PinnedBuffer::alloc(8 * n + 8)?;
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i + 1 < n {
            let left = n - 1 - i;
            let b = graphs.iter().map(|(b, _)| *b as usize).find(|&b| b <= left).unwrap_or(1);
            chunks.push((i, b));
            i += b;
        }
        chunks.push((n - 1, 1));
        {
            let h = host.as_mut_slice();
            for (j, &t) in prompt.iter().enumerate() {
                h[4 * j..4 * j + 4].copy_from_slice(&(t as i32).to_ne_bytes());
            }
            for (c, &(start, _)) in chunks.iter().enumerate() {
                let o = 4 * n + 4 * c;
                h[o..o + 4].copy_from_slice(&(start as i32).to_ne_bytes());
            }
        }
        let d_tok = self.state.get(self.state.s.d_tokens);
        let d_pos = self.state.get(self.state.s.d_pos).sub(0, 4);
        for (c, &(start, b)) in chunks.iter().enumerate() {
            host.copy_range_to_device_async(4 * start, &d_tok.sub(0, 4 * b), self.stream)?;
            host.copy_range_to_device_async(4 * n + 4 * c, &d_pos, self.stream)?;
            if b == 1 {
                g1.launch(self.stream)?;
            } else {
                graphs.iter().find(|(gb, _)| *gb as usize == b).expect("graph for chunk").1.launch(self.stream)?;
            }
        }
        self.stream.synchronize()?;
        Ok(self.state.get(self.state.s.d_next_tokens).download::<i32>(1)?[0] as u32)
    }

    pub fn logits(&self) -> Result<Vec<f32>, String> {
        self.fwd.logits_slice().download(dims::VOCAB).map_err(|e| e.to_string())
    }
}
