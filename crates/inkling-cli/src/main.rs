//! `inkling` — inkling-rs command line.
//!
//!   inkling load-check [--model-dir DIR] [--buffered] [--readback-window BYTES]
//!       Loads the exact-mode weights into one arena, reports load time and VRAM, and compares a sample
//!       of every tensor read back from the GPU with the file.
//!   inkling tokenize [--model-dir DIR] [--effort E] TEXT
//!       Renders TEXT as one user turn with the model's chat template and prints the token IDs.
//!   Engine commands take --mode exact|nvfp4|fp8 (default exact). fp8 is the shippable mode: FP8 dense
//!   attention with NVFP4 experts, head, router and controllers.
//!   inkling generate [--max-tokens N] [--effort E] [--cubin FILE] TEXT
//!       Greedy generation for one user turn (eager steps).
//!   inkling bench [--max-tokens N] [--effort E] [RUNS]
//!       ingine-bench's chat and code prompts. Prefill replays the captured step graph once per prompt
//!       token (INKLING_EAGER_PREFILL=1 runs it eagerly instead); decode replays it once per token with a
//!       host sync per token, as a streaming server does. Default effort low.
//!   inkling kld FILE [PREFIX] [EVAL]
//!       KL divergence of the shipped fp8 mode (token and wide prefill) from exact mode, see kld.rs.
//!   inkling serve --mode fp8 --draft-vocab FILE [--max-ctx N] [--effort E]
//!       OpenAI-compatible chat server on 127.0.0.1:$INKLING_PORT (default 8013), see server.rs.

mod engine;
mod kld;
mod lookup;
mod server;
mod spec;

use inkling_model::loader::LoadOptions;
use inkling_model::{Config, weights};
use inkling_runtime::{Gpu, Stream};
use inkling_text::TextModel;
use std::path::PathBuf;
use std::process::ExitCode;

/// Checkpoint folder when neither --model-dir nor INKLING_MODEL_DIR is given: %USERPROFILE%\Models\Inkling-2-Mini.
fn default_model_dir() -> PathBuf {
    std::env::var_os("USERPROFILE")
        .map(|h| PathBuf::from(h).join("Models").join("Inkling-2-Mini"))
        .unwrap_or_else(|| PathBuf::from("Inkling-2-Mini"))
}

pub(crate) struct Opts {
    model_dir: PathBuf,
    buffered: bool,
    readback_window: usize,
    effort: Option<String>,
    cubin: Option<PathBuf>,
    max_tokens: usize,
    max_ctx: usize,
    nvfp4: bool,
    batch: usize,
    /// DSpark speculative decoding with K drafts verified per step (verify width K + 1).
    spec: Option<usize>,
    /// Draft vocabulary shortlist (little-endian i32 token ids); default: the full vocabulary.
    draft_vocab: Option<PathBuf>,
    /// Adaptive verify width (2/4/8 by a CUDA-graph switch on draft confidence >= threshold).
    spec_threshold: Option<f32>,
    /// Prompt-lookup drafting in the spec bench (n-gram matches go to a B = 8 verify).
    lookup: bool,
    rest: Vec<String>,
}

fn parse(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        model_dir: std::env::var_os("INKLING_MODEL_DIR").map(PathBuf::from).unwrap_or_else(default_model_dir),
        buffered: false,
        readback_window: 4096,
        effort: None,
        cubin: None,
        max_tokens: 256,
        max_ctx: 8192,
        nvfp4: false,
        batch: 1,
        spec: None,
        draft_vocab: None,
        spec_threshold: None,
        lookup: false,
        rest: Vec::new(),
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--model-dir" => o.model_dir = value(a)?.into(),
            "--buffered" => o.buffered = true,
            "--readback-window" => o.readback_window = value(a)?.parse().map_err(|_| "bad --readback-window")?,
            "--effort" => o.effort = Some(value(a)?),
            "--cubin" => o.cubin = Some(value(a)?.into()),
            "--max-tokens" => o.max_tokens = value(a)?.parse().map_err(|_| "bad --max-tokens")?,
            "--max-ctx" => o.max_ctx = value(a)?.parse().map_err(|_| "bad --max-ctx")?,
            "--batch" => o.batch = value(a)?.parse().map_err(|_| "bad --batch")?,
            "--spec" => o.spec = Some(3),
            "--spec-k" => o.spec = Some(value(a)?.parse().map_err(|_| "bad --spec-k")?),
            "--draft-vocab" => o.draft_vocab = Some(value(a)?.into()),
            "--lookup" => o.lookup = true,
            "--spec-adaptive" => {
                o.spec_threshold = Some(value(a)?.parse().map_err(|_| "bad --spec-adaptive")?);
                o.spec.get_or_insert(7);
            }
            "--mode" => {
                o.nvfp4 = match value(a)?.as_str() {
                    "exact" => false,
                    "nvfp4" => true,
                    // The shippable mode: FP8 dense attention, NVFP4 experts/head/router/controllers.
                    "fp8" => {
                        // SAFETY: set during argument parsing, before any other thread starts.
                        unsafe { std::env::set_var("INKLING_NVFP4_PARTS", "experts,head,router,ctrl,dense_fp8") };
                        true
                    }
                    m => return Err(format!("unknown --mode {m} (exact, nvfp4 or fp8)")),
                }
            }
            _ => o.rest.push(a.clone()),
        }
    }
    Ok(o)
}

fn load_check(o: &Opts) -> Result<(), String> {
    Config::load(&o.model_dir)?;
    let gpu = Gpu::open().map_err(|e| e.to_string())?;
    let (free0, total) = gpu.mem_info().map_err(|e| e.to_string())?;
    println!(
        "device: {} sm_{}{}, {} SMs, driver {}, {:.2} of {:.2} GiB free",
        gpu.name,
        gpu.compute_capability.0,
        gpu.compute_capability.1,
        gpu.sm_count,
        gpu.driver_version,
        free0 as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    let stream = Stream::new().map_err(|e| e.to_string())?;
    let opts = LoadOptions { unbuffered: !o.buffered, ..LoadOptions::default() };
    let w = weights::load_exact(&o.model_dir, &stream, opts)?;
    let (free1, _) = gpu.mem_info().map_err(|e| e.to_string())?;
    let s = w.stats;
    println!(
        "loaded {} tensors into one arena of {:.3} GB ({} bytes padding) in {:.2} s: read {:.2} GB in {} chunks, copied {:.3} GB ({:.2} GB/s), {} reads",
        w.names().count(),
        w.arena.total_bytes() as f64 / 1e9,
        w.arena.padding_bytes(),
        s.seconds,
        s.bytes_read as f64 / 1e9,
        s.chunks,
        s.bytes_copied as f64 / 1e9,
        s.bytes_copied as f64 / 1e9 / s.seconds,
        if o.buffered { "buffered" } else { "unbuffered" }
    );
    println!("VRAM used by this process: {:.2} GiB", (free0 - free1) as f64 / (1u64 << 30) as f64);
    let compared = weights::verify_readback(&o.model_dir, &w, o.readback_window)?;
    println!("readback: {:.1} MB sampled across every tensor match the file", compared as f64 / 1e6);
    Ok(())
}

fn tokenize(o: &Opts) -> Result<(), String> {
    let text = TextModel::load(&o.model_dir)?;
    let msgs = serde_json::json!([{"role": "user", "content": o.rest.join(" ")}]);
    let effort = o.effort.as_ref().map(|e| e.parse::<f64>().map(serde_json::Value::from).unwrap_or(e.clone().into()));
    let ids = text.encode_chat(&msgs, None, effort.as_ref())?;
    println!("{}", serde_json::to_string(&ids).unwrap());
    Ok(())
}

pub(crate) fn effort_value(o: &Opts) -> Option<serde_json::Value> {
    o.effort.as_ref().map(|e| e.parse::<f64>().map(serde_json::Value::from).unwrap_or(e.clone().into()))
}

fn generate_cmd(o: &Opts) -> Result<(), String> {
    let text = TextModel::load(&o.model_dir)?;
    let msgs = serde_json::json!([{"role": "user", "content": o.rest.join(" ")}]);
    let prompt = text.encode_chat(&msgs, None, effort_value(o).as_ref())?;
    let eng = engine::Engine::open(&o.model_dir, &engine::cubin_path(o.cubin.as_ref()))?;
    let state = inkling_engine::State::new(o.max_ctx.max(prompt.len() + o.max_tokens + 16)).map_err(|e| e.to_string())?;
    let packed = if o.nvfp4 { Some(inkling_engine::Nvfp4Weights::pack(&eng.module, &eng.weights, &eng.stream)?) } else { None };
    if let Some(q) = &packed {
        println!("nvfp4 mode: packed {:.2} GB of NVFP4 weights in {:.2} s", q.arena.total_bytes() as f64 / 1e9, q.pack_seconds);
    }
    let mut fwd = inkling_engine::Forward::with_mode(&eng.module, &eng.weights, packed.as_ref(), &state)?;
    if let (true, Ok(spec)) = (o.nvfp4, std::env::var("INKLING_NVFP4_PARTS")) {
        fwd.parts = inkling_engine::Nvfp4Parts::parse(&spec)?;
        println!("nvfp4 parts: {:?}", fwd.parts);
    }
    let mut run = engine::Runner { fwd, state: &state, stream: &eng.stream };
    run.reset()?;
    let t0 = std::time::Instant::now();
    let mut next = 0;
    for &tok in &prompt {
        next = run.step(tok, false)?;
    }
    let prefill = t0.elapsed().as_secs_f64();
    let mut out = vec![next];
    let t1 = std::time::Instant::now();
    while out.len() < o.max_tokens && *out.last().unwrap() != inkling_model::dims::EOS {
        out.push(run.step(*out.last().unwrap(), false)?);
    }
    let decode = t1.elapsed().as_secs_f64();
    println!("{}", text.decode(&out, false)?);
    eprintln!(
        "prompt {} tokens in {:.3} s; {} tokens decoded in {:.3} s = {:.1} tok/s (eager)",
        prompt.len(),
        prefill,
        out.len() - 1,
        decode,
        (out.len() - 1) as f64 / decode
    );
    Ok(())
}

/// The two `ingine-bench` prompts (third_party/ingine/native/apps/bench/main.cpp:34-35), same effort.
pub(crate) const BENCH_PROMPTS: [(&str, &str); 2] = [
    ("chat", "Explain how a hash map handles collisions, with a short Python example."),
    ("code", "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests."),
];

fn bench_cmd(o: &Opts) -> Result<(), String> {
    if o.spec.is_some() {
        return spec::bench(o);
    }
    let runs: usize = o.rest.first().and_then(|s| s.parse().ok()).unwrap_or(6);
    let eager_prefill = std::env::var_os("INKLING_EAGER_PREFILL").is_some();
    let text = TextModel::load(&o.model_dir)?;
    let effort = effort_value(o).unwrap_or("low".into());
    let eng = engine::Engine::open(&o.model_dir, &engine::cubin_path(o.cubin.as_ref()))?;
    let state = inkling_engine::State::new(o.max_ctx).map_err(|e| e.to_string())?;
    let packed = if o.nvfp4 { Some(inkling_engine::Nvfp4Weights::pack(&eng.module, &eng.weights, &eng.stream)?) } else { None };
    if let Some(q) = &packed {
        println!("nvfp4 mode: packed {:.2} GB of NVFP4 weights in {:.2} s", q.arena.total_bytes() as f64 / 1e9, q.pack_seconds);
    }
    let mut fwd = inkling_engine::Forward::with_mode(&eng.module, &eng.weights, packed.as_ref(), &state)?;
    if let (true, Ok(spec)) = (o.nvfp4, std::env::var("INKLING_NVFP4_PARTS")) {
        fwd.parts = inkling_engine::Nvfp4Parts::parse(&spec)?;
        println!("nvfp4 parts: {:?}", fwd.parts);
    }
    let mut run = engine::Runner { fwd, state: &state, stream: &eng.stream };
    let (free, total) = eng.gpu.mem_info().map_err(|e| e.to_string())?;
    println!(
        "{} driver {}: weights {:.2} s, state {:.2} GB, GPU {:.2} of {:.2} GiB in use, {} runs x {} tokens, effort {effort}, {:?} mode, PDL {}",
        eng.gpu.name,
        eng.gpu.driver_version,
        eng.load_seconds,
        state.total_bytes() as f64 / 1e9,
        (total - free) as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
        runs,
        o.max_tokens,
        run.fwd.mode(),
        run.fwd.pdl
    );
    // One captured step, replayed per token: the graph reads d_pos/d_tokens and advances them itself.
    run.reset()?;
    let t = std::time::Instant::now();
    let graph = inkling_runtime::capture::<String>(&eng.stream, |s| run.fwd.record_step(s))?;
    graph.upload(&eng.stream).map_err(|e| e.to_string())?;
    println!("captured {} nodes in {:.1} ms", graph.node_count(), t.elapsed().as_secs_f64() * 1e3);
    // Chunked prefill graphs (nvfp4 mode with every part on); INKLING_TOKEN_PREFILL=1 keeps per-token prefill.
    let mut batch_graphs: Vec<(i32, inkling_runtime::GraphExec)> = Vec::new();
    if run.fwd.parts == inkling_engine::Nvfp4Parts::ALL && std::env::var_os("INKLING_TOKEN_PREFILL").is_none() {
        for b in [8, 4, 2] {
            let g = inkling_runtime::capture::<String>(&eng.stream, |s| run.fwd.record_batch(s, b))?;
            g.upload(&eng.stream)?;
            batch_graphs.push((b, g));
        }
        println!("chunked prefill: B=8/4/2 graphs captured");
    }
    let wide = match &packed {
        Some(q) => match inkling_engine::WidePrefill::new(&eng.module, &eng.weights, q, &run.fwd, &state, &eng.stream, None) {
            Ok(w) => {
                println!("wide prefill: chunks {:?}", w.chunk_sizes());
                Some(w)
            }
            Err(e) => {
                println!("wide prefill off: {e}");
                None
            }
        },
        _ => None,
    };
    let next_tok = state.get(state.s.d_next_tokens);

    let mut summary = Vec::new();
    for (name, prompt_text) in BENCH_PROMPTS {
        let msgs = serde_json::json!([{"role": "user", "content": prompt_text}]);
        let prompt = text.encode_chat(&msgs, None, Some(&effort))?;
        run.ensure_room(prompt.len() + o.max_tokens)?;
        let mut rates = Vec::new();
        let mut ttfts = Vec::new();
        let mut first_text: Option<Vec<u32>> = None;
        for r in 0..runs + 1 {
            run.reset()?;
            let t0 = std::time::Instant::now();
            let next = if eager_prefill {
                let mut next = 0;
                for &tok in &prompt {
                    next = run.step(tok, false)?;
                }
                next
            } else if let Some(w) = &wide {
                w.prefill(&batch_graphs, &graph, &prompt)?
            } else if !batch_graphs.is_empty() {
                run.prefill_chunked(&batch_graphs, &graph, &prompt)?
            } else {
                run.prefill_graph(&graph, &prompt)?
            };
            let ttft = t0.elapsed().as_secs_f64();
            // d_tokens already holds `next` (inc_pos_and_copy_tok); decode by graph replay.
            let mut out = vec![next];
            let mut step_us: Vec<f64> = Vec::with_capacity(o.max_tokens);
            let wall = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            let t1 = std::time::Instant::now();
            while out.len() < o.max_tokens && *out.last().unwrap() != inkling_model::dims::EOS {
                let ts = std::time::Instant::now();
                graph.launch(&eng.stream).map_err(|e| e.to_string())?;
                eng.stream.synchronize().map_err(|e| e.to_string())?;
                out.push(next_tok.download::<i32>(1).map_err(|e| e.to_string())?[0] as u32);
                step_us.push(ts.elapsed().as_secs_f64() * 1e6);
            }
            let secs = t1.elapsed().as_secs_f64();
            let rate = (out.len() - 1) as f64 / secs;
            let mut st = step_us.clone();
            st.sort_by(f64::total_cmp);
            let q = |f: f64| st[((st.len() - 1) as f64 * f).round() as usize];
            // Per-token time over the run's halves shows whether a slow run is slow throughout.
            let half = step_us.len() / 2;
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
            println!(
                "    {name} run {r} @ {}.{:03}: TTFT {:.3} s, {:.1} tok/s, step us p50 {:.0} p95 {:.0} max {:.0}, mean first/second half {:.0}/{:.0}",
                wall.as_secs(), wall.subsec_millis(), ttft, rate, q(0.5), q(0.95), q(1.0),
                mean(&step_us[..half]), mean(&step_us[half..])
            );
            if r > 0 {
                rates.push(rate);
                ttfts.push(ttft);
            }
            match &first_text {
                None => first_text = Some(out.clone()),
                Some(f) if *f != out => println!("    run {r}: output differs from the first run (nondeterminism)"),
                _ => {}
            }
        }
        let mut sorted = rates.clone();
        sorted.sort_by(f64::total_cmp);
        let p50 = sorted[sorted.len() / 2];
        let shown = text.decode(first_text.as_ref().unwrap(), false)?;
        let mut tt = ttfts.clone();
        tt.sort_by(f64::total_cmp);
        println!(
            "{name}: decode best {:.1} p50 {:.1} min {:.1} tok/s; TTFT best {:.3} p50 {:.3} s; over {} runs (warm-up excluded); {:?}",
            sorted[sorted.len() - 1], p50, sorted[0], tt[0], tt[tt.len() / 2], runs,
            shown.chars().take(160).collect::<String>()
        );
        summary.push(serde_json::json!({"prompt": name, "runs": rates, "p50": p50, "ttft_s": ttfts, "tokens": first_text.unwrap().len()}));
    }
    println!("{}", serde_json::to_string(&summary).unwrap());
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        eprintln!("usage: inkling <load-check|tokenize|generate|bench|kld|serve> [options]");
        return ExitCode::from(2);
    };
    let result = parse(rest).and_then(|o| match cmd.as_str() {
        "load-check" => load_check(&o),
        "tokenize" => tokenize(&o),
        "generate" => generate_cmd(&o),
        "bench" => bench_cmd(&o),
        "kld" => kld::kld_cmd(&o),
        "serve" => server::serve(&o),
        other => Err(format!("unknown command {other}")),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("inkling: {e}");
            ExitCode::FAILURE
        }
    }
}
