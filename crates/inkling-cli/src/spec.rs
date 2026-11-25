//! `inkling bench --mode nvfp4 --spec [--spec-k K]`: DSpark speculative decoding with a fixed verify
//! width K + 1 (K = 3 by default), greedy acceptance on the device (`spec_verify`), and the G4 check
//! against plain nvfp4 greedy decode.
//!
//! Per verify round: the draft graph (7 drafts after the anchor) then the verify graph (assemble
//! `[anchor, drafts[0..K)]`, the target forward at width K + 1 with the drafter's commit, `spec_verify`),
//! then one async copy of the accepted tokens and a host sync. No rollback: positions past the accepted
//! prefix are overwritten by the next round before anything reads them.

use crate::engine::{self, Runner};
use crate::lookup::PromptLookupIndex;
use crate::{BENCH_PROMPTS, Opts, effort_value};
use inkling_engine::{DsBuf, Dspark, DsparkWeights, Nvfp4Parts};
use inkling_model::dims;
use inkling_model::loader::LoadOptions;
use inkling_runtime::{GraphExec, PinnedBuffer, Stream, capture};
use inkling_text::TextModel;

/// Reads a draft-vocabulary shortlist (little-endian i32 ids) and checks it: in range, unique, sorted,
/// and containing EOS.
pub fn read_shortlist(path: &std::path::Path) -> Result<Vec<i32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 || bytes.is_empty() {
        return Err(format!("{}: not a list of i32 ids", path.display()));
    }
    let ids: Vec<i32> = bytes.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
    if ids.iter().any(|&t| t < 0 || t as usize >= dims::VOCAB) {
        return Err(format!("{}: id out of range", path.display()));
    }
    if ids.windows(2).any(|w| w[0] >= w[1]) {
        return Err(format!("{}: ids must be sorted and unique", path.display()));
    }
    if ids.binary_search(&(dims::EOS as i32)).is_err() {
        return Err(format!("{}: EOS {} missing", path.display(), dims::EOS));
    }
    Ok(ids)
}

struct SpecGraphs<'w> {
    /// Wide prefill (16-256-token chunks) with the drafter's taps and commit, if available.
    wide: Option<inkling_engine::WidePrefill<'w>>,
    prefill: Vec<(i32, GraphExec)>,
    prefill1: GraphExec,
    draft: GraphExec,
    /// Assemble + target forward + commit + spec_verify, ending with async copies of the accepted
    /// tokens, their count and (adaptive) the chosen width into the pinned buffers.
    verify: GraphExec,
    /// Prompt lookup: assemble [anchor, 7 lookup drafts] + B = 8 forward + commit + spec_verify + copies.
    verify8: Option<GraphExec>,
}

/// A copy/RAG-style prompt (the model is asked to repeat a ~300-token code block with one rename), where
/// prompt lookup should find long matches.
pub(crate) const COPY_PROMPT: (&str, &str) = (
    "copy",
    "Repeat the following Python module back exactly, except rename the class `LRUCache` to `BoundedCache` everywhere. Output only the code.\n\n```python\nfrom collections import OrderedDict\nfrom typing import Generic, Hashable, Optional, TypeVar\n\nK = TypeVar(\"K\", bound=Hashable)\nV = TypeVar(\"V\")\n\n\nclass LRUCache(Generic[K, V]):\n    \"\"\"A least-recently-used cache with O(1) get and put.\"\"\"\n\n    def __init__(self, capacity: int) -> None:\n        if capacity <= 0:\n            raise ValueError(\"capacity must be positive\")\n        self.capacity = capacity\n        self._data: OrderedDict[K, V] = OrderedDict()\n        self.hits = 0\n        self.misses = 0\n\n    def get(self, key: K) -> Optional[V]:\n        if key not in self._data:\n            self.misses += 1\n            return None\n        self.hits += 1\n        self._data.move_to_end(key)\n        return self._data[key]\n\n    def put(self, key: K, value: V) -> None:\n        if key in self._data:\n            self._data.move_to_end(key)\n        self._data[key] = value\n        if len(self._data) > self.capacity:\n            self._data.popitem(last=False)\n\n    def __len__(self) -> int:\n        return len(self._data)\n\n    def __contains__(self, key: object) -> bool:\n        return key in self._data\n\n    def clear(self) -> None:\n        self._data.clear()\n        self.hits = 0\n        self.misses = 0\n\n\ndef demo() -> None:\n    cache: LRUCache[str, int] = LRUCache(2)\n    cache.put(\"a\", 1)\n    cache.put(\"b\", 2)\n    assert cache.get(\"a\") == 1\n    cache.put(\"c\", 3)\n    assert cache.get(\"b\") is None\n    print(len(cache), cache.hits, cache.misses)\n```",
);

/// Pinned host buffers the verify graph copies its results into.
struct Results {
    accepted: PinnedBuffer,
    n: PinnedBuffer,
    width: PinnedBuffer,
}

impl Results {
    fn record_copies(&mut self, ds: &Dspark, s: &Stream, adaptive: bool) -> Result<(), String> {
        self.accepted.copy_from_device_async(&ds.slice(DsBuf::Accepted).sub(0, 32), s)?;
        self.n.copy_from_device_async(&ds.slice(DsBuf::NumAccepted).sub(0, 4), s)?;
        if adaptive {
            self.width.copy_from_device_async(&ds.slice(DsBuf::Width).sub(0, 4), s)?;
        }
        Ok(())
    }

    fn i32_at(b: &PinnedBuffer, i: usize) -> i32 {
        i32::from_ne_bytes(b.as_slice()[4 * i..4 * i + 4].try_into().unwrap())
    }
}

struct RunOut {
    tokens: Vec<u32>,
    ttft: f64,
    decode_s: f64,
    rounds: usize,
    accepted_hist: [usize; 9],
    width_hist: [usize; 9],
    /// Rounds whose drafts came from prompt lookup.
    lookup_hits: usize,
    /// GPU milliseconds in the draft and verify graphs (INKLING_SPEC_PROFILE=1), summed over rounds.
    draft_ms: f64,
    verify_ms: f64,
}

#[allow(clippy::too_many_arguments)]
fn spec_run(
    run: &Runner,
    ds: &Dspark,
    g: &SpecGraphs<'_>,
    stream: &Stream,
    prompt: &[u32],
    max_tokens: usize,
    res: &Results,
    fixed_width: Option<usize>,
    done: &inkling_runtime::Event,
    mut lookup: Option<(&mut PromptLookupIndex, &mut PinnedBuffer)>,
) -> Result<RunOut, String> {
    run.reset()?;
    ds.reset(stream)?;
    let t0 = std::time::Instant::now();
    let next = match &g.wide {
        Some(w) => w.prefill(&g.prefill, &g.prefill1, prompt)?,
        None => run.prefill_chunked(&g.prefill, &g.prefill1, prompt)?,
    };
    // Anchor = the token after the prompt, at position prompt.len() (already in d_pos).
    run.state.get(run.state.s.d_next_tokens).sub(0, 4).copy_to(&ds.slice(DsBuf::Anchor), stream)?;
    let ttft = t0.elapsed().as_secs_f64();
    let mut out = vec![next];
    if let Some((ix, _)) = lookup.as_mut() {
        ix.reset();
        for &t in prompt.iter().chain(std::iter::once(&next)) {
            ix.push(t as i32);
        }
    }
    let mut lookup_hits = 0usize;
    let d_drafts = ds.slice(DsBuf::Drafts).sub(0, 4 * crate::lookup::DRAFTS);
    let mut rounds = 0usize;
    let mut hist = [0usize; 9];
    let mut whist = [0usize; 9];
    let profile = std::env::var_os("INKLING_SPEC_PROFILE").is_some();
    let ev: Vec<inkling_runtime::Event> = if profile { (0..3).map(|_| inkling_runtime::Event::new()).collect::<Result<_, _>>()? } else { Vec::new() };
    // Overlap: launch the next round's draft before reading this round's result (squid5090's
    // draft_in_flight). The profile run keeps the rounds serial so the event pairs bracket one graph each.
    let overlap = !profile && std::env::var_os("INKLING_SPEC_NO_OVERLAP").is_none();
    let (mut draft_ms, mut verify_ms) = (0f64, 0f64);
    let mut draft_in_flight = false;
    let t1 = std::time::Instant::now();
    while out.len() < max_tokens && *out.last().unwrap() != dims::EOS {
        // Prompt lookup first (squid5090): on a hit the 7 drafts go straight to the B = 8 verify and any
        // in-flight DSpark draft is discarded (its d_drafts are overwritten by the copy, in stream order).
        let hit = match (lookup.as_mut(), g.verify8.as_ref()) {
            (Some((ix, pin)), Some(v8)) => match ix.find() {
                Some(drafts) => {
                    for (j, t) in drafts.iter().enumerate() {
                        pin.as_mut_slice()[4 * j..4 * j + 4].copy_from_slice(&t.to_ne_bytes());
                    }
                    pin.copy_to_device_async(d_drafts.bytes, &d_drafts, stream)?;
                    v8.launch(stream)?;
                    stream.synchronize()?;
                    draft_in_flight = false;
                    lookup_hits += 1;
                    true
                }
                None => false,
            },
            _ => false,
        };
        if hit {
            // Results read below, width 8.
        } else {
        if profile {
            ev[0].record(stream)?;
        }
        if !draft_in_flight {
            g.draft.launch(stream)?;
        }
        if profile {
            ev[1].record(stream)?;
        }
        g.verify.launch(stream)?;
        if profile {
            ev[2].record(stream)?;
        }
        if overlap {
            done.record(stream)?;
            g.draft.launch(stream)?;
            draft_in_flight = true;
            done.synchronize()?;
        } else {
            stream.synchronize()?;
        }
        }
        let n = Results::i32_at(&res.n, 0) as usize;
        let w = if hit { 8 } else { fixed_width.unwrap_or_else(|| Results::i32_at(&res.width, 0) as usize) };
        if n == 0 || n > w || w > 8 {
            return Err(format!("spec_verify accepted {n} tokens at width {w}"));
        }
        rounds += 1;
        hist[n] += 1;
        whist[w] += 1;
        if profile && !hit {
            draft_ms += ev[1].elapsed_ms_since(&ev[0])? as f64;
            verify_ms += ev[2].elapsed_ms_since(&ev[1])? as f64;
        }
        for j in 0..n {
            let t = Results::i32_at(&res.accepted, j) as u32;
            out.push(t);
            if let Some((ix, _)) = lookup.as_mut() {
                ix.push(t as i32);
            }
            if t == dims::EOS || out.len() >= max_tokens {
                break;
            }
        }
    }
    let decode_s = t1.elapsed().as_secs_f64();
    stream.synchronize()?; // a trailing draft may still be running
    out.truncate(max_tokens);
    Ok(RunOut { tokens: out, ttft, decode_s, rounds, accepted_hist: hist, width_hist: whist, lookup_hits, draft_ms, verify_ms })
}

pub fn bench(o: &Opts) -> Result<(), String> {
    let adaptive = o.spec_threshold;
    let k = if adaptive.is_some() { inkling_engine::NUM_DRAFTS } else { o.spec.unwrap() };
    if !o.nvfp4 {
        return Err("--spec needs --mode nvfp4".into());
    }
    let width = (k + 1) as i32;
    if ![2, 4, 8].contains(&width) {
        return Err(format!("--spec-k {k}: verify width {width} must be 2, 4 or 8"));
    }
    let runs: usize = o.rest.first().and_then(|s| s.parse().ok()).unwrap_or(6);
    let text = TextModel::load(&o.model_dir)?;
    let effort = effort_value(o).unwrap_or("low".into());
    let eng = engine::Engine::open(&o.model_dir, &engine::cubin_path(o.cubin.as_ref()))?;
    let stream = &eng.stream;
    let state = inkling_engine::State::new(o.max_ctx).map_err(|e| e.to_string())?;
    let packed = inkling_engine::Nvfp4Weights::pack(&eng.module, &eng.weights, stream)?;
    let t = std::time::Instant::now();
    let mtp = inkling_model::weights::load_mtp(&o.model_dir, stream, LoadOptions::default())?;
    let mtp_load = t.elapsed().as_secs_f64();
    let shortlist = o.draft_vocab.as_deref().map(read_shortlist).transpose()?;
    let dw = DsparkWeights::prepare(&eng.module, &eng.weights, &packed, mtp, shortlist.as_deref(), stream)?;
    let mut ds = Dspark::new(&eng.module, &eng.weights, &dw, &state)?;
    ds.commit_on_aux = std::env::var_os("INKLING_SPEC_SERIAL_COMMIT").is_none();
    let ds = ds;
    let fwd = inkling_engine::Forward::with_mode(&eng.module, &eng.weights, Some(&packed), &state)?;
    let mut run = Runner { fwd, state: &state, stream };
    if let Ok(spec) = std::env::var("INKLING_NVFP4_PARTS") {
        run.fwd.parts = Nvfp4Parts::parse(&spec)?;
        println!("nvfp4 parts: {:?}", run.fwd.parts);
    }
    if ![Nvfp4Parts::ALL, Nvfp4Parts::BATCH_EXACT_DENSE, Nvfp4Parts::BATCH_FP8_DENSE].contains(&run.fwd.parts) {
        return Err("--spec needs every nvfp4 part, every part but dense, or FP8 dense".into());
    }
    let (free, total) = eng.gpu.mem_info().map_err(|e| e.to_string())?;
    println!(
        "{} driver {}: weights {:.2} s, mtp {:.2} s, drafter prepared in {:.2} s (draft vocab {}), state {:.2} GB + drafter {:.2} GB, GPU {:.2} of {:.2} GiB in use; {}, {runs} runs x {} tokens, effort {effort}",
        eng.gpu.name,
        eng.gpu.driver_version,
        eng.load_seconds,
        mtp_load,
        dw.prepare_seconds,
        dw.vocab,
        state.total_bytes() as f64 / 1e9,
        (ds.total_bytes() + dw.arena.total_bytes() + dw.mtp.arena.total_bytes()) as f64 / 1e9,
        (total - free) as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64,
        match adaptive {
            Some(t) => format!("adaptive widths 2/4/8 at threshold {t}"),
            None => format!("K = {k} (verify width {width})"),
        },
        o.max_tokens
    );

    // Plain graphs (reference decode), then the same with the drafter's taps + commit.
    run.reset()?;
    let plain1 = capture::<String>(stream, |s| run.fwd.record_step(s))?;
    let mut plain_prefill = Vec::new();
    for b in [8, 4, 2] {
        plain_prefill.push((b, capture::<String>(stream, |s| run.fwd.record_batch(s, b))?));
    }
    run.fwd.dspark = Some(&ds);
    let prefill1 = capture::<String>(stream, |s| run.fwd.record_step(s))?;
    let mut prefill = Vec::new();
    for b in [8, 4, 2] {
        prefill.push((b, capture::<String>(stream, |s| run.fwd.record_batch(s, b))?));
    }
    let draft = capture::<String>(stream, |s| ds.record_draft(s))?;
    let mut res = Results { accepted: PinnedBuffer::alloc(32)?, n: PinnedBuffer::alloc(4)?, width: PinnedBuffer::alloc(4)? };
    let verify = match adaptive {
        None => capture::<String>(stream, |s| {
            ds.record_assemble(s, k as i32)?;
            run.fwd.record_batch(s, width)?;
            ds.record_spec_verify(s, width)?;
            res.record_copies(&ds, s, false)
        })?,
        Some(threshold) => {
            // The adaptive switch: z_adaptive_select_width (CUDA C++ cubin) picks width 2, 4 or 8;
            // each switch case runs the target forward + commit at that width and spec_verify.
            let path = engine::cubin_path(o.cubin.as_ref()).with_file_name("z_adaptive_select_width.cubin");
            let image = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let m = inkling_runtime::Module::from_cubin(&image, path.display().to_string())?;
            let select = m.function("z_adaptive_select_width").ok_or("z_adaptive_select_width missing")?;
            let fwd = &run.fwd;
            let g = inkling_runtime::capture_switch::<String>(
                stream,
                3,
                |s, handle| ds.record_select_width(s, select, threshold, handle),
                |s| res.record_copies(&ds, s, true),
                |i, s| {
                    let w = [2, 4, 8][i as usize];
                    fwd.record_batch(s, w)?;
                    ds.record_spec_verify(s, w)
                },
            )?;
            std::mem::forget(m); // the graph references the module's function for the process lifetime
            g
        }
    };
    let verify8 = if o.lookup {
        Some(capture::<String>(stream, |s| {
            ds.record_assemble(s, crate::lookup::DRAFTS as i32)?;
            run.fwd.record_batch(s, 8)?;
            ds.record_spec_verify(s, 8)?;
            res.record_copies(&ds, s, false)
        })?)
    } else {
        None
    };
    run.fwd.dspark = None;
    if let Some(v8) = &verify8 {
        v8.upload(stream)?;
    }
    for gr in [&plain1, &prefill1, &draft, &verify] {
        gr.upload(stream)?;
    }
    println!(
        "graphs: plain step {} nodes, draft {} nodes, verify (B={width}) {} nodes",
        plain1.node_count(),
        draft.node_count(),
        verify.node_count()
    );
    // Wide prefill for both the plain reference and the speculative runs, so G4 compares like with like.
    let (wide_plain, wide) = if std::env::var_os("INKLING_WIDE_PREFILL").is_none_or(|v| v != "0") {
        let fwd = &run.fwd;
        let mut plain_fwd = inkling_engine::WidePrefill::new(&eng.module, &eng.weights, &packed, fwd, &state, stream, None);
        let spec_fwd = inkling_engine::WidePrefill::new(&eng.module, &eng.weights, &packed, fwd, &state, stream, Some(&ds));
        if let Err(e) = &plain_fwd {
            println!("wide prefill off: {e}");
        }
        if spec_fwd.is_err() {
            plain_fwd = Err("no spec wide prefill".into());
        }
        (plain_fwd.ok(), spec_fwd.ok())
    } else {
        (None, None)
    };
    println!("wide prefill: {}", wide.as_ref().map_or("off".to_string(), |w| format!("on (chunks {:?})", w.chunk_sizes())));
    let g = SpecGraphs { wide, prefill, prefill1, draft, verify, verify8 };
    let mut lookup_ix = PromptLookupIndex::default();
    let mut lookup_pin = PinnedBuffer::alloc(4 * crate::lookup::DRAFTS)?;
    println!("prompt lookup: {}", if o.lookup { "on (min match 16, 7 drafts, B = 8 verify)" } else { "off" });
    let done = inkling_runtime::Event::new_untimed()?;
    let next_tok = state.get(state.s.d_next_tokens);

    let mut summary = Vec::new();
    for (name, prompt_text) in BENCH_PROMPTS.into_iter().chain(std::iter::once(COPY_PROMPT)) {
        let msgs = serde_json::json!([{"role": "user", "content": prompt_text}]);
        let prompt = text.encode_chat(&msgs, None, Some(&effort))?;
        // Verify steps write up to 8 positions past the last accepted token.
        run.ensure_room(prompt.len() + o.max_tokens + 8)?;

        // Plain greedy reference (graph replay, host sync per token).
        run.reset()?;
        let t0 = std::time::Instant::now();
        let first = match &wide_plain {
            Some(w) => w.prefill(&plain_prefill, &plain1, &prompt)?,
            None => run.prefill_chunked(&plain_prefill, &plain1, &prompt)?,
        };
        let _ = t0;
        let mut plain = vec![first];
        let tp = std::time::Instant::now();
        while plain.len() < o.max_tokens && *plain.last().unwrap() != dims::EOS {
            plain1.launch(stream)?;
            stream.synchronize()?;
            plain.push(next_tok.download::<i32>(1)?[0] as u32);
        }
        let plain_rate = (plain.len() - 1) as f64 / tp.elapsed().as_secs_f64();

        let mut rates = Vec::new();
        let mut per_tok = Vec::new();
        let mut tpv = Vec::new();
        let mut ttfts = Vec::new();
        let mut first_out: Option<RunOut> = None;
        for r in 0..runs + 1 {
            let ro = spec_run(
                &run,
                &ds,
                &g,
                stream,
                &prompt,
                o.max_tokens,
                &res,
                adaptive.is_none().then_some(width as usize),
                &done,
                o.lookup.then_some((&mut lookup_ix, &mut lookup_pin)),
            )?;
            let ngen = ro.tokens.len() - 1;
            let rate = ngen as f64 / ro.decode_s;
            println!(
                "    {name} run {r}: TTFT {:.3} s, {ngen} tokens in {:.3} s = {:.1} tok/s, {} rounds, {:.2} tokens/verify, accepted-count histogram {:?}",
                ro.ttft,
                ro.decode_s,
                rate,
                ro.rounds,
                ngen as f64 / ro.rounds.max(1) as f64,
                &ro.accepted_hist[1..=width as usize]
            );
            if adaptive.is_some() || o.lookup {
                println!(
                    "        verify widths 2/4/8: {} / {} / {}; prompt-lookup rounds {}",
                    ro.width_hist[2], ro.width_hist[4], ro.width_hist[8], ro.lookup_hits
                );
            }
            if ro.draft_ms > 0.0 {
                println!(
                    "        GPU per round: draft {:.0} us, verify {:.0} us; host+sync overhead {:.0} us",
                    ro.draft_ms * 1e3 / ro.rounds as f64,
                    ro.verify_ms * 1e3 / ro.rounds as f64,
                    (ro.decode_s * 1e3 - ro.draft_ms - ro.verify_ms) * 1e3 / ro.rounds as f64
                );
            }
            if r > 0 {
                rates.push(rate);
                per_tok.push(ro.decode_s * 1e6 / ngen as f64);
                tpv.push(ngen as f64 / ro.rounds.max(1) as f64);
                ttfts.push(ro.ttft);
            }
            match &first_out {
                None => first_out = Some(ro),
                Some(f) if f.tokens != ro.tokens => println!("    run {r}: speculative output differs from run 0 (nondeterminism)"),
                _ => {}
            }
        }
        let spec_tokens = first_out.unwrap().tokens;

        // G4: token-for-token against plain greedy; at the first divergence, the plain path's top-2 margin.
        let n = plain.len().min(spec_tokens.len());
        let div = (0..n).find(|&i| plain[i] != spec_tokens[i]);
        let g4 = match div {
            None if plain.len() == spec_tokens.len() => {
                println!("    G4 {name}: speculative == plain greedy for all {} tokens", plain.len());
                serde_json::json!({"identical": true, "tokens": plain.len()})
            }
            None => {
                println!("    G4 {name}: identical over {n} tokens; lengths differ ({} plain, {} spec)", plain.len(), spec_tokens.len());
                serde_json::json!({"identical": true, "tokens": n, "len_plain": plain.len(), "len_spec": spec_tokens.len()})
            }
            Some(i) => {
                // Teacher-force the plain path eagerly up to the divergence and read its logits there.
                run.reset()?;
                let mut feed: Vec<u32> = prompt.clone();
                feed.extend_from_slice(&plain[..i]);
                for (j, &tok) in feed.iter().enumerate() {
                    run.step(tok, j + 1 == feed.len())?;
                }
                let logits = run.logits()?;
                let (lp, ls) = (logits[plain[i] as usize], logits[spec_tokens[i] as usize]);
                let mut sorted: Vec<f32> = logits.clone();
                sorted.sort_by(|a, b| b.total_cmp(a));
                let margin = sorted[0] - sorted[1];
                println!(
                    "    G4 {name}: first divergence at generated token {i}: plain {} (logit {lp:.4}) vs spec {} (logit {ls:.4}); plain top-1/top-2 margin {margin:.4}, plain-spec logit gap {:.4}",
                    plain[i],
                    spec_tokens[i],
                    lp - ls
                );
                serde_json::json!({"identical": false, "first_divergence": i, "plain_token": plain[i], "spec_token": spec_tokens[i], "top2_margin": margin, "logit_gap": lp - ls})
            }
        };

        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (p50, us50, tpv50, ttft50) = (med(&mut rates.clone()), med(&mut per_tok.clone()), med(&mut tpv.clone()), med(&mut ttfts.clone()));
        let shown = text.decode(&spec_tokens, false)?;
        println!(
            "{name}: spec decode p50 {p50:.1} tok/s ({us50:.0} us per token), best {:.1}; {tpv50:.2} tokens/verify; TTFT p50 {ttft50:.3} s; plain greedy {plain_rate:.1} tok/s (one run, host sync per token); {:?}",
            rates.iter().cloned().fold(0.0, f64::max),
            shown.chars().take(120).collect::<String>()
        );
        summary.push(serde_json::json!({
            "prompt": name, "k": k, "adaptive_threshold": adaptive, "runs": rates, "p50": p50, "us_per_token_p50": us50,
            "tokens_per_verify": tpv, "ttft_s": ttfts, "tokens": spec_tokens.len(), "plain_rate": plain_rate, "g4": g4
        }));
    }
    println!("{}", serde_json::to_string(&summary).unwrap());
    Ok(())
}
