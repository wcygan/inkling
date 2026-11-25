//! `inkling kld FILE [PREFIX] [EVAL]`: how far the shipped precisions move the model's next-token distribution.
//!
//! FILE's text becomes one user turn. Its first PREFIX tokens (default 7000) are prefilled and the next EVAL
//! tokens (default 512) are fed one at a time (teacher forcing) with the LM head writing full logits.
//!
//! * Reference: exact mode (BF16 weights, BF16 KV), every token through the B = 1 step.
//! * `fp8, token prefill`: the shipped mode (FP8 dense, NVFP4 experts, FP8 global KV) with the prefix through
//!   the B = 1 step, so only the decode kernels' precisions differ from the reference.
//! * `fp8, wide prefill`: the same with the prefix through the wide prefill, as the server runs it. The
//!   difference to the previous row is what the wide prefill's own arithmetic adds. The `INKLING_WIDE_*`
//!   variables select its kernels, so a new prefill kernel is judged by this row.
//!
//! * `wide vs token prefill`: the wide row measured against the token row instead of the exact reference. Both
//!   run the same precisions, so this is the size of a change in summation order alone.
//!
//! `INKLING_KLD_SELF=1` adds a self-check (exact mode through eager steps against the reference; all zeros)
//! and `INKLING_KLD_WORST=1` lists each row's five most divergent positions.
//!
//! Per row: KL(reference || test) in nats over the EVAL positions (mean, median, 95th percentile, maximum), how
//! often both pick the same top token, and the perplexity of the file's own next tokens. Needs about 25 GiB of
//! free VRAM (both weight sets are resident), so the server must be stopped.

use crate::engine::{self, Runner};
use crate::{Opts, effort_value};
use inkling_engine::{Forward, Nvfp4Parts, Nvfp4Weights, State, WidePrefill};
use inkling_runtime::capture;
use inkling_text::TextModel;

const FP8_PARTS: &str = "experts,head,router,ctrl,dense_fp8";

/// Feeds `tokens` one at a time from the current position and returns the logits after each.
fn eval_logits(run: &mut Runner, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
    let mut out = Vec::with_capacity(tokens.len());
    for &t in tokens {
        run.step(t, true)?;
        out.push(run.logits()?);
    }
    Ok(out)
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let mx = logits.iter().cloned().fold(f32::MIN, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - mx).exp()).sum();
    let lz = mx + sum.ln();
    logits.iter().map(|&x| x as f64 - lz).collect()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn report(name: &str, reference: &[Vec<f32>], got: &[Vec<f32>], targets: &[u32]) -> serde_json::Value {
    let mut klds = Vec::with_capacity(got.len());
    let (mut same, mut nll_ref, mut nll) = (0usize, 0f64, 0f64);
    for ((r, g), &target) in reference.iter().zip(got).zip(targets) {
        let (lr, lg) = (log_softmax(r), log_softmax(g));
        let kl: f64 = lr.iter().zip(&lg).map(|(&a, &b)| a.exp() * (a - b)).sum();
        klds.push(kl.max(0.0));
        same += (argmax(r) == argmax(g)) as usize;
        nll_ref -= lr[target as usize];
        nll -= lg[target as usize];
    }
    let n = klds.len() as f64;
    let mean = klds.iter().sum::<f64>() / n;
    if std::env::var_os("INKLING_KLD_WORST").is_some() {
        // The five positions with the largest divergence: index, KLD, each side's top token and its probability.
        let mut order: Vec<usize> = (0..klds.len()).collect();
        order.sort_by(|&x, &y| klds[y].total_cmp(&klds[x]));
        for &i in order.iter().take(5) {
            let (lr, lg) = (log_softmax(&reference[i]), log_softmax(&got[i]));
            let (ar, ag) = (argmax(&reference[i]), argmax(&got[i]));
            println!("    position {i}: KLD {:.4}; reference top {ar} p {:.3} (test gives it {:.3}); test top {ag} p {:.3}; target {}",
                klds[i], lr[ar].exp(), lg[ar].exp(), lg[ag].exp(), targets[i]);
        }
    }
    let mut sorted = klds.clone();
    sorted.sort_by(f64::total_cmp);
    let q = |f: f64| sorted[((sorted.len() - 1) as f64 * f).round() as usize];
    let (ppl_ref, ppl) = ((nll_ref / n).exp(), (nll / n).exp());
    println!(
        "{name:<22} KLD mean {mean:.6} median {:.6} p95 {:.6} max {:.4}; top-1 same {:.2}%; PPL {ppl:.4} (reference {ppl_ref:.4}, {:+.4})",
        q(0.5),
        q(0.95),
        q(1.0),
        100.0 * same as f64 / n,
        ppl - ppl_ref
    );
    serde_json::json!({"config": name, "kld_mean": mean, "kld_median": q(0.5), "kld_p95": q(0.95), "kld_max": q(1.0),
        "top1_same": same as f64 / n, "ppl": ppl, "ppl_reference": ppl_ref, "positions": klds.len()})
}

pub fn kld_cmd(o: &Opts) -> Result<(), String> {
    let file = o.rest.first().ok_or("usage: inkling kld FILE [PREFIX_TOKENS] [EVAL_TOKENS]")?;
    let num = |i: usize, default: usize| o.rest.get(i).map_or(Ok(default), |s| s.parse::<usize>().map_err(|_| format!("bad number {s}")));
    let (prefix, n_eval) = (num(1, 7000)?, num(2, 512)?);
    if prefix < 2 || n_eval < 1 {
        return Err("PREFIX must be at least 2 and EVAL at least 1".into());
    }
    let text = TextModel::load(&o.model_dir)?;
    let body = std::fs::read_to_string(file).map_err(|e| format!("{file}: {e}"))?;
    let msgs = serde_json::json!([{"role": "user", "content": body}]);
    let effort = effort_value(o).unwrap_or("low".into());
    let all = text.encode_chat(&msgs, None, Some(&effort))?;
    let need = prefix + n_eval + 1;
    if all.len() < need {
        return Err(format!("{file} gives {} tokens; PREFIX + EVAL + 1 = {need} are needed", all.len()));
    }
    let tokens = &all[..need];
    let (fed, targets) = (&tokens[prefix..prefix + n_eval], &tokens[prefix + 1..]);
    let ctx = (need + 64).max(4096);
    let eng = engine::Engine::open(&o.model_dir, &engine::cubin_path(o.cubin.as_ref()))?;
    let stream = &eng.stream;
    println!("{file}: {prefix} prefix tokens, {n_eval} evaluated positions, weights loaded in {:.1} s", eng.load_seconds);

    // Reference: exact mode, BF16 KV. SAFETY (both blocks): single-threaded, before any state is built.
    unsafe {
        std::env::remove_var("INKLING_KV_FP8");
        std::env::remove_var("INKLING_NVFP4_PARTS");
    }
    let t = std::time::Instant::now();
    let reference = {
        let state = State::new(ctx).map_err(|e| e.to_string())?;
        let fwd = Forward::new(&eng.module, &eng.weights, &state)?;
        let mut run = Runner { fwd, state: &state, stream };
        run.reset()?;
        let g1 = capture::<String>(stream, |s| run.fwd.record_step(s))?;
        g1.upload(stream).map_err(|e| e.to_string())?;
        run.prefill_graph(&g1, &tokens[..prefix])?;
        eval_logits(&mut run, fed)?
    };
    println!("reference (exact, BF16 KV) done in {:.1} s", t.elapsed().as_secs_f64());
    let mut rows = Vec::new();
    if std::env::var_os("INKLING_KLD_SELF").is_some() {
        // Self-check: exact mode again with every token through eager steps (no graph). Must be all zeros.
        let state = State::new(ctx).map_err(|e| e.to_string())?;
        let fwd = Forward::new(&eng.module, &eng.weights, &state)?;
        let mut run = Runner { fwd, state: &state, stream };
        run.reset()?;
        for &tok in &tokens[..prefix] {
            run.step(tok, false)?;
        }
        let got = eval_logits(&mut run, fed)?;
        rows.push(report("exact, eager steps", &reference, &got, targets));
    }

    unsafe {
        std::env::set_var("INKLING_KV_FP8", "1");
        std::env::set_var("INKLING_NVFP4_PARTS", FP8_PARTS);
    }
    let packed = Nvfp4Weights::pack(&eng.module, &eng.weights, stream)?;
    let mut token_logits: Option<Vec<Vec<f32>>> = None;
    for wide in [false, true] {
        let t = std::time::Instant::now();
        let state = State::new(ctx).map_err(|e| e.to_string())?;
        let mut fwd = Forward::with_mode(&eng.module, &eng.weights, Some(&packed), &state)?;
        fwd.parts = Nvfp4Parts::parse(FP8_PARTS)?;
        let mut run = Runner { fwd, state: &state, stream };
        run.reset()?;
        let g1 = capture::<String>(stream, |s| run.fwd.record_step(s))?;
        g1.upload(stream).map_err(|e| e.to_string())?;
        if wide {
            let mut batch = Vec::new();
            for b in [8, 4, 2] {
                let g = capture::<String>(stream, |s| run.fwd.record_batch(s, b))?;
                g.upload(stream).map_err(|e| e.to_string())?;
                batch.push((b, g));
            }
            let w = WidePrefill::new(&eng.module, &eng.weights, &packed, &run.fwd, &state, stream, None)?;
            w.prefill(&batch, &g1, &tokens[..prefix])?;
        } else {
            run.prefill_graph(&g1, &tokens[..prefix])?;
        }
        let prefill_s = t.elapsed().as_secs_f64();
        let got = eval_logits(&mut run, fed)?;
        let name = if wide { "fp8, wide prefill" } else { "fp8, token prefill" };
        let mut row = report(name, &reference, &got, targets);
        row["prefix"] = prefix.into();
        row["prefill_s"] = prefill_s.into();
        rows.push(row);
        match &token_logits {
            // Same precisions, different order of sums: what any re-ordered prefill kernel moves the output by.
            Some(tok) => rows.push(report("wide vs token prefill", tok, &got, targets)),
            None => token_logits = Some(got),
        }
    }
    println!("{}", serde_json::to_string(&rows).unwrap());
    Ok(())
}
