//! `inkling serve`: an OpenAI-compatible chat server over the shippable mode (fp8 + DSpark adaptive
//! speculation), for coding agents such as Pi.
//!
//!   inkling serve --mode fp8 --draft-vocab assets\draft_vocab_32k.bin [--max-ctx 32768] [--effort low]
//!   Environment: INKLING_PORT (8013), INKLING_SERVE_LOG (per-request JSONL), INKLING_SERVE_LOG_BODIES=1
//!   (adds prompt text around cache divergences to the log), INKLING_MAX_REQUEST_S (600).
//!
//! Endpoints:
//! * `POST /v1/chat/completions` handles streaming and non-streaming requests with `tools`, `tool_choice`
//!   (auto or none), `reasoning_effort`, `enable_thinking`, `max_tokens` and `stop`.
//! * `GET /health` returns 503 while loading. `GET /v1/models` and `GET /metrics` (Prometheus) are also
//!   served, as is `POST /apply-template` (the rendered prompt, for debugging cache misses).
//!
//! Behaviour:
//! * **Decoding** is greedy speculative decoding, identical to plain greedy. Sampling fields are ignored,
//!   and a loop guard stops degenerate repetition (`finish_reason: "length"`).
//! * **Threads:** a control thread answers health, models and metrics at once. Chat requests queue for the
//!   engine thread: one runs and up to 4 wait, after which the server returns 503 with `Retry-After`.
//! * **Turn reuse:**
//!   * The live cache is extended when the new prompt continues it (the usual agent turn).
//!   * Otherwise the longest matching checkpoint is restored. A checkpoint snapshots the rolling state
//!     (short-conv rings and local-window KV) at the end of every prompt, and at the end of the
//!     system/tools header. Full-attention KV is indexed by position, so it stays valid below a
//!     checkpoint while the live tokens still match.
//!   * Anything else prefills from scratch.
//! * **Reasoning** streams as `reasoning_content`. Assistant history may carry it as `reasoning_content`,
//!   `reasoning` or `reasoning_text`. When a client drops it, a server-side memory restores it so the
//!   re-rendered history matches what the model generated.
//! * **Tool calls** stream as they decode: the name first, then `arguments` deltas. A call cut off by
//!   `max_tokens` is dropped (`finish_reason: "length"`). A call whose JSON does not parse is returned as
//!   text.

use crate::engine::{self, Runner};
use crate::spec::read_shortlist;
use crate::{Opts, effort_value};
use inkling_engine::{DsBuf, Dspark, DsparkWeights, Nvfp4Parts};
use inkling_model::dims;
use inkling_model::loader::LoadOptions;
use inkling_runtime::{DevSlice, DeviceBuffer, GraphExec, PinnedBuffer, Stream, capture};
use inkling_text::TextModel;
use serde_json::{Value as Json, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

const MODEL_ID: &str = "inkling-2-mini";
const MSG_MODEL: &str = "<|message_model|>";
const MSG_SYSTEM: &str = "<|message_system|>";
const THINK: &str = "<|content_thinking|>";
const TEXT: &str = "<|content_text|>";
const TOOL_JSON: &str = "<|content_invoke_tool_json|>";
const TOOL_TEXT: &str = "<|content_invoke_tool_text|>";
const END_MSG: &str = "<|end_message|>";
const END_SAMPLING: &str = "<|content_model_end_sampling|>";
const QUEUE_WAITING: usize = 4;
/// Checkpoints kept (each ~239 MB of VRAM); INKLING_CHECKPOINTS overrides, 0 disables them.
fn max_checkpoints() -> usize {
    std::env::var("INKLING_CHECKPOINTS").ok().and_then(|v| v.parse().ok()).unwrap_or(6)
}
const PING_EVERY: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------------------------------
// Output parsing

#[derive(Debug, Clone)]
enum Seg {
    Reasoning(String),
    Content(String),
    Tool { name: String, body: String, json: bool, complete: bool },
}

/// Splits decoded output (special tokens kept) into reasoning, content and tool-call segments.
fn parse_output(s: &str) -> Vec<Seg> {
    let s = s.replace(END_SAMPLING, "");
    let mut segs = Vec::new();
    let mut parts = s.split(MSG_MODEL);
    if let Some(head) = parts.next() {
        let head = head.replace(END_MSG, "");
        if !head.trim().is_empty() {
            segs.push(Seg::Content(head));
        }
    }
    for p in parts {
        let (body, complete) = match p.find(END_MSG) {
            Some(i) => (&p[..i], true),
            None => (p, false),
        };
        if let Some(t) = body.strip_prefix(THINK) {
            segs.push(Seg::Reasoning(t.to_owned()));
        } else if let Some(t) = body.strip_prefix(TEXT) {
            segs.push(Seg::Content(t.to_owned()));
        } else if let Some(i) = body.find(TOOL_JSON) {
            segs.push(Seg::Tool { name: body[..i].to_owned(), body: body[i + TOOL_JSON.len()..].to_owned(), json: true, complete });
        } else if let Some(i) = body.find(TOOL_TEXT) {
            segs.push(Seg::Tool { name: body[..i].to_owned(), body: body[i + TOOL_TEXT.len()..].to_owned(), json: false, complete });
        } else if complete {
            segs.push(Seg::Content(body.to_owned()));
        }
    }
    segs
}

/// The arguments of a complete tool call, or None when its JSON does not parse.
fn tool_args(body: &str, json_form: bool) -> Option<Json> {
    if !json_form {
        return Some(json!({ "input": body }));
    }
    let v: Json = serde_json::from_str(body.trim()).ok()?;
    match v.get("args") {
        Some(a) if a.is_object() => Some(a.clone()),
        Some(_) => None,
        None if v.is_object() => Some(v),
        None => None,
    }
}

/// The raw text of the `args` object inside a (possibly partial) `{"name":…,"args":{…}}` body, as it
/// streams. Once the args object closes, the text ends at its own closing brace, whether or not the
/// outer `}` or the end marker has been decoded yet: a streamed delta can't be taken back, so a stray
/// outer `}` sent early would reach the client as `{…}}`.
fn args_text(body: &str, complete: bool) -> Option<&str> {
    let i = body.find("\"args\":")? + "\"args\":".len();
    let rest = &body[i..];
    let object = rest.trim_start();
    if object.starts_with('{') {
        let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
        for (at, c) in object.char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_string = false;
                }
            } else if c == '"' {
                in_string = true;
            } else if c == '{' {
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    return Some(&object[..at + 1]);
                }
            }
        }
        return Some(object); // still inside the args object
    }
    // Args that are not an object are rejected by `tool_args`; keep the old text for them.
    Some(if complete { rest.trim_end().strip_suffix('}').unwrap_or(rest) } else { rest })
}

/// Text that is safe to emit: drops a trailing incomplete UTF-8 sequence (decoded as U+FFFD).
fn stable(s: &str) -> &str {
    s.trim_end_matches('\u{FFFD}')
}

/// Sorts object keys recursively so equal JSON compares equal as text.
fn canonical(v: &Json) -> Json {
    match v {
        Json::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            Json::Object(keys.into_iter().map(|k| (k.clone(), canonical(&m[k]))).collect())
        }
        Json::Array(a) => Json::Array(a.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

fn text_of(content: &Json) -> String {
    match content {
        Json::String(s) => s.clone(),
        Json::Array(parts) => parts
            .iter()
            .filter_map(|p| p.as_str().map(str::to_owned).or_else(|| p.get("text").and_then(Json::as_str).map(str::to_owned)))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// True when the tail of `out` repeats: a period of at most 64 tokens seen 8 times in a row, or the
/// last 200 tokens appearing twice before.
fn looping(out: &[u32], check_long: bool) -> bool {
    for p in 1..=64usize {
        let need = p * 8;
        if out.len() < need {
            break;
        }
        let tail = &out[out.len() - need..];
        if (p..need).all(|i| tail[i] == tail[i - p]) {
            return true;
        }
    }
    if check_long && out.len() >= 600 {
        let w = &out[out.len() - 200..];
        let hay = &out[..out.len() - 200];
        return hay.windows(200).filter(|x| *x == w).take(2).count() >= 2;
    }
    false
}

fn random_id(prefix: &str, seed: &mut u64) -> String {
    const A: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut s = String::from(prefix);
    for _ in 0..24 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        s.push(A[(*seed % A.len() as u64) as usize] as char);
    }
    s
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------------
// HTTP

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

fn read_request(conn: &mut BufReader<TcpStream>) -> Result<Option<Request>, String> {
    let mut line = String::new();
    if conn.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
        return Ok(None);
    }
    let mut it = line.split_whitespace();
    let method = it.next().unwrap_or("").to_owned();
    let path = it.next().unwrap_or("").to_owned();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if conn.read_line(&mut h).map_err(|e| e.to_string())? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_owned(), v.trim().to_owned()));
        }
    }
    let mut req = Request { method, path, headers, body: Vec::new() };
    if req.header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
        loop {
            let mut sz = String::new();
            conn.read_line(&mut sz).map_err(|e| e.to_string())?;
            let n = usize::from_str_radix(sz.trim().split(';').next().unwrap_or("0"), 16).map_err(|_| "bad chunk size")?;
            let mut chunk = vec![0u8; n + 2];
            conn.read_exact(&mut chunk).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            req.body.extend_from_slice(&chunk[..n]);
        }
    } else if let Some(len) = req.header("content-length") {
        let len: usize = len.parse().map_err(|_| "bad content-length")?;
        req.body = vec![0u8; len];
        conn.read_exact(&mut req.body).map_err(|e| e.to_string())?;
    }
    Ok(Some(req))
}

fn respond_raw(conn: &mut TcpStream, status: &str, ctype: &str, extra: &str, body: &[u8]) -> std::io::Result<()> {
    write!(conn, "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n", body.len())?;
    conn.write_all(body)?;
    conn.flush()
}

fn respond(conn: &mut TcpStream, status: &str, body: &Json) -> std::io::Result<()> {
    respond_raw(conn, status, "application/json", "", &serde_json::to_vec(body).unwrap())
}

fn error_body(msg: &str, kind: &str, code: Option<&str>) -> Json {
    json!({"error": {"message": msg, "type": kind, "code": code}})
}

/// True when the peer closed the connection (checked without blocking).
fn peer_closed(conn: &TcpStream) -> bool {
    if conn.set_nonblocking(true).is_err() {
        return false;
    }
    let mut b = [0u8; 1];
    let closed = matches!(conn.peek(&mut b), Ok(0)) || matches!(conn.peek(&mut b), Err(e) if e.kind() != std::io::ErrorKind::WouldBlock);
    let _ = conn.set_nonblocking(false);
    closed
}

// ---------------------------------------------------------------------------------------------------
// Metrics

const BUCKETS: [f64; 10] = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0];

#[derive(Default)]
struct Hist {
    counts: [u64; 10],
    count: u64,
    sum: f64,
}

impl Hist {
    fn observe(&mut self, v: f64) {
        for (i, b) in BUCKETS.iter().enumerate() {
            if v <= *b {
                self.counts[i] += 1;
            }
        }
        self.count += 1;
        self.sum += v;
    }
    fn render(&self, name: &str, help: &str, out: &mut String) {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
        for (i, b) in BUCKETS.iter().enumerate() {
            out.push_str(&format!("{name}_bucket{{le=\"{b}\"}} {}\n", self.counts[i]));
        }
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {}\n{name}_sum {}\n{name}_count {}\n", self.count, self.sum, self.count));
    }
}

#[derive(Default)]
struct Metrics {
    requests: std::collections::BTreeMap<String, u64>,
    reuse: std::collections::BTreeMap<String, u64>,
    prompt_tokens: u64,
    cached_tokens: u64,
    generated: u64,
    rejected: u64,
    ttft: Hist,
    request: Hist,
    queue: Hist,
    last_tok_s: f64,
    last_tpv: f64,
}

impl Metrics {
    fn render(&self, queue_depth: usize) -> String {
        let mut o = String::new();
        o.push_str("# TYPE inkling_requests_total counter\n");
        for (k, v) in &self.requests {
            o.push_str(&format!("inkling_requests_total{{finish=\"{k}\"}} {v}\n"));
        }
        o.push_str("# TYPE inkling_reuse_total counter\n");
        for (k, v) in &self.reuse {
            o.push_str(&format!("inkling_reuse_total{{mode=\"{k}\"}} {v}\n"));
        }
        for (n, v) in [
            ("inkling_prompt_tokens_total", self.prompt_tokens),
            ("inkling_prompt_tokens_cached_total", self.cached_tokens),
            ("inkling_generation_tokens_total", self.generated),
            ("inkling_rejected_total", self.rejected),
        ] {
            o.push_str(&format!("# TYPE {n} counter\n{n} {v}\n"));
        }
        self.ttft.render("inkling_ttft_seconds", "Time from engine start of a request to its first token.", &mut o);
        self.request.render("inkling_request_seconds", "Engine time per request.", &mut o);
        self.queue.render("inkling_queue_seconds", "Time a request waited for the engine.", &mut o);
        for (n, v) in [
            ("inkling_decode_tokens_per_second", self.last_tok_s),
            ("inkling_tokens_per_verify", self.last_tpv),
            ("inkling_queue_depth", queue_depth as f64),
        ] {
            o.push_str(&format!("# TYPE {n} gauge\n{n} {v}\n"));
        }
        o
    }
}

struct Shared {
    ready: AtomicBool,
    /// Requests waiting or running.
    depth: AtomicUsize,
    metrics: Mutex<Metrics>,
    max_ctx: usize,
    default_effort: String,
}

struct Job {
    conn: TcpStream,
    body: Vec<u8>,
    session: Option<String>,
    enqueued: Instant,
}

// ---------------------------------------------------------------------------------------------------
// Engine side

struct Results {
    accepted: PinnedBuffer,
    n: PinnedBuffer,
    width: PinnedBuffer,
}

impl Results {
    fn record_copies(&mut self, ds: &Dspark, s: &Stream) -> Result<(), String> {
        self.accepted.copy_from_device_async(&ds.slice(DsBuf::Accepted).sub(0, 32), s)?;
        self.n.copy_from_device_async(&ds.slice(DsBuf::NumAccepted).sub(0, 4), s)?;
        self.width.copy_from_device_async(&ds.slice(DsBuf::Width).sub(0, 4), s)?;
        Ok(())
    }
    fn i32_at(b: &PinnedBuffer, i: usize) -> i32 {
        i32::from_ne_bytes(b.as_slice()[4 * i..4 * i + 4].try_into().unwrap())
    }
}

/// A snapshot of the rolling state after `tokens` were fed.
struct Checkpoint {
    tokens: Vec<u32>,
    buf: DeviceBuffer,
    used: Instant,
    header: bool,
}

struct ReasoningMemory {
    entries: Vec<(String, String)>,
}

impl ReasoningMemory {
    fn key(content: &str, tool_calls: &[(String, Json)]) -> String {
        let calls: Vec<Json> = tool_calls.iter().map(|(n, a)| json!([n, canonical(a)])).collect();
        format!("{}\u{1}{}", content.trim(), serde_json::to_string(&calls).unwrap())
    }
    fn remember(&mut self, key: String, reasoning: String) {
        if reasoning.is_empty() {
            return;
        }
        self.entries.retain(|(k, _)| *k != key);
        self.entries.push((key, reasoning));
        if self.entries.len() > 256 {
            self.entries.remove(0);
        }
    }
    fn find(&self, key: &str) -> Option<&str> {
        self.entries.iter().rev().find(|(k, _)| k == key).map(|(_, r)| r.as_str())
    }
}

/// Streams SSE events; any write error means the client went away.
struct Sse {
    conn: TcpStream,
    id: String,
    created: u64,
}

impl Sse {
    fn event(&mut self, v: &Json) -> std::io::Result<()> {
        write!(self.conn, "data: {}\n\n", serde_json::to_string(v).unwrap())?;
        self.conn.flush()
    }
    fn chunk(&mut self, delta: Json, finish: Option<&str>) -> std::io::Result<()> {
        let c = json!({"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": MODEL_ID,
                       "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
        self.event(&c)
    }
    fn ping(&mut self) -> std::io::Result<()> {
        self.conn.write_all(b": ping\n\n")?;
        self.conn.flush()
    }
}

/// What has been streamed so far.
#[derive(Default)]
struct Emitted {
    text_bytes: Vec<usize>,
    /// Per tool segment: (id, arguments bytes sent).
    tools: Vec<(String, usize)>,
}

struct Engine<'a> {
    run: Runner<'a>,
    ds: &'a Dspark<'a>,
    stream: &'a Stream,
    text: TextModel,
    prefill: Vec<(i32, GraphExec)>,
    prefill1: GraphExec,
    wide: Option<inkling_engine::WidePrefill<'a>>,
    draft: GraphExec,
    verify: GraphExec,
    res: Results,
    done: inkling_runtime::Event,
    rolling: Vec<DevSlice>,
    rolling_bytes: usize,
    /// Tokens with valid KV at positions `0..live.len()`, and the highest position any step may have
    /// written (verify rounds write up to 8 past the accepted tokens).
    live: Vec<u32>,
    hwm: usize,
    checkpoints: Vec<Checkpoint>,
    memory: ReasoningMemory,
    shared: Arc<Shared>,
    log: Option<std::fs::File>,
    log_bodies: bool,
    max_request: Duration,
    seed: u64,
    served: u64,
    sys_id: u32,
    text_id: u32,
}

/// How a request's prompt reached the KV cache.
struct Prefill {
    mode: &'static str,
    start: usize,
    lcp: usize,
    seconds: f64,
}

enum Outcome {
    Done,
    ClientGone,
}

impl Engine<'_> {
    fn normalize(&self, msgs: &Json) -> Result<(Json, usize), String> {
        let arr = msgs.as_array().ok_or("messages must be an array")?;
        let mut out = Vec::new();
        let mut restored = 0;
        for m in arr {
            let mut m = m.clone();
            let role = m.get("role").and_then(Json::as_str).unwrap_or("").to_owned();
            if role == "developer" {
                m["role"] = "system".into();
            }
            let flat = match m.get("content") {
                Some(c @ Json::Array(parts)) => {
                    if parts.iter().any(|p| p.get("type").and_then(Json::as_str).is_some_and(|t| t.contains("image"))) {
                        return Err("this server is text only; image content is not supported".into());
                    }
                    Some(text_of(c))
                }
                _ => None,
            };
            if let Some(f) = flat {
                m["content"] = Json::String(f);
            }
            if role == "assistant" {
                for alt in ["reasoning", "reasoning_text"] {
                    if m.get("reasoning_content").is_none_or(Json::is_null) {
                        if let Some(r) = m.get(alt).cloned() {
                            m["reasoning_content"] = r;
                        }
                    }
                }
                let mut calls: Vec<(String, Json)> = Vec::new();
                if let Some(Json::Array(tcs)) = m.get_mut("tool_calls") {
                    for tc in tcs.iter_mut() {
                        let f = &mut tc["function"];
                        if let Some(Json::String(a)) = f.get("arguments").cloned() {
                            f["arguments"] = serde_json::from_str(&a).unwrap_or_else(|_| json!({ "input": a }));
                        }
                        if f.get("arguments").is_none_or(|a| !a.is_object()) {
                            f["arguments"] = json!({});
                        }
                        calls.push((f["name"].as_str().unwrap_or("").to_owned(), f["arguments"].clone()));
                    }
                }
                let has = m.get("reasoning_content").and_then(Json::as_str).is_some_and(|s| !s.is_empty());
                if !has {
                    let key = ReasoningMemory::key(&m.get("content").map(text_of).unwrap_or_default(), &calls);
                    if let Some(r) = self.memory.find(&key) {
                        m["reasoning_content"] = r.into();
                        restored += 1;
                    }
                }
                if m.get("content").is_some_and(Json::is_null) {
                    m.as_object_mut().unwrap().remove("content");
                }
            }
            out.push(m);
        }
        Ok((Json::Array(out), restored))
    }

    /// End of the system/tools header: the position of the effort message the template inserts before
    /// the first non-system turn.
    fn header_end(&self, prompt: &[u32]) -> Option<usize> {
        (0..prompt.len().saturating_sub(2)).find(|&i| {
            prompt[i] == self.sys_id
                && prompt[i + 1] == self.text_id
                && self.text.decode(&prompt[i + 2..(i + 4).min(prompt.len())], false).is_ok_and(|s| s.starts_with("Thinking"))
        })
    }

    fn snapshot(&mut self, tokens: &[u32], header: bool) -> Result<(), String> {
        if self.checkpoints.iter().any(|c| c.tokens == tokens) {
            return Ok(());
        }
        if max_checkpoints() == 0 {
            return Ok(());
        }
        let buf = if self.checkpoints.len() >= max_checkpoints() {
            // Evict the least recently used, keeping the newest header checkpoint when possible.
            let newest_header = self.checkpoints.iter().enumerate().filter(|(_, c)| c.header).max_by_key(|(_, c)| c.used).map(|(i, _)| i);
            let victim = self
                .checkpoints
                .iter()
                .enumerate()
                .filter(|(i, _)| Some(*i) != newest_header)
                .min_by_key(|(_, c)| c.used)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.checkpoints.swap_remove(victim).buf
        } else {
            DeviceBuffer::alloc(self.rolling_bytes)?
        };
        let dst = buf.slice();
        let mut off = 0;
        for s in &self.rolling {
            s.copy_to(&dst.sub(off, s.bytes), self.stream)?;
            off += s.bytes;
        }
        self.checkpoints.push(Checkpoint { tokens: tokens.to_vec(), buf, used: Instant::now(), header });
        Ok(())
    }

    fn restore(&mut self, i: usize) -> Result<(), String> {
        let src = self.checkpoints[i].buf.slice();
        let mut off = 0;
        for s in &self.rolling {
            src.sub(off, s.bytes).copy_to(s, self.stream)?;
            off += s.bytes;
        }
        self.checkpoints[i].used = Instant::now();
        Ok(())
    }

    /// Prefill `prompt[start..end]` at positions `start..end`, the last token through the B = 1 graph.
    /// Syncs every 512 tokens to call `tick`, which may ping the client; returns the greedy next token.
    /// With wide prefill, ranges of at least INKLING_WIDE_MIN (256) tokens use its T-token chunks.
    fn prefill_range(&self, prompt: &[u32], start: usize, end: usize, tick: &mut dyn FnMut() -> bool) -> Result<Option<u32>, String> {
        if let Some(w) = &self.wide {
            return w.prefill_range(&self.prefill, &self.prefill1, prompt, start, end, tick);
        }
        let (st, stream) = (self.run.state, self.stream);
        let mut chunks: Vec<(usize, usize)> = Vec::new();
        let mut i = start;
        while i + 1 < end {
            let left = end - 1 - i;
            let b = self.prefill.iter().map(|(b, _)| *b as usize).find(|&b| b <= left).unwrap_or(1);
            chunks.push((i, b));
            i += b;
        }
        chunks.push((end - 1, 1));
        let n = end;
        let mut host = PinnedBuffer::alloc(4 * n + 4 * chunks.len() + 8)?;
        {
            let h = host.as_mut_slice();
            for (j, &t) in prompt[..n].iter().enumerate() {
                h[4 * j..4 * j + 4].copy_from_slice(&(t as i32).to_ne_bytes());
            }
            for (c, &(s0, _)) in chunks.iter().enumerate() {
                let o = 4 * n + 4 * c;
                h[o..o + 4].copy_from_slice(&(s0 as i32).to_ne_bytes());
            }
        }
        let d_tok = st.get(st.s.d_tokens);
        let d_pos = st.get(st.s.d_pos).sub(0, 4);
        let mut since_sync = 0;
        for (c, &(s0, b)) in chunks.iter().enumerate() {
            host.copy_range_to_device_async(4 * s0, &d_tok.sub(0, 4 * b), stream)?;
            host.copy_range_to_device_async(4 * n + 4 * c, &d_pos, stream)?;
            if b == 1 {
                self.prefill1.launch(stream)?;
            } else {
                self.prefill.iter().find(|(gb, _)| *gb as usize == b).expect("graph for chunk").1.launch(stream)?;
            }
            since_sync += b;
            if since_sync >= 512 {
                since_sync = 0;
                stream.synchronize()?;
                if !tick() {
                    return Ok(None);
                }
            }
        }
        stream.synchronize()?;
        Ok(Some(st.get(st.s.d_next_tokens).download::<i32>(1)?[0] as u32))
    }

    /// Brings the KV cache to `prompt` (extend, checkpoint or from scratch), taking checkpoints at the end
    /// of the header and of the prompt. Returns None when the client went away during prefill.
    fn load_prompt(&mut self, prompt: &[u32], tick: &mut dyn FnMut() -> bool) -> Result<Option<(u32, Prefill)>, String> {
        let t0 = Instant::now();
        let lcp = self.live.iter().zip(prompt).take_while(|(a, b)| a == b).count().min(prompt.len() - 1);
        let (mode, start) = if lcp > 0 && lcp + 13 >= self.hwm {
            ("extend", lcp)
        } else if let Some((i, len)) = self
            .checkpoints
            .iter()
            .enumerate()
            .filter(|(_, c)| c.tokens.len() <= lcp && prompt.starts_with(&c.tokens))
            .map(|(i, c)| (i, c.tokens.len()))
            .max_by_key(|&(_, len)| len)
        {
            self.restore(i)?;
            ("checkpoint", len)
        } else {
            self.run.reset()?;
            self.ds.reset(self.stream)?;
            ("none", 0)
        };
        // Invalidate checkpoints whose tokens the new prompt overwrites (positions >= start).
        self.live.truncate(start);
        self.checkpoints.retain(|c| c.tokens.len() <= start || prompt.starts_with(&c.tokens));
        let mut from = start;
        if let Some(h) = self.header_end(prompt) {
            if from < h && h + 1 < prompt.len() {
                if self.prefill_range(prompt, from, h, tick)?.is_none() {
                    return Ok(None);
                }
                self.snapshot(&prompt[..h], true)?;
                from = h;
            }
        }
        // Snapshot one token before the end: a resent prompt can reuse at most len - 1 positions (the last
        // token must run again for its logits), so a checkpoint at len would never match it.
        let n = prompt.len();
        if from + 1 < n {
            if self.prefill_range(prompt, from, n - 1, tick)?.is_none() {
                return Ok(None);
            }
            self.snapshot(&prompt[..n - 1], false)?;
            from = n - 1;
        }
        let Some(next) = self.prefill_range(prompt, from, n, tick)? else { return Ok(None) };
        self.live = prompt.to_vec();
        self.hwm = prompt.len();
        Ok(Some((next, Prefill { mode, start, lcp, seconds: t0.elapsed().as_secs_f64() })))
    }

    fn handle(&mut self, job: Job) {
        let queue_s = job.enqueued.elapsed().as_secs_f64();
        self.shared.metrics.lock().unwrap().queue.observe(queue_s);
        let mut conn = job.conn;
        let req: Json = match serde_json::from_slice(&job.body) {
            Ok(v) => v,
            Err(e) => {
                let _ = respond(&mut conn, "400 Bad Request", &error_body(&format!("bad JSON: {e}"), "invalid_request_error", None));
                return;
            }
        };
        let mut sse: Option<Sse> = None;
        let result = self.chat(&mut conn, &req, &mut sse, job.session.as_deref(), queue_s);
        if let Err(e) = result {
            eprintln!("[serve] request failed: {e}");
            // The engine state may not match `live` after a failure: forget the caches.
            self.live.clear();
            self.hwm = 0;
            self.checkpoints.clear();
            match sse.as_mut() {
                Some(s) => {
                    let _ = s.event(&error_body(&e, "server_error", None));
                    let _ = s.conn.write_all(b"data: [DONE]\n\n");
                }
                None => {
                    let _ = respond(&mut conn, "500 Internal Server Error", &error_body(&e, "server_error", None));
                }
            }
        }
    }

    fn chat(&mut self, conn: &mut TcpStream, req: &Json, sse_slot: &mut Option<Sse>, session: Option<&str>, queue_s: f64) -> Result<(), String> {
        let t_req = Instant::now();
        let stream_mode = req.get("stream").and_then(Json::as_bool).unwrap_or(false);
        let include_usage = req.pointer("/stream_options/include_usage").and_then(Json::as_bool).unwrap_or(false);
        let bad = |conn: &mut TcpStream, msg: &str, code: Option<&str>, extra: Json| -> Result<(), String> {
            let mut b = error_body(msg, "invalid_request_error", code);
            if let (Json::Object(e), Json::Object(x)) = (&mut b["error"], extra) {
                e.extend(x);
            }
            respond(conn, "400 Bad Request", &b).map_err(|e| e.to_string())
        };
        let (msgs, restored) = match self.normalize(req.get("messages").unwrap_or(&Json::Null)) {
            Ok(m) => m,
            Err(e) => return bad(conn, &e, None, json!({})),
        };
        let tool_choice = match req.get("tool_choice") {
            None | Some(Json::Null) => "auto",
            Some(Json::String(s)) if s == "auto" || s == "none" => s.as_str(),
            Some(_) => return bad(conn, "tool_choice: only \"auto\" and \"none\" are supported", Some("unsupported_tool_choice"), json!({})),
        };
        let tools = req.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
        let n_tools = tools.and_then(Json::as_array).map_or(0, Vec::len);
        let thinking_off = req.get("enable_thinking").and_then(Json::as_bool) == Some(false)
            || req.pointer("/chat_template_kwargs/enable_thinking").and_then(Json::as_bool) == Some(false);
        let effort = if thinking_off {
            Json::from("none")
        } else {
            match req.get("reasoning_effort") {
                Some(Json::String(s)) if s == "minimal" => Json::from("none"),
                Some(e) if !e.is_null() => e.clone(),
                _ => Json::from(self.shared.default_effort.clone()),
            }
        };
        let prompt = match self.text.encode_chat(&msgs, tools, Some(&effort)) {
            Ok(p) => p,
            Err(e) => return bad(conn, &e, None, json!({})),
        };
        let max_ctx = self.shared.max_ctx;
        let room = max_ctx.saturating_sub(prompt.len() + 16);
        if room < 16 {
            let msg = format!(
                "This model's maximum context length is {max_ctx} tokens. However, your messages resulted in {} tokens. Please reduce the length of the messages.",
                prompt.len()
            );
            return bad(conn, &msg, Some("context_length_exceeded"), json!({"n_prompt_tokens": prompt.len(), "n_ctx": max_ctx}));
        }
        let max_tokens = req
            .get("max_completion_tokens")
            .or_else(|| req.get("max_tokens"))
            .and_then(Json::as_u64)
            .map(|v| v as usize)
            .unwrap_or(16384)
            .min(room);
        let stops: Vec<String> = match req.get("stop") {
            Some(Json::String(s)) => vec![s.clone()],
            Some(Json::Array(a)) => a.iter().filter_map(|s| s.as_str().map(str::to_owned)).collect(),
            _ => Vec::new(),
        };
        let sampled = req.get("temperature").and_then(Json::as_f64).is_some_and(|t| t > 0.0);

        let id = random_id("chatcmpl-", &mut self.seed);
        let created = now_secs();
        if stream_mode {
            write!(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n").map_err(|e| e.to_string())?;
            let mut s = Sse { conn: conn.try_clone().map_err(|e| e.to_string())?, id: id.clone(), created };
            if s.chunk(json!({"role": "assistant", "content": ""}), None).is_err() {
                return Ok(());
            }
            *sse_slot = Some(s);
        }

        // Prefill, pinging the client every 5 s (stream) or checking it is still there.
        let mut last_ping = Instant::now();
        let probe = conn.try_clone().map_err(|e| e.to_string())?;
        let pf = {
            let sse_ref = &mut *sse_slot;
            let mut tick = || -> bool {
                if last_ping.elapsed() < PING_EVERY {
                    return true;
                }
                last_ping = Instant::now();
                match sse_ref.as_mut() {
                    Some(s) => s.ping().is_ok(),
                    None => !peer_closed(&probe),
                }
            };
            self.load_prompt(&prompt, &mut tick)?
        };
        let Some((next, pf)) = pf else {
            self.log_line(json!({"t": now_secs(), "id": id, "prompt": prompt.len(), "finish": "client_gone_in_prefill"}));
            return Ok(());
        };
        let st = self.run.state;
        st.get(st.s.d_next_tokens).sub(0, 4).copy_to(&self.ds.slice(DsBuf::Anchor), self.stream)?;

        // Decode.
        let mut out = vec![next];
        let mut emitted = Emitted::default();
        let mut outcome = Outcome::Done;
        let (mut stopped_by_string, mut loop_stop, mut tool_cut, mut timed_out) = (false, false, false, false);
        let (mut rounds, mut draft_in_flight) = (0usize, false);
        let t_dec = Instant::now();
        let mut last_emit = Instant::now();
        let mut ttft_first: Option<f64> = None;
        while out.len() < max_tokens && *out.last().unwrap() != dims::EOS {
            if !draft_in_flight {
                self.draft.launch(self.stream)?;
            }
            self.verify.launch(self.stream)?;
            self.done.record(self.stream)?;
            self.draft.launch(self.stream)?;
            draft_in_flight = true;
            self.done.synchronize()?;
            let n = Results::i32_at(&self.res.n, 0) as usize;
            let w = Results::i32_at(&self.res.width, 0) as usize;
            if n == 0 || n > w || w > 8 {
                return Err(format!("spec_verify accepted {n} tokens at width {w}"));
            }
            rounds += 1;
            for j in 0..n {
                let t = Results::i32_at(&self.res.accepted, j) as u32;
                out.push(t);
                if t == dims::EOS || out.len() >= max_tokens {
                    break;
                }
            }
            if looping(&out, rounds % 50 == 0) {
                loop_stop = true;
                break;
            }
            if t_req.elapsed() > self.max_request {
                timed_out = true;
                break;
            }
            if sse_slot.is_none() && rounds % 50 == 0 && peer_closed(conn) {
                outcome = Outcome::ClientGone;
                break;
            }
            let check = sse_slot.is_some() || !stops.is_empty() || tool_choice == "none";
            if check && last_emit.elapsed().as_millis() >= 20 {
                last_emit = Instant::now();
                let segs = parse_output(&self.text.decode(&out, false)?);
                if !stops.is_empty() && segs.iter().any(|s| matches!(s, Seg::Content(t) if stops.iter().any(|x| t.contains(x)))) {
                    stopped_by_string = true;
                    break;
                }
                if tool_choice == "none" && segs.iter().any(|s| matches!(s, Seg::Tool { .. })) {
                    tool_cut = true;
                    break;
                }
                if let Some(s) = sse_slot.as_mut() {
                    if ttft_first.is_none() && !segs.is_empty() {
                        ttft_first = Some(t_req.elapsed().as_secs_f64());
                    }
                    if emit(s, &segs, &mut emitted, &mut self.seed, false).is_err() {
                        outcome = Outcome::ClientGone;
                        break;
                    }
                }
            }
        }
        let decode_s = t_dec.elapsed().as_secs_f64();
        self.stream.synchronize()?;
        out.truncate(max_tokens);

        // Fed tokens (KV valid): the prompt and every generated token but the last.
        self.live.extend_from_slice(&out[..out.len() - 1]);
        self.hwm = self.live.len() + 8;

        // Final segments.
        let mut segs = parse_output(&self.text.decode(&out, false)?);
        if !stops.is_empty() {
            for s in segs.iter_mut() {
                if let Seg::Content(t) = s {
                    if let Some(i) = stops.iter().filter_map(|x| t.find(x.as_str())).min() {
                        t.truncate(i);
                        stopped_by_string = true;
                    }
                }
            }
        }
        if tool_choice == "none" {
            segs.retain(|s| !matches!(s, Seg::Tool { .. }));
        }
        let mut truncated_call = false;
        let mut bad_calls = 0;
        let mut calls: Vec<(String, Json)> = Vec::new();
        let mut final_segs: Vec<Seg> = Vec::new();
        for s in segs {
            match s {
                Seg::Tool { complete: false, .. } => truncated_call = true,
                Seg::Tool { name, body, json, complete: true } => match tool_args(&body, json) {
                    Some(a) => {
                        calls.push((name.clone(), a));
                        final_segs.push(Seg::Tool { name, body, json, complete: true });
                    }
                    None => {
                        bad_calls += 1;
                        final_segs.push(Seg::Content(format!("{name}{body}")));
                    }
                },
                other => final_segs.push(other),
            }
        }
        let reasoning: String = final_segs.iter().filter_map(|s| if let Seg::Reasoning(t) = s { Some(t.as_str()) } else { None }).collect();
        let content: String = final_segs.iter().filter_map(|s| if let Seg::Content(t) = s { Some(t.as_str()) } else { None }).collect();
        self.memory.remember(ReasoningMemory::key(&content, &calls), reasoning.clone());
        let finish = if truncated_call || loop_stop || timed_out {
            "length"
        } else if !calls.is_empty() {
            "tool_calls"
        } else if stopped_by_string || tool_cut || out.last() == Some(&dims::EOS) {
            "stop"
        } else {
            "length"
        };

        let completion = out.len();
        let ngen = completion.saturating_sub(1);
        let tps = if decode_s > 0.0 { ngen as f64 / decode_s } else { 0.0 };
        let tpv = if rounds > 0 { ngen as f64 / rounds as f64 } else { 0.0 };
        let prefilled = prompt.len() - pf.start;
        let timings = json!({
            "prompt_n": prefilled, "prompt_ms": pf.seconds * 1e3, "prompt_per_second": prefilled as f64 / pf.seconds.max(1e-9),
            "cache_n": pf.start, "reuse": pf.mode, "predicted_n": ngen, "predicted_ms": decode_s * 1e3,
            "predicted_per_second": tps, "tokens_per_verify": tpv
        });
        let usage = json!({
            "prompt_tokens": prompt.len(), "completion_tokens": completion, "total_tokens": prompt.len() + completion,
            "prompt_tokens_details": {"cached_tokens": pf.start}
        });
        {
            let mut m = self.shared.metrics.lock().unwrap();
            let fin = if matches!(outcome, Outcome::ClientGone) { "cancelled" } else { finish };
            *m.requests.entry(fin.into()).or_default() += 1;
            *m.reuse.entry(pf.mode.into()).or_default() += 1;
            m.prompt_tokens += prompt.len() as u64;
            m.cached_tokens += pf.start as u64;
            m.generated += completion as u64;
            m.ttft.observe(pf.seconds);
            m.request.observe(t_req.elapsed().as_secs_f64());
            m.last_tok_s = tps;
            m.last_tpv = tpv;
        }
        let mut line = json!({
            "t": now_secs(), "id": id, "session": session, "queue_s": queue_s, "prompt": prompt.len(), "reuse": pf.mode,
            "lcp": pf.lcp, "reused": pf.start, "prefilled": prefilled, "ttft_s": pf.seconds,
            "prefill_tok_s": prefilled as f64 / pf.seconds.max(1e-9), "completion": completion, "decode_s": decode_s,
            "tok_s": tps, "rounds": rounds, "tokens_per_verify": tpv, "tool_calls": calls.len(), "bad_tool_calls": bad_calls,
            "truncated_tool_call": truncated_call, "finish": finish, "loop_stop": loop_stop, "timed_out": timed_out,
            "stream": stream_mode, "client_gone": matches!(outcome, Outcome::ClientGone), "effort": effort,
            "n_tools": n_tools, "max_tokens": max_tokens, "reasoning_restored": restored, "sampled_request": sampled,
            "checkpoints": self.checkpoints.len(), "total_s": t_req.elapsed().as_secs_f64(),
            "first_delta_s": ttft_first
        });
        if pf.mode == "none" && pf.lcp > 0 {
            line["diverge_at"] = pf.lcp.into();
            if self.log_bodies {
                let a = pf.lcp.saturating_sub(12);
                line["diverge_context"] = json!({
                    "before": self.text.decode(&prompt[a..pf.lcp], false).unwrap_or_default(),
                    "prompt_after": self.text.decode(&prompt[pf.lcp..(pf.lcp + 12).min(prompt.len())], false).unwrap_or_default(),
                });
            }
        }
        if pf.seconds > 5.0 || (pf.mode == "none" && prefilled > 4096) {
            line["slow"] = true.into();
        }
        eprintln!(
            "[serve] prompt {} ({} reuse {}, prefilled {} in {:.3} s) -> {} tokens in {:.2} s = {:.0} tok/s, {:.2} tok/verify, {} tool calls, {}{}{}",
            prompt.len(),
            pf.mode,
            pf.start,
            prefilled,
            pf.seconds,
            completion,
            decode_s,
            tps,
            tpv,
            calls.len(),
            finish,
            if loop_stop { " (loop guard)" } else { "" },
            if matches!(outcome, Outcome::ClientGone) { " (client gone)" } else { "" }
        );
        self.log_line(line);
        self.served += 1;
        if matches!(outcome, Outcome::ClientGone) {
            return Ok(());
        }

        let mut tool_ids: Vec<String> = emitted.tools.iter().map(|(id, _)| id.clone()).collect();
        while tool_ids.len() < calls.len() {
            tool_ids.push(random_id("call_", &mut self.seed));
        }
        let tool_json: Vec<Json> = calls
            .iter()
            .zip(&tool_ids)
            .map(|((name, args), id)| json!({"id": id, "type": "function", "function": {"name": name, "arguments": serde_json::to_string(args).unwrap()}}))
            .collect();
        if let Some(s) = sse_slot.as_mut() {
            let _ = (|| -> std::io::Result<()> {
                emit(s, &final_segs, &mut emitted, &mut self.seed, true)?;
                let mut c = json!({"id": s.id, "object": "chat.completion.chunk", "created": s.created, "model": MODEL_ID,
                                   "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "timings": timings});
                if include_usage {
                    s.event(&c)?;
                    c = json!({"id": s.id, "object": "chat.completion.chunk", "created": s.created, "model": MODEL_ID, "choices": [], "usage": usage});
                }
                s.event(&c)?;
                s.conn.write_all(b"data: [DONE]\n\n")?;
                s.conn.flush()
            })();
            return Ok(());
        }
        let mut msg = json!({"role": "assistant", "content": if content.is_empty() && !tool_json.is_empty() { Json::Null } else { Json::from(content) }});
        if !reasoning.is_empty() {
            msg["reasoning_content"] = reasoning.into();
        }
        if !tool_json.is_empty() {
            msg["tool_calls"] = Json::Array(tool_json);
        }
        let body = json!({
            "id": id, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage, "timings": timings
        });
        respond(conn, "200 OK", &body).map_err(|e| e.to_string())
    }

    fn log_line(&mut self, line: Json) {
        if let Some(f) = self.log.as_mut() {
            let _ = writeln!(f, "{line}");
        }
    }
}

/// Streams new reasoning and content text, and tool calls: the name and id as soon as the header is
/// complete, then `arguments` deltas of the raw args JSON. With `last`, tool calls that never streamed
/// (non-JSON forms) are sent whole, and calls that ended up dropped are not.
fn emit(s: &mut Sse, segs: &[Seg], em: &mut Emitted, seed: &mut u64, last: bool) -> std::io::Result<()> {
    let (mut ti, mut ci) = (0, 0);
    for seg in segs {
        match seg {
            Seg::Reasoning(t) | Seg::Content(t) => {
                let field = if matches!(seg, Seg::Reasoning(_)) { "reasoning_content" } else { "content" };
                if em.text_bytes.len() <= ti {
                    em.text_bytes.push(0);
                }
                let safe = if last { t.as_str() } else { stable(t) };
                if safe.len() > em.text_bytes[ti] && safe.is_char_boundary(em.text_bytes[ti]) {
                    s.chunk(json!({ field: &safe[em.text_bytes[ti]..] }), None)?;
                    em.text_bytes[ti] = safe.len();
                }
                ti += 1;
            }
            Seg::Tool { name, body, json, complete } => {
                // Stream only calls that will be kept: complete ones at the end must parse.
                if last && (!*complete || tool_args(body, *json).is_none()) {
                    ci += 1;
                    continue;
                }
                if em.tools.len() <= ci {
                    if !last && name.is_empty() {
                        break;
                    }
                    let id = random_id("call_", seed);
                    s.chunk(json!({"tool_calls": [{"index": ci, "id": id, "type": "function", "function": {"name": name, "arguments": ""}}]}), None)?;
                    em.tools.push((id, 0));
                }
                let args = if *json {
                    args_text(body, *complete).map(|a| if last { a.to_owned() } else { stable(a).to_owned() })
                } else if last {
                    tool_args(body, false).map(|a| serde_json::to_string(&a).unwrap())
                } else {
                    None
                };
                if let Some(a) = args {
                    let sent = em.tools[ci].1;
                    if a.len() > sent && a.is_char_boundary(sent) {
                        s.chunk(json!({"tool_calls": [{"index": ci, "function": {"arguments": &a[sent..]}}]}), None)?;
                        em.tools[ci].1 = a.len();
                    }
                }
                ci += 1;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------
// Control thread

fn control(listener: TcpListener, shared: Arc<Shared>, tx: mpsc::SyncSender<Job>, text: Arc<Mutex<Option<TextModel>>>) {
    for conn in listener.incoming() {
        let Ok(mut conn) = conn else { continue };
        let _ = conn.set_read_timeout(Some(Duration::from_secs(30)));
        let Ok(clone) = conn.try_clone() else { continue };
        let mut reader = BufReader::new(clone);
        let req = match read_request(&mut reader) {
            Ok(Some(r)) => r,
            Ok(None) => continue,
            Err(e) => {
                let _ = respond(&mut conn, "400 Bad Request", &error_body(&e, "invalid_request_error", None));
                continue;
            }
        };
        let _ = conn.set_read_timeout(None);
        let path = req.path.split('?').next().unwrap_or("").to_owned();
        let ready = shared.ready.load(Ordering::SeqCst);
        match (req.method.as_str(), path.as_str()) {
            ("GET", "/health") => {
                let (st, body) = if ready { ("200 OK", json!({"status": "ok"})) } else { ("503 Service Unavailable", json!({"status": "loading"})) };
                let _ = respond(&mut conn, st, &body);
            }
            ("GET", "/v1/models") => {
                let _ = respond(
                    &mut conn,
                    "200 OK",
                    &json!({"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "inkling-rs",
                        "context_length": shared.max_ctx,
                        "meta": {"context_length": shared.max_ctx, "default_effort": shared.default_effort, "decoding": "greedy-spec",
                                 "reasoning_field": "reasoning_content", "tool_choice": ["auto", "none"]}}]}),
                );
            }
            ("GET", "/metrics") => {
                let body = shared.metrics.lock().unwrap().render(shared.depth.load(Ordering::SeqCst));
                let _ = respond_raw(&mut conn, "200 OK", "text/plain; version=0.0.4", "", body.as_bytes());
            }
            ("POST", "/apply-template") => {
                let reply = (|| -> Result<Json, String> {
                    let b: Json = serde_json::from_slice(&req.body).map_err(|e| e.to_string())?;
                    let guard = text.lock().unwrap();
                    let t = guard.as_ref().ok_or("loading")?;
                    let effort = b.get("reasoning_effort").cloned().unwrap_or(Json::from(shared.default_effort.clone()));
                    let prompt = t.render(b.get("messages").unwrap_or(&Json::Null), b.get("tools"), Some(&effort))?;
                    Ok(json!({"prompt": prompt}))
                })();
                let _ = match reply {
                    Ok(v) => respond(&mut conn, "200 OK", &v),
                    Err(e) => respond(&mut conn, "400 Bad Request", &error_body(&e, "invalid_request_error", None)),
                };
            }
            ("POST", "/v1/chat/completions") => {
                if !ready {
                    let _ = respond_raw(&mut conn, "503 Service Unavailable", "application/json", "Retry-After: 5\r\n", br#"{"error":{"message":"model is loading","type":"server_error"}}"#);
                    continue;
                }
                let session = req.header("x-session-affinity").or_else(|| req.header("session_id")).map(str::to_owned);
                let depth = shared.depth.fetch_add(1, Ordering::SeqCst);
                let job = Job { conn, body: req.body, session, enqueued: Instant::now() };
                if depth > QUEUE_WAITING {
                    shared.depth.fetch_sub(1, Ordering::SeqCst);
                    shared.metrics.lock().unwrap().rejected += 1;
                    let mut c = job.conn;
                    let _ = respond_raw(&mut c, "503 Service Unavailable", "application/json", "Retry-After: 2\r\n", br#"{"error":{"message":"server busy: queue full","type":"server_error","code":"queue_full"}}"#);
                    continue;
                }
                if let Err(mpsc::TrySendError::Full(job) | mpsc::TrySendError::Disconnected(job)) = tx.try_send(job) {
                    shared.depth.fetch_sub(1, Ordering::SeqCst);
                    let mut c = job.conn;
                    let _ = respond_raw(&mut c, "503 Service Unavailable", "application/json", "Retry-After: 2\r\n", br#"{"error":{"message":"server busy","type":"server_error"}}"#);
                }
            }
            _ => {
                let _ = respond(&mut conn, "404 Not Found", &error_body("not found", "not_found", None));
            }
        }
    }
}

pub fn serve(o: &Opts) -> Result<(), String> {
    if !o.nvfp4 {
        return Err("serve needs --mode fp8 (or nvfp4)".into());
    }
    let port: u16 = std::env::var("INKLING_PORT").ok().and_then(|p| p.parse().ok()).unwrap_or(8013);
    let default_effort = match effort_value(o) {
        Some(Json::String(s)) => s,
        Some(v) => v.to_string(),
        None => "low".into(),
    };
    let shared = Arc::new(Shared {
        ready: AtomicBool::new(false),
        depth: AtomicUsize::new(0),
        metrics: Mutex::new(Metrics::default()),
        max_ctx: o.max_ctx,
        default_effort: default_effort.clone(),
    });
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("bind 127.0.0.1:{port}: {e}"))?;
    let (tx, rx) = mpsc::sync_channel::<Job>(QUEUE_WAITING + 1);
    let control_text = Arc::new(Mutex::new(None));
    {
        let (shared, text) = (shared.clone(), control_text.clone());
        std::thread::spawn(move || control(listener, shared, tx, text));
    }
    eprintln!("[serve] listening on http://127.0.0.1:{port}/v1, loading the model");

    let threshold = o.spec_threshold.unwrap_or(0.8);
    *control_text.lock().unwrap() = Some(TextModel::load(&o.model_dir)?);
    let text = TextModel::load(&o.model_dir)?;
    // Weights only read while packing go to a transient arena, freed below once everything is built
    // (INKLING_KEEP_BF16=1 keeps them).
    let trim = std::env::var_os("INKLING_KEEP_BF16").is_none();
    let eng = engine::Engine::open_split(&o.model_dir, &engine::cubin_path(o.cubin.as_ref()), |n| {
        trim && inkling_model::weights::fp8_serving_transient(n)
    })?;
    let stream = &eng.stream;
    let state = inkling_engine::State::new(o.max_ctx).map_err(|e| e.to_string())?;
    let packed = inkling_engine::Nvfp4Weights::pack(&eng.module, &eng.weights, stream)?;
    let mtp = inkling_model::weights::load_mtp_split(&o.model_dir, stream, LoadOptions::default(), |n| {
        trim && inkling_model::weights::mtp_transient(n)
    })?;
    let shortlist = o.draft_vocab.as_deref().map(read_shortlist).transpose()?;
    let dw = DsparkWeights::prepare(&eng.module, &eng.weights, &packed, mtp, shortlist.as_deref(), stream)?;
    let mut ds = Dspark::new(&eng.module, &eng.weights, &dw, &state)?;
    ds.commit_on_aux = std::env::var_os("INKLING_SPEC_SERIAL_COMMIT").is_none();
    let ds = ds;
    let fwd = inkling_engine::Forward::with_mode(&eng.module, &eng.weights, Some(&packed), &state)?;
    let mut run = Runner { fwd, state: &state, stream };
    if let Ok(spec) = std::env::var("INKLING_NVFP4_PARTS") {
        run.fwd.parts = Nvfp4Parts::parse(&spec)?;
    }
    if ![Nvfp4Parts::ALL, Nvfp4Parts::BATCH_EXACT_DENSE, Nvfp4Parts::BATCH_FP8_DENSE].contains(&run.fwd.parts) {
        return Err("serve needs every nvfp4 part, every part but dense, or FP8 dense".into());
    }
    run.reset()?;
    run.fwd.dspark = Some(&ds);
    let prefill1 = capture::<String>(stream, |s| run.fwd.record_step(s))?;
    let mut prefill = Vec::new();
    for b in [8, 4, 2] {
        prefill.push((b, capture::<String>(stream, |s| run.fwd.record_batch(s, b))?));
    }
    let draft = capture::<String>(stream, |s| ds.record_draft(s))?;
    let mut res = Results { accepted: PinnedBuffer::alloc(32)?, n: PinnedBuffer::alloc(4)?, width: PinnedBuffer::alloc(4)? };
    let path = engine::cubin_path(o.cubin.as_ref()).with_file_name("z_adaptive_select_width.cubin");
    let image = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let m = inkling_runtime::Module::from_cubin(&image, path.display().to_string())?;
    let select = m.function("z_adaptive_select_width").ok_or("z_adaptive_select_width missing")?;
    let verify = {
        let fwd = &run.fwd;
        inkling_runtime::capture_switch::<String>(
            stream,
            3,
            |s, handle| ds.record_select_width(s, select, threshold, handle),
            |s| res.record_copies(&ds, s),
            |i, s| {
                let w = [2, 4, 8][i as usize];
                fwd.record_batch(s, w)?;
                ds.record_spec_verify(s, w)
            },
        )?
    };
    std::mem::forget(m); // the graph references the module's function for the process lifetime
    for gr in [&prefill1, &draft, &verify] {
        gr.upload(stream)?;
    }
    // Wide prefill: on unless INKLING_WIDE_PREFILL=0; with the drafter attached, as spec does.
    let wide = if std::env::var_os("INKLING_WIDE_PREFILL").is_none_or(|v| v != "0") {
        match inkling_engine::WidePrefill::new(&eng.module, &eng.weights, &packed, &run.fwd, &state, stream, Some(&ds)) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("[serve] wide prefill off: {e}");
                None
            }
        }
    } else {
        None
    };
    let wide_desc = wide.as_ref().map_or("off".to_string(), |w| format!("on, chunks {:?}", w.chunk_sizes()));
    let mut rolling = state.rolling_slices();
    rolling.extend(ds.rolling_slices());
    let rolling_bytes: usize = rolling.iter().map(|s| s.bytes).sum();
    let log = std::env::var_os("INKLING_SERVE_LOG").map(|p| std::fs::OpenOptions::new().create(true).append(true).open(p)).transpose().map_err(|e| e.to_string())?;
    let sys_id = text.token_id(MSG_SYSTEM).ok_or("no <|message_system|> token")?;
    let text_id = text.token_id(TEXT).ok_or("no <|content_text|> token")?;
    let mut eng_srv = Engine {
        run,
        ds: &ds,
        stream,
        text,
        prefill,
        prefill1,
        wide,
        draft,
        verify,
        res,
        done: inkling_runtime::Event::new_untimed()?,
        rolling,
        rolling_bytes,
        live: Vec::new(),
        hwm: 0,
        checkpoints: Vec::new(),
        memory: ReasoningMemory { entries: Vec::new() },
        shared: shared.clone(),
        log,
        log_bodies: std::env::var_os("INKLING_SERVE_LOG_BODIES").is_some(),
        max_request: Duration::from_secs(std::env::var("INKLING_MAX_REQUEST_S").ok().and_then(|v| v.parse().ok()).unwrap_or(600)),
        seed: (now_secs() ^ 0x9E37_79B9_7F4A_7C15) | 1,
        served: 0,
        sys_id,
        text_id,
    };
    // Every quantized copy, drafter, wide-prefill object and captured graph exists now, and none of
    // the kernels that run reads the transient weights: release them.
    let freed = eng.weights.free_transient() + dw.mtp.free_transient();
    let (free, total) = eng.gpu.mem_info().map_err(|e| e.to_string())?;
    shared.ready.store(true, Ordering::SeqCst);
    eprintln!(
        "[serve] {} ready on http://127.0.0.1:{port}/v1 (model {MODEL_ID}): fp8 + DSpark adaptive {threshold}, wide prefill {wide_desc}, max ctx {}, default effort {default_effort}, checkpoints {:.1} MB each (up to {}), weights {:.1} s, GPU {:.2} of {:.2} GiB in use",
        eng.gpu.name,
        o.max_ctx,
        rolling_bytes as f64 / 1e6,
        max_checkpoints(),
        eng.load_seconds,
        (total - free) as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    // Where the VRAM goes (GB), from each arena's planned size.
    let gb = |b: usize| b as f64 / 1e9;
    let global_kv: usize = state
        .visits
        .iter()
        .filter(|v| v.kv_cap > state.max_ctx)
        .map(|v| state.get(v.k_cache).bytes + state.get(v.v_cache).bytes)
        .sum();
    let parts = [
        ("checkpoint, resident (native NVFP4 experts 0-19, embed, small)", eng.weights.device_bytes().0),
        ("nvfp4 packed (experts, head, router, ctrl, dense)", packed.arena.total_bytes()),
        ("fp8 dense", eng_srv.run.fwd.fp8.as_ref().map_or(0, |f| f.arena.total_bytes())),
        ("drafter mtp, resident", dw.mtp.device_bytes().0),
        ("drafter packed", dw.arena.total_bytes()),
        ("drafter state", ds.total_bytes()),
        ("global KV (max ctx)", global_kv),
        ("other decode state (local KV, rings, scratch)", state.total_bytes() - global_kv),
        ("wide prefill buffers", eng_srv.wide.as_ref().map_or(0, |w| w.total_bytes())),
    ];
    let sum: usize = parts.iter().map(|p| p.1).sum();
    eprintln!("[serve] VRAM by component (GB), after freeing {:.2} GB of weights read only while packing:", gb(freed));
    for (name, b) in parts {
        eprintln!("[serve]   {:>7.2}  {name}", gb(b));
    }
    eprintln!(
        "[serve]   {:>7.2}  sum; plus checkpoints {:.2} GB each when used; GPU in use by all processes {:.2} GB",
        gb(sum),
        gb(rolling_bytes),
        gb(total - free)
    );
    while let Ok(job) = rx.recv() {
        eng_srv.handle(job);
        shared.depth.fetch_sub(1, Ordering::SeqCst);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_reasoning_text_and_tool_calls() {
        let s = "<|message_model|><|content_thinking|>plan<|end_message|><|message_model|><|content_text|>hi<|end_message|><|message_model|>read<|content_invoke_tool_json|>{\"name\":\"read\",\"args\":{\"path\":\"a.rs\"}}<|end_message|><|content_model_end_sampling|>";
        let segs = parse_output(s);
        assert!(matches!(&segs[0], Seg::Reasoning(t) if t == "plan"));
        assert!(matches!(&segs[1], Seg::Content(t) if t == "hi"));
        match &segs[2] {
            Seg::Tool { name, body, json, complete } => {
                assert_eq!(name, "read");
                assert!(*json && *complete);
                assert_eq!(tool_args(body, true), Some(json!({"path": "a.rs"})));
                assert_eq!(args_text(body, true), Some("{\"path\":\"a.rs\"}"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn partial_tool_args_stream() {
        let segs = parse_output("<|message_model|>edit<|content_invoke_tool_json|>{\"name\":\"edit\",\"args\":{\"path\":\"x");
        match &segs[0] {
            Seg::Tool { body, complete: false, .. } => assert_eq!(args_text(body, false), Some("{\"path\":\"x")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn streamed_args_never_include_the_outer_brace() {
        // The outer `}` can be decoded one step before the end marker, while `complete` is still false.
        let args = "{\"command\":\"grep -n \\\"a}b\\\\|{x\\\" f | head\"}";
        let full = format!("{{\"name\":\"bash\",\"args\":{args}}}");
        for cut in 1..=full.len() {
            if !full.is_char_boundary(cut) {
                continue;
            }
            let streamed = args_text(&full[..cut], false).unwrap_or("");
            assert!(args.starts_with(streamed), "cut {cut}: {streamed}");
            if cut >= full.len() - 1 {
                assert_eq!(streamed, args, "cut {cut}");
            }
        }
        assert_eq!(args_text(&full, true), Some(args));
        assert_eq!(args_text("{\"name\":\"t\",\"args\":\"s\"}", true), Some("\"s\""));
    }

    #[test]
    fn malformed_tool_json_is_rejected() {
        assert_eq!(tool_args("{\"name\":\"x\",\"args\":{\"a\":", true), None);
    }

    #[test]
    fn reasoning_key_ignores_argument_order() {
        let a = ReasoningMemory::key("x", &[("f".into(), json!({"b": 1, "a": 2}))]);
        let b = ReasoningMemory::key("x", &[("f".into(), json!({"a": 2, "b": 1}))]);
        assert_eq!(a, b);
    }

    #[test]
    fn loop_guard() {
        let mut v: Vec<u32> = (0..100).collect();
        assert!(!looping(&v, true));
        for _ in 0..8 {
            v.extend([7, 8, 9]);
        }
        assert!(looping(&v, false));
        let mut w: Vec<u32> = (0..1000).collect();
        let block: Vec<u32> = (5000..5200).collect();
        for _ in 0..3 {
            w.extend(&block);
            w.extend([1, 2, 3]);
        }
        w.extend(&block);
        assert!(looping(&w, true));
    }
}
