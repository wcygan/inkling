#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Prefill speed and output check for a build of the engine, on a private server (port 8099).

    uv run --script scripts/prefill_bench.py run LABEL [--exe FILE] [--cubin FILE] [--quick] [KEY=VALUE ...]
    uv run --script scripts/prefill_bench.py cmp BASE LABEL

`run` starts `inkling serve` with the shipped settings (fp8, FP8 KV caches, 256K context) and writes
logs/prefill-bench/LABEL.json:
  * speed: cold prefill of random-word prompts (about 19K x 5, 64K x 3 and 128K x 2 tokens; one output token);
  * outputs: greedy decodes after prompts of 0.3K to 120K tokens built from this repository's sources at a pinned
    commit, and a three-turn conversation that extends the cache by a long and by a short range.
`cmp` prints the speed of both runs and whether every output is the same text. A change that keeps the arithmetic
must give the same text; one that does not shows up here as the first prompt whose output differs.

The repository's own server must be stopped first (the GPU holds one copy of the model). KEY=VALUE pairs are
environment variables for the server (for example INKLING_WIDE_MOE=mma).
"""
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "prefill-bench"
PORT = 8099
BASE = f"http://127.0.0.1:{PORT}"
PIN = "403798d"  # the commit whose sources make the prompts
FILES = ["crates/inkling-cli/src/server.rs", "crates/inkling-engine/src/forward.rs", "crates/inkling-kernels/src/attn.rs",
         "README.md", "crates/inkling-engine/src/prefill.rs", "crates/inkling-kernels/src/dspark.rs",
         "crates/inkling-kernels/src/nvfp4.rs", "crates/inkling-engine/src/dspark.rs", "crates/inkling-kernels/src/moe.rs",
         "crates/inkling-kernels/src/mma.rs", "crates/inkling-kernels/src/ctrl2.rs", "crates/inkling-kernels/src/prefill_moe.rs"]
WORDS = ("river stone cloud engine garden paper signal market window forest bridge candle harbor meadow "
         "copper violin lantern orbit summit canyon ledger anchor pepper marble falcon timber glacier "
         "compass velvet thunder orchard beacon quartz saddle ribbon furnace island mirror pocket").split()
QUESTIONS = ["Summarize what the code above does, in one paragraph.",
             "List the five most important functions above and what each one is for.",
             "Write a short Python function that parses the first struct above into a dict."]


def corpus() -> str:
    parts = []
    for f in FILES:
        text = subprocess.run(["git", "-C", str(ROOT), "show", f"{PIN}:{f}"], capture_output=True, check=True).stdout.decode("utf-8")
        parts.append(f"// ==== {f}\n{text}")
    return "\n".join(parts)


def source_prompt(text: str, tokens: int, case: int) -> str:
    """About `tokens` tokens of the corpus (3.4 characters per token), from a start that differs per case."""
    chars = int(tokens * 3.4)
    start = (case * 7919) % 5000
    body = (text * (chars // len(text) + 2))[start:start + chars]
    return f"{body}\n\n{QUESTIONS[case % len(QUESTIONS)]}"


def word_prompt(tokens: int, seed: int) -> str:
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(int(tokens / 1.513)))


class Server:
    def __init__(self, label: str, exe: Path, cubin: Path | None, extra: dict[str, str]):
        self.log = OUT / f"{label}.serve.jsonl"
        self.log.unlink(missing_ok=True)
        env = dict(os.environ, INKLING_PORT=str(PORT), INKLING_CHECKPOINTS="6", INKLING_SERVE_LOG=str(self.log),
                   INKLING_KV_FP8="1", INKLING_DRAFT_KV_FP8="1", **extra)
        cuda = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5"))
        env["PATH"] = f"{cuda / 'bin' / 'x64'}{os.pathsep}{env['PATH']}"
        model = os.environ.get("INKLING_MODEL_DIR") or str(Path(os.environ["USERPROFILE"]) / "Models" / "Inkling-2-Mini")
        argv = [str(exe), "serve", "--mode", "fp8", "--draft-vocab", str(ROOT / "assets" / "draft_vocab_32k.bin"),
                "--max-ctx", "262144", "--effort", "low", "--model-dir", model]
        if cubin:
            argv += ["--cubin", str(cubin)]
        self.console = open(OUT / f"{label}.server.txt", "w")
        self.proc = subprocess.Popen(argv, env=env, cwd=ROOT, stdout=self.console, stderr=self.console)
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen(BASE + "/health", timeout=2).read()
                return
            except Exception:
                if self.proc.poll() is not None or time.time() - t0 > 240:
                    self.stop()
                    raise SystemExit(f"the server did not become ready; see {self.console.name}")
                time.sleep(1)

    def chat(self, messages: list, max_tokens: int) -> tuple[str, dict]:
        body = {"model": "inkling-2-mini", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "reasoning_effort": "low"}
        req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(req, timeout=900))
        msg = reply["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + "\x1f" + (msg.get("content") or "")
        for _ in range(200):
            for line in self.log.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["id"] == reply["id"]:
                    return text, row
            time.sleep(0.05)
        raise SystemExit("request missing from the server log")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.console.close()


def run(label: str, exe: Path, cubin: Path | None, quick: bool, extra: dict[str, str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = corpus()
    srv = Server(label, exe, cubin, extra)
    result = {"label": label, "exe": str(exe), "cubin": str(cubin) if cubin else None, "env": extra, "speed": [], "outputs": []}
    try:
        srv.chat([{"role": "user", "content": word_prompt(3000, 1)}], 1)  # warm-up
        seed = 100
        for name, tokens, reps in (("19K", 19000, 5), ("64K", 64000, 3), ("128K", 128000, 2)):
            for _ in range(1 if quick else reps):
                seed += 1
                _, row = srv.chat([{"role": "user", "content": word_prompt(tokens, seed)}], 1)
                assert row["reuse"] == "none" or row["reused"] < 64, row
                result["speed"].append({"size": name, "prefilled": row["prefilled"], "seconds": row["ttft_s"]})
                print(f"  speed {name}: {row['prefilled']} tokens in {row['ttft_s']:.3f} s = {row['prefilled'] / row['ttft_s']:.0f} tok/s", flush=True)
        sizes = [300, 1000, 3000, 8000, 20000, 40000, 64000, 120000]
        for case, tokens in enumerate(sizes[:5] if quick else sizes):
            out, row = srv.chat([{"role": "user", "content": source_prompt(text, tokens, case)}], 160)
            result["outputs"].append({"case": f"cold-{tokens}", "prompt": row["prompt"], "prefilled": row["prefilled"], "text": out,
                                      "tokens_per_verify": row["tokens_per_verify"], "completion": row["completion"]})
            print(f"  output cold-{tokens}: prompt {row['prompt']}, {row['completion']} tokens, {row['tokens_per_verify']:.2f} tok/verify", flush=True)
        # A conversation: 8K, then a turn that adds about 3K (wide chunks from a non-zero start), then a short one.
        msgs = [{"role": "user", "content": source_prompt(text, 8000, 11)}]
        for turn, more in enumerate((None, source_prompt(text, 3000, 12), "Now say the same in two sentences.")):
            if more:
                msgs.append({"role": "user", "content": more})
            out, row = srv.chat(msgs, 120)
            msgs.append({"role": "assistant", "content": out.split("\x1f", 1)[1]})
            result["outputs"].append({"case": f"turn-{turn}", "prompt": row["prompt"], "prefilled": row["prefilled"], "text": out,
                                      "tokens_per_verify": row["tokens_per_verify"], "completion": row["completion"]})
            print(f"  output turn-{turn}: prompt {row['prompt']}, prefilled {row['prefilled']} ({row['reuse']}), {row['tokens_per_verify']:.2f} tok/verify", flush=True)
    finally:
        srv.stop()
    (OUT / f"{label}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


def rates(result: dict) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in result["speed"]:
        by.setdefault(r["size"], []).append(r["prefilled"] / r["seconds"])
    return {k: statistics.median(v) for k, v in by.items()}


def cmp(base: str, label: str) -> int:
    a, b = (json.loads((OUT / f"{n}.json").read_text(encoding="utf-8")) for n in (base, label))
    ra, rb = rates(a), rates(b)
    for size in ra:
        if size in rb:
            print(f"prefill {size:>5}: {ra[size]:7.0f} -> {rb[size]:7.0f} tok/s ({rb[size] / ra[size]:.3f}x)")
    ob = {o["case"]: o for o in b["outputs"]}
    differ = 0
    for o in a["outputs"]:
        n = ob.get(o["case"])
        if n is None:
            continue
        same = o["text"] == n["text"]
        differ += not same
        at = next((i for i, (x, y) in enumerate(zip(o["text"], n["text"])) if x != y), min(len(o["text"]), len(n["text"])))
        print(f"output {o['case']:>12}: {'same' if same else f'DIFFERS at character {at} of {len(o["text"])}'}"
              f"   tok/verify {o['tokens_per_verify']:.2f} -> {n['tokens_per_verify']:.2f}")
    print("all outputs the same" if not differ else f"{differ} outputs differ")
    return 1 if differ else 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) >= 3 and args[0] == "cmp":
        return cmp(args[1], args[2])
    if len(args) < 2 or args[0] != "run":
        print(__doc__)
        return 2
    label, rest = args[1], args[2:]
    exe, cubin, quick, extra = ROOT / "target" / "release" / "inkling.exe", None, False, {}
    while rest:
        a = rest.pop(0)
        if a == "--exe":
            exe = Path(rest.pop(0)).resolve()
        elif a == "--cubin":
            cubin = Path(rest.pop(0)).resolve()
        elif a == "--quick":
            quick = True
        elif "=" in a:
            k, v = a.split("=", 1)
            extra[k] = v
        else:
            raise SystemExit(f"unknown argument {a}")
    run(label, exe, cubin, quick, extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
