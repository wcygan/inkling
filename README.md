# inkling

A native Windows inference engine and OpenAI-compatible server for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini),
written in Rust with GPU kernels in Rust ([cuda-oxide](https://github.com/NVIDIA/cuda-rust)). It is built for one
GPU, the **NVIDIA RTX 5090** (Blackwell, `sm_120a`), and runs the model with:

* **FP8 dense attention and NVFP4 experts**, so the 20 GB checkpoint fits next to a long context in 32 GB;
* **DSpark speculative decoding** with an adaptive verify width (2, 4 or 8 tokens per step);
* **Wide prefill**, a batched prompt path for long prompts, with chunks up to 2,048 tokens;
* an **FP8 KV cache**, which makes a context of up to **256K tokens** fit in about 22.5 GiB.

`inkling serve` speaks the OpenAI chat API (streaming, reasoning, tool calls) and reuses the cache across turns,
so it works as the backend for coding agents such as [Pi](https://www.npmjs.com/package/@mariozechner/pi-coding-agent).

## Start here

On a new machine the only thing to install by hand is `uv`. Everything else is checked for you:

```powershell
winget install --id astral-sh.uv -e
setup.cmd                 # or: uv run --script scripts\inkling.py doctor
```

`doctor` checks every tool the project needs (the NVIDIA driver, CUDA 13.5, the Visual Studio C++ tools, Rust, LLVM,
libffi, git, free disk space), says which step needs each one, and prints the `winget` command for every tool that is
missing, in the order to run them. It installs nothing itself. Run the commands it lists, open a **new** terminal so
the installs are on `PATH`, and run it again until it says everything is installed. `doctor --for run` checks only
what running the server needs, and the other commands refuse with the same recommendation when something they need
is missing.

## Quick start

```powershell
just doctor          # is everything installed? how to install what is missing
just status          # what is ready, and the next step
just weights         # download the checkpoint (about 20 GB) and verify it
just kernels         # build the GPU kernels from source (the first run fetches cuda-oxide)
just build           # build the engine
just serve           # start the server on http://127.0.0.1:8013/v1
```

Without `just`, run `uv run --script scripts\inkling.py <command>` (`status`, `weights`, `check-weights`,
`doctor`, `kernels`, `build`, `serve`, `stop`). `status` always tells you what to run next, and every command that cannot run
yet says why and what to do, so you can run them in any order and the tool will tell you.

## Requirements

`just doctor` checks all of this and tells you how to install what is missing.

| | |
| --- | --- |
| GPU | NVIDIA RTX 5090. The kernels are compiled for `sm_120a`; other GPUs are refused. About 22.5 GiB of VRAM is used at 256K context |
| OS | Windows 11 x64 |
| Driver and CUDA | A current NVIDIA driver and the CUDA 13.5 Toolkit. `cublas64_13.dll` (in the Toolkit's `bin\x64`) must be on `PATH`, and the kernel build needs `nvcc` |
| Tools | [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`; it runs the scripts and brings its own Python), [rustup](https://rustup.rs) (the pinned Rust 1.99.1 installs itself), git, and [just](https://github.com/casey/just) (`winget install Casey.Just`, optional) |
| Build tools | Visual Studio 2022 or its Build Tools with the C++ workload; LLVM (`winget install LLVM.LLVM`); vcpkg with `libffi:x64-windows`. These are needed only for `just kernels` |
| Disk | About 20 GB for the weights |

## Get the weights

The weights are not part of this repository. `just weights` downloads the original checkpoint from Hugging Face,
[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0), at the
revision this project was built against (`7a67ae23`), into `%USERPROFILE%\Models\Inkling-2-Mini`, and then checks
every file against the hashes Hugging Face publishes for that revision. The same download by hand:

```powershell
uvx --from huggingface_hub hf download thinkingmachines/Inkling-2-Mini `
    --revision 7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6 `
    --local-dir "$env:USERPROFILE\Models\Inkling-2-Mini"
just check-weights
```

The download is about 20 GB: `model-00001-of-00001.safetensors` (19.5 GB), `mtp.safetensors` (0.6 GB, the
speculative drafter), the tokenizer and chat template, and the config files. An interrupted download resumes when
you run the command again. If the repository asks you to accept its terms first, run
`uvx --from huggingface_hub hf auth login` and try again. The engine reads the safetensors as published; nothing is
converted.

To keep the checkpoint elsewhere, set `INKLING_MODEL_DIR`, set `model_dir` in `serve.config.json`, or pass
`--model-dir`.

## Build

```powershell
just kernels         # the two GPU kernel files, into kernels\
just build           # target\release\inkling.exe
```

The host build takes under a minute. `just kernels` takes a few minutes the first time and under a minute after
that. It:

1. clones [cuda-oxide](https://github.com/NVIDIA/cuda-rust) into `third_party\cuda-rust` at the pinned commit
   (`5e5c77b19a4b3204d6542bab2f0fde2a4eed824b`) and applies `patches\cuda-rust-windows-lean.patch`, a small Windows
   fix;
2. installs `cargo-oxide` and builds its codegen backend (`cargo oxide setup`), if that has not been done;
3. compiles `crates\inkling-kernels` for `sm_120a`, and the one CUDA C++ kernel with `nvcc`;
4. records the hash of the sources and of both files in `kernels\build-stamp.json`, last.

It first checks that git, Rust, uv, the CUDA Toolkit, LLVM, libffi and Visual Studio are present and lists any that
are missing. [`scripts\vsdev.cmd`](scripts/vsdev.cmd) sets up the compiler environment; it assumes the default
install locations, and `CUDA_PATH`, `LIBCLANG_PATH` and `VCPKG_ROOT` override them. The kernels are reproducible:
the same sources give the same bytes (see [`kernels/README.md`](kernels/README.md)). If the build cannot find
`lake.exe`, that is the case the patch fixes.

## Run the server

```powershell
just serve
```

The server loads the weights (about 7 seconds with the files in the disk cache, longer on a cold read), then
listens on `http://127.0.0.1:8013/v1`. `GET /health` returns 503 until it is ready. Starting `serve` again stops this
repository's running server first, so there is only ever one. Ctrl+C stops it. `just stop` stops it from another
terminal, and `just serve --dry-run` runs every check and prints the command without starting anything.

Settings are in [`serve.config.json`](serve.config.json) (an unknown or mistyped setting is an error), and any of
them can be overridden for one run:

```powershell
just serve --max-ctx 131072 --checkpoints 4
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_ctx` | 262144 | Context length in tokens. The KV cache is allocated up front, so a smaller value frees VRAM |
| `checkpoints` | 6 | Saved cache states for turn reuse, about 0.3 GB each |
| `effort` | `low` | Default reasoning effort (`reasoning_effort` in a request overrides it) |
| `port` | 8013 | Listen port, on 127.0.0.1 only |
| `kv_fp8`, `draft_kv_fp8` | true | FP8 KV cache for the model and for the drafter |
| `model_dir` | (empty) | Checkpoint folder; empty means `INKLING_MODEL_DIR`, then `%USERPROFILE%\Models\Inkling-2-Mini` |
| `pi_models_json` | (empty) | Path of a Pi `models.json` whose `inkling` provider should follow `port` and `max_ctx` |

`serve` starts only when the checkpoint is verified, the kernels match the current sources, the GPU is an RTX 5090
class card (compute capability 12.0), and the port is free or held by this repository's own server. It never stops
another program's server. [docs/lifecycle](docs/lifecycle/README.md) explains these checks and the model that was
used to design them.

Try it:

```powershell
curl http://127.0.0.1:8013/health
curl http://127.0.0.1:8013/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "max_tokens": 200
}'
```

What the server supports:

* **API:** `/v1/chat/completions` with streaming, `reasoning_content`, tool calls whose arguments stream as deltas,
  `tool_choice` set to `auto` or `none`, and `reasoning_effort`. Also `/v1/models`, `/health`, `/metrics`
  (Prometheus) and `/apply-template`. Text only. The server does not check API keys.
* **Turn reuse:** a new request extends the live cache when it continues the last conversation, and otherwise restores
  the nearest saved checkpoint (taken at the end of every prompt and at the end of the system and tools header).
* **Limits:** one request runs at a time with up to four waiting, then it returns 503. A prompt over `max_ctx` returns
  `context_length_exceeded`. A request is cut off after 600 seconds (`INKLING_MAX_REQUEST_S`).
* **Logging:** one JSON line per request, with timings and cache reuse, goes to `logs\serve-log.jsonl`.
* **Memory:** after loading, the server frees the 7 GB of BF16 weights that the fp8 path never reads
  (`INKLING_KEEP_BF16=1` keeps them).

### Use it with Pi

```powershell
npm i -g @mariozechner/pi-coding-agent
```

Add an `inkling` provider to `~\.pi\agent\models.json`, with `contextWindow` equal to `max_ctx`:

```json
{
  "providers": {
    "inkling": {
      "baseUrl": "http://127.0.0.1:8013/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsStore": false,
        "supportsStrictMode": false,
        "supportsReasoningEffort": true,
        "maxTokensField": "max_tokens"
      },
      "models": [{
        "id": "inkling-2-mini",
        "name": "Inkling 2 Mini (local RTX 5090)",
        "reasoning": true,
        "thinkingLevelMap": { "off": "none", "minimal": "none", "xhigh": "xhigh" },
        "input": ["text"],
        "contextWindow": 262144,
        "maxTokens": 16384
      }]
    }
  }
}
```

Then run `pi --model inkling/inkling-2-mini` in any project folder, or `just pi`. Set `pi_models_json` in
`serve.config.json` to that file, and `just serve` keeps its port and `contextWindow` in step with the server. Any
other OpenAI-compatible client works with the same base URL.

## Other commands

```text
inkling load-check [--model-dir DIR]     load the weights and report time and VRAM
inkling tokenize TEXT                    render TEXT with the chat template and print the token IDs
inkling generate [--max-tokens N] TEXT   greedy generation for one user turn
inkling bench [--mode fp8] [RUNS]        decode speed on a chat and a code prompt
```

Run these as `target\release\inkling.exe ...`; they need the kernels built. Useful environment variables for the
server: `INKLING_PORT`, `INKLING_CHECKPOINTS`, `INKLING_KV_FP8`, `INKLING_DRAFT_KV_FP8`, `INKLING_WIDE_PREFILL=0`
(turn wide prefill off), `INKLING_WIDE_MIN` (the shortest prompt that uses it, 256 tokens by default) and
`INKLING_WIDE_MAX=1024` (smaller prefill chunks, which use less VRAM). `scripts\inkling.py serve` sets the first
four from the settings above.

## Why it is specific to the RTX 5090

* **Blackwell instructions.** The kernels use FP8 (E4M3) and NVFP4 (E2M1) tensor-core paths, Programmatic Dependent
  Launch to overlap one kernel's start-up with the previous kernel's tail, and `cp.async` pipelines. They are compiled
  for `sm_120a` only.
* **Sized for 170 SMs.** Launch grids, tile shapes and the fused attention groups are chosen so that the work fills the
  5090's 170 streaming multiprocessors.
* **Memory-bound decode.** Decoding is limited by memory bandwidth (about 1.7 TB/s on this card), so the design counts
  bytes: FP8 and NVFP4 weights, an FP8 KV cache and weight prefetch.
* **A 32 GB budget.** The memory plan, which frees weights the fp8 path never reads, stores the KV cache in FP8 and
  sizes the checkpoints, is what lets a 256K context fit next to the model.
* **Windows.** Under WDDM, launching many small kernels from the host is slow, so every decode step is captured once
  as a CUDA graph and replayed.

## Layout

| Path | Contents |
| --- | --- |
| `crates/inkling-cli` | The `inkling` binary and the server (`server.rs`) |
| `crates/inkling-engine` | The decode and prefill engine: step graphs, speculative decoding, FP8 and NVFP4 weights |
| `crates/inkling-model` | Config, safetensors and weight loading |
| `crates/inkling-runtime` | CUDA driver bindings: streams, graphs, memory arenas, module loading |
| `crates/inkling-text` | Tokenizer and chat template |
| `crates/inkling-kernels` | The GPU kernels (cuda-oxide Rust, and one CUDA C++ file) |
| `crates/kernel-build` | Build driver for the kernels |
| `kernels/` | Where `just kernels` writes the built kernels |
| `assets/` | The draft-vocabulary shortlist for the speculative drafter |
| `scripts/` | `inkling.py` (every command above), `vsdev.cmd`, `build_draft_vocab.py` |
| `docs/lifecycle/` | The rules the commands follow, as a TLA+ model with its results and a chart |

## License and credits

Apache-2.0, see [LICENSE](LICENSE). The kernels and launch order were ported from
[squid5090](https://github.com/tensorfish/squid5090), and the design follows
[ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine); see [NOTICE](NOTICE) for the full
credits. The model is Inkling 2 Mini by Thinking Machines (Apache-2.0). CUDA components keep NVIDIA's license terms.
