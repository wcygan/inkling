# Prefill performance: where it stands and what is next

Written on 2026-11-25, with main at `f47d671`. Everything below was measured on one RTX 5090 (Windows 11, CUDA 13.5,
P1 memory clock) with `scripts/prefill_bench.py` and `inkling kld`. Raw results are in `logs/prefill-bench/` (not in
git).

## Where it stands

Cold prefill, tokens per second (medians):

| Prompt | Start of the day (`403798d`) | Now, default (`f47d671`) | With `INKLING_WIDE_MOE=bs` |
| --- | ---: | ---: | ---: |
| 19K | 7,094 | 8,758 | 11,121 |
| 64K | 6,197 | 7,544 | 9,211 |
| 128K | 5,287 | 6,276 | 7,390 |
| 256K | about 4,000 (65 s) | 4,650 (55 s) | not measured |

Follow-up turns of about 90 to 250 new tokens reach their first token 1.2 to 1.8 times sooner (`INKLING_WIDE_MIN` 64
instead of 256). Decode speed is unchanged all day (436 to 449 tok/s over the benchmark's decodes).

What changed, by commit:

| Commit | Change | Outputs |
| --- | --- | --- |
| `9bcb551` | Chunk-wide DSpark commit, controller clusters that run 16 tokens with the weights loaded once, local attention in 512-token launches with `prefill_attn_flash5`; `scripts/prefill_bench.py` | identical text |
| `ebe0961` | `inkling kld` | no change |
| `bfb3ab3` | Block-scaled FP4 expert kernel, opt-in (`INKLING_WIDE_MOE=bs`): activations quantized to NVFP4 | differ |
| `9b4609c` | Port of ingine's grouped expert GEMM, the default above 512-token chunks (`INKLING_WIDE_MOE=grouped` restores the old kernels) | differ (order of sums) |
| `f47d671` | `INKLING_WIDE_MIN` 256 -> 64; README numbers | differ on 64 to 255-token ranges (order of sums) |

Published figures of other engines, each on its own setup: ingine 10,900 tok/s at 20K, 9,600 at 64K, 8,100 at 131K,
6,200 at 254K; squid5090 16,855 at 19K (`fp8/fp8`); basalt 15,820 at 16K; kern 32K in 2.19 s and 128K in 12.46 s on an
RTX PRO 5000.

## What is known about accuracy

`inkling kld FILE PREFIX EVAL` compares next-token distributions with exact mode (BF16 weights and KV). On
`logs/prefill-bench/kld-corpus.txt` (this repository's Rust sources at `403798d`), wide prefill against exact mode:

| Sample | Expert kernel | Mean KLD | Median | Same top token | Same top token as token prefill |
| --- | --- | ---: | ---: | ---: | ---: |
| 30K prefix, 512 positions | grouped (old default) | 0.0095 | 0.00018 | 97.5% | 97.7% |
| | `wg` (default) | 0.0105 | 0.00026 | 97.5% | 98.6% |
| | `bs` (FP4) | 0.0104 | 0.00023 | 98.0% | 97.3% |
| 7K prefix, 2,048 positions | grouped | 0.094 | 0.00073 | 94.8% | 95.7% |
| | `wg` | 0.121 | 0.00076 | 95.2% | 96.3% |
| | `bs` | 0.126 | 0.00065 | 94.0% | 94.7% |

* The self-check (`INKLING_KLD_SELF=1`) is exactly zero.
* The test cannot yet separate these kernels: two paths with the same precisions already disagree on 2.5 to 4.3% of
  top tokens, and the 7K sample has a heavy tail (a few positions with KLD above 10, mostly indentation tokens).
* At very short context (prefix 100 to 200, 1,024 positions) the fp8 mode picks the same top token as exact mode in
  only 87% of positions, against 95 to 97% at 7K and 30K. Not investigated.
* The reference is this engine's own exact mode, not the PyTorch model.

## Next steps, in the order I would take them

1. **A better divergence corpus.** basalt uses 100 chunks of 512 tokens of wikitext-2 and gets numbers stable to three
   digits. Add prose and chat text and several thousand positions, and cache the reference logits (the exact pass is
   most of the run time), so that the choices below can be made on evidence.
2. **Native MXFP8 experts for layers 20 to 23.** The release ships those experts in MXFP8 and this engine repacks them
   to NVFP4. basalt measured the repack as worse (KLD 0.0043 against 0.0027 over 20 chunks). Blackwell has
   `mma.sync kind::mxf8f6f4` for it (squid5090 and TensorFold use it). Should help accuracy in decode and prefill.
3. **Decide the default for the FP4 expert kernel.** basalt's precision notes say the release marks NVFP4 activation
   quantization as required (its own serving runs W4A4); not checked against the model card. basalt also measured two
   FP4 terms per activation (KLD 0.0026) against one (0.0041).
4. **Dense projections on 8-bit tensor cores** (about 17% of a chunk). squid5090: 43 -> 16 ms per chunk with cuBLASLt
   native FP8; needs per-token activation quantization.
5. **Controllers** (about 12% of a chunk): a batched version; squid5090 got 1.8x by vectorizing. No precision change.
6. **Global attention at long context** (about 45% of a chunk at 124K). ingine and kern are well ahead at 128K.
7. **Host-RAM prefix caches** (squid5090, ingine) so that saved states stop costing 0.36 GB of VRAM each.
8. Small items: the ingine port's 32-row variant for chunks of 512 tokens or fewer; `prefill_bench.py`'s conversation
   turns stop at 120 tokens, inside the reasoning, so their replies are probably empty (use `reasoning_effort` "none"
   as `--small` does).

## How to measure

The repository's server must be stopped first (`just stop`); another session shares this GPU through
`logs/gpu-claim.txt` (session, start time, expected minutes; delete it when done; whoever stops the server restarts it).

```powershell
uv run --script scripts\prefill_bench.py run LABEL            # speed at 19K/64K/128K and 11 greedy outputs
uv run --script scripts\prefill_bench.py cmp BASE LABEL       # speed ratio, and whether each output is the same text
uv run --script scripts\prefill_bench.py run LABEL --small    # follow-up turns of 16 to 700 new tokens
uv run --script scripts\prefill_bench.py run LABEL --speed 1000:3,10000:3,32000:3,64000:3,128000:3,256000:3
target\release\inkling.exe kld logs\prefill-bench\kld-corpus.txt 30000 512   # about 4 minutes; 7000 2048 about 2.5
```

`run` takes `--exe`, `--cubin` and `KEY=VALUE` environment settings, so an experimental kernel can be built into a
scratch folder (`cargo oxide build --arch sm_120a` in `crates\kernel-build`, through `scripts\vsdev.cmd`) and compared
without touching `kernels\`. `just kernels` is needed before `just serve` whenever a kernel source changed.

Switches: `INKLING_WIDE_MOE` (`wg` default, `grouped`, `mma`, `bs`), `INKLING_WIDE_MIN` (64), `INKLING_LOCAL_SLICE`
(512), `INKLING_WIDE_COMMIT=0`, `INKLING_WIDE_CTRL_TB=0`.

Per-kernel timings came from Nsight Systems (`nsys launch --trace=cuda --cuda-graph-trace=node`, then `nsys start` /
`stop` around one request, `nsys export --type sqlite`, and a sum over `CUPTI_ACTIVITY_KIND_KERNEL`). Per 2,048-token
chunk at about 5K context before today's work: experts 119 ms, dense projections 41, controllers 37, local attention
31, norms and convolutions 20, drafter commit 14, global attention 13 (224 ms at 124K context).
