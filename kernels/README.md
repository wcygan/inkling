# GPU kernels

This folder is where the two GPU kernel files the engine loads are built. They are not stored in the repository;
build them once with:

```powershell
just kernels        # or: uv run --script scripts\inkling.py kernels
```

| File | Source | Compiled with |
| --- | --- | --- |
| `inkling_kernels.cubin` | `crates/inkling-kernels` (Rust) | cuda-oxide: `cargo oxide build --arch sm_120a` |
| `z_adaptive_select_width.cubin` | `crates/inkling-kernels/cuda/z_adaptive_select_width.cu` | `nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -cubin` |
| `build-stamp.json` | written last by `just kernels` | the hash of the sources and the hashes of both files |

The second kernel stays in CUDA C++ because it calls `cudaGraphSetConditional`, which cuda-oxide does not expose.

`build-stamp.json` is what marks the kernels fresh. It is deleted when a build starts and written only after both
files are complete and the sources did not change meanwhile, so an interrupted or out-of-date build can never be
used by `serve`. After you edit a kernel source, `just serve` refuses until you run `just kernels` again.

The build is reproducible: from the same sources and the pinned cuda-oxide commit, `inkling_kernels.cubin` has the
SHA-256 `033558d235ae907be72e13967662e4da3a75f3fb046520666a8ee70898a0cf04` and `z_adaptive_select_width.cubin`
starts with `cd0675a184345ad8`.
