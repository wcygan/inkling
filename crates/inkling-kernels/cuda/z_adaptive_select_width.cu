// z_adaptive_select_width: the one DSpark kernel that stays CUDA C++.
//
// It ends by calling the device runtime's cudaGraphSetConditional, which sets
// the value of the switch node that picks the verify width. nvcc lowers it to
// a call to `.extern .func cudaGraphSetConditional` that ptxas resolves in a
// whole-program cubin. cuda-oxide cannot emit that call: its LLVM verifier
// rejects the unknown external symbol, and inline PTX cannot declare a
// module-scope `.extern .func`.
//
// Body copied from squid5090 @ 3ee093c cuda/cuda_abi.cu:226
// (z_adaptive_select_width_kernel), with an unmangled extern "C" name.
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -cubin
// (no -rdc needed). Launch: grid 1, 1 thread (squid5090 uses <<<1, 1>>>),
// captured as the first node of the adaptive graph whose 3-way switch node
// owns `cond_handle` (cudaGraphConditionalHandleCreate on the root graph).

#include <cstdint>
#include <cuda_runtime.h>

constexpr int MTP_MAX_DRAFTS = 7;

extern "C" __global__ void z_adaptive_select_width(
    const int32_t* __restrict__ d_anchor,
    const int32_t* __restrict__ d_drafts,
    const float* __restrict__ d_draft_conf,
    float threshold,
    int32_t* __restrict__ d_tokens,
    int32_t* __restrict__ d_chosen_width,
    cudaGraphConditionalHandle cond_handle) {
    if (threadIdx.x != 0) return;
    int n_conf = 0;
    while (n_conf < MTP_MAX_DRAFTS && d_draft_conf[n_conf] >= threshold) {
        ++n_conf;
    }
    unsigned int case_idx = (n_conf <= 1) ? 0u : ((n_conf <= 3) ? 1u : 2u);
    int width = (case_idx == 0u) ? 2 : ((case_idx == 1u) ? 4 : 8);
    d_tokens[0] = *d_anchor;
    for (int i = 0; i < width - 1; ++i) d_tokens[1 + i] = d_drafts[i];
    *d_chosen_width = width;
    cudaGraphSetConditional(cond_handle, case_idx);
}
