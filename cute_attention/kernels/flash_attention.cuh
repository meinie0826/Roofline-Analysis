/**
 * FlashAttention Kernels - Main Entry Point
 */

#pragma once

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace flash_attention {

using Element = cutlass::bfloat16_t;
using ElementAccum = float;

// Forward declarations
template <int Stage>
struct FlashAttentionKernel {
    static void launch(
        const Element* Q, const Element* K, const Element* V, Element* O,
        int batch_size, int nheads, int seq_len, int head_dim,
        bool causal, cudaStream_t stream
    );
};

// Unified interface
extern "C" {
    void launch_flash_attention(
        const void* Q, const void* K, const void* V, void* O,
        int batch_size, int nheads, int seq_len, int head_dim,
        int stage, bool causal, cudaStream_t stream
    );
}

} // namespace flash_attention
