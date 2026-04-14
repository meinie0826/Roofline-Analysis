/**
 * FlashAttention Kernels - Main Entry Point
 * 
 * Provides unified interface for all stages
 */

#pragma once

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "stage0_naive.cuh"
#include "stage1_tiled.cuh"
#include "stage2_smem.cuh"
#include "stage3_mma.cuh"
#include "stage4_final.cuh"

namespace flash_attention {

using Element = cutlass::bfloat16_t;
using ElementAccum = float;

// Stage configurations
struct Stage0Naive {
    static constexpr int kBlockM = 1;
    static constexpr int kBlockN = 128;
    static constexpr int kHeadDim = 128;
    static constexpr const char* name = "Stage0-Naive";
    static constexpr const char* description = "Baseline: single thread per query";
};

struct Stage1Tiled {
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;
    static constexpr int kHeadDim = 128;
    static constexpr const char* name = "Stage1-Tiled";
    static constexpr const char* description = "Tiled computation with SMEM caching";
};

struct Stage2Smem {
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;
    static constexpr int kHeadDim = 128;
    static constexpr const char* name = "Stage2-Smem";
    static constexpr const char* description = "Optimized SMEM layout + vectorization";
};

struct Stage3MMA {
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;
    static constexpr int kHeadDim = 128;
    static constexpr const char* name = "Stage3-MMA";
    static constexpr const char* description = "Tensor Core MMA operations";
};

struct Stage4Final {
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 64;
    static constexpr int kHeadDim = 128;
    static constexpr const char* name = "Stage4-Final";
    static constexpr const char* description = "Online softmax + software pipelining";
};

// Kernel launcher
template <int Stage>
struct FlashAttentionKernel;

template <>
struct FlashAttentionKernel<0> {
    static void launch(
        const Element* Q, const Element* K, const Element* V, Element* O,
        int batch_size, int nheads, int seq_len, int head_dim,
        bool causal, cudaStream_t stream
    );
};

template <>
struct FlashAttentionKernel<1> {
    static void launch(
        const Element* Q, const Element* K, const Element* V, Element* O,
        int batch_size, int nheads, int seq_len, int head_dim,
        bool causal, cudaStream_t stream
    );
};

template <>
struct FlashAttentionKernel<2> {
    static void launch(
        const Element* Q, const Element* K, const Element* V, Element* O,
        int batch_size, int nheads, int seq_len, int head_dim,
        bool causal, cudaStream_t stream
    );
};

template <>
struct FlashAttentionKernel<3> {
    static void launch(
        const Element* Q, const Element* K, const Element* V, Element* O,
        int batch_size, int nheads, int seq_len, int head_dim,
        bool causal, cudaStream_t stream
    );
};

template <>
struct FlashAttentionKernel<4> {
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
