/**
 * Stage 0: Naive FlashAttention Kernel (CuTe 版本)
 * 
 * 最基础的实现：
 * - 每个 CTA 处理一个 query position
 * - 直接从全局内存读取 Q, K, V
 * - 无 shared memory 优化
 * 
 * 性能目标：~0.5-1 TFLOPs/s
 */

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

using namespace cute;

template <typename Element, typename ElementAccum>
struct FlashAttentionNaive {
    using ElementMma = cutlass::bfloat16_t;
    
    // Tile sizes - 最小粒度
    static constexpr int kBlockM = 1;    // 每个 CTA 处理 1 个 query
    static constexpr int kBlockN = 128;  // 一次处理 128 个 keys
    static constexpr int kHeadDim = 128;
    
    // 每个 thread 处理部分 head dimension
    static constexpr int kThreads = 128;
    
    // Shared memory layout (无优化，直接访问 global)
    
    struct SharedStorage {
        // Stage 0 不使用 shared memory
        alignas(16) ElementAccum accum[kHeadDim];
    };
    
    static constexpr int SharedStorageSize = sizeof(SharedStorage);
    
    // Kernel
    template <typename TensorQ, typename TensorK, typename TensorV, typename TensorO>
    __device__ static void apply(
        TensorQ&& gQ,      // [seq_len, head_dim] for current batch/head
        TensorK&& gK,      // [seq_len, head_dim]
        TensorV&& gV,      // [seq_len, head_dim]
        TensorO&& gO,      // [seq_len, head_dim]
        int seq_len,
        float scale
    ) {
        // 每个 thread 处理一个 query position
        int m_idx = blockIdx.x;
        int tid = threadIdx.x;
        
        if (m_idx >= seq_len) return;
        
        // 加载 Q[m_idx, :] 到寄存器
        auto q_vec = make_tensor<Element>(Int<kHeadDim>{});
        for (int d = tid; d < kHeadDim; d += kThreads) {
            q_vec[d] = gQ(m_idx, d);
        }
        
        // Step 1: 计算 max(s_m)
        float max_score = -INFINITY;
        
        for (int n_start = 0; n_start < seq_len; n_start += kBlockN) {
            int n_end = min(n_start + kBlockN, seq_len);
            
            for (int n = n_start + tid; n < n_end; n += kThreads) {
                // 计算 Q[m] · K[n]
                float score = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    float q_val = static_cast<float>(q_vec[d]);
                    float k_val = static_cast<float>(gK(n, d));
                    score += q_val * k_val;
                }
                score *= scale;
                max_score = fmaxf(max_score, score);
            }
        }
        
        // Block reduce for max
        // (简化：假设单 thread 处理)
        __shared__ float shared_max;
        if (tid == 0) shared_max = max_score;
        __syncthreads();
        max_score = shared_max;
        
        // Step 2: 计算 sum(exp(s - max))
        float sum_exp = 0.0f;
        
        for (int n_start = 0; n_start < seq_len; n_start += kBlockN) {
            int n_end = min(n_start + kBlockN, seq_len);
            
            for (int n = n_start + tid; n < n_end; n += kThreads) {
                float score = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    float q_val = static_cast<float>(q_vec[d]);
                    float k_val = static_cast<float>(gK(n, d));
                    score += q_val * k_val;
                }
                score = expf(score * scale - max_score);
                sum_exp += score;
            }
        }
        
        __shared__ float shared_sum;
        if (tid == 0) shared_sum = sum_exp;
        __syncthreads();
        sum_exp = shared_sum;
        
        // Step 3: 计算 O[m, d] = sum_n(exp(s_mn - max) / sum_exp * V[n, d])
        for (int d = tid; d < kHeadDim; d += kThreads) {
            float o_val = 0.0f;
            
            for (int n = 0; n < seq_len; ++n) {
                float score = 0.0f;
                for (int dd = 0; dd < kHeadDim; ++dd) {
                    float q_val = static_cast<float>(q_vec[dd]);
                    float k_val = static_cast<float>(gK(n, dd));
                    score += q_val * k_val;
                }
                
                float attn = expf(score * scale - max_score) / sum_exp;
                float v_val = static_cast<float>(gV(n, d));
                o_val += attn * v_val;
            }
            
            gO(m_idx, d) = static_cast<Element>(o_val);
        }
    }
};

// Launcher
template <typename Element>
void launch_naive_attention(
    const Element* Q, const Element* K, const Element* V, Element* O,
    int batch_size, int nheads, int seq_len, int head_dim,
    cudaStream_t stream
);
