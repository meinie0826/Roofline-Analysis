/**
 * Stage 1: Tiled FlashAttention Kernel (CuTe 版本)
 * 
 * 优化点：
 * - 每个 CTA 处理一个 tile of queries (kBlockM positions)
 * - Shared memory 缓存 Q, K tiles
 * - Block-level reduction for softmax
 * 
 * 性能目标：~2-4 TFLOPs/s
 */

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

using namespace cute;

template <typename Element, typename ElementAccum, int kBlockM, int kBlockN, int kHeadDim>
struct FlashAttentionTiled {
    static_assert(kHeadDim % 32 == 0, "Head dimension must be multiple of 32");
    
    using ElementMma = cutlass::bfloat16_t;
    
    // Thread counts
    static constexpr int kThreads = 128;
    
    // Shared memory for Q, K tiles
    struct SharedStorage {
        alignas(128) Element sQ[kBlockM * kHeadDim];
        alignas(128) Element sK[kBlockN * kHeadDim];
        alignas(128) Element sV[kHeadDim * kBlockN];  // Transposed for coalescing
        alignas(128) ElementAccum sO[kBlockM * kHeadDim];
        alignas(128) float sScores[kBlockM * kBlockN];
        float max_scores[kBlockM];
        float sum_exp[kBlockM];
    };
    
    template <typename TensorQ, typename TensorK, typename TensorV, typename TensorO>
    __device__ static void apply(
        TensorQ&& gQ,
        TensorK&& gK,
        TensorV&& gV,
        TensorO&& gO,
        int seq_len,
        float scale
    ) {
        int tid = threadIdx.x;
        int m_tile = blockIdx.x;
        int m_start = m_tile * kBlockM;
        
        if (m_start >= seq_len) return;
        
        int m_end = min(m_start + kBlockM, seq_len);
        int m_size = m_end - m_start;
        
        extern __shared__ char smem[];
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        
        // Initialize output accumulator
        for (int i = tid; i < kBlockM * kHeadDim; i += kThreads) {
            storage.sO[i] = ElementAccum(0);
        }
        
        // Initialize softmax state
        for (int i = tid; i < kBlockM; i += kThreads) {
            storage.max_scores[i] = -INFINITY;
            storage.sum_exp[i] = 0.0f;
        }
        
        __syncthreads();
        
        // Load Q tile to shared memory
        // Each thread loads multiple elements
        for (int i = tid; i < m_size * kHeadDim; i += kThreads) {
            int m = i / kHeadDim;
            int d = i % kHeadDim;
            storage.sQ[i] = gQ(m_start + m, d);
        }
        
        __syncthreads();
        
        // Iterate over K, V tiles
        for (int n_start = 0; n_start < seq_len; n_start += kBlockN) {
            int n_end = min(n_start + kBlockN, seq_len);
            int n_size = n_end - n_start;
            
            // Load K tile
            for (int i = tid; i < n_size * kHeadDim; i += kThreads) {
                int n = i / kHeadDim;
                int d = i % kHeadDim;
                storage.sK[i] = gK(n_start + n, d);
            }
            
            // Load V tile (transposed)
            for (int i = tid; i < n_size * kHeadDim; i += kThreads) {
                int n = i / kHeadDim;
                int d = i % kHeadDim;
                storage.sV[d * kBlockN + n] = gV(n_start + n, d);
            }
            
            __syncthreads();
            
            // Compute QK^T and online softmax
            // Each thread computes one row of S
            for (int m = tid; m < m_size; m += kThreads) {
                float row_max = storage.max_scores[m];
                
                // Compute attention scores for this row
                for (int n = 0; n < n_size; ++n) {
                    float score = 0.0f;
                    for (int d = 0; d < kHeadDim; ++d) {
                        float q = static_cast<float>(storage.sQ[m * kHeadDim + d]);
                        float k = static_cast<float>(storage.sK[n * kHeadDim + d]);
                        score += q * k;
                    }
                    score *= scale;
                    storage.sS[m * kBlockN + n] = score;
                    row_max = fmaxf(row_max, score);
                }
                
                // Update global max
                float old_max = storage.max_scores[m];
                float new_max = row_max;
                
                // Rescale previous accumulator
                if (new_max > old_max) {
                    float rescale = expf(old_max - new_max);
                    storage.sum_exp[m] *= rescale;
                    
                    for (int d = 0; d < kHeadDim; ++d) {
                        storage.sO[m * kHeadDim + d] *= rescale;
                    }
                }
                
                storage.max_scores[m] = new_max;
            }
            
            __syncthreads();
            
            // Compute exp and accumulate
            for (int m = tid; m < m_size; m += kThreads) {
                float row_max = storage.max_scores[m];
                float row_sum = 0.0f;
                
                for (int n = 0; n < n_size; ++n) {
                    float s = expf(storage.sS[m * kBlockN + n] - row_max);
                    storage.sS[m * kBlockN + n] = s;  // Store for later use
                    row_sum += s;
                }
                
                // Accumulate output
                for (int d = 0; d < kHeadDim; ++d) {
                    float o_val = 0.0f;
                    for (int n = 0; n < n_size; ++n) {
                        float v = static_cast<float>(storage.sV[d * kBlockN + n]);
                        o_val += storage.sS[m * kBlockN + n] * v;
                    }
                    storage.sO[m * kHeadDim + d] += o_val;
                }
                
                storage.sum_exp[m] += row_sum;
            }
            
            __syncthreads();
        }
        
        // Normalize and write output
        for (int i = tid; i < m_size * kHeadDim; i += kThreads) {
            int m = i / kHeadDim;
            int d = i % kHeadDim;
            float o_val = storage.sO[i] / storage.sum_exp[m];
            gO(m_start + m, d) = static_cast<Element>(o_val);
        }
    }
};
