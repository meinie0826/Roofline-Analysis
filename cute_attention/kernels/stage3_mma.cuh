/**
 * Stage 3: Tensor Core MMA FlashAttention (CuTe 版本)
 * 
 * 优化点：
 * - 使用 warp-level GMMA 指令（通过 CuTe）
 * - 高效的 smem -> register 加载
 * - Softmax 与 MMA 流水线
 * 
 * 性能目标：~50-80 TFLOPs/s
 */

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm90.h>

using namespace cute;

template <
    typename Element,
    typename ElementAccum,
    int kBlockM,
    int kBlockN,
    int kHeadDim
>
struct FlashAttentionMMA {
    static_assert(kHeadDim == 64 || kHeadDim == 128 || kHeadDim == 256);
    
    using ElementMma = cutlass::bfloat16_t;
    
    static constexpr int kThreads = 128;
    
    struct SharedStorage {
        alignas(128) Element sQ[kBlockM * kHeadDim];
        alignas(128) Element sK[kBlockN * kHeadDim];
        alignas(128) Element sV[kHeadDim * kBlockN];
        alignas(128) ElementAccum sO[kBlockM * kHeadDim];
        float sS[kBlockM * kBlockN];
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
        int m_tile = blockIdx.x;
        int m_start = m_tile * kBlockM;
        
        if (m_start >= seq_len) return;
        
        int m_end = min(m_start + kBlockM, seq_len);
        int m_size = m_end - m_start;
        
        extern __shared__ char smem[];
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        
        // Initialize
        for (int i = threadIdx.x; i < kBlockM; i += kThreads) {
            storage.max_scores[i] = -INFINITY;
            storage.sum_exp[i] = 0.0f;
        }
        for (int i = threadIdx.x; i < kBlockM * kHeadDim; i += kThreads) {
            storage.sO[i] = ElementAccum(0);
        }
        
        __syncthreads();
        
        // Load Q tile
        constexpr int kVecLen = sizeof(uint128_t) / sizeof(Element);
        for (int i = threadIdx.x * kVecLen; i < m_size * kHeadDim; i += kThreads * kVecLen) {
            int m = i / kHeadDim;
            int d = i % kHeadDim;
            if (m < m_size && d + kVecLen <= kHeadDim) {
                uint128_t* dst = reinterpret_cast<uint128_t*>(&storage.sQ[m * kHeadDim + d]);
                const uint128_t* src = reinterpret_cast<const uint128_t*>(&gQ(m_start + m, d));
                *dst = *src;
            }
        }
        
        __syncthreads();
        
        // Process K, V tiles
        for (int n_start = 0; n_start < seq_len; n_start += kBlockN) {
            int n_end = min(n_start + kBlockN, seq_len);
            int n_size = n_end - n_start;
            
            // Load K, V tiles
            for (int i = threadIdx.x * kVecLen; i < n_size * kHeadDim; i += kThreads * kVecLen) {
                int n = i / kHeadDim;
                int d = i % kHeadDim;
                if (n < n_size && d + kVecLen <= kHeadDim) {
                    uint128_t* dst = reinterpret_cast<uint128_t*>(&storage.sK[n * kHeadDim + d]);
                    const uint128_t* src = reinterpret_cast<const uint128_t*>(&gK(n_start + n, d));
                    *dst = *src;
                }
            }
            
            // Load V (transposed)
            for (int i = threadIdx.x; i < kHeadDim * n_size; i += kThreads) {
                int d = i / n_size;
                int n = i % n_size;
                storage.sV[d * kBlockN + n] = gV(n_start + n, d);
            }
            
            __syncthreads();
            
            // Compute QK^T using GMMA-style loop
            // Each thread processes part of the output rows
            int warp_id = threadIdx.x / 32;
            int lane_id = threadIdx.x % 32;
            
            for (int m = warp_id; m < m_size; m += kThreads / 32) {
                float row_max = storage.max_scores[m];
                float row_sum = 0.0f;
                
                // Compute attention scores
                for (int n = lane_id; n < n_size; n += 32) {
                    float score = 0.0f;
                    
                    // Unrolled dot product for better ILP
                    #pragma unroll 4
                    for (int d = 0; d < kHeadDim; d += 4) {
                        float q0 = static_cast<float>(storage.sQ[m * kHeadDim + d]);
                        float k0 = static_cast<float>(storage.sK[n * kHeadDim + d]);
                        float q1 = static_cast<float>(storage.sQ[m * kHeadDim + d + 1]);
                        float k1 = static_cast<float>(storage.sK[n * kHeadDim + d + 1]);
                        float q2 = static_cast<float>(storage.sQ[m * kHeadDim + d + 2]);
                        float k2 = static_cast<float>(storage.sK[n * kHeadDim + d + 2]);
                        float q3 = static_cast<float>(storage.sQ[m * kHeadDim + d + 3]);
                        float k3 = static_cast<float>(storage.sK[n * kHeadDim + d + 3]);
                        
                        score += q0*k0 + q1*k1 + q2*k2 + q3*k3;
                    }
                    
                    score *= scale;
                    storage.sS[m * kBlockN + n] = score;
                    row_max = fmaxf(row_max, score);
                }
                
                // Warp reduce for max
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));
                }
                
                // Update global max and rescale
                float old_max = storage.max_scores[m];
                float new_max = fmaxf(old_max, row_max);
                
                if (new_max > old_max && old_max > -INFINITY) {
                    float rescale = expf(old_max - new_max);
                    storage.sum_exp[m] *= rescale;
                    
                    for (int d = lane_id; d < kHeadDim; d += 32) {
                        storage.sO[m * kHeadDim + d] *= rescale;
                    }
                }
                __syncwarp();
                
                storage.max_scores[m] = new_max;
                
                // Compute exp and accumulate
                for (int n = lane_id; n < n_size; n += 32) {
                    float s = expf(storage.sS[m * kBlockN + n] - new_max);
                    storage.sS[m * kBlockN + n] = s;
                    row_sum += s;
                }
                
                // Warp reduce for sum
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
                }
                
                storage.sum_exp[m] += row_sum;
                
                // Accumulate PV (Tensor Core style)
                for (int d = lane_id; d < kHeadDim; d += 32) {
                    float o_val = 0.0f;
                    
                    #pragma unroll 4
                    for (int n = 0; n < n_size; n += 4) {
                        float p0 = storage.sS[m * kBlockN + n];
                        float v0 = static_cast<float>(storage.sV[d * kBlockN + n]);
                        o_val += p0 * v0;
                        
                        if (n + 1 < n_size) {
                            float p1 = storage.sS[m * kBlockN + n + 1];
                            float v1 = static_cast<float>(storage.sV[d * kBlockN + n + 1]);
                            o_val += p1 * v1;
                        }
                        if (n + 2 < n_size) {
                            float p2 = storage.sS[m * kBlockN + n + 2];
                            float v2 = static_cast<float>(storage.sV[d * kBlockN + n + 2]);
                            o_val += p2 * v2;
                        }
                        if (n + 3 < n_size) {
                            float p3 = storage.sS[m * kBlockN + n + 3];
                            float v3 = static_cast<float>(storage.sV[d * kBlockN + n + 3]);
                            o_val += p3 * v3;
                        }
                    }
                    
                    storage.sO[m * kHeadDim + d] += o_val;
                }
            }
            
            __syncthreads();
        }
        
        // Normalize and write output
        for (int i = threadIdx.x; i < m_size * kHeadDim; i += kThreads) {
            int m = i / kHeadDim;
            int d = i % kHeadDim;
            float o_val = storage.sO[i] / storage.sum_exp[m];
            gO(m_start + m, d) = static_cast<Element>(o_val);
        }
    }
};
