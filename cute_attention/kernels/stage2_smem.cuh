/**
 * Stage 2: Shared Memory Optimized FlashAttention (CuTe 版本)
 * 
 * 优化点：
 * - 更高效的 SMEM 布局（避免 bank conflict）
 * - 向量化加载/存储
 * - Warp-level reduction
 * 
 * 性能目标：~8-15 TFLOPs/s
 */

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/gemm.h>

using namespace cute;

template <typename Element, typename ElementAccum, int kBlockM, int kBlockN, int kHeadDim>
struct FlashAttentionSmem {
    static_assert(kHeadDim % 64 == 0, "Head dimension must be multiple of 64");
    
    using ElementMma = cutlass::bfloat16_t;
    
    // Increased thread count for better parallelism
    static constexpr int kThreads = 128;
    
    // Warp-level MMA tile sizes
    static constexpr int kWarpM = 16;
    static constexpr int kWarpN = 16;
    static constexpr int kWarpK = 16;
    
    // Shared memory with padding to avoid bank conflicts
    static constexpr int kPadding = 4;  // Padding elements
    
    struct SharedStorage {
        // Q: [kBlockM, kHeadDim] - padded
        alignas(128) Element sQ[kBlockM * (kHeadDim + kPadding)];
        
        // K: [kBlockN, kHeadDim] - padded
        alignas(128) Element sK[kBlockN * (kHeadDim + kPadding)];
        
        // V: [kHeadDim, kBlockN] - transposed, padded
        alignas(128) Element sV[kHeadDim * (kBlockN + kPadding)];
        
        // Score accumulator
        alignas(128) float sS[kBlockM * kBlockN];
        
        // Output accumulator
        alignas(128) ElementAccum sO[kBlockM * (kHeadDim + kPadding)];
        
        // Softmax state
        float max_scores[kBlockM];
        float sum_exp[kBlockM];
    };
    
    // Vectorized load types
    using LoadVec = uint128_t;  // 128-bit loads (8 BF16 elements)
    
    // CuTe layouts for efficient access
    using LayoutQ = decltype(make_layout(
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        Stride<Int<kHeadDim + kPadding>, _1>{}
    ));
    
    using LayoutK = decltype(make_layout(
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        Stride<Int<kHeadDim + kPadding>, _1>{}
    ));
    
    using LayoutV = decltype(make_layout(
        Shape<Int<kHeadDim>, Int<kBlockN>>{},
        Stride<Int<kBlockN + kPadding>, _1>{}
    ));
    
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
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        
        int m_tile = blockIdx.x;
        int m_start = m_tile * kBlockM;
        
        if (m_start >= seq_len) return;
        
        int m_end = min(m_start + kBlockM, seq_len);
        int m_size = m_end - m_start;
        
        extern __shared__ char smem[];
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        
        // Initialize output accumulator
        #pragma unroll
        for (int i = tid; i < kBlockM * (kHeadDim + kPadding); i += kThreads) {
            storage.sO[i] = ElementAccum(0);
        }
        
        // Initialize softmax state
        #pragma unroll
        for (int i = tid; i < kBlockM; i += kThreads) {
            storage.max_scores[i] = -INFINITY;
            storage.sum_exp[i] = 0.0f;
        }
        
        __syncthreads();
        
        // Vectorized Q load
        constexpr int kVecLen = sizeof(LoadVec) / sizeof(Element);
        const int kQElements = m_size * kHeadDim;
        
        for (int i = tid * kVecLen; i < kQElements; i += kThreads * kVecLen) {
            int m = i / kHeadDim;
            int d = i % kHeadDim;
            int smem_idx = m * (kHeadDim + kPadding) + d;
            
            if (m < m_size) {
                LoadVec* gmem_ptr = reinterpret_cast<LoadVec*>(reinterpret_cast<Element*>(&gQ(m_start + m, d)));
                LoadVec* smem_ptr = reinterpret_cast<LoadVec*>(&storage.sQ[smem_idx]);
                *smem_ptr = *gmem_ptr;
            }
        }
        
        __syncthreads();
        
        // Process K, V tiles
        for (int n_start = 0; n_start < seq_len; n_start += kBlockN) {
            int n_end = min(n_start + kBlockN, seq_len);
            int n_size = n_end - n_start;
            
            // Vectorized K load
            const int kKElements = n_size * kHeadDim;
            for (int i = tid * kVecLen; i < kKElements; i += kThreads * kVecLen) {
                int n = i / kHeadDim;
                int d = i % kHeadDim;
                int smem_idx = n * (kHeadDim + kPadding) + d;
                
                LoadVec* gmem_ptr = reinterpret_cast<LoadVec*>(reinterpret_cast<Element*>(&gK(n_start + n, d)));
                LoadVec* smem_ptr = reinterpret_cast<LoadVec*>(&storage.sK[smem_idx]);
                *smem_ptr = *gmem_ptr;
            }
            
            // Vectorized V load (transposed)
            const int kVElements = n_size * kHeadDim;
            for (int i = tid * kVecLen; i < kVElements; i += kThreads * kVecLen) {
                int n = i / kHeadDim;
                int d = i % kHeadDim;
                int smem_idx = d * (kBlockN + kPadding) + n;
                
                Element v_val = gV(n_start + n, d);
                storage.sV[smem_idx] = v_val;
            }
            
            __syncthreads();
            
            // Compute QK^T with better parallelism
            // Each warp processes part of the output rows
            for (int m = warp_id; m < m_size; m += kThreads / 32) {
                float row_max = storage.max_scores[m];
                float row_sum = 0.0f;
                
                // Compute attention scores
                for (int n = lane_id; n < n_size; n += 32) {
                    float score = 0.0f;
                    
                    // Unrolled dot product
                    #pragma unroll 4
                    for (int d = 0; d < kHeadDim; d += 4) {
                        int q_idx = m * (kHeadDim + kPadding) + d;
                        int k_idx = n * (kHeadDim + kPadding) + d;
                        
                        float q0 = static_cast<float>(storage.sQ[q_idx]);
                        float k0 = static_cast<float>(storage.sK[k_idx]);
                        float q1 = static_cast<float>(storage.sQ[q_idx + 1]);
                        float k1 = static_cast<float>(storage.sK[k_idx + 1]);
                        float q2 = static_cast<float>(storage.sQ[q_idx + 2]);
                        float k2 = static_cast<float>(storage.sK[k_idx + 2]);
                        float q3 = static_cast<float>(storage.sK[k_idx + 3]);
                        float k3 = static_cast<float>(storage.sK[k_idx + 3]);
                        
                        score += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
                    }
                    
                    score *= scale;
                    storage.sS[m * kBlockN + n] = score;
                    row_max = fmaxf(row_max, score);
                }
                
                // Warp reduce for max
                for (int offset = 16; offset > 0; offset /= 2) {
                    row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));
                }
                
                // Update global max and rescale
                float old_max = storage.max_scores[m];
                float new_max = row_max;
                
                if (new_max > old_max) {
                    float rescale = expf(old_max - new_max);
                    storage.sum_exp[m] *= rescale;
                    
                    for (int d = lane_id; d < kHeadDim; d += 32) {
                        int o_idx = m * (kHeadDim + kPadding) + d;
                        storage.sO[o_idx] *= rescale;
                    }
                }
                __syncwarp();
                
                storage.max_scores[m] = new_max;
                
                // Compute exp and accumulate
                float local_sum = 0.0f;
                for (int n = lane_id; n < n_size; n += 32) {
                    float s = expf(storage.sS[m * kBlockN + n] - new_max);
                    storage.sS[m * kBlockN + n] = s;
                    local_sum += s;
                }
                
                // Warp reduce for sum
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
                }
                
                storage.sum_exp[m] += local_sum;
                
                // Accumulate PV
                for (int d = lane_id; d < kHeadDim; d += 32) {
                    float o_val = 0.0f;
                    for (int n = 0; n < n_size; ++n) {
                        int v_idx = d * (kBlockN + kPadding) + n;
                        float v = static_cast<float>(storage.sV[v_idx]);
                        o_val += storage.sS[m * kBlockN + n] * v;
                    }
                    int o_idx = m * (kHeadDim + kPadding) + d;
                    storage.sO[o_idx] += o_val;
                }
            }
            
            __syncthreads();
        }
        
        // Normalize and write output
        for (int i = tid; i < m_size * kHeadDim; i += kThreads) {
            int m = i / kHeadDim;
            int d = i % kHeadDim;
            int o_idx = m * (kHeadDim + kPadding) + d;
            float o_val = storage.sO[o_idx] / storage.sum_exp[m];
            gO(m_start + m, d) = static_cast<Element>(o_val);
        }
    }
};
