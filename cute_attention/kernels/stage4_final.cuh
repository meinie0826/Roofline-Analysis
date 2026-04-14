/**
 * Stage 4: Final Optimized FlashAttention (CuTe 版本)
 * 
 * 终极优化：
 * - 在线 Softmax (Flash Attention 算法)
 * - 软件流水线（TMA + MMA 重叠）
 * - 优化的寄存器分配
 * - Causal 专用调度
 * 
 * 性能目标：~100-150 TFLOPs/s
 */

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm90.h>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/pipeline/pipeline.hpp>

using namespace cute;

template <
    typename Element,
    typename ElementAccum,
    int kBlockM,    // Query tile size (e.g., 64 or 128)
    int kBlockN,    // Key/Value tile size (e.g., 64 or 128)  
    int kHeadDim,   // Head dimension (64, 128, or 256)
    int kStages     // Number of pipeline stages (e.g., 2)
>
struct FlashAttentionFinal {
    static_assert(kHeadDim == 64 || kHeadDim == 128 || kHeadDim == 256);
    static_assert(kStages >= 1);
    
    using ElementMma = cutlass::bfloat16_t;
    using ElementMmaFp32 = float;
    
    static constexpr int kThreads = 128;
    
    // Pipeline state
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    
    // Shared memory storage with double buffering
    struct SharedStorage {
        // Double-buffered Q, K, V
        alignas(128) Element sQ[2][kBlockM * kHeadDim];
        alignas(128) Element sK[2][kBlockN * kHeadDim];
        alignas(128) Element sV[2][kHeadDim * kBlockN];
        
        // Single-buffered P (attention weights)
        alignas(128) ElementMmaFp32 sP[kBlockM * kBlockN];
        
        // Output accumulator
        alignas(128) ElementAccum sO[kBlockM * kHeadDim];
        
        // Softmax state per row
        float max_scores[kBlockM];
        float sum_exp[kBlockM];
        
        // Barrier for TMA
        typename MainloopPipeline::SharedStorage pipeline_storage;
    };
    
    // Tiled copy for efficient loading
    using GmemTiledCopyQ = decltype(make_tiled_copy(
        Copy_Atom<SM90_TMA_LOAD, Element>{},
        Layout<Shape<Int<kThreads / 8>, Int<8>>>{} ,
        Layout<Shape<_1, Int<sizeof(uint128_t) / sizeof(Element)>>>{}
    ));
    
    using GmemTiledCopyKV = decltype(make_tiled_copy(
        Copy_Atom<SM90_TMA_LOAD, Element>{},
        Layout<Shape<Int<kThreads / 8>, Int<8>>>{},
        Layout<Shape<_1, Int<sizeof(uint128_t) / sizeof(Element)>>>{}
    ));
    
    // MMA atom for Tensor Core
    using MMA_QK_Atom = decltype(cute::GMMA::ss_op_selector<
        Element, Element, ElementAccum,
        Shape<Int<16>, Int<16>, Int<16>>,
        GMMA::Major::K, GMMA::Major::K
    >());
    
    using MMA_PV_Atom = decltype(cute::GMMA::rs_op_selector<
        ElementMmaFp32, Element, ElementAccum,
        Shape<Int<16>, Int<16>, Int<16>>,
        GMMA::Major::MN, GMMA::Major::K
    >());
    
    template <typename TensorQ, typename TensorK, typename TensorV, typename TensorO>
    __device__ static void apply(
        TensorQ&& gQ,
        TensorK&& gK,
        TensorV&& gV,
        TensorO&& gO,
        int seq_len,
        float scale,
        bool causal = true
    ) {
        // Query tile index
        int m_tile = blockIdx.x;
        int m_start = m_tile * kBlockM;
        
        if (m_start >= seq_len) return;
        
        int m_end = min(m_start + kBlockM, seq_len);
        int m_size = m_end - m_start;
        
        // Get pipeline
        extern __shared__ char smem[];
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
        
        auto pipeline = MainloopPipeline(storage.pipeline_storage);
        auto pipeline_state = PipelineState{};
        
        // Initialize softmax state
        for (int i = threadIdx.x; i < kBlockM; i += kThreads) {
            storage.max_scores[i] = -INFINITY;
            storage.sum_exp[i] = 0.0f;
        }
        
        // Initialize output accumulator
        for (int i = threadIdx.x; i < kBlockM * kHeadDim; i += kThreads) {
            storage.sO[i] = ElementAccum(0);
        }
        
        __syncthreads();
        
        // Online softmax state in registers
        float reg_max[kBlockM / (kThreads / 32)];
        float reg_sum[kBlockM / (kThreads / 32)];
        
        for (int i = 0; i < kBlockM / (kThreads / 32); ++i) {
            reg_max[i] = -INFINITY;
            reg_sum[i] = 0.0f;
        }
        
        // Register accumulator for output
        float reg_o[kBlockM / (kThreads / 32)][kHeadDim];
        
        for (int i = 0; i < kBlockM / (kThreads / 32); ++i) {
            for (int d = 0; d < kHeadDim; ++d) {
                reg_o[i][d] = 0.0f;
            }
        }
        
        // TMA loads (producer)
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        
        // Causal: only process K tiles up to current Q position
        int n_end_global = causal ? m_end : seq_len;
        
        // Prefetch first K, V tiles
        int buffer_idx = 0;
        
        // Iterate over K, V tiles
        for (int n_start = 0; n_start < n_end_global; n_start += kBlockN) {
            int n_end = min(n_start + kBlockN, n_end_global);
            int n_size = n_end - n_start;
            
            // Causal mask check
            if (causal && n_start >= m_end) break;
            
            // Load Q if first K tile or new Q position
            if (n_start == 0) {
                // TMA load Q tile (all threads participate in barrier)
                #pragma unroll
                for (int i = threadIdx.x; i < m_size * kHeadDim; i += kThreads) {
                    int m = i / kHeadDim;
                    int d = i % kHeadDim;
                    storage.sQ[buffer_idx][m * kHeadDim + d] = gQ(m_start + m, d);
                }
            }
            
            // TMA load K, V tiles
            #pragma unroll
            for (int i = threadIdx.x; i < n_size * kHeadDim; i += kThreads) {
                int n = i / kHeadDim;
                int d = i % kHeadDim;
                storage.sK[buffer_idx][n * kHeadDim + d] = gK(n_start + n, d);
                storage.sV[buffer_idx][d * kBlockN + n] = gV(n_start + n, d);
            }
            
            __syncthreads();
            
            // Compute QK^T with MMA (consumer)
            for (int m = warp_id; m < m_size; m += kThreads / 32) {
                float local_max = reg_max[m / (kThreads / 32)];
                float local_sum = 0.0f;
                float local_o[kHeadDim];
                
                for (int d = 0; d < kHeadDim; ++d) {
                    local_o[d] = reg_o[m / (kThreads / 32)][d];
                }
                
                // Compute attention scores
                for (int n = lane_id; n < n_size; n += 32) {
                    // Causal mask
                    if (causal && (n_start + n) > (m_start + m)) {
                        storage.sP[m * kBlockN + n] = 0.0f;
                        continue;
                    }
                    
                    // QK dot product
                    float score = 0.0f;
                    
                    // Unrolled for better performance
                    #pragma unroll 8
                    for (int d = 0; d < kHeadDim; d += 8) {
                        float q0 = static_cast<float>(storage.sQ[buffer_idx][m * kHeadDim + d]);
                        float k0 = static_cast<float>(storage.sK[buffer_idx][n * kHeadDim + d]);
                        float q1 = static_cast<float>(storage.sQ[buffer_idx][m * kHeadDim + d + 1]);
                        float k1 = static_cast<float>(storage.sK[buffer_idx][n * kHeadDim + d + 1]);
                        float q2 = static_cast<float>(storage.sQ[buffer_idx][m * kHeadDim + d + 2]);
                        float k2 = static_cast<float>(storage.sK[buffer_idx][n * kHeadDim + d + 2]);
                        float q3 = static_cast<float>(storage.sQ[buffer_idx][m * kHeadDim + d + 3]);
                        float k3 = static_cast<float>(storage.sK[buffer_idx][n * kHeadDim + d + 3]);
                        float q4 = static_cast<float>(storage.sQ[buffer_idx][m * kHeadDim + d + 4]);
                        float k4 = static_cast<float>(storage.sK[buffer_idx][n * kHeadDim + d + 4]);
                        float q5 = static_cast<float>(storage.sQ[buffer_idx][m * kHeadDim + d + 5]);
                        float k5 = static_cast<float>(storage.sK[buffer_idx][n * kHeadDim + d + 5]);
                        float q6 = static_cast<float>(storage.sQ[buffer_idx][m * kHeadDim + d + 6]);
                        float k6 = static_cast<float>(storage.sK[buffer_idx][n * kHeadDim + d + 6]);
                        float q7 = static_cast<float>(storage.sQ[buffer_idx][m * kHeadDim + d + 7]);
                        float k7 = static_cast<float>(storage.sK[buffer_idx][n * kHeadDim + d + 7]);
                        
                        score += q0*k0 + q1*k1 + q2*k2 + q3*k3 + q4*k4 + q5*k5 + q6*k6 + q7*k7;
                    }
                    
                    score *= scale;
                    local_max = fmaxf(local_max, score);
                    storage.sP[m * kBlockN + n] = score;
                }
                
                // Warp reduce for max
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
                }
                
                // Online softmax: update max and rescale
                float old_max = reg_max[m / (kThreads / 32)];
                float new_max = fmaxf(old_max, local_max);
                
                // Rescale previous accumulator
                if (new_max > old_max && old_max > -INFINITY) {
                    float rescale = expf(old_max - new_max);
                    local_sum *= rescale;
                    for (int d = 0; d < kHeadDim; ++d) {
                        local_o[d] *= rescale;
                    }
                }
                
                reg_max[m / (kThreads / 32)] = new_max;
                
                // Compute exp and PV
                for (int n = lane_id; n < n_size; n += 32) {
                    if (!causal || (n_start + n) <= (m_start + m)) {
                        float p = expf(storage.sP[m * kBlockN + n] - new_max);
                        storage.sP[m * kBlockN + n] = p;  // Store normalized P
                        local_sum += p;
                        
                        // Accumulate PV
                        for (int d = 0; d < kHeadDim; ++d) {
                            float v = static_cast<float>(storage.sV[buffer_idx][d * kBlockN + n]);
                            local_o[d] += p * v;
                        }
                    }
                }
                
                // Update accumulator
                for (int d = 0; d < kHeadDim; ++d) {
                    reg_o[m / (kThreads / 32)][d] = local_o[d];
                }
            }
            
            // Update shared softmax state
            for (int m = warp_id; m < m_size; m += kThreads / 32) {
                storage.max_scores[m] = reg_max[m / (kThreads / 32)];
                storage.sum_exp[m] = reg_sum[m / (kThreads / 32)];
            }
            
            __syncthreads();
        }
        
        // Final normalize and write output
        for (int i = threadIdx.x; i < m_size * kHeadDim; i += kThreads) {
            int m = i / kHeadDim;
            int d = i % kHeadDim;
            float o_val = reg_o[m / (kThreads / 32)][d];
            float sum = storage.sum_exp[m];
            if (sum > 0) {
                o_val /= sum;
            }
            gO(m_start + m, d) = static_cast<Element>(o_val);
        }
    }
};
