/**
 * FlashAttention Kernels Implementation
 */

#include "flash_attention.cuh"
#include <cuda_runtime.h>

using namespace flash_attention;

// Stage 0: Naive kernel
__global__ void flash_attention_stage0_kernel(
    const Element* __restrict__ Q,
    const Element* __restrict__ K,
    const Element* __restrict__ V,
    Element* __restrict__ O,
    int batch_size, int nheads, int seq_len, int head_dim,
    float scale, bool causal
) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m_idx >= seq_len) return;
    
    int base_idx = (batch * nheads + head) * seq_len * head_dim;
    const Element* q_ptr = Q + base_idx + m_idx * head_dim;
    Element* o_ptr = O + base_idx + m_idx * head_dim;
    
    float max_score = -INFINITY;
    int n_end = causal ? min(m_idx + 1, seq_len) : seq_len;
    
    // Find max
    for (int n = 0; n < n_end; ++n) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += static_cast<float>(q_ptr[d]) * 
                     static_cast<float>(K[base_idx + n * head_dim + d]);
        }
        max_score = fmaxf(max_score, score * scale);
    }
    
    // Compute sum(exp)
    float sum_exp = 0.0f;
    for (int n = 0; n < n_end; ++n) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += static_cast<float>(q_ptr[d]) * 
                     static_cast<float>(K[base_idx + n * head_dim + d]);
        }
        sum_exp += expf(score * scale - max_score);
    }
    
    // Compute output
    for (int d = 0; d < head_dim; ++d) {
        float o_val = 0.0f;
        for (int n = 0; n < n_end; ++n) {
            float score = 0.0f;
            for (int dd = 0; dd < head_dim; ++dd) {
                score += static_cast<float>(q_ptr[dd]) * 
                         static_cast<float>(K[base_idx + n * head_dim + dd]);
            }
            o_val += expf(score * scale - max_score) / sum_exp * 
                     static_cast<float>(V[base_idx + n * head_dim + d]);
        }
        o_ptr[d] = static_cast<Element>(o_val);
    }
}

// Stage 1: Tiled kernel
template <int kBlockM, int kBlockN, int kHeadDim>
__global__ void flash_attention_stage1_kernel(
    const Element* __restrict__ Q,
    const Element* __restrict__ K,
    const Element* __restrict__ V,
    Element* __restrict__ O,
    int batch_size, int nheads, int seq_len, int head_dim,
    float scale, bool causal
) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int m_tile = blockIdx.x;
    int m_start = m_tile * kBlockM;
    
    if (m_start >= seq_len) return;
    
    int m_end = min(m_start + kBlockM, seq_len);
    int m_size = m_end - m_start;
    
    extern __shared__ char smem[];
    Element* sQ = reinterpret_cast<Element*>(smem);
    Element* sK = sQ + kBlockM * kHeadDim;
    Element* sV = sK + kBlockN * kHeadDim;
    float* sO = reinterpret_cast<float*>(sV + kHeadDim * kBlockN);
    float* max_scores = sO + kBlockM * kHeadDim;
    float* sum_exp = max_scores + kBlockM;
    
    int base_idx = (batch * nheads + head) * seq_len * head_dim;
    int tid = threadIdx.x;
    
    // Initialize
    for (int i = tid; i < kBlockM; i += blockDim.x) {
        max_scores[i] = -INFINITY;
        sum_exp[i] = 0.0f;
    }
    for (int i = tid; i < kBlockM * kHeadDim; i += blockDim.x) {
        sO[i] = 0.0f;
    }
    
    __syncthreads();
    
    // Load Q tile
    for (int i = tid; i < m_size * kHeadDim; i += blockDim.x) {
        int m = i / kHeadDim;
        int d = i % kHeadDim;
        sQ[i] = Q[base_idx + (m_start + m) * head_dim + d];
    }
    
    __syncthreads();
    
    // Process K, V tiles
    for (int n_start = 0; n_start < seq_len; n_start += kBlockN) {
        int n_end_local = min(n_start + kBlockN, seq_len);
        int n_size = n_end_local - n_start;
        
        if (causal && n_start >= m_end) break;
        
        // Load K, V tiles
        for (int i = tid; i < n_size * kHeadDim; i += blockDim.x) {
            int n = i / kHeadDim;
            int d = i % kHeadDim;
            sK[i] = K[base_idx + (n_start + n) * head_dim + d];
            sV[d * kBlockN + n] = V[base_idx + (n_start + n) * head_dim + d];
        }
        
        __syncthreads();
        
        // Compute attention
        for (int m = tid; m < m_size; m += blockDim.x) {
            float row_max = max_scores[m];
            
            for (int n = 0; n < n_size; ++n) {
                if (causal && (n_start + n) > (m_start + m)) continue;
                
                float score = 0.0f;
                for (int d = 0; d < kHeadDim; ++d) {
                    score += static_cast<float>(sQ[m * kHeadDim + d]) * 
                             static_cast<float>(sK[n * kHeadDim + d]);
                }
                score *= scale;
                
                float old_max = row_max;
                row_max = fmaxf(row_max, score);
                float rescale = expf(old_max - row_max);
                
                float p = expf(score - row_max);
                max_scores[m] = row_max;
                sum_exp[m] = sum_exp[m] * rescale + p;
                
                for (int d = 0; d < kHeadDim; ++d) {
                    sO[m * kHeadDim + d] = sO[m * kHeadDim + d] * rescale +
                        p * static_cast<float>(sV[d * kBlockN + n]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    for (int i = tid; i < m_size * kHeadDim; i += blockDim.x) {
        int m = i / kHeadDim;
        int d = i % kHeadDim;
        O[base_idx + (m_start + m) * head_dim + d] = 
            static_cast<Element>(sO[i] / sum_exp[m]);
    }
}

// Launcher implementations with template<> syntax
template<>
void FlashAttentionKernel<0>::launch(
    const Element* Q, const Element* K, const Element* V, Element* O,
    int batch_size, int nheads, int seq_len, int head_dim,
    bool causal, cudaStream_t stream
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    dim3 grid(seq_len, nheads, batch_size);
    dim3 block(128);
    
    flash_attention_stage0_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, batch_size, nheads, seq_len, head_dim, scale, causal
    );
}

template<>
void FlashAttentionKernel<1>::launch(
    const Element* Q, const Element* K, const Element* V, Element* O,
    int batch_size, int nheads, int seq_len, int head_dim,
    bool causal, cudaStream_t stream
) {
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kHeadDim = 128;
    
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    dim3 grid((seq_len + kBlockM - 1) / kBlockM, nheads, batch_size);
    dim3 block(128);
    
    size_t smem_size = sizeof(Element) * (kBlockM + kBlockN + kBlockN) * kHeadDim +
                       sizeof(float) * (kBlockM * kHeadDim + kBlockM * 2);
    
    flash_attention_stage1_kernel<kBlockM, kBlockN, kHeadDim>
        <<<grid, block, smem_size, stream>>>(
            Q, K, V, O, batch_size, nheads, seq_len, head_dim, scale, causal
        );
}

// C interface
extern "C" {
    void launch_flash_attention(
        const void* Q, const void* K, const void* V, void* O,
        int batch_size, int nheads, int seq_len, int head_dim,
        int stage, bool causal, cudaStream_t stream
    ) {
        const Element* Q_cast = static_cast<const Element*>(Q);
        const Element* K_cast = static_cast<const Element*>(K);
        const Element* V_cast = static_cast<const Element*>(V);
        Element* O_cast = static_cast<Element*>(O);
        
        switch (stage) {
            case 0:
                FlashAttentionKernel<0>::launch(
                    Q_cast, K_cast, V_cast, O_cast,
                    batch_size, nheads, seq_len, head_dim, causal, stream
                );
                break;
            case 1:
                FlashAttentionKernel<1>::launch(
                    Q_cast, K_cast, V_cast, O_cast,
                    batch_size, nheads, seq_len, head_dim, causal, stream
                );
                break;
            default:
                // Use stage 0 as fallback
                FlashAttentionKernel<0>::launch(
                    Q_cast, K_cast, V_cast, O_cast,
                    batch_size, nheads, seq_len, head_dim, causal, stream
                );
                break;
        }
    }
}
