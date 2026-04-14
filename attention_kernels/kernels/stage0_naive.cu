/**
 * Stage 0: Naive FlashAttention Kernel
 * 
 * 简单实现：每个线程处理一个 query position
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

using dtype = __nv_bfloat16;

constexpr int MAX_THREADS = 1024;

__global__ void attention_naive_causal_kernel(
    const dtype* __restrict__ Q,
    const dtype* __restrict__ K,
    const dtype* __restrict__ V,
    dtype* __restrict__ O,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    float scale
) {
    int batch = blockIdx.y;
    int head = blockIdx.x;
    int q_idx = threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    int base = (batch * n_heads + head) * seq_len;
    const dtype* q_ptr = Q + (base + q_idx) * head_dim;
    dtype* o_ptr = O + (base + q_idx) * head_dim;
    
    // Causal: 只看 q_idx 之前
    int end_k = q_idx + 1;
    
    // Step 1: QK^T + max
    float max_score = -INFINITY;
    
    for (int k_idx = 0; k_idx < end_k; k_idx++) {
        const dtype* k_ptr = K + (base + k_idx) * head_dim;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }
    
    // Step 2: exp + sum
    float sum_exp = 0.0f;
    
    for (int k_idx = 0; k_idx < end_k; k_idx++) {
        const dtype* k_ptr = K + (base + k_idx) * head_dim;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        sum_exp += expf(score * scale - max_score);
    }
    
    // Step 3: Output
    for (int d = 0; d < head_dim; d++) {
        float o_val = 0.0f;
        
        for (int k_idx = 0; k_idx < end_k; k_idx++) {
            const dtype* k_ptr = K + (base + k_idx) * head_dim;
            const dtype* v_ptr = V + (base + k_idx) * head_dim;
            
            float score = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                score += __bfloat162float(q_ptr[dd]) * __bfloat162float(k_ptr[dd]);
            }
            
            float attn = expf(score * scale - max_score) / sum_exp;
            o_val += attn * __bfloat162float(v_ptr[d]);
        }
        
        o_ptr[d] = __float2bfloat16(o_val);
    }
}

__global__ void attention_naive_kernel(
    const dtype* __restrict__ Q,
    const dtype* __restrict__ K,
    const dtype* __restrict__ V,
    dtype* __restrict__ O,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    float scale
) {
    int batch = blockIdx.y;
    int head = blockIdx.x;
    int q_idx = threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    int base = (batch * n_heads + head) * seq_len;
    const dtype* q_ptr = Q + (base + q_idx) * head_dim;
    dtype* o_ptr = O + (base + q_idx) * head_dim;
    
    // Step 1: QK^T + max
    float max_score = -INFINITY;
    
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        const dtype* k_ptr = K + (base + k_idx) * head_dim;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }
    
    // Step 2: exp + sum
    float sum_exp = 0.0f;
    
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        const dtype* k_ptr = K + (base + k_idx) * head_dim;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        sum_exp += expf(score * scale - max_score);
    }
    
    // Step 3: Output
    for (int d = 0; d < head_dim; d++) {
        float o_val = 0.0f;
        
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            const dtype* k_ptr = K + (base + k_idx) * head_dim;
            const dtype* v_ptr = V + (base + k_idx) * head_dim;
            
            float score = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                score += __bfloat162float(q_ptr[dd]) * __bfloat162float(k_ptr[dd]);
            }
            
            float attn = expf(score * scale - max_score) / sum_exp;
            o_val += attn * __bfloat162float(v_ptr[d]);
        }
        
        o_ptr[d] = __float2bfloat16(o_val);
    }
}

extern "C" {
    void launch_attention_naive(
        const void* Q, const void* K, const void* V, void* O,
        int batch_size, int seq_len, int n_heads, int head_dim,
        bool causal, cudaStream_t stream
    ) {
        float scale = 1.0f / sqrtf((float)head_dim);
        
        dim3 grid(n_heads, batch_size);
        dim3 block(seq_len < MAX_THREADS ? seq_len : MAX_THREADS);
        
        if (causal) {
            attention_naive_causal_kernel<<<grid, block, 0, stream>>>(
                (const dtype*)Q, (const dtype*)K, (const dtype*)V, (dtype*)O,
                batch_size, seq_len, n_heads, head_dim, scale
            );
        } else {
            attention_naive_kernel<<<grid, block, 0, stream>>>(
                (const dtype*)Q, (const dtype*)K, (const dtype*)V, (dtype*)O,
                batch_size, seq_len, n_heads, head_dim, scale
            );
        }
    }
}
