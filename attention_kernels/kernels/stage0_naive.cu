/**
 * Stage 0: Naive FlashAttention Kernel
 * 
 * 改进版本：只计算一次 QK^T，存在 shared memory
 * Grid: (batch, head), Block: threads = seq_len
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

using dtype = __nv_bfloat16;

constexpr int MAX_THREADS = 1024;

extern "C" {

__global__ void __launch_bounds__(1024)
attention_naive_causal_kernel(
    const dtype* __restrict__ Q,
    const dtype* __restrict__ K,
    const dtype* __restrict__ V,
    dtype* __restrict__ O,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    float scale,
    float* __restrict__ smem  // shared memory for attention weights
) {
    int batch = blockIdx.y;
    int head = blockIdx.x;
    int q_idx = threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    int base = (batch * n_heads + head) * seq_len;
    const dtype* q_ptr = Q + (base + q_idx) * head_dim;
    dtype* o_ptr = O + (base + q_idx) * head_dim;
    
    float* attn_weights = smem + q_idx * seq_len;
    
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
        attn_weights[k_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Step 2: exp + sum
    float sum_exp = 0.0f;
    
    for (int k_idx = 0; k_idx < end_k; k_idx++) {
        attn_weights[k_idx] = expf(attn_weights[k_idx] - max_score);
        sum_exp += attn_weights[k_idx];
    }
    
    // Step 3: Output
    for (int d = 0; d < head_dim; d++) {
        float o_val = 0.0f;
        
        for (int k_idx = 0; k_idx < end_k; k_idx++) {
            const dtype* v_ptr = V + (base + k_idx) * head_dim;
            o_val += attn_weights[k_idx] * __bfloat162float(v_ptr[d]);
        }
        
        o_ptr[d] = __float2bfloat16(o_val / sum_exp);
    }
}

__global__ void __launch_bounds__(1024)
attention_naive_kernel(
    const dtype* __restrict__ Q,
    const dtype* __restrict__ K,
    const dtype* __restrict__ V,
    dtype* __restrict__ O,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    float scale,
    float* __restrict__ smem
) {
    int batch = blockIdx.y;
    int head = blockIdx.x;
    int q_idx = threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    int base = (batch * n_heads + head) * seq_len;
    const dtype* q_ptr = Q + (base + q_idx) * head_dim;
    dtype* o_ptr = O + (base + q_idx) * head_dim;
    
    float* attn_weights = smem + q_idx * seq_len;
    
    // Step 1: QK^T + max
    float max_score = -INFINITY;
    
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        const dtype* k_ptr = K + (base + k_idx) * head_dim;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        score *= scale;
        attn_weights[k_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Step 2: exp + sum
    float sum_exp = 0.0f;
    
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        attn_weights[k_idx] = expf(attn_weights[k_idx] - max_score);
        sum_exp += attn_weights[k_idx];
    }
    
    // Step 3: Output
    for (int d = 0; d < head_dim; d++) {
        float o_val = 0.0f;
        
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            const dtype* v_ptr = V + (base + k_idx) * head_dim;
            o_val += attn_weights[k_idx] * __bfloat162float(v_ptr[d]);
        }
        
        o_ptr[d] = __float2bfloat16(o_val / sum_exp);
    }
}

void launch_attention_naive(
    const void* Q, const void* K, const void* V, void* O,
    int batch_size, int seq_len, int n_heads, int head_dim,
    bool causal, cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    
    dim3 grid(n_heads, batch_size);
    dim3 block(min(seq_len, MAX_THREADS));
    
    // 动态 shared memory 存储 attention weights
    size_t smem_size = sizeof(float) * seq_len * block.x;
    
    void* args[] = {&Q, &K, &V, &O, &batch_size, &seq_len, &n_heads, &head_dim, &scale};
    
    if (causal) {
        cudaLaunchKernel((void*)attention_naive_causal_kernel, grid, block, args, smem_size, stream);
    } else {
        cudaLaunchKernel((void*)attention_naive_kernel, grid, block, args, smem_size, stream);
    }
}

} // extern "C"
