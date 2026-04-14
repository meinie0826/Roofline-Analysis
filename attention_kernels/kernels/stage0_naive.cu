/**
 * Stage 0: Naive FlashAttention Kernel
 * 
 * 最基础的实现：每个线程计算一个输出元素
 * 
 * 特点：
 * - 直接从全局内存读取 Q, K, V
 * - 每个线程独立计算完整的 attention score
 * - O(N²) 全局内存访问
 * - 无数据复用
 * 
 * 性能：~0.5-1 TFLOPs/s (严重带宽受限)
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// 使用 BF16 作为默认精度（与 FA4 一致）
using dtype = __nv_bfloat16;

__global__ void attention_naive_kernel(
    const dtype* __restrict__ Q,    // [batch, seq_len, n_heads, head_dim]
    const dtype* __restrict__ K,
    const dtype* __restrict__ V,
    dtype* __restrict__ O,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    float scale
) {
    // 每个线程计算 O[b, h, q_idx, :] 的一行
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    // 指向当前 query 的起始位置
    const dtype* q_ptr = Q + (batch * seq_len * n_heads + head * seq_len + q_idx) * head_dim;
    dtype* o_ptr = O + (batch * seq_len * n_heads + head * seq_len + q_idx) * head_dim;
    
    // 1. 计算 QK^T 的最大值（用于数值稳定性）
    float max_score = -INFINITY;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        const dtype* k_ptr = K + (batch * seq_len * n_heads + head * seq_len + k_idx) * head_dim;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = __bfloat162float(q_ptr[d]);
            float k_val = __bfloat162float(k_ptr[d]);
            score += q_val * k_val;
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }
    
    // 2. 计算 exp(QK^T - max) 和 sum(exp)
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        const dtype* k_ptr = K + (batch * seq_len * n_heads + head * seq_len + k_idx) * head_dim;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float q_val = __bfloat162float(q_ptr[d]);
            float k_val = __bfloat162float(k_ptr[d]);
            score += q_val * k_val;
        }
        score = expf(score * scale - max_score);
        sum_exp += score;
    }
    
    // 3. 计算 O = softmax(QK^T) @ V
    for (int d = 0; d < head_dim; d++) {
        float o_val = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            const dtype* k_ptr = K + (batch * seq_len * n_heads + head * seq_len + k_idx) * head_dim;
            const dtype* v_ptr = V + (batch * seq_len * n_heads + head * seq_len + k_idx) * head_dim;
            
            float score = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                float q_val = __bfloat162float(q_ptr[dd]);
                float k_val = __bfloat162float(k_ptr[dd]);
                score += q_val * k_val;
            }
            score = expf(score * scale - max_score) / sum_exp;
            o_val += score * __bfloat162float(v_ptr[d]);
        }
        o_ptr[d] = __float2bfloat16(o_val);
    }
}

// Causal attention 版本（只关注当前位置之前的）
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
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    const dtype* q_ptr = Q + (batch * seq_len * n_heads + head * seq_len + q_idx) * head_dim;
    dtype* o_ptr = O + (batch * seq_len * n_heads + head * seq_len + q_idx) * head_dim;
    
    // 只计算 q_idx 之前的位置（causal mask）
    int end_k = q_idx + 1;
    
    // 1. Find max score
    float max_score = -INFINITY;
    for (int k_idx = 0; k_idx < end_k; k_idx++) {
        const dtype* k_ptr = K + (batch * seq_len * n_heads + head * seq_len + k_idx) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }
    
    // 2. Compute sum(exp)
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < end_k; k_idx++) {
        const dtype* k_ptr = K + (batch * seq_len * n_heads + head * seq_len + k_idx) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __bfloat162float(q_ptr[d]) * __bfloat162float(k_ptr[d]);
        }
        score = expf(score * scale - max_score);
        sum_exp += score;
    }
    
    // 3. Compute output
    for (int d = 0; d < head_dim; d++) {
        float o_val = 0.0f;
        for (int k_idx = 0; k_idx < end_k; k_idx++) {
            const dtype* k_ptr = K + (batch * seq_len * n_heads + head * seq_len + k_idx) * head_dim;
            const dtype* v_ptr = V + (batch * seq_len * n_heads + head * seq_len + k_idx) * head_dim;
            
            float score = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                score += __bfloat162float(q_ptr[dd]) * __bfloat162float(k_ptr[dd]);
            }
            score = expf(score * scale - max_score) / sum_exp;
            o_val += score * __bfloat162float(v_ptr[d]);
        }
        o_ptr[d] = __float2bfloat16(o_val);
    }
}

// C 接口
extern "C" {
    void launch_attention_naive(
        const void* Q, const void* K, const void* V, void* O,
        int batch_size, int seq_len, int n_heads, int head_dim,
        bool causal, cudaStream_t stream
    ) {
        float scale = 1.0f / sqrtf((float)head_dim);
        
        dim3 grid((seq_len + 127) / 128, n_heads, batch_size);
        dim3 block(128);
        
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
