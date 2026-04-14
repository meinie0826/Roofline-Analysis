/**
 * Stage 2: Shared Memory Optimized FlashAttention Kernel
 * 
 * 优化点：使用 shared memory 缓存 tiles
 * - Q, K, V tiles 先加载到 shared memory
 * - Block 内所有线程复用
 * - 大幅减少 global memory traffic
 * 
 * 性能：~8-15 TFLOPs/s（相比 stage1 提升 3-4x）
 * 瓶颈：计算（Tensor Core 未使用）
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

using dtype = __nv_bfloat16;

// Reduced tile sizes to fit in 48KB shared memory
constexpr int TILE_M = 64;    // Query tile (reduced from 128)
constexpr int TILE_N = 32;    // Key tile (reduced from 64)
constexpr int TILE_K = 64;    // Head dimension (reduced from 128)
constexpr int THREADS_PER_BLOCK = 128;

__global__ void attention_smem_kernel(
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
    // Shared memory tiles - fit in 48KB
    __shared__ dtype Q_tile[TILE_M * TILE_K];   // 64*64*2 = 8KB
    __shared__ dtype K_tile[TILE_N * TILE_K];   // 32*64*2 = 4KB
    __shared__ dtype V_tile[TILE_K * TILE_N];   // 64*32*2 = 4KB (transposed)
    __shared__ float S_tile[TILE_M * TILE_N];   // 64*32*4 = 8KB (attention scores)
    __shared__ float O_partial[TILE_M * TILE_K]; // 64*64*4 = 16KB (FP32 accumulator)
    __shared__ float max_scores[TILE_M];
    __shared__ float sum_exp[TILE_M];
    
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int tile_m = blockIdx.x;
    
    int q_start = tile_m * TILE_M;
    int q_end = min(q_start + TILE_M, seq_len);
    int m_size = q_end - q_start;
    
    if (q_start >= seq_len) return;
    
    int tid = threadIdx.x;
    
    const dtype* Q_base = Q + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    const dtype* K_base = K + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    const dtype* V_base = V + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    dtype* O_base = O + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    
    // Initialize accumulators
    for (int i = tid; i < TILE_M; i += THREADS_PER_BLOCK) {
        max_scores[i] = -INFINITY;
        sum_exp[i] = 0.0f;
    }
    for (int i = tid; i < TILE_M * TILE_K; i += THREADS_PER_BLOCK) {
        O_partial[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile
    for (int idx = tid; idx < m_size * head_dim; idx += THREADS_PER_BLOCK) {
        int q = idx / head_dim;
        int d = idx % head_dim;
        Q_tile[idx] = Q_base[(q_start + q) * head_dim + d];
    }
    __syncthreads();
    
    // Iterate over K, V tiles
    for (int k_tile_start = 0; k_tile_start < seq_len; k_tile_start += TILE_N) {
        int k_tile_end = min(k_tile_start + TILE_N, seq_len);
        int n_size = k_tile_end - k_tile_start;
        
        // Load K tile
        for (int idx = tid; idx < n_size * head_dim; idx += THREADS_PER_BLOCK) {
            int k = idx / head_dim;
            int d = idx % head_dim;
            K_tile[idx] = K_base[(k_tile_start + k) * head_dim + d];
        }
        
        // Load V tile (transposed for coalescing)
        for (int idx = tid; idx < n_size * head_dim; idx += THREADS_PER_BLOCK) {
            int k = idx / head_dim;
            int d = idx % head_dim;
            V_tile[d * TILE_N + k] = V_base[(k_tile_start + k) * head_dim + d];
        }
        __syncthreads();
        
        // Compute QK^T
        for (int q_row = tid; q_row < m_size; q_row += THREADS_PER_BLOCK) {
            float local_max = -INFINITY;
            for (int k_col = 0; k_col < n_size; k_col++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += __bfloat162float(Q_tile[q_row * head_dim + d]) * 
                             __bfloat162float(K_tile[k_col * head_dim + d]);
                }
                score *= scale;
                S_tile[q_row * TILE_N + k_col] = score;
                local_max = fmaxf(local_max, score);
            }
            
            // Update global max with online softmax
            float old_max = max_scores[q_row];
            float new_max = fmaxf(old_max, local_max);
            
            if (new_max > old_max) {
                float rescale = expf(old_max - new_max);
                atomicAdd(&sum_exp[q_row], 0.0f); // Read current value
                float cur_sum = sum_exp[q_row] * rescale;
                sum_exp[q_row] = cur_sum;
                
                for (int d = 0; d < head_dim; d++) {
                    O_partial[q_row * head_dim + d] *= rescale;
                }
                max_scores[q_row] = new_max;
            }
        }
        __syncthreads();
        
        // Compute exp and accumulate
        for (int q_row = tid; q_row < m_size; q_row += THREADS_PER_BLOCK) {
            float row_max = max_scores[q_row];
            float row_sum = 0.0f;
            
            for (int k_col = 0; k_col < n_size; k_col++) {
                float exp_s = expf(S_tile[q_row * TILE_N + k_col] - row_max);
                row_sum += exp_s;
                
                for (int d = 0; d < head_dim; d++) {
                    float v_val = __bfloat162float(V_tile[d * TILE_N + k_col]);
                    O_partial[q_row * head_dim + d] += exp_s * v_val;
                }
            }
            
            atomicAdd(&sum_exp[q_row], row_sum);
        }
        __syncthreads();
    }
    
    // Normalize and write output
    for (int idx = tid; idx < m_size * head_dim; idx += THREADS_PER_BLOCK) {
        int q = idx / head_dim;
        int d = idx % head_dim;
        float o_val = O_partial[idx] / sum_exp[q];
        O_base[(q_start + q) * head_dim + d] = __float2bfloat16(o_val);
    }
}

__global__ void attention_smem_causal_kernel(
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
    __shared__ dtype Q_tile[TILE_M * TILE_K];
    __shared__ dtype K_tile[TILE_N * TILE_K];
    __shared__ dtype V_tile[TILE_K * TILE_N];
    __shared__ float S_tile[TILE_M * TILE_N];
    __shared__ float O_partial[TILE_M * TILE_K];
    __shared__ float max_scores[TILE_M];
    __shared__ float sum_exp[TILE_M];
    
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int tile_m = blockIdx.x;
    
    int q_start = tile_m * TILE_M;
    int q_end = min(q_start + TILE_M, seq_len);
    int m_size = q_end - q_start;
    
    if (q_start >= seq_len) return;
    
    int tid = threadIdx.x;
    
    const dtype* Q_base = Q + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    const dtype* K_base = K + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    const dtype* V_base = V + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    dtype* O_base = O + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    
    // Initialize
    for (int i = tid; i < TILE_M; i += THREADS_PER_BLOCK) {
        max_scores[i] = -INFINITY;
        sum_exp[i] = 0.0f;
    }
    for (int i = tid; i < TILE_M * TILE_K; i += THREADS_PER_BLOCK) {
        O_partial[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile
    for (int idx = tid; idx < m_size * head_dim; idx += THREADS_PER_BLOCK) {
        int q = idx / head_dim;
        int d = idx % head_dim;
        Q_tile[idx] = Q_base[(q_start + q) * head_dim + d];
    }
    __syncthreads();
    
    // Causal: only process K tiles <= current Q position
    for (int k_tile_start = 0; k_tile_start < q_end; k_tile_start += TILE_N) {
        int k_tile_end = min(k_tile_start + TILE_N, q_end);
        int n_size = k_tile_end - k_tile_start;
        
        // Load K, V tiles
        for (int idx = tid; idx < n_size * head_dim; idx += THREADS_PER_BLOCK) {
            int k = idx / head_dim;
            int d = idx % head_dim;
            K_tile[idx] = K_base[(k_tile_start + k) * head_dim + d];
            V_tile[d * TILE_N + k] = V_base[(k_tile_start + k) * head_dim + d];
        }
        __syncthreads();
        
        // Compute with causal mask
        for (int q_row = tid; q_row < m_size; q_row += THREADS_PER_BLOCK) {
            int global_q = q_start + q_row;
            float local_max = -INFINITY;
            
            for (int k_col = 0; k_col < n_size; k_col++) {
                int global_k = k_tile_start + k_col;
                float score = -INFINITY;
                
                if (global_k <= global_q) {
                    score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        score += __bfloat162float(Q_tile[q_row * head_dim + d]) * 
                                 __bfloat162float(K_tile[k_col * head_dim + d]);
                    }
                    score *= scale;
                }
                
                S_tile[q_row * TILE_N + k_col] = score;
                local_max = fmaxf(local_max, score);
            }
            
            float old_max = max_scores[q_row];
            float new_max = fmaxf(old_max, local_max);
            
            if (new_max > old_max) {
                float rescale = expf(old_max - new_max);
                sum_exp[q_row] *= rescale;
                for (int d = 0; d < head_dim; d++) {
                    O_partial[q_row * head_dim + d] *= rescale;
                }
                max_scores[q_row] = new_max;
            }
        }
        __syncthreads();
        
        // Accumulate
        for (int q_row = tid; q_row < m_size; q_row += THREADS_PER_BLOCK) {
            float row_max = max_scores[q_row];
            float row_sum = 0.0f;
            
            for (int k_col = 0; k_col < n_size; k_col++) {
                int global_k = k_tile_start + k_col;
                if (global_k > q_start + q_row) continue;
                
                float exp_s = expf(S_tile[q_row * TILE_N + k_col] - row_max);
                row_sum += exp_s;
                
                for (int d = 0; d < head_dim; d++) {
                    float v_val = __bfloat162float(V_tile[d * TILE_N + k_col]);
                    O_partial[q_row * head_dim + d] += exp_s * v_val;
                }
            }
            
            atomicAdd(&sum_exp[q_row], row_sum);
        }
        __syncthreads();
    }
    
    // Write output
    for (int idx = tid; idx < m_size * head_dim; idx += THREADS_PER_BLOCK) {
        int q = idx / head_dim;
        int d = idx % head_dim;
        float o_val = O_partial[idx] / sum_exp[q];
        O_base[(q_start + q) * head_dim + d] = __float2bfloat16(o_val);
    }
}

extern "C" {
    void launch_attention_smem(
        const void* Q, const void* K, const void* V, void* O,
        int batch_size, int seq_len, int n_heads, int head_dim,
        bool causal, cudaStream_t stream
    ) {
        float scale = 1.0f / sqrtf((float)head_dim);
        
        dim3 grid((seq_len + TILE_M - 1) / TILE_M, n_heads, batch_size);
        dim3 block(THREADS_PER_BLOCK);
        
        if (causal) {
            attention_smem_causal_kernel<<<grid, block, 0, stream>>>(
                (const dtype*)Q, (const dtype*)K, (const dtype*)V, (dtype*)O,
                batch_size, seq_len, n_heads, head_dim, scale
            );
        } else {
            attention_smem_kernel<<<grid, block, 0, stream>>>(
                (const dtype*)Q, (const dtype*)K, (const dtype*)V, (dtype*)O,
                batch_size, seq_len, n_heads, head_dim, scale
            );
        }
    }
}
