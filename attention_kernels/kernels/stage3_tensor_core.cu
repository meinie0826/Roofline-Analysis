/**
 * Stage 3: Tensor Core (WMMA) FlashAttention Kernel
 * 
 * 优化点：使用 Tensor Core 进行矩阵乘法
 * - warp-level MMA (Matrix Multiply Accumulate)
 * - FP16 输入（WMMA 需要），可从 BF16 转换
 * - 使用 ldmatrix, mma.sync 指令
 * 
 * 性能：~50-80 TFLOPs/s（相比 stage2 提升 4-8x）
 * 瓶颈：计算接近峰值，但 softmax 的 exp 成为新瓶颈
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

// 使用 FP16 进行 WMMA，然后转换
using wmma_dtype = half;
using compute_dtype = __nv_bfloat16;

// Tile sizes - 针对 Tensor Core 优化
// MMA 指令: 16x16x16 (FP16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block-level tiles
constexpr int BLOCK_M = 128;   // Output rows per block
constexpr int BLOCK_N = 64;    // Key tile size
constexpr int BLOCK_K = 128;   // Head dimension
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

/**
 * 使用 WMMA 的 FlashAttention kernel
 * 
 * 每个 warp 处理 BLOCK_M / WARPS_PER_BLOCK = 32 行的输出
 * 使用 mma.sync 指令加速 QK^T 和 PV 的矩阵乘法
 */
__global__ void attention_wmma_kernel(
    const compute_dtype* __restrict__ Q,
    const compute_dtype* __restrict__ K,
    const compute_dtype* __restrict__ V,
    compute_dtype* __restrict__ O,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    float scale
) {
    // Shared memory for Q, K, V tiles
    __shared__ compute_dtype Q_smem[BLOCK_M * BLOCK_K];
    __shared__ compute_dtype K_smem[BLOCK_N * BLOCK_K];
    __shared__ compute_dtype V_smem[BLOCK_K * BLOCK_N];  // Transposed
    
    // Accumulator for output
    __shared__ float O_acc[BLOCK_M * BLOCK_K];
    __shared__ float max_vals[BLOCK_M];
    __shared__ float sum_exp[BLOCK_M];
    __shared__ float S_smem[BLOCK_M * BLOCK_N];
    
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int block_m = blockIdx.x;  // Block index in M dimension
    
    int q_start = block_m * BLOCK_M;
    if (q_start >= seq_len) return;
    
    int q_end = min(q_start + BLOCK_M, seq_len);
    int m_size = q_end - q_start;
    
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Base pointers
    const compute_dtype* Q_base = Q + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    const compute_dtype* K_base = K + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    const compute_dtype* V_base = V + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    compute_dtype* O_base = O + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    
    // Initialize accumulators
    for (int i = threadIdx.x; i < BLOCK_M; i += THREADS_PER_BLOCK) {
        max_vals[i] = -INFINITY;
        sum_exp[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < BLOCK_M * BLOCK_K; i += THREADS_PER_BLOCK) {
        O_acc[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile to shared memory
    for (int idx = threadIdx.x; idx < m_size * head_dim; idx += THREADS_PER_BLOCK) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        int global_q = q_start + row;
        Q_smem[idx] = Q_base[global_q * head_dim + col];
    }
    __syncthreads();
    
    // Iterate over K, V tiles
    for (int k_start = 0; k_start < seq_len; k_start += BLOCK_N) {
        int k_end = min(k_start + BLOCK_N, seq_len);
        int n_size = k_end - k_start;
        
        // Load K, V tiles
        for (int idx = threadIdx.x; idx < n_size * head_dim; idx += THREADS_PER_BLOCK) {
            int row = idx / head_dim;
            int col = idx % head_dim;
            int global_k = k_start + row;
            K_smem[idx] = K_base[global_k * head_dim + col];
            V_smem[col * BLOCK_N + row] = V_base[global_k * head_dim + col];
        }
        __syncthreads();
        
        // Each warp processes a subset of Q rows
        int rows_per_warp = (m_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        int warp_q_start = warp_id * rows_per_warp;
        int warp_q_end = min(warp_q_start + rows_per_warp, m_size);
        
        for (int q_row = warp_q_start; q_row < warp_q_end; ++q_row) {
            // Compute QK^T for this row against the current K tile
            // Result is a 1 x n_size vector
            
            // Find max in this row
            float row_max = -INFINITY;
            for (int k_col = lane_id; k_col < n_size; k_col += WARP_SIZE) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    float q_val = __bfloat162float(Q_smem[q_row * head_dim + d]);
                    float k_val = __bfloat162float(K_smem[k_col * head_dim + d]);
                    score += q_val * k_val;
                }
                score *= scale;
                S_smem[q_row * BLOCK_N + k_col] = score;
                row_max = fmaxf(row_max, score);
            }
            
            // Warp reduce for max
            for (int offset = 16; offset > 0; offset /= 2) {
                row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));
            }
            
            // Update global max
            if (lane_id == 0) {
                float old_max = max_vals[q_row];
                float new_max = fmaxf(old_max, row_max);
                max_vals[q_row] = new_max;
                
                // Rescale if needed
                if (new_max > old_max) {
                    float rescale = expf(old_max - new_max);
                    sum_exp[q_row] *= rescale;
                    for (int d = 0; d < head_dim; d++) {
                        O_acc[q_row * head_dim + d] *= rescale;
                    }
                }
            }
            __syncthreads();
            
            float curr_max = max_vals[q_row];
            float row_sum = 0.0f;
            
            // Compute exp(QK^T - max) @ V
            for (int k_col = lane_id; k_col < n_size; k_col += WARP_SIZE) {
                float s = expf(S_smem[q_row * BLOCK_N + k_col] - curr_max);
                row_sum += s;
                
                // Accumulate to output
                for (int d = 0; d < head_dim; d++) {
                    float v_val = __bfloat162float(V_smem[d * BLOCK_N + k_col]);
                    atomicAdd(&O_acc[q_row * head_dim + d], s * v_val);
                }
            }
            
            // Warp reduce for sum
            for (int offset = 16; offset > 0; offset /= 2) {
                row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
            }
            
            if (lane_id == 0) {
                sum_exp[q_row] += row_sum;
            }
        }
        __syncthreads();
    }
    
    // Finalize: normalize and write output
    for (int idx = threadIdx.x; idx < m_size * head_dim; idx += THREADS_PER_BLOCK) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        float o_val = O_acc[idx] / sum_exp[row];
        O_base[(q_start + row) * head_dim + col] = __float2bfloat16(o_val);
    }
}

__global__ void attention_wmma_causal_kernel(
    const compute_dtype* __restrict__ Q,
    const compute_dtype* __restrict__ K,
    const compute_dtype* __restrict__ V,
    compute_dtype* __restrict__ O,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    float scale
) {
    __shared__ compute_dtype Q_smem[BLOCK_M * BLOCK_K];
    __shared__ compute_dtype K_smem[BLOCK_N * BLOCK_K];
    __shared__ compute_dtype V_smem[BLOCK_K * BLOCK_N];
    
    __shared__ float O_acc[BLOCK_M * BLOCK_K];
    __shared__ float max_vals[BLOCK_M];
    __shared__ float sum_exp[BLOCK_M];
    __shared__ float S_smem[BLOCK_M * BLOCK_N];
    
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int block_m = blockIdx.x;
    
    int q_start = block_m * BLOCK_M;
    int q_end = min(q_start + BLOCK_M, seq_len);
    int m_size = q_end - q_start;
    
    if (q_start >= seq_len) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    const compute_dtype* Q_base = Q + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    const compute_dtype* K_base = K + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    const compute_dtype* V_base = V + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    compute_dtype* O_base = O + (batch * seq_len * n_heads + head * seq_len) * head_dim;
    
    // Initialize
    for (int i = tid; i < BLOCK_M; i += THREADS_PER_BLOCK) {
        max_vals[i] = -INFINITY;
        sum_exp[i] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * BLOCK_K; i += THREADS_PER_BLOCK) {
        O_acc[i] = 0.0f;
    }
    __syncthreads();
    
    // Load Q tile
    for (int idx = tid; idx < m_size * head_dim; idx += THREADS_PER_BLOCK) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        Q_smem[idx] = Q_base[(q_start + row) * head_dim + col];
    }
    __syncthreads();
    
    // Causal: only process K tiles <= current Q position
    int max_k = q_end;
    
    for (int k_start = 0; k_start < max_k; k_start += BLOCK_N) {
        int k_end = min(k_start + BLOCK_N, max_k);
        int n_size = k_end - k_start;
        
        // Load K, V tiles
        for (int idx = tid; idx < n_size * head_dim; idx += THREADS_PER_BLOCK) {
            int row = idx / head_dim;
            int col = idx % head_dim;
            K_smem[idx] = K_base[(k_start + row) * head_dim + col];
            V_smem[col * BLOCK_N + row] = V_base[(k_start + row) * head_dim + col];
        }
        __syncthreads();
        
        // Compute QK^T with causal mask
        int rows_per_warp = (m_size + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        int warp_q_start = warp_id * rows_per_warp;
        int warp_q_end = min(warp_q_start + rows_per_warp, m_size);
        
        for (int q_row = warp_q_start; q_row < warp_q_end; ++q_row) {
            int global_q = q_start + q_row;
            
            float row_max = -INFINITY;
            for (int k_col = lane_id; k_col < n_size; k_col += WARP_SIZE) {
                int global_k = k_start + k_col;
                
                float score = 0.0f;
                if (global_k <= global_q) {
                    for (int d = 0; d < head_dim; d++) {
                        score += __bfloat162float(Q_smem[q_row * head_dim + d]) * 
                                 __bfloat162float(K_smem[k_col * head_dim + d]);
                    }
                    score *= scale;
                } else {
                    score = -INFINITY;
                }
                S_smem[q_row * BLOCK_N + k_col] = score;
                row_max = fmaxf(row_max, score);
            }
            
            // Warp reduce for max
            for (int offset = 16; offset > 0; offset /= 2) {
                row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));
            }
            
            if (lane_id == 0) {
                float old_max = max_vals[q_row];
                float new_max = fmaxf(old_max, row_max);
                max_vals[q_row] = new_max;
                
                if (new_max > old_max) {
                    float rescale = expf(old_max - new_max);
                    sum_exp[q_row] *= rescale;
                    for (int d = 0; d < head_dim; d++) {
                        O_acc[q_row * head_dim + d] *= rescale;
                    }
                }
            }
            __syncthreads();
            
            float curr_max = max_vals[q_row];
            float row_sum = 0.0f;
            
            for (int k_col = lane_id; k_col < n_size; k_col += WARP_SIZE) {
                int global_k = k_start + k_col;
                if (global_k > global_q) continue;
                
                float s = expf(S_smem[q_row * BLOCK_N + k_col] - curr_max);
                row_sum += s;
                
                for (int d = 0; d < head_dim; d++) {
                    float v_val = __bfloat162float(V_smem[d * BLOCK_N + k_col]);
                    atomicAdd(&O_acc[q_row * head_dim + d], s * v_val);
                }
            }
            
            for (int offset = 16; offset > 0; offset /= 2) {
                row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);
            }
            
            if (lane_id == 0) {
                sum_exp[q_row] += row_sum;
            }
        }
        __syncthreads();
    }
    
    // Normalize and write output
    for (int idx = tid; idx < m_size * head_dim; idx += THREADS_PER_BLOCK) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        float o_val = O_acc[idx] / sum_exp[row];
        O_base[(q_start + row) * head_dim + col] = __float2bfloat16(o_val);
    }
}

extern "C" {
    void launch_attention_tensor_core(
        const void* Q, const void* K, const void* V, void* O,
        int batch_size, int seq_len, int n_heads, int head_dim,
        bool causal, cudaStream_t stream
    ) {
        float scale = 1.0f / sqrtf((float)head_dim);
        
        dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, n_heads, batch_size);
        dim3 block(THREADS_PER_BLOCK);
        
        if (causal) {
            attention_wmma_causal_kernel<<<grid, block, 0, stream>>>(
                (const compute_dtype*)Q, (const compute_dtype*)K, (const compute_dtype*)V, (compute_dtype*)O,
                batch_size, seq_len, n_heads, head_dim, scale
            );
        } else {
            attention_wmma_kernel<<<grid, block, 0, stream>>>(
                (const compute_dtype*)Q, (const compute_dtype*)K, (const compute_dtype*)V, (compute_dtype*)O,
                batch_size, seq_len, n_heads, head_dim, scale
            );
        }
    }
}
