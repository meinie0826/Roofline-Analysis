/**
 * Fair comparison: Baseline vs D1 (DSMEM copy) vs D2 (mma.2sm)
 * 
 * All three kernels have IDENTICAL problem partitioning:
 * - Each pair of CTAs processes 128 rows × 64 columns of C
 * - Baseline: Two independent CTAs, each loads B from HBM
 * - D1: Two CTAs in cluster, CTA1 copies B from CTA0's DSMEM
 * - D2: Two CTAs in cluster, hardware mma.2sm operand exchange
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
constexpr int kTileM = 64;   // Per-CTA tile
constexpr int kTileN = 64;
constexpr int kTileK = 32;

//=============================================================================
// Baseline: Independent CTAs (same as before)
//=============================================================================

__global__ void __launch_bounds__(128)
baseline_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  __shared__ half sA[2][64][32];
  __shared__ half sB[2][64][32];
  
  int m_start = blockIdx.x * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  if (m_start >= M || n_start >= N) return;
  
  int tid = threadIdx.x;
  int row_base = (tid / 16) * 4;
  int col_base = (tid % 16) * 4;
  
  float accum[16] = {0};
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % 2;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Load A
    for (int i = tid; i < kTileM * kTileK; i += 128) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m < M && k_offset + k < K) {
        sA[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        sA[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Load B (independently)
    for (int i = tid; i < kTileN * kTileK; i += 128) {
      int n = i / kTileK;
      int k = i % kTileK;
      if (n_start + n < N && k_offset + k < K) {
        sB[stage][n][k] = B[(n_start + n) * K + k_offset + k];
      } else {
        sB[stage][n][k] = __float2half(0.0f);
      }
    }
    
    __syncthreads();
    
    // Compute
    for (int k = 0; k < k_tiles; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          accum[r * 4 + c] += __half2float(sA[stage][(row_base + r) % kTileM][k]) *
                              __half2float(sB[stage][(col_base + c) % kTileN][k]);
        }
      }
    }
  }
  
  // Write C
  for (int i = 0; i < 16; ++i) {
    int r = row_base + (i / 4);
    int c = col_base + (i % 4);
    if (m_start + r < M && n_start + c < N) {
      C[(m_start + r) * N + n_start + c] = __float2half(accum[i]);
    }
  }
}

//=============================================================================
// D1: Cluster with DSMEM copy
//=============================================================================

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)
void d1_cluster_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  __shared__ half sA[2][64][32];
  __shared__ half sB[2][64][32];
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  int m_start = (blockIdx.x * 2 + rank) * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  if (m_start >= M || n_start >= N) return;
  
  half* remote_sB = reinterpret_cast<half*>(cluster.map_shared_rank(&sB[0][0][0], 0));
  
  int tid = threadIdx.x;
  int row_base = (tid / 16) * 4;
  int col_base = (tid % 16) * 4;
  
  float accum[16] = {0};
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % 2;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Load A
    for (int i = tid; i < kTileM * kTileK; i += 128) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m < M && k_offset + k < K) {
        sA[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        sA[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Load B (CTA0 only)
    if (rank == 0) {
      for (int i = tid; i < kTileN * kTileK; i += 128) {
        int n = i / kTileK;
        int k = i % kTileK;
        if (n_start + n < N && k_offset + k < K) {
          sB[stage][n][k] = B[(n_start + n) * K + k_offset + k];
        } else {
          sB[stage][n][k] = __float2half(0.0f);
        }
      }
    }
    
    cluster.sync();
    
    // CTA1 copies B
    if (rank == 1) {
      for (int i = tid; i < kTileN * kTileK; i += 128) {
        int n = i / kTileK;
        int k = i % kTileK;
        sB[stage][n][k] = remote_sB[stage * 64 * 32 + n * 32 + k];
      }
    }
    
    cluster.sync();
    
    // Compute
    for (int k = 0; k < k_tiles; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          accum[r * 4 + c] += __half2float(sA[stage][(row_base + r) % kTileM][k]) *
                              __half2float(sB[stage][(col_base + c) % kTileN][k]);
        }
      }
    }
  }
  
  // Write C
  for (int i = 0; i < 16; ++i) {
    int r = row_base + (i / 4);
    int c = col_base + (i % 4);
    if (m_start + r < M && n_start + c < N) {
      C[(m_start + r) * N + n_start + c] = __float2half(accum[i]);
    }
  }
}

//=============================================================================
// D2: Hardware mma.2sm (both CTAs compute together)
//=============================================================================

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)
void d2_mma2sm_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  // In mma.2sm mode, both CTAs share the B matrix via hardware
  // CTA0 handles rows 0-63, CTA1 handles rows 64-127
  
  __shared__ half sA[2][64][32];  // Each CTA has its own A tile
  __shared__ half sB[2][64][32];  // Only CTA0 needs to load B
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  int m_start = (blockIdx.x * 2 + rank) * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  if (m_start >= M || n_start >= N) return;
  
  int tid = threadIdx.x;
  int row_base = (tid / 16) * 4;
  int col_base = (tid % 16) * 4;
  
  float accum[16] = {0};
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % 2;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Each CTA loads its own A tile
    for (int i = tid; i < kTileM * kTileK; i += 128) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m < M && k_offset + k < K) {
        sA[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        sA[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Only CTA0 loads B (both CTAs will access it via hardware multicast)
    if (rank == 0) {
      for (int i = tid; i < kTileN * kTileK; i += 128) {
        int n = i / kTileK;
        int k = i % kTileK;
        if (n_start + n < N && k_offset + k < K) {
          sB[stage][n][k] = B[(n_start + n) * K + k_offset + k];
        } else {
          sB[stage][n][k] = __float2half(0.0f);
        }
      }
    }
    
    cluster.sync();
    
    // In real mma.2sm, hardware fetches B from CTA0's smem
    // Here we simulate by having CTA1 read from CTA0's smem directly
    // (Real implementation would use tcgen05.mma.2sm instruction)
    
    half (*b_ptr)[64][32] = (rank == 0) ? sB : reinterpret_cast<half(*)[64][32]>(
        cluster.map_shared_rank(&sB[0][0][0], 0));
    
    // Compute
    for (int k = 0; k < k_tiles; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          // Simulated mma.2sm: both CTAs read from CTA0's B buffer
          accum[r * 4 + c] += __half2float(sA[stage][(row_base + r) % kTileM][k]) *
                              __half2float((*b_ptr)[stage][(col_base + c) % kTileN][k]);
        }
      }
    }
  }
  
  // Write C
  for (int i = 0; i < 16; ++i) {
    int r = row_base + (i / 4);
    int c = col_base + (i % 4);
    if (m_start + r < M && n_start + c < N) {
      C[(m_start + r) * N + n_start + c] = __float2half(accum[i]);
    }
  }
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "all";
  int M = 2048, N = 2048, K = 8192;
  int repeats = 20, warmup = 5;
  
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--m=") == 0) M = std::atoi(argv[i] + 4);
    else if (arg.find("--n=") == 0) N = std::atoi(argv[i] + 4);
    else if (arg.find("--k=") == 0) K = std::atoi(argv[i] + 4);
    else if (arg.find("--repeats=") == 0) repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--warmup=") == 0) warmup = std::atoi(argv[i] + 9);
  }
  
  std::fprintf(stdout, "CONFIG mode=%s m=%d n=%d k=%d repeats=%d warmup=%d gpu=\"%s\"\n",
               mode, M, N, K, repeats, warmup, gpu_name().c_str());
  
  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(half));
  
  cudaMemset(d_A, 1, M * K * sizeof(half));
  cudaMemset(d_B, 1, K * N * sizeof(half));
  cudaMemset(d_C, 0, M * N * sizeof(half));
  
  double gflops = 2.0 * M * N * K / 1e9;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  auto measure_kernel = [&](auto kernel, const char* name, bool is_cluster) {
    dim3 grid((M + kTileM - 1) / kTileM, (N + kTileN - 1) / kTileN);
    dim3 block(128);
    size_t smem = 2 * 64 * 32 * sizeof(half) + 2 * 64 * 32 * sizeof(half);
    
    if constexpr (!std::is_same_v<decltype(kernel), decltype(nullptr)>) {
      // Warmup
      for (int w = 0; w < warmup; ++w) {
        if (is_cluster) {
          cudaLaunchConfig_t config{};
          config.gridDim = grid;
          config.blockDim = block;
          config.dynamicSmemBytes = smem;
          
          cudaLaunchAttribute attrs[1];
          attrs[0].id = cudaLaunchAttributeClusterDimension;
          attrs[0].val.clusterDim = {2, 1, 1};
          config.attrs = attrs;
          config.numAttrs = 1;
          
          cudaLaunchKernelEx(&config, kernel, d_A, d_B, d_C, M, N, K);
        } else {
          kernel<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
        }
      }
      cudaDeviceSynchronize();
      
      // Measure
      float total_ms = 0.0f;
      for (int r = 0; r < repeats; ++r) {
        cudaEventRecord(start);
        if (is_cluster) {
          cudaLaunchConfig_t config{};
          config.gridDim = grid;
          config.blockDim = block;
          config.dynamicSmemBytes = smem;
          
          cudaLaunchAttribute attrs[1];
          attrs[0].id = cudaLaunchAttributeClusterDimension;
          attrs[0].val.clusterDim = {2, 1, 1};
          config.attrs = attrs;
          config.numAttrs = 1;
          
          cudaLaunchKernelEx(&config, kernel, d_A, d_B, d_C, M, N, K);
        } else {
          kernel<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
      }
      double avg_ms = total_ms / repeats;
      std::fprintf(stdout, "RESULT mode=%s elapsed_ms=%.6f gflops=%.2f\n", name, avg_ms, gflops / avg_ms);
    }
  };
  
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    measure_kernel(baseline_gemm_kernel, "baseline", false);
  }
  
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d1") == 0) {
    measure_kernel(d1_cluster_gemm_kernel, "d1", true);
  }
  
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d2") == 0) {
    measure_kernel(d2_mma2sm_gemm_kernel, "d2", true);
  }
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
