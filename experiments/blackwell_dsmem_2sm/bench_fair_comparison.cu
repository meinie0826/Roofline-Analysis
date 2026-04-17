/**
 * Complete GEMM kernels with proper work distribution
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 64;
constexpr int kStages = 2;

//=============================================================================
struct GmemTile {
  alignas(128) half A[kStages][kTileM][kTileK];
  alignas(128) half B[kStages][kTileN][kTileK];
};

//=============================================================================
// Baseline: Independent CTAs, each loads B from HBM
//=============================================================================

__global__ void __launch_bounds__(64)
baseline_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  __shared__ GmemTile tile;
  
  int m_tile = blockIdx.x;
  int n_tile = blockIdx.y;
  int m_start = m_tile * kTileM;
  int n_start = n_tile * kTileN;
  
  if (m_start >= M || n_start >= N) return;
  
  int tid = threadIdx.x;
  
  // Each thread handles 4x4 elements = 16 accumulators
  int row_tile = tid / 8;      // 0-7
  int col_tile = tid % 8;      // 0-7
  
  int row_start = row_tile * 8;   // Each thread handles 8 rows
  int col_start = col_tile * 8;   // Each thread handles 8 cols
  
  float accum[64] = {0};  // 8x8 = 64 accumulators per thread
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % kStages;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Cooperative load A
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK;
      int k = i % kTileK;
      half val = __float2half(0.0f);
      if (m_start + m < M && k_offset + k < K) {
        val = A[(m_start + m) * K + k_offset + k];
      }
      tile.A[stage][m][k] = val;
    }
    
    // Cooperative load B (each CTA loads independently)
    for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
      int n = i / kTileK;
      int k = i % kTileK;
      half val = __float2half(0.0f);
      if (n_start + n < N && k_offset + k < K) {
        val = B[(n_start + n) * K + k_offset + k];
      }
      tile.B[stage][n][k] = val;
    }
    
    __syncthreads();
    
    // Accumulate
    for (int k = 0; k < k_tiles; ++k) {
      #pragma unroll
      for (int r = 0; r < 8; ++r) {
        int row = row_start + r;
        half a_val = tile.A[stage][row % kTileM][k];
        float a_f = __half2float(a_val);
        
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
          int col = col_start + c;
          half b_val = tile.B[stage][col % kTileN][k];
          accum[r * 8 + c] += a_f * __half2float(b_val);
        }
      }
    }
  }
  
  // Write C
  #pragma unroll
  for (int r = 0; r < 8; ++r) {
    int row = row_start + r;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      int col = col_start + c;
      if (m_start + row < M && n_start + col < N) {
        C[(m_start + row) * N + n_start + col] = __float2half(accum[r * 8 + c]);
      }
    }
  }
}

//=============================================================================
// D1: Cluster with DSMEM B sharing
//=============================================================================

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(64)
void d1_cluster_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  __shared__ GmemTile tile;
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  // Two CTAs process adjacent tiles in M dimension
  int m_start = (blockIdx.x * 2 + rank) * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  if (m_start >= M || n_start >= N) return;
  
  // Pointer to CTA0's smem
  GmemTile* tile0 = reinterpret_cast<GmemTile*>(cluster.map_shared_rank(&tile, 0));
  
  int tid = threadIdx.x;
  int row_tile = tid / 8;
  int col_tile = tid % 8;
  int row_start = row_tile * 8;
  int col_start = col_tile * 8;
  
  float accum[64] = {0};
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % kStages;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Load A (each CTA loads its own)
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK;
      int k = i % kTileK;
      half val = __float2half(0.0f);
      if (m_start + m < M && k_offset + k < K) {
        val = A[(m_start + m) * K + k_offset + k];
      }
      tile.A[stage][m][k] = val;
    }
    
    // Load B (CTA0 only from HBM)
    if (rank == 0) {
      for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
        int n = i / kTileK;
        int k = i % kTileK;
        half val = __float2half(0.0f);
        if (n_start + n < N && k_offset + k < K) {
          val = B[(n_start + n) * K + k_offset + k];
        }
        tile.B[stage][n][k] = val;
      }
    }
    
    cluster.sync();
    
    // CTA1 copies B from CTA0's DSMEM
    if (rank == 1) {
      for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
        int n = i / kTileK;
        int k = i % kTileK;
        tile.B[stage][n][k] = tile0->B[stage][n][k];
      }
    }
    
    cluster.sync();
    
    // Accumulate (same as baseline)
    for (int k = 0; k < k_tiles; ++k) {
      #pragma unroll
      for (int r = 0; r < 8; ++r) {
        int row = row_start + r;
        half a_val = tile.A[stage][row % kTileM][k];
        float a_f = __half2float(a_val);
        
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
          int col = col_start + c;
          half b_val = tile.B[stage][col % kTileN][k];
          accum[r * 8 + c] += a_f * __half2float(b_val);
        }
      }
    }
  }
  
  // Write C
  #pragma unroll
  for (int r = 0; r < 8; ++r) {
    int row = row_start + r;
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
      int col = col_start + c;
      if (m_start + row < M && n_start + col < N) {
        C[(m_start + row) * N + n_start + col] = __float2half(accum[r * 8 + c]);
      }
    }
  }
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "all";
  int M = 1024, N = 1024, K = 4096;
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
  
  // Allocate
  half *d_A, *d_B, *d_C;
  size_t A_bytes = M * K * sizeof(half);
  size_t B_bytes = K * N * sizeof(half);
  size_t C_bytes = M * N * sizeof(half);
  
  cudaMalloc(&d_A, A_bytes);
  cudaMalloc(&d_B, B_bytes);
  cudaMalloc(&d_C, C_bytes);
  
  cudaMemset(d_A, 1, A_bytes);
  cudaMemset(d_B, 1, B_bytes);
  cudaMemset(d_C, 0, C_bytes);
  
  double gflops = 2.0 * M * N * K / 1e9;
  
  dim3 grid((M + kTileM - 1) / kTileM, (N + kTileN - 1) / kTileN);
  dim3 block(64);
  size_t smem = sizeof(GmemTile);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // Baseline
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    for (int w = 0; w < warmup; ++w) {
      baseline_gemm_kernel<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    float total_ms = 0.0f;
    for (int r = 0; r < repeats; ++r) {
      cudaEventRecord(start);
      baseline_gemm_kernel<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
    }
    double avg_ms = total_ms / repeats;
    std::fprintf(stdout, "RESULT mode=baseline elapsed_ms=%.6f gflops=%.2f\n", avg_ms, gflops / avg_ms);
  }
  
  // D1 Cluster
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d1") == 0) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;
    
    for (int w = 0; w < warmup; ++w) {
      cudaLaunchKernelEx(&config, d1_cluster_gemm_kernel, d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    float total_ms = 0.0f;
    for (int r = 0; r < repeats; ++r) {
      cudaEventRecord(start);
      cudaLaunchKernelEx(&config, d1_cluster_gemm_kernel, d_A, d_B, d_C, M, N, K);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
    }
    double avg_ms = total_ms / repeats;
    std::fprintf(stdout, "RESULT mode=d1 elapsed_ms=%.6f gflops=%.2f\n", avg_ms, gflops / avg_ms);
  }
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
