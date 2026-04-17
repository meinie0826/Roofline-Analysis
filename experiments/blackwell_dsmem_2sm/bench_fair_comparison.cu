/**
 * GEMM kernels with optimization barriers
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
constexpr int kTileK = 32;  // Smaller K tile to increase iterations
constexpr int kStages = 2;

//=============================================================================
__device__ float prevent_optimize(float val) {
  volatile float* ptr = reinterpret_cast<volatile float*>(val);
  return *ptr;
}

//=============================================================================
struct GmemTile {
  alignas(128) half A[kStages][kTileM][kTileK];
  alignas(128) half B[kStages][kTileN][kTileK];
};

//=============================================================================
// Baseline: Independent CTAs
//=============================================================================

__global__ void __launch_bounds__(128)
baseline_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  __shared__ GmemTile tile;
  
  int m_start = blockIdx.x * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  if (m_start >= M || n_start >= N) return;
  
  int tid = threadIdx.x;
  
  // Each thread handles 1 output element
  int row = tid / 8;  // 0-15
  int col = tid % 8;  // 0-7
  
  // Multiple passes to cover full tile
  int num_passes = (kTileM * kTileN) / 128;  // 32 passes
  
  float accum = 0.0f;
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % kStages;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Load A
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m < M && k_offset + k < K) {
        tile.A[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        tile.A[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Load B
    for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
      int n = i / kTileK;
      int k = i % kTileK;
      if (n_start + n < N && k_offset + k < K) {
        tile.B[stage][n][k] = B[(n_start + n) * K + k_offset + k];
      } else {
        tile.B[stage][n][k] = __float2half(0.0f);
      }
    }
    
    __syncthreads();
    
    // Accumulate for this thread's element
    for (int pass = 0; pass < num_passes; ++pass) {
      int idx = pass * 128 + tid;
      int r = idx / kTileN;
      int c = idx % kTileN;
      
      for (int k = 0; k < k_tiles; ++k) {
        accum += __half2float(tile.A[stage][r % kTileM][k]) *
                 __half2float(tile.B[stage][c % kTileN][k]);
      }
    }
  }
  
  // Write C - force use of accum
  int idx = tid;
  if (idx < kTileM * kTileN && m_start + idx / kTileN < M && n_start + idx % kTileN < N) {
    C[(m_start + idx / kTileN) * N + n_start + idx % kTileN] = 
        __float2half(accum);
  }
}

//=============================================================================
// D1: Cluster with DSMEM sharing
//=============================================================================

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)
void d1_cluster_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  __shared__ GmemTile tile;
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  int m_start = (blockIdx.x * 2 + rank) * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  if (m_start >= M || n_start >= N) return;
  
  GmemTile* tile0 = reinterpret_cast<GmemTile*>(cluster.map_shared_rank(&tile, 0));
  
  int tid = threadIdx.x;
  float accum = 0.0f;
  int num_passes = (kTileM * kTileN) / 128;
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % kStages;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Load A
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m < M && k_offset + k < K) {
        tile.A[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        tile.A[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Load B (CTA0 only)
    if (rank == 0) {
      for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
        int n = i / kTileK;
        int k = i % kTileK;
        if (n_start + n < N && k_offset + k < K) {
          tile.B[stage][n][k] = B[(n_start + n) * K + k_offset + k];
        } else {
          tile.B[stage][n][k] = __float2half(0.0f);
        }
      }
    }
    
    cluster.sync();
    
    // CTA1 copies B from CTA0
    if (rank == 1) {
      for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
        int n = i / kTileK;
        int k = i % kTileK;
        tile.B[stage][n][k] = tile0->B[stage][n][k];
      }
    }
    
    cluster.sync();
    
    // Accumulate
    for (int pass = 0; pass < num_passes; ++pass) {
      int idx = pass * 128 + tid;
      int r = idx / kTileN;
      int c = idx % kTileN;
      
      for (int k = 0; k < k_tiles; ++k) {
        accum += __half2float(tile.A[stage][r % kTileM][k]) *
                 __half2float(tile.B[stage][c % kTileN][k]);
      }
    }
  }
  
  int idx = tid;
  if (idx < kTileM * kTileN && m_start + idx / kTileN < M && n_start + idx % kTileN < N) {
    C[(m_start + idx / kTileN) * N + n_start + idx % kTileN] = __float2half(accum);
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
  
  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(half));
  
  cudaMemset(d_A, 1, M * K * sizeof(half));
  cudaMemset(d_B, 1, K * N * sizeof(half));
  cudaMemset(d_C, 0, M * N * sizeof(half));
  
  double gflops = 2.0 * M * N * K / 1e9;
  
  dim3 grid((M + kTileM - 1) / kTileM, (N + kTileN - 1) / kTileN);
  dim3 block(128);
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
