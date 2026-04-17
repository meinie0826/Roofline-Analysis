/**
 * Final comparison using existing benchmarks:
 * - Baseline/D2: Use existing bench_cutlass_2sm_gemm
 * - D1: Custom cluster kernel with DSMEM copy
 * 
 * Run separately and compare results.
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
// D1: Custom cluster kernel with DSMEM copy (tile 128×64×32)
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
  
  int m_start = blockIdx.x * 128 + rank * 64;
  int n_start = blockIdx.y * 64;
  
  if (m_start >= M || n_start >= N) return;
  
  half* remote_sB = reinterpret_cast<half*>(cluster.map_shared_rank(&sB[0][0][0], 0));
  
  int tid = threadIdx.x;
  float accum[16] = {0};
  
  for (int k_offset = 0; k_offset < K; k_offset += 32) {
    int stage = (k_offset / 32) % 2;
    int k_tiles = min(32, K - k_offset);
    
    // Load A
    for (int i = tid; i < 64 * 32; i += 128) {
      int m = i / 32, k = i % 32;
      if (m_start + m < M && k_offset + k < K) {
        sA[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        sA[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Load B (CTA0 only)
    if (rank == 0) {
      for (int i = tid; i < 64 * 32; i += 128) {
        int n = i / 32, k = i % 32;
        if (n_start + n < N && k_offset + k < K) {
          sB[stage][n][k] = B[(n_start + n) * K + k_offset + k];
        } else {
          sB[stage][n][k] = __float2half(0.0f);
        }
      }
    }
    
    cluster.sync();
    
    // CTA1 copies B from CTA0
    if (rank == 1) {
      for (int i = tid; i < 64 * 32; i += 128) {
        int n = i / 32, k = i % 32;
        sB[stage][n][k] = remote_sB[stage * 64 * 32 + n * 32 + k];
      }
    }
    
    cluster.sync();
    
    // Compute
    for (int k = 0; k < k_tiles; ++k) {
      for (int i = 0; i < 16; ++i) {
        int r = (tid / 4) + ((i / 4) * 16);
        int c = ((tid % 4) * 4) + (i % 4);
        if (r < 64 && c < 64) {
          accum[i] += __half2float(sA[stage][r % 64][k]) *
                      __half2float(sB[stage][c % 64][k]);
        }
      }
    }
  }
  
  // Write C
  for (int i = 0; i < 16; ++i) {
    int r = (tid / 4) + ((i / 4) * 16);
    int c = ((tid % 4) * 4) + (i % 4);
    if (m_start + r < M && n_start + c < N) {
      C[(m_start + r) * N + n_start + c] = __float2half(accum[i]);
    }
  }
}

//=============================================================================
double measure_d1(const half* d_A, const half* d_B, half* d_C,
                 int M, int N, int K, int repeats, int warmup) {
  dim3 grid((M + 127) / 128, (N + 63) / 64);
  dim3 block(128);
  size_t smem = 2 * 64 * 32 * sizeof(half) + 2 * 64 * 32 * sizeof(half);
  
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
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
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
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  return total_ms / repeats;
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "d1";
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
  
  double ms = measure_d1(d_A, d_B, d_C, M, N, K, repeats, warmup);
  std::fprintf(stdout, "RESULT mode=d1 elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
