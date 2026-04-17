/**
 * GEMM kernels that cannot be optimized away
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
constexpr int kTileK = 32;
constexpr int kStages = 2;

//=============================================================================
struct GmemTile {
  alignas(128) half A[kStages][kTileM][kTileK];
  alignas(128) half B[kStages][kTileN][kTileK];
};

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
  
  // Each thread accumulates multiple elements
  int row_base = (tid / 16) * 4;  // 0-60 step 4
  int col_base = (tid % 16) * 4;  // 0-60 step 4
  
  float accum0 = 0.0f, accum1 = 0.0f, accum2 = 0.0f, accum3 = 0.0f;
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % kStages;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Load A cooperatively
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m < M && k_offset + k < K) {
        tile.A[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        tile.A[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Load B cooperatively
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
    
    // Accumulate
    for (int k = 0; k < k_tiles; ++k) {
      half a_vals[4], b_vals[4];
      
      // Load 4 A values
      a_vals[0] = tile.A[stage][row_base][k];
      a_vals[1] = tile.A[stage][row_base + 1][k];
      a_vals[2] = tile.A[stage][row_base + 2][k];
      a_vals[3] = tile.A[stage][row_base + 3][k];
      
      // Load 4 B values
      b_vals[0] = tile.B[stage][col_base][k];
      b_vals[1] = tile.B[stage][col_base + 1][k];
      b_vals[2] = tile.B[stage][col_base + 2][k];
      b_vals[3] = tile.B[stage][col_base + 3][k];
      
      // 4x4 outer product = 16 MACs
      accum0 += __half2float(a_vals[0]) * __half2float(b_vals[0]);
      accum1 += __half2float(a_vals[0]) * __half2float(b_vals[1]);
      accum2 += __half2float(a_vals[0]) * __half2float(b_vals[2]);
      accum3 += __half2float(a_vals[0]) * __half2float(b_vals[3]);
      
      accum0 += __half2float(a_vals[1]) * __half2float(b_vals[0]);
      accum1 += __half2float(a_vals[1]) * __half2float(b_vals[1]);
      accum2 += __half2float(a_vals[1]) * __half2float(b_vals[2]);
      accum3 += __half2float(a_vals[1]) * __half2float(b_vals[3]);
      
      accum0 += __half2float(a_vals[2]) * __half2float(b_vals[0]);
      accum1 += __half2float(a_vals[2]) * __half2float(b_vals[1]);
      accum2 += __half2float(a_vals[2]) * __half2float(b_vals[2]);
      accum3 += __half2float(a_vals[2]) * __half2float(b_vals[3]);
      
      accum0 += __half2float(a_vals[3]) * __half2float(b_vals[0]);
      accum1 += __half2float(a_vals[3]) * __half2float(b_vals[1]);
      accum2 += __half2float(a_vals[3]) * __half2float(b_vals[2]);
      accum3 += __half2float(a_vals[3]) * __half2float(b_vals[3]);
    }
  }
  
  // Write 4x4 block
  if (m_start + row_base < M && n_start + col_base < N) {
    C[(m_start + row_base) * N + n_start + col_base] = __float2half(accum0);
    C[(m_start + row_base) * N + n_start + col_base + 1] = __float2half(accum1);
    C[(m_start + row_base) * N + n_start + col_base + 2] = __float2half(accum2);
    C[(m_start + row_base) * N + n_start + col_base + 3] = __float2half(accum3);
  }
}

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
  int row_base = (tid / 16) * 4;
  int col_base = (tid % 16) * 4;
  
  float accum0 = 0.0f, accum1 = 0.0f, accum2 = 0.0f, accum3 = 0.0f;
  
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
    
    // Accumulate (same as baseline)
    for (int k = 0; k < k_tiles; ++k) {
      half a_vals[4], b_vals[4];
      
      a_vals[0] = tile.A[stage][row_base][k];
      a_vals[1] = tile.A[stage][row_base + 1][k];
      a_vals[2] = tile.A[stage][row_base + 2][k];
      a_vals[3] = tile.A[stage][row_base + 3][k];
      
      b_vals[0] = tile.B[stage][col_base][k];
      b_vals[1] = tile.B[stage][col_base + 1][k];
      b_vals[2] = tile.B[stage][col_base + 2][k];
      b_vals[3] = tile.B[stage][col_base + 3][k];
      
      accum0 += __half2float(a_vals[0]) * __half2float(b_vals[0]);
      accum1 += __half2float(a_vals[0]) * __half2float(b_vals[1]);
      accum2 += __half2float(a_vals[0]) * __half2float(b_vals[2]);
      accum3 += __half2float(a_vals[0]) * __half2float(b_vals[3]);
      
      accum0 += __half2float(a_vals[1]) * __half2float(b_vals[0]);
      accum1 += __half2float(a_vals[1]) * __half2float(b_vals[1]);
      accum2 += __half2float(a_vals[1]) * __half2float(b_vals[2]);
      accum3 += __half2float(a_vals[1]) * __half2float(b_vals[3]);
      
      accum0 += __half2float(a_vals[2]) * __half2float(b_vals[0]);
      accum1 += __half2float(a_vals[2]) * __half2float(b_vals[1]);
      accum2 += __half2float(a_vals[2]) * __half2float(b_vals[2]);
      accum3 += __half2float(a_vals[2]) * __half2float(b_vals[3]);
      
      accum0 += __half2float(a_vals[3]) * __half2float(b_vals[0]);
      accum1 += __half2float(a_vals[3]) * __half2float(b_vals[1]);
      accum2 += __half2float(a_vals[3]) * __half2float(b_vals[2]);
      accum3 += __half2float(a_vals[3]) * __half2float(b_vals[3]);
    }
  }
  
  if (m_start + row_base < M && n_start + col_base < N) {
    C[(m_start + row_base) * N + n_start + col_base] = __float2half(accum0);
    C[(m_start + row_base) * N + n_start + col_base + 1] = __float2half(accum1);
    C[(m_start + row_base) * N + n_start + col_base + 2] = __float2half(accum2);
    C[(m_start + row_base) * N + n_start + col_base + 3] = __float2half(accum3);
  }
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "all";
  int M = 2048, N = 2048, K = 4096;
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
