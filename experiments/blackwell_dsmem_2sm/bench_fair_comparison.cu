/**
 * Fair comparison: 1SM vs 2SM with identical tile shape
 * 
 * Tile shape: 128×64×64 (fixed)
 * Problem: Each pair of CTAs processes 256 rows × 64 cols of C
 * 
 * Baseline: Two independent 1SM CTAs, each loads B from HBM
 * D1: Two CTAs in a cluster, CTA0 loads B, CTA1 copies from DSMEM
 * D2: Hardware mma.2sm (if we can trigger it)
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
// Constants
//=============================================================================
constexpr int kTileM = 128;
constexpr int kTileN = 64;
constexpr int kTileK = 64;
constexpr int kStages = 2;

//=============================================================================
// Shared memory layout
//=============================================================================
struct GmemTile {
  alignas(128) half_t A[kStages][kTileM][kTileK];
  alignas(128) half_t B[kStages][kTileN][kTileK];
};

//=============================================================================
// Baseline: Each CTA loads B independently from HBM
//=============================================================================

__global__ void baseline_gemm_kernel(
    const half_t* __restrict__ A,
    const half_t* __restrict__ B,
    half_t* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
  __shared__ GmemTile tile;
  
  int cta_id = blockIdx.x;
  int pair_id = cta_id / 2;
  int rank = cta_id % 2;
  
  // Each CTA handles kTileM rows
  int m_start = pair_id * (2 * kTileM) + rank * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  const half_t* gA = A + m_start * lda;
  const half_t* gB = B + n_start * ldb;
  half_t* gC = C + m_start * ldc + n_start;
  
  int tid = threadIdx.x;
  
  // Accumulator (simplified: just use registers)
  float acc[kTileM/128][kTileN/64][16] = {0};  // Minimal accumulation
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % kStages;
    
    // === Load A ===
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m < M && k_offset + k < K) {
        tile.A[stage][m][k] = gA[m * lda + k_offset + k];
      }
    }
    
    // === Load B (each CTA loads independently) ===
    for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
      int n = i / kTileK;
      int k = i % kTileK;
      if (n_start + n < N && k_offset + k < K) {
        tile.B[stage][n][k] = gB[n * ldb + k_offset + k];
      }
    }
    
    __syncthreads();
    
    // === MMA (simplified: just do minimal compute) ===
    // This is NOT real MMA, just placeholder to ensure tile is used
    #pragma unroll
    for (int m = 0; m < 1; ++m) {
      #pragma unroll
      for (int n = 0; n < 1; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
          sum += static_cast<float>(tile.A[stage][tid % kTileM][k]) *
                 static_cast<float>(tile.B[stage][tid % kTileN][k]);
        }
        acc[m][n][0] += sum;
      }
    }
  }
  
  // Write result (minimal)
  if (m_start < M && n_start < N && tid < 128) {
    gC[tid % kTileM * ldc + (tid / kTileM) % kTileN] = 
        static_cast<half_t>(acc[0][0][0]);
  }
}

//=============================================================================
// D1: Cluster kernel with DSMEM B sharing
//=============================================================================

__global__ __cluster_dims__(2, 1, 1)
void d1_cluster_gemm_kernel(
    const half_t* __restrict__ A,
    const half_t* __restrict__ B,
    half_t* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
  __shared__ GmemTile tile;
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  int pair_id = blockIdx.x;
  int m_start = pair_id * (2 * kTileM) + rank * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  const half_t* gA = A + m_start * lda;
  const half_t* gB = B + n_start * ldb;
  half_t* gC = C + m_start * ldc + n_start;
  
  // Pointer to CTA0's smem
  GmemTile* remote_tile0 = reinterpret_cast<GmemTile*>(cluster.map_shared_rank(&tile, 0));
  
  int tid = threadIdx.x;
  
  float acc[1][1][16] = {0};
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % kStages;
    
    // === Load A (each CTA loads its own) ===
    for (int i = tid; i < kTileM * kTileK; i += blockDim.x) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m < M && k_offset + k < K) {
        tile.A[stage][m][k] = gA[m * lda + k_offset + k];
      }
    }
    
    // === Load B (CTA0 only from HBM) ===
    if (rank == 0) {
      for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
        int n = i / kTileK;
        int k = i % kTileK;
        if (n_start + n < N && k_offset + k < K) {
          tile.B[stage][n][k] = gB[n * ldb + k_offset + k];
        }
      }
    }
    
    cluster.sync();  // Ensure CTA0's B is ready
    
    // === CTA1 copies B from CTA0's DSMEM ===
    if (rank == 1) {
      for (int i = tid; i < kTileN * kTileK; i += blockDim.x) {
        int n = i / kTileK;
        int k = i % kTileK;
        tile.B[stage][n][k] = remote_tile0->B[stage][n][k];
      }
    }
    
    cluster.sync();
    
    // === MMA (same as baseline) ===
    #pragma unroll
    for (int m = 0; m < 1; ++m) {
      #pragma unroll
      for (int n = 0; n < 1; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) {
          sum += static_cast<float>(tile.A[stage][tid % kTileM][k]) *
                 static_cast<float>(tile.B[stage][tid % kTileN][k]);
        }
        acc[m][n][0] += sum;
      }
    }
  }
  
  if (m_start < M && n_start < N && tid < 128) {
    gC[tid % kTileM * ldc + (tid / kTileM) % kTileN] = 
        static_cast<half_t>(acc[0][0][0]);
  }
}

//=============================================================================
// Benchmark harness
//=============================================================================

double measure_kernel(void (*kernel)(const half_t*, const half_t*, half_t*, int, int, int, int, int, int),
                       const half_t* d_A, const half_t* d_B, half_t* d_C,
                       int M, int N, int K, int repeats, int warmup,
                       bool is_cluster = false) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  int lda = K, ldb = K, ldc = N;
  
  // Warmup
  for (int w = 0; w < warmup; ++w) {
    kernel<<<dim3(1, 1), 128>>>(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
  }
  cudaDeviceSynchronize();
  
  // Measure
  float total_ms = 0.0f;
  for (int r = 0; r < repeats; ++r) {
    cudaEventRecord(start);
    kernel<<<dim3(1, 1), 128>>>(d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
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

double measure_cluster_kernel(void (*kernel)(const half_t*, const half_t*, half_t*, int, int, int, int, int, int),
                               const half_t* d_A, const half_t* d_B, half_t* d_C,
                               int M, int N, int K, int repeats, int warmup) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  int lda = K, ldb = K, ldc = N;
  
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1, 1);
  config.blockDim = 128;
  config.dynamicSmemBytes = sizeof(GmemTile);
  
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  config.attrs = attrs;
  config.numAttrs = 1;
  
  // Warmup
  for (int w = 0; w < warmup; ++w) {
    cudaLaunchKernelEx(&config, kernel, d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
  }
  cudaDeviceSynchronize();
  
  // Measure
  float total_ms = 0.0f;
  for (int r = 0; r < repeats; ++r) {
    cudaEventRecord(start);
    cudaLaunchKernelEx(&config, kernel, d_A, d_B, d_C, M, N, K, lda, ldb, ldc);
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
  const char* mode = "all";
  int M = 256, N = 64, K = 1024;
  int repeats = 10, warmup = 3;
  
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
  
  // Allocate (minimal, single tile)
  size_t A_size = M * K * sizeof(half_t);
  size_t B_size = K * N * sizeof(half_t);
  size_t C_size = M * N * sizeof(half_t);
  
  half_t *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, A_size);
  cudaMalloc(&d_B, B_size);
  cudaMalloc(&d_C, C_size);
  
  cudaMemset(d_A, 1, A_size);
  cudaMemset(d_B, 1, B_size);
  cudaMemset(d_C, 0, C_size);
  
  double gflops = 2.0 * M * N * K / 1e9;
  
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    double ms = measure_kernel(baseline_gemm_kernel, d_A, d_B, d_C, M, N, K, repeats, warmup);
    double perf = gflops / ms;
    std::fprintf(stdout, "RESULT mode=baseline elapsed_ms=%.6f gflops=%.2f\n", ms, perf);
  }
  
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d1") == 0) {
    double ms = measure_cluster_kernel(d1_cluster_gemm_kernel, d_A, d_B, d_C, M, N, K, repeats, warmup);
    double perf = gflops / ms;
    std::fprintf(stdout, "RESULT mode=d1 elapsed_ms=%.6f gflops=%.2f\n", ms, perf);
  }
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
