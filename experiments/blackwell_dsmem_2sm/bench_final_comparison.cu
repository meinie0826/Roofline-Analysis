/**
 * Comprehensive comparison:
 * - Baseline: CUTLASS 1SM (tile 128×64×32)
 * - D1: Custom cluster kernel with DSMEM copy (tile 128×64×32)
 * - D2: CUTLASS 2SM with real mma.2sm (tile 256×64×32)
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdio>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
// CUTLASS type aliases
//=============================================================================
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

//=============================================================================
// Helper to run CUTLASS kernel
//=============================================================================
template<typename Gemm>
double run_cutlass(const half* d_A, const half* d_B, half* d_C,
                   int M, int N, int K, int repeats, int warmup) {
  using StrideA = cutlass::detail::TagToStrideB<LayoutA>::type;
  using StrideB = cutlass::detail::TagToStrideB<LayoutB>::type;
  using StrideC = cutlass::detail::TagToStrideB<LayoutC>::type;
  
  typename Gemm::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_size = {M, N, K};
  args.ptr_A = reinterpret_cast<const ElementA*>(d_A);
  args.ptr_B = reinterpret_cast<const ElementB*>(d_B);
  args.ptr_C = reinterpret_cast<ElementC*>(d_C);
  args.ptr_D = reinterpret_cast<ElementC*>(d_C);
  args.dA = StrideA(K);
  args.dB = StrideB(K);
  args.dC = StrideC(N);
  args.dD = StrideC(N);
  
  Gemm gemm;
  auto status = gemm.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS initialization failed\n");
    return -1.0;
  }
  
  // Warmup
  for (int w = 0; w < warmup; ++w) {
    gemm.run();
  }
  cudaDeviceSynchronize();
  
  // Measure
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  float total_ms = 0.0f;
  for (int r = 0; r < repeats; ++r) {
    cudaEventRecord(start);
    gemm.run();
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
  
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  
  // Baseline: CUTLASS 1SM (tile 128×64)
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<
      cutlass::gemm::kernel::GemmUniversal<
        cutlass::gemm::collective::CollectiveMma<
          cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
          cutlass::gemm::GemmShape<128, 64, 32>,
          ElementA, LayoutA,
          ElementB, LayoutB,
          ElementAccumulator, LayoutC
        >,
        cutlass::epilogue::collective::CollectiveEpilogue<
          cutlass::epilogue::TmaWarpSpecialized1Sm,
          cutlass::gemm::GemmShape<128, 64, 32>,
          ElementAccumulator, ElementC, LayoutC, ElementC, LayoutC
        >
      >
    >;
    
    double ms = run_cutlass<Gemm1SM>(d_A, d_B, d_C, M, N, K, repeats, warmup);
    if (ms > 0) {
      std::fprintf(stdout, "RESULT mode=baseline elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
    }
  }
  
  // D1: Custom DSMEM copy (tile 128×64)
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d1") == 0) {
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
  
  // D2: CUTLASS 2SM (tile 256×64, real mma.2sm)
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d2") == 0) {
    using Gemm2SM = cutlass::gemm::device::GemmUniversalAdapter<
      cutlass::gemm::kernel::GemmUniversal<
        cutlass::gemm::collective::CollectiveMma<
          cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
          cutlass::gemm::GemmShape<256, 64, 32>,
          ElementA, LayoutA,
          ElementB, LayoutB,
          ElementAccumulator, LayoutC
        >,
        cutlass::epilogue::collective::CollectiveEpilogue<
          cutlass::epilogue::TmaWarpSpecialized2Sm,
          cutlass::gemm::GemmShape<256, 64, 32>,
          ElementAccumulator, ElementC, LayoutC, ElementC, LayoutC
        >
      >
    >;
    
    double ms = run_cutlass<Gemm2SM>(d_A, d_B, d_C, M, N, K, repeats, warmup);
    if (ms > 0) {
      std::fprintf(stdout, "RESULT mode=d2 elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
    }
  }
  
#else
  std::fprintf(stderr, "CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined\n");
#endif
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
