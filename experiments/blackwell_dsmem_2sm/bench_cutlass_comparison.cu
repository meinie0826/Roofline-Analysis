/**
 * Fair comparison using real CUTLASS kernels
 * 
 * - Baseline: CUTLASS 1SM TMA kernel (tile: 128×64×32)
 * - D1: Custom cluster kernel with DSMEM copy (tile: 128×64×32)  
 * - D2: CUTLASS 2SM TMA kernel (tile: 256×64×32, uses mma.2sm)
 * 
 * Note: D2 tile shape is necessarily larger because mma.2sm requires
 * two CTAs to work together on a 256×M tile.
 */

#include "common.h"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/epilogue/collective/collective_epilogue.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/packed_stride.hpp>

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
double run_cutlass_gemm(const half* d_A, const half* d_B, half* d_C,
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
  if (gemm(args) != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS kernel failed to initialize\n");
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
  
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  
  // Baseline: 1SM kernel
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
    
    double ms = run_cutlass_gemm<Gemm1SM>(d_A, d_B, d_C, M, N, K, repeats, warmup);
    if (ms > 0) {
      std::fprintf(stdout, "RESULT mode=baseline elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
    }
  }
  
  // D2: 2SM kernel (real mma.2sm)
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d2") == 0) {
    using Gemm2SM = cutlass::gemm::device::GemmUniversalAdapter<
      cutlass::gemm::kernel::GemmUniversal<
        cutlass::gemm::collective::CollectiveMma<
          cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
          cutlass::gemm::GemmShape<256, 64, 32>,  // 2SM requires 256 rows
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
    
    double ms = run_cutlass_gemm<Gemm2SM>(d_A, d_B, d_C, M, N, K, repeats, warmup);
    if (ms > 0) {
      std::fprintf(stdout, "RESULT mode=d2 elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
    }
  }
  
#else
  std::fprintf(stderr, "CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined\n");
#endif
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
