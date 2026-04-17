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
#include "cutlass/util/device_memory.h"

// BlockFill constants
#define CUTLASS_BLOCKFILL_GRID  256
#define CUTLASS_BLOCKFILL_BLOCK 128

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
template<int TileN, class MainloopSchedule, class EpilogueSchedule, class StageCountTag>
struct CutlassRunner {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  static constexpr bool kUse2Sm = std::is_same_v<MainloopSchedule, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>;
  
  using TileShapeMNK = std::conditional_t<kUse2Sm,
    cute::Shape<cute::_256, cute::Int<TileN>, cute::_32>,
    cute::Shape<cute::_128, cute::Int<TileN>, cute::_32>>;
  using ClusterShapeMNK = cute::Shape<cute::_2, cute::_1, cute::_1>;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      TileShapeMNK,
      ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementD,
      LayoutD,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShapeMNK,
      ClusterShapeMNK,
      StageCountTag,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShape = typename Gemm::GemmKernel::ProblemShape;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  cutlass::DeviceAllocation<ElementA> A;
  cutlass::DeviceAllocation<ElementB> B;
  cutlass::DeviceAllocation<ElementC> C;
  cutlass::DeviceAllocation<ElementD> D;

  double run(int m, int n, int k, int repeats, int warmup) {
    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    A.reset(m * k);
    B.reset(k * n);
    C.reset(m * n);
    D.reset(m * n);

    // Initialize with simple fill (no random)
    cudaMemset(A.get(), 1, A.size() * sizeof(ElementA));
    cudaMemset(B.get(), 1, B.size() * sizeof(ElementB));
    cudaMemset(C.get(), 0, C.size() * sizeof(ElementC));

    typename Gemm::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = ProblemShape{m, n, k, 1};
    args.ptr_A = A.get();
    args.ptr_B = B.get();
    args.ptr_C = C.get();
    args.ptr_D = D.get();
    args.dA = stride_a;
    args.dB = stride_b;
    args.dC = stride_c;
    args.dD = stride_d;

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
};

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
  
  double gflops = 2.0 * M * N * K / 1e9;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  
  // Baseline: CUTLASS 1SM (tile 128×N)
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    using Runner = CutlassRunner<64, 
        cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
        cutlass::epilogue::TmaWarpSpecialized1Sm,
        cute::_2>;
    
    Runner runner;
    double ms = runner.run(M, N, K, repeats, warmup);
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
    
    // Allocate memory
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    
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
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }
  
  // D2: CUTLASS 2SM (tile 256×N, real mma.2sm)
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d2") == 0) {
    using Runner = CutlassRunner<64,
        cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
        cutlass::epilogue::TmaWarpSpecialized2Sm,
        cute::_2>;
    
    Runner runner;
    double ms = runner.run(M, N, K, repeats, warmup);
    if (ms > 0) {
      std::fprintf(stdout, "RESULT mode=d2 elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
    }
  }
  
#else
  std::fprintf(stderr, "CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined\n");
#endif
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  return 0;
}
