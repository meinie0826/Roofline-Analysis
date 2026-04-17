/**
 * Simplified 2SM comparison benchmark - reuses existing CUTLASS runners
 */

#include "common.h"
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

using namespace cute;
using namespace blackwell_dsmem_2sm;

//=============================================================================
// Copy definitions from bench_cutlass_2sm_gemm.cu
//=============================================================================

template <int TileN>
struct TileShapeForN;

template <> struct TileShapeForN<64> { using Type = Shape<_128, _64, _64>; };
template <> struct TileShapeForN<128> { using Type = Shape<_128, _128, _64>; };
template <> struct TileShapeForN<256> { using Type = Shape<_128, _256, _64>; };

template <int TileN>
struct TileShape2SmForN;

template <> struct TileShape2SmForN<64> { using Type = Shape<_256, _64, _64>; };
template <> struct TileShape2SmForN<128> { using Type = Shape<_256, _128, _64>; };
template <> struct TileShape2SmForN<256> { using Type = Shape<_256, _256, _64>; };

template <int TileN, class MainloopSchedule, class EpilogueSchedule, class StageCountTag>
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
  using ClusterShapeMNK = Shape<_2, _1, _1>;
  using TileShape = std::conditional_t<kUse2Sm, typename TileShape2SmForN<TileN>::Type, typename TileShapeForN<TileN>::Type>;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      TileShape,
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
      TileShape,
      ClusterShapeMNK,
      StageCountTag,
      MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
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

  StrideA strideA;
  StrideB strideB;
  StrideC strideC;
  StrideD strideD;

  Gemm gemm_op_;
  cutlass::device_memory::allocation<uint8_t> workspace_;

  void initialize(int m, int n, int k) {
    strideA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    strideB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    strideC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    strideD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    A.reset(static_cast<std::size_t>(m) * k);
    B.reset(static_cast<std::size_t>(k) * n);
    C.reset(static_cast<std::size_t>(m) * n);
    D.reset(static_cast<std::size_t>(m) * n);
  }

  bool setup(int m, int n, int k) {
    ProblemShape problem = ProblemShape{m, n, k, 1};
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        {A.get(), strideA, B.get(), strideB},
        {{1.0f, 0.0f}, C.get(), strideC, D.get(), strideD},
        {}};
    gemm_op_ = Gemm{};
    std::size_t workspace_size = Gemm::get_workspace_size(arguments);
    workspace_ = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    cutlass::Status status = gemm_op_.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return false;
    status = gemm_op_.initialize(arguments, workspace_.get());
    return status == cutlass::Status::kSuccess;
  }

  bool run() {
    cutlass::Status status = gemm_op_.run();
    if (status != cutlass::Status::kSuccess) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
  }
};

//=============================================================================
// Benchmark wrappers
//=============================================================================

double measure_gemm(int M, int N, int K, int repeats, int warmup, const char* mode) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = 148;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total_ms = 0.0;

  // Dispatch based on mode
  if (std::strcmp(mode, "baseline") == 0) {
    // Baseline: 1SM kernel (each CTA would load B independently)
    using Runner = CutlassRunner<64, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
                                 cutlass::epilogue::TmaWarpSpecialized1Sm, cute::_2>;
    Runner runner;
    runner.initialize(M, N, K);

    for (int w = 0; w < warmup; ++w) {
      runner.setup(M, N, K);
      runner.run();
    }

    for (int r = 0; r < repeats; ++r) {
      runner.setup(M, N, K);
      cudaEventRecord(start);
      runner.run();
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
    }
  } else if (std::strcmp(mode, "d1") == 0) {
    // D1: Same as baseline for now - would need custom kernel for DSMEM sharing
    using Runner = CutlassRunner<64, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
                                 cutlass::epilogue::TmaWarpSpecialized1Sm, cute::_2>;
    Runner runner;
    runner.initialize(M, N, K);

    for (int w = 0; w < warmup; ++w) {
      runner.setup(M, N, K);
      runner.run();
    }

    for (int r = 0; r < repeats; ++r) {
      runner.setup(M, N, K);
      cudaEventRecord(start);
      runner.run();
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
    }
  } else if (std::strcmp(mode, "d2") == 0) {
    // D2: 2SM hardware shared B
    using Runner = CutlassRunner<64, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                                 cutlass::epilogue::TmaWarpSpecialized2Sm, cute::_2>;
    Runner runner;
    runner.initialize(M, N, K);

    for (int w = 0; w < warmup; ++w) {
      runner.setup(M, N, K);
      runner.run();
    }

    for (int r = 0; r < repeats; ++r) {
      runner.setup(M, N, K);
      cudaEventRecord(start);
      runner.run();
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);

      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
    }
  } else {
    std::fprintf(stderr, "Unknown mode: %s\n", mode);
    return -1.0;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return total_ms / repeats;
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "baseline";
  int M = 4096, N = 64, K = 4096;
  int repeats = 10, warmup = 3;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--m=") == 0) M = std::atoi(argv[i] + 4);
    else if (arg.find("--n=") == 0) N = std::atoi(argv[i] + 4);
    else if (arg.find("--k=") == 0) K = std::atoi(argv[i] + 4);
    else if (arg.find("--repeats=") == 0) repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--warmup=") == 0) warmup = std::atoi(argv[i] + 9);
    else if (arg == "--help" || arg == "-h") {
      std::cerr << "Usage: " << argv[0] << " --mode=baseline|d1|d2 --m=N --n=N --k=N\n";
      return 0;
    }
  }

  std::fprintf(stdout, "CONFIG mode=%s m=%d n=%d k=%d repeats=%d warmup=%d gpu=\"%s\"\n",
               mode, M, N, K, repeats, warmup, gpu_name().c_str());

  double avg_ms = measure_gemm(M, N, K, repeats, warmup, mode);
  double gflops = 2.0 * M * N * K / avg_ms / 1.0e6;

  std::fprintf(stdout, "RESULT elapsed_ms=%.6f gflops=%.2f\n", avg_ms, gflops);

  return 0;
}
