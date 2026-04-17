#include "common.h"

#include <iostream>

// Workaround: provide fixed grid/block sizes to bypass cudaOccupancyMaxPotentialBlockSize
// which fails on sm_100a kernels. We use BlockForEach directly instead of
// BlockFillRandomUniform so we can pass explicit launch parameters.
#define CUTLASS_BLOCKFILL_GRID  256
#define CUTLASS_BLOCKFILL_BLOCK 128

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

using namespace cute;
using namespace blackwell_dsmem_2sm;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <int TileN>
struct TileShapeForN;

template <>
struct TileShapeForN<64> { using Type = Shape<_128, _64, _64>; };

template <>
struct TileShapeForN<128> { using Type = Shape<_128, _128, _64>; };

template <>
struct TileShapeForN<256> { using Type = Shape<_128, _256, _64>; };

template <int TileN>
struct TileShape2SmForN;

template <>
struct TileShape2SmForN<64> { using Type = Shape<_256, _64, _64>; };

template <>
struct TileShape2SmForN<128> { using Type = Shape<_256, _128, _64>; };

template <>
struct TileShape2SmForN<256> { using Type = Shape<_256, _256, _64>; };

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

  void initialize(int m, int n, int k) {
    strideA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    strideB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    strideC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    strideD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    A.reset(static_cast<std::size_t>(m) * k);
    B.reset(static_cast<std::size_t>(k) * n);
    C.reset(static_cast<std::size_t>(m) * n);
    D.reset(static_cast<std::size_t>(m) * n);

    {
      using RF = cutlass::reference::device::detail::RandomUniformFunc<ElementA>;
      cutlass::reference::device::BlockForEach<ElementA, RF>(
        A.get(), A.size(), typename RF::Params(2023, ElementA(3), ElementA(-3), 0, 0),
        CUTLASS_BLOCKFILL_GRID, CUTLASS_BLOCKFILL_BLOCK);
    }
    {
      using RF = cutlass::reference::device::detail::RandomUniformFunc<ElementB>;
      cutlass::reference::device::BlockForEach<ElementB, RF>(
        B.get(), B.size(), typename RF::Params(2024, ElementB(3), ElementB(-3), 0, 0),
        CUTLASS_BLOCKFILL_GRID, CUTLASS_BLOCKFILL_BLOCK);
    }
    {
      using RF = cutlass::reference::device::detail::RandomUniformFunc<ElementC>;
      cutlass::reference::device::BlockForEach<ElementC, RF>(
        C.get(), C.size(), typename RF::Params(2025, ElementC(3), ElementC(-3), 0, 0),
        CUTLASS_BLOCKFILL_GRID, CUTLASS_BLOCKFILL_BLOCK);
    }
  }

  bool run_once(const GemmOptions& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShape problem = ProblemShape{options.m, options.n, options.k, 1};
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        {A.get(), strideA, B.get(), strideB},
        {{1.0f, 0.0f}, C.get(), strideC, D.get(), strideD},
        hw_info};

    Gemm gemm_op;
    std::size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return false;
    status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) return false;
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
  }
};

template <typename Runner>
double measure_runner(Runner& runner, const GemmOptions& options, const cutlass::KernelHardwareInfo& hw_info) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  double total_ms = 0.0;
  runner.initialize(options.m, options.n, options.k);
  for (int i = 0; i < options.warmup_repeats; ++i) {
    if (!runner.run_once(options, hw_info)) return -1.0;
  }
  for (int i = 0; i < options.repeats; ++i) {
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    if (!runner.run_once(options, hw_info)) return -1.0;
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    total_ms += elapsed_ms(start, stop);
  }

  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
  return total_ms / static_cast<double>(options.repeats);
}

template <int TileN, class StageCountTag>
double dispatch_cutlass_1sm(const GemmOptions& options, const cutlass::KernelHardwareInfo& hw_info) {
  using Runner = CutlassRunner<TileN, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100, cutlass::epilogue::TmaWarpSpecialized1Sm, StageCountTag>;
  Runner runner;
  return measure_runner(runner, options, hw_info);
}

template <int TileN, class StageCountTag>
double dispatch_cutlass_2sm(const GemmOptions& options, const cutlass::KernelHardwareInfo& hw_info) {
  using Runner = CutlassRunner<TileN, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100, cutlass::epilogue::TmaWarpSpecialized2Sm, StageCountTag>;
  Runner runner;
  return measure_runner(runner, options, hw_info);
}

template <class StageCountTag>
double dispatch_cutlass_by_mode(const GemmOptions& options, const cutlass::KernelHardwareInfo& hw_info) {
  if (std::strcmp(options.mode, "1sm") == 0) {
    if (options.tile_n == 64) return dispatch_cutlass_1sm<64, StageCountTag>(options, hw_info);
    if (options.tile_n == 128) return dispatch_cutlass_1sm<128, StageCountTag>(options, hw_info);
    return -1.0;
  }

  if (options.tile_n == 64) return dispatch_cutlass_2sm<64, StageCountTag>(options, hw_info);
  if (options.tile_n == 128) return dispatch_cutlass_2sm<128, StageCountTag>(options, hw_info);
  if (options.tile_n == 256) return dispatch_cutlass_2sm<256, StageCountTag>(options, hw_info);
  return -1.0;
}

#endif

int main(int argc, char** argv) {
  GemmOptions options;
  parse_gemm_options(argc, argv, &options);

  if (!is_valid_tile_n(options.tile_n) || !is_valid_stages(options.stages)) {
    std::fprintf(stderr, "Unsupported tile_n=%d or stages=%d\n", options.tile_n, options.stages);
    return 1;
  }
  if (std::strcmp(options.mode, "1sm") != 0 && std::strcmp(options.mode, "2sm") != 0) {
    std::fprintf(stderr, "mode must be 1sm or 2sm\n");
    return 1;
  }
  if (options.stages == 1) {
    std::fprintf(stderr, "CUTLASS SM100 benchmark does not support stages=1 in this configuration. Use --stages=2 or --stages=4.\n");
    return 1;
  }
  if (std::strcmp(options.mode, "1sm") == 0 && options.tile_n == 256) {
    std::fprintf(stderr, "CUTLASS 1SM benchmark with tile_n=256 exceeds SMEM capacity in this configuration. Use tile_n=64 or 128.\n");
    return 1;
  }

  cudaDeviceProp props{};
  check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  std::printf(
      "CONFIG benchmark=bench_cutlass_2sm_gemm mode=%s m=%d n=%d k=%d tile_n=%d stages=%d repeats=%d warmup_repeats=%d gpu=\"%s\"\n",
      options.mode,
      options.m,
      options.n,
      options.k,
      options.tile_n,
      options.stages,
      options.repeats,
      options.warmup_repeats,
      props.name);

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  std::fprintf(stderr, "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not enabled for this build.\n");
  return 1;
#else
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  double avg_ms = -1.0;
  if (options.stages == 2) avg_ms = dispatch_cutlass_by_mode<cute::_2>(options, hw_info);
  if (options.stages == 4) avg_ms = dispatch_cutlass_by_mode<cute::_4>(options, hw_info);

  if (avg_ms <= 0.0) {
    std::fprintf(stderr, "CUTLASS benchmark failed to run.\n");
    return 1;
  }

  const double flops = 2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) * static_cast<double>(options.k);
  const double gflops = flops / (avg_ms * 1.0e6);
  const double bytes_b_share = static_cast<double>(options.tile_n) * 64.0 * sizeof(cutlass::half_t);

  std::printf(
      "RESULT benchmark=bench_cutlass_2sm_gemm mode=%s m=%d n=%d k=%d tile_n=%d stages=%d avg_ms=%.6f gflops=%.4f bytes_b_share=%.0f\n",
      options.mode,
      options.m,
      options.n,
      options.k,
      options.tile_n,
      options.stages,
      avg_ms,
      gflops,
      bytes_b_share);
  return 0;
#endif
}
