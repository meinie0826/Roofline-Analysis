#include "common.h"

#include <type_traits>

// Work around cudaOccupancyMaxPotentialBlockSize issues on sm_100a.
#define CUTLASS_BLOCKFILL_GRID  256
#define CUTLASS_BLOCKFILL_BLOCK 128

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

template <int TileN, bool Use2Sm, class StageCountTag>
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

  using ClusterShapeMNK = cute::conditional_t<Use2Sm, Shape<_2, _1, _1>, Shape<_1, _1, _1>>;
  using TileShape = std::conditional_t<
      Use2Sm,
      typename TileShape2SmForN<TileN>::Type,
      typename TileShapeForN<TileN>::Type>;

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
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

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
      cute::conditional_t<
          cute::is_same_v<StageCountTag, cutlass::gemm::collective::StageCountAuto>,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          StageCountTag>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

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
          A.get(),
          A.size(),
          typename RF::Params(2023, ElementA(3), ElementA(-3), 0, 0),
          CUTLASS_BLOCKFILL_GRID,
          CUTLASS_BLOCKFILL_BLOCK);
    }
    {
      using RF = cutlass::reference::device::detail::RandomUniformFunc<ElementB>;
      cutlass::reference::device::BlockForEach<ElementB, RF>(
          B.get(),
          B.size(),
          typename RF::Params(2024, ElementB(3), ElementB(-3), 0, 0),
          CUTLASS_BLOCKFILL_GRID,
          CUTLASS_BLOCKFILL_BLOCK);
    }
    {
      using RF = cutlass::reference::device::detail::RandomUniformFunc<ElementC>;
      cutlass::reference::device::BlockForEach<ElementC, RF>(
          C.get(),
          C.size(),
          typename RF::Params(2025, ElementC(3), ElementC(-3), 0, 0),
          CUTLASS_BLOCKFILL_GRID,
          CUTLASS_BLOCKFILL_BLOCK);
    }
  }

  bool setup(const GemmOptions& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShape problem = ProblemShape{options.m, options.n, options.k, 1};
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        {A.get(), strideA, B.get(), strideB},
        {{1.0f, 0.0f}, C.get(), strideC, D.get(), strideD},
        hw_info};

    gemm_op_ = Gemm{};
    std::size_t workspace_size = Gemm::get_workspace_size(arguments);
    workspace_ = cutlass::device_memory::allocation<uint8_t>(workspace_size);

    cutlass::Status status = gemm_op_.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "can_implement failed: %s\n", cutlass::cutlassGetStatusString(status));
      return false;
    }

    status = gemm_op_.initialize(arguments, workspace_.get());
    if (status != cutlass::Status::kSuccess) {
      cudaError_t cuda_err = cudaGetLastError();
      std::fprintf(stderr,
                   "initialize failed: %s (cuda: %s)\n",
                   cutlass::cutlassGetStatusString(status),
                   cudaGetErrorString(cuda_err));
      return false;
    }
    return true;
  }

  bool run_kernel() {
    cutlass::Status status = gemm_op_.run();
    if (status != cutlass::Status::kSuccess) {
      std::fprintf(stderr, "run failed: %s\n", cutlass::cutlassGetStatusString(status));
      return false;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
      return false;
    }
    return true;
  }

 private:
  Gemm gemm_op_;
  cutlass::device_memory::allocation<uint8_t> workspace_;
};

struct KernelMetrics {
  double avg_ms = -1.0;
  double tflops = -1.0;
  double ns_per_flop = -1.0;
  double est_b_bytes = 0.0;
  double est_b_bandwidth_gbps = 0.0;
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 64;
};

template <typename Runner>
KernelMetrics measure_runner(
    Runner& runner,
    const GemmOptions& options,
    const cutlass::KernelHardwareInfo& hw_info,
    int tile_m,
    int tile_n) {
  KernelMetrics metrics;
  metrics.tile_m = tile_m;
  metrics.tile_n = tile_n;

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  runner.initialize(options.m, options.n, options.k);
  if (!runner.setup(options, hw_info)) {
    return metrics;
  }

  for (int i = 0; i < options.warmup_repeats; ++i) {
    if (!runner.run_kernel()) {
      return metrics;
    }
  }

  double total_ms = 0.0;
  for (int i = 0; i < options.repeats; ++i) {
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    if (!runner.run_kernel()) {
      return metrics;
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    total_ms += elapsed_ms(start, stop);
  }

  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

  metrics.avg_ms = total_ms / static_cast<double>(options.repeats);

  const double flops =
      2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) * static_cast<double>(options.k);
  metrics.tflops = flops / (metrics.avg_ms * 1.0e9);
  metrics.ns_per_flop = (metrics.avg_ms * 1.0e6) / flops;

  const int grid_m = div_up(options.m, tile_m);
  const int grid_n = div_up(options.n, tile_n);
  const int grid_k = div_up(options.k, metrics.tile_k);
  metrics.est_b_bytes = static_cast<double>(grid_m) * grid_n * grid_k *
                        static_cast<double>(tile_n) * metrics.tile_k * sizeof(cutlass::half_t);
  metrics.est_b_bandwidth_gbps = metrics.avg_ms > 0.0
      ? metrics.est_b_bytes / (metrics.avg_ms * 1.0e6)
      : 0.0;

  return metrics;
}

template <int TileN, class StageCountTag>
KernelMetrics dispatch_cutlass_1sm(
    const GemmOptions& options,
    const cutlass::KernelHardwareInfo& hw_info) {
  using Runner = CutlassRunner<TileN, false, StageCountTag>;
  Runner runner;
  return measure_runner(runner, options, hw_info, 128, TileN);
}

template <int TileN, class StageCountTag>
KernelMetrics dispatch_cutlass_2sm(
    const GemmOptions& options,
    const cutlass::KernelHardwareInfo& hw_info) {
  using Runner = CutlassRunner<TileN, true, StageCountTag>;
  Runner runner;
  return measure_runner(runner, options, hw_info, 256, TileN);
}

template <class StageCountTag>
KernelMetrics dispatch_cutlass_mode(
    const char* mode,
    const GemmOptions& options,
    const cutlass::KernelHardwareInfo& hw_info) {
  if (std::strcmp(mode, "1sm") == 0) {
    if (options.tile_n == 64) return dispatch_cutlass_1sm<64, StageCountTag>(options, hw_info);
    if (options.tile_n == 128) return dispatch_cutlass_1sm<128, StageCountTag>(options, hw_info);
  } else if (std::strcmp(mode, "2sm") == 0) {
    if (options.tile_n == 64) return dispatch_cutlass_2sm<64, StageCountTag>(options, hw_info);
    if (options.tile_n == 128) return dispatch_cutlass_2sm<128, StageCountTag>(options, hw_info);
    if (options.tile_n == 256) return dispatch_cutlass_2sm<256, StageCountTag>(options, hw_info);
  }

  return KernelMetrics{};
}

void print_result_line(const char* mode, const GemmOptions& options, const KernelMetrics& metrics) {
  std::printf(
      "RESULT benchmark=bench_cutlass_mma2sm_cost mode=%s m=%d n=%d k=%d tile_m=%d tile_n=%d tile_k=%d stages=%d avg_ms=%.6f tflops=%.4f ns_per_flop=%.6e est_b_bytes=%.0f est_b_bandwidth_gbps=%.4f\n",
      mode,
      options.m,
      options.n,
      options.k,
      metrics.tile_m,
      metrics.tile_n,
      metrics.tile_k,
      options.stages,
      metrics.avg_ms,
      metrics.tflops,
      metrics.ns_per_flop,
      metrics.est_b_bytes,
      metrics.est_b_bandwidth_gbps);
}

void print_summary_line(const GemmOptions& options, const KernelMetrics& one_sm, const KernelMetrics& two_sm) {
  const double speedup = two_sm.avg_ms > 0.0 ? one_sm.avg_ms / two_sm.avg_ms : 0.0;
  const double tflops_ratio = one_sm.tflops > 0.0 ? two_sm.tflops / one_sm.tflops : 0.0;
  const double ns_per_flop_ratio = one_sm.ns_per_flop > 0.0 ? two_sm.ns_per_flop / one_sm.ns_per_flop : 0.0;
  const double b_bytes_ratio = one_sm.est_b_bytes > 0.0 ? two_sm.est_b_bytes / one_sm.est_b_bytes : 0.0;
  const double b_bw_ratio = one_sm.est_b_bandwidth_gbps > 0.0
      ? two_sm.est_b_bandwidth_gbps / one_sm.est_b_bandwidth_gbps
      : 0.0;

  std::printf(
      "SUMMARY benchmark=bench_cutlass_mma2sm_cost compare=2sm_vs_1sm m=%d n=%d k=%d tile_n=%d stages=%d speedup=%.6f tflops_ratio=%.6f ns_per_flop_ratio=%.6f est_b_bytes_ratio=%.6f est_b_bw_ratio=%.6f\n",
      options.m,
      options.n,
      options.k,
      options.tile_n,
      options.stages,
      speedup,
      tflops_ratio,
      ns_per_flop_ratio,
      b_bytes_ratio,
      b_bw_ratio);
}

#endif

int main(int argc, char** argv) {
  GemmOptions options;
  parse_gemm_options(argc, argv, &options);

  const char* mode = options.mode;
  if (std::strcmp(mode, "compare") != 0 &&
      std::strcmp(mode, "1sm") != 0 &&
      std::strcmp(mode, "2sm") != 0) {
    std::fprintf(stderr, "mode must be compare, 1sm, or 2sm\n");
    return 1;
  }

  if (!is_valid_tile_n(options.tile_n) || !is_valid_stages(options.stages)) {
    std::fprintf(stderr, "Unsupported tile_n=%d or stages=%d\n", options.tile_n, options.stages);
    return 1;
  }

  if (options.stages == 1) {
    std::fprintf(stderr,
                 "CUTLASS SM100 benchmark does not support stages=1 in this configuration. "
                 "Use --stages=2 or --stages=4.\n");
    return 1;
  }

  if (options.tile_n == 256) {
    std::fprintf(stderr,
                 "tile_n=256 is not enabled because the 1SM reference kernel over-allocates "
                 "SMEM in this configuration. Use tile_n=64 or 128.\n");
    return 1;
  }

  cudaDeviceProp props{};
  check_cuda(cudaGetDeviceProperties(&props, 0), "cudaGetDeviceProperties");
  std::printf(
      "CONFIG benchmark=bench_cutlass_mma2sm_cost mode=%s m=%d n=%d k=%d tile_n=%d stages=%d repeats=%d warmup_repeats=%d gpu=\"%s\"\n",
      mode,
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

  auto dispatch = [&](const char* kernel_mode) {
    if (options.stages == 2) return dispatch_cutlass_mode<cute::_2>(kernel_mode, options, hw_info);
    return dispatch_cutlass_mode<cute::_4>(kernel_mode, options, hw_info);
  };

  if (std::strcmp(mode, "1sm") == 0 || std::strcmp(mode, "2sm") == 0) {
    KernelMetrics metrics = dispatch(mode);
    if (metrics.avg_ms <= 0.0) {
      std::fprintf(stderr, "CUTLASS benchmark failed to run.\n");
      return 1;
    }
    print_result_line(mode, options, metrics);
    return 0;
  }

  KernelMetrics one_sm = dispatch("1sm");
  KernelMetrics two_sm = dispatch("2sm");
  if (one_sm.avg_ms <= 0.0 || two_sm.avg_ms <= 0.0) {
    std::fprintf(stderr, "CUTLASS benchmark failed to run.\n");
    return 1;
  }

  print_result_line("1sm", options, one_sm);
  print_result_line("2sm", options, two_sm);
  print_summary_line(options, one_sm, two_sm);
  return 0;
#endif
}
