#include "gemm_reference_common.h"

#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cluster_launch.hpp>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace cute;

#ifndef CUBLAS_COMPUTE_32F_FAST_16BF
#define CUBLAS_COMPUTE_32F_FAST_16BF CUBLAS_COMPUTE_32F
#endif

using Element = cutlass::bfloat16_t;
using Accumulator = float;

constexpr int kInstrM = 128;
constexpr int kInstrN = 64;
constexpr int kInstrK = 16;
constexpr int kThreads = 128;

struct Step7Options {
  int m = 128;
  int n = 64;
  int k = 32;
  int warmup = 5;
  int iters = 20;
  int seed = 2026;
};

std::optional<std::string_view> maybe_value(const std::string& arg,
                                            const std::string& prefix) {
  if (arg.rfind(prefix, 0) != 0) {
    return std::nullopt;
  }
  return std::string_view(arg).substr(prefix.size());
}

template <typename T>
T parse_number(std::string_view text, const char* name) {
  std::string buffer(text);
  std::istringstream iss(buffer);
  T value{};
  iss >> value;
  if (!iss || !iss.eof()) {
    std::ostringstream oss;
    oss << "Invalid value for " << name << ": " << buffer;
    throw std::runtime_error(oss.str());
  }
  return value;
}

void print_usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --m=<int>        GEMM M, multiple of 128\n"
      << "  --n=<int>        GEMM N, multiple of 64\n"
      << "  --k=<int>        GEMM K, multiple of 16\n"
      << "  --warmup=<int>   Warmup iterations\n"
      << "  --iters=<int>    Timed iterations\n"
      << "  --seed=<int>     Deterministic exact init seed\n";
}

Step7Options parse_options(int argc, char** argv) {
  Step7Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (auto value = maybe_value(arg, "--m=")) {
      options.m = parse_number<int>(*value, "m");
    } else if (auto value = maybe_value(arg, "--n=")) {
      options.n = parse_number<int>(*value, "n");
    } else if (auto value = maybe_value(arg, "--k=")) {
      options.k = parse_number<int>(*value, "k");
    } else if (auto value = maybe_value(arg, "--warmup=")) {
      options.warmup = parse_number<int>(*value, "warmup");
    } else if (auto value = maybe_value(arg, "--iters=")) {
      options.iters = parse_number<int>(*value, "iters");
    } else if (auto value = maybe_value(arg, "--seed=")) {
      options.seed = parse_number<int>(*value, "seed");
    } else {
      std::ostringstream oss;
      oss << "Unknown option: " << arg;
      throw std::runtime_error(oss.str());
    }
  }

  if (options.m <= 0 || options.n <= 0 || options.k <= 0) {
    throw std::runtime_error("m, n and k must be positive");
  }
  if (options.warmup < 0 || options.iters <= 0) {
    throw std::runtime_error("warmup must be >= 0 and iters must be > 0");
  }
  if (options.m % kInstrM != 0 || options.n % kInstrN != 0 ||
      options.k % kInstrK != 0) {
    throw std::runtime_error(
        "step7 tcgen05 umma currently requires m multiple of 128, n multiple of 64, k multiple of 16");
  }
  return options;
}

void initialize_exact_tensor(std::vector<Element>& tensor, int seed) {
  for (std::size_t i = 0; i < tensor.size(); ++i) {
    int value = ((static_cast<int>(i) * 17 + seed * 13) % 5) - 2;
    tensor[i] = Element(static_cast<float>(value));
  }
}

void run_cublas_reference(cublasHandle_t handle,
                          const Step7Options& options,
                          const Element* d_a,
                          const Element* d_b,
                          float* d_d,
                          cublasComputeType_t compute_type) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CHECK_CUBLAS(cublasGemmEx(handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            options.n,
                            options.m,
                            options.k,
                            &alpha,
                            d_b,
                            CUDA_R_16BF,
                            options.k,
                            d_a,
                            CUDA_R_16BF,
                            options.k,
                            &beta,
                            d_d,
                            CUDA_R_32F,
                            options.n,
                            compute_type,
                            CUBLAS_GEMM_DEFAULT));
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA() {
    return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_sB() {
    return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{});
  }
};

template <class SharedStorageType,
          class ATensor,
          class BTensor,
          class CTensor,
          class DTensor,
          class MmaTiler_MNK,
          class TiledMMA,
          class ClusterShape_MNK,
          class Alpha,
          class Beta>
__global__ static void step7_tcgen05_umma_kernel(
    ATensor mA,
    BTensor mB,
    CTensor mC,
    DTensor mD,
    MmaTiler_MNK mma_tiler,
    TiledMMA tiled_mma,
    ClusterShape_MNK cluster_shape,
    Alpha alpha,
    Beta beta) {
  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape),
      make_tile(typename TiledMMA::AtomThrID{}));

  auto mma_coord_vmnk = make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                                   blockIdx.x / size<0>(cluster_layout_vmnk),
                                   blockIdx.y,
                                   _);
  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);

  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  SharedStorageType& shared_storage =
      *reinterpret_cast<SharedStorageType*>(shared_memory);

  Tensor tCsA = shared_storage.tensor_sA();
  Tensor tCsB = shared_storage.tensor_sB();

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
  }
  __syncthreads();
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(shared_storage.mma_barrier, 1);
  }
  int mma_barrier_phase_bit = 0;
  __syncthreads();

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    cooperative_copy<kThreads>(threadIdx.x, tCgA(_, _, _, k_tile), tCsA);
    cooperative_copy<kThreads>(threadIdx.x, tCgB(_, _, _, k_tile), tCsB);
    __syncthreads();

    if (elect_one_warp) {
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
    }

    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  TiledCopy tiled_t2r_copy =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDgC = thr_t2r_copy.partition_D(tCgC);
  Tensor tDrC = make_fragment_like(tDgC);
  copy(tDgC, tDrC);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  axpby(alpha, tDrAcc, beta, tDrC);
  copy(tDrC, tDgD);

  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
}

void launch_cutlass_tcgen05_umma_gemm(const Element* device_ptr_A,
                                      const Element* device_ptr_B,
                                      const float* device_ptr_C,
                                      float* device_ptr_D,
                                      const Step7Options& options) {
  auto layout_A =
      make_layout(make_shape(options.m, options.k),
                  make_stride(options.k, Int<1>{}));
  auto layout_B =
      make_layout(make_shape(options.n, options.k),
                  make_stride(options.k, Int<1>{}));
  auto layout_C =
      make_layout(make_shape(options.m, options.n),
                  make_stride(options.n, Int<1>{}));
  auto layout_D =
      make_layout(make_shape(options.m, options.n),
                  make_stride(options.n, Int<1>{}));

  Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);
  Tensor mB = make_tensor(make_gmem_ptr(device_ptr_B), layout_B);
  Tensor mC = make_tensor(make_gmem_ptr(device_ptr_C), layout_C);
  Tensor mD = make_tensor(make_gmem_ptr(device_ptr_D), layout_D);

  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<Element,
                           Element,
                           Accumulator,
                           kInstrM,
                           kInstrN,
                           UMMA::Major::K,
                           UMMA::Major::K>{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma);
  auto mma_tiler = make_shape(bM, bN, bK);

  auto mma_shape_A =
      partition_shape_A(tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B =
      partition_shape_B(tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{}, mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<Element>{}, mma_shape_B);

  using SMEMStorage =
      SharedStorage<Element, Element, decltype(sA_layout), decltype(sB_layout)>;

  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape),
      make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  dim3 dimBlock(kThreads);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                  size<2>(cluster_shape));
  dim3 dimGrid(
      size(ceil_div(options.m, size(bM) * size<1>(cluster_layout_vmnk))) *
          dimCluster.x,
      size(ceil_div(options.n, size(bN) * size<2>(cluster_layout_vmnk))) *
          dimCluster.y);
  int smem_bytes = sizeof(SMEMStorage);

  auto* kernel_ptr = &step7_tcgen05_umma_kernel<SMEMStorage,
                                                decltype(mA),
                                                decltype(mB),
                                                decltype(mC),
                                                decltype(mD),
                                                decltype(mma_tiler),
                                                decltype(tiled_mma),
                                                decltype(cluster_shape),
                                                float,
                                                float>;

  CHECK_CUDA(cudaFuncSetAttribute(kernel_ptr,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_bytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smem_bytes};
  cutlass::Status status =
      cutlass::launch_kernel_on_cluster(params,
                                        reinterpret_cast<void const*>(kernel_ptr),
                                        mA,
                                        mB,
                                        mC,
                                        mD,
                                        mma_tiler,
                                        tiled_mma,
                                        cluster_shape,
                                        1.0f,
                                        0.0f);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass::launch_kernel_on_cluster failed");
  }
  CHECK_CUDA(cudaGetLastError());
}

#else

void launch_cutlass_tcgen05_umma_gemm(const Element*,
                                      const Element*,
                                      const float*,
                                      float*,
                                      const Step7Options&) {
  throw std::runtime_error(
      "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not enabled for this build");
}

#endif

template <typename LaunchFn>
TimingStats benchmark_kernel(const Step7Options& options,
                             LaunchFn&& launch_fn,
                             float* d_out,
                             std::vector<float>& host_out) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(options.m) * options.n * sizeof(float);

  for (int i = 0; i < options.warmup; ++i) {
    launch_fn();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> times_ms;
  times_ms.reserve(options.iters);
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < options.iters; ++i) {
    CHECK_CUDA(cudaEventRecord(start));
    launch_fn();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    times_ms.push_back(elapsed);
  }

  CHECK_CUDA(cudaMemcpy(host_out.data(), d_out, output_bytes,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  Options stats_options;
  stats_options.m = options.m;
  stats_options.n = options.n;
  stats_options.k = options.k;
  return finalize_timing_stats(stats_options, times_ms, host_out);
}

CompareStats compare_exact(const std::vector<float>& reference,
                           const std::vector<float>& actual) {
  CompareStats stats;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double diff = std::abs(static_cast<double>(reference[i]) -
                                 static_cast<double>(actual[i]));
    stats.max_abs = std::max(stats.max_abs, diff);
    if (diff != 0.0) {
      ++stats.fail_count;
    }
  }
  stats.pass = (stats.fail_count == 0);
  return stats;
}

void print_step7_result_line(const Step7Options& options,
                             const char* backend,
                             const TimingStats& stats) {
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT benchmark=bench_step7_tcgen05_umma"
            << " backend=" << backend
            << " m=" << options.m
            << " n=" << options.n
            << " k=" << options.k
            << " warmup=" << options.warmup
            << " iters=" << options.iters
            << " avg_ms=" << stats.avg_ms
            << " median_ms=" << stats.median_ms
            << " min_ms=" << stats.min_ms
            << " gflops=" << stats.gflops
            << " tflops=" << (stats.gflops / 1000.0)
            << " checksum=" << stats.checksum
            << '\n';
}

void print_step7_check_line(const Step7Options& options,
                            const char* backend,
                            const CompareStats& compare,
                            bool exact_required) {
  std::cout << std::scientific << std::setprecision(6)
            << "CHECK benchmark=bench_step7_tcgen05_umma"
            << " backend=" << backend
            << " m=" << options.m
            << " n=" << options.n
            << " k=" << options.k
            << " pass=" << static_cast<int>(compare.pass)
            << " max_abs=" << compare.max_abs
            << " fail_count=" << compare.fail_count;
  if (!exact_required) {
    std::cout << " note=informational_only";
  }
  std::cout << '\n';
}

void print_step7_summary(const Step7Options& options,
                         const TimingStats& tcgen05_stats,
                         const TimingStats& cublas_fast_stats,
                         const TimingStats& cublas_pedantic_stats,
                         const CompareStats& tcgen05_compare,
                         const CompareStats& fast_compare) {
  auto print_perf_row = [&](const char* backend, const TimingStats& stats) {
    const double speedup_vs_fast =
        cublas_fast_stats.median_ms / stats.median_ms;
    const double speedup_vs_pedantic =
        cublas_pedantic_stats.median_ms / stats.median_ms;
    std::cout << std::fixed << std::setprecision(6)
              << "  " << std::left << std::setw(24) << backend
              << std::right << std::setw(12) << stats.median_ms
              << std::setw(12) << (stats.gflops / 1000.0)
              << std::setw(12) << speedup_vs_fast
              << std::setw(16) << speedup_vs_pedantic
              << std::setw(16) << stats.checksum << '\n';
  };

  std::cout << "SUMMARY benchmark=bench_step7_tcgen05_umma"
            << " m=" << options.m
            << " n=" << options.n
            << " k=" << options.k
            << " baseline=cublas_fast_16bf"
            << '\n';
  std::cout << "  correctness: cutlass_tcgen05_umma vs cublas_pedantic exact_pass="
            << static_cast<int>(tcgen05_compare.pass)
            << " max_abs=" << std::scientific << std::setprecision(6)
            << tcgen05_compare.max_abs
            << " fail_count=" << tcgen05_compare.fail_count << '\n';
  std::cout << "  fast_math_delta: cublas_fast_16bf vs cublas_pedantic pass="
            << static_cast<int>(fast_compare.pass)
            << " max_abs=" << std::scientific << std::setprecision(6)
            << fast_compare.max_abs
            << " fail_count=" << fast_compare.fail_count << '\n';
  std::cout << "  "
            << std::left << std::setw(24) << "backend"
            << std::right << std::setw(12) << "median_ms"
            << std::setw(12) << "tflops"
            << std::setw(12) << "vs_fast"
            << std::setw(16) << "vs_pedantic"
            << std::setw(16) << "checksum" << '\n';
  print_perf_row("cutlass_tcgen05_umma", tcgen05_stats);
  print_perf_row("cublas_fast_16bf", cublas_fast_stats);
  print_perf_row("cublas_pedantic", cublas_pedantic_stats);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Step7Options options = parse_options(argc, argv);
    print_device_info();

    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    if (props.major < 10) {
      throw std::runtime_error(
          "step7 tcgen05 umma path requires Blackwell SM100 or newer");
    }

    const std::size_t a_elems =
        static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_elems =
        static_cast<std::size_t>(options.n) * options.k;
    const std::size_t c_elems =
        static_cast<std::size_t>(options.m) * options.n;

    std::vector<Element> host_A(a_elems);
    std::vector<Element> host_B(b_elems);
    initialize_exact_tensor(host_A, options.seed);
    initialize_exact_tensor(host_B, options.seed + 1);

    Element* device_A = nullptr;
    Element* device_B = nullptr;
    float* device_C = nullptr;
    float* device_tcgen05_D = nullptr;
    float* device_cublas_pedantic_D = nullptr;
    float* device_cublas_fast_D = nullptr;
    CHECK_CUDA(cudaMalloc(&device_A, a_elems * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&device_B, b_elems * sizeof(Element)));
    CHECK_CUDA(cudaMalloc(&device_C, c_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_tcgen05_D, c_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_cublas_pedantic_D, c_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_cublas_fast_D, c_elems * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_A, host_A.data(), a_elems * sizeof(Element),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_B, host_B.data(), b_elems * sizeof(Element),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(device_C, 0, c_elems * sizeof(float)));

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    auto tcgen05_launch = [&] {
      launch_cutlass_tcgen05_umma_gemm(device_A, device_B, device_C,
                                       device_tcgen05_D, options);
    };
    auto cublas_pedantic_launch = [&] {
      run_cublas_reference(handle, options, device_A, device_B,
                           device_cublas_pedantic_D,
                           CUBLAS_COMPUTE_32F_PEDANTIC);
    };
    auto cublas_fast_launch = [&] {
      run_cublas_reference(handle, options, device_A, device_B,
                           device_cublas_fast_D,
                           CUBLAS_COMPUTE_32F_FAST_16BF);
    };

    cublas_pedantic_launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    cublas_fast_launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    tcgen05_launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> host_tcgen05_D(c_elems, 0.0f);
    std::vector<float> host_cublas_pedantic_D(c_elems, 0.0f);
    std::vector<float> host_cublas_fast_D(c_elems, 0.0f);
    CHECK_CUDA(cudaMemcpy(host_tcgen05_D.data(), device_tcgen05_D,
                          c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_cublas_pedantic_D.data(),
                          device_cublas_pedantic_D,
                          c_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_cublas_fast_D.data(), device_cublas_fast_D,
                          c_elems * sizeof(float), cudaMemcpyDeviceToHost));

    const CompareStats tcgen05_compare =
        compare_exact(host_cublas_pedantic_D, host_tcgen05_D);
    const CompareStats fast_compare =
        compare_exact(host_cublas_pedantic_D, host_cublas_fast_D);
    print_step7_check_line(options, "cutlass_tcgen05_umma_vs_cublas_pedantic",
                           tcgen05_compare, true);
    print_step7_check_line(options, "cublas_fast_16bf_vs_cublas_pedantic",
                           fast_compare, false);

    if (!tcgen05_compare.pass) {
      throw std::runtime_error(
          "cutlass tcgen05 umma output does not exactly match cuBLAS");
    }

    TimingStats tcgen05_stats =
        benchmark_kernel(options, tcgen05_launch, device_tcgen05_D,
                         host_tcgen05_D);
    TimingStats cublas_fast_stats =
        benchmark_kernel(options, cublas_fast_launch, device_cublas_fast_D,
                         host_cublas_fast_D);
    TimingStats cublas_pedantic_stats =
        benchmark_kernel(options, cublas_pedantic_launch,
                         device_cublas_pedantic_D, host_cublas_pedantic_D);

    print_step7_result_line(options, "cutlass_tcgen05_umma", tcgen05_stats);
    print_step7_result_line(options, "cublas_fast_16bf", cublas_fast_stats);
    print_step7_result_line(options, "cublas_pedantic", cublas_pedantic_stats);
    print_step7_summary(options, tcgen05_stats, cublas_fast_stats,
                        cublas_pedantic_stats, tcgen05_compare, fast_compare);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(device_A));
    CHECK_CUDA(cudaFree(device_B));
    CHECK_CUDA(cudaFree(device_C));
    CHECK_CUDA(cudaFree(device_tcgen05_D));
    CHECK_CUDA(cudaFree(device_cublas_pedantic_D));
    CHECK_CUDA(cudaFree(device_cublas_fast_D));
  } catch (const std::exception& error) {
    std::cerr << "ERROR " << error.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
