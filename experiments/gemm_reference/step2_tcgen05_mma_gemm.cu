#include "gemm_reference_common.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace cute;

namespace {

struct Step2Options {
  int m = 128;
  int n = 256;
  int k = 64;
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
      << "  --n=<int>        GEMM N, multiple of 256\n"
      << "  --k=<int>        GEMM K, multiple of 64\n"
      << "  --warmup=<int>   Warmup iterations\n"
      << "  --iters=<int>    Timed iterations\n"
      << "  --seed=<int>     Deterministic exact init seed\n";
}

Step2Options parse_options(int argc, char** argv) {
  Step2Options options;
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
  if (options.m % 128 != 0 || options.n % 256 != 0 || options.k % 64 != 0) {
    throw std::runtime_error(
        "step2 currently requires m multiple of 128, n multiple of 256, k multiple of 64");
  }
  return options;
}

template <class Tensor>
void initialize_exact_tensor(Tensor& tensor, int seed) {
  using DataType = typename Tensor::element_type;
  for (int i = 0; i < cute::size(tensor); ++i) {
    int value = ((i * 17 + seed * 13) % 5) - 2;
    tensor(i) = DataType(value);
  }
}

template <class Tensor>
void zero_tensor(Tensor& tensor) {
  using DataType = typename Tensor::element_type;
  for (int i = 0; i < cute::size(tensor); ++i) {
    tensor(i) = DataType(0);
  }
}

template <int NumThreads, class SrcTensor, class DstTensor>
CUTE_DEVICE void cooperative_copy_fallback(uint32_t tid,
                                           const SrcTensor& src,
                                           DstTensor& dst) {
  static_assert(NumThreads > 0, "NumThreads must be positive");
  for (int i = tid; i < size(dst); i += NumThreads) {
    dst(i) = src(i);
  }
}

void run_cublas_reference(cublasHandle_t handle,
                          const Step2Options& options,
                          const cutlass::half_t* d_a,
                          const cutlass::half_t* d_b,
                          float* d_d) {
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
                            CUDA_R_16F,
                            options.k,
                            d_a,
                            CUDA_R_16F,
                            options.k,
                            &beta,
                            d_d,
                            CUDA_R_32F,
                            options.n,
                            CUBLAS_COMPUTE_32F_PEDANTIC,
                            CUBLAS_GEMM_DEFAULT));
}

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorageStep2 {
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

template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler_MNK, class TiledMMA,
          class ClusterShape_MNK, class Alpha, class Beta>
__global__ static void gemm_device_step2(
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
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step< X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  auto& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

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
    cooperative_copy_fallback<128>(threadIdx.x, tCgA(_, _, _, k_tile), tCsA);
    cooperative_copy_fallback<128>(threadIdx.x, tCgB(_, _, _, k_tile), tCsB);
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

  TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
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

template <class TypeA, class LayoutA, class TypeB, class LayoutB, class TypeC,
          class LayoutC, class TypeD, class LayoutD, class Alpha, class Beta>
void launch_tcgen05_gemm(const TypeA* device_ptr_A,
                         LayoutA layout_A,
                         const TypeB* device_ptr_B,
                         LayoutB layout_B,
                         const TypeC* device_ptr_C,
                         LayoutC layout_C,
                         TypeD* device_ptr_D,
                         LayoutD layout_D,
                         Alpha alpha,
                         Beta beta) {
  if (shape<0>(layout_A) != shape<0>(layout_C) ||
      shape<0>(layout_A) != shape<0>(layout_D) ||
      shape<0>(layout_B) != shape<1>(layout_C) ||
      shape<0>(layout_B) != shape<1>(layout_D) ||
      shape<1>(layout_A) != shape<1>(layout_B)) {
    throw std::runtime_error("step2 layout mismatch");
  }

  Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);
  Tensor mB = make_tensor(make_gmem_ptr(device_ptr_B), layout_B);
  Tensor mC = make_tensor(make_gmem_ptr(device_ptr_C), layout_C);
  Tensor mD = make_tensor(make_gmem_ptr(device_ptr_D), layout_D);

  using TiledMMA = decltype(make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K,
                           UMMA::Major::K>{}));
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256, UMMA::Major::K,
                           UMMA::Major::K>{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  if (!evenly_divides(shape(mma_tiler), tile_shape(tiled_mma))) {
    throw std::runtime_error("step2 mma_tiler must evenly divide tiled_mma");
  }
  if (!evenly_divides(
          make_shape(shape<0>(layout_A), shape<0>(layout_B), shape<1>(layout_A)),
          mma_tiler)) {
    throw std::runtime_error("step2 does not support OOB tiles");
  }

  using ASmemLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<TypeA>{},
      partition_shape_A(TiledMMA{}, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)))));
  using BSmemLayout = decltype(UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<TypeB>{},
      partition_shape_B(TiledMMA{}, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)))));
  using SharedStorage =
      SharedStorageStep2<TypeA, TypeB, ASmemLayout, BSmemLayout>;

  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape),
      make_tile(typename TiledMMA::AtomThrID{}));

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                  size<2>(cluster_shape));
  dim3 dimGrid(
      size(ceil_div(shape<0>(layout_A), bM * size<1>(cluster_layout_vmnk))) *
          dimCluster.x,
      size(ceil_div(shape<0>(layout_B), bN * size<2>(cluster_layout_vmnk))) *
          dimCluster.y);
  int smemBytes = sizeof(SharedStorage);

  auto* kernel_ptr = &gemm_device_step2<SharedStorage, decltype(mA), decltype(mB),
                                        decltype(mC), decltype(mD),
                                        decltype(mma_tiler), TiledMMA,
                                        decltype(cluster_shape), Alpha, Beta>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, reinterpret_cast<void const*>(kernel_ptr), mA, mB, mC, mD,
      mma_tiler, tiled_mma, cluster_shape, alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("Failed to launch tcgen05 step2 kernel");
  }
}

#endif

template <typename LaunchFn>
TimingStats benchmark_kernel(const Step2Options& options,
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

void print_step2_result_line(const Step2Options& options,
                             const char* backend,
                             const TimingStats& stats) {
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT benchmark=bench_step2_tcgen05_mma"
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Step2Options options = parse_options(argc, argv);
    print_device_info();

    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    if (props.major != 10) {
      throw std::runtime_error("step2 requires Blackwell tcgen05 support");
    }

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
    throw std::runtime_error(
        "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not enabled for this build");
#else
    using TypeA = cutlass::half_t;
    using TypeB = cutlass::half_t;
    using TypeC = float;
    using TypeD = float;

    auto layout_A = make_layout(make_shape(options.m, options.k),
                                make_stride(options.k, Int<1>{}));
    auto layout_B = make_layout(make_shape(options.n, options.k),
                                make_stride(options.k, Int<1>{}));
    auto layout_C = make_layout(make_shape(options.m, options.n),
                                make_stride(options.n, Int<1>{}));
    auto layout_D = make_layout(make_shape(options.m, options.n),
                                make_stride(options.n, Int<1>{}));

    thrust::host_vector<TypeA> host_A(options.m * options.k);
    thrust::host_vector<TypeB> host_B(options.n * options.k);
    thrust::host_vector<TypeC> host_C(options.m * options.n);

    auto host_tensor_A = make_tensor(host_A.data(), layout_A);
    auto host_tensor_B = make_tensor(host_B.data(), layout_B);
    auto host_tensor_C = make_tensor(host_C.data(), layout_C);
    initialize_exact_tensor(host_tensor_A, options.seed);
    initialize_exact_tensor(host_tensor_B, options.seed + 1);
    zero_tensor(host_tensor_C);

    thrust::device_vector<TypeA> device_A = host_A;
    thrust::device_vector<TypeB> device_B = host_B;
    thrust::device_vector<TypeC> device_C = host_C;
    thrust::device_vector<TypeD> device_tcgen05_D(options.m * options.n);
    thrust::device_vector<TypeD> device_cublas_D(options.m * options.n);

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    auto tcgen05_launch = [&] {
      launch_tcgen05_gemm(
          thrust::raw_pointer_cast(device_A.data()), layout_A,
          thrust::raw_pointer_cast(device_B.data()), layout_B,
          thrust::raw_pointer_cast(device_C.data()), layout_C,
          thrust::raw_pointer_cast(device_tcgen05_D.data()), layout_D, 1.0f, 0.0f);
    };
    auto cublas_launch = [&] {
      run_cublas_reference(handle, options,
                           thrust::raw_pointer_cast(device_A.data()),
                           thrust::raw_pointer_cast(device_B.data()),
                           thrust::raw_pointer_cast(device_cublas_D.data()));
    };

    tcgen05_launch();
    cublas_launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> host_tcgen05_D(options.m * options.n, 0.0f);
    std::vector<float> host_cublas_D(options.m * options.n, 0.0f);
    CHECK_CUDA(cudaMemcpy(host_tcgen05_D.data(),
                          thrust::raw_pointer_cast(device_tcgen05_D.data()),
                          host_tcgen05_D.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_cublas_D.data(),
                          thrust::raw_pointer_cast(device_cublas_D.data()),
                          host_cublas_D.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    const CompareStats compare = compare_exact(host_cublas_D, host_tcgen05_D);
    std::cout << std::scientific << std::setprecision(6)
              << "CHECK benchmark=bench_step2_tcgen05_mma"
              << " backend=tcgen05_mma"
              << " m=" << options.m
              << " n=" << options.n
              << " k=" << options.k
              << " pass=" << static_cast<int>(compare.pass)
              << " max_abs=" << compare.max_abs
              << " fail_count=" << compare.fail_count
              << '\n';

    if (!compare.pass) {
      throw std::runtime_error("tcgen05 output does not exactly match cuBLAS");
    }

    TimingStats tcgen05_stats = benchmark_kernel(
        options, tcgen05_launch, thrust::raw_pointer_cast(device_tcgen05_D.data()),
        host_tcgen05_D);
    TimingStats cublas_stats = benchmark_kernel(
        options, cublas_launch, thrust::raw_pointer_cast(device_cublas_D.data()),
        host_cublas_D);

    print_step2_result_line(options, "tcgen05_mma", tcgen05_stats);
    print_step2_result_line(options, "cublas", cublas_stats);

    CHECK_CUBLAS(cublasDestroy(handle));
    return 0;
#endif
  } catch (const std::exception& e) {
    std::cerr << "ERROR " << e.what() << '\n';
    return 1;
  }
}
