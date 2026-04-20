#include "gemm_reference_common.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

#ifndef CUBLAS_COMPUTE_32F_FAST_16BF
#define CUBLAS_COMPUTE_32F_FAST_16BF CUBLAS_COMPUTE_32F
#endif

constexpr int kWarpSize = 32;
constexpr int kWarpTileM = 64;
constexpr int kWarpTileN = 64;
constexpr int kWarpTileK = 16;
constexpr int kThreads = 128;

struct Step7Options {
  int m = 128;
  int n = 64;
  int k = 32;
  int warmup = 5;
  int iters = 20;
  int seed = 2026;
};

template <typename T, typename U>
__host__ __device__ constexpr auto align_up_local(T x, U boundary) {
  return (x + boundary - 1) & ~(boundary - 1);
}

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
      << "  --m=<int>        GEMM M, multiple of 64\n"
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
  if (options.m % kWarpTileM != 0 || options.n % kWarpTileN != 0 ||
      options.k % kWarpTileK != 0) {
    throw std::runtime_error(
        "step7 tcgen05 umma currently requires m multiple of 64, n multiple of 64, k multiple of 16");
  }
  return options;
}

void initialize_exact_tensor(std::vector<nv_bfloat16>& tensor, int seed) {
  for (std::size_t i = 0; i < tensor.size(); ++i) {
    int value = ((static_cast<int>(i) * 17 + seed * 13) % 5) - 2;
    tensor[i] = __float2bfloat16(static_cast<float>(value));
  }
}

void run_cublas_reference(cublasHandle_t handle,
                          const Step7Options& options,
                          const nv_bfloat16* d_a,
                          const nv_bfloat16* d_b,
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

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

template <int M, int N>
__device__ constexpr uint32_t make_i_desc_bf16() {
  uint32_t desc = 0;
  desc |= (1U << 4);
  desc |= (1U << 7);
  desc |= (1U << 10);
  desc |= ((N >> 3) << 17);
  desc |= ((M >> 4) << 24);
  return desc;
}

__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr, int height) {
  int addr = static_cast<int>(__cvta_generic_to_shared(ptr));
  uint64_t desc = 0;
  desc |= (addr >> 4) & 0x3FFF;
  desc |= ((height * 16) >> 4) << 16;
  desc |= (8ULL << 32);
  desc |= (1ULL << 46);
  return desc;
}

__device__ __forceinline__ void mbarrier_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(addr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint32_t addr,
                                                     uint32_t phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}"
      :
      : "r"(addr), "r"(phase)
      : "memory");
}

__device__ __forceinline__ void tcgen05_alloc(uint32_t addr_ptr,
                                              uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :
               : "r"(addr_ptr), "r"(cols)
               : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr,
                                                uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :
               : "r"(tmem_addr), "r"(cols)
               : "memory");
}

__device__ __forceinline__ void tcgen05_mma(uint32_t tmem_d,
                                            uint64_t desc_a,
                                            uint64_t desc_b,
                                            uint32_t i_desc,
                                            uint32_t pred) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}"
      :
      : "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(i_desc), "r"(pred)
      : "memory");
}

__device__ __forceinline__ void tcgen05_commit(uint32_t mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :
      : "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tmem_load4(float& d0,
                                           float& d1,
                                           float& d2,
                                           float& d3,
                                           uint32_t src_addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(src_addr));
}

__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t make_tmem_addr_row_col(int row, int col) {
  const uint32_t dp =
      static_cast<uint32_t>((row & 0xF) + ((row >> 4) * 32));
  return static_cast<uint32_t>(col) | (dp << 16);
}

struct SharedStorage {
  __align__(128) nv_bfloat16 a[kWarpTileM][kWarpTileK];
  __align__(128) nv_bfloat16 b[kWarpTileN][kWarpTileK];
  uint64_t mbar;
  uint32_t tmem_addr;
};

__global__ __cluster_dims__(1, 1, 1) __launch_bounds__(kThreads)
void step7_tcgen05_umma_kernel(const nv_bfloat16* __restrict__ A,
                               const nv_bfloat16* __restrict__ B,
                               float* __restrict__ D,
                               int M,
                               int N,
                               int K) {
  __shared__ SharedStorage shared;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int block_m = blockIdx.y * kWarpTileM;
  const int block_n = blockIdx.x * kWarpTileN;

  if (warp_id == 0 && elect_sync()) {
    const uint32_t mbar_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(&shared.mbar));
    mbarrier_init(mbar_addr, 1);
    fence_mbarrier_init();
  }

  const uint32_t tmem_cols =
      static_cast<uint32_t>(align_up_local(kWarpTileN < 32 ? 32 : kWarpTileN, 32));
  if (warp_id == 0) {
    const uint32_t tmem_addr_smem =
        static_cast<uint32_t>(__cvta_generic_to_shared(&shared.tmem_addr));
    tcgen05_alloc(tmem_addr_smem, tmem_cols);
  }
  __syncthreads();

  const uint32_t tmem_d = shared.tmem_addr;
  const uint32_t mbar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&shared.mbar));
  const uint64_t a_desc = make_smem_desc(shared.a, kWarpTileM);
  const uint64_t b_desc = make_smem_desc(shared.b, kWarpTileN);
  constexpr uint32_t i_desc = make_i_desc_bf16<kWarpTileM, kWarpTileN>();

  uint32_t phase = 0;
  for (int k_base = 0; k_base < K; k_base += kWarpTileK) {
    for (int idx = tid; idx < kWarpTileM * kWarpTileK; idx += blockDim.x) {
      const int row = idx / kWarpTileK;
      const int col = idx % kWarpTileK;
      shared.a[row][col] = A[(block_m + row) * K + (k_base + col)];
    }
    for (int idx = tid; idx < kWarpTileN * kWarpTileK; idx += blockDim.x) {
      const int row = idx / kWarpTileK;
      const int col = idx % kWarpTileK;
      shared.b[row][col] = B[(block_n + row) * K + (k_base + col)];
    }
    __syncthreads();

    if (warp_id == 0 && elect_sync()) {
      tcgen05_mma(tmem_d, a_desc, b_desc, i_desc, (k_base == 0) ? 0 : 1);
      tcgen05_commit(mbar_addr);
    }

    mbarrier_wait_parity(mbar_addr, phase);
    phase ^= 1;
    __syncthreads();
  }

  __syncthreads();
  tcgen05_fence_after_sync();
  __syncthreads();

  if (tid < kWarpTileM) {
    const int row = tid;
    const int global_row = block_m + row;
    for (int col = 0; col < kWarpTileN; col += 4) {
      float d0 = 0.0f;
      float d1 = 0.0f;
      float d2 = 0.0f;
      float d3 = 0.0f;
      const uint32_t src_addr = make_tmem_addr_row_col(row, col);
      tmem_load4(d0, d1, d2, d3, src_addr);
      tmem_wait_ld();
      D[global_row * N + (block_n + col + 0)] = d0;
      D[global_row * N + (block_n + col + 1)] = d1;
      D[global_row * N + (block_n + col + 2)] = d2;
      D[global_row * N + (block_n + col + 3)] = d3;
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    tcgen05_dealloc(tmem_d, tmem_cols);
  }
}

void launch_tcgen05_umma_gemm(const nv_bfloat16* d_a,
                              const nv_bfloat16* d_b,
                              float* d_d,
                              const Step7Options& options) {
  dim3 block(kThreads);
  dim3 grid(options.n / kWarpTileN, options.m / kWarpTileM);
  step7_tcgen05_umma_kernel<<<grid, block>>>(d_a, d_b, d_d, options.m,
                                             options.n, options.k);
  CHECK_CUDA(cudaGetLastError());
}

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
  auto print_row = [&](const char* backend,
                       const TimingStats& stats,
                       double baseline_ms) {
    const double speedup = baseline_ms / stats.median_ms;
    std::cout << std::fixed << std::setprecision(6)
              << "  " << std::left << std::setw(20) << backend
              << " median_ms=" << std::setw(10) << stats.median_ms
              << " tflops=" << std::setw(9) << (stats.gflops / 1000.0)
              << " speedup_vs_fast=" << std::setw(8) << speedup
              << " checksum=" << stats.checksum << '\n';
  };

  std::cout << "SUMMARY benchmark=bench_step7_tcgen05_umma"
            << " m=" << options.m
            << " n=" << options.n
            << " k=" << options.k
            << " baseline=cublas_fast_16bf"
            << '\n';
  std::cout << "  correctness: tcgen05_umma vs cublas_pedantic exact_pass="
            << static_cast<int>(tcgen05_compare.pass)
            << " max_abs=" << std::scientific << std::setprecision(6)
            << tcgen05_compare.max_abs
            << " fail_count=" << tcgen05_compare.fail_count << '\n';
  std::cout << "  fast_math_delta: cublas_fast_16bf vs cublas_pedantic pass="
            << static_cast<int>(fast_compare.pass)
            << " max_abs=" << std::scientific << std::setprecision(6)
            << fast_compare.max_abs
            << " fail_count=" << fast_compare.fail_count << '\n';
  print_row("tcgen05_umma", tcgen05_stats, cublas_fast_stats.median_ms);
  print_row("cublas_fast_16bf", cublas_fast_stats, cublas_fast_stats.median_ms);
  print_row("cublas_pedantic", cublas_pedantic_stats,
            cublas_fast_stats.median_ms);
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
          "step7 tcgen05 umma path requires SM100 or newer");
    }

    const std::size_t a_elems =
        static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_elems =
        static_cast<std::size_t>(options.n) * options.k;
    const std::size_t d_elems =
        static_cast<std::size_t>(options.m) * options.n;

    std::vector<nv_bfloat16> host_A(a_elems);
    std::vector<nv_bfloat16> host_B(b_elems);
    initialize_exact_tensor(host_A, options.seed);
    initialize_exact_tensor(host_B, options.seed + 1);

    nv_bfloat16* device_A = nullptr;
    nv_bfloat16* device_B = nullptr;
    float* device_tcgen05_D = nullptr;
    float* device_cublas_pedantic_D = nullptr;
    float* device_cublas_fast_D = nullptr;
    CHECK_CUDA(cudaMalloc(&device_A, a_elems * sizeof(nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&device_B, b_elems * sizeof(nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&device_tcgen05_D, d_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_cublas_pedantic_D, d_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_cublas_fast_D, d_elems * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_A, host_A.data(),
                          a_elems * sizeof(nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_B, host_B.data(),
                          b_elems * sizeof(nv_bfloat16),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    auto tcgen05_launch = [&] {
      launch_tcgen05_umma_gemm(device_A, device_B, device_tcgen05_D, options);
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

    std::vector<float> host_tcgen05_D(d_elems, 0.0f);
    std::vector<float> host_cublas_pedantic_D(d_elems, 0.0f);
    std::vector<float> host_cublas_fast_D(d_elems, 0.0f);
    CHECK_CUDA(cudaMemcpy(host_tcgen05_D.data(), device_tcgen05_D,
                          d_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_cublas_pedantic_D.data(), device_cublas_pedantic_D,
                          d_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_cublas_fast_D.data(), device_cublas_fast_D,
                          d_elems * sizeof(float), cudaMemcpyDeviceToHost));

    const CompareStats tcgen05_compare =
        compare_exact(host_cublas_pedantic_D, host_tcgen05_D);
    const CompareStats fast_compare =
        compare_exact(host_cublas_pedantic_D, host_cublas_fast_D);
    print_step7_check_line(options, "tcgen05_umma_vs_cublas_pedantic",
                           tcgen05_compare, true);
    print_step7_check_line(options, "cublas_fast_16bf_vs_cublas_pedantic",
                           fast_compare, false);

    if (!tcgen05_compare.pass) {
      throw std::runtime_error(
          "tcgen05 umma output does not exactly match cuBLAS");
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

    print_step7_result_line(options, "tcgen05_umma", tcgen05_stats);
    print_step7_result_line(options, "cublas_fast_16bf", cublas_fast_stats);
    print_step7_result_line(options, "cublas_pedantic", cublas_pedantic_stats);
    print_step7_summary(options, tcgen05_stats, cublas_fast_stats,
                        cublas_pedantic_stats, tcgen05_compare, fast_compare);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(device_A));
    CHECK_CUDA(cudaFree(device_B));
    CHECK_CUDA(cudaFree(device_tcgen05_D));
    CHECK_CUDA(cudaFree(device_cublas_pedantic_D));
    CHECK_CUDA(cudaFree(device_cublas_fast_D));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR " << e.what() << '\n';
    return 1;
  }
}
