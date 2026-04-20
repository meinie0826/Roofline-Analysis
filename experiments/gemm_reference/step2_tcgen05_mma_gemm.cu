#include "gemm_reference_common.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kTileM = 128;
constexpr int kTileN = 64;
constexpr int kUmmaK = 16;
constexpr uint32_t kUmmaLayoutNone = 0;
constexpr int kCanonicalInnerMn = 8;
constexpr int kCanonicalInnerK = 8;

struct Step2Options {
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
      << "  --m=<int>        GEMM M, multiple of 128\n"
      << "  --n=<int>        GEMM N, multiple of 64\n"
      << "  --k=<int>        GEMM K, multiple of 16\n"
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
  if (options.m % kTileM != 0 || options.n % kTileN != 0 ||
      options.k % kUmmaK != 0) {
    throw std::runtime_error(
        "step2 currently requires m multiple of 128, n multiple of 64, k multiple of 16");
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
                          const Step2Options& options,
                          const nv_bfloat16* d_a,
                          const nv_bfloat16* d_b,
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
                            CUDA_R_16BF,
                            options.k,
                            d_a,
                            CUDA_R_16BF,
                            options.k,
                            &beta,
                            d_d,
                            CUDA_R_32F,
                            options.n,
                            CUBLAS_COMPUTE_32F_PEDANTIC,
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

__device__ __forceinline__ void mbarrier_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(addr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_wait(uint32_t addr, uint32_t phase) {
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
                                              uint32_t num_cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :
               : "r"(addr_ptr), "r"(num_cols)
               : "memory");
}

__device__ __forceinline__ void tcgen05_dealloc(uint32_t tmem_addr,
                                                uint32_t num_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :
               : "r"(tmem_addr), "r"(num_cols)
               : "memory");
}

__device__ __forceinline__ void tcgen05_commit(uint32_t mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :
      : "r"(mbar_addr)
      : "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_ld_1x32(float& d0, uint32_t addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
               : "=f"(d0)
               : "r"(addr));
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
               :
               :
               : "memory");
}

__host__ __device__ constexpr int canonical_k_major_lbo_bytes(int mn_dim) {
  return mn_dim * 16;
}

__host__ __device__ constexpr int canonical_k_major_sbo_bytes(int mn_dim) {
  return mn_dim * 8;
}

__host__ __device__ constexpr int canonical_k_major_lbo_elems(int mn_dim) {
  return canonical_k_major_lbo_bytes(mn_dim) / sizeof(nv_bfloat16);
}

__host__ __device__ constexpr int canonical_k_major_sbo_elems(int mn_dim) {
  return canonical_k_major_sbo_bytes(mn_dim) / sizeof(nv_bfloat16);
}

__host__ __device__ constexpr int canonical_k_major_smem_elems(int mn_dim,
                                                               int k_dim) {
  return ((mn_dim / kCanonicalInnerMn - 1) * canonical_k_major_sbo_elems(mn_dim)) +
         ((k_dim / kCanonicalInnerK - 1) * canonical_k_major_lbo_elems(mn_dim)) +
         ((kCanonicalInnerMn - 1) * kCanonicalInnerK) +
         (kCanonicalInnerK - 1) + 1;
}

__host__ __device__ constexpr int canonical_k_major_index(int mn_dim,
                                                          int row,
                                                          int col) {
  return ((row & (kCanonicalInnerMn - 1)) * kCanonicalInnerK) +
         ((row / kCanonicalInnerMn) * canonical_k_major_sbo_elems(mn_dim)) +
         (col & (kCanonicalInnerK - 1)) +
         ((col / kCanonicalInnerK) * canonical_k_major_lbo_elems(mn_dim));
}

__host__ __device__ constexpr uint32_t tmem_addr_f32_1sm(uint32_t base,
                                                         int row,
                                                         int col) {
  return base + (static_cast<uint32_t>(row) << 16) +
         static_cast<uint32_t>(col * kTileM);
}

constexpr std::size_t kPackedABytes =
    canonical_k_major_smem_elems(kTileM, kUmmaK) * sizeof(nv_bfloat16);
constexpr std::size_t kPackedBOffsetBytes = align_up_local(kPackedABytes, 128);
constexpr std::size_t kPackedBBytes =
    canonical_k_major_smem_elems(kTileN, kUmmaK) * sizeof(nv_bfloat16);
constexpr std::size_t kStep2SmemBytes =
    align_up_local(kPackedBOffsetBytes + kPackedBBytes, 128);

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t addr,
                                                   uint32_t leading_byte_offset,
                                                   uint32_t stride_byte_offset,
                                                   uint32_t layout_type) {
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((addr >> 4) & 0x3FFF);
  desc |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16;
  desc |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;
  desc |= 1ULL << 46;  // version = 1
  desc |= static_cast<uint64_t>(layout_type & 0x7) << 61;
  return desc;
}

__device__ __forceinline__ uint32_t make_i_desc_f16bf16() {
  uint32_t desc = 0;
  desc |= 1U << 4;                       // c_format = F32
  desc |= 1U << 7;                       // a_format = BF16
  desc |= 1U << 10;                      // b_format = BF16
  desc |= (kTileN >> 3) << 17;           // n_dim
  desc |= (kTileM >> 4) << 24;           // m_dim
  return desc;
}

__device__ __forceinline__ void tcgen05_mma_bf16(uint32_t tmem_d,
                                                 uint64_t a_desc,
                                                 uint64_t b_desc,
                                                 uint32_t i_desc,
                                                 int accumulate) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
      "}\n"
      :
      : "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(accumulate)
      : "memory");
}

__global__ __launch_bounds__(kThreads) void step2_tcgen05_kernel(
    const nv_bfloat16* __restrict__ A,
    const nv_bfloat16* __restrict__ B,
    float* __restrict__ D,
    int M,
    int N,
    int K) {
  extern __shared__ __align__(128) unsigned char smem[];
  auto* sA = reinterpret_cast<nv_bfloat16*>(smem);
  auto* sB = reinterpret_cast<nv_bfloat16*>(smem + kPackedBOffsetBytes);

  __shared__ uint64_t mbar;
  __shared__ int tmem_addr;

  const int tid = threadIdx.x;
  const int lane_id = tid & 31;
  const int warp_id = tid / 32;
  const int tile_m = blockIdx.y * kTileM;
  const int tile_n = blockIdx.x * kTileN;

  if (tile_m >= M || tile_n >= N) {
    return;
  }

  const uint32_t mbar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
  if (tid == 0) {
    mbarrier_init(mbar_addr, 1);
    fence_mbarrier_init_release_cluster();
  }
  if (warp_id == 0) {
    uint32_t tmem_addr_smem =
        static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr));
    tcgen05_alloc(tmem_addr_smem, align_up_local(kTileN, 32));
  }
  __syncthreads();

  const uint32_t tmem_d = static_cast<uint32_t>(tmem_addr);
  const uint32_t sA_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
  const uint32_t sB_addr = static_cast<uint32_t>(__cvta_generic_to_shared(sB));
  const uint64_t a_desc =
      make_smem_desc(sA_addr,
                     canonical_k_major_lbo_bytes(kTileM),
                     canonical_k_major_sbo_bytes(kTileM),
                     kUmmaLayoutNone);
  const uint64_t b_desc =
      make_smem_desc(sB_addr,
                     canonical_k_major_lbo_bytes(kTileN),
                     canonical_k_major_sbo_bytes(kTileN),
                     kUmmaLayoutNone);
  const uint32_t i_desc = make_i_desc_f16bf16();

  int phase = 0;
  for (int k_base = 0; k_base < K; k_base += kUmmaK) {
    // PTX K-major no-swizzle canonical layout:
    // ((8, mn/8), (8, k/8*2)) : ((8, SBO), (1, LBO)) for bf16.
    for (int i = tid; i < kTileM * kUmmaK; i += blockDim.x) {
      int row = i / kUmmaK;
      int col = i % kUmmaK;
      int packed = canonical_k_major_index(kTileM, row, col);
      sA[packed] = A[(tile_m + row) * K + (k_base + col)];
    }
    for (int i = tid; i < kTileN * kUmmaK; i += blockDim.x) {
      int row = i / kUmmaK;
      int col = i % kUmmaK;
      int packed = canonical_k_major_index(kTileN, row, col);
      sB[packed] = B[(tile_n + row) * K + (k_base + col)];
    }
    __syncthreads();

    if (warp_id == 0 && elect_sync()) {
      tcgen05_mma_bf16(tmem_d, a_desc, b_desc, i_desc, k_base != 0);
      tcgen05_commit(mbar_addr);
    }
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;
    __syncthreads();
  }

  tcgen05_fence_after_thread_sync();

  if (warp_id < (kTileM / 32)) {
    const int row = warp_id * 32 + lane_id;
    for (int col = 0; col < kTileN; ++col) {
      float value = 0.0f;
      uint32_t tmem_ptr = tmem_addr_f32_1sm(tmem_d, row, col);
      tcgen05_ld_1x32(value, tmem_ptr);
      tcgen05_wait_ld();
      D[(tile_m + row) * N + (tile_n + col)] = value;
    }
  }

  __syncthreads();
  tcgen05_fence_after_thread_sync();
  __syncthreads();
  if (warp_id == 0) {
    tcgen05_relinquish_alloc_permit();
    tcgen05_dealloc(tmem_d, align_up_local(kTileN, 32));
  }
}

void launch_tcgen05_gemm(const nv_bfloat16* d_a,
                         const nv_bfloat16* d_b,
                         float* d_d,
                         const Step2Options& options) {
  dim3 block(kThreads);
  dim3 grid(options.n / kTileN, options.m / kTileM);
  std::size_t smem_bytes = kStep2SmemBytes;
  step2_tcgen05_kernel<<<grid, block, smem_bytes>>>(d_a, d_b, d_d, options.m,
                                                    options.n, options.k);
  CHECK_CUDA(cudaGetLastError());
}

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
    float* device_cublas_D = nullptr;
    CHECK_CUDA(cudaMalloc(&device_A, a_elems * sizeof(nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&device_B, b_elems * sizeof(nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&device_tcgen05_D, d_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_cublas_D, d_elems * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_A, host_A.data(), a_elems * sizeof(nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_B, host_B.data(), b_elems * sizeof(nv_bfloat16),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    auto tcgen05_launch = [&] {
      launch_tcgen05_gemm(device_A, device_B, device_tcgen05_D, options);
    };
    auto cublas_launch = [&] {
      run_cublas_reference(handle, options, device_A, device_B, device_cublas_D);
    };

    cublas_launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    tcgen05_launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> host_tcgen05_D(d_elems, 0.0f);
    std::vector<float> host_cublas_D(d_elems, 0.0f);
    CHECK_CUDA(cudaMemcpy(host_tcgen05_D.data(), device_tcgen05_D,
                          d_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_cublas_D.data(), device_cublas_D,
                          d_elems * sizeof(float), cudaMemcpyDeviceToHost));

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

    TimingStats tcgen05_stats =
        benchmark_kernel(options, tcgen05_launch, device_tcgen05_D, host_tcgen05_D);
    TimingStats cublas_stats =
        benchmark_kernel(options, cublas_launch, device_cublas_D, host_cublas_D);

    print_step2_result_line(options, "tcgen05_mma", tcgen05_stats);
    print_step2_result_line(options, "cublas", cublas_stats);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(device_A));
    CHECK_CUDA(cudaFree(device_B));
    CHECK_CUDA(cudaFree(device_tcgen05_D));
    CHECK_CUDA(cudaFree(device_cublas_D));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR " << e.what() << '\n';
    return 1;
  }
}
