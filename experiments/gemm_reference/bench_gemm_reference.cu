#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(expr)                                                        \
  do {                                                                          \
    cudaError_t status__ = (expr);                                              \
    if (status__ != cudaSuccess) {                                              \
      std::ostringstream oss__;                                                 \
      oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "          \
            << cudaGetErrorString(status__);                                    \
      throw std::runtime_error(oss__.str());                                    \
    }                                                                           \
  } while (0)

#define CHECK_CUBLAS(expr)                                                      \
  do {                                                                          \
    cublasStatus_t status__ = (expr);                                           \
    if (status__ != CUBLAS_STATUS_SUCCESS) {                                    \
      std::ostringstream oss__;                                                 \
      oss__ << "cuBLAS error at " << __FILE__ << ":" << __LINE__                \
            << ": status=" << static_cast<int>(status__);                       \
      throw std::runtime_error(oss__.str());                                    \
    }                                                                           \
  } while (0)

namespace {

struct Options {
  int m = 512;
  int n = 512;
  int k = 512;
  int warmup = 5;
  int iters = 20;
  int block_m = 16;
  int block_n = 16;
  int seed = 2026;
  float alpha = 1.0f;
  float beta = 0.0f;
  float atol = 1.0e-3f;
  float rtol = 1.0e-3f;
  std::string backend = "all";
  bool check = true;
};

struct TimingStats {
  double avg_ms = 0.0;
  double median_ms = 0.0;
  double min_ms = 0.0;
  double gflops = 0.0;
  double checksum = 0.0;
};

struct CompareStats {
  bool pass = true;
  double max_abs = 0.0;
  double max_rel = 0.0;
  double l2_rel = 0.0;
  std::size_t fail_count = 0;
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

bool parse_bool(std::string_view text, const char* name) {
  if (text == "1" || text == "true") {
    return true;
  }
  if (text == "0" || text == "false") {
    return false;
  }
  std::ostringstream oss;
  oss << "Invalid boolean for " << name << ": " << text;
  throw std::runtime_error(oss.str());
}

void print_usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --m=<int>             Rows of A/C\n"
      << "  --n=<int>             Columns of B/C\n"
      << "  --k=<int>             Columns of A / rows of B\n"
      << "  --warmup=<int>        Warmup iterations per backend\n"
      << "  --iters=<int>         Timed iterations per backend\n"
      << "  --block-m=<int>       Naive kernel block height\n"
      << "  --block-n=<int>       Naive kernel block width\n"
      << "  --seed=<int>          RNG seed\n"
      << "  --alpha=<float>       GEMM alpha\n"
      << "  --beta=<float>        GEMM beta\n"
      << "  --atol=<float>        Correctness absolute tolerance\n"
      << "  --rtol=<float>        Correctness relative tolerance\n"
      << "  --backend=<all|naive|cublas>\n"
      << "  --check=<0|1>         Compare naive output against cuBLAS\n"
      << "  --help                Show this message\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
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
    } else if (auto value = maybe_value(arg, "--block-m=")) {
      options.block_m = parse_number<int>(*value, "block-m");
    } else if (auto value = maybe_value(arg, "--block-n=")) {
      options.block_n = parse_number<int>(*value, "block-n");
    } else if (auto value = maybe_value(arg, "--seed=")) {
      options.seed = parse_number<int>(*value, "seed");
    } else if (auto value = maybe_value(arg, "--alpha=")) {
      options.alpha = parse_number<float>(*value, "alpha");
    } else if (auto value = maybe_value(arg, "--beta=")) {
      options.beta = parse_number<float>(*value, "beta");
    } else if (auto value = maybe_value(arg, "--atol=")) {
      options.atol = parse_number<float>(*value, "atol");
    } else if (auto value = maybe_value(arg, "--rtol=")) {
      options.rtol = parse_number<float>(*value, "rtol");
    } else if (auto value = maybe_value(arg, "--backend=")) {
      options.backend = std::string(*value);
    } else if (auto value = maybe_value(arg, "--check=")) {
      options.check = parse_bool(*value, "check");
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
  if (options.block_m <= 0 || options.block_n <= 0) {
    throw std::runtime_error("block-m and block-n must be positive");
  }
  if (options.backend != "all" && options.backend != "naive" &&
      options.backend != "cublas") {
    throw std::runtime_error("backend must be one of: all, naive, cublas");
  }
  return options;
}

void fill_random(std::vector<float>& data, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : data) {
    value = dist(rng);
  }
}

double checksum_host(const std::vector<float>& data) {
  double checksum = 0.0;
  for (float value : data) {
    checksum += static_cast<double>(value);
  }
  return checksum;
}

double median_ms(std::vector<float> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return 0.5 * (static_cast<double>(values[mid - 1]) +
                  static_cast<double>(values[mid]));
  }
  return static_cast<double>(values[mid]);
}

CompareStats compare_outputs(const std::vector<float>& reference,
                             const std::vector<float>& actual,
                             float atol,
                             float rtol) {
  if (reference.size() != actual.size()) {
    throw std::runtime_error("compare_outputs size mismatch");
  }

  CompareStats stats;
  long double ref_norm_sq = 0.0;
  long double diff_norm_sq = 0.0;

  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double ref = static_cast<double>(reference[i]);
    const double got = static_cast<double>(actual[i]);
    const double abs_diff = std::abs(ref - got);
    const double rel_diff =
        abs_diff / std::max(std::abs(ref), std::numeric_limits<double>::min());

    stats.max_abs = std::max(stats.max_abs, abs_diff);
    stats.max_rel = std::max(stats.max_rel, rel_diff);

    const double allowed = static_cast<double>(atol) +
                           static_cast<double>(rtol) * std::abs(ref);
    if (abs_diff > allowed) {
      ++stats.fail_count;
    }

    ref_norm_sq += ref * ref;
    diff_norm_sq += abs_diff * abs_diff;
  }

  stats.l2_rel = std::sqrt(static_cast<double>(
      diff_norm_sq / std::max(ref_norm_sq, static_cast<long double>(1.0e-30))));
  stats.pass = (stats.fail_count == 0);
  return stats;
}

__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  const float* __restrict__ c_in,
                                  float* __restrict__ d_out,
                                  int m,
                                  int n,
                                  int k,
                                  float alpha,
                                  float beta) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    const std::size_t a_idx = static_cast<std::size_t>(row) * k + kk;
    const std::size_t b_idx = static_cast<std::size_t>(kk) * n + col;
    acc += a[a_idx] * b[b_idx];
  }

  const std::size_t c_idx = static_cast<std::size_t>(row) * n + col;
  d_out[c_idx] = alpha * acc + beta * c_in[c_idx];
}

void maybe_reset_output(float* dst,
                        const float* src,
                        std::size_t bytes,
                        float beta) {
  if (beta != 0.0f) {
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  }
}

void launch_naive(const Options& options,
                  const float* d_a,
                  const float* d_b,
                  const float* d_c_in,
                  float* d_out) {
  const dim3 block(options.block_n, options.block_m);
  const dim3 grid((options.n + block.x - 1) / block.x,
                  (options.m + block.y - 1) / block.y);
  naive_gemm_kernel<<<grid, block>>>(d_a,
                                     d_b,
                                     d_c_in,
                                     d_out,
                                     options.m,
                                     options.n,
                                     options.k,
                                     options.alpha,
                                     options.beta);
  CHECK_CUDA(cudaGetLastError());
}

void launch_cublas(cublasHandle_t handle,
                   const Options& options,
                   const float* d_a,
                   const float* d_b,
                   float* d_out) {
  CHECK_CUBLAS(cublasSgemm(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           options.n,
                           options.m,
                           options.k,
                           &options.alpha,
                           d_b,
                           options.n,
                           d_a,
                           options.k,
                           &options.beta,
                           d_out,
                           options.n));
}

TimingStats benchmark_naive(const Options& options,
                            const float* d_a,
                            const float* d_b,
                            const float* d_c_in,
                            float* d_out,
                            std::vector<float>& host_out) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(options.m) * options.n * sizeof(float);
  for (int i = 0; i < options.warmup; ++i) {
    maybe_reset_output(d_out, d_c_in, output_bytes, options.beta);
    launch_naive(options, d_a, d_b, d_c_in, d_out);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> times_ms;
  times_ms.reserve(options.iters);
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < options.iters; ++i) {
    maybe_reset_output(d_out, d_c_in, output_bytes, options.beta);
    CHECK_CUDA(cudaEventRecord(start));
    launch_naive(options, d_a, d_b, d_c_in, d_out);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    times_ms.push_back(elapsed);
  }

  CHECK_CUDA(cudaMemcpy(host_out.data(),
                        d_out,
                        output_bytes,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  TimingStats stats;
  double sum_ms = 0.0;
  stats.min_ms = std::numeric_limits<double>::max();
  for (float value : times_ms) {
    sum_ms += value;
    stats.min_ms = std::min(stats.min_ms, static_cast<double>(value));
  }
  stats.avg_ms = sum_ms / times_ms.size();
  stats.median_ms = median_ms(times_ms);
  const double flops =
      2.0 * static_cast<double>(options.m) * options.n * options.k;
  stats.gflops = flops / (stats.median_ms * 1.0e-3) / 1.0e9;
  stats.checksum = checksum_host(host_out);
  return stats;
}

TimingStats benchmark_cublas(const Options& options,
                             cublasHandle_t handle,
                             const float* d_a,
                             const float* d_b,
                             const float* d_c_in,
                             float* d_out,
                             std::vector<float>& host_out) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(options.m) * options.n * sizeof(float);
  for (int i = 0; i < options.warmup; ++i) {
    maybe_reset_output(d_out, d_c_in, output_bytes, options.beta);
    launch_cublas(handle, options, d_a, d_b, d_out);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> times_ms;
  times_ms.reserve(options.iters);
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < options.iters; ++i) {
    maybe_reset_output(d_out, d_c_in, output_bytes, options.beta);
    CHECK_CUDA(cudaEventRecord(start));
    launch_cublas(handle, options, d_a, d_b, d_out);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    times_ms.push_back(elapsed);
  }

  CHECK_CUDA(cudaMemcpy(host_out.data(),
                        d_out,
                        output_bytes,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  TimingStats stats;
  double sum_ms = 0.0;
  stats.min_ms = std::numeric_limits<double>::max();
  for (float value : times_ms) {
    sum_ms += value;
    stats.min_ms = std::min(stats.min_ms, static_cast<double>(value));
  }
  stats.avg_ms = sum_ms / times_ms.size();
  stats.median_ms = median_ms(times_ms);
  const double flops =
      2.0 * static_cast<double>(options.m) * options.n * options.k;
  stats.gflops = flops / (stats.median_ms * 1.0e-3) / 1.0e9;
  stats.checksum = checksum_host(host_out);
  return stats;
}

void print_device_info() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp props{};
  CHECK_CUDA(cudaGetDeviceProperties(&props, device));
  std::cout << "INFO device=" << device
            << " name=" << props.name
            << " sm=" << props.major << props.minor
            << " max_threads_per_block=" << props.maxThreadsPerBlock
            << " multi_processor_count=" << props.multiProcessorCount
            << '\n';
}

void print_result_line(const Options& options,
                       const char* backend,
                       const TimingStats& stats) {
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT benchmark=bench_gemm_reference"
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
    const Options options = parse_options(argc, argv);
    print_device_info();

    std::cout << "CONFIG m=" << options.m
              << " n=" << options.n
              << " k=" << options.k
              << " warmup=" << options.warmup
              << " iters=" << options.iters
              << " block_m=" << options.block_m
              << " block_n=" << options.block_n
              << " alpha=" << options.alpha
              << " beta=" << options.beta
              << " atol=" << options.atol
              << " rtol=" << options.rtol
              << " backend=" << options.backend
              << " check=" << static_cast<int>(options.check)
              << '\n';

    const std::size_t a_elems =
        static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_elems =
        static_cast<std::size_t>(options.k) * options.n;
    const std::size_t c_elems =
        static_cast<std::size_t>(options.m) * options.n;

    std::vector<float> h_a(a_elems);
    std::vector<float> h_b(b_elems);
    std::vector<float> h_c_init(c_elems);
    std::vector<float> h_naive(c_elems, 0.0f);
    std::vector<float> h_cublas(c_elems, 0.0f);

    fill_random(h_a, options.seed);
    fill_random(h_b, options.seed + 1);
    fill_random(h_c_init, options.seed + 2);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c_init = nullptr;
    float* d_naive = nullptr;
    float* d_cublas = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, a_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, b_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c_init, c_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_naive, c_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cublas, c_elems * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(
        d_a, h_a.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_b, h_b.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c_init,
                          h_c_init.data(),
                          c_elems * sizeof(float),
                          cudaMemcpyHostToDevice));

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    if (options.check) {
      maybe_reset_output(
          d_naive, d_c_init, c_elems * sizeof(float), options.beta);
      launch_naive(options, d_a, d_b, d_c_init, d_naive);
      maybe_reset_output(
          d_cublas, d_c_init, c_elems * sizeof(float), options.beta);
      launch_cublas(handle, options, d_a, d_b, d_cublas);
      CHECK_CUDA(cudaDeviceSynchronize());

      CHECK_CUDA(cudaMemcpy(h_naive.data(),
                            d_naive,
                            c_elems * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CHECK_CUDA(cudaMemcpy(h_cublas.data(),
                            d_cublas,
                            c_elems * sizeof(float),
                            cudaMemcpyDeviceToHost));

      const CompareStats compare =
          compare_outputs(h_cublas, h_naive, options.atol, options.rtol);
      std::cout << std::scientific << std::setprecision(6)
                << "CHECK benchmark=bench_gemm_reference"
                << " m=" << options.m
                << " n=" << options.n
                << " k=" << options.k
                << " pass=" << static_cast<int>(compare.pass)
                << " atol=" << options.atol
                << " rtol=" << options.rtol
                << " max_abs=" << compare.max_abs
                << " max_rel=" << compare.max_rel
                << " l2_rel=" << compare.l2_rel
                << " fail_count=" << compare.fail_count
                << '\n';

      if (!compare.pass) {
        std::cerr << "Correctness check failed.\n";
        CHECK_CUBLAS(cublasDestroy(handle));
        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c_init));
        CHECK_CUDA(cudaFree(d_naive));
        CHECK_CUDA(cudaFree(d_cublas));
        return 1;
      }
    }

    if (options.backend == "all" || options.backend == "naive") {
      const TimingStats stats =
          benchmark_naive(options, d_a, d_b, d_c_init, d_naive, h_naive);
      print_result_line(options, "naive", stats);
    }

    if (options.backend == "all" || options.backend == "cublas") {
      const TimingStats stats = benchmark_cublas(
          options, handle, d_a, d_b, d_c_init, d_cublas, h_cublas);
      print_result_line(options, "cublas", stats);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c_init));
    CHECK_CUDA(cudaFree(d_naive));
    CHECK_CUDA(cudaFree(d_cublas));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR " << e.what() << '\n';
    return 1;
  }
}
