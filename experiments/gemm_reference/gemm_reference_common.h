#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
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

inline bool should_run_backend(const Options& options, std::string_view backend) {
  return options.backend == "all" || options.backend == backend;
}

inline void maybe_reset_output(float* dst,
                               const float* src,
                               std::size_t bytes,
                               float beta) {
  if (beta != 0.0f) {
    CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
  }
}

inline double checksum_host(const std::vector<float>& data) {
  double checksum = 0.0;
  for (float value : data) {
    checksum += static_cast<double>(value);
  }
  return checksum;
}

inline double median_ms(std::vector<float> values) {
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

inline TimingStats finalize_timing_stats(const Options& options,
                                         const std::vector<float>& times_ms,
                                         const std::vector<float>& host_out) {
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

inline CompareStats compare_outputs(const std::vector<float>& reference,
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

inline void print_device_info() {
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

inline void print_result_line(const Options& options,
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

inline void print_check_line(const Options& options,
                             const char* backend,
                             const CompareStats& compare) {
  std::cout << std::scientific << std::setprecision(6)
            << "CHECK benchmark=bench_gemm_reference"
            << " backend=" << backend
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
}

void run_naive_once(const Options& options,
                    const float* d_a,
                    const float* d_b,
                    const float* d_c_in,
                    float* d_out);

TimingStats benchmark_naive(const Options& options,
                            const float* d_a,
                            const float* d_b,
                            const float* d_c_in,
                            float* d_out,
                            std::vector<float>& host_out);

void run_cublas_reference_once(cublasHandle_t handle,
                               const Options& options,
                               const float* d_a,
                               const float* d_b,
                               float* d_out);

TimingStats benchmark_cublas_reference(const Options& options,
                                       cublasHandle_t handle,
                                       const float* d_a,
                                       const float* d_b,
                                       const float* d_c_in,
                                       float* d_out,
                                       std::vector<float>& host_out);

void run_simt_cp_async_once(const Options& options,
                            const float* d_a,
                            const float* d_b,
                            const float* d_c_in,
                            float* d_out);

TimingStats benchmark_simt_cp_async(const Options& options,
                                    const float* d_a,
                                    const float* d_b,
                                    const float* d_c_in,
                                    float* d_out,
                                    std::vector<float>& host_out);
