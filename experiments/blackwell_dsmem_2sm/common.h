#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace blackwell_dsmem_2sm {

inline void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

inline bool starts_with(const char* arg, const char* prefix) {
  return std::strncmp(arg, prefix, std::strlen(prefix)) == 0;
}

inline int parse_int_arg(const char* arg, const char* prefix, int current) {
  return starts_with(arg, prefix) ? std::atoi(arg + std::strlen(prefix)) : current;
}

inline long long parse_i64_arg(const char* arg, const char* prefix, long long current) {
  return starts_with(arg, prefix) ? std::atoll(arg + std::strlen(prefix)) : current;
}

inline const char* parse_str_arg(const char* arg, const char* prefix, const char* current) {
  return starts_with(arg, prefix) ? (arg + std::strlen(prefix)) : current;
}

struct StreamOptions {
  int repeats = 20;
  int warmup_repeats = 5;
  int iters = 2048;
  int buffer_bytes = 65536;
  int align_bytes = 128;
  int vec_bytes = 16;
  int cluster_dim_x = 2;
  const char* mode = "remote";
};

inline void parse_stream_options(int argc, char** argv, StreamOptions* options) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    options->repeats = parse_int_arg(arg, "--repeats=", options->repeats);
    options->warmup_repeats = parse_int_arg(arg, "--warmup-repeats=", options->warmup_repeats);
    options->iters = parse_int_arg(arg, "--iters=", options->iters);
    options->buffer_bytes = parse_int_arg(arg, "--buffer-bytes=", options->buffer_bytes);
    options->align_bytes = parse_int_arg(arg, "--align-bytes=", options->align_bytes);
    options->vec_bytes = parse_int_arg(arg, "--vec-bytes=", options->vec_bytes);
    options->cluster_dim_x = parse_int_arg(arg, "--cluster-dim-x=", options->cluster_dim_x);
    options->mode = parse_str_arg(arg, "--mode=", options->mode);
  }
}

struct PingPongOptions {
  int repeats = 20;
  int warmup_repeats = 5;
  int iters = 10000;
  int cluster_dim_x = 2;
};

inline void parse_pingpong_options(int argc, char** argv, PingPongOptions* options) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    options->repeats = parse_int_arg(arg, "--repeats=", options->repeats);
    options->warmup_repeats = parse_int_arg(arg, "--warmup-repeats=", options->warmup_repeats);
    options->iters = parse_int_arg(arg, "--iters=", options->iters);
    options->cluster_dim_x = parse_int_arg(arg, "--cluster-dim-x=", options->cluster_dim_x);
  }
}

struct GemmOptions {
  int m = 4096;
  int n = 128;
  int k = 4096;
  int tile_n = 128;
  int stages = 2;
  int repeats = 20;
  int warmup_repeats = 5;
  int cluster_dim_x = 2;
  int verify = 0;
  const char* mode = "remote";
};

inline void parse_gemm_options(int argc, char** argv, GemmOptions* options) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    options->m = parse_int_arg(arg, "--m=", options->m);
    options->n = parse_int_arg(arg, "--n=", options->n);
    options->k = parse_int_arg(arg, "--k=", options->k);
    options->tile_n = parse_int_arg(arg, "--tile-n=", options->tile_n);
    options->stages = parse_int_arg(arg, "--stages=", options->stages);
    options->repeats = parse_int_arg(arg, "--repeats=", options->repeats);
    options->warmup_repeats = parse_int_arg(arg, "--warmup-repeats=", options->warmup_repeats);
    options->cluster_dim_x = parse_int_arg(arg, "--cluster-dim-x=", options->cluster_dim_x);
    options->verify = parse_int_arg(arg, "--verify=", options->verify);
    options->mode = parse_str_arg(arg, "--mode=", options->mode);
  }
}

inline bool is_valid_vec_bytes(int vec_bytes) {
  return vec_bytes == 4 || vec_bytes == 8 || vec_bytes == 16;
}

inline bool is_valid_align_bytes(int align_bytes) {
  return align_bytes == 32 || align_bytes == 64 || align_bytes == 128;
}

inline bool is_valid_stages(int stages) {
  return stages == 1 || stages == 2 || stages == 4;
}

inline bool is_valid_tile_n(int tile_n) {
  return tile_n == 64 || tile_n == 128 || tile_n == 256;
}

inline int div_up(int x, int y) {
  return (x + y - 1) / y;
}

template <typename T>
inline T* alloc_device(std::size_t count) {
  T* ptr = nullptr;
  check_cuda(cudaMalloc(&ptr, sizeof(T) * count), "cudaMalloc");
  return ptr;
}

template <typename T>
inline void copy_to_host(T* dst, const T* src, std::size_t count) {
  check_cuda(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost), "cudaMemcpyDtoH");
}

template <typename T>
inline void copy_to_device(T* dst, const T* src, std::size_t count) {
  check_cuda(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice), "cudaMemcpyHtoD");
}

inline double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  return static_cast<double>(ms);
}

struct StreamResult {
  unsigned long long cycles = 0;
  unsigned long long checksum = 0;
  unsigned long long bytes = 0;
};

inline void print_stream_header(const char* benchmark, const StreamOptions& options) {
  std::printf(
      "CONFIG benchmark=%s mode=%s repeats=%d warmup_repeats=%d iters=%d buffer_bytes=%d align_bytes=%d vec_bytes=%d cluster_dim_x=%d\n",
      benchmark,
      options.mode,
      options.repeats,
      options.warmup_repeats,
      options.iters,
      options.buffer_bytes,
      options.align_bytes,
      options.vec_bytes,
      options.cluster_dim_x);
}

inline void print_stream_result(
    const char* benchmark,
    const StreamOptions& options,
    int repeat,
    double sm_clock_ghz,
    const StreamResult& result) {
  const double elapsed_ns = sm_clock_ghz > 0.0 ? static_cast<double>(result.cycles) / sm_clock_ghz : 0.0;
  const double bandwidth_gbps = elapsed_ns > 0.0 ? static_cast<double>(result.bytes) / elapsed_ns : 0.0;
  std::printf(
      "RESULT benchmark=%s mode=%s repeat=%d iters=%d buffer_bytes=%d align_bytes=%d vec_bytes=%d cycles=%llu bytes=%llu elapsed_ns=%.4f bandwidth_gbps=%.4f checksum=%llu\n",
      benchmark,
      options.mode,
      repeat,
      options.iters,
      options.buffer_bytes,
      options.align_bytes,
      options.vec_bytes,
      result.cycles,
      result.bytes,
      elapsed_ns,
      bandwidth_gbps,
      result.checksum);
}

struct PingPongResult {
  unsigned long long cycles = 0;
};

inline void print_pingpong_header(const PingPongOptions& options) {
  std::printf(
      "CONFIG benchmark=bench_dsmem_pingpong repeats=%d warmup_repeats=%d iters=%d cluster_dim_x=%d\n",
      options.repeats,
      options.warmup_repeats,
      options.iters,
      options.cluster_dim_x);
}

inline void print_pingpong_result(
    const PingPongOptions& options,
    int repeat,
    double sm_clock_ghz,
    const PingPongResult& result) {
  const double elapsed_ns = sm_clock_ghz > 0.0 ? static_cast<double>(result.cycles) / sm_clock_ghz : 0.0;
  const double cycles_per_roundtrip = options.iters > 0 ? static_cast<double>(result.cycles) / static_cast<double>(options.iters) : 0.0;
  std::printf(
      "RESULT benchmark=bench_dsmem_pingpong repeat=%d iters=%d cycles=%llu elapsed_ns=%.4f cycles_per_roundtrip=%.4f ns_per_roundtrip=%.4f\n",
      repeat,
      options.iters,
      result.cycles,
      elapsed_ns,
      cycles_per_roundtrip,
      options.iters > 0 ? elapsed_ns / static_cast<double>(options.iters) : 0.0);
}

inline double query_sm_clock_ghz() {
  int khz = 0;
  check_cuda(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0), "cudaDeviceGetAttribute clockRate");
  return static_cast<double>(khz) * 1.0e-6;
}

inline std::string gpu_name() {
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  return std::string(prop.name);
}

}  // namespace blackwell_dsmem_2sm
