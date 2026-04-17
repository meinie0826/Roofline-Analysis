#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace v100_hmma884 {

constexpr int kThreadsPerBlock = 32;
constexpr int kBlocks = 1;

struct DeviceResult {
    unsigned long long cycles;
    unsigned long long sink_lo;
    unsigned long long sink_hi;
};

struct BenchOptions {
    int loop_iters = 4096;
    int warmup_iters = 64;
    int warmup_launches = 3;
    int repeats = 10;
    int unroll = 8;
    int streams = 1;
    const char* dtype = "f16";
    const char* mode = "dep";
};

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

inline const char* parse_str_arg(const char* arg, const char* prefix, const char* current) {
    return starts_with(arg, prefix) ? arg + std::strlen(prefix) : current;
}

inline void parse_common_args(int argc, char** argv, BenchOptions* options) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        options->loop_iters = parse_int_arg(arg, "--loop-iters=", options->loop_iters);
        options->warmup_iters = parse_int_arg(arg, "--warmup-iters=", options->warmup_iters);
        options->warmup_launches = parse_int_arg(arg, "--warmup-launches=", options->warmup_launches);
        options->repeats = parse_int_arg(arg, "--repeats=", options->repeats);
        options->unroll = parse_int_arg(arg, "--unroll=", options->unroll);
        options->streams = parse_int_arg(arg, "--streams=", options->streams);
        options->dtype = parse_str_arg(arg, "--dtype=", options->dtype);
        options->mode = parse_str_arg(arg, "--mode=", options->mode);
    }
}

inline void print_result_line(
    const char* benchmark,
    const char* dtype,
    const char* mode,
    int streams,
    int repeat,
    int loop_iters,
    int unroll,
    unsigned long long total_mma,
    const DeviceResult& result) {
    const double cycles_per_mma = total_mma ? static_cast<double>(result.cycles) / static_cast<double>(total_mma) : 0.0;
    std::printf(
        "RESULT benchmark=%s dtype=%s mode=%s streams=%d repeat=%d loop_iters=%d unroll=%d total_mma=%llu cycles=%llu cycles_per_mma=%.8f sink_lo=%llu sink_hi=%llu\n",
        benchmark,
        dtype,
        mode,
        streams,
        repeat,
        loop_iters,
        unroll,
        total_mma,
        result.cycles,
        cycles_per_mma,
        result.sink_lo,
        result.sink_hi);
}

inline void print_options(const char* benchmark, const BenchOptions& options) {
    std::printf(
        "CONFIG benchmark=%s dtype=%s mode=%s streams=%d loop_iters=%d warmup_iters=%d warmup_launches=%d repeats=%d unroll=%d\n",
        benchmark,
        options.dtype,
        options.mode,
        options.streams,
        options.loop_iters,
        options.warmup_iters,
        options.warmup_launches,
        options.repeats,
        options.unroll);
}

inline DeviceResult fetch_result(DeviceResult* d_result) {
    DeviceResult h_result{};
    check_cuda(cudaMemcpy(&h_result, d_result, sizeof(DeviceResult), cudaMemcpyDeviceToHost), "cudaMemcpy result");
    return h_result;
}

inline DeviceResult* alloc_device_result() {
    DeviceResult* d_result = nullptr;
    check_cuda(cudaMalloc(&d_result, sizeof(DeviceResult)), "cudaMalloc result");
    return d_result;
}

inline void free_device_result(DeviceResult* d_result) {
    check_cuda(cudaFree(d_result), "cudaFree result");
}

inline void check_kernel(const char* benchmark) {
    (void)benchmark;
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "kernel synchronize");
}

inline bool valid_unroll(int unroll) {
    return unroll == 1 || unroll == 2 || unroll == 4 || unroll == 8 || unroll == 16;
}

inline bool valid_streams(int streams) {
    return streams == 1 || streams == 2 || streams == 4 || streams == 8;
}

}  // namespace v100_hmma884
