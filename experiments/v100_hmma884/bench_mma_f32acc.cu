#include "common.h"
#include <cuda_fp16.h>

namespace {

using namespace v100_hmma884;

struct F32Acc {
    float x[8];
};

__device__ __forceinline__ void mma_f32_once(unsigned a0, unsigned a1, unsigned b0, unsigned b1, unsigned c0, unsigned c1, unsigned c2, unsigned c3, F32Acc& acc) {
    float d0, d1, d2, d3, d4, d5, d6, d7;
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11}, "
        "{%12, %13, %14, %15};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    acc.x[0] = d0;
    acc.x[1] = d1;
    acc.x[2] = d2;
    acc.x[3] = d3;
    acc.x[4] = d4;
    acc.x[5] = d5;
    acc.x[6] = d6;
    acc.x[7] = d7;
}

template <int STREAMS>
__device__ __forceinline__ void mma_f32_streams_once(
    unsigned a0,
    unsigned a1,
    unsigned b0,
    unsigned b1,
    unsigned (&c)[STREAMS][4],
    F32Acc (&acc)[STREAMS]) {
#pragma unroll
    for (int s = 0; s < STREAMS; ++s) {
        mma_f32_once(a0, a1, b0, b1, c[s][0], c[s][1], c[s][2], c[s][3], acc[s]);
    }
}

template <int STREAMS>
__global__ void bench_mma_f32acc_kernel(DeviceResult* out, int warmup_iters, int loop_iters, int unroll) {
    if (threadIdx.x >= kThreadsPerBlock) return;

    unsigned a0 = 0x3c003c00u;
    unsigned a1 = 0x3c003c00u;
    unsigned b0 = 0x3c003c00u;
    unsigned b1 = 0x3c003c00u;
    unsigned c[STREAMS][4] = {};
    F32Acc acc[STREAMS] = {};

    #pragma unroll 1
    for (int i = 0; i < warmup_iters; ++i) {
        mma_f32_streams_once<STREAMS>(a0, a1, b0, b1, c, acc);
    }

    __syncthreads();
    const unsigned long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < loop_iters; ++i) {
        if (unroll >= 1) mma_f32_streams_once<STREAMS>(a0, a1, b0, b1, c, acc);
        if (unroll >= 2) mma_f32_streams_once<STREAMS>(a0, a1, b0, b1, c, acc);
        if (unroll >= 4) {
            mma_f32_streams_once<STREAMS>(a0, a1, b0, b1, c, acc);
            mma_f32_streams_once<STREAMS>(a0, a1, b0, b1, c, acc);
        }
        if (unroll >= 8) {
            for (int j = 0; j < 4; ++j) mma_f32_streams_once<STREAMS>(a0, a1, b0, b1, c, acc);
        }
        if (unroll >= 16) {
            for (int j = 0; j < 8; ++j) mma_f32_streams_once<STREAMS>(a0, a1, b0, b1, c, acc);
        }
    }

    const unsigned long long stop = clock64();

    if (threadIdx.x == 0) {
        unsigned long long lo = 0;
        unsigned long long hi = 0;
#pragma unroll
        for (int s = 0; s < STREAMS; ++s) {
            lo ^= (static_cast<unsigned long long>(__float_as_uint(acc[s].x[1])) << 32) | __float_as_uint(acc[s].x[0]);
            hi ^= (static_cast<unsigned long long>(__float_as_uint(acc[s].x[3])) << 32) | __float_as_uint(acc[s].x[2]);
            lo ^= (static_cast<unsigned long long>(__float_as_uint(acc[s].x[5])) << 32) | __float_as_uint(acc[s].x[4]);
            hi ^= (static_cast<unsigned long long>(__float_as_uint(acc[s].x[7])) << 32) | __float_as_uint(acc[s].x[6]);
        }
        out->cycles = stop - start;
        out->sink_lo = lo;
        out->sink_hi = hi;
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace v100_hmma884;

    BenchOptions options;
    options.dtype = "f32_f16_f16_f16";
    options.mode = "fixedc";
    options.streams = 1;
    parse_common_args(argc, argv, &options);

    if (!valid_unroll(options.unroll)) {
        std::fprintf(stderr, "Unsupported unroll=%d\n", options.unroll);
        return 1;
    }
    if (!valid_streams(options.streams)) {
        std::fprintf(stderr, "Unsupported streams=%d\n", options.streams);
        return 1;
    }
    options.mode = options.streams == 1 ? "fixedc" : "fixedc_indep";

    DeviceResult* d_result = alloc_device_result();
    print_options("bench_mma_f32acc", options);

    for (int i = 0; i < options.warmup_launches; ++i) {
        switch (options.streams) {
            case 1: bench_mma_f32acc_kernel<1><<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll); break;
            case 2: bench_mma_f32acc_kernel<2><<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll); break;
            case 4: bench_mma_f32acc_kernel<4><<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll); break;
            case 8: bench_mma_f32acc_kernel<8><<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll); break;
            default: std::fprintf(stderr, "Unsupported streams=%d\n", options.streams); return 1;
        }
        check_kernel("bench_mma_f32acc warmup");
    }

    const unsigned long long total_mma = static_cast<unsigned long long>(options.loop_iters) * static_cast<unsigned long long>(options.unroll) * static_cast<unsigned long long>(options.streams);
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        switch (options.streams) {
            case 1: bench_mma_f32acc_kernel<1><<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll); break;
            case 2: bench_mma_f32acc_kernel<2><<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll); break;
            case 4: bench_mma_f32acc_kernel<4><<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll); break;
            case 8: bench_mma_f32acc_kernel<8><<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll); break;
            default: std::fprintf(stderr, "Unsupported streams=%d\n", options.streams); return 1;
        }
        check_kernel("bench_mma_f32acc");
        const DeviceResult result = fetch_result(d_result);
        print_result_line("bench_mma_f32acc", options.dtype, options.mode, options.streams, repeat, options.loop_iters, options.unroll, total_mma, result);
    }

    free_device_result(d_result);
    return 0;
}
