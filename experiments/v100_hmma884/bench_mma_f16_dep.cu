#include "common.h"
#include <cuda_fp16.h>

namespace {

using namespace v100_hmma884;

__device__ __forceinline__ void mma_f16_once(
    unsigned a0,
    unsigned a1,
    unsigned b0,
    unsigned b1,
    unsigned& c0,
    unsigned& c1,
    unsigned& c2,
    unsigned& c3) {
    unsigned d0, d1, d2, d3;
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
    c0 = d0;
    c1 = d1;
    c2 = d2;
    c3 = d3;
}

__global__ void bench_mma_f16_dep_kernel(DeviceResult* out, int warmup_iters, int loop_iters, int unroll) {
    if (threadIdx.x >= kThreadsPerBlock) return;

    unsigned a0 = 0x3c003c00u;
    unsigned a1 = 0x3c003c00u;
    unsigned b0 = 0x3c003c00u;
    unsigned b1 = 0x3c003c00u;
    unsigned c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    #pragma unroll 1
    for (int i = 0; i < warmup_iters; ++i) {
        mma_f16_once(a0, a1, b0, b1, c0, c1, c2, c3);
    }

    __syncthreads();
    const unsigned long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < loop_iters; ++i) {
        if (unroll >= 1) mma_f16_once(a0, a1, b0, b1, c0, c1, c2, c3);
        if (unroll >= 2) mma_f16_once(a0, a1, b0, b1, c0, c1, c2, c3);
        if (unroll >= 4) {
            mma_f16_once(a0, a1, b0, b1, c0, c1, c2, c3);
            mma_f16_once(a0, a1, b0, b1, c0, c1, c2, c3);
        }
        if (unroll >= 8) {
            for (int j = 0; j < 4; ++j) mma_f16_once(a0, a1, b0, b1, c0, c1, c2, c3);
        }
        if (unroll >= 16) {
            for (int j = 0; j < 8; ++j) mma_f16_once(a0, a1, b0, b1, c0, c1, c2, c3);
        }
    }

    const unsigned long long stop = clock64();

    if (threadIdx.x == 0) {
        out->cycles = stop - start;
        out->sink_lo = (static_cast<unsigned long long>(c1) << 32) | c0;
        out->sink_hi = (static_cast<unsigned long long>(c3) << 32) | c2;
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace v100_hmma884;

    BenchOptions options;
    options.dtype = "f16_f16_f16_f16";
    options.mode = "dep";
    options.streams = 1;
    parse_common_args(argc, argv, &options);

    if (!valid_unroll(options.unroll)) {
        std::fprintf(stderr, "Unsupported unroll=%d\n", options.unroll);
        return 1;
    }

    DeviceResult* d_result = alloc_device_result();
    print_options("bench_mma_f16_dep", options);

    for (int i = 0; i < options.warmup_launches; ++i) {
        bench_mma_f16_dep_kernel<<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll);
        check_kernel("bench_mma_f16_dep warmup");
    }

    const unsigned long long total_mma = static_cast<unsigned long long>(options.loop_iters) * static_cast<unsigned long long>(options.unroll);
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        bench_mma_f16_dep_kernel<<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll);
        check_kernel("bench_mma_f16_dep");
        const DeviceResult result = fetch_result(d_result);
        print_result_line("bench_mma_f16_dep", options.dtype, options.mode, options.streams, repeat, options.loop_iters, options.unroll, total_mma, result);
    }

    free_device_result(d_result);
    return 0;
}
