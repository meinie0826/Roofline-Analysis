#include "common.h"
#include <cuda_fp16.h>

namespace {

using namespace v100_hmma884;

struct F32Acc {
    float x[8];
};

__device__ __forceinline__ void mma_f32_dep_once(
    unsigned a0,
    unsigned a1,
    unsigned b0,
    unsigned b1,
    F32Acc& acc) {
    float d0, d1, d2, d3, d4, d5, d6, d7;
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11}, "
        "{%12, %13, %14, %15, %16, %17, %18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1),
          "f"(acc.x[0]), "f"(acc.x[1]), "f"(acc.x[2]), "f"(acc.x[3]),
          "f"(acc.x[4]), "f"(acc.x[5]), "f"(acc.x[6]), "f"(acc.x[7]));
    acc.x[0] = d0;
    acc.x[1] = d1;
    acc.x[2] = d2;
    acc.x[3] = d3;
    acc.x[4] = d4;
    acc.x[5] = d5;
    acc.x[6] = d6;
    acc.x[7] = d7;
}

__global__ void bench_mma_f32_dep_kernel(DeviceResult* out, int warmup_iters, int loop_iters, int unroll) {
    if (threadIdx.x >= kThreadsPerBlock) return;

    unsigned a0 = 0x3c003c00u;
    unsigned a1 = 0x3c003c00u;
    unsigned b0 = 0x3c003c00u;
    unsigned b1 = 0x3c003c00u;
    F32Acc acc = {};

    #pragma unroll 1
    for (int i = 0; i < warmup_iters; ++i) {
        mma_f32_dep_once(a0, a1, b0, b1, acc);
    }

    __syncthreads();
    const unsigned long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < loop_iters; ++i) {
        if (unroll >= 1) mma_f32_dep_once(a0, a1, b0, b1, acc);
        if (unroll >= 2) mma_f32_dep_once(a0, a1, b0, b1, acc);
        if (unroll >= 4) {
            mma_f32_dep_once(a0, a1, b0, b1, acc);
            mma_f32_dep_once(a0, a1, b0, b1, acc);
        }
        if (unroll >= 8) {
            for (int j = 0; j < 4; ++j) mma_f32_dep_once(a0, a1, b0, b1, acc);
        }
        if (unroll >= 16) {
            for (int j = 0; j < 8; ++j) mma_f32_dep_once(a0, a1, b0, b1, acc);
        }
    }

    const unsigned long long stop = clock64();

    if (threadIdx.x == 0) {
        unsigned long long lo =
            (static_cast<unsigned long long>(__float_as_uint(acc.x[1])) << 32) |
            __float_as_uint(acc.x[0]);
        unsigned long long hi =
            (static_cast<unsigned long long>(__float_as_uint(acc.x[3])) << 32) |
            __float_as_uint(acc.x[2]);
        lo ^= (static_cast<unsigned long long>(__float_as_uint(acc.x[5])) << 32) |
              __float_as_uint(acc.x[4]);
        hi ^= (static_cast<unsigned long long>(__float_as_uint(acc.x[7])) << 32) |
              __float_as_uint(acc.x[6]);
        out->cycles = stop - start;
        out->sink_lo = lo;
        out->sink_hi = hi;
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace v100_hmma884;

    BenchOptions options;
    options.dtype = "f32_f16_f16_f32";
    options.mode = "dep";
    options.streams = 1;
    parse_common_args(argc, argv, &options);

    if (!valid_unroll(options.unroll)) {
        std::fprintf(stderr, "Unsupported unroll=%d\n", options.unroll);
        return 1;
    }

    DeviceResult* d_result = alloc_device_result();
    print_options("bench_mma_f32_dep", options);

    for (int i = 0; i < options.warmup_launches; ++i) {
        bench_mma_f32_dep_kernel<<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll);
        check_kernel("bench_mma_f32_dep warmup");
    }

    const unsigned long long total_mma =
        static_cast<unsigned long long>(options.loop_iters) *
        static_cast<unsigned long long>(options.unroll);
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        bench_mma_f32_dep_kernel<<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll);
        check_kernel("bench_mma_f32_dep");
        const DeviceResult result = fetch_result(d_result);
        print_result_line("bench_mma_f32_dep", options.dtype, options.mode, options.streams, repeat, options.loop_iters, options.unroll, total_mma, result);
    }

    free_device_result(d_result);
    return 0;
}
