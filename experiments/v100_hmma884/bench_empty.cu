#include "common.h"

namespace {

using namespace v100_hmma884;

__global__ void bench_empty_kernel(DeviceResult* out, int warmup_iters, int loop_iters, int unroll) {
    if (threadIdx.x >= kThreadsPerBlock) return;

    unsigned x0 = static_cast<unsigned>(threadIdx.x);
    unsigned x1 = x0 ^ 0x9e3779b9u;
    unsigned x2 = x0 + 0x7f4a7c15u;
    unsigned x3 = x0 ^ 0x85ebca6bu;

    #pragma unroll 1
    for (int i = 0; i < warmup_iters; ++i) {
        x0 = x0 * 1664525u + 1013904223u;
        x1 = x1 * 22695477u + 1u;
        x2 ^= x0 + x1;
        x3 += x2 ^ 0x27d4eb2du;
    }

    __syncthreads();
    const unsigned long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < loop_iters; ++i) {
        if (unroll >= 1) {
            x0 = x0 * 1664525u + 1013904223u;
            x1 = x1 * 22695477u + 1u;
            x2 ^= x0 + x1;
            x3 += x2 ^ 0x27d4eb2du;
        }
        if (unroll >= 2) {
            x0 = x0 * 1664525u + 1013904223u;
            x1 = x1 * 22695477u + 1u;
            x2 ^= x0 + x1;
            x3 += x2 ^ 0x27d4eb2du;
        }
        if (unroll >= 4) {
            x0 = x0 * 1664525u + 1013904223u;
            x1 = x1 * 22695477u + 1u;
            x2 ^= x0 + x1;
            x3 += x2 ^ 0x27d4eb2du;

            x0 = x0 * 1664525u + 1013904223u;
            x1 = x1 * 22695477u + 1u;
            x2 ^= x0 + x1;
            x3 += x2 ^ 0x27d4eb2du;
        }
        if (unroll >= 8) {
            for (int j = 0; j < 4; ++j) {
                x0 = x0 * 1664525u + 1013904223u;
                x1 = x1 * 22695477u + 1u;
                x2 ^= x0 + x1;
                x3 += x2 ^ 0x27d4eb2du;
            }
        }
        if (unroll >= 16) {
            for (int j = 0; j < 8; ++j) {
                x0 = x0 * 1664525u + 1013904223u;
                x1 = x1 * 22695477u + 1u;
                x2 ^= x0 + x1;
                x3 += x2 ^ 0x27d4eb2du;
            }
        }
    }

    const unsigned long long stop = clock64();

    if (threadIdx.x == 0) {
        out->cycles = stop - start;
        out->sink_lo = (static_cast<unsigned long long>(x1) << 32) | x0;
        out->sink_hi = (static_cast<unsigned long long>(x3) << 32) | x2;
    }
}

}  // namespace

int main(int argc, char** argv) {
    using namespace v100_hmma884;

    BenchOptions options;
    options.dtype = "none";
    options.mode = "empty";
    options.streams = 0;
    parse_common_args(argc, argv, &options);

    if (!valid_unroll(options.unroll)) {
        std::fprintf(stderr, "Unsupported unroll=%d\n", options.unroll);
        return 1;
    }

    DeviceResult* d_result = alloc_device_result();
    print_options("bench_empty", options);

    for (int i = 0; i < options.warmup_launches; ++i) {
        bench_empty_kernel<<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll);
        check_kernel("bench_empty warmup");
    }

    const unsigned long long total_mma = 0;
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        bench_empty_kernel<<<kBlocks, kThreadsPerBlock>>>(d_result, options.warmup_iters, options.loop_iters, options.unroll);
        check_kernel("bench_empty");
        const DeviceResult result = fetch_result(d_result);
        print_result_line("bench_empty", options.dtype, options.mode, options.streams, repeat, options.loop_iters, options.unroll, total_mma, result);
    }

    free_device_result(d_result);
    return 0;
}
