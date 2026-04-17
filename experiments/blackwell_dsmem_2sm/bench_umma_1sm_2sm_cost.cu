#include "common.h"

#include <cuda_bf16.h>

#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace blackwell_dsmem_2sm;

namespace {

struct UmmaOptions {
  const char* mode = "compare";
  int tile_n = 128;
  int depth = 256;
  int iters = 1000;
  int repeats = 20;
  int warmup_repeats = 5;
};

void parse_umma_options(int argc, char** argv, UmmaOptions* options) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    options->mode = parse_str_arg(arg, "--mode=", options->mode);
    options->tile_n = parse_int_arg(arg, "--tile-n=", options->tile_n);
    options->depth = parse_int_arg(arg, "--depth=", options->depth);
    options->iters = parse_int_arg(arg, "--iters=", options->iters);
    options->repeats = parse_int_arg(arg, "--repeats=", options->repeats);
    options->warmup_repeats = parse_int_arg(arg, "--warmup-repeats=", options->warmup_repeats);
  }
}

bool is_valid_umma_tile_n(int tile_n) {
  return tile_n == 64 || tile_n == 128 || tile_n == 256;
}

template <typename T, typename U>
__host__ __device__ constexpr auto align_up_local(T x, U boundary) {
  return (x + boundary - 1) & ~(boundary - 1);
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

template <int CtaGroup>
__device__ __forceinline__ void barrier_sync();

template <>
__device__ __forceinline__ void barrier_sync<1>() {
  __syncthreads();
}

template <>
__device__ __forceinline__ void barrier_sync<2>() {
  asm volatile("barrier.cluster.arrive.release.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");
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

template <typename T>
__device__ __forceinline__ T fill_value(int i) {
  T val;
  uint8_t* p = reinterpret_cast<uint8_t*>(&val);
  #pragma unroll
  for (int b = 0; b < static_cast<int>(sizeof(T)); ++b) {
    p[b] = static_cast<uint8_t>(((i + b) % 127) + 1);
  }
  return val;
}

struct KernelResult {
  unsigned long long cycles;
};

template <int CtaGroup, int M, int N, int K>
__global__ __cluster_dims__(CtaGroup, 1, 1) __launch_bounds__(128)
void umma_cost_kernel(KernelResult* result, int depth, int iters) {
  static_assert(K == 16, "BF16 UMMA benchmark currently assumes K=16.");

  constexpr int MPerCta = M / CtaGroup;
  constexpr int ASize = (MPerCta * K * int(sizeof(nv_bfloat16)));
  constexpr int BSize = (N * K * int(sizeof(nv_bfloat16)));

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;

  int cta_rank = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

  extern __shared__ __align__(128) char smem[];
  auto* A = reinterpret_cast<nv_bfloat16*>(smem);
  auto* B = reinterpret_cast<nv_bfloat16*>(smem + ASize);

  constexpr int ANumel = ASize / int(sizeof(nv_bfloat16));
  constexpr int BNumel = BSize / int(sizeof(nv_bfloat16));

  for (int i = tid; i < ANumel; i += blockDim.x) {
    A[i] = fill_value<nv_bfloat16>(i + cta_rank * ANumel);
  }
  for (int i = tid; i < BNumel; i += blockDim.x) {
    B[i] = fill_value<nv_bfloat16>(i);
  }

  barrier_sync<CtaGroup>();

  __shared__ uint64_t mbar;
  __shared__ int tmem_addr;

  const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(&mbar));
  if (warp_id == 0 && elect_sync()) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }

  const int tmem_cols = align_up_local(N < 32 ? 32 : N, 32);
  if (warp_id == 0) {
    const int tmem_addr_smem = static_cast<int>(__cvta_generic_to_shared(&tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(tmem_addr_smem), "r"(tmem_cols), "n"(CtaGroup));
  }
  barrier_sync<CtaGroup>();

  const uint32_t tmem_d = tmem_addr;
  const uint64_t a_desc = make_smem_desc(A, MPerCta);
  const uint64_t b_desc = make_smem_desc(B, N);
  constexpr uint32_t i_desc = make_i_desc_bf16<M, N>();

  auto mma = [&](int pred) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "tcgen05.mma.cta_group::%4.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}"
        :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc), "n"(CtaGroup), "r"(pred));
  };

  barrier_sync<CtaGroup>();

  uint64_t start_clock = 0;
  uint64_t end_clock = 0;

  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock));

  int phase = 0;
  for (int iter = 0; iter < iters; ++iter) {
    if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
      mma(0);
      for (int m = 1; m < depth; ++m) {
        mma(1);
      }

      const uint16_t cta_mask = static_cast<uint16_t>((1 << CtaGroup) - 1);
      asm volatile(
          "tcgen05.commit.cta_group::%2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 "
          "[%0], %1;"
          :: "r"(mbar_addr), "h"(cta_mask), "n"(CtaGroup)
          : "memory");
    }

    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}"
        :: "r"(mbar_addr), "r"(phase));
    phase ^= 1;
  }

  asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock));
  asm volatile("tcgen05.fence::after_thread_sync;");

  barrier_sync<CtaGroup>();
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(tmem_cols), "n"(CtaGroup));
  }

  if (cta_rank == 0 && warp_id == 0 && elect_sync()) {
    result->cycles = end_clock - start_clock;
  }
}

struct UmmaMetrics {
  double avg_cycles = -1.0;
  double cycles_per_mma = -1.0;
  double cycles_per_flop = -1.0;
  double flops_per_cycle = -1.0;
  double tflops_est = -1.0;
  int mma_m = 0;
  int mma_n = 0;
  int mma_k = 16;
};

template <int CtaGroup, int M, int N, int K>
UmmaMetrics measure_mode(const UmmaOptions& options) {
  UmmaMetrics metrics;
  metrics.mma_m = M;
  metrics.mma_n = N;
  metrics.mma_k = K;

  KernelResult* device_result = alloc_device<KernelResult>(1);
  KernelResult host_result{};

  prepare_cluster_kernel(umma_cost_kernel<CtaGroup, M, N, K>, 0);

  constexpr int MPerCta = M / CtaGroup;
  constexpr int ASize = (MPerCta * K * int(sizeof(nv_bfloat16)));
  constexpr int BSize = (N * K * int(sizeof(nv_bfloat16)));
  const int smem_bytes = ASize + BSize;

  for (int i = 0; i < options.warmup_repeats; ++i) {
    umma_cost_kernel<CtaGroup, M, N, K><<<CtaGroup, 128, smem_bytes>>>(
        device_result, options.depth, options.iters);
    check_cuda(cudaGetLastError(), "umma_cost_kernel warmup launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");
  }

  double total_cycles = 0.0;
  for (int i = 0; i < options.repeats; ++i) {
    umma_cost_kernel<CtaGroup, M, N, K><<<CtaGroup, 128, smem_bytes>>>(
        device_result, options.depth, options.iters);
    check_cuda(cudaGetLastError(), "umma_cost_kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    copy_to_host(&host_result, device_result, 1);
    total_cycles += static_cast<double>(host_result.cycles);
  }

  check_cuda(cudaFree(device_result), "cudaFree");

  const double total_mmas = static_cast<double>(options.depth) * static_cast<double>(options.iters);
  const double flops_per_mma = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
  const double sm_clock_ghz = query_sm_clock_ghz();

  metrics.avg_cycles = total_cycles / static_cast<double>(options.repeats);
  metrics.cycles_per_mma = metrics.avg_cycles / total_mmas;
  metrics.cycles_per_flop = metrics.cycles_per_mma / flops_per_mma;
  metrics.flops_per_cycle = flops_per_mma / metrics.cycles_per_mma;
  metrics.tflops_est = sm_clock_ghz * 1.0e9 * metrics.flops_per_cycle / 1.0e12;
  return metrics;
}

template <int N>
UmmaMetrics dispatch_1sm(const UmmaOptions& options) {
  return measure_mode<1, 128, N, 16>(options);
}

template <int N>
UmmaMetrics dispatch_2sm(const UmmaOptions& options) {
  return measure_mode<2, 256, N, 16>(options);
}

UmmaMetrics dispatch_mode(const char* mode, const UmmaOptions& options) {
  if (std::strcmp(mode, "1sm") == 0) {
    if (options.tile_n == 64) return dispatch_1sm<64>(options);
    if (options.tile_n == 128) return dispatch_1sm<128>(options);
    return dispatch_1sm<256>(options);
  }

  if (options.tile_n == 64) return dispatch_2sm<64>(options);
  if (options.tile_n == 128) return dispatch_2sm<128>(options);
  return dispatch_2sm<256>(options);
}

void print_result_line(const char* mode, const UmmaOptions& options, const UmmaMetrics& metrics) {
  std::printf(
      "RESULT benchmark=bench_umma_1sm_2sm_cost mode=%s mma_m=%d mma_n=%d mma_k=%d depth=%d "
      "iters=%d avg_cycles=%.4f cycles_per_mma=%.6f cycles_per_flop=%.12e flops_per_cycle=%.4f "
      "tflops_est=%.4f\n",
      mode,
      metrics.mma_m,
      metrics.mma_n,
      metrics.mma_k,
      options.depth,
      options.iters,
      metrics.avg_cycles,
      metrics.cycles_per_mma,
      metrics.cycles_per_flop,
      metrics.flops_per_cycle,
      metrics.tflops_est);
}

void print_summary_line(const UmmaOptions& options, const UmmaMetrics& one_sm, const UmmaMetrics& two_sm) {
  const double cycles_per_mma_ratio =
      one_sm.cycles_per_mma > 0.0 ? two_sm.cycles_per_mma / one_sm.cycles_per_mma : 0.0;
  const double cycles_per_flop_ratio =
      one_sm.cycles_per_flop > 0.0 ? two_sm.cycles_per_flop / one_sm.cycles_per_flop : 0.0;
  const double flops_per_cycle_ratio =
      one_sm.flops_per_cycle > 0.0 ? two_sm.flops_per_cycle / one_sm.flops_per_cycle : 0.0;
  const double tflops_ratio = one_sm.tflops_est > 0.0 ? two_sm.tflops_est / one_sm.tflops_est : 0.0;

  std::printf(
      "SUMMARY benchmark=bench_umma_1sm_2sm_cost compare=2sm_vs_1sm tile_n=%d depth=%d iters=%d "
      "cycles_per_mma_ratio=%.6f cycles_per_flop_ratio=%.6f flops_per_cycle_ratio=%.6f "
      "tflops_ratio=%.6f\n",
      options.tile_n,
      options.depth,
      options.iters,
      cycles_per_mma_ratio,
      cycles_per_flop_ratio,
      flops_per_cycle_ratio,
      tflops_ratio);
}

}  // namespace

int main(int argc, char** argv) {
  UmmaOptions options;
  parse_umma_options(argc, argv, &options);

  if (std::strcmp(options.mode, "compare") != 0 &&
      std::strcmp(options.mode, "1sm") != 0 &&
      std::strcmp(options.mode, "2sm") != 0) {
    std::fprintf(stderr, "mode must be compare, 1sm, or 2sm\n");
    return 1;
  }

  if (!is_valid_umma_tile_n(options.tile_n)) {
    std::fprintf(stderr, "Unsupported tile_n=%d\n", options.tile_n);
    return 1;
  }
  if (options.depth <= 0 || options.iters <= 0 || options.repeats <= 0 || options.warmup_repeats < 0) {
    std::fprintf(stderr, "depth/iters/repeats must be positive and warmup_repeats must be non-negative\n");
    return 1;
  }

  std::printf(
      "CONFIG benchmark=bench_umma_1sm_2sm_cost mode=%s tile_n=%d depth=%d iters=%d repeats=%d "
      "warmup_repeats=%d gpu=\"%s\"\n",
      options.mode,
      options.tile_n,
      options.depth,
      options.iters,
      options.repeats,
      options.warmup_repeats,
      gpu_name().c_str());

  if (std::strcmp(options.mode, "1sm") == 0 || std::strcmp(options.mode, "2sm") == 0) {
    UmmaMetrics metrics = dispatch_mode(options.mode, options);
    print_result_line(options.mode, options, metrics);
    return 0;
  }

  UmmaMetrics one_sm = dispatch_mode("1sm", options);
  UmmaMetrics two_sm = dispatch_mode("2sm", options);
  print_result_line("1sm", options, one_sm);
  print_result_line("2sm", options, two_sm);
  print_summary_line(options, one_sm, two_sm);
  return 0;
}
