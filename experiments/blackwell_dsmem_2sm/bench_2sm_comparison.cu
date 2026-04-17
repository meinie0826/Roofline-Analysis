#include "common.h"
#include <cooperative_groups.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

namespace cg = cooperative_groups;
using namespace cute;
using namespace blackwell_dsmem_2sm;

//=============================================================================
// D1: Software DSMEM Sharing Kernel
// Two CTAs in one cluster, CTA1 copies B from CTA0's smem
//=============================================================================

template <typename ElementA, typename ElementB, typename ElementC,
          typename ElementAcc, int kTileM, int kTileN, int kTileK>
struct D1SharedBParams {
  ElementA* A;
  ElementB* B;
  ElementC* C;
  ElementC* D;
  int M, N, K;
  int lda, ldb, ldc;
};

template <int kTileM, int kTileN, int kTileK, int kStages>
struct D1SharedBSmem {
  alignas(128) half_t A[kStages][kTileM][kTileK];
  alignas(128) half_t B[kStages][kTileN][kTileK];
};

template <typename Params, int kTileM, int kTileN, int kTileK, int kStages>
__global__ __cluster_dims__(2, 1, 1)
void d1_software_share_b_kernel(
    Params params,
    int* barrier,
    unsigned long long* timer_out)  // 返回 CTA1 的计时
{
  using Smem = D1SharedBSmem<kTileM, kTileN, kTileK, kStages>;
  extern __shared__ char smem_raw[];
  Smem* smem = reinterpret_cast<Smem*>(smem_raw);

  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  int pair_id = blockIdx.x / 2;  // Which pair this cluster belongs to

  // Global tile coordinates
  int m_tile = pair_id;          // Each pair handles one M tile (2*kTileM)
  int n_tile = blockIdx.y;
  int m_offset = m_tile * (2 * kTileM) + rank * kTileM;

  // Pointers
  half_t* gA = params.A + m_offset * params.lda;
  half_t* gB = params.B + n_tile * kTileN * params.ldb;
  half_t* gC = params.C + m_offset * params.ldc + n_tile * kTileN;
  half_t* gD = params.D + m_offset * params.ldc + n_tile * kTileN;

  // Remote smem pointer (CTA1 sees CTA0's smem)
  Smem* remote_smem0 = cluster.map_shared_rank(smem, 0);

  // Timer for CTA1
  unsigned long long start = 0, stop = 0;

  // === K loop ===
  for (int k_offset = 0; k_offset < params.K; k_offset += kTileK) {
    int k_stage = (k_offset / kTileK) % kStages;

    // --- Phase 1: Load A (each CTA loads its own) ---
    if (rank == 0) {
      // CTA0 loads A[0:kTileM, k:k+kTileK]
      for (int m = threadIdx.x; m < kTileM * kTileK; m += blockDim.x) {
        int mi = m / kTileK;
        int ki = m % kTileK;
        smem->A[k_stage][mi][ki] = gA[mi * params.lda + (k_offset + ki)];
      }
    } else {
      // CTA1 loads A[kTileM:2*kTileM, k:k+kTileK]
      for (int m = threadIdx.x; m < kTileM * kTileK; m += blockDim.x) {
        int mi = m / kTileK;
        int ki = m % kTileK;
        smem->A[k_stage][mi][ki] = gA[mi * params.lda + (k_offset + ki)];
      }
    }

    // --- Phase 2: Load B (CTA0 only from gmem) ---
    if (rank == 0) {
      for (int n = threadIdx.x; n < kTileN * kTileK; n += blockDim.x) {
        int ni = n / kTileK;
        int ki = n % kTileK;
        smem->B[k_stage][ni][ki] = gB[ni * params.ldb + (k_offset + ki)];
      }
    }

    cluster.sync();  // Ensure CTA0's B is ready

    // --- Phase 3: CTA1 copies B from CTA0's DSMEM ===
    if (rank == 1) {
      if (k_offset == 0) start = clock64();  // 开始计时（只计一次）

      for (int n = threadIdx.x; n < kTileN * kTileK; n += blockDim.x) {
        int ni = n / kTileK;
        int ki = n % kTileK;
        smem->B[k_stage][ni][ki] = remote_smem0->B[k_stage][ni][ki];
      }
    }

    cluster.sync();

    // --- Phase 4: Both CTAs run MMA (simplified: just load and compute checksum) ---
    // 这里简化：不实际跑 MMA，只做 smem read 来防止被优化掉
    float sum = 0.0f;
    for (int m = threadIdx.x; m < kTileM * kTileN; m += blockDim.x) {
      int mi = m / kTileN;
      int ni = m % kTileN;
      half_t a = smem->A[k_stage][mi][0];
      half_t b = smem->B[k_stage][ni][0];
      sum += static_cast<float>(a) * static_cast<float>(b);
    }
    __syncwarp();

    __syncthreads();
  }

  // Stop timer for CTA1
  if (rank == 1) {
    stop = clock64();
    if (threadIdx.x == 0) {
      *timer_out = stop - start;
    }
  }

  // Prevent optimization
  if (params.D[0] > 0.0f) {
    printf("");
  }

  cluster.sync();
}

//=============================================================================
// Benchmark launcher
//=============================================================================

struct BenchResult {
  double elapsed_ms;
  double gflops;
  double bytes_b_share;  // B tile 总字节数（用于计算 DSMEM 带宽）
};

struct BenchConfig {
  int M = 4096;
  int N = 64;
  int K = 4096;
  int tile_m = 128;
  int tile_n = 64;
  int tile_k = 64;
  int stages = 2;
  int repeats = 10;
  int warmup = 3;
  const char* mode = "d1";
};

void print_usage(const char* prog) {
  std::fprintf(stderr,
    "Usage: %s [options]\n"
    "Options:\n"
    "  --mode=baseline|d1|d2    (default: d1)\n"
    "  --m=N --n=N --k=N        GEMM dimensions\n"
    "  --tile-m=N --tile-n=N    Tile sizes\n"
    "  --stages=N               Pipeline stages\n"
    "  --repeats=N              Measurement repeats\n"
    "  --warmup=N               Warmup iterations\n",
    prog);
}

BenchResult run_baseline(const BenchConfig& cfg) {
  // 使用 CUTLASS 1SM kernel
  // Baseline: 两个独立 CTA 各自加载 B（实际上就是一个 1SM kernel 跑，测量两倍时间）
  using Runner = CutlassRunner<64, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
                               cutlass::epilogue::TmaWarpSpecialized1Sm,
                               cutlass::gemm::KernelTmaWarpSpecialized1SmSm100::StageCount>;
  
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = 148;  // B300

  Runner runner;
  runner.initialize(cfg.M, cfg.N, cfg.K);

  GemmOptions opts;
  opts.m = cfg.M;
  opts.n = cfg.N;
  opts.k = cfg.K;
  opts.repeats = 1;
  opts.warmup_repeats = 0;

  // Warmup
  for (int i = 0; i < cfg.warmup; ++i) {
    runner.setup(opts, hw_info);
    runner.run_kernel();
  }

  // Measure
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total_ms = 0.0;
  for (int r = 0; r < cfg.repeats; ++r) {
    runner.setup(opts, hw_info);
    cudaEventRecord(start);
    runner.run_kernel();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  double avg_ms = total_ms / cfg.repeats;
  double gflops = 2.0 * cfg.M * cfg.N * cfg.K / avg_ms / 1.0e6;

  // Baseline 的 B 字节传输量是 2x（两个 CTA 各加载一次）
  double bytes_b = 2.0 * cfg.tile_n * cfg.tile_k * sizeof(half_t);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return {avg_ms, gflops, bytes_b};
}

BenchResult run_d1(const BenchConfig& cfg) {
  // D1: Software DSMEM sharing
  using Params = D1SharedBParams<half_t, half_t, half_t, float, 128, 64, 64>;

  Params params;
  params.M = cfg.M;
  params.N = cfg.N;
  params.K = cfg.K;
  params.lda = cfg.K;
  params.ldb = cfg.K;
  params.ldc = cfg.N;

  // Allocate
  size_t A_size = cfg.M * cfg.K * sizeof(half_t);
  size_t B_size = cfg.K * cfg.N * sizeof(half_t);
  size_t C_size = cfg.M * cfg.N * sizeof(half_t);

  cudaMalloc(&params.A, A_size);
  cudaMalloc(&params.B, B_size);
  cudaMalloc(&params.C, C_size);
  cudaMalloc(&params.D, C_size);

  // Init
  cudaMemset(params.A, 1, A_size);
  cudaMemset(params.B, 1, B_size);
  cudaMemset(params.C, 0, C_size);

  // Timer buffer
  unsigned long long* d_timer;
  cudaMalloc(&d_timer, sizeof(unsigned long long));

  int* d_barrier;
  cudaMalloc(&d_barrier, sizeof(int));
  cudaMemset(d_barrier, 0, sizeof(int));

  // Grid
  int pairs = (cfg.M + 2 * cfg.tile_m - 1) / (2 * cfg.tile_m);
  dim3 grid(pairs * 2, (cfg.N + cfg.tile_n - 1) / cfg.tile_n);
  dim3 block(128);

  size_t smem_bytes = sizeof(D1SharedBSmem<128, 64, 64, 2>);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_bytes;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  config.attrs = attrs;
  config.numAttrs = 1;

  // Warmup
  for (int i = 0; i < cfg.warmup; ++i) {
    cudaLaunchKernelEx(&config, d1_software_share_b_kernel<Params, 128, 64, 64, 2>,
                       params, d_barrier, d_timer);
    cudaDeviceSynchronize();
  }

  // Measure
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total_ms = 0.0;
  unsigned long long total_cycles = 0;

  for (int r = 0; r < cfg.repeats; ++r) {
    cudaEventRecord(start);
    cudaLaunchKernelEx(&config, d1_software_share_b_kernel<Params, 128, 64, 64, 2>,
                       params, d_barrier, d_timer);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;

    unsigned long long cycles;
    cudaMemcpy(&cycles, d_timer, sizeof(cycles), cudaMemcpyDeviceToHost);
    total_cycles += cycles;
  }

  // Query clock
  int clock_khz;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
  double clock_ghz = clock_khz * 1e-6;

  double avg_cycles = double(total_cycles) / cfg.repeats;
  double elapsed_ns = avg_cycles / clock_ghz;
  double avg_ms = total_ms / cfg.repeats;

  double gflops = 2.0 * cfg.M * cfg.N * cfg.K / avg_ms / 1.0e6;

  // D1 的 B 字节传输量是 1x（只有 CTA0 加载）
  double bytes_b = 1.0 * cfg.tile_n * cfg.tile_k * sizeof(half_t);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(params.A);
  cudaFree(params.B);
  cudaFree(params.C);
  cudaFree(params.D);
  cudaFree(d_timer);
  cudaFree(d_barrier);

  return {avg_ms, gflops, bytes_b};
}

BenchResult run_d2(const BenchConfig& cfg) {
  // D2: CUTLASS mma.2sm
  using Runner = CutlassRunner<64, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                               cutlass::epilogue::TmaWarpSpecialized2Sm,
                               cutlass::gemm::KernelTmaWarpSpecialized2SmSm100::StageCount>;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = 148;

  Runner runner;
  runner.initialize(cfg.M, cfg.N, cfg.K);

  GemmOptions opts;
  opts.m = cfg.M;
  opts.n = cfg.N;
  opts.k = cfg.K;
  opts.repeats = 1;
  opts.warmup_repeats = 0;

  for (int i = 0; i < cfg.warmup; ++i) {
    runner.setup(opts, hw_info);
    runner.run_kernel();
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total_ms = 0.0;
  for (int r = 0; r < cfg.repeats; ++r) {
    runner.setup(opts, hw_info);
    cudaEventRecord(start);
    runner.run_kernel();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }

  double avg_ms = total_ms / cfg.repeats;
  double gflops = 2.0 * cfg.M * cfg.N * cfg.K / avg_ms / 1.0e6;

  // D2 的 B 字节传输量是 0.5x（硬件共享，实际上加载一次）
  double bytes_b = 0.5 * cfg.tile_n * cfg.tile_k * sizeof(half_t);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return {avg_ms, gflops, bytes_b};
}

//=============================================================================
int main(int argc, char** argv) {
  BenchConfig cfg;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) cfg.mode = argv[i] + 7;
    else if (arg.find("--m=") == 0) cfg.M = std::atoi(argv[i] + 4);
    else if (arg.find("--n=") == 0) cfg.N = std::atoi(argv[i] + 4);
    else if (arg.find("--k=") == 0) cfg.K = std::atoi(argv[i] + 4);
    else if (arg.find("--tile-m=") == 0) cfg.tile_m = std::atoi(argv[i] + 9);
    else if (arg.find("--tile-n=") == 0) cfg.tile_n = std::atoi(argv[i] + 9);
    else if (arg.find("--stages=") == 0) cfg.stages = std::atoi(argv[i] + 9);
    else if (arg.find("--repeats=") == 0) cfg.repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--warmup=") == 0) cfg.warmup = std::atoi(argv[i] + 9);
    else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  std::fprintf(stdout, "CONFIG mode=%s m=%d n=%d k=%d tile_m=%d tile_n=%d stages=%d repeats=%d warmup=%d gpu=\"%s\"\n",
               cfg.mode, cfg.M, cfg.N, cfg.K, cfg.tile_m, cfg.tile_n,
               cfg.stages, cfg.repeats, cfg.warmup, gpu_name().c_str());

  BenchResult result;
  if (std::strcmp(cfg.mode, "baseline") == 0) {
    result = run_baseline(cfg);
  } else if (std::strcmp(cfg.mode, "d1") == 0) {
    result = run_d1(cfg);
  } else if (std::strcmp(cfg.mode, "d2") == 0) {
    result = run_d2(cfg);
  } else {
    std::fprintf(stderr, "Unknown mode: %s\n", cfg.mode);
    return 1;
  }

  std::fprintf(stdout, "RESULT elapsed_ms=%.6f gflops=%.2f bytes_b_share=%.0f\n",
               result.elapsed_ms, result.gflops, result.bytes_b_share);

  return 0;
}
