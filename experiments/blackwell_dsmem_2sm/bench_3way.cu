/**
 * Three-way comparison (all hand-written, no CUTLASS):
 *
 * All kernels: tile 128×64, each pair of CTAs processes the same workload
 *
 * Baseline: Two independent CTAs, each loads B from HBM
 * D1: Cluster, CTA1 copies B from CTA0's DSMEM into local smem, then computes
 * D2: Cluster, CTA1 reads B directly from CTA0's DSMEM during computation (no copy)
 *
 * Key insight:
 *   D1: 1 extra memcpy pass (DSMEM copy to local) + compute with local smem
 *   D2: No copy pass, but compute reads cross-SM (simulates mma.2sm behavior)
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
// Constants: Both CTAs share one 128×64 tile
// CTA0 -> rows 0..63, CTA1 -> rows 64..127
//=============================================================================
constexpr int kCTATileM = 64;
constexpr int kTileN    = 64;
constexpr int kTileK    = 32;
constexpr int kStages   = 2;

//=============================================================================
// Shared memory for one CTA: A tile [kStages][kCTATileM][kTileK] + B tile [kStages][kTileN][kTileK]
//=============================================================================
struct SmemTile {
  half A[kStages][kCTATileM][kTileK];
  half B[kStages][kTileN][kTileK];
};

//=============================================================================
// Baseline: independent CTAs (no cluster)
//=============================================================================
__global__ void __launch_bounds__(128)
baseline_kernel(const half* __restrict__ A,
                const half* __restrict__ B,
                half* __restrict__ C,
                int M, int N, int K)
{
  __shared__ SmemTile sm;

  const int m0 = blockIdx.x * kCTATileM;
  const int n0 = blockIdx.y * kTileN;
  if (m0 >= M || n0 >= N) return;

  const int tid = threadIdx.x;
  const int rbase = (tid / 16) * 4;
  const int cbase = (tid % 16) * 4;
  float acc[16] = {};

  for (int ko = 0; ko < K; ko += kTileK) {
    const int stage = (ko / kTileK) % kStages;
    const int kt    = min(kTileK, K - ko);

    // Load A
    for (int i = tid; i < kCTATileM * kTileK; i += 128) {
      int m = i / kTileK, k = i % kTileK;
      sm.A[stage][m][k] = (m0 + m < M && ko + k < K)
                          ? A[(m0 + m) * K + ko + k]
                          : __float2half(0.f);
    }
    // Load B (independent)
    for (int i = tid; i < kTileN * kTileK; i += 128) {
      int n = i / kTileK, k = i % kTileK;
      sm.B[stage][n][k] = (n0 + n < N && ko + k < K)
                          ? B[(n0 + n) * K + ko + k]
                          : __float2half(0.f);
    }
    __syncthreads();

    // Compute
    for (int k = 0; k < kt; ++k) {
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          acc[r*4+c] += __half2float(sm.A[stage][rbase+r][k])
                      * __half2float(sm.B[stage][cbase+c][k]);
    }
  }

  // Write C
  for (int i = 0; i < 16; ++i) {
    int r = rbase + i/4, c = cbase + i%4;
    if (m0+r < M && n0+c < N)
      C[(m0+r)*N + n0+c] = __float2half(acc[i]);
  }
}

//=============================================================================
// D1: Cluster, CTA1 copies B into local smem first, then computes
//=============================================================================
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)
void d1_kernel(const half* __restrict__ A,
               const half* __restrict__ B,
               half* __restrict__ C,
               int M, int N, int K)
{
  __shared__ SmemTile sm;

  auto cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int m0 = (blockIdx.x * 2 + rank) * kCTATileM;
  const int n0 = blockIdx.y * kTileN;
  if (m0 >= M || n0 >= N) return;

  // Pointer into CTA0's B buffer (flat byte pointer for easier indexing)
  const half* remB = reinterpret_cast<SmemTile*>(
      cluster.map_shared_rank(&sm, 0))->B[0][0];

  const int tid = threadIdx.x;
  const int rbase = (tid / 16) * 4;
  const int cbase = (tid % 16) * 4;
  float acc[16] = {};

  for (int ko = 0; ko < K; ko += kTileK) {
    const int stage = (ko / kTileK) % kStages;
    const int kt    = min(kTileK, K - ko);

    // Load A (each CTA loads its own rows)
    for (int i = tid; i < kCTATileM * kTileK; i += 128) {
      int m = i / kTileK, k = i % kTileK;
      sm.A[stage][m][k] = (m0 + m < M && ko + k < K)
                          ? A[(m0 + m) * K + ko + k]
                          : __float2half(0.f);
    }

    // Only CTA0 loads B from HBM
    if (rank == 0) {
      for (int i = tid; i < kTileN * kTileK; i += 128) {
        int n = i / kTileK, k = i % kTileK;
        sm.B[stage][n][k] = (n0 + n < N && ko + k < K)
                            ? B[(n0 + n) * K + ko + k]
                            : __float2half(0.f);
      }
    }

    cluster.sync(); // wait for CTA0's B to be ready

    // CTA1: copy B from CTA0's DSMEM into local smem (explicit copy)
    if (rank == 1) {
      for (int i = tid; i < kTileN * kTileK; i += 128) {
        sm.B[stage][i / kTileK][i % kTileK] = remB[stage * kTileN * kTileK + i];
      }
    }

    cluster.sync(); // wait for copy to finish

    // Compute from LOCAL smem (same as baseline after copy)
    for (int k = 0; k < kt; ++k) {
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          acc[r*4+c] += __half2float(sm.A[stage][rbase+r][k])
                      * __half2float(sm.B[stage][cbase+c][k]);
    }
  }

  for (int i = 0; i < 16; ++i) {
    int r = rbase + i/4, c = cbase + i%4;
    if (m0+r < M && n0+c < N)
      C[(m0+r)*N + n0+c] = __float2half(acc[i]);
  }
}

//=============================================================================
// D2: Cluster, CTA1 reads B directly from CTA0's DSMEM during compute (no copy)
// This simulates mma.2sm hardware behavior: operand B fetched from peer smem
//=============================================================================
__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)
void d2_kernel(const half* __restrict__ A,
               const half* __restrict__ B,
               half* __restrict__ C,
               int M, int N, int K)
{
  __shared__ SmemTile sm;

  auto cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int m0 = (blockIdx.x * 2 + rank) * kCTATileM;
  const int n0 = blockIdx.y * kTileN;
  if (m0 >= M || n0 >= N) return;

  // Get pointer to CTA0's shared memory, then compute B buffer offset
  // A buffer size = kStages * kCTATileM * kTileK halves
  constexpr int kASize = kStages * kCTATileM * kTileK;
  const half* B0 = reinterpret_cast<const half*>(
      cluster.map_shared_rank(&sm, 0)) + kASize;

  const int tid = threadIdx.x;
  const int rbase = (tid / 16) * 4;
  const int cbase = (tid % 16) * 4;
  float acc[16] = {};

  for (int ko = 0; ko < K; ko += kTileK) {
    const int stage = (ko / kTileK) % kStages;
    const int kt    = min(kTileK, K - ko);

    // Load A (each CTA loads its own rows)
    for (int i = tid; i < kCTATileM * kTileK; i += 128) {
      int m = i / kTileK, k = i % kTileK;
      sm.A[stage][m][k] = (m0 + m < M && ko + k < K)
                          ? A[(m0 + m) * K + ko + k]
                          : __float2half(0.f);
    }

    // Only CTA0 loads B from HBM (no copy step for CTA1)
    if (rank == 0) {
      for (int i = tid; i < kTileN * kTileK; i += 128) {
        int n = i / kTileK, k = i % kTileK;
        sm.B[stage][n][k] = (n0 + n < N && ko + k < K)
                            ? B[(n0 + n) * K + ko + k]
                            : __float2half(0.f);
      }
    }

    cluster.sync(); // wait for CTA0's B

    // Compute: BOTH CTAs read B from CTA0's smem via flat pointer
    // CTA0 reads local, CTA1 reads remote DSMEM
    for (int k = 0; k < kt; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          // B index: [stage][cbase+c][k] flattened
          int b_idx = stage * kTileN * kTileK + (cbase + c) * kTileK + k;
          acc[r*4+c] += __half2float(sm.A[stage][rbase+r][k])
                      * __half2float(B0[b_idx]);
        }
      }
    }
  }

  for (int i = 0; i < 16; ++i) {
    int r = rbase + i/4, c = cbase + i%4;
    if (m0+r < M && n0+c < N)
      C[(m0+r)*N + n0+c] = __float2half(acc[i]);
  }
}

//=============================================================================
// Benchmark helper
//=============================================================================
template<typename LaunchFn>
double measure(LaunchFn launch_fn, int repeats, int warmup) {
  for (int w = 0; w < warmup; ++w) launch_fn();
  cudaDeviceSynchronize();

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  float total = 0.f;
  for (int r = 0; r < repeats; ++r) {
    cudaEventRecord(t0);
    launch_fn();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, t0, t1);
    total += ms;
  }
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return total / repeats;
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "all";
  int M = 2048, N = 2048, K = 8192;
  int repeats = 20, warmup = 5;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if      (arg.find("--mode=")    == 0) mode    = argv[i] + 7;
    else if (arg.find("--m=")       == 0) M       = std::atoi(argv[i] + 4);
    else if (arg.find("--n=")       == 0) N       = std::atoi(argv[i] + 4);
    else if (arg.find("--k=")       == 0) K       = std::atoi(argv[i] + 4);
    else if (arg.find("--repeats=") == 0) repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--warmup=")  == 0) warmup  = std::atoi(argv[i] + 9);
  }

  std::fprintf(stdout,
    "CONFIG mode=%s m=%d n=%d k=%d repeats=%d warmup=%d gpu=\"%s\"\n",
    mode, M, N, K, repeats, warmup, gpu_name().c_str());

  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, (size_t)M * K * sizeof(half));
  cudaMalloc(&d_B, (size_t)K * N * sizeof(half));
  cudaMalloc(&d_C, (size_t)M * N * sizeof(half));
  cudaMemset(d_A, 1, (size_t)M * K * sizeof(half));
  cudaMemset(d_B, 1, (size_t)K * N * sizeof(half));
  cudaMemset(d_C, 0, (size_t)M * N * sizeof(half));

  const double gflops = 2.0 * M * N * K / 1e9;

  // Grid for baseline (one CTA per row-tile)
  dim3 grid_base((M + kCTATileM - 1) / kCTATileM,
                 (N + kTileN    - 1) / kTileN);
  // Grid for cluster kernels (one cluster per two row-tiles)
  dim3 grid_clus((M + 2*kCTATileM - 1) / (2*kCTATileM),
                 (N + kTileN    - 1) / kTileN);
  dim3 block(128);

  // Cluster launch config (no dynamic smem needed - all static)
  cudaLaunchConfig_t cfg{};
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = 0;  // static shared, no dynamic needed
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim = {2, 1, 1};
  cfg.attrs = attr;
  cfg.numAttrs = 1;

  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    auto fn = [&]() {
      baseline_kernel<<<grid_base, block>>>(d_A, d_B, d_C, M, N, K);
    };
    double ms = measure(fn, repeats, warmup);
    std::fprintf(stdout,
      "RESULT mode=baseline elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
  }

  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d1") == 0) {
    cfg.gridDim = grid_clus;
    auto fn = [&]() {
      cudaLaunchKernelEx(&cfg, d1_kernel, d_A, d_B, d_C, M, N, K);
    };
    double ms = measure(fn, repeats, warmup);
    std::fprintf(stdout,
      "RESULT mode=d1 elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
  }

  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d2") == 0) {
    cfg.gridDim = grid_clus;
    auto fn = [&]() {
      cudaLaunchKernelEx(&cfg, d2_kernel, d_A, d_B, d_C, M, N, K);
    };
    double ms = measure(fn, repeats, warmup);
    std::fprintf(stdout,
      "RESULT mode=d2 elapsed_ms=%.6f gflops=%.2f\n", ms, gflops / ms);
  }

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  return 0;
}
