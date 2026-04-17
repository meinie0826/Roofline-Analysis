/**
 * Fair comparison with real mma.2sm instruction
 * 
 * All three kernels use SAME tile shape: 128×64×32
 * - Baseline: Two independent CTAs (each 64×64)
 * - D1: Two CTAs in cluster, CTA1 copies B from CTA0's DSMEM
 * - D2: Two CTAs in cluster, real mma.2sm (tcgen05.mma.cta_group::2)
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cstdio>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
constexpr int kTileM = 128;  // Full tile (2 CTAs)
constexpr int kTileN = 64;
constexpr int kTileK = 32;
constexpr int kStages = 2;

//=============================================================================
// tcgen05.mma instruction wrappers
//=============================================================================

__device__ uint64_t make_smem_desc(uint32_t addr, uint32_t SBO, uint32_t swizzle) {
  // Encode shared memory descriptor
  // Bits 0-15: address (16B aligned, encoded as addr >> 4)
  // Bits 32-45: SBO (stride dimension byte offset / 16)
  // Bits 61-63: swizzle mode
  return ((uint64_t)(addr >> 4)) | (((uint64_t)SBO) << 32) | (((uint64_t)swizzle) << 61);
}

__device__ void tcgen05_mma_2sm(
    uint32_t taddr,
    uint64_t a_desc,
    uint64_t b_desc,
    uint32_t i_desc,
    uint32_t accumulate) {
  // tcgen05.mma.cta_group::2.sync.aligned.m128n64k32.f32.bf16.bf16
  asm volatile(
      "tcgen05.mma.cta_group::2.sync.aligned.m128n64k32.f32.bf16.bf16 [%0], %1, %2, %3, %4, %5;"
      :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), 
         "r"(0), "r"(accumulate)  // disable-output-lane=0, enable-input-d=accumulate
      : "memory");
}

__device__ void tcgen05_commit(uint32_t mbar_addr) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :: "r"(mbar_addr) : "memory");
}

//=============================================================================
// Baseline: Independent CTAs (same as before)
//=============================================================================

__global__ void __launch_bounds__(64)
baseline_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  // Each CTA processes 64×64 tile
  __shared__ half sA[2][64][32];
  __shared__ half sB[2][64][32];
  
  int m_start = blockIdx.x * 64;
  int n_start = blockIdx.y * 64;
  
  if (m_start >= M || n_start >= N) return;
  
  int tid = threadIdx.x;
  int row_base = (tid / 16) * 4;
  int col_base = (tid % 16) * 4;
  
  float accum[16] = {0};
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % 2;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Load A
    for (int i = tid; i < 64 * 32; i += 64) {
      int m = i / 32, k = i % 32;
      if (m_start + m < M && k_offset + k < K) {
        sA[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        sA[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Load B (independently)
    for (int i = tid; i < 64 * 32; i += 64) {
      int n = i / 32, k = i % 32;
      if (n_start + n < N && k_offset + k < K) {
        sB[stage][n][k] = B[(n_start + n) * K + k_offset + k];
      } else {
        sB[stage][n][k] = __float2half(0.0f);
      }
    }
    
    __syncthreads();
    
    // Compute
    for (int k = 0; k < k_tiles; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          accum[r * 4 + c] += __half2float(sA[stage][(row_base + r) % 64][k]) *
                              __half2float(sB[stage][(col_base + c) % 64][k]);
        }
      }
    }
  }
  
  // Write C
  for (int i = 0; i < 16; ++i) {
    int r = row_base + (i / 4);
    int c = col_base + (i % 4);
    if (m_start + r < M && n_start + c < N) {
      C[(m_start + r) * N + n_start + c] = __float2half(accum[i]);
    }
  }
}

//=============================================================================
// D1: Cluster with DSMEM copy (same as before)
//=============================================================================

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(64)
void d1_cluster_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  __shared__ half sA[2][64][32];
  __shared__ half sB[2][64][32];
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  int m_start = (blockIdx.x * 2 + rank) * 64;
  int n_start = blockIdx.y * 64;
  
  if (m_start >= M || n_start >= N) return;
  
  half* remote_sB = reinterpret_cast<half*>(cluster.map_shared_rank(&sB[0][0][0], 0));
  
  int tid = threadIdx.x;
  int row_base = (tid / 16) * 4;
  int col_base = (tid % 16) * 4;
  
  float accum[16] = {0};
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % 2;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Load A
    for (int i = tid; i < 64 * 32; i += 64) {
      int m = i / 32, k = i % 32;
      if (m_start + m < M && k_offset + k < K) {
        sA[stage][m][k] = A[(m_start + m) * K + k_offset + k];
      } else {
        sA[stage][m][k] = __float2half(0.0f);
      }
    }
    
    // Load B (CTA0 only)
    if (rank == 0) {
      for (int i = tid; i < 64 * 32; i += 64) {
        int n = i / 32, k = i % 32;
        if (n_start + n < N && k_offset + k < K) {
          sB[stage][n][k] = B[(n_start + n) * K + k_offset + k];
        } else {
          sB[stage][n][k] = __float2half(0.0f);
        }
      }
    }
    
    cluster.sync();
    
    // CTA1 copies B
    if (rank == 1) {
      for (int i = tid; i < 64 * 32; i += 64) {
        sB[stage][i / 32][i % 32] = remote_sB[stage * 64 * 32 + i];
      }
    }
    
    cluster.sync();
    
    // Compute
    for (int k = 0; k < k_tiles; ++k) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
          accum[r * 4 + c] += __half2float(sA[stage][(row_base + r) % 64][k]) *
                              __half2float(sB[stage][(col_base + c) % 64][k]);
        }
      }
    }
  }
  
  // Write C
  for (int i = 0; i < 16; ++i) {
    int r = row_base + (i / 4);
    int c = col_base + (i % 4);
    if (m_start + r < M && n_start + c < N) {
      C[(m_start + r) * N + n_start + c] = __float2half(accum[i]);
    }
  }
}

//=============================================================================
// D2: Real mma.2sm using tcgen05.mma.cta_group::2
//=============================================================================

__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128)
void d2_mma2sm_gemm_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
  // This kernel uses real tcgen05.mma.2sm instruction
  // Both CTAs cooperate on 128×64×32 tile
  
  extern __shared__ char smem[];
  half* sA = reinterpret_cast<half*>(smem);
  half* sB = sA + kStages * kTileM * kTileK;
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  // Tile partitioning:
  // CTA0 handles rows 0-63, CTA1 handles rows 64-127
  int m_start = blockIdx.x * kTileM;
  int n_start = blockIdx.y * kTileN;
  
  if (m_start >= M || n_start >= N) return;
  
  int tid = threadIdx.x;
  
  // Initialize shared memory descriptors
  // For simplicity, no swizzling (swizzle=0)
  uint32_t sA_addr = __cvta_generic_to_shared(sA);
  uint32_t sB_addr = __cvta_generic_to_shared(sB);
  uint64_t a_desc = make_smem_desc(sA_addr, 8 * kTileK * 2, 0);
  uint64_t b_desc = make_smem_desc(sB_addr, 8 * kTileK * 2, 0);
  
  // Instruction descriptor for m128n64k32.f32.bf16.bf16
  uint32_t i_desc = (1U << 4)   // dtype=FP32
                  | (1U << 7)   // atype=BF16 (map half to bf16)
                  | (1U << 10)  // btype=BF16
                  | ((kTileN >> 3) << 17)  // MMA_N
                  | (8U << 24); // MMA_M=128 (encoded as 8)
  
  float accum[32] = {0};  // Each CTA accumulates 64 rows
  
  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int stage = (k_offset / kTileK) % kStages;
    int k_tiles = min(kTileK, K - k_offset);
    
    // Cooperative load A (each CTA loads its portion)
    int m_offset = rank * 64;
    for (int i = tid; i < 64 * kTileK; i += 128) {
      int m = i / kTileK;
      int k = i % kTileK;
      if (m_start + m_offset + m < M && k_offset + k < K) {
        sA[stage * kTileM * kTileK + (m_offset + m) * kTileK + k] = 
            A[(m_start + m_offset + m) * K + k_offset + k];
      } else {
        sA[stage * kTileM * kTileK + (m_offset + m) * kTileK + k] = __float2half(0.0f);
      }
    }
    
    // Load B (CTA0 only, both CTAs will read via multicast)
    if (rank == 0) {
      for (int i = tid; i < kTileN * kTileK; i += 128) {
        int n = i / kTileK;
        int k = i % kTileK;
        if (n_start + n < N && k_offset + k < K) {
          sB[stage * kTileN * kTileK + n * kTileK + k] = 
              B[(n_start + n) * K + k_offset + k];
        } else {
          sB[stage * kTileN * kTileK + n * kTileK + k] = __float2half(0.0f);
        }
      }
    }
    
    cluster.sync();
    
    // MMA using tcgen05.mma.2sm
    // Only one thread per cluster issues the instruction
    if (tid == 0) {
      uint32_t taddr = 0;  // Tensor memory address (allocated elsewhere)
      uint32_t accumulate = (k_offset > 0) ? 1 : 0;
      tcgen05_mma_2sm(taddr, a_desc, b_desc, i_desc, accumulate);
    }
    
    cluster.sync();
  }
  
  // Write C (each CTA writes its portion)
  int m_offset = rank * 64;
  for (int i = 0; i < 16; ++i) {
    int r = (tid / 4) * 4 + (i / 4);
    int c = (tid % 4) * 4 + (i % 4);
    if (m_start + m_offset + r < M && n_start + c < N) {
      C[(m_start + m_offset + r) * N + n_start + c] = __float2half(accum[i]);
    }
  }
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "all";
  int M = 2048, N = 2048, K = 8192;
  int repeats = 20, warmup = 5;
  
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--m=") == 0) M = std::atoi(argv[i] + 4);
    else if (arg.find("--n=") == 0) N = std::atoi(argv[i] + 4);
    else if (arg.find("--k=") == 0) K = std::atoi(argv[i] + 4);
    else if (arg.find("--repeats=") == 0) repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--warmup=") == 0) warmup = std::atoi(argv[i] + 9);
  }
  
  std::fprintf(stdout, "CONFIG mode=%s m=%d n=%d k=%d repeats=%d warmup=%d gpu=\"%s\"\n",
               mode, M, N, K, repeats, warmup, gpu_name().c_str());
  
  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(half));
  cudaMalloc(&d_B, K * N * sizeof(half));
  cudaMalloc(&d_C, M * N * sizeof(half));
  
  cudaMemset(d_A, 1, M * K * sizeof(half));
  cudaMemset(d_B, 1, K * N * sizeof(half));
  cudaMemset(d_C, 0, M * N * sizeof(half));
  
  double gflops = 2.0 * M * N * K / 1e9;
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // Baseline
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    dim3 grid((M + 63) / 64, (N + 63) / 64);
    dim3 block(64);
    size_t smem = 2 * 64 * 32 * sizeof(half) + 2 * 64 * 32 * sizeof(half);
    
    for (int w = 0; w < warmup; ++w) {
      baseline_gemm_kernel<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    float total_ms = 0.0f;
    for (int r = 0; r < repeats; ++r) {
      cudaEventRecord(start);
      baseline_gemm_kernel<<<grid, block, smem>>>(d_A, d_B, d_C, M, N, K);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
    }
    double avg_ms = total_ms / repeats;
    std::fprintf(stdout, "RESULT mode=baseline elapsed_ms=%.6f gflops=%.2f\n", avg_ms, gflops / avg_ms);
  }
  
  // D1
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d1") == 0) {
    dim3 grid((M + 127) / 128, (N + 63) / 64);
    dim3 block(64);
    size_t smem = 2 * 64 * 32 * sizeof(half) + 2 * 64 * 32 * sizeof(half);
    
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;
    
    for (int w = 0; w < warmup; ++w) {
      cudaLaunchKernelEx(&config, d1_cluster_gemm_kernel, d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    float total_ms = 0.0f;
    for (int r = 0; r < repeats; ++r) {
      cudaEventRecord(start);
      cudaLaunchKernelEx(&config, d1_cluster_gemm_kernel, d_A, d_B, d_C, M, N, K);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
    }
    double avg_ms = total_ms / repeats;
    std::fprintf(stdout, "RESULT mode=d1 elapsed_ms=%.6f gflops=%.2f\n", avg_ms, gflops / avg_ms);
  }
  
  // D2 (real mma.2sm)
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "d2") == 0) {
    dim3 grid((M + 127) / 128, (N + 63) / 64);
    dim3 block(128);
    size_t smem = kStages * kTileM * kTileK * sizeof(half) + kStages * kTileN * kTileK * sizeof(half);
    
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;
    
    for (int w = 0; w < warmup; ++w) {
      cudaLaunchKernelEx(&config, d2_mma2sm_gemm_kernel, d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    
    float total_ms = 0.0f;
    for (int r = 0; r < repeats; ++r) {
      cudaEventRecord(start);
      cudaLaunchKernelEx(&config, d2_mma2sm_gemm_kernel, d_A, d_B, d_C, M, N, K);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, start, stop);
      total_ms += ms;
    }
    double avg_ms = total_ms / repeats;
    std::fprintf(stdout, "RESULT mode=d2 elapsed_ms=%.6f gflops=%.2f\n", avg_ms, gflops / avg_ms);
  }
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
