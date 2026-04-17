/**
 * Pure bandwidth comparison: HBM vs DSMEM vs mma.2sm
 * 
 * Measures tile loading bandwidth under three scenarios:
 * - baseline: Each CTA loads tile from HBM (2x traffic)
 * - dsmem: CTA0 loads, CTA1 copies from CTA0's smem
 * - mma2sm: Hardware mma.2sm operand exchange (if measurable)
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

// Use char for byte-level access
using byte_t = char;

//=============================================================================
// Baseline: Each CTA loads tile from HBM independently
//=============================================================================

template <int TileBytes>
__global__ void baseline_load_kernel(
    const byte_t* __restrict__ src,
    byte_t* __restrict__ dst,
    int* __restrict__ cycles_out,
    int iters)
{
  // Each CTA loads TileBytes from HBM
  // Two CTAs -> 2 * TileBytes total traffic
  
  const char* gmem_in = reinterpret_cast<const char*>(src);
  char* gmem_out = reinterpret_cast<char*>(dst);
  
  int tid = threadIdx.x;
  int cta_id = blockIdx.x;
  int offset = cta_id * TileBytes;
  
  // Use ld.global.cg to bypass L1
  unsigned long long start = clock64();
  
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int j = tid; j < TileBytes; j += blockDim.x) {
      unsigned int val;
      asm volatile("ld.global.cg.b8 %0, [%1];" : "=r"(val) : "l"(reinterpret_cast<unsigned long long>(gmem_in + offset + j)));
      gmem_out[offset + j] = static_cast<char>(val);
    }
    __syncthreads();
  }
  
  unsigned long long stop = clock64();
  
  if (tid == 0) {
    atomicAdd(cycles_out, static_cast<int>(stop - start));
  }
}

//=============================================================================
// DSMEM: CTA0 loads, CTA1 copies from CTA0's smem
//=============================================================================

template <int TileBytes, int kStages>
__global__ __cluster_dims__(2, 1, 1)
void dsmem_copy_kernel(
    const byte_t* __restrict__ src,
    byte_t* __restrict__ dst,
    int* __restrict__ cycles_out,
    int iters)
{
  __shared__ char smem[2][TileBytes];  // [stage][byte]
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  // Pointer to CTA0's smem
  char* remote_smem = reinterpret_cast<char*>(cluster.map_shared_rank(smem, 0));
  
  const char* gmem_in = reinterpret_cast<const char*>(src);
  char* gmem_out = reinterpret_cast<char*>(dst);
  
  int tid = threadIdx.x;
  int stage = 0;
  
  unsigned long long start = 0, stop = 0;
  
  for (int i = 0; i < iters; ++i) {
    // === Phase 1: CTA0 loads from HBM ===
    if (rank == 0) {
      for (int j = tid; j < TileBytes; j += blockDim.x) {
        unsigned int val;
        asm volatile("ld.global.cg.b8 %0, [%1];" : "=r"(val) : "l"(reinterpret_cast<unsigned long long>(gmem_in + j)));
        smem[stage][j] = static_cast<char>(val);
      }
    }
    
    cluster.sync();  // Ensure CTA0's load is complete
    
    // === Phase 2: CTA1 copies from CTA0's DSMEM ===
    if (rank == 1) {
      if (i == 0) start = clock64();  // Start timing after warmup
      
      for (int j = tid; j < TileBytes; j += blockDim.x) {
        smem[stage][j] = remote_smem[stage * TileBytes + j];
      }
    }
    
    cluster.sync();
    
    // === Phase 3: Both CTAs write to HBM ===
    for (int j = tid; j < TileBytes; j += blockDim.x) {
      gmem_out[rank * TileBytes + j] = smem[stage][j];
    }
    
    cluster.sync();
    stage = (stage + 1) % kStages;
  }
  
  if (rank == 1 && tid == 0) {
    stop = clock64();
    atomicAdd(cycles_out, static_cast<int>(stop - start));
  }
}

//=============================================================================
// Pure DSMEM bandwidth test (no HBM)
//=============================================================================

template <int TileBytes>
__global__ __cluster_dims__(2, 1, 1)
void pure_dsmem_bandwidth_kernel(
    int* __restrict__ cycles_out,
    int iters)
{
  __shared__ char smem_src[TileBytes];
  __shared__ char smem_dst[TileBytes];
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  // CTA0: producer, CTA1: consumer
  if (rank == 0) {
    // Initialize source
    for (int j = threadIdx.x; j < TileBytes; j += blockDim.x) {
      smem_src[j] = static_cast<char>(j);
    }
  }
  
  cluster.sync();
  
  char* remote_src = reinterpret_cast<char*>(cluster.map_shared_rank(smem_src, 0));
  
  unsigned long long start = 0, stop = 0;
  
  // Measure pure DSMEM copy bandwidth
  if (rank == 1) {
    start = clock64();
    
    for (int i = 0; i < iters; ++i) {
      #pragma unroll 4
      for (int j = threadIdx.x; j < TileBytes; j += blockDim.x) {
        smem_dst[j] = remote_src[j];
      }
      __syncthreads();
    }
    
    stop = clock64();
    
    if (threadIdx.x == 0) {
      atomicAdd(cycles_out, static_cast<int>(stop - start));
    }
  }
}

//=============================================================================
// Launcher
//=============================================================================

struct BenchResult {
  double elapsed_ms;
  double bandwidth_gbps;
  double bytes_transferred;
};

double measure_baseline(const byte_t* d_src, byte_t* d_dst, int tile_bytes, int iters, int warmup) {
  int* d_cycles;
  cudaMalloc(&d_cycles, sizeof(int));
  cudaMemset(d_cycles, 0, sizeof(int));
  
  dim3 grid(2);
  dim3 block(128);
  
  // Warmup
  baseline_load_kernel<16384><<<grid, block>>>(d_src, d_dst, d_cycles, warmup);
  cudaDeviceSynchronize();
  cudaMemset(d_cycles, 0, sizeof(int));
  
  // Measure
  if (tile_bytes == 16384) baseline_load_kernel<16384><<<grid, block>>>(d_src, d_dst, d_cycles, iters);
  else if (tile_bytes == 32768) baseline_load_kernel<32768><<<grid, block>>>(d_src, d_dst, d_cycles, iters);
  else if (tile_bytes == 65536) baseline_load_kernel<65536><<<grid, block>>>(d_src, d_dst, d_cycles, iters);
  else if (tile_bytes == 131072) baseline_load_kernel<131072><<<grid, block>>>(d_src, d_dst, d_cycles, iters);
  cudaDeviceSynchronize();
  
  int h_cycles;
  cudaMemcpy(&h_cycles, d_cycles, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_cycles);
  
  int clock_khz;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
  double clock_ghz = clock_khz * 1e-6;
  
  return double(h_cycles) / clock_ghz / 1e6;  // ms
}

double measure_dsmem(const byte_t* d_src, byte_t* d_dst, int tile_bytes, int iters, int warmup) {
  int* d_cycles;
  cudaMalloc(&d_cycles, sizeof(int));
  cudaMemset(d_cycles, 0, sizeof(int));
  
  dim3 grid(1);
  dim3 block(128);
  
  size_t smem_bytes = 2 * tile_bytes;
  
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
  if (tile_bytes == 16384) cudaLaunchKernelEx(&config, dsmem_copy_kernel<16384, 2>, d_src, d_dst, d_cycles, warmup);
  else if (tile_bytes == 32768) cudaLaunchKernelEx(&config, dsmem_copy_kernel<32768, 2>, d_src, d_dst, d_cycles, warmup);
  else if (tile_bytes == 65536) cudaLaunchKernelEx(&config, dsmem_copy_kernel<65536, 2>, d_src, d_dst, d_cycles, warmup);
  else if (tile_bytes == 131072) cudaLaunchKernelEx(&config, dsmem_copy_kernel<131072, 2>, d_src, d_dst, d_cycles, warmup);
  cudaDeviceSynchronize();
  cudaMemset(d_cycles, 0, sizeof(int));
  
  // Measure
  if (tile_bytes == 16384) cudaLaunchKernelEx(&config, dsmem_copy_kernel<16384, 2>, d_src, d_dst, d_cycles, iters);
  else if (tile_bytes == 32768) cudaLaunchKernelEx(&config, dsmem_copy_kernel<32768, 2>, d_src, d_dst, d_cycles, iters);
  else if (tile_bytes == 65536) cudaLaunchKernelEx(&config, dsmem_copy_kernel<65536, 2>, d_src, d_dst, d_cycles, iters);
  else if (tile_bytes == 131072) cudaLaunchKernelEx(&config, dsmem_copy_kernel<131072, 2>, d_src, d_dst, d_cycles, iters);
  cudaDeviceSynchronize();
  
  int h_cycles;
  cudaMemcpy(&h_cycles, d_cycles, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_cycles);
  
  int clock_khz;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
  double clock_ghz = clock_khz * 1e-6;
  
  return double(h_cycles) / clock_ghz / 1e6;  // ms
}

double measure_pure_dsmem(int tile_bytes, int iters) {
  int* d_cycles;
  cudaMalloc(&d_cycles, sizeof(int));
  cudaMemset(d_cycles, 0, sizeof(int));
  
  dim3 grid(1);
  dim3 block(128);
  
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 2 * tile_bytes;
  
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  config.attrs = attrs;
  config.numAttrs = 1;
  
  if (tile_bytes == 16384) cudaLaunchKernelEx(&config, pure_dsmem_bandwidth_kernel<16384>, d_cycles, iters);
  else if (tile_bytes == 32768) cudaLaunchKernelEx(&config, pure_dsmem_bandwidth_kernel<32768>, d_cycles, iters);
  else if (tile_bytes == 65536) cudaLaunchKernelEx(&config, pure_dsmem_bandwidth_kernel<65536>, d_cycles, iters);
  else if (tile_bytes == 131072) cudaLaunchKernelEx(&config, pure_dsmem_bandwidth_kernel<131072>, d_cycles, iters);
  cudaDeviceSynchronize();
  
  int h_cycles;
  cudaMemcpy(&h_cycles, d_cycles, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_cycles);
  
  int clock_khz;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
  double clock_ghz = clock_khz * 1e-6;
  
  return double(h_cycles) / clock_ghz / 1e6;  // ms
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "all";
  int tile_bytes = 16384;
  int iters = 1000;
  int warmup = 100;
  
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--tile-bytes=") == 0) tile_bytes = std::atoi(argv[i] + 13);
    else if (arg.find("--iters=") == 0) iters = std::atoi(argv[i] + 8);
    else if (arg.find("--warmup=") == 0) warmup = std::atoi(argv[i] + 9);
  }
  
  std::fprintf(stdout, "CONFIG mode=%s tile_bytes=%d iters=%d warmup=%d gpu=\"%s\"\n",
               mode, tile_bytes, iters, warmup, gpu_name().c_str());
  
  // Allocate
  byte_t *d_src, *d_dst;
  cudaMalloc(&d_src, 2 * tile_bytes);
  cudaMalloc(&d_dst, 2 * tile_bytes);
  cudaMemset(d_src, 1, 2 * tile_bytes);
  cudaMemset(d_dst, 0, 2 * tile_bytes);
  
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    double ms = measure_baseline(d_src, d_dst, tile_bytes, iters, warmup);
    double bytes = 2.0 * tile_bytes * iters;  // Two CTAs, each loads tile_bytes
    double bw = bytes / ms / 1e6;  // GB/s
    std::fprintf(stdout, "RESULT mode=baseline elapsed_ms=%.6f bytes=%.0f bandwidth_gbps=%.2f\n", 
                 ms, bytes, bw);
  }
  
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "dsmem") == 0) {
    double ms = measure_dsmem(d_src, d_dst, tile_bytes, iters, warmup);
    double bytes = 1.0 * tile_bytes * iters;  // Only CTA0 loads from HBM, CTA1 copies from DSMEM
    double bw = bytes / ms / 1e6;  // GB/s
    std::fprintf(stdout, "RESULT mode=dsmem elapsed_ms=%.6f bytes=%.0f bandwidth_gbps=%.2f\n", 
                 ms, bytes, bw);
  }
  
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "pure_dsmem") == 0) {
    double ms = measure_pure_dsmem(tile_bytes, iters);
    double bytes = 1.0 * tile_bytes * iters;  // CTA1 copies from CTA0's smem
    double bw = bytes / ms / 1e6;  // GB/s
    std::fprintf(stdout, "RESULT mode=pure_dsmem elapsed_ms=%.6f bytes=%.0f bandwidth_gbps=%.2f\n", 
                 ms, bytes, bw);
  }
  
  cudaFree(d_src);
  cudaFree(d_dst);
  
  return 0;
}
