/**
 * Simple bandwidth test: HBM vs DSMEM
 */

#include "common.h"
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

//=============================================================================
// Baseline: Each CTA loads from HBM independently
//=============================================================================

__global__ void baseline_load_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int tile_bytes,
    int iters)
{
  int cta_id = blockIdx.x;
  const char* gmem_in = src + cta_id * tile_bytes;
  char* gmem_out = dst + cta_id * tile_bytes;
  
  for (int i = 0; i < iters; ++i) {
    for (int j = threadIdx.x; j < tile_bytes; j += blockDim.x) {
      gmem_out[j] = gmem_in[j];
    }
    __syncthreads();
  }
}

//=============================================================================
// DSMEM: CTA0 loads, CTA1 copies from CTA0's smem
//=============================================================================

__global__ __cluster_dims__(2, 1, 1)
void dsmem_copy_kernel(
    const char* __restrict__ src,
    char* __restrict__ dst,
    int tile_bytes,
    int iters)
{
  extern __shared__ char smem[];
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  const char* gmem_in = src + rank * tile_bytes;
  char* gmem_out = dst + rank * tile_bytes;
  
  char* remote_smem = reinterpret_cast<char*>(cluster.map_shared_rank(smem, 0));
  
  for (int i = 0; i < iters; ++i) {
    // Phase 1: Both CTAs load their portion from HBM (or CTA0 loads, CTA1 zeros)
    if (rank == 0) {
      for (int j = threadIdx.x; j < tile_bytes; j += blockDim.x) {
        smem[j] = gmem_in[j];
      }
    } else {
      for (int j = threadIdx.x; j < tile_bytes; j += blockDim.x) {
        smem[j] = 0;  // Will be overwritten
      }
    }
    
    cluster.sync();
    
    // Phase 2: CTA1 copies from CTA0's DSMEM
    if (rank == 1) {
      for (int j = threadIdx.x; j < tile_bytes; j += blockDim.x) {
        smem[j] = remote_smem[j];
      }
    }
    
    cluster.sync();
    
    // Phase 3: Both CTAs write to HBM
    for (int j = threadIdx.x; j < tile_bytes; j += blockDim.x) {
      gmem_out[j] = smem[j];
    }
    
    cluster.sync();
  }
}

//=============================================================================
// Pure DSMEM copy (no HBM)
//=============================================================================

__global__ __cluster_dims__(2, 1, 1)
void pure_dsmem_kernel(
    int tile_bytes,
    int iters)
{
  extern __shared__ char smem[];
  char* smem_dst = smem;
  char* smem_src = smem + tile_bytes;
  
  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  
  // CTA0's source buffer, CTA1's destination buffer
  char* remote_src = reinterpret_cast<char*>(cluster.map_shared_rank(smem_src, 0));
  
  // Initialize source
  if (rank == 0) {
    for (int j = threadIdx.x; j < tile_bytes; j += blockDim.x) {
      smem_src[j] = static_cast<char>(j);
    }
  }
  
  cluster.sync();
  
  // Measure pure DSMEM copy
  for (int i = 0; i < iters; ++i) {
    if (rank == 1) {
      for (int j = threadIdx.x; j < tile_bytes; j += blockDim.x) {
        smem_dst[j] = remote_src[j];
      }
    }
    cluster.sync();
  }
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "all";
  int tile_bytes = 16384;
  int iters = 100;
  
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--tile-bytes=") == 0) tile_bytes = std::atoi(argv[i] + 13);
    else if (arg.find("--iters=") == 0) iters = std::atoi(argv[i] + 8);
  }
  
  std::fprintf(stdout, "CONFIG mode=%s tile_bytes=%d iters=%d gpu=\"%s\"\n",
               mode, tile_bytes, iters, gpu_name().c_str());
  
  // Allocate
  char *d_src, *d_dst;
  cudaMalloc(&d_src, 2 * tile_bytes);
  cudaMalloc(&d_dst, 2 * tile_bytes);
  cudaMemset(d_src, 0xAB, 2 * tile_bytes);
  cudaMemset(d_dst, 0, 2 * tile_bytes);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // Baseline
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "baseline") == 0) {
    cudaEventRecord(start);
    baseline_load_kernel<<<2, 128>>>(d_src, d_dst, tile_bytes, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double bytes = 2.0 * tile_bytes * iters;
    double bw = bytes / ms / 1e6;
    std::fprintf(stdout, "RESULT mode=baseline elapsed_ms=%.6f bytes=%.0f bandwidth_gbps=%.2f\n", 
                 ms, bytes, bw);
  }
  
  // DSMEM copy
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "dsmem") == 0) {
    dim3 grid(1);
    dim3 block(128);
    
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = tile_bytes;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;
    
    cudaEventRecord(start);
    cudaLaunchKernelEx(&config, dsmem_copy_kernel, d_src, d_dst, tile_bytes, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    // HBM bytes: only CTA0 loads from HBM (CTA1 copies from DSMEM)
    double bytes = 1.0 * tile_bytes * iters;
    double bw = bytes / ms / 1e6;
    std::fprintf(stdout, "RESULT mode=dsmem elapsed_ms=%.6f bytes=%.0f bandwidth_gbps=%.2f\n", 
                 ms, bytes, bw);
  }
  
  // Pure DSMEM
  if (std::strcmp(mode, "all") == 0 || std::strcmp(mode, "pure_dsmem") == 0) {
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
    
    cudaEventRecord(start);
    cudaLaunchKernelEx(&config, pure_dsmem_kernel, tile_bytes, iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double bytes = 1.0 * tile_bytes * iters;
    double bw = bytes / ms / 1e6;
    std::fprintf(stdout, "RESULT mode=pure_dsmem elapsed_ms=%.6f bytes=%.0f bandwidth_gbps=%.2f\n", 
                 ms, bytes, bw);
  }
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_src);
  cudaFree(d_dst);
  
  return 0;
}
