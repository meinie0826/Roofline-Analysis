#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>

namespace cg = cooperative_groups;

#define CUDA_CHECK(x) do {                                      \
  cudaError_t err__ = (x);                                      \
  if (err__ != cudaSuccess) {                                   \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(err__));     \
    std::exit(1);                                               \
  }                                                             \
} while (0)

// ============================================================================
// Local SMEM read: 所有 thread 同步计时
// ============================================================================
template <int TILE_BYTES, int THREADS>
__global__ void local_smem_read_kernel(
    unsigned long long* __restrict__ cycles_out,
    int repeats)
{
  __shared__ unsigned char smem_raw[TILE_BYTES];
  int4* sB = reinterpret_cast<int4*>(smem_raw);
  int vec_elems = TILE_BYTES / 16;

  // 初始化
  for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
    sB[i] = make_int4(i, i+1, i+2, i+3);
  }
  __syncthreads();

  // === 同步后开始计时 ===
  __syncthreads();
  unsigned long long start = clock64();
  __syncthreads();

  unsigned long long checksum = 0;
  for (int r = 0; r < repeats; ++r) {
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      int4 v = sB[i];
      checksum ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }

  __syncthreads();
  unsigned long long stop = clock64();
  __syncthreads();

  // 防止优化掉
  if (checksum == 0xDEADBEEF) {
    printf("impossible\n");
  }

  if (threadIdx.x == 0) {
    cycles_out[blockIdx.x] = stop - start;
  }
}

// ============================================================================
// Remote DSMEM read: CTA1 读 CTA0 的 smem
// ============================================================================
template <int TILE_BYTES, int THREADS>
__global__ __cluster_dims__(2, 1, 1)
void remote_dsmem_read_kernel(
    unsigned long long* __restrict__ cycles_out,
    int repeats)
{
  extern __shared__ unsigned char smem_raw[];
  int4* sB_local = reinterpret_cast<int4*>(smem_raw);
  int vec_elems = TILE_BYTES / 16;

  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();

  // CTA0: 初始化 smem
  if (rank == 0) {
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      sB_local[i] = make_int4(i, i+1, i+2, i+3);
    }
  }

  cluster.sync();

  // CTA1: 拿到 remote pointer
  const int4* remote_sB0 = cluster.map_shared_rank(sB_local, 0);

  // === CTA1 同步后开始计时 ===
  unsigned long long start = 0, stop = 0;
  unsigned long long checksum = 0;

  if (rank == 1) {
    __syncthreads();
    start = clock64();
    __syncthreads();

    for (int r = 0; r < repeats; ++r) {
      for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
        int4 v = remote_sB0[i];
        checksum ^= v.x ^ v.y ^ v.z ^ v.w;
      }
    }

    __syncthreads();
    stop = clock64();
    __syncthreads();
  }

  // 防止优化掉
  if (checksum == 0xDEADBEEF) {
    printf("impossible\n");
  }

  if (rank == 1 && threadIdx.x == 0) {
    cycles_out[blockIdx.x / 2] = stop - start;
  }

  cluster.sync();
}

// ============================================================================
// GMEM baseline: 从 global memory 加载
// ============================================================================
template <int TILE_BYTES, int THREADS>
__global__ void gmem_read_kernel(
    const int4* __restrict__ gmem_b,
    unsigned long long* __restrict__ cycles_out,
    int repeats)
{
  extern __shared__ unsigned char smem_raw[];
  int4* sB_local = reinterpret_cast<int4*>(smem_raw);
  int vec_elems = TILE_BYTES / 16;

  const int4* gB = gmem_b + blockIdx.x * vec_elems;

  // === 同步后开始计时 ===
  __syncthreads();
  unsigned long long start = clock64();
  __syncthreads();

  unsigned long long checksum = 0;
  for (int r = 0; r < repeats; ++r) {
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      int4 v = gB[i];
      checksum ^= v.x ^ v.y ^ v.z ^ v.w;
      sB_local[i] = v;
    }
  }

  __syncthreads();
  unsigned long long stop = clock64();
  __syncthreads();

  if (checksum == 0xDEADBEEF) {
    printf("impossible\n");
  }

  if (threadIdx.x == 0) {
    cycles_out[blockIdx.x] = stop - start;
  }
}

// ============================================================================
int main(int argc, char** argv) {
  const char* mode = "local";
  int tile_bytes = 16384;
  int repeats = 1000;
  int blocks = 120;
  int threads = 128;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--tile-bytes=") == 0) tile_bytes = std::atoi(argv[i] + 13);
    else if (arg.find("--repeats=") == 0) repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--blocks=") == 0) blocks = std::atoi(argv[i] + 9);
    else if (arg == "--help" || arg == "-h") {
      std::cerr << "Usage: " << argv[0] << " --mode=local|remote|gmem --tile-bytes=N --repeats=N --blocks=N\n";
      return 0;
    }
  }

  // Query SM clock
  int clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  double clock_ghz = clock_khz * 1e-6;

  unsigned long long* d_cycles = nullptr;
  int4* d_gmem = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cycles, blocks * sizeof(unsigned long long)));
  
  if (std::strcmp(mode, "gmem") == 0) {
    size_t total_vecs = size_t(blocks) * (tile_bytes / 16);
    CUDA_CHECK(cudaMalloc(&d_gmem, total_vecs * sizeof(int4)));
  }

  size_t smem_bytes = tile_bytes;

  if (std::strcmp(mode, "local") == 0) {
    local_smem_read_kernel<16384, 128><<<blocks, 128>>>(
        d_cycles, repeats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

  } else if (std::strcmp(mode, "remote") == 0) {
    // remote 模式：每个 cluster 有 2 个 CTA
    dim3 grid(blocks * 2);
    dim3 block(128);
    
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, remote_dsmem_read_kernel<16384, 128>, d_cycles, repeats));
    CUDA_CHECK(cudaDeviceSynchronize());

  } else if (std::strcmp(mode, "gmem") == 0) {
    gmem_read_kernel<16384, 128><<<blocks, 128, smem_bytes>>>(
        d_gmem, d_cycles, repeats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

  } else {
    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
  }

  // 读取结果
  std::vector<unsigned long long> h_cycles(blocks);
  CUDA_CHECK(cudaMemcpy(h_cycles.data(), d_cycles, blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  double total_cycles = 0.0;
  for (int i = 0; i < blocks; ++i) {
    total_cycles += static_cast<double>(h_cycles[i]);
  }
  double avg_cycles = total_cycles / blocks;
  double elapsed_ns = avg_cycles / clock_ghz;

  // 带宽计算
  double bytes_per_block = double(repeats) * double(tile_bytes);
  double bandwidth_gbps = bytes_per_block / (elapsed_ns * 1e-9) / 1e9;

  std::cout << "CONFIG mode=" << mode
            << " tile_bytes=" << tile_bytes
            << " repeats=" << repeats
            << " blocks=" << blocks << "\n";
  std::cout << "RESULT avg_cycles=" << avg_cycles
            << " elapsed_ns=" << elapsed_ns
            << " bandwidth_GB/s=" << bandwidth_gbps << "\n";

  CUDA_CHECK(cudaFree(d_cycles));
  if (d_gmem) CUDA_CHECK(cudaFree(d_gmem));

  return 0;
}
