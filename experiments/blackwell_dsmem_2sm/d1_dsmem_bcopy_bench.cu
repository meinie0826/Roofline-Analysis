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

template <typename T>
__device__ __forceinline__ T ld_gmem(const T* p) { return *p; }

template <>
__device__ __forceinline__ int4 ld_gmem<int4>(const int4* p) { return *p; }

template <typename T>
__device__ __forceinline__ void st_smem(T* p, T v) { *p = v; }

template <>
__device__ __forceinline__ void st_smem<int4>(int4* p, int4 v) { *p = v; }

// ============================================================================
// Kernel: CTA1 从 CTA0 的 DSMEM 拷贝 B tile 到本地 SMEM
// ============================================================================
template <int TILE_BYTES, int THREADS>
__global__ __cluster_dims__(2, 1, 1)
void d1_remote_bcopy_kernel(
    const int4* __restrict__ gmem_b,
    float* __restrict__ sink,
    int repeats)
{
  extern __shared__ __align__(128) unsigned char smem_raw[];
  int4* sB_local = reinterpret_cast<int4*>(smem_raw);

  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();
  int pair_id = blockIdx.x / 2;
  int vec_elems = TILE_BYTES / sizeof(int4);

  // 每个 pair 对应一块独立 B tile
  const int4* gB = gmem_b + pair_id * vec_elems;

  // CTA0: gmem -> local smem
  if (rank == 0) {
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      sB_local[i] = ld_gmem(gB + i);
    }
  }

  cluster.sync();

  // CTA1: remote DSMEM -> local smem
  int4* remote_sB0 = cluster.map_shared_rank(sB_local, 0);

  for (int r = 0; r < repeats; ++r) {
    if (rank == 1) {
      for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
        int4 v = remote_sB0[i];
        st_smem(sB_local + i, v);
      }
    }
    cluster.sync();
  }

  // 防止编译器优化掉拷贝
  float acc = 0.0f;
  if (rank == 1) {
    int words = TILE_BYTES / sizeof(float);
    float* fptr = reinterpret_cast<float*>(sB_local);
    for (int i = threadIdx.x; i < words; i += THREADS) {
      acc += fptr[i];
    }
  }

  __shared__ float red[THREADS];
  red[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
    __syncthreads();
  }

  if (rank == 1 && threadIdx.x == 0) {
    sink[pair_id] = red[0];
  }

  cluster.sync();
}

// ============================================================================
// Baseline: 两个 CTA 各自从 gmem 加载 B，无共享
// ============================================================================
template <int TILE_BYTES, int THREADS>
__global__ void baseline_dupB_kernel(
    const int4* __restrict__ gmem_b,
    float* __restrict__ sink,
    int repeats)
{
  extern __shared__ __align__(128) unsigned char smem_raw[];
  int4* sB_local = reinterpret_cast<int4*>(smem_raw);

  int pair_id = blockIdx.x;
  int vec_elems = TILE_BYTES / sizeof(int4);
  const int4* gB = gmem_b + pair_id * vec_elems;

  for (int r = 0; r < repeats; ++r) {
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      sB_local[i] = ld_gmem(gB + i);
    }
    __syncthreads();
  }

  float acc = 0.0f;
  int words = TILE_BYTES / sizeof(float);
  float* fptr = reinterpret_cast<float*>(sB_local);
  for (int i = threadIdx.x; i < words; i += THREADS) {
    acc += fptr[i];
  }

  __shared__ float red[THREADS];
  red[threadIdx.x] = acc;
  __syncthreads();

  for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    sink[pair_id] = red[0];
  }
}

// ============================================================================
// Host launcher
// ============================================================================
void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  --mode=remote|baseline  (default: remote)\n"
            << "  --tile-bytes=N          (default: 16384)\n"
            << "  --repeats=N             (default: 1000)\n"
            << "  --pairs=N               (default: 120)\n"
            << "  --threads=N             (default: 128)\n";
}

int main(int argc, char** argv) {
  const char* mode = "remote";
  int tile_bytes = 16384;  // 128*64*2 = 16KB
  int repeats = 1000;
  int pairs = 120;
  int threads = 128;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--tile-bytes=") == 0) tile_bytes = std::atoi(argv[i] + 13);
    else if (arg.find("--repeats=") == 0) repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--pairs=") == 0) pairs = std::atoi(argv[i] + 8);
    else if (arg.find("--threads=") == 0) threads = std::atoi(argv[i] + 10);
    else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    }
  }

  if (tile_bytes % 16 != 0) {
    std::cerr << "tile_bytes must be multiple of 16\n";
    return 1;
  }

  size_t vec_elems = tile_bytes / 16;
  size_t total_vecs = size_t(pairs) * vec_elems;

  std::vector<int4> hB(total_vecs);
  for (size_t i = 0; i < total_vecs; ++i) {
    hB[i] = make_int4(int(i), int(i+1), int(i+2), int(i+3));
  }

  int4* dB = nullptr;
  float* dSink = nullptr;
  CUDA_CHECK(cudaMalloc(&dB, total_vecs * sizeof(int4)));
  CUDA_CHECK(cudaMalloc(&dSink, pairs * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), total_vecs * sizeof(int4), cudaMemcpyHostToDevice));

  size_t smem_bytes = tile_bytes;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  if (std::strcmp(mode, "remote") == 0) {
    // D1: remote DSMEM copy
    dim3 grid(pairs * 2);
    dim3 block(threads);

    CUDA_CHECK(cudaFuncSetAttribute(
        d1_remote_bcopy_kernel<16384, 128>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes)));

    CUDA_CHECK(cudaEventRecord(start));
    d1_remote_bcopy_kernel<16384, 128><<<grid, block, smem_bytes>>>(dB, dSink, repeats);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

  } else {
    // Baseline: duplicate B load
    dim3 grid(pairs);
    dim3 block(threads);

    CUDA_CHECK(cudaFuncSetAttribute(
        baseline_dupB_kernel<16384, 128>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes)));

    CUDA_CHECK(cudaEventRecord(start));
    baseline_dupB_kernel<16384, 128><<<grid, block, smem_bytes>>>(dB, dSink, repeats);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
  }

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  // remote 模式：每个 pair 只传输一次 B tile（从 CTA0 到 CTA1）
  // baseline 模式：每个 CTA 都从 gmem 加载，总加载量是 2x
  double total_bytes;
  if (std::strcmp(mode, "remote") == 0) {
    total_bytes = double(pairs) * double(tile_bytes) * double(repeats);
  } else {
    total_bytes = double(pairs) * double(tile_bytes) * double(repeats) * 2.0;
  }
  double gbps = total_bytes / (ms * 1.0e6);

  std::cout << "CONFIG mode=" << mode
            << " tile_bytes=" << tile_bytes
            << " repeats=" << repeats
            << " pairs=" << pairs
            << " threads=" << threads << "\n";
  std::cout << "RESULT elapsed_ms=" << ms
            << " total_bytes=" << total_bytes
            << " aggregate_GB/s=" << gbps << "\n";

  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dSink));
  return 0;
}
