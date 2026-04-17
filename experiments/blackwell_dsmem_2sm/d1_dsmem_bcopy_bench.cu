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
// Kernel: CTA1 从 CTA0 的 DSMEM 拷贝 B tile 到本地 SMEM
// 只同步一次，测量纯传输带宽
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

  const int4* gB = gmem_b + pair_id * vec_elems;

  // CTA0: gmem -> local smem (只做一次)
  if (rank == 0) {
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      sB_local[i] = gB[i];
    }
  }

  cluster.sync();

  // CTA1: 拿到 remote pointer
  int4* remote_sB0 = cluster.map_shared_rank(sB_local, 0);

  // 预热：先做一次拷贝填满本地 smem
  if (rank == 1) {
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      sB_local[i] = remote_sB0[i];
    }
  }
  cluster.sync();

  // === 测量：多次 remote read，只有 CTA1 在跑 ===
  unsigned long long start = 0, stop = 0;
  
  if (rank == 1 && threadIdx.x == 0) {
    start = clock64();
  }

  if (rank == 1) {
    for (int r = 0; r < repeats; ++r) {
      for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
        int4 v = remote_sB0[i];  // remote DSMEM read
        sB_local[i] = v;         // write to local smem
      }
    }
  }

  if (rank == 1 && threadIdx.x == 0) {
    stop = clock64();
    // 写回结果供 host 读取
    sink[pair_id * 2] = static_cast<float>(stop - start);
    sink[pair_id * 2 + 1] = static_cast<float>(repeats * vec_elems * sizeof(int4));
  }

  // 防止优化掉
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
    sink[pair_id] += red[0];
  }

  cluster.sync();
}

// ============================================================================
// Baseline: 单 CTA 从 gmem 加载 B (真实 HBM 带宽)
// ============================================================================
template <int TILE_BYTES, int THREADS>
__global__ void baseline_gmem_bcopy_kernel(
    const int4* __restrict__ gmem_b,
    float* __restrict__ sink,
    int repeats)
{
  extern __shared__ __align__(128) unsigned char smem_raw[];
  int4* sB_local = reinterpret_cast<int4*>(smem_raw);

  int pair_id = blockIdx.x;
  int vec_elems = TILE_BYTES / sizeof(int4);
  const int4* gB = gmem_b + pair_id * vec_elems;

  unsigned long long start = 0, stop = 0;
  
  if (threadIdx.x == 0) {
    start = clock64();
  }

  for (int r = 0; r < repeats; ++r) {
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      sB_local[i] = gB[i];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    stop = clock64();
    sink[pair_id * 2] = static_cast<float>(stop - start);
    sink[pair_id * 2 + 1] = static_cast<float>(repeats * vec_elems * sizeof(int4));
  }

  // 防止优化掉
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
    sink[pair_id] += red[0];
  }
}

// ============================================================================
// Local SMEM baseline: 单 CTA 读自己的 smem
// ============================================================================
template <int TILE_BYTES, int THREADS>
__global__ void local_smem_read_kernel(
    float* __restrict__ sink,
    int repeats)
{
  extern __shared__ __align__(128) unsigned char smem_raw[];
  int4* sB = reinterpret_cast<int4*>(smem_raw);
  int vec_elems = TILE_BYTES / sizeof(int4);

  // 初始化 smem
  for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
    sB[i] = make_int4(threadIdx.x, threadIdx.x+1, threadIdx.x+2, threadIdx.x+3);
  }
  __syncthreads();

  unsigned long long start = 0, stop = 0;
  unsigned long long checksum = 0;

  if (threadIdx.x == 0) {
    start = clock64();
  }

  for (int r = 0; r < repeats; ++r) {
    unsigned long long local_sum = 0;
    for (int i = threadIdx.x; i < vec_elems; i += THREADS) {
      int4 v = sB[i];
      local_sum ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    checksum ^= local_sum;
  }

  if (threadIdx.x == 0) {
    stop = clock64();
    sink[blockIdx.x * 3] = static_cast<float>(stop - start);
    sink[blockIdx.x * 3 + 1] = static_cast<float>(repeats * vec_elems * sizeof(int4));
    sink[blockIdx.x * 3 + 2] = static_cast<float>(checksum);
  }
}

// ============================================================================
int main(int argc, char** argv) {
  const char* mode = "remote";
  int tile_bytes = 16384;
  int repeats = 1000;
  int pairs = 120;
  int threads = 128;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--tile-bytes=") == 0) tile_bytes = std::atoi(argv[i] + 13);
    else if (arg.find("--repeats=") == 0) repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--pairs=") == 0) pairs = std::atoi(argv[i] + 8);
    else if (arg == "--help" || arg == "-h") {
      std::cerr << "Usage: " << argv[0] << " --mode=remote|baseline|local --tile-bytes=N --repeats=N --pairs=N\n";
      return 0;
    }
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
  CUDA_CHECK(cudaMalloc(&dSink, pairs * 3 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), total_vecs * sizeof(int4), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dSink, 0, pairs * 3 * sizeof(float)));

  size_t smem_bytes = tile_bytes;

  // Query SM clock for cycle->ns conversion
  int clock_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  double clock_ghz = clock_khz * 1e-6;

  if (std::strcmp(mode, "remote") == 0) {
    dim3 grid(pairs * 2);
    dim3 block(threads);

    CUDA_CHECK(cudaFuncSetAttribute(
        reinterpret_cast<const void*>(d1_remote_bcopy_kernel<16384, 128>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        int(smem_bytes)));

    d1_remote_bcopy_kernel<16384, 128><<<grid, block, smem_bytes>>>(dB, dSink, repeats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

  } else if (std::strcmp(mode, "baseline") == 0) {
    dim3 grid(pairs);
    dim3 block(threads);

    baseline_gmem_bcopy_kernel<16384, 128><<<grid, block, smem_bytes>>>(dB, dSink, repeats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

  } else if (std::strcmp(mode, "local") == 0) {
    dim3 grid(pairs);
    dim3 block(threads);

    local_smem_read_kernel<16384, 128><<<grid, block, smem_bytes>>>(dSink, repeats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

  } else {
    std::cerr << "Unknown mode: " << mode << "\n";
    return 1;
  }

  // 读取结果，计算平均
  std::vector<float> hSink(pairs * 3);
  CUDA_CHECK(cudaMemcpy(hSink.data(), dSink, pairs * 3 * sizeof(float), cudaMemcpyDeviceToHost));

  double total_cycles = 0.0;
  double total_bytes = 0.0;
  for (int i = 0; i < pairs; ++i) {
    total_cycles += hSink[i * 3];
    total_bytes += hSink[i * 3 + 1];
  }
  double avg_cycles = total_cycles / pairs;
  double elapsed_ns = avg_cycles / clock_ghz;
  double total_bytes_all = double(pairs) * double(repeats) * double(tile_bytes);
  double gbps = total_bytes_all / (elapsed_ns * 1e-9 * pairs) / 1e9;

  std::cout << "CONFIG mode=" << mode
            << " tile_bytes=" << tile_bytes
            << " repeats=" << repeats
            << " pairs=" << pairs << "\n";
  std::cout << "RESULT avg_cycles=" << avg_cycles
            << " elapsed_ns=" << elapsed_ns
            << " bytes_per_pair=" << (repeats * tile_bytes)
            << " bandwidth_GB/s=" << gbps << "\n";

  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dSink));
  return 0;
}
