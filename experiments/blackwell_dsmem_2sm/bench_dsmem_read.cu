#include "common.h"

#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

template <int VecBytes>
struct VecType;

template <>
struct VecType<4> { using Type = uint32_t; };

template <>
struct VecType<8> { using Type = uint2; };

template <>
struct VecType<16> { using Type = uint4; };

template <int VecBytes>
__device__ __forceinline__ unsigned long long checksum_vec(typename VecType<VecBytes>::Type v);

template <>
__device__ __forceinline__ unsigned long long checksum_vec<4>(uint32_t v) {
  return static_cast<unsigned long long>(v);
}

template <>
__device__ __forceinline__ unsigned long long checksum_vec<8>(uint2 v) {
  return static_cast<unsigned long long>(v.x) ^ (static_cast<unsigned long long>(v.y) << 32);
}

template <>
__device__ __forceinline__ unsigned long long checksum_vec<16>(uint4 v) {
  return static_cast<unsigned long long>(v.x) ^
         (static_cast<unsigned long long>(v.y) << 11) ^
         (static_cast<unsigned long long>(v.z) << 23) ^
         (static_cast<unsigned long long>(v.w) << 37);
}

__device__ __forceinline__ uintptr_t align_up_ptr(uintptr_t ptr, int align_bytes) {
  return (ptr + static_cast<uintptr_t>(align_bytes - 1)) & ~static_cast<uintptr_t>(align_bytes - 1);
}

template <bool Remote, int VecBytes>
__global__ void dsmem_read_kernel(StreamResult* out, int iters, int buffer_bytes, int align_bytes) {
  extern __shared__ __align__(16) unsigned char smem_raw[];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const bool timed_block = (rank == 1);

  uintptr_t base_ptr = reinterpret_cast<uintptr_t>(smem_raw);
  uintptr_t aligned_ptr = align_up_ptr(base_ptr, align_bytes > VecBytes ? align_bytes : VecBytes);
  unsigned char* local_ptr = reinterpret_cast<unsigned char*>(aligned_ptr);

  for (int i = threadIdx.x; i < buffer_bytes; i += blockDim.x) {
    local_ptr[i] = static_cast<unsigned char>((rank * 17 + i) & 0xff);
  }
  cluster.sync();

  unsigned char* src_ptr = local_ptr;
  if constexpr (Remote) {
    src_ptr = reinterpret_cast<unsigned char*>(cluster.map_shared_rank(local_ptr, 0));
  }

  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  const Vec* vec_ptr = reinterpret_cast<const Vec*>(src_ptr);

  unsigned long long start = 0;
  unsigned long long stop = 0;
  unsigned long long checksum = 0;

  if (threadIdx.x == 0 && timed_block) {
    start = clock64();
  }
  __syncthreads();

  for (int iter = 0; iter < iters; ++iter) {
    unsigned long long local_checksum = 0;
    for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
      Vec value = vec_ptr[idx];
      local_checksum ^= checksum_vec<VecBytes>(value) + static_cast<unsigned long long>(iter * 131 + idx);
    }
    checksum ^= local_checksum;
  }

  __syncthreads();
  if (threadIdx.x == 0 && timed_block) {
    stop = clock64();
    out->cycles = stop - start;
    out->checksum = checksum;
    out->bytes = static_cast<unsigned long long>(iters) * static_cast<unsigned long long>(vec_count) * VecBytes;
  }
}

template <bool Remote>
void run_for_vec_bytes(StreamResult* d_result, const StreamOptions& options) {
  dim3 block(128, 1, 1);
  dim3 grid(options.cluster_dim_x, 1, 1);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = options.buffer_bytes + options.align_bytes + 32;
  config.stream = nullptr;

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = options.cluster_dim_x;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  if (options.vec_bytes == 4) {
    void* args[] = {&d_result, const_cast<int*>(&options.iters), const_cast<int*>(&options.buffer_bytes), const_cast<int*>(&options.align_bytes)};
    check_cuda(cudaLaunchKernelEx(&config, dsmem_read_kernel<Remote, 4>, args), "cudaLaunchKernelEx read 4");
  } else if (options.vec_bytes == 8) {
    void* args[] = {&d_result, const_cast<int*>(&options.iters), const_cast<int*>(&options.buffer_bytes), const_cast<int*>(&options.align_bytes)};
    check_cuda(cudaLaunchKernelEx(&config, dsmem_read_kernel<Remote, 8>, args), "cudaLaunchKernelEx read 8");
  } else {
    void* args[] = {&d_result, const_cast<int*>(&options.iters), const_cast<int*>(&options.buffer_bytes), const_cast<int*>(&options.align_bytes)};
    check_cuda(cudaLaunchKernelEx(&config, dsmem_read_kernel<Remote, 16>, args), "cudaLaunchKernelEx read 16");
  }
}

int main(int argc, char** argv) {
  StreamOptions options;
  parse_stream_options(argc, argv, &options);

  if (!is_valid_vec_bytes(options.vec_bytes) || !is_valid_align_bytes(options.align_bytes)) {
    std::fprintf(stderr, "Unsupported vec/alignment combination vec_bytes=%d align_bytes=%d\n", options.vec_bytes, options.align_bytes);
    return 1;
  }
  if (options.cluster_dim_x != 2) {
    std::fprintf(stderr, "This benchmark currently expects --cluster-dim-x=2\n");
    return 1;
  }
  if (std::strcmp(options.mode, "local") != 0 && std::strcmp(options.mode, "remote") != 0) {
    std::fprintf(stderr, "mode must be local or remote\n");
    return 1;
  }

  print_stream_header("bench_dsmem_read", options);
  const double sm_clock_ghz = query_sm_clock_ghz();
  StreamResult* d_result = alloc_device<StreamResult>(1);

  for (int i = 0; i < options.warmup_repeats; ++i) {
    if (std::strcmp(options.mode, "remote") == 0) {
      run_for_vec_bytes<true>(d_result, options);
    } else {
      run_for_vec_bytes<false>(d_result, options);
    }
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
  }

  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    if (std::strcmp(options.mode, "remote") == 0) {
      run_for_vec_bytes<true>(d_result, options);
    } else {
      run_for_vec_bytes<false>(d_result, options);
    }
    check_cuda(cudaDeviceSynchronize(), "benchmark synchronize");
    StreamResult result{};
    copy_to_host(&result, d_result, 1);
    print_stream_result("bench_dsmem_read", options, repeat, sm_clock_ghz, result);
  }

  check_cuda(cudaFree(d_result), "cudaFree");
  return 0;
}
