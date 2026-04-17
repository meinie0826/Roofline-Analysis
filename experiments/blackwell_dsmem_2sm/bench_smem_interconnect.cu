#include "common.h"

#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

template <int VecBytes>
struct VecType;

template <> struct VecType<4> { using Type = uint32_t; };
template <> struct VecType<8> { using Type = uint2; };
template <> struct VecType<16> { using Type = uint4; };

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

//=============================================================================
// Mode 0: Local smem read (baseline)
// CTA reads its own smem
//=============================================================================
template <int VecBytes>
__global__ void local_smem_kernel(StreamResult* out, int iters, int buffer_bytes) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  
  const int rank = blockIdx.x;
  uintptr_t base_ptr = reinterpret_cast<uintptr_t>(smem_raw);
  uintptr_t aligned_ptr = align_up_ptr(base_ptr, VecBytes);
  unsigned char* local_ptr = reinterpret_cast<unsigned char*>(aligned_ptr);
  
  // Initialize smem
  for (int i = threadIdx.x; i < buffer_bytes; i += blockDim.x) {
    local_ptr[i] = static_cast<unsigned char>((rank * 17 + i) & 0xff);
  }
  __syncthreads();
  
  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  const Vec* vec_ptr = reinterpret_cast<const Vec*>(local_ptr);
  
  unsigned long long start = 0, stop = 0;
  unsigned long long checksum = 0;
  
  if (threadIdx.x == 0) {
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
  if (threadIdx.x == 0) {
    stop = clock64();
    if (rank == 0) {
      out->cycles = stop - start;
      out->checksum = checksum;
      out->bytes = static_cast<unsigned long long>(iters) * static_cast<unsigned long long>(vec_count) * VecBytes;
    }
  }
}

//=============================================================================
// Mode 1: DSMEM remote read (same cluster)
// CTA1 reads CTA0's smem via cluster.map_shared_rank
//=============================================================================
template <int VecBytes>
__global__ void dsmem_remote_kernel(StreamResult* out, int iters, int buffer_bytes) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  
  uintptr_t base_ptr = reinterpret_cast<uintptr_t>(smem_raw);
  uintptr_t aligned_ptr = align_up_ptr(base_ptr, VecBytes);
  unsigned char* local_ptr = reinterpret_cast<unsigned char*>(aligned_ptr);
  
  // Initialize smem
  for (int i = threadIdx.x; i < buffer_bytes; i += blockDim.x) {
    local_ptr[i] = static_cast<unsigned char>((rank * 17 + i) & 0xff);
  }
  cluster.sync();
  
  // CTA1 reads CTA0's smem
  unsigned char* src_ptr = local_ptr;
  if (rank == 1) {
    src_ptr = reinterpret_cast<unsigned char*>(cluster.map_shared_rank(local_ptr, 0));
  }
  
  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  const Vec* vec_ptr = reinterpret_cast<const Vec*>(src_ptr);
  
  unsigned long long start = 0, stop = 0;
  unsigned long long checksum = 0;
  
  if (threadIdx.x == 0 && rank == 1) {
    start = clock64();
  }
  __syncthreads();
  
  if (rank == 1) {
    for (int iter = 0; iter < iters; ++iter) {
      unsigned long long local_checksum = 0;
      for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
        Vec value = vec_ptr[idx];
        local_checksum ^= checksum_vec<VecBytes>(value) + static_cast<unsigned long long>(iter * 131 + idx);
      }
      checksum ^= local_checksum;
    }
  }
  
  cluster.sync();
  if (threadIdx.x == 0 && rank == 1) {
    stop = clock64();
    out->cycles = stop - start;
    out->checksum = checksum;
    out->bytes = static_cast<unsigned long long>(iters) * static_cast<unsigned long long>(vec_count) * VecBytes;
  }
}

//=============================================================================
// Mode 2: Global memory relay (simulates TPC-internal cross-cluster)
// CTA0 writes to global buffer, CTA1 reads from same buffer
// Uses L2 cache persistence to keep data in L2 (not HBM)
//=============================================================================
template <int VecBytes>
__global__ void global_relay_kernel(
    StreamResult* out,
    int iters,
    int buffer_bytes,
    unsigned char* global_buffer  // L2 persistent buffer
) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  
  const int rank = blockIdx.x;
  
  uintptr_t base_ptr = reinterpret_cast<uintptr_t>(smem_raw);
  uintptr_t aligned_ptr = align_up_ptr(base_ptr, VecBytes);
  unsigned char* local_ptr = reinterpret_cast<unsigned char*>(aligned_ptr);
  
  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  
  if (rank == 0) {
    // Producer: write data to global buffer (L2 cached)
    Vec* gmem_ptr = reinterpret_cast<Vec*>(global_buffer);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      Vec value{};
      gmem_ptr[i] = value;
    }
    __threadfence();  // Ensure writes visible
  }
  __syncthreads();
  
  unsigned long long start = 0, stop = 0;
  unsigned long long checksum = 0;
  
  // Measure: CTA1 reads from global buffer (L2 hit path)
  if (rank == 1) {
    const Vec* gmem_ptr = reinterpret_cast<const Vec*>(global_buffer);
    
    if (threadIdx.x == 0) {
      start = clock64();
    }
    __syncthreads();
    
    for (int iter = 0; iter < iters; ++iter) {
      unsigned long long local_checksum = 0;
      for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
        Vec value = gmem_ptr[idx];
        local_checksum ^= checksum_vec<VecBytes>(value) + static_cast<unsigned long long>(iter * 131 + idx);
      }
      checksum ^= local_checksum;
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
      stop = clock64();
      out->cycles = stop - start;
      out->checksum = checksum;
      out->bytes = static_cast<unsigned long long>(iters) * static_cast<unsigned long long>(vec_count) * VecBytes;
    }
  }
}

//=============================================================================
// Mode 3: Two-kernel relay (true cross-cluster via global memory)
// Kernel 0: CTA0 writes to global buffer
// Kernel 1: CTA1 reads from global buffer (separate launch)
//=============================================================================
template <int VecBytes>
__global__ void producer_kernel(
    unsigned char* global_buffer,
    int buffer_bytes,
    int rank
) {
  if (rank != 0) return;
  
  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  Vec* gmem_ptr = reinterpret_cast<Vec*>(global_buffer);
  
  for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
    Vec value{};
    // Fill with known pattern
    gmem_ptr[idx] = value;
  }
}

template <int VecBytes>
__global__ void consumer_kernel(
    StreamResult* out,
    unsigned char* global_buffer,
    int buffer_bytes,
    int iters,
    int rank
) {
  if (rank != 1) return;
  
  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  const Vec* gmem_ptr = reinterpret_cast<const Vec*>(global_buffer);
  
  unsigned long long checksum = 0;
  
  unsigned long long start = clock64();
  
  for (int iter = 0; iter < iters; ++iter) {
    unsigned long long local_checksum = 0;
    for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
      Vec value = gmem_ptr[idx];
      local_checksum ^= checksum_vec<VecBytes>(value) + static_cast<unsigned long long>(iter * 131 + idx);
    }
    checksum ^= local_checksum;
  }
  
  unsigned long long stop = clock64();
  
  if (threadIdx.x == 0) {
    out->cycles = stop - start;
    out->checksum = checksum;
    out->bytes = static_cast<unsigned long long>(iters) * static_cast<unsigned long long>(vec_count) * VecBytes;
  }
}

//=============================================================================
// Benchmark runner
//=============================================================================
enum class InterconnectMode {
  LOCAL_SMEM,      // Local smem read
  DSMEM_REMOTE,    // Cluster-internal DSMEM
  GLOBAL_RELAY,    // Single kernel, L2 relay
  TWO_KERNEL       // Two kernels, true cross-cluster
};

template <int VecBytes>
void run_benchmark(
    StreamResult* d_result,
    unsigned char* d_global_buffer,
    const StreamOptions& options,
    InterconnectMode mode,
    double sm_clock_ghz
) {
  dim3 block(128, 1, 1);
  
  const int smem_bytes = options.buffer_bytes + 32;
  
  if (mode == InterconnectMode::LOCAL_SMEM) {
    // Single CTA, local smem read
    dim3 grid(1, 1, 1);
    for (int r = 0; r < options.repeats; ++r) {
      local_smem_kernel<VecBytes><<<grid, block, smem_bytes>>>(d_result, options.iters, options.buffer_bytes);
      check_cuda(cudaDeviceSynchronize(), "local_smem_kernel");
      StreamResult result;
      copy_to_host(&result, d_result, 1);
      print_stream_result("local_smem", options, r, sm_clock_ghz, result);
    }
  }
  else if (mode == InterconnectMode::DSMEM_REMOTE) {
    // 2-CTA cluster, DSMEM read
    dim3 grid(2, 1, 1);
    
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2, 1, 1};
    config.attrs = &attr;
    config.numAttrs = 1;
    
    for (int r = 0; r < options.repeats; ++r) {
      check_cuda(cudaLaunchKernelEx(&config, dsmem_remote_kernel<VecBytes>, d_result, options.iters, options.buffer_bytes), "dsmem_remote_kernel");
      check_cuda(cudaDeviceSynchronize(), "dsmem_remote sync");
      StreamResult result;
      copy_to_host(&result, d_result, 1);
      print_stream_result("dsmem_remote", options, r, sm_clock_ghz, result);
    }
  }
  else if (mode == InterconnectMode::GLOBAL_RELAY) {
    // Single kernel, two CTAs not in cluster, communicate via global buffer
    dim3 grid(2, 1, 1);
    
    for (int r = 0; r < options.repeats; ++r) {
      global_relay_kernel<VecBytes><<<grid, block, smem_bytes>>>(d_result, options.iters, options.buffer_bytes, d_global_buffer);
      check_cuda(cudaDeviceSynchronize(), "global_relay_kernel");
      StreamResult result;
      copy_to_host(&result, d_result, 1);
      print_stream_result("global_relay", options, r, sm_clock_ghz, result);
    }
  }
  else if (mode == InterconnectMode::TWO_KERNEL) {
    // Two separate kernels, simulate true cross-cluster
    dim3 grid(1, 1, 1);
    
    for (int r = 0; r < options.repeats; ++r) {
      // Producer
      producer_kernel<VecBytes><<<grid, block>>>(d_global_buffer, options.buffer_bytes, 0);
      check_cuda(cudaDeviceSynchronize(), "producer_kernel");
      
      // Consumer
      consumer_kernel<VecBytes><<<grid, block>>>(d_result, d_global_buffer, options.buffer_bytes, options.iters, 1);
      check_cuda(cudaDeviceSynchronize(), "consumer_kernel");
      
      StreamResult result;
      copy_to_host(&result, d_result, 1);
      print_stream_result("two_kernel", options, r, sm_clock_ghz, result);
    }
  }
}

void print_usage(const char* prog) {
  std::fprintf(stderr, "Usage: %s [options]\n", prog);
  std::fprintf(stderr, "Options:\n");
  std::fprintf(stderr, "  --mode=local|dsmem|global|two_kernel  (default: local)\n");
  std::fprintf(stderr, "  --buffer-bytes=N    buffer size (default: 65536)\n");
  std::fprintf(stderr, "  --iters=N           iterations (default: 2048)\n");
  std::fprintf(stderr, "  --repeats=N         measurement repeats (default: 20)\n");
  std::fprintf(stderr, "  --vec-bytes=N       vector width 4|8|16 (default: 16)\n");
}

int main(int argc, char** argv) {
  StreamOptions options;
  const char* mode_str = "local";
  
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    options.buffer_bytes = static_cast<int>(parse_i64_arg(arg, "--buffer-bytes=", options.buffer_bytes));
    options.iters = parse_int_arg(arg, "--iters=", options.iters);
    options.repeats = parse_int_arg(arg, "--repeats=", options.repeats);
    options.vec_bytes = parse_int_arg(arg, "--vec-bytes=", options.vec_bytes);
    if (starts_with(arg, "--mode=")) {
      mode_str = parse_str_arg(arg, "--mode=", mode_str);
    }
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }
  
  InterconnectMode mode;
  if (std::strcmp(mode_str, "local") == 0) {
    mode = InterconnectMode::LOCAL_SMEM;
  } else if (std::strcmp(mode_str, "dsmem") == 0) {
    mode = InterconnectMode::DSMEM_REMOTE;
  } else if (std::strcmp(mode_str, "global") == 0) {
    mode = InterconnectMode::GLOBAL_RELAY;
  } else if (std::strcmp(mode_str, "two_kernel") == 0) {
    mode = InterconnectMode::TWO_KERNEL;
  } else {
    std::fprintf(stderr, "Unknown mode: %s\n", mode_str);
    print_usage(argv[0]);
    return 1;
  }
  
  if (!is_valid_vec_bytes(options.vec_bytes)) {
    std::fprintf(stderr, "Unsupported vec_bytes=%d (use 4, 8, or 16)\n", options.vec_bytes);
    return 1;
  }
  
  const double sm_clock_ghz = query_sm_clock_ghz();
  StreamResult* d_result = alloc_device<StreamResult>(1);
  unsigned char* d_global_buffer = alloc_device<unsigned char>(options.buffer_bytes);
  
  std::fprintf(stdout, "CONFIG benchmark=bench_smem_interconnect mode=%s buffer_bytes=%d iters=%d repeats=%d vec_bytes=%d gpu=\"%s\"\n",
               mode_str, options.buffer_bytes, options.iters, options.repeats, options.vec_bytes,
               query_gpu_name().c_str());
  
  if (options.vec_bytes == 4) {
    run_benchmark<4>(d_result, d_global_buffer, options, mode, sm_clock_ghz);
  } else if (options.vec_bytes == 8) {
    run_benchmark<8>(d_result, d_global_buffer, options, mode, sm_clock_ghz);
  } else {
    run_benchmark<16>(d_result, d_global_buffer, options, mode, sm_clock_ghz);
  }
  
  check_cuda(cudaFree(d_global_buffer), "cudaFree global buffer");
  check_cuda(cudaFree(d_result), "cudaFree result");
  
  return 0;
}
