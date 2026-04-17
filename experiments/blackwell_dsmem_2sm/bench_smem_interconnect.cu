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

//=============================================================================
// Mode 0: Local smem read
//=============================================================================
template <int VecBytes>
__global__ void local_read_kernel(StreamResult* out, int iters, int buffer_bytes) {
  __shared__ unsigned char smem[65536];
  
  // Initialize
  for (int i = threadIdx.x; i < buffer_bytes; i += blockDim.x) {
    smem[i] = static_cast<unsigned char>(i & 0xff);
  }
  __syncthreads();
  
  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  const Vec* vec_ptr = reinterpret_cast<const Vec*>(smem);
  
  unsigned long long checksum = 0;
  
  unsigned long long start = clock64();
  for (int iter = 0; iter < iters; ++iter) {
    unsigned long long local_sum = 0;
    for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
      Vec value = vec_ptr[idx];
      local_sum ^= checksum_vec<VecBytes>(value);
    }
    checksum ^= local_sum;
  }
  unsigned long long stop = clock64();
  
  if (threadIdx.x == 0) {
    out->cycles = stop - start;
    out->checksum = checksum;
    out->bytes = static_cast<unsigned long long>(iters) * vec_count * VecBytes;
  }
}

//=============================================================================
// Mode 1: DSMEM remote read
//=============================================================================
template <int VecBytes>
__global__ void dsmem_read_kernel(StreamResult* out, int iters, int buffer_bytes) {
  extern __shared__ unsigned char smem_raw[];
  
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  
  // Each CTA has its own smem region
  unsigned char* my_smem = smem_raw;
  
  // Initialize own smem
  for (int i = threadIdx.x; i < buffer_bytes; i += blockDim.x) {
    my_smem[i] = static_cast<unsigned char>((rank * 17 + i) & 0xff);
  }
  cluster.sync();
  
  // CTA1 reads from CTA0's smem
  const unsigned char* src_smem = reinterpret_cast<const unsigned char*>(
      cluster.map_shared_rank(my_smem, 0));
  
  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  const Vec* vec_ptr = reinterpret_cast<const Vec*>(src_smem);
  
  unsigned long long checksum = 0;
  unsigned long long cycles = 0;
  
  if (rank == 1) {
    unsigned long long start = clock64();
    for (int iter = 0; iter < iters; ++iter) {
      unsigned long long local_sum = 0;
      for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
        Vec value = vec_ptr[idx];
        local_sum ^= checksum_vec<VecBytes>(value);
      }
      checksum ^= local_sum;
    }
    unsigned long long stop = clock64();
    cycles = stop - start;
  }
  
  cluster.sync();
  if (rank == 1 && threadIdx.x == 0) {
    out->cycles = cycles;
    out->checksum = checksum;
    out->bytes = static_cast<unsigned long long>(iters) * vec_count * VecBytes;
  }
}

//=============================================================================
// Mode 2: Global memory (L2 cached) relay
//=============================================================================
template <int VecBytes>
__global__ void global_read_kernel(
    StreamResult* out,
    const unsigned char* __restrict__ gmem_buffer,
    int iters,
    int buffer_bytes
) {
  using Vec = typename VecType<VecBytes>::Type;
  const int vec_count = buffer_bytes / VecBytes;
  const Vec* vec_ptr = reinterpret_cast<const Vec*>(gmem_buffer);
  
  unsigned long long checksum = 0;
  
  unsigned long long start = clock64();
  for (int iter = 0; iter < iters; ++iter) {
    unsigned long long local_sum = 0;
    for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
      Vec value = vec_ptr[idx];
      local_sum ^= checksum_vec<VecBytes>(value);
    }
    checksum ^= local_sum;
  }
  unsigned long long stop = clock64();
  
  if (threadIdx.x == 0) {
    out->cycles = stop - start;
    out->checksum = checksum;
    out->bytes = static_cast<unsigned long long>(iters) * vec_count * VecBytes;
  }
}

//=============================================================================
// Launcher
//=============================================================================
void print_usage(const char* prog) {
  std::fprintf(stderr, "Usage: %s --mode=local|dsmem|global --buffer-bytes=N --iters=N --repeats=N\n", prog);
}

int main(int argc, char** argv) {
  const char* mode_str = "local";
  int buffer_bytes = 65536;
  int iters = 2048;
  int repeats = 10;
  int vec_bytes = 16;
  
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (starts_with(arg, "--mode=")) mode_str = arg + 7;
    else if (starts_with(arg, "--buffer-bytes=")) buffer_bytes = std::atoi(arg + 15);
    else if (starts_with(arg, "--iters=")) iters = std::atoi(arg + 8);
    else if (starts_with(arg, "--repeats=")) repeats = std::atoi(arg + 10);
    else if (starts_with(arg, "--vec-bytes=")) vec_bytes = std::atoi(arg + 12);
    else if (std::strcmp(arg, "--help") == 0) {
      print_usage(argv[0]);
      return 0;
    }
  }
  
  if (vec_bytes != 4 && vec_bytes != 8 && vec_bytes != 16) {
    std::fprintf(stderr, "Unsupported vec_bytes=%d\n", vec_bytes);
    return 1;
  }
  
  const double sm_clock_ghz = query_sm_clock_ghz();
  StreamResult* d_result = alloc_device<StreamResult>(1);
  unsigned char* d_buffer = nullptr;
  
  bool is_global = (std::strcmp(mode_str, "global") == 0);
  if (is_global) {
    d_buffer = alloc_device<unsigned char>(buffer_bytes);
    // Initialize with known pattern
    unsigned char* h_buffer = new unsigned char[buffer_bytes];
    for (int i = 0; i < buffer_bytes; ++i) h_buffer[i] = static_cast<unsigned char>(i & 0xff);
    check_cuda(cudaMemcpy(d_buffer, h_buffer, buffer_bytes, cudaMemcpyHostToDevice), "init buffer");
    delete[] h_buffer;
  }
  
  std::fprintf(stdout, "CONFIG mode=%s buffer_bytes=%d iters=%d repeats=%d vec_bytes=%d gpu=\"%s\"\n",
               mode_str, buffer_bytes, iters, repeats, vec_bytes, gpu_name().c_str());
  
  dim3 block(128, 1, 1);
  
  auto run_local = [&](int repeat) {
    local_read_kernel<16><<<1, block, buffer_bytes>>>(d_result, iters, buffer_bytes);
    check_cuda(cudaGetLastError(), "local kernel launch");
    check_cuda(cudaDeviceSynchronize(), "local sync");
    StreamResult result;
    copy_to_host(&result, d_result, 1);
    double elapsed_ns = static_cast<double>(result.cycles) / sm_clock_ghz;
    double bw_gbps = static_cast<double>(result.bytes) / elapsed_ns;
    std::fprintf(stdout, "RESULT mode=local repeat=%d cycles=%llu bytes=%llu bw_gbps=%.4f checksum=%llu\n",
                 repeat, result.cycles, result.bytes, bw_gbps, result.checksum);
  };
  
  auto run_dsmem = [&](int repeat) {
    dim3 grid(2, 1, 1);
    size_t smem_per_cta = buffer_bytes;
    
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_per_cta;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim = {2, 1, 1};
    config.attrs = attrs;
    config.numAttrs = 1;
    
    check_cuda(cudaLaunchKernelEx(&config, dsmem_read_kernel<16>, d_result, iters, buffer_bytes), "dsmem launch");
    check_cuda(cudaDeviceSynchronize(), "dsmem sync");
    StreamResult result;
    copy_to_host(&result, d_result, 1);
    double elapsed_ns = static_cast<double>(result.cycles) / sm_clock_ghz;
    double bw_gbps = static_cast<double>(result.bytes) / elapsed_ns;
    std::fprintf(stdout, "RESULT mode=dsmem repeat=%d cycles=%llu bytes=%llu bw_gbps=%.4f checksum=%llu\n",
                 repeat, result.cycles, result.bytes, bw_gbps, result.checksum);
  };
  
  auto run_global = [&](int repeat) {
    global_read_kernel<16><<<1, block>>>(d_result, d_buffer, iters, buffer_bytes);
    check_cuda(cudaGetLastError(), "global kernel launch");
    check_cuda(cudaDeviceSynchronize(), "global sync");
    StreamResult result;
    copy_to_host(&result, d_result, 1);
    double elapsed_ns = static_cast<double>(result.cycles) / sm_clock_ghz;
    double bw_gbps = static_cast<double>(result.bytes) / elapsed_ns;
    std::fprintf(stdout, "RESULT mode=global repeat=%d cycles=%llu bytes=%llu bw_gbps=%.4f checksum=%llu\n",
                 repeat, result.cycles, result.bytes, bw_gbps, result.checksum);
  };
  
  for (int r = 0; r < repeats; ++r) {
    if (std::strcmp(mode_str, "local") == 0) {
      run_local(r);
    } else if (std::strcmp(mode_str, "dsmem") == 0) {
      run_dsmem(r);
    } else if (std::strcmp(mode_str, "global") == 0) {
      run_global(r);
    } else {
      std::fprintf(stderr, "Unknown mode: %s\n", mode_str);
      return 1;
    }
  }
  
  if (d_buffer) check_cuda(cudaFree(d_buffer), "free buffer");
  check_cuda(cudaFree(d_result), "free result");
  
  return 0;
}
