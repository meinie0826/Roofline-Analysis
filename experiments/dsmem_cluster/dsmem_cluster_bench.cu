#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

namespace {

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

bool starts_with(const char* arg, const char* prefix) {
  return std::strncmp(arg, prefix, std::strlen(prefix)) == 0;
}

int parse_int(const char* arg, const char* prefix, int current) {
  return starts_with(arg, prefix) ? std::atoi(arg + std::strlen(prefix)) : current;
}

size_t parse_size(const char* arg, const char* prefix, size_t current) {
  return starts_with(arg, prefix) ? static_cast<size_t>(std::strtoull(arg + std::strlen(prefix), nullptr, 10)) : current;
}

const char* parse_str(const char* arg, const char* prefix, const char* current) {
  return starts_with(arg, prefix) ? arg + std::strlen(prefix) : current;
}

std::vector<int> parse_cluster_sizes(const char* text) {
  std::vector<int> values;
  const char* cursor = text;
  while (*cursor != '\0') {
    char* end = nullptr;
    long value = std::strtol(cursor, &end, 10);
    if (end == cursor) break;
    if (value > 0) values.push_back(static_cast<int>(value));
    cursor = (*end == ',') ? end + 1 : end;
  }
  if (values.empty()) values = {1, 2, 4, 8, 16};
  return values;
}

std::string gpu_name() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp prop{};
  check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  return std::string(prop.name);
}

int sm_count() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  int value = 0;
  check_cuda(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device), "cudaDevAttrMultiProcessorCount");
  return value;
}

int max_blocks_per_cluster() {
  // Some CUDA 12.x headers that can compile Blackwell targets do not expose
  // cudaDevAttrMaxBlocksPerCluster. The benchmark still validates each tested
  // cluster size by launching it, so keep this metadata field best-effort.
  return 0;
}

double sm_clock_ghz() {
  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  int khz = 0;
  check_cuda(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, device), "cudaDevAttrClockRate");
  return static_cast<double>(khz) / 1.0e6;
}

float elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
  float ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
  return ms;
}

template <typename Kernel>
bool prepare_cluster_kernel(Kernel kernel, int dynamic_smem_bytes) {
  if (dynamic_smem_bytes > 48 * 1024) {
    cudaError_t smem_err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem_bytes);
    if (smem_err != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
  }
  cudaError_t cluster_err = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (cluster_err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

void print_unsupported(const char* metric, int cluster_size, const char* reason) {
  std::printf("{\"metric\":\"%s\",\"cluster_size\":%d,\"supported\":false,\"reason\":\"%s\"}\n",
              metric, cluster_size, reason);
}

struct Options {
  int repeats = 30;
  int warmup = 5;
  int latency_iters = 200000;
  int bandwidth_iters = 4096;
  int block_threads = 128;
  int latency_elems = 4096;
  int bandwidth_bytes = 32768;
  size_t global_latency_elems = 64ull * 1024ull * 1024ull;
  size_t global_bandwidth_bytes = 512ull * 1024ull * 1024ull;
  int max_grid_blocks = 0;
  const char* gpu_label = "auto";
  const char* cluster_sizes_text = "1,2,4,8,16";
};

struct LatencyResult {
  unsigned long long cycles;
  unsigned int sink;
};

__global__ void init_pointer_chase(unsigned int* data, size_t elems, unsigned int stride) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (; idx < elems; idx += step) {
    data[idx] = static_cast<unsigned int>((idx + stride) % elems);
  }
}

__global__ void dsmem_latency_kernel(LatencyResult* out, int iters, int elems, int peer_delta) {
  extern __shared__ unsigned int latency_smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_size = static_cast<int>(cluster.num_blocks());
  const unsigned int stride = 131u;

  for (int i = threadIdx.x; i < elems; i += blockDim.x) {
    latency_smem[i] = static_cast<unsigned int>((i + stride) % elems);
  }
  cluster.sync();

  if (rank == 0 && threadIdx.x == 0) {
    const int peer_rank = (cluster_size == 1) ? 0 : ((rank + peer_delta) % cluster_size);
    volatile unsigned int* peer = reinterpret_cast<volatile unsigned int*>(cluster.map_shared_rank(latency_smem, peer_rank));
    unsigned int index = static_cast<unsigned int>(peer[0]);
    unsigned long long start = clock64();
    for (int iter = 0; iter < iters; ++iter) {
      index = peer[index];
    }
    unsigned long long stop = clock64();
    out->cycles = stop - start;
    out->sink = index;
  }
  cluster.sync();
}

__global__ void global_latency_kernel(LatencyResult* out, const unsigned int* data, int iters, size_t elems) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    unsigned int index = data[0];
    unsigned long long start = clock64();
    for (int iter = 0; iter < iters; ++iter) {
      index = data[index];
    }
    unsigned long long stop = clock64();
    out->cycles = stop - start;
    out->sink = index;
  }
}

__global__ void dsmem_bandwidth_kernel(unsigned long long* sinks, int iters, int buffer_bytes) {
  extern __shared__ __align__(16) unsigned char bandwidth_smem[];
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_size = static_cast<int>(cluster.num_blocks());

  for (int i = threadIdx.x * 16; i < buffer_bytes; i += blockDim.x * 16) {
    uint4 value{static_cast<unsigned int>(rank + i), static_cast<unsigned int>(rank + i + 1),
                static_cast<unsigned int>(rank + i + 2), static_cast<unsigned int>(rank + i + 3)};
    *reinterpret_cast<uint4*>(bandwidth_smem + i) = value;
  }
  cluster.sync();

  const int peer_rank = (cluster_size == 1) ? rank : ((rank + 1) % cluster_size);
  const unsigned char* peer = reinterpret_cast<const unsigned char*>(cluster.map_shared_rank(bandwidth_smem, peer_rank));
  const int vec_count = buffer_bytes / 16;
  const uint4* vec = reinterpret_cast<const uint4*>(peer);
  unsigned long long checksum = 0;

  for (int iter = 0; iter < iters; ++iter) {
    unsigned long long local = 0;
    for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
      uint4 v = vec[idx];
      local += static_cast<unsigned long long>(v.x) + v.y + v.z + v.w + static_cast<unsigned int>(iter);
    }
    checksum ^= local;
  }

  cluster.sync();
  const int global_block = blockIdx.x;
  if (threadIdx.x == 0) sinks[global_block] = checksum;
}

__global__ void local_smem_bandwidth_kernel(unsigned long long* sinks, int iters, int buffer_bytes) {
  extern __shared__ __align__(16) unsigned char local_bandwidth_smem[];

  for (int i = threadIdx.x * 16; i < buffer_bytes; i += blockDim.x * 16) {
    uint4 value{static_cast<unsigned int>(blockIdx.x + i), static_cast<unsigned int>(blockIdx.x + i + 1),
                static_cast<unsigned int>(blockIdx.x + i + 2), static_cast<unsigned int>(blockIdx.x + i + 3)};
    *reinterpret_cast<uint4*>(local_bandwidth_smem + i) = value;
  }
  __syncthreads();

  const int vec_count = buffer_bytes / 16;
  const uint4* vec = reinterpret_cast<const uint4*>(local_bandwidth_smem);
  unsigned long long checksum = 0;

  for (int iter = 0; iter < iters; ++iter) {
    unsigned long long local = 0;
    for (int idx = threadIdx.x; idx < vec_count; idx += blockDim.x) {
      uint4 v = vec[idx];
      local += static_cast<unsigned long long>(v.x) + v.y + v.z + v.w + static_cast<unsigned int>(iter);
    }
    checksum ^= local;
  }

  __syncthreads();
  if (threadIdx.x == 0) sinks[blockIdx.x] = checksum;
}

__device__ __forceinline__ unsigned int read_smid() {
  unsigned int smid = 0;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

__global__ void smid_residency_kernel(unsigned int* smids) {
  cg::cluster_group cluster = cg::this_cluster();
  if (threadIdx.x == 0) {
    smids[blockIdx.x] = read_smid();
  }
  cluster.sync();
}

__global__ void global_bandwidth_kernel(unsigned long long* sinks, const uint4* data, size_t vec_count, int iters) {
  unsigned long long checksum = 0;
  const size_t start = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (int iter = 0; iter < iters; ++iter) {
    unsigned long long local = 0;
    for (size_t idx = start; idx < vec_count; idx += stride) {
      uint4 v = data[idx];
      local += static_cast<unsigned long long>(v.x) + v.y + v.z + v.w + static_cast<unsigned int>(iter);
    }
    checksum ^= local;
  }
  if (threadIdx.x == 0) sinks[blockIdx.x] = checksum;
}

__global__ void init_uint4(uint4* data, size_t vec_count) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (; idx < vec_count; idx += step) {
    data[idx] = uint4{static_cast<unsigned int>(idx), static_cast<unsigned int>(idx >> 3),
                      static_cast<unsigned int>(idx * 17), static_cast<unsigned int>(idx * 31)};
  }
}

std::vector<double> sorted_values(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values;
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  values = sorted_values(values);
  const size_t mid = values.size() / 2;
  return (values.size() % 2 == 0) ? 0.5 * (values[mid - 1] + values[mid]) : values[mid];
}

double mean(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

int estimate_active_clusters(int cluster_size, int block_threads, int dynamic_smem_bytes) {
#if CUDART_VERSION >= 12000
  if (!prepare_cluster_kernel(dsmem_bandwidth_kernel, dynamic_smem_bytes)) {
    return -1;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(cluster_size, 1, 1);
  config.blockDim = dim3(block_threads, 1, 1);
  config.dynamicSmemBytes = dynamic_smem_bytes;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_size;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  int active_clusters = 0;
  cudaError_t err = cudaOccupancyMaxActiveClusters(&active_clusters,
                                                   reinterpret_cast<const void*>(dsmem_bandwidth_kernel),
                                                   &config);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  return active_clusters;
#else
  (void)cluster_size;
  (void)block_threads;
  (void)dynamic_smem_bytes;
  return -1;
#endif
}

bool launch_dsmem_latency(int cluster_size, int peer_delta, const Options& options, LatencyResult* d_result) {
  if (!prepare_cluster_kernel(dsmem_latency_kernel, options.latency_elems * static_cast<int>(sizeof(unsigned int)))) {
    return false;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(cluster_size, 1, 1);
  config.blockDim = dim3(options.block_threads, 1, 1);
  config.dynamicSmemBytes = options.latency_elems * sizeof(unsigned int);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_size;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(&config, dsmem_latency_kernel, d_result, options.latency_iters, options.latency_elems, peer_delta);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return cudaDeviceSynchronize() == cudaSuccess;
}

bool run_dsmem_latency(int cluster_size, const Options& options) {
  LatencyResult* d_result = nullptr;
  check_cuda(cudaMalloc(&d_result, sizeof(LatencyResult)), "cudaMalloc latency result");
  const int first_delta = cluster_size == 1 ? 0 : 1;
  const int last_delta = cluster_size == 1 ? 0 : cluster_size - 1;
  std::vector<double> peer_medians;
  std::vector<double> all_cycles;

  for (int peer_delta = first_delta; peer_delta <= last_delta; ++peer_delta) {
    std::vector<double> cycles;
    cycles.reserve(options.repeats);

    for (int i = 0; i < options.warmup; ++i) {
      if (!launch_dsmem_latency(cluster_size, peer_delta, options, d_result)) {
        cudaFree(d_result);
        print_unsupported("dsmem_latency", cluster_size, "launch_failed");
        return false;
      }
    }
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      if (!launch_dsmem_latency(cluster_size, peer_delta, options, d_result)) {
        cudaFree(d_result);
        print_unsupported("dsmem_latency", cluster_size, "launch_failed");
        return false;
      }
      LatencyResult h{};
      check_cuda(cudaMemcpy(&h, d_result, sizeof(h), cudaMemcpyDeviceToHost), "cudaMemcpy latency result");
      cycles.push_back(static_cast<double>(h.cycles) / static_cast<double>(options.latency_iters));
    }
    const double peer_median = median(cycles);
    peer_medians.push_back(peer_median);
    all_cycles.insert(all_cycles.end(), cycles.begin(), cycles.end());
    std::printf("{\"metric\":\"dsmem_latency_peer\",\"cluster_size\":%d,\"peer_delta\":%d,\"supported\":true,\"cycles_per_load_median\":%.6f,\"cycles_per_load_mean\":%.6f,\"iters\":%d,\"repeats\":%d}\n",
                cluster_size, peer_delta, peer_median, mean(cycles), options.latency_iters, options.repeats);
  }

  std::printf("{\"metric\":\"dsmem_latency\",\"cluster_size\":%d,\"supported\":true,\"cycles_per_load_median\":%.6f,\"cycles_per_load_mean\":%.6f,\"peer_delta_count\":%zu,\"iters\":%d,\"repeats\":%d}\n",
              cluster_size, mean(peer_medians), mean(all_cycles), peer_medians.size(), options.latency_iters, options.repeats);
  cudaFree(d_result);
  return true;
}

void run_global_latency(const Options& options) {
  unsigned int* d_data = nullptr;
  LatencyResult* d_result = nullptr;
  check_cuda(cudaMalloc(&d_data, options.global_latency_elems * sizeof(unsigned int)), "cudaMalloc global latency data");
  check_cuda(cudaMalloc(&d_result, sizeof(LatencyResult)), "cudaMalloc global latency result");
  init_pointer_chase<<<1024, 256>>>(d_data, options.global_latency_elems, 1048583u);
  check_cuda(cudaGetLastError(), "init_pointer_chase launch");
  check_cuda(cudaDeviceSynchronize(), "init_pointer_chase sync");

  std::vector<double> cycles;
  cycles.reserve(options.repeats);
  for (int i = 0; i < options.warmup; ++i) {
    global_latency_kernel<<<1, 1>>>(d_result, d_data, options.latency_iters, options.global_latency_elems);
    check_cuda(cudaDeviceSynchronize(), "global latency warmup");
  }
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    global_latency_kernel<<<1, 1>>>(d_result, d_data, options.latency_iters, options.global_latency_elems);
    check_cuda(cudaDeviceSynchronize(), "global latency sync");
    LatencyResult h{};
    check_cuda(cudaMemcpy(&h, d_result, sizeof(h), cudaMemcpyDeviceToHost), "cudaMemcpy global latency result");
    cycles.push_back(static_cast<double>(h.cycles) / static_cast<double>(options.latency_iters));
  }

  std::printf("{\"metric\":\"global_latency\",\"cluster_size\":0,\"supported\":true,\"cycles_per_load_median\":%.6f,\"cycles_per_load_mean\":%.6f,\"iters\":%d,\"repeats\":%d,\"elems\":%zu}\n",
              median(cycles), mean(cycles), options.latency_iters, options.repeats, options.global_latency_elems);
  cudaFree(d_data);
  cudaFree(d_result);
}

bool launch_dsmem_bandwidth(int cluster_size, const Options& options, int grid_blocks, unsigned long long* d_sinks, cudaEvent_t start, cudaEvent_t stop, float* ms) {
  if (!prepare_cluster_kernel(dsmem_bandwidth_kernel, options.bandwidth_bytes)) {
    return false;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid_blocks, 1, 1);
  config.blockDim = dim3(options.block_threads, 1, 1);
  config.dynamicSmemBytes = options.bandwidth_bytes;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_size;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  check_cuda(cudaEventRecord(start), "cudaEventRecord start dsmem bandwidth");
  cudaError_t err = cudaLaunchKernelEx(&config, dsmem_bandwidth_kernel, d_sinks, options.bandwidth_iters, options.bandwidth_bytes);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop dsmem bandwidth");
  cudaError_t sync_err = cudaEventSynchronize(stop);
  if (sync_err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  *ms = elapsed_ms(start, stop);
  return true;
}

bool run_dsmem_bandwidth(int cluster_size, const Options& options) {
  const int active_clusters = estimate_active_clusters(cluster_size, options.block_threads, options.bandwidth_bytes);
  int grid_blocks = (active_clusters > 0 ? active_clusters : std::max(1, sm_count() / std::max(1, cluster_size))) * cluster_size;
  if (options.max_grid_blocks > 0) grid_blocks = std::min(grid_blocks, options.max_grid_blocks);
  grid_blocks = std::max(cluster_size, (grid_blocks / cluster_size) * cluster_size);

  unsigned long long* d_sinks = nullptr;
  check_cuda(cudaMalloc(&d_sinks, sizeof(unsigned long long) * grid_blocks), "cudaMalloc dsmem sinks");
  cudaEvent_t start{}, stop{};
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  float ms = 0.0f;
  for (int i = 0; i < options.warmup; ++i) {
    if (!launch_dsmem_bandwidth(cluster_size, options, grid_blocks, d_sinks, start, stop, &ms)) {
      cudaFree(d_sinks);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      print_unsupported("dsmem_bandwidth", cluster_size, "launch_failed");
      return false;
    }
  }

  std::vector<double> tbps;
  tbps.reserve(options.repeats);
  const double bytes = static_cast<double>(grid_blocks) * static_cast<double>(options.bandwidth_iters) * static_cast<double>(options.bandwidth_bytes);
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    if (!launch_dsmem_bandwidth(cluster_size, options, grid_blocks, d_sinks, start, stop, &ms)) {
      cudaFree(d_sinks);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      print_unsupported("dsmem_bandwidth", cluster_size, "launch_failed");
      return false;
    }
    tbps.push_back(bytes / (static_cast<double>(ms) * 1.0e-3) / 1.0e12);
  }

  std::printf("{\"metric\":\"dsmem_bandwidth\",\"cluster_size\":%d,\"supported\":true,\"bandwidth_tb_s_median\":%.6f,\"bandwidth_tb_s_mean\":%.6f,\"grid_blocks\":%d,\"active_clusters_estimate\":%d,\"iters\":%d,\"buffer_bytes\":%d,\"repeats\":%d}\n",
              cluster_size, median(tbps), mean(tbps), grid_blocks, active_clusters, options.bandwidth_iters, options.bandwidth_bytes, options.repeats);
  cudaFree(d_sinks);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return true;
}

void run_local_smem_bandwidth(const Options& options) {
  const int blocks = options.max_grid_blocks > 0 ? options.max_grid_blocks : sm_count();
  unsigned long long* d_sinks = nullptr;
  check_cuda(cudaMalloc(&d_sinks, sizeof(unsigned long long) * blocks), "cudaMalloc local smem sinks");

  cudaEvent_t start{}, stop{};
  check_cuda(cudaEventCreate(&start), "cudaEventCreate local smem start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate local smem stop");

  if (options.bandwidth_bytes > 48 * 1024) {
    check_cuda(cudaFuncSetAttribute(local_smem_bandwidth_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, options.bandwidth_bytes),
               "cudaFuncSetAttribute local smem MaxDynamicSharedMemorySize");
  }

  for (int i = 0; i < options.warmup; ++i) {
    local_smem_bandwidth_kernel<<<blocks, options.block_threads, options.bandwidth_bytes>>>(d_sinks, options.bandwidth_iters, options.bandwidth_bytes);
    check_cuda(cudaDeviceSynchronize(), "local smem bandwidth warmup");
  }

  std::vector<double> tbps;
  tbps.reserve(options.repeats);
  const double bytes = static_cast<double>(blocks) * static_cast<double>(options.bandwidth_iters) * static_cast<double>(options.bandwidth_bytes);
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    check_cuda(cudaEventRecord(start), "local smem bandwidth event start");
    local_smem_bandwidth_kernel<<<blocks, options.block_threads, options.bandwidth_bytes>>>(d_sinks, options.bandwidth_iters, options.bandwidth_bytes);
    check_cuda(cudaEventRecord(stop), "local smem bandwidth event stop");
    check_cuda(cudaEventSynchronize(stop), "local smem bandwidth event sync");
    float ms = elapsed_ms(start, stop);
    tbps.push_back(bytes / (static_cast<double>(ms) * 1.0e-3) / 1.0e12);
  }

  std::printf("{\"metric\":\"local_smem_bandwidth\",\"cluster_size\":0,\"supported\":true,\"bandwidth_tb_s_median\":%.6f,\"bandwidth_tb_s_mean\":%.6f,\"grid_blocks\":%d,\"iters\":%d,\"buffer_bytes\":%d,\"repeats\":%d}\n",
              median(tbps), mean(tbps), blocks, options.bandwidth_iters, options.bandwidth_bytes, options.repeats);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_sinks);
}

void run_global_bandwidth(const Options& options) {
  const size_t vec_count = options.global_bandwidth_bytes / sizeof(uint4);
  uint4* d_data = nullptr;
  unsigned long long* d_sinks = nullptr;
  const int blocks = options.max_grid_blocks > 0 ? options.max_grid_blocks : sm_count() * 4;
  check_cuda(cudaMalloc(&d_data, vec_count * sizeof(uint4)), "cudaMalloc global bandwidth data");
  check_cuda(cudaMalloc(&d_sinks, blocks * sizeof(unsigned long long)), "cudaMalloc global bandwidth sinks");
  init_uint4<<<1024, 256>>>(d_data, vec_count);
  check_cuda(cudaGetLastError(), "init_uint4 launch");
  check_cuda(cudaDeviceSynchronize(), "init_uint4 sync");

  cudaEvent_t start{}, stop{};
  check_cuda(cudaEventCreate(&start), "cudaEventCreate global start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate global stop");
  for (int i = 0; i < options.warmup; ++i) {
    global_bandwidth_kernel<<<blocks, options.block_threads>>>(d_sinks, d_data, vec_count, options.bandwidth_iters);
    check_cuda(cudaDeviceSynchronize(), "global bandwidth warmup");
  }

  std::vector<double> tbps;
  const double bytes = static_cast<double>(vec_count * sizeof(uint4)) * static_cast<double>(options.bandwidth_iters);
  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    check_cuda(cudaEventRecord(start), "global bandwidth event start");
    global_bandwidth_kernel<<<blocks, options.block_threads>>>(d_sinks, d_data, vec_count, options.bandwidth_iters);
    check_cuda(cudaEventRecord(stop), "global bandwidth event stop");
    check_cuda(cudaEventSynchronize(stop), "global bandwidth event sync");
    float ms = elapsed_ms(start, stop);
    tbps.push_back(bytes / (static_cast<double>(ms) * 1.0e-3) / 1.0e12);
  }
  std::printf("{\"metric\":\"global_bandwidth\",\"cluster_size\":0,\"supported\":true,\"bandwidth_tb_s_median\":%.6f,\"bandwidth_tb_s_mean\":%.6f,\"grid_blocks\":%d,\"iters\":%d,\"bytes_per_iter\":%zu,\"repeats\":%d}\n",
              median(tbps), mean(tbps), blocks, options.bandwidth_iters, vec_count * sizeof(uint4), options.repeats);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_data);
  cudaFree(d_sinks);
}

void print_active_sm(int cluster_size, const Options& options) {
  const int active_clusters = estimate_active_clusters(cluster_size, options.block_threads, options.bandwidth_bytes);
  if (active_clusters < 0) {
    print_unsupported("active_sm", cluster_size, "occupancy_api_failed");
    return;
  }
  std::printf("{\"metric\":\"active_sm\",\"cluster_size\":%d,\"supported\":true,\"active_clusters_estimate\":%d,\"active_sms_estimate\":%d}\n",
              cluster_size, active_clusters, active_clusters * cluster_size);
}

bool print_smid_residency(int cluster_size, const Options& options) {
  if (!prepare_cluster_kernel(smid_residency_kernel, 0)) {
    print_unsupported("smid_residency", cluster_size, "prepare_failed");
    return false;
  }

  const int active_clusters = estimate_active_clusters(cluster_size, options.block_threads, options.bandwidth_bytes);
  int grid_blocks = (active_clusters > 0 ? active_clusters : std::max(1, sm_count() / std::max(1, cluster_size))) * cluster_size;
  if (options.max_grid_blocks > 0) grid_blocks = std::min(grid_blocks, options.max_grid_blocks);
  grid_blocks = std::max(cluster_size, (grid_blocks / cluster_size) * cluster_size);

  unsigned int* d_smids = nullptr;
  std::vector<unsigned int> h_smids(static_cast<size_t>(grid_blocks));
  check_cuda(cudaMalloc(&d_smids, sizeof(unsigned int) * grid_blocks), "cudaMalloc smid residency");

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid_blocks, 1, 1);
  config.blockDim = dim3(options.block_threads, 1, 1);
  config.dynamicSmemBytes = 0;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = cluster_size;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  cudaError_t err = cudaLaunchKernelEx(&config, smid_residency_kernel, d_smids);
  if (err != cudaSuccess) {
    cudaGetLastError();
    cudaFree(d_smids);
    print_unsupported("smid_residency", cluster_size, "launch_failed");
    return false;
  }
  check_cuda(cudaDeviceSynchronize(), "smid residency sync");
  check_cuda(cudaMemcpy(h_smids.data(), d_smids, sizeof(unsigned int) * grid_blocks, cudaMemcpyDeviceToHost), "cudaMemcpy smid residency");

  std::vector<unsigned int> unique_smids = h_smids;
  std::sort(unique_smids.begin(), unique_smids.end());
  unique_smids.erase(std::unique(unique_smids.begin(), unique_smids.end()), unique_smids.end());

  std::printf("{\"metric\":\"smid_residency\",\"cluster_size\":%d,\"supported\":true,\"grid_blocks\":%d,\"active_clusters_estimate\":%d,\"unique_smid_count\":%zu,\"smids\":[",
              cluster_size, grid_blocks, active_clusters, unique_smids.size());
  for (size_t i = 0; i < unique_smids.size(); ++i) {
    std::printf("%s%u", i == 0 ? "" : ",", unique_smids[i]);
  }
  std::printf("]}\n");

  cudaFree(d_smids);
  return true;
}

void print_usage(const char* prog) {
  std::printf("Usage: %s [options]\n", prog);
  std::printf("  --gpu-label=NAME                 Label stored in JSON output, default auto\n");
  std::printf("  --cluster-sizes=1,2,4,8,16       Cluster sizes to test\n");
  std::printf("  --repeats=N                      Timed repeats, default 30\n");
  std::printf("  --warmup=N                       Warmup repeats, default 5\n");
  std::printf("  --latency-iters=N                Dependent-load iterations, default 200000\n");
  std::printf("  --bandwidth-iters=N              Bandwidth kernel iterations, default 4096\n");
  std::printf("  --latency-elems=N                Per-CTA DSMEM pointer-chase elems, default 4096\n");
  std::printf("  --bandwidth-bytes=N              Per-CTA DSMEM buffer bytes, default 32768\n");
  std::printf("  --global-latency-elems=N         Global pointer-chase elems, default 67108864\n");
  std::printf("  --global-bandwidth-bytes=N       Global bandwidth bytes per iter, default 536870912\n");
  std::printf("  --max-grid-blocks=N              Cap bandwidth grid blocks, default auto\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    }
    options.repeats = parse_int(arg, "--repeats=", options.repeats);
    options.warmup = parse_int(arg, "--warmup=", options.warmup);
    options.latency_iters = parse_int(arg, "--latency-iters=", options.latency_iters);
    options.bandwidth_iters = parse_int(arg, "--bandwidth-iters=", options.bandwidth_iters);
    options.block_threads = parse_int(arg, "--block-threads=", options.block_threads);
    options.latency_elems = parse_int(arg, "--latency-elems=", options.latency_elems);
    options.bandwidth_bytes = parse_int(arg, "--bandwidth-bytes=", options.bandwidth_bytes);
    options.global_latency_elems = parse_size(arg, "--global-latency-elems=", options.global_latency_elems);
    options.global_bandwidth_bytes = parse_size(arg, "--global-bandwidth-bytes=", options.global_bandwidth_bytes);
    options.max_grid_blocks = parse_int(arg, "--max-grid-blocks=", options.max_grid_blocks);
    options.gpu_label = parse_str(arg, "--gpu-label=", options.gpu_label);
    options.cluster_sizes_text = parse_str(arg, "--cluster-sizes=", options.cluster_sizes_text);
  }

  if (options.latency_elems <= 0 || (options.latency_elems & (options.latency_elems - 1)) != 0) {
    std::fprintf(stderr, "--latency-elems must be a positive power of two\n");
    return 1;
  }
  if (options.bandwidth_bytes <= 0 || options.bandwidth_bytes % 16 != 0) {
    std::fprintf(stderr, "--bandwidth-bytes must be positive and divisible by 16\n");
    return 1;
  }

  const std::vector<int> cluster_sizes = parse_cluster_sizes(options.cluster_sizes_text);
  const std::string name = gpu_name();
  std::printf("{\"metric\":\"metadata\",\"gpu_label\":\"%s\",\"gpu_name\":\"%s\",\"sm_count\":%d,\"max_blocks_per_cluster_attr\":%d,\"sm_clock_ghz_attr\":%.6f}\n",
              options.gpu_label, name.c_str(), sm_count(), max_blocks_per_cluster(), sm_clock_ghz());

  run_global_latency(options);
  run_global_bandwidth(options);
  run_local_smem_bandwidth(options);

  for (int cluster_size : cluster_sizes) {
    print_active_sm(cluster_size, options);
    print_smid_residency(cluster_size, options);
    run_dsmem_latency(cluster_size, options);
    run_dsmem_bandwidth(cluster_size, options);
  }
  return 0;
}
