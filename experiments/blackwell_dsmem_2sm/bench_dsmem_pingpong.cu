#include "common.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

__global__ void dsmem_pingpong_kernel(PingPongResult* out, int iters) {
  extern __shared__ __align__(16) int smem_raw[];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());

  volatile int* my_flag = reinterpret_cast<volatile int*>(smem_raw);
  volatile int* peer_flag = reinterpret_cast<volatile int*>(cluster.map_shared_rank(reinterpret_cast<int*>(smem_raw), rank == 0 ? 1 : 0));

  if (threadIdx.x == 0) {
    *my_flag = 0;
  }
  cluster.sync();

  if (threadIdx.x == 0) {
    if (rank == 0) {
      unsigned long long start = clock64();
      for (int i = 1; i <= iters; ++i) {
        *peer_flag = i;
        __threadfence_cluster();
        while (*my_flag != i) {
        }
      }
      unsigned long long stop = clock64();
      out->cycles = stop - start;
    } else {
      for (int i = 1; i <= iters; ++i) {
        while (*my_flag != i) {
        }
        *peer_flag = i;
        __threadfence_cluster();
      }
    }
  }
}

void launch_pingpong(PingPongResult* d_result, const PingPongOptions& options) {
  dim3 block(32, 1, 1);
  dim3 grid(options.cluster_dim_x, 1, 1);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 64;
  config.stream = nullptr;

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = dim3(options.cluster_dim_x, 1, 1);
  config.attrs = &attr;
  config.numAttrs = 1;

  void* args[] = {&d_result, const_cast<int*>(&options.iters)};
  check_cuda(cudaLaunchKernelEx(&config, dsmem_pingpong_kernel, args), "cudaLaunchKernelEx pingpong");
}

int main(int argc, char** argv) {
  PingPongOptions options;
  parse_pingpong_options(argc, argv, &options);

  if (options.cluster_dim_x != 2) {
    std::fprintf(stderr, "This benchmark currently expects --cluster-dim-x=2\n");
    return 1;
  }

  print_pingpong_header(options);
  const double sm_clock_ghz = query_sm_clock_ghz();
  PingPongResult* d_result = alloc_device<PingPongResult>(1);

  for (int i = 0; i < options.warmup_repeats; ++i) {
    launch_pingpong(d_result, options);
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
  }

  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    launch_pingpong(d_result, options);
    check_cuda(cudaDeviceSynchronize(), "benchmark synchronize");
    PingPongResult result{};
    copy_to_host(&result, d_result, 1);
    print_pingpong_result(options, repeat, sm_clock_ghz, result);
  }

  check_cuda(cudaFree(d_result), "cudaFree");
  return 0;
}
