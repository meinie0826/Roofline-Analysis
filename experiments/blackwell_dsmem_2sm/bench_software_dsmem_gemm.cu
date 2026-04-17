#include "common.h"

#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <vector>

namespace cg = cooperative_groups;
using namespace blackwell_dsmem_2sm;

constexpr int kSoftwareBlockThreads = 128;

struct SoftwareGemmResult {
  float elapsed_ms = 0.0f;
  float gflops = 0.0f;
  float bytes_b_share = 0.0f;
  float checksum = 0.0f;
};

template <int TileN, int Stages, bool RemoteB>
__global__ void software_dsmem_gemm_kernel(const half* A, const half* B, float* D, int lda, int ldb, int ldd, int k_total) {
  extern __shared__ __align__(16) unsigned char storage_raw[];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int cluster_id_x = blockIdx.x / 2;
  const int m_group = blockIdx.y;
  const int row_tile = m_group * 2 + rank;
  const int row_base = row_tile * 16;
  const int col_base = cluster_id_x * TileN;

  half* a_stage = reinterpret_cast<half*>(storage_raw);
  half* b_stage = a_stage + Stages * 16 * 16;
  const int b_stage_elems = 16 * TileN;

  for (int stage = 0; stage < Stages; ++stage) {
    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
      a_stage[stage * 16 * 16 + i] = __float2half(0.0f);
    }
    for (int i = threadIdx.x; i < b_stage_elems; i += blockDim.x) {
      b_stage[stage * b_stage_elems + i] = __float2half(0.0f);
    }
  }
  cluster.sync();

  constexpr int TileElems = 16 * TileN;
  constexpr int OutputsPerThread = (TileElems + kSoftwareBlockThreads - 1) / kSoftwareBlockThreads;
  float acc[OutputsPerThread];
  int out_idx[OutputsPerThread];

  #pragma unroll
  for (int slot = 0; slot < OutputsPerThread; ++slot) {
    out_idx[slot] = threadIdx.x + slot * blockDim.x;
    acc[slot] = 0.0f;
  }

  const int k_tiles = k_total / 16;
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    const int stage = k_tile % Stages;
    half* a_buf = a_stage + stage * 16 * 16;
    half* b_buf_local = b_stage + stage * b_stage_elems;

    for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
      const int r = i / 16;
      const int c = i % 16;
      a_buf[i] = A[(row_base + r) * lda + k_tile * 16 + c];
    }

    if (!RemoteB || rank == 0) {
      for (int i = threadIdx.x; i < b_stage_elems; i += blockDim.x) {
        const int r = i / TileN;
        const int c = i % TileN;
        b_buf_local[i] = B[(k_tile * 16 + r) * ldb + col_base + c];
      }
    }
    cluster.sync();

    half* b_buf = b_buf_local;
    if constexpr (RemoteB) {
      if (rank == 1) {
        b_buf = reinterpret_cast<half*>(cluster.map_shared_rank(b_buf_local, 0));
      }
    }

    #pragma unroll
    for (int slot = 0; slot < OutputsPerThread; ++slot) {
      const int linear_idx = out_idx[slot];
      if (linear_idx < TileElems) {
        const int row = linear_idx / TileN;
        const int col = linear_idx % TileN;
        float sum = acc[slot];
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
          sum += __half2float(a_buf[row * 16 + kk]) * __half2float(b_buf[kk * TileN + col]);
        }
        acc[slot] = sum;
      }
    }
    cluster.sync();
  }

  #pragma unroll
  for (int slot = 0; slot < OutputsPerThread; ++slot) {
    const int linear_idx = out_idx[slot];
    if (linear_idx < TileElems) {
      const int row = linear_idx / TileN;
      const int col = linear_idx % TileN;
      D[(row_base + row) * ldd + col_base + col] = acc[slot];
    }
  }
}

template <int TileN, int Stages, bool RemoteB>
void launch_software_gemm(const half* dA, const half* dB, float* dD, const GemmOptions& options) {
  dim3 block(kSoftwareBlockThreads, 1, 1);
  dim3 grid((options.n / TileN) * 2, options.m / 32, 1);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = Stages * (16 * 16 + 16 * TileN) * static_cast<int>(sizeof(half)) + 128;
  config.stream = nullptr;

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = 2;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;
  config.attrs = &attr;
  config.numAttrs = 1;

  int lda = options.k;
  int ldb = options.n;
  int ldd = options.n;
  int k_total = options.k;
  prepare_cluster_kernel(software_dsmem_gemm_kernel<TileN, Stages, RemoteB>, config.dynamicSmemBytes);
  check_cuda(
      cudaLaunchKernelEx(
          &config,
          software_dsmem_gemm_kernel<TileN, Stages, RemoteB>,
          dA,
          dB,
          dD,
          lda,
          ldb,
          ldd,
          k_total),
      "cudaLaunchKernelEx software gemm");
}

template <int TileN, int Stages>
void run_software_case(const half* dA, const half* dB, float* dD, const GemmOptions& options) {
  if (std::strcmp(options.mode, "remote") == 0) {
    launch_software_gemm<TileN, Stages, true>(dA, dB, dD, options);
  } else {
    launch_software_gemm<TileN, Stages, false>(dA, dB, dD, options);
  }
}

int main(int argc, char** argv) {
  GemmOptions options;
  parse_gemm_options(argc, argv, &options);

  if (!is_valid_tile_n(options.tile_n) || !is_valid_stages(options.stages)) {
    std::fprintf(stderr, "Unsupported tile_n=%d or stages=%d\n", options.tile_n, options.stages);
    return 1;
  }
  if (options.m % 32 != 0 || options.n % options.tile_n != 0 || options.k % 16 != 0) {
    std::fprintf(stderr, "Require m %% 32 == 0, n %% tile_n == 0, k %% 16 == 0\n");
    return 1;
  }
  if (std::strcmp(options.mode, "local") != 0 && std::strcmp(options.mode, "remote") != 0) {
    std::fprintf(stderr, "mode must be local or remote\n");
    return 1;
  }

  std::vector<half> hA(options.m * options.k);
  std::vector<half> hB(options.k * options.n);
  for (int i = 0; i < options.m * options.k; ++i) {
    hA[i] = __float2half(static_cast<float>((i % 7) - 3));
  }
  for (int i = 0; i < options.k * options.n; ++i) {
    hB[i] = __float2half(static_cast<float>((i % 5) - 2));
  }

  half* dA = alloc_device<half>(hA.size());
  half* dB = alloc_device<half>(hB.size());
  float* dD = alloc_device<float>(static_cast<std::size_t>(options.m) * options.n);
  copy_to_device(dA, hA.data(), hA.size());
  copy_to_device(dB, hB.data(), hB.size());
  check_cuda(cudaMemset(dD, 0, sizeof(float) * options.m * options.n), "cudaMemset D");

  std::printf(
      "CONFIG benchmark=bench_software_dsmem_gemm mode=%s m=%d n=%d k=%d tile_n=%d stages=%d repeats=%d warmup_repeats=%d\n",
      options.mode,
      options.m,
      options.n,
      options.k,
      options.tile_n,
      options.stages,
      options.repeats,
      options.warmup_repeats);

  for (int i = 0; i < options.warmup_repeats; ++i) {
    if (options.tile_n == 64 && options.stages == 1) run_software_case<64, 1>(dA, dB, dD, options);
    if (options.tile_n == 64 && options.stages == 2) run_software_case<64, 2>(dA, dB, dD, options);
    if (options.tile_n == 64 && options.stages == 4) run_software_case<64, 4>(dA, dB, dD, options);
    if (options.tile_n == 128 && options.stages == 1) run_software_case<128, 1>(dA, dB, dD, options);
    if (options.tile_n == 128 && options.stages == 2) run_software_case<128, 2>(dA, dB, dD, options);
    if (options.tile_n == 128 && options.stages == 4) run_software_case<128, 4>(dA, dB, dD, options);
    if (options.tile_n == 256 && options.stages == 1) run_software_case<256, 1>(dA, dB, dD, options);
    if (options.tile_n == 256 && options.stages == 2) run_software_case<256, 2>(dA, dB, dD, options);
    if (options.tile_n == 256 && options.stages == 4) run_software_case<256, 4>(dA, dB, dD, options);
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
  }

  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  std::vector<float> hD(static_cast<std::size_t>(options.m) * options.n);
  const double total_flops = 2.0 * static_cast<double>(options.m) * static_cast<double>(options.n) * static_cast<double>(options.k);
  const double bytes_b_share = static_cast<double>(options.tile_n) * 16.0 * sizeof(half);

  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    if (options.tile_n == 64 && options.stages == 1) run_software_case<64, 1>(dA, dB, dD, options);
    if (options.tile_n == 64 && options.stages == 2) run_software_case<64, 2>(dA, dB, dD, options);
    if (options.tile_n == 64 && options.stages == 4) run_software_case<64, 4>(dA, dB, dD, options);
    if (options.tile_n == 128 && options.stages == 1) run_software_case<128, 1>(dA, dB, dD, options);
    if (options.tile_n == 128 && options.stages == 2) run_software_case<128, 2>(dA, dB, dD, options);
    if (options.tile_n == 128 && options.stages == 4) run_software_case<128, 4>(dA, dB, dD, options);
    if (options.tile_n == 256 && options.stages == 1) run_software_case<256, 1>(dA, dB, dD, options);
    if (options.tile_n == 256 && options.stages == 2) run_software_case<256, 2>(dA, dB, dD, options);
    if (options.tile_n == 256 && options.stages == 4) run_software_case<256, 4>(dA, dB, dD, options);
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

    copy_to_host(hD.data(), dD, hD.size());
    double checksum = 0.0;
    for (int i = 0; i < options.n && i < 32; ++i) {
      checksum += static_cast<double>(hD[i]);
    }

    const double ms = elapsed_ms(start, stop);
    const double gflops = ms > 0.0 ? total_flops / (ms * 1.0e6) : 0.0;
    std::printf(
        "RESULT benchmark=bench_software_dsmem_gemm mode=%s repeat=%d m=%d n=%d k=%d tile_n=%d stages=%d elapsed_ms=%.6f gflops=%.4f bytes_b_share=%.0f checksum=%.6f\n",
        options.mode,
        repeat,
        options.m,
        options.n,
        options.k,
        options.tile_n,
        options.stages,
        ms,
        gflops,
        bytes_b_share,
        checksum);
  }

  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
  check_cuda(cudaFree(dA), "cudaFree A");
  check_cuda(cudaFree(dB), "cudaFree B");
  check_cuda(cudaFree(dD), "cudaFree D");
  return 0;
}
