#include "gemm_reference_common.h"

#include <cuda_pipeline.h>

namespace {

constexpr int kSimtBlockM = 16;
constexpr int kSimtBlockN = 16;
constexpr int kSimtBlockK = 16;

__device__ inline void cp_async_or_zero(float* smem_ptr,
                                        const float* gmem_ptr,
                                        bool predicate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  if (predicate) {
    __pipeline_memcpy_async(smem_ptr, gmem_ptr, sizeof(float));
  } else {
    *smem_ptr = 0.0f;
  }
#else
  *smem_ptr = predicate ? *gmem_ptr : 0.0f;
#endif
}

__global__ void simt_gemm_cp_async_kernel(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          const float* __restrict__ c_in,
                                          float* __restrict__ d_out,
                                          int m,
                                          int n,
                                          int k,
                                          float alpha,
                                          float beta) {
  __shared__ float sA[2][kSimtBlockM][kSimtBlockK];
  __shared__ float sB[2][kSimtBlockK][kSimtBlockN];

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int global_row = blockIdx.y * kSimtBlockM + local_row;
  const int global_col = blockIdx.x * kSimtBlockN + local_col;
  const int num_k_tiles = (k + kSimtBlockK - 1) / kSimtBlockK;

  if (num_k_tiles == 0) {
    return;
  }

  const int first_k_base = 0;
  const int first_a_col = first_k_base + local_col;
  const int first_b_row = first_k_base + local_row;
  const bool first_a_valid = global_row < m && first_a_col < k;
  const bool first_b_valid = first_b_row < k && global_col < n;
  const float* first_a_ptr =
      first_a_valid ? (a + static_cast<std::size_t>(global_row) * k + first_a_col) : a;
  const float* first_b_ptr =
      first_b_valid ? (b + static_cast<std::size_t>(first_b_row) * n + global_col) : b;
  cp_async_or_zero(&sA[0][local_row][local_col], first_a_ptr, first_a_valid);
  cp_async_or_zero(&sB[0][local_row][local_col], first_b_ptr, first_b_valid);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  __pipeline_commit();
  __pipeline_wait_prior(0);
#endif
  __syncthreads();

  float acc = 0.0f;

  for (int tile_idx = 0; tile_idx < num_k_tiles; ++tile_idx) {
    const int current_stage = tile_idx & 1;
    const int next_tile_idx = tile_idx + 1;
    const int next_stage = current_stage ^ 1;

    if (next_tile_idx < num_k_tiles) {
      const int next_k_base = next_tile_idx * kSimtBlockK;
      const int next_a_col = next_k_base + local_col;
      const int next_b_row = next_k_base + local_row;
      const bool next_a_valid = global_row < m && next_a_col < k;
      const bool next_b_valid = next_b_row < k && global_col < n;
      const float* next_a_ptr =
          next_a_valid ? (a + static_cast<std::size_t>(global_row) * k + next_a_col) : a;
      const float* next_b_ptr =
          next_b_valid ? (b + static_cast<std::size_t>(next_b_row) * n + global_col) : b;
      cp_async_or_zero(&sA[next_stage][local_row][local_col], next_a_ptr, next_a_valid);
      cp_async_or_zero(&sB[next_stage][local_row][local_col], next_b_ptr, next_b_valid);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      __pipeline_commit();
#endif
    }

#pragma unroll
    for (int kk = 0; kk < kSimtBlockK; ++kk) {
      acc += sA[current_stage][local_row][kk] * sB[current_stage][kk][local_col];
    }

    if (next_tile_idx < num_k_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      __pipeline_wait_prior(0);
#endif
      __syncthreads();
    }
  }

  if (global_row < m && global_col < n) {
    const std::size_t c_idx = static_cast<std::size_t>(global_row) * n + global_col;
    d_out[c_idx] = alpha * acc + beta * c_in[c_idx];
  }
}

}  // namespace

void run_simt_cp_async_once(const Options& options,
                            const float* d_a,
                            const float* d_b,
                            const float* d_c_in,
                            float* d_out) {
  if (options.block_m != kSimtBlockM || options.block_n != kSimtBlockN) {
    std::ostringstream oss;
    oss << "simt backend currently requires --block-m=" << kSimtBlockM
        << " and --block-n=" << kSimtBlockN;
    throw std::runtime_error(oss.str());
  }

  const dim3 block(kSimtBlockN, kSimtBlockM);
  const dim3 grid((options.n + kSimtBlockN - 1) / kSimtBlockN,
                  (options.m + kSimtBlockM - 1) / kSimtBlockM);
  simt_gemm_cp_async_kernel<<<grid, block>>>(d_a,
                                             d_b,
                                             d_c_in,
                                             d_out,
                                             options.m,
                                             options.n,
                                             options.k,
                                             options.alpha,
                                             options.beta);
  CHECK_CUDA(cudaGetLastError());
}

TimingStats benchmark_simt_cp_async(const Options& options,
                                    const float* d_a,
                                    const float* d_b,
                                    const float* d_c_in,
                                    float* d_out,
                                    std::vector<float>& host_out) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(options.m) * options.n * sizeof(float);

  for (int i = 0; i < options.warmup; ++i) {
    maybe_reset_output(d_out, d_c_in, output_bytes, options.beta);
    run_simt_cp_async_once(options, d_a, d_b, d_c_in, d_out);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> times_ms;
  times_ms.reserve(options.iters);
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < options.iters; ++i) {
    maybe_reset_output(d_out, d_c_in, output_bytes, options.beta);
    CHECK_CUDA(cudaEventRecord(start));
    run_simt_cp_async_once(options, d_a, d_b, d_c_in, d_out);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    times_ms.push_back(elapsed);
  }

  CHECK_CUDA(cudaMemcpy(host_out.data(),
                        d_out,
                        output_bytes,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  return finalize_timing_stats(options, times_ms, host_out);
}
