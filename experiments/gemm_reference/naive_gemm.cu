#include "gemm_reference_common.h"

namespace {

__global__ void naive_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  const float* __restrict__ c_in,
                                  float* __restrict__ d_out,
                                  int m,
                                  int n,
                                  int k,
                                  float alpha,
                                  float beta) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    const std::size_t a_idx = static_cast<std::size_t>(row) * k + kk;
    const std::size_t b_idx = static_cast<std::size_t>(kk) * n + col;
    acc += a[a_idx] * b[b_idx];
  }

  const std::size_t c_idx = static_cast<std::size_t>(row) * n + col;
  d_out[c_idx] = alpha * acc + beta * c_in[c_idx];
}

}  // namespace

void run_naive_once(const Options& options,
                    const float* d_a,
                    const float* d_b,
                    const float* d_c_in,
                    float* d_out) {
  const dim3 block(options.block_n, options.block_m);
  const dim3 grid((options.n + block.x - 1) / block.x,
                  (options.m + block.y - 1) / block.y);
  naive_gemm_kernel<<<grid, block>>>(d_a,
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

TimingStats benchmark_naive(const Options& options,
                            const float* d_a,
                            const float* d_b,
                            const float* d_c_in,
                            float* d_out,
                            std::vector<float>& host_out) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(options.m) * options.n * sizeof(float);

  for (int i = 0; i < options.warmup; ++i) {
    maybe_reset_output(d_out, d_c_in, output_bytes, options.beta);
    run_naive_once(options, d_a, d_b, d_c_in, d_out);
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
    run_naive_once(options, d_a, d_b, d_c_in, d_out);
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
