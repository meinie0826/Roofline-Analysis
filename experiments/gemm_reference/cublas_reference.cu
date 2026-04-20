#include "gemm_reference_common.h"

void run_cublas_reference_once(cublasHandle_t handle,
                               const Options& options,
                               const float* d_a,
                               const float* d_b,
                               float* d_out) {
  CHECK_CUBLAS(cublasSgemm(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           options.n,
                           options.m,
                           options.k,
                           &options.alpha,
                           d_b,
                           options.n,
                           d_a,
                           options.k,
                           &options.beta,
                           d_out,
                           options.n));
}

TimingStats benchmark_cublas_reference(const Options& options,
                                       cublasHandle_t handle,
                                       const float* d_a,
                                       const float* d_b,
                                       const float* d_c_in,
                                       float* d_out,
                                       std::vector<float>& host_out) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(options.m) * options.n * sizeof(float);

  for (int i = 0; i < options.warmup; ++i) {
    maybe_reset_output(d_out, d_c_in, output_bytes, options.beta);
    run_cublas_reference_once(handle, options, d_a, d_b, d_out);
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
    run_cublas_reference_once(handle, options, d_a, d_b, d_out);
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
