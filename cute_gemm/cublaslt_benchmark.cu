#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                        \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA error: ") +                 \
                               cudaGetErrorString(status));                    \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                        \
    cublasStatus_t status = (expr);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error("cuBLASLt error: " + std::to_string(status));  \
    }                                                                         \
  } while (0)

namespace {

struct Args {
  int m = 4096;
  int n = 2048;
  int k = 512;
  int warmup = 20;
  int iters = 100;
  int requested_algos = 32;
  size_t workspace_bytes = 64ull * 1024ull * 1024ull;
};

std::tuple<int, int, int> parse_mnk(const std::string& text) {
  size_t first = text.find(',');
  size_t second = text.find(',', first == std::string::npos ? first : first + 1);
  if (first == std::string::npos || second == std::string::npos) {
    throw std::runtime_error("--mnk must be M,N,K");
  }
  return {
      std::stoi(text.substr(0, first)),
      std::stoi(text.substr(first + 1, second - first - 1)),
      std::stoi(text.substr(second + 1)),
  };
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
      }
      return argv[++i];
    };

    if (key == "--mnk") {
      std::tie(args.m, args.n, args.k) = parse_mnk(require_value());
    } else if (key == "--warmup") {
      args.warmup = std::stoi(require_value());
    } else if (key == "--iters") {
      args.iters = std::stoi(require_value());
    } else if (key == "--algos") {
      args.requested_algos = std::stoi(require_value());
    } else if (key == "--workspace-mb") {
      args.workspace_bytes = static_cast<size_t>(std::stoull(require_value())) * 1024ull * 1024ull;
    } else if (key == "--help") {
      std::cout << "Usage: cublaslt_benchmark --mnk M,N,K [--warmup N] [--iters N] "
                   "[--algos N] [--workspace-mb N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  return args;
}

std::vector<__half> make_half_data(size_t size) {
  std::vector<__half> data(size);
  for (size_t i = 0; i < size; ++i) {
    int value = static_cast<int>(i % 5) - 2;
    data[i] = __float2half(static_cast<float>(value));
  }
  return data;
}

void set_row_major(cublasLtMatrixLayout_t layout) {
  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
      layout,
      CUBLASLT_MATRIX_LAYOUT_ORDER,
      &order,
      sizeof(order)));
}

float time_algo(
    cublasLtHandle_t lt,
    cublasLtMatmulDesc_t op_desc,
    const float* alpha,
    const __half* d_a,
    cublasLtMatrixLayout_t a_desc,
    const __half* d_b,
    cublasLtMatrixLayout_t b_desc,
    const float* beta,
    __half* d_c,
    cublasLtMatrixLayout_t c_desc,
    __half* d_d,
    cublasLtMatrixLayout_t d_desc,
    const cublasLtMatmulAlgo_t& algo,
    void* workspace,
    size_t workspace_bytes,
    int warmup,
    int iters) {
  for (int i = 0; i < warmup; ++i) {
    CUBLAS_CHECK(cublasLtMatmul(
        lt,
        op_desc,
        alpha,
        d_a,
        a_desc,
        d_b,
        b_desc,
        beta,
        d_c,
        c_desc,
        d_d,
        d_desc,
        &algo,
        workspace,
        workspace_bytes,
        0));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUBLAS_CHECK(cublasLtMatmul(
        lt,
        op_desc,
        alpha,
        d_a,
        a_desc,
        d_b,
        b_desc,
        beta,
        d_c,
        c_desc,
        d_d,
        d_desc,
        &algo,
        workspace,
        workspace_bytes,
        0));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / static_cast<float>(iters);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);

    std::vector<__half> h_a = make_half_data(static_cast<size_t>(args.m) * args.k);
    std::vector<__half> h_b = make_half_data(static_cast<size_t>(args.n) * args.k);

    __half* d_a = nullptr;
    __half* d_b = nullptr;
    __half* d_c = nullptr;
    void* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(args.m) * args.n * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&workspace, args.workspace_bytes));

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    cublasLtMatmulDesc_t op_desc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, args.m, args.k, args.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, args.n, args.k, args.k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, args.m, args.n, args.n));
    set_row_major(a_desc);
    set_row_major(b_desc);
    set_row_major(c_desc);

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &args.workspace_bytes,
        sizeof(args.workspace_bytes)));

    std::vector<cublasLtMatmulHeuristicResult_t> heuristic(args.requested_algos);
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        lt,
        op_desc,
        a_desc,
        b_desc,
        c_desc,
        c_desc,
        pref,
        args.requested_algos,
        heuristic.data(),
        &returned));
    if (returned == 0) {
      throw std::runtime_error("cuBLASLt returned no algorithms");
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    float best_ms = 1.0e30f;
    int best_algo = -1;

    for (int i = 0; i < returned; ++i) {
      float ms = time_algo(
          lt,
          op_desc,
          &alpha,
          d_a,
          a_desc,
          d_b,
          b_desc,
          &beta,
          d_c,
          c_desc,
          d_c,
          c_desc,
          heuristic[i].algo,
          workspace,
          args.workspace_bytes,
          args.warmup,
          args.iters);
      if (ms < best_ms) {
        best_ms = ms;
        best_algo = i;
      }
      std::cout << "ALGO index=" << i << " ms=" << std::fixed << std::setprecision(6) << ms << "\n";
    }

    double flops = 2.0 * static_cast<double>(args.m) * args.n * args.k;
    double tflops = flops / (static_cast<double>(best_ms) * 1.0e-3) / 1.0e12;
    std::cout << "RESULT benchmark=cublaslt"
              << " m=" << args.m
              << " n=" << args.n
              << " k=" << args.k
              << " best_algo_index=" << best_algo
              << " ms=" << std::fixed << std::setprecision(6) << best_ms
              << " tflops=" << std::fixed << std::setprecision(3) << tflops
              << " returned_algos=" << returned
              << " workspace_mb=" << (args.workspace_bytes / 1024 / 1024)
              << "\n";

    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
    CUBLAS_CHECK(cublasLtDestroy(lt));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << "\n";
    return 1;
  }
}
