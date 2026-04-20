#include "gemm_reference_common.h"

#include <cstdlib>
#include <optional>
#include <random>

namespace {

std::optional<std::string_view> maybe_value(const std::string& arg,
                                            const std::string& prefix) {
  if (arg.rfind(prefix, 0) != 0) {
    return std::nullopt;
  }
  return std::string_view(arg).substr(prefix.size());
}

template <typename T>
T parse_number(std::string_view text, const char* name) {
  std::string buffer(text);
  std::istringstream iss(buffer);
  T value{};
  iss >> value;
  if (!iss || !iss.eof()) {
    std::ostringstream oss;
    oss << "Invalid value for " << name << ": " << buffer;
    throw std::runtime_error(oss.str());
  }
  return value;
}

bool parse_bool(std::string_view text, const char* name) {
  if (text == "1" || text == "true") {
    return true;
  }
  if (text == "0" || text == "false") {
    return false;
  }
  std::ostringstream oss;
  oss << "Invalid boolean for " << name << ": " << text;
  throw std::runtime_error(oss.str());
}

void fill_random(std::vector<float>& data, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (float& value : data) {
    value = dist(rng);
  }
}

void print_usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --m=<int>             Rows of A/C\n"
      << "  --n=<int>             Columns of B/C\n"
      << "  --k=<int>             Columns of A / rows of B\n"
      << "  --warmup=<int>        Warmup iterations per backend\n"
      << "  --iters=<int>         Timed iterations per backend\n"
      << "  --block-m=<int>       Thread-block rows (naive / simt expects 16)\n"
      << "  --block-n=<int>       Thread-block cols (naive / simt expects 16)\n"
      << "  --seed=<int>          RNG seed\n"
      << "  --alpha=<float>       GEMM alpha\n"
      << "  --beta=<float>        GEMM beta\n"
      << "  --atol=<float>        Correctness absolute tolerance\n"
      << "  --rtol=<float>        Correctness relative tolerance\n"
      << "  --backend=<all|naive|simt|cublas>\n"
      << "  --check=<0|1>         Compare custom kernels against cuBLAS\n"
      << "  --help                Show this message\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (auto value = maybe_value(arg, "--m=")) {
      options.m = parse_number<int>(*value, "m");
    } else if (auto value = maybe_value(arg, "--n=")) {
      options.n = parse_number<int>(*value, "n");
    } else if (auto value = maybe_value(arg, "--k=")) {
      options.k = parse_number<int>(*value, "k");
    } else if (auto value = maybe_value(arg, "--warmup=")) {
      options.warmup = parse_number<int>(*value, "warmup");
    } else if (auto value = maybe_value(arg, "--iters=")) {
      options.iters = parse_number<int>(*value, "iters");
    } else if (auto value = maybe_value(arg, "--block-m=")) {
      options.block_m = parse_number<int>(*value, "block-m");
    } else if (auto value = maybe_value(arg, "--block-n=")) {
      options.block_n = parse_number<int>(*value, "block-n");
    } else if (auto value = maybe_value(arg, "--seed=")) {
      options.seed = parse_number<int>(*value, "seed");
    } else if (auto value = maybe_value(arg, "--alpha=")) {
      options.alpha = parse_number<float>(*value, "alpha");
    } else if (auto value = maybe_value(arg, "--beta=")) {
      options.beta = parse_number<float>(*value, "beta");
    } else if (auto value = maybe_value(arg, "--atol=")) {
      options.atol = parse_number<float>(*value, "atol");
    } else if (auto value = maybe_value(arg, "--rtol=")) {
      options.rtol = parse_number<float>(*value, "rtol");
    } else if (auto value = maybe_value(arg, "--backend=")) {
      options.backend = std::string(*value);
    } else if (auto value = maybe_value(arg, "--check=")) {
      options.check = parse_bool(*value, "check");
    } else {
      std::ostringstream oss;
      oss << "Unknown option: " << arg;
      throw std::runtime_error(oss.str());
    }
  }

  if (options.m <= 0 || options.n <= 0 || options.k <= 0) {
    throw std::runtime_error("m, n and k must be positive");
  }
  if (options.warmup < 0 || options.iters <= 0) {
    throw std::runtime_error("warmup must be >= 0 and iters must be > 0");
  }
  if (options.block_m <= 0 || options.block_n <= 0) {
    throw std::runtime_error("block-m and block-n must be positive");
  }
  if (options.backend != "all" && options.backend != "naive" &&
      options.backend != "simt" && options.backend != "cublas") {
    throw std::runtime_error("backend must be one of: all, naive, simt, cublas");
  }
  return options;
}

void check_backend_against_reference(const Options& options,
                                     const char* backend,
                                     const std::vector<float>& reference,
                                     const std::vector<float>& actual) {
  const CompareStats compare =
      compare_outputs(reference, actual, options.atol, options.rtol);
  print_check_line(options, backend, compare);
  if (!compare.pass) {
    std::ostringstream oss;
    oss << "Correctness check failed for backend=" << backend;
    throw std::runtime_error(oss.str());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    print_device_info();

    std::cout << "CONFIG m=" << options.m
              << " n=" << options.n
              << " k=" << options.k
              << " warmup=" << options.warmup
              << " iters=" << options.iters
              << " block_m=" << options.block_m
              << " block_n=" << options.block_n
              << " alpha=" << options.alpha
              << " beta=" << options.beta
              << " atol=" << options.atol
              << " rtol=" << options.rtol
              << " backend=" << options.backend
              << " check=" << static_cast<int>(options.check)
              << '\n';

    const std::size_t a_elems =
        static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_elems =
        static_cast<std::size_t>(options.k) * options.n;
    const std::size_t c_elems =
        static_cast<std::size_t>(options.m) * options.n;
    const std::size_t c_bytes = c_elems * sizeof(float);

    std::vector<float> h_a(a_elems);
    std::vector<float> h_b(b_elems);
    std::vector<float> h_c_init(c_elems);
    std::vector<float> h_naive(c_elems, 0.0f);
    std::vector<float> h_simt(c_elems, 0.0f);
    std::vector<float> h_cublas(c_elems, 0.0f);

    fill_random(h_a, options.seed);
    fill_random(h_b, options.seed + 1);
    fill_random(h_c_init, options.seed + 2);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c_init = nullptr;
    float* d_naive = nullptr;
    float* d_simt = nullptr;
    float* d_cublas = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, a_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, b_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c_init, c_bytes));
    CHECK_CUDA(cudaMalloc(&d_naive, c_bytes));
    CHECK_CUDA(cudaMalloc(&d_simt, c_bytes));
    CHECK_CUDA(cudaMalloc(&d_cublas, c_bytes));

    CHECK_CUDA(cudaMemcpy(
        d_a, h_a.data(), a_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(
        d_b, h_b.data(), b_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_c_init, h_c_init.data(), c_bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    const bool need_reference_check =
        options.check &&
        (should_run_backend(options, "naive") || should_run_backend(options, "simt"));

    if (need_reference_check) {
      maybe_reset_output(d_cublas, d_c_init, c_bytes, options.beta);
      run_cublas_reference_once(handle, options, d_a, d_b, d_cublas);
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(
          cudaMemcpy(h_cublas.data(), d_cublas, c_bytes, cudaMemcpyDeviceToHost));

      if (should_run_backend(options, "naive")) {
        maybe_reset_output(d_naive, d_c_init, c_bytes, options.beta);
        run_naive_once(options, d_a, d_b, d_c_init, d_naive);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(
            h_naive.data(), d_naive, c_bytes, cudaMemcpyDeviceToHost));
        check_backend_against_reference(options, "naive", h_cublas, h_naive);
      }

      if (should_run_backend(options, "simt")) {
        maybe_reset_output(d_simt, d_c_init, c_bytes, options.beta);
        run_simt_cp_async_once(options, d_a, d_b, d_c_init, d_simt);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(
            h_simt.data(), d_simt, c_bytes, cudaMemcpyDeviceToHost));
        check_backend_against_reference(options, "simt", h_cublas, h_simt);
      }
    }

    if (should_run_backend(options, "naive")) {
      const TimingStats stats =
          benchmark_naive(options, d_a, d_b, d_c_init, d_naive, h_naive);
      print_result_line(options, "naive", stats);
    }

    if (should_run_backend(options, "simt")) {
      const TimingStats stats =
          benchmark_simt_cp_async(options, d_a, d_b, d_c_init, d_simt, h_simt);
      print_result_line(options, "simt", stats);
    }

    if (should_run_backend(options, "cublas")) {
      const TimingStats stats = benchmark_cublas_reference(
          options, handle, d_a, d_b, d_c_init, d_cublas, h_cublas);
      print_result_line(options, "cublas", stats);
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c_init));
    CHECK_CUDA(cudaFree(d_naive));
    CHECK_CUDA(cudaFree(d_simt));
    CHECK_CUDA(cudaFree(d_cublas));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR " << e.what() << '\n';
    return 1;
  }
}
