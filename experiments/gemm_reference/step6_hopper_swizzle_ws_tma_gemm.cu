#include "gemm_reference_common.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace wmma = nvcuda::wmma;

#ifndef CUBLAS_COMPUTE_32F_FAST_16BF
#define CUBLAS_COMPUTE_32F_FAST_16BF CUBLAS_COMPUTE_32F
#endif

constexpr int kWarpSize = 32;
constexpr int kWarpsPerWarpGroup = 4;
constexpr int kProducerWarpGroups = 1;
constexpr int kConsumerWarpGroups = 2;
constexpr int kProducerWarps = kProducerWarpGroups * kWarpsPerWarpGroup;
constexpr int kConsumerWarps = kConsumerWarpGroups * kWarpsPerWarpGroup;
constexpr int kThreads = (kProducerWarps + kConsumerWarps) * kWarpSize;

constexpr int kStages = 4;
constexpr int kWarpTileM = 16;
constexpr int kWarpTileN = 16;
constexpr int kWarpTileK = 16;
constexpr int kWarpsPerBlockM = 4;
constexpr int kWarpsPerBlockN = 2;
constexpr int kTileM = kWarpsPerBlockM * kWarpTileM;
constexpr int kTileN = kWarpsPerBlockN * kWarpTileN;
constexpr int kL2BlockSwizzleGroupM = 8;
constexpr int kSharedWarpSlotSwizzleMask = 0x1;
constexpr int kWarpScratchALd = kWarpTileK + 8;
constexpr int kWarpScratchBLd = kWarpTileK + 8;

template <typename T, typename U>
__host__ __device__ constexpr auto align_up_local(T x, U boundary) {
  return (x + boundary - 1) & ~(boundary - 1);
}

constexpr std::size_t kStageAElements = kTileM * kWarpTileK;
constexpr std::size_t kStageBElements = kTileN * kWarpTileK;
constexpr std::size_t kSharedABytes =
    static_cast<std::size_t>(kStages) * kStageAElements * sizeof(nv_bfloat16);
constexpr std::size_t kSharedBOffsetBytes = align_up_local(kSharedABytes, 128);
constexpr std::size_t kSharedBBytes =
    static_cast<std::size_t>(kStages) * kStageBElements * sizeof(nv_bfloat16);
constexpr std::size_t kFilledBarrierOffsetBytes =
    align_up_local(kSharedBOffsetBytes + kSharedBBytes, alignof(uint64_t));
constexpr std::size_t kReadyBarrierOffsetBytes =
    align_up_local(kFilledBarrierOffsetBytes + kStages * sizeof(uint64_t),
                   alignof(uint64_t));
constexpr std::size_t kSharedScratchAOffsetBytes =
    align_up_local(kReadyBarrierOffsetBytes + kStages * sizeof(uint64_t), 128);
constexpr std::size_t kWarpScratchAElements = kWarpTileM * kWarpScratchALd;
constexpr std::size_t kSharedScratchABytes =
    static_cast<std::size_t>(kStages) * kConsumerWarps * kWarpScratchAElements *
    sizeof(nv_bfloat16);
constexpr std::size_t kSharedScratchBOffsetBytes =
    align_up_local(kSharedScratchAOffsetBytes + kSharedScratchABytes, 128);
constexpr std::size_t kWarpScratchBElements = kWarpTileN * kWarpScratchBLd;
constexpr std::size_t kSharedScratchBBytes =
    static_cast<std::size_t>(kStages) * kConsumerWarps * kWarpScratchBElements *
    sizeof(nv_bfloat16);
constexpr std::size_t kStep6SmemBytes =
    align_up_local(kSharedScratchBOffsetBytes + kSharedScratchBBytes, 128);

struct Step6Options {
  int m = 128;
  int n = 64;
  int k = 32;
  int warmup = 5;
  int iters = 20;
  int seed = 2026;
};

void check_cu(CUresult status, const char* expr, const char* file, int line) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* name = "unknown";
  const char* desc = "unknown";
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &desc);
  std::ostringstream oss;
  oss << "Driver API error at " << file << ":" << line
      << " for " << expr << ": " << name << " (" << desc << ")";
  throw std::runtime_error(oss.str());
}

#define CHECK_CU(expr) check_cu((expr), #expr, __FILE__, __LINE__)

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

void print_usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --m=<int>        GEMM M, multiple of 64\n"
      << "  --n=<int>        GEMM N, multiple of 32\n"
      << "  --k=<int>        GEMM K, multiple of 16\n"
      << "  --warmup=<int>   Warmup iterations\n"
      << "  --iters=<int>    Timed iterations\n"
      << "  --seed=<int>     Deterministic exact init seed\n";
}

Step6Options parse_options(int argc, char** argv) {
  Step6Options options;
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
    } else if (auto value = maybe_value(arg, "--seed=")) {
      options.seed = parse_number<int>(*value, "seed");
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
  if (options.m % kTileM != 0 || options.n % kTileN != 0 ||
      options.k % kWarpTileK != 0) {
    throw std::runtime_error(
        "step6 hopper swizzle ws tma currently requires m multiple of 64, n multiple of 32, k multiple of 16");
  }
  return options;
}

void initialize_exact_tensor(std::vector<nv_bfloat16>& tensor, int seed) {
  for (std::size_t i = 0; i < tensor.size(); ++i) {
    int value = ((static_cast<int>(i) * 17 + seed * 13) % 5) - 2;
    tensor[i] = __float2bfloat16(static_cast<float>(value));
  }
}

void run_cublas_reference(cublasHandle_t handle,
                          const Step6Options& options,
                          const nv_bfloat16* d_a,
                          const nv_bfloat16* d_b,
                          float* d_d,
                          cublasComputeType_t compute_type) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  CHECK_CUBLAS(cublasGemmEx(handle,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            options.n,
                            options.m,
                            options.k,
                            &alpha,
                            d_b,
                            CUDA_R_16BF,
                            options.k,
                            d_a,
                            CUDA_R_16BF,
                            options.k,
                            &beta,
                            d_d,
                            CUDA_R_32F,
                            options.n,
                            compute_type,
                            CUBLAS_GEMM_DEFAULT));
}

void create_tensor_map_2d(CUtensorMap* tensor_map,
                          const nv_bfloat16* ptr,
                          int rows,
                          int cols,
                          int tile_rows,
                          int tile_cols) {
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {static_cast<uint64_t>(cols),
                                static_cast<uint64_t>(rows)};
  uint64_t global_stride[kRank - 1] = {
      static_cast<uint64_t>(cols) * sizeof(nv_bfloat16)};
  uint32_t box_dim[kRank] = {static_cast<uint32_t>(tile_cols),
                             static_cast<uint32_t>(tile_rows)};
  uint32_t elem_stride[kRank] = {1u, 1u};

  CHECK_CU(cuTensorMapEncodeTiled(tensor_map,
                                  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                  kRank,
                                  const_cast<nv_bfloat16*>(ptr),
                                  global_dim,
                                  global_stride,
                                  box_dim,
                                  elem_stride,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_NONE,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

__device__ __forceinline__ void mbarrier_init_shared(uint32_t addr,
                                                     uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :
               : "r"(addr), "r"(count)
               : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_shared(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :
               : "r"(addr)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint32_t addr,
                                                          uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(addr), "r"(tx_bytes)
               : "memory");
}

__device__ __forceinline__ void mbarrier_wait_parity(uint32_t addr,
                                                     uint32_t phase) {
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t"
        "}"
        : "=r"(done)
        : "r"(addr), "r"(phase), "r"(0x989680)
        : "memory");
  }
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst,
                                            const void* tensor_map,
                                            int coord_k,
                                            int coord_row,
                                            uint32_t mbarrier_addr) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%3, %4}], [%2];"
      :
      : "r"(dst), "l"(tensor_map), "r"(mbarrier_addr), "r"(coord_k),
        "r"(coord_row)
      : "memory");
}

__device__ __forceinline__ int consumer_slot_swizzle(int stage,
                                                     int consumer_warp_id) {
  return consumer_warp_id ^ (stage & kSharedWarpSlotSwizzleMask);
}

__device__ __forceinline__ void compute_l2_swizzled_tile(int pid,
                                                         int grid_m_tiles,
                                                         int grid_n_tiles,
                                                         int& tile_m,
                                                         int& tile_n) {
  const int group_m = min(kL2BlockSwizzleGroupM, grid_m_tiles);
  const int tiles_per_group = group_m * grid_n_tiles;
  const int group_id = pid / tiles_per_group;
  const int first_tile_m = group_id * group_m;
  const int current_group_m = min(grid_m_tiles - first_tile_m, group_m);
  const int pid_in_group = pid % tiles_per_group;
  tile_m = first_tile_m + (pid_in_group % current_group_m);
  tile_n = pid_in_group / current_group_m;
}

__device__ __forceinline__ void warp_copy_a_to_swizzled_smem(
    nv_bfloat16* dst,
    const nv_bfloat16* src,
    int lane_id) {
  for (int idx = lane_id; idx < kWarpTileM * kWarpTileK; idx += kWarpSize) {
    const int row = idx / kWarpTileK;
    const int col = idx % kWarpTileK;
    dst[row * kWarpScratchALd + col] = src[row * kWarpTileK + col];
  }
}

__device__ __forceinline__ void warp_copy_b_to_swizzled_smem(
    nv_bfloat16* dst,
    const nv_bfloat16* src,
    int lane_id) {
  for (int idx = lane_id; idx < kWarpTileN * kWarpTileK; idx += kWarpSize) {
    const int n = idx / kWarpTileK;
    const int k = idx % kWarpTileK;
    dst[k + n * kWarpScratchBLd] = src[n * kWarpTileK + k];
  }
}

__global__ __launch_bounds__(kThreads) void step6_hopper_swizzle_ws_tma_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    float* __restrict__ D,
    int M,
    int N,
    int K) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto* sA_tma = reinterpret_cast<nv_bfloat16*>(smem_raw);
  auto* sB_tma =
      reinterpret_cast<nv_bfloat16*>(smem_raw + kSharedBOffsetBytes);
  auto* filled =
      reinterpret_cast<uint64_t*>(smem_raw + kFilledBarrierOffsetBytes);
  auto* ready =
      reinterpret_cast<uint64_t*>(smem_raw + kReadyBarrierOffsetBytes);
  // Keep TMA landing buffers linear for correctness, then remap into a
  // per-warp padded scratch layout before WMMA loads.
  auto* sA_swizzled =
      reinterpret_cast<nv_bfloat16*>(smem_raw + kSharedScratchAOffsetBytes);
  auto* sB_swizzled =
      reinterpret_cast<nv_bfloat16*>(smem_raw + kSharedScratchBOffsetBytes);

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const bool is_producer = (warp_id < kProducerWarps);
  const bool is_consumer = !is_producer;
  const int consumer_warp_id = warp_id - kProducerWarps;
  const int grid_m_tiles = M / kTileM;
  const int grid_n_tiles = N / kTileN;
  int tile_m_idx = 0;
  int tile_n_idx = 0;
  compute_l2_swizzled_tile(static_cast<int>(blockIdx.x), grid_m_tiles,
                           grid_n_tiles, tile_m_idx, tile_n_idx);
  const int block_m = tile_m_idx * kTileM;
  const int block_n = tile_n_idx * kTileN;

  if (tid == 0) {
    for (int stage = 0; stage < kStages; ++stage) {
      mbarrier_init_shared(
          static_cast<uint32_t>(__cvta_generic_to_shared(&filled[stage])), 1);
      mbarrier_init_shared(
          static_cast<uint32_t>(__cvta_generic_to_shared(&ready[stage])),
          kConsumerWarps);
    }
  }
  __syncthreads();
  fence_mbarrier_init();
  __syncthreads();

  uint32_t filled_phase[kStages] = {0, 0};
  uint32_t ready_phase[kStages] = {0, 0};
  const int k_tiles = K / kWarpTileK;

  if (tid == 0) {
    const int prologue_tiles = min(kStages, k_tiles);
    for (int tile_idx = 0; tile_idx < prologue_tiles; ++tile_idx) {
      const int stage = tile_idx % kStages;
      nv_bfloat16* stage_a = sA_tma + stage * kStageAElements;
      nv_bfloat16* stage_b = sB_tma + stage * kStageBElements;
      const uint32_t filled_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(&filled[stage]));
      const uint32_t stage_a_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(stage_a));
      const uint32_t stage_b_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(stage_b));
      mbarrier_arrive_expect_tx(
          filled_addr,
          static_cast<uint32_t>((kStageAElements + kStageBElements) *
                                sizeof(nv_bfloat16)));
      tma_load_2d(stage_a_addr, &A_tmap, tile_idx * kWarpTileK, block_m,
                  filled_addr);
      tma_load_2d(stage_b_addr, &B_tmap, tile_idx * kWarpTileK, block_n,
                  filled_addr);
    }
  }

  wmma::fragment<wmma::accumulator,
                 kWarpTileM,
                 kWarpTileN,
                 kWarpTileK,
                 float>
      acc_frag;
  if (is_consumer) {
    wmma::fill_fragment(acc_frag, 0.0f);
  }

  for (int tile_idx = 0; tile_idx < k_tiles; ++tile_idx) {
    const int stage = tile_idx % kStages;
    const int future_tile = tile_idx + kStages;

    if (is_consumer) {
      const uint32_t filled_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(&filled[stage]));
      mbarrier_wait_parity(filled_addr, filled_phase[stage]);
      filled_phase[stage] ^= 1;

      const int warp_m = consumer_warp_id / kWarpsPerBlockN;
      const int warp_n = consumer_warp_id % kWarpsPerBlockN;
      nv_bfloat16* stage_a = sA_tma + stage * kStageAElements;
      nv_bfloat16* stage_b = sB_tma + stage * kStageBElements;
      const nv_bfloat16* warp_a_src =
          stage_a + warp_m * kWarpTileM * kWarpTileK;
      const nv_bfloat16* warp_b_src =
          stage_b + warp_n * kWarpTileN * kWarpTileK;
      const int consumer_slot = consumer_slot_swizzle(stage, consumer_warp_id);
      nv_bfloat16* warp_a_ptr =
          sA_swizzled +
          (static_cast<std::size_t>(stage) * kConsumerWarps + consumer_slot) *
              kWarpScratchAElements;
      nv_bfloat16* warp_b_ptr =
          sB_swizzled +
          (static_cast<std::size_t>(stage) * kConsumerWarps + consumer_slot) *
              kWarpScratchBElements;

      warp_copy_a_to_swizzled_smem(warp_a_ptr, warp_a_src, lane_id);
      warp_copy_b_to_swizzled_smem(warp_b_ptr, warp_b_src, lane_id);
      __syncwarp();

      wmma::fragment<wmma::matrix_a,
                     kWarpTileM,
                     kWarpTileN,
                     kWarpTileK,
                     nv_bfloat16,
                     wmma::row_major>
          a_frag;
      wmma::fragment<wmma::matrix_b,
                     kWarpTileM,
                     kWarpTileN,
                     kWarpTileK,
                     nv_bfloat16,
                     wmma::col_major>
          b_frag;
      wmma::load_matrix_sync(a_frag, warp_a_ptr, kWarpScratchALd);
      wmma::load_matrix_sync(b_frag, warp_b_ptr, kWarpScratchBLd);
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      if (lane_id == 0) {
        const uint32_t ready_addr =
            static_cast<uint32_t>(__cvta_generic_to_shared(&ready[stage]));
        mbarrier_arrive_shared(ready_addr);
      }
    }

    if (tid == 0 && future_tile < k_tiles) {
      const uint32_t ready_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(&ready[stage]));
      mbarrier_wait_parity(ready_addr, ready_phase[stage]);
      ready_phase[stage] ^= 1;

      nv_bfloat16* stage_a = sA_tma + stage * kStageAElements;
      nv_bfloat16* stage_b = sB_tma + stage * kStageBElements;
      const uint32_t filled_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(&filled[stage]));
      const uint32_t stage_a_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(stage_a));
      const uint32_t stage_b_addr =
          static_cast<uint32_t>(__cvta_generic_to_shared(stage_b));
      mbarrier_arrive_expect_tx(
          filled_addr,
          static_cast<uint32_t>((kStageAElements + kStageBElements) *
                                sizeof(nv_bfloat16)));
      tma_load_2d(stage_a_addr, &A_tmap, future_tile * kWarpTileK, block_m,
                  filled_addr);
      tma_load_2d(stage_b_addr, &B_tmap, future_tile * kWarpTileK, block_n,
                  filled_addr);
    }
  }

  if (is_consumer) {
    const int warp_m = consumer_warp_id / kWarpsPerBlockN;
    const int warp_n = consumer_warp_id % kWarpsPerBlockN;
    const int warp_row = block_m + warp_m * kWarpTileM;
    const int warp_col = block_n + warp_n * kWarpTileN;
    wmma::store_matrix_sync(D + warp_row * N + warp_col, acc_frag, N,
                            wmma::mem_row_major);
  }
}

void launch_hopper_swizzle_ws_tma_gemm(const CUtensorMap& a_tmap,
                                       const CUtensorMap& b_tmap,
                                       float* d_d,
                                       const Step6Options& options) {
  dim3 block(kThreads);
  dim3 grid((options.n / kTileN) * (options.m / kTileM));
  CHECK_CUDA(cudaFuncSetAttribute(
      step6_hopper_swizzle_ws_tma_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kStep6SmemBytes)));
  step6_hopper_swizzle_ws_tma_kernel<<<grid, block, kStep6SmemBytes>>>(
      a_tmap, b_tmap, d_d, options.m, options.n, options.k);
  CHECK_CUDA(cudaGetLastError());
}

template <typename LaunchFn>
TimingStats benchmark_kernel(const Step6Options& options,
                             LaunchFn&& launch_fn,
                             float* d_out,
                             std::vector<float>& host_out) {
  const std::size_t output_bytes =
      static_cast<std::size_t>(options.m) * options.n * sizeof(float);

  for (int i = 0; i < options.warmup; ++i) {
    launch_fn();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> times_ms;
  times_ms.reserve(options.iters);
  cudaEvent_t start{};
  cudaEvent_t stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  for (int i = 0; i < options.iters; ++i) {
    CHECK_CUDA(cudaEventRecord(start));
    launch_fn();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
    times_ms.push_back(elapsed);
  }

  CHECK_CUDA(cudaMemcpy(host_out.data(), d_out, output_bytes,
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  Options stats_options;
  stats_options.m = options.m;
  stats_options.n = options.n;
  stats_options.k = options.k;
  return finalize_timing_stats(stats_options, times_ms, host_out);
}

CompareStats compare_exact(const std::vector<float>& reference,
                           const std::vector<float>& actual) {
  CompareStats stats;
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double diff = std::abs(static_cast<double>(reference[i]) -
                                 static_cast<double>(actual[i]));
    stats.max_abs = std::max(stats.max_abs, diff);
    if (diff != 0.0) {
      ++stats.fail_count;
    }
  }
  stats.pass = (stats.fail_count == 0);
  return stats;
}

void print_step6_result_line(const Step6Options& options,
                             const char* backend,
                             const TimingStats& stats) {
  std::cout << std::fixed << std::setprecision(6)
            << "RESULT benchmark=bench_step6_hopper_swizzle_ws_tma"
            << " backend=" << backend
            << " m=" << options.m
            << " n=" << options.n
            << " k=" << options.k
            << " warmup=" << options.warmup
            << " iters=" << options.iters
            << " avg_ms=" << stats.avg_ms
            << " median_ms=" << stats.median_ms
            << " min_ms=" << stats.min_ms
            << " gflops=" << stats.gflops
            << " tflops=" << (stats.gflops / 1000.0)
            << " checksum=" << stats.checksum
            << '\n';
}

void print_step6_check_line(const Step6Options& options,
                            const char* backend,
                            const CompareStats& compare,
                            bool exact_required) {
  std::cout << std::scientific << std::setprecision(6)
            << "CHECK benchmark=bench_step6_hopper_swizzle_ws_tma"
            << " backend=" << backend
            << " m=" << options.m
            << " n=" << options.n
            << " k=" << options.k
            << " pass=" << static_cast<int>(compare.pass)
            << " max_abs=" << compare.max_abs
            << " fail_count=" << compare.fail_count;
  if (!exact_required) {
    std::cout << " note=informational_only";
  }
  std::cout << '\n';
}

void print_step6_summary(const Step6Options& options,
                         const TimingStats& hopper_swizzle_stats,
                         const TimingStats& cublas_fast_stats,
                         const TimingStats& cublas_pedantic_stats,
                         const CompareStats& hopper_compare,
                         const CompareStats& fast_compare) {
  auto print_row = [&](const char* backend,
                       const TimingStats& stats,
                       double baseline_ms) {
    const double speedup = baseline_ms / stats.median_ms;
    std::cout << std::fixed << std::setprecision(6)
              << "  " << std::left << std::setw(20) << backend
              << " median_ms=" << std::setw(10) << stats.median_ms
              << " tflops=" << std::setw(9) << (stats.gflops / 1000.0)
              << " speedup_vs_fast=" << std::setw(8) << speedup
              << " checksum=" << stats.checksum << '\n';
  };

  std::cout << "SUMMARY benchmark=bench_step6_hopper_swizzle_ws_tma"
            << " m=" << options.m
            << " n=" << options.n
            << " k=" << options.k
            << " baseline=cublas_fast_16bf"
            << '\n';
  std::cout << "  correctness: hopper_swizzle_ws_tma_mma vs cublas_pedantic exact_pass="
            << static_cast<int>(hopper_compare.pass)
            << " max_abs=" << std::scientific << std::setprecision(6)
            << hopper_compare.max_abs
            << " fail_count=" << hopper_compare.fail_count << '\n';
  std::cout << "  fast_math_delta: cublas_fast_16bf vs cublas_pedantic pass="
            << static_cast<int>(fast_compare.pass)
            << " max_abs=" << std::scientific << std::setprecision(6)
            << fast_compare.max_abs
            << " fail_count=" << fast_compare.fail_count << '\n';
  print_row("hopper_swizzle_ws_tma_mma", hopper_swizzle_stats,
            cublas_fast_stats.median_ms);
  print_row("cublas_fast_16bf", cublas_fast_stats, cublas_fast_stats.median_ms);
  print_row("cublas_pedantic", cublas_pedantic_stats,
            cublas_fast_stats.median_ms);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Step6Options options = parse_options(argc, argv);
    print_device_info();

    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
    if (props.major < 9) {
      throw std::runtime_error(
          "step6 hopper swizzle warp-specialized tma path requires SM90 or newer");
    }

    CHECK_CU(cuInit(0));

    const std::size_t a_elems =
        static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_elems =
        static_cast<std::size_t>(options.n) * options.k;
    const std::size_t d_elems =
        static_cast<std::size_t>(options.m) * options.n;

    std::vector<nv_bfloat16> host_A(a_elems);
    std::vector<nv_bfloat16> host_B(b_elems);
    initialize_exact_tensor(host_A, options.seed);
    initialize_exact_tensor(host_B, options.seed + 1);

    nv_bfloat16* device_A = nullptr;
    nv_bfloat16* device_B = nullptr;
    float* device_hopper_swizzle_D = nullptr;
    float* device_cublas_pedantic_D = nullptr;
    float* device_cublas_fast_D = nullptr;
    CHECK_CUDA(cudaMalloc(&device_A, a_elems * sizeof(nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&device_B, b_elems * sizeof(nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&device_hopper_swizzle_D, d_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_cublas_pedantic_D, d_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&device_cublas_fast_D, d_elems * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_A, host_A.data(), a_elems * sizeof(nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(device_B, host_B.data(), b_elems * sizeof(nv_bfloat16),
                          cudaMemcpyHostToDevice));

    CUtensorMap a_tmap{};
    CUtensorMap b_tmap{};
    create_tensor_map_2d(&a_tmap, device_A, options.m, options.k, kTileM,
                         kWarpTileK);
    create_tensor_map_2d(&b_tmap, device_B, options.n, options.k, kTileN,
                         kWarpTileK);

    cublasHandle_t handle{};
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    auto hopper_swizzle_launch = [&] {
      launch_hopper_swizzle_ws_tma_gemm(a_tmap, b_tmap,
                                        device_hopper_swizzle_D, options);
    };
    auto cublas_pedantic_launch = [&] {
      run_cublas_reference(handle, options, device_A, device_B,
                           device_cublas_pedantic_D,
                           CUBLAS_COMPUTE_32F_PEDANTIC);
    };
    auto cublas_fast_launch = [&] {
      run_cublas_reference(handle, options, device_A, device_B,
                           device_cublas_fast_D,
                           CUBLAS_COMPUTE_32F_FAST_16BF);
    };

    cublas_pedantic_launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    cublas_fast_launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    hopper_swizzle_launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> host_hopper_swizzle_D(d_elems, 0.0f);
    std::vector<float> host_cublas_pedantic_D(d_elems, 0.0f);
    std::vector<float> host_cublas_fast_D(d_elems, 0.0f);
    CHECK_CUDA(cudaMemcpy(host_hopper_swizzle_D.data(), device_hopper_swizzle_D,
                          d_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_cublas_pedantic_D.data(), device_cublas_pedantic_D,
                          d_elems * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(host_cublas_fast_D.data(), device_cublas_fast_D,
                          d_elems * sizeof(float), cudaMemcpyDeviceToHost));

    const CompareStats hopper_compare =
        compare_exact(host_cublas_pedantic_D, host_hopper_swizzle_D);
    const CompareStats fast_compare =
        compare_exact(host_cublas_pedantic_D, host_cublas_fast_D);
    print_step6_check_line(options, "hopper_swizzle_ws_tma_mma_vs_cublas_pedantic",
                           hopper_compare, true);
    print_step6_check_line(options, "cublas_fast_16bf_vs_cublas_pedantic",
                           fast_compare, false);

    if (!hopper_compare.pass) {
      throw std::runtime_error(
          "hopper swizzle ws tma mma output does not exactly match cuBLAS");
    }

    TimingStats hopper_swizzle_stats =
        benchmark_kernel(options, hopper_swizzle_launch,
                         device_hopper_swizzle_D, host_hopper_swizzle_D);
    TimingStats cublas_fast_stats =
        benchmark_kernel(options, cublas_fast_launch, device_cublas_fast_D,
                         host_cublas_fast_D);
    TimingStats cublas_pedantic_stats =
        benchmark_kernel(options, cublas_pedantic_launch,
                         device_cublas_pedantic_D, host_cublas_pedantic_D);

    print_step6_result_line(options, "hopper_swizzle_ws_tma_mma",
                            hopper_swizzle_stats);
    print_step6_result_line(options, "cublas_fast_16bf", cublas_fast_stats);
    print_step6_result_line(options, "cublas_pedantic", cublas_pedantic_stats);
    print_step6_summary(options, hopper_swizzle_stats, cublas_fast_stats,
                        cublas_pedantic_stats, hopper_compare, fast_compare);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(device_A));
    CHECK_CUDA(cudaFree(device_B));
    CHECK_CUDA(cudaFree(device_hopper_swizzle_D));
    CHECK_CUDA(cudaFree(device_cublas_pedantic_D));
    CHECK_CUDA(cudaFree(device_cublas_fast_D));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR " << e.what() << '\n';
    return 1;
  }
}
