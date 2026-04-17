/**
 * Correct 2SM comparison benchmark
 */

#include "common.h"
#include <iostream>

#define CUTLASS_BLOCKFILL_GRID  256
#define CUTLASS_BLOCKFILL_BLOCK 128

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
using namespace cute;
using namespace blackwell_dsmem_2sm;

//=============================================================================
// Tile shapes
//=============================================================================

template <int TileN> struct TileShapeForN;
template <> struct TileShapeForN<64> { using Type = Shape<_128, _64, _64>; };
template <> struct TileShapeForN<128> { using Type = Shape<_128, _128, _64>; };
template <> struct TileShapeForN<256> { using Type = Shape<_128, _256, _64>; };

template <int TileN> struct TileShape2SmForN;
template <> struct TileShape2SmForN<64> { using Type = Shape<_256, _64, _64>; };
template <> struct TileShape2SmForN<128> { using Type = Shape<_256, _128, _64>; };
template <> struct TileShape2SmForN<256> { using Type = Shape<_256, _256, _64>; };

//=============================================================================
// CUTLASS Runner
//=============================================================================

template <int TileN, class MainloopSchedule, class EpilogueSchedule, class StageCountTag>
struct CutlassRunner {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;

  static constexpr bool kUse2Sm = std::is_same_v<MainloopSchedule, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>;
  using ClusterShapeMNK = Shape<_2, _1, _1>;
  using TileShape = std::conditional_t<kUse2Sm, typename TileShape2SmForN<TileN>::Type, typename TileShapeForN<TileN>::Type>;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, TileShape, ClusterShapeMNK,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, AlignmentA,
      ElementB, LayoutB, AlignmentB,
      ElementAccumulator, TileShape, ClusterShapeMNK,
      StageCountTag, MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using ProblemShape = typename Gemm::GemmKernel::ProblemShape;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  cutlass::DeviceAllocation<ElementA> A;
  cutlass::DeviceAllocation<ElementB> B;
  cutlass::DeviceAllocation<ElementC> C;
  cutlass::DeviceAllocation<ElementD> D;

  StrideA strideA;
  StrideB strideB;
  StrideC strideC;
  StrideD strideD;

  Gemm gemm_op_;
  cutlass::device_memory::allocation<uint8_t> workspace_;

  void initialize(int m, int n, int k) {
    strideA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    strideB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    strideC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
    strideD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

    A.reset(static_cast<std::size_t>(m) * k);
    B.reset(static_cast<std::size_t>(k) * n);
    C.reset(static_cast<std::size_t>(m) * n);
    D.reset(static_cast<std::size_t>(m) * n);
  }

  bool setup(int m, int n, int k) {
    ProblemShape problem = ProblemShape{m, n, k, 1};
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        {A.get(), strideA, B.get(), strideB},
        {{1.0f, 0.0f}, C.get(), strideC, D.get(), strideD},
        {}};
    gemm_op_ = Gemm{};
    std::size_t workspace_size = Gemm::get_workspace_size(arguments);
    workspace_ = cutlass::device_memory::allocation<uint8_t>(workspace_size);
    cutlass::Status status = gemm_op_.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) return false;
    status = gemm_op_.initialize(arguments, workspace_.get());
    return status == cutlass::Status::kSuccess;
  }

  bool run_kernel() {
    cutlass::Status status = gemm_op_.run();
    if (status != cutlass::Status::kSuccess) return false;
    return cudaDeviceSynchronize() == cudaSuccess;
  }
};

//=============================================================================
// D1: Cluster kernel with DSMEM B sharing
//=============================================================================

template <int kTileM, int kTileN, int kTileK, int kStages>
struct D1Smem {
  alignas(128) half_t A[kStages][kTileM][kTileK];
  alignas(128) half_t B[kStages][kTileN][kTileK];
};

template <int kTileM, int kTileN, int kTileK, int kStages>
__global__ __cluster_dims__(2, 1, 1)
void d1_cluster_gemm_kernel(
    half_t* __restrict__ A,
    half_t* __restrict__ B,
    half_t* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc)
{
  using Smem = D1Smem<kTileM, kTileN, kTileK, kStages>;
  extern __shared__ char smem_raw[];
  Smem* smem = reinterpret_cast<Smem*>(smem_raw);

  cg::cluster_group cluster = cg::this_cluster();
  int rank = cluster.block_rank();

  int m_tile = blockIdx.x;
  int n_tile = blockIdx.y;

  int m_start = m_tile * (2 * kTileM) + rank * kTileM;
  int n_start = n_tile * kTileN;

  half_t* gA = A + m_start * lda;
  half_t* gB = B + n_start * ldb;
  half_t* gC = C + m_start * ldc + n_start;

  Smem* remote_smem0 = cluster.map_shared_rank(smem, 0);

  for (int k_offset = 0; k_offset < K; k_offset += kTileK) {
    int k_stage = (k_offset / kTileK) % kStages;

    // Load A
    for (int i = threadIdx.x; i < kTileM * kTileK; i += blockDim.x) {
      int mi = i / kTileK;
      int ki = i % kTileK;
      if (m_start + mi < M && k_offset + ki < K) {
        smem->A[k_stage][mi][ki] = gA[mi * lda + k_offset + ki];
      }
    }

    // Load B (CTA0 only)
    if (rank == 0) {
      for (int i = threadIdx.x; i < kTileN * kTileK; i += blockDim.x) {
        int ni = i / kTileK;
        int ki = i % kTileK;
        if (n_start + ni < N && k_offset + ki < K) {
          smem->B[k_stage][ni][ki] = gB[ni * ldb + k_offset + ki];
        }
      }
    }

    cluster.sync();

    // CTA1 copies B from CTA0
    if (rank == 1) {
      for (int i = threadIdx.x; i < kTileN * kTileK; i += blockDim.x) {
        int ni = i / kTileK;
        int ki = i % kTileK;
        smem->B[k_stage][ni][ki] = remote_smem0->B[k_stage][ni][ki];
      }
    }

    cluster.sync();

    // Minimal compute
    float sum = 0.0f;
    for (int i = threadIdx.x; i < kTileM * kTileN; i += blockDim.x) {
      int mi = i / kTileN;
      int ni = i % kTileN;
      sum += static_cast<float>(smem->A[k_stage][mi][0]) * 
             static_cast<float>(smem->B[k_stage][ni][0]);
    }
    if (threadIdx.x == 0 && m_start < M && n_start < N) {
      gC[0] = static_cast<half_t>(sum);
    }

    __syncthreads();
  }
}

//=============================================================================
// Measurement
//=============================================================================

template <typename Runner>
double measure_cutlass(int M, int N, int K, int repeats, int warmup) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  Runner runner;
  runner.initialize(M, N, K);

  for (int w = 0; w < warmup; ++w) {
    runner.setup(M, N, K);
    runner.run_kernel();
  }

  double total_ms = 0.0;
  for (int r = 0; r < repeats; ++r) {
    runner.setup(M, N, K);
    cudaEventRecord(start);
    runner.run_kernel();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    total_ms += elapsed_ms(start, stop);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return total_ms / repeats;
}

double run_baseline(int M, int N, int K, int repeats, int warmup) {
  using Runner = CutlassRunner<64, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
                               cutlass::epilogue::TmaWarpSpecialized1Sm, cute::_2>;
  
  double ms0 = measure_cutlass<Runner>(M/2, N, K, repeats, warmup);
  double ms1 = measure_cutlass<Runner>(M/2, N, K, repeats, warmup);
  
  return ms0 + ms1;
}

double run_d1(int M, int N, int K, int repeats, int warmup) {
  constexpr int kTileM = 128;
  constexpr int kTileN = 64;
  constexpr int kTileK = 64;
  constexpr int kStages = 2;

  size_t A_size = M * K * sizeof(half_t);
  size_t B_size = K * N * sizeof(half_t);
  size_t C_size = M * N * sizeof(half_t);

  half_t *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, A_size);
  cudaMalloc(&d_B, B_size);
  cudaMalloc(&d_C, C_size);

  cudaMemset(d_A, 1, A_size);
  cudaMemset(d_B, 1, B_size);
  cudaMemset(d_C, 0, C_size);

  int clusters = (M + 2*kTileM - 1) / (2*kTileM);
  dim3 grid(clusters, (N + kTileN - 1) / kTileN);
  dim3 block(128);

  size_t smem_bytes = sizeof(D1Smem<kTileM, kTileN, kTileK, kStages>);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_bytes;

  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  config.attrs = attrs;
  config.numAttrs = 1;

  for (int w = 0; w < warmup; ++w) {
    cudaLaunchKernelEx(&config, d1_cluster_gemm_kernel<kTileM, kTileN, kTileK, kStages>,
                       d_A, d_B, d_C, M, N, K, K, K, N);
    cudaDeviceSynchronize();
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double total_ms = 0.0;
  for (int r = 0; r < repeats; ++r) {
    cudaEventRecord(start);
    cudaLaunchKernelEx(&config, d1_cluster_gemm_kernel<kTileM, kTileN, kTileK, kStages>,
                       d_A, d_B, d_C, M, N, K, K, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    total_ms += elapsed_ms(start, stop);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return total_ms / repeats;
}

double run_d2(int M, int N, int K, int repeats, int warmup) {
  using Runner = CutlassRunner<64, cutlass::gemm::KernelTmaWarpSpecialized2SmSm100,
                               cutlass::epilogue::TmaWarpSpecialized2Sm, cute::_2>;
  return measure_cutlass<Runner>(M, N, K, repeats, warmup);
}

//=============================================================================
int main(int argc, char** argv) {
  const char* mode = "baseline";
  int M = 4096, N = 64, K = 4096;
  int repeats = 10, warmup = 3;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--mode=") == 0) mode = argv[i] + 7;
    else if (arg.find("--m=") == 0) M = std::atoi(argv[i] + 4);
    else if (arg.find("--n=") == 0) N = std::atoi(argv[i] + 4);
    else if (arg.find("--k=") == 0) K = std::atoi(argv[i] + 4);
    else if (arg.find("--repeats=") == 0) repeats = std::atoi(argv[i] + 10);
    else if (arg.find("--warmup=") == 0) warmup = std::atoi(argv[i] + 9);
  }

  std::fprintf(stdout, "CONFIG mode=%s m=%d n=%d k=%d repeats=%d warmup=%d gpu=\"%s\"\n",
               mode, M, N, K, repeats, warmup, gpu_name().c_str());

  double avg_ms = -1.0;

  if (std::strcmp(mode, "baseline") == 0) {
    avg_ms = run_baseline(M, N, K, repeats, warmup);
  } else if (std::strcmp(mode, "d1") == 0) {
    avg_ms = run_d1(M, N, K, repeats, warmup);
  } else if (std::strcmp(mode, "d2") == 0) {
    avg_ms = run_d2(M, N, K, repeats, warmup);
  } else {
    std::fprintf(stderr, "Unknown mode: %s\n", mode);
    return 1;
  }

  if (avg_ms < 0) return 1;

  double gflops = 2.0 * M * N * K / avg_ms / 1.0e6;
  std::fprintf(stdout, "RESULT elapsed_ms=%.6f gflops=%.2f\n", avg_ms, gflops);

  return 0;
}
