/**
 * Complete Tensor Core GEMM kernel for Blackwell SM100
 * Reference: https://gau-nernst.github.io/tcgen05/
 * 
 * Key components:
 * 1. TMA descriptor creation (host)
 * 2. mbarrier for synchronization
 * 3. Tensor memory allocation
 * 4. Shared memory descriptor encoding
 * 5. Instruction descriptor encoding
 * 6. tcgen05.mma execution
 * 7. Epilogue using tcgen05.ld
 */

#include "common.h"
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using namespace blackwell_dsmem_2sm;

//=============================================================================
// Configuration - using smaller tiles for simpler implementation
//=============================================================================
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
constexpr int TB_SIZE = 128;

// MMA shapes for BF16
constexpr int MMA_M = 64;   // tcgen05.mma supports up to 128
constexpr int MMA_N = 64;   // tcgen05.mma supports up to 256
constexpr int MMA_K = 16;   // 8 BF16 elements = 16 bytes

// Number of MMA iterations per K block
constexpr int MMA_PER_K = BLOCK_K / MMA_K;

//=============================================================================
// PTX inline assembly - Low level primitives
//=============================================================================

// mbarrier initialization
__device__ __forceinline__
void mbarrier_init(uint32_t addr, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" 
               :: "r"(addr), "r"(count) : "memory");
}

// mbarrier arrive with expect_tx
__device__ __forceinline__
void mbarrier_arrive_expect_tx(uint32_t addr, uint32_t tx_count) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
               :: "r"(addr), "r"(tx_count) : "memory");
}

// mbarrier wait
__device__ __forceinline__
void mbarrier_wait(uint32_t addr, uint32_t phase) {
  uint32_t ticks = 0x989680;  // ~10M cycles timeout
  asm volatile(
    "{\n\t"
    ".reg .pred P;\n\t"
    "LOOP:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, %2;\n\t"
    "@!P bra LOOP;\n\t"
    "}"
    :: "r"(addr), "r"(phase), "r"(ticks) : "memory"
  );
}

// Fence after mbarrier init
__device__ __forceinline__
void fence_mbarrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// TMA load 2D
__device__ __forceinline__
void tma_load_2d(uint32_t dst, const void* tmap, uint32_t coord0, uint32_t coord1, uint32_t mbar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1, {%3, %4}], [%2];"
               :: "r"(dst), "l"(tmap), "r"(mbar), "r"(coord0), "r"(coord1) : "memory");
}

// Tensor memory allocation
__device__ __forceinline__
void tmem_alloc(uint32_t addr_ptr, uint32_t num_cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
               :: "r"(addr_ptr), "r"(num_cols) : "memory");
}

// Tensor memory deallocation
__device__ __forceinline__
void tmem_dealloc(uint32_t taddr, uint32_t num_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
               :: "r"(taddr), "r"(num_cols) : "memory");
}

// tcgen05.mma - the main compute instruction
// Note: This is complex and requires proper descriptor encoding
__device__ __forceinline__
void tcgen05_mma(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, 
                 uint32_t desc_i, uint32_t pred_d) {
  asm volatile("tcgen05.mma.cta_group::1 [%0], %1, %2, %3, %4;"
               :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), 
                  "r"(desc_i), "r"(pred_d) : "memory");
}

// tcgen05.commit - signal completion of MMA
__device__ __forceinline__
void tcgen05_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(mbar) : "memory");
}

// tcgen05.fence
__device__ __forceinline__
void tcgen05_fence() {
  asm volatile("tcgen05.fence;" ::: "memory");
}

// tcgen05.ld - load from tensor memory
__device__ __forceinline__
void tmem_load(float* data, uint32_t addr) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];"
               : "=f"(data[0]), "=f"(data[1]), "=f"(data[2]), "=f"(data[3])
               : "r"(addr));
}

// tcgen05.wait::ld
__device__ __forceinline__
void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

// elect.sync - elect one thread in warp
__device__ __forceinline__
uint32_t elect_sync() {
  uint32_t pred;
  asm volatile("elect.sync |_|b32 %0, 0xffff;" : "=r"(pred));
  return pred;
}

//=============================================================================
// Helper: Encode shared memory descriptor
//=============================================================================
__device__ __forceinline__
uint64_t encode_smem_desc(uint32_t addr, uint32_t leading_dim_offset, 
                          uint32_t stride_dim_offset, uint32_t swizzle_mode) {
  // Descriptor encoding (64-bit):
  // [15:0]  address (16-bit)
  // [31:16] LBO - leading dimension byte offset
  // [47:32] SBO - stride dimension byte offset
  // [46]    leading dimension offset mode (0 = byte, 1 = element)
  // [61:60] swizzle mode
  // [63:62] reserved
  
  uint64_t desc = static_cast<uint64_t>(addr & 0xFFFF);
  desc |= (static_cast<uint64_t>(leading_dim_offset) << 16);
  desc |= (static_cast<uint64_t>(stride_dim_offset) << 32);
  desc |= (static_cast<uint64_t>(swizzle_mode) << 61);
  return desc;
}

//=============================================================================
// Helper: Encode instruction descriptor for BF16 MMA
//=============================================================================
__device__ __forceinline__
uint32_t encode_inst_desc_bf16(uint32_t mma_m, uint32_t mma_n) {
  // Instruction descriptor encoding (32-bit):
  // [3:0]   dtype (1 = FP32)
  // [6:4]   atype (1 = BF16)
  // [9:7]   btype (1 = BF16)
  // [16:10] mma_n
  // [23:17] mma_m
  
  uint32_t desc = (1U << 0);         // FP32 output
  desc |= (1U << 4);                  // BF16 A
  desc |= (1U << 7);                  // BF16 B
  desc |= ((mma_n >> 3) << 10);       // MMA_N / 8
  desc |= ((mma_m >> 4) << 17);       // MMA_M / 16
  
  return desc;
}

//=============================================================================
// Host: Create TMA descriptor
//=============================================================================
void create_tma_descriptor_2d(CUtensorMap* tmap, const nv_bfloat16* ptr,
                              cuuint64_t global_dim0, cuuint64_t global_dim1,
                              cuuint32_t box_dim0, cuuint32_t box_dim1) {
  constexpr uint32_t rank = 2;
  
  // Global dimensions (reversed order for TMA)
  cuuint64_t globalDim[rank] = {global_dim0, global_dim1};
  cuuint64_t globalStrides[rank-1] = {global_dim0 * sizeof(nv_bfloat16)};
  
  // Box dimensions (tile size in shared memory)
  cuuint32_t boxDim[rank] = {box_dim0, box_dim1};
  cuuint32_t elementStrides[rank] = {1, 1};
  
  CUresult err = cuTensorMapEncodeTiled(
    tmap,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    (void*)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CU_TENSOR_MAP_INTERLEAVE_NONE,
    CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  
  if (err != CUDA_SUCCESS) {
    std::fprintf(stderr, "TMA descriptor creation failed: %d\n", err);
  }
}

//=============================================================================
// GEMM Kernel using tcgen05.mma
//=============================================================================
__global__ __launch_bounds__(TB_SIZE)
void tc_gemm_kernel(
    const __grid_constant__ CUtensorMap A_tmap,
    const __grid_constant__ CUtensorMap B_tmap,
    nv_bfloat16* C,
    int M, int N, int K)
{
  // Thread identification
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  
  // Block identification
  const int bid_m = blockIdx.y;
  const int bid_n = blockIdx.x;
  
  // Output tile offset
  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;
  
  // Early exit if tile is out of bounds
  if (off_m >= M || off_n >= N) return;
  
  //===========================================================================
  // Shared memory allocation
  //===========================================================================
  extern __shared__ __align__(16) char smem_buffer[];
  
  // A tile: [BLOCK_M, BLOCK_K] BF16 = BLOCK_M * BLOCK_K * 2 bytes
  nv_bfloat16* A_smem = reinterpret_cast<nv_bfloat16*>(smem_buffer);
  
  // B tile: [BLOCK_N, BLOCK_K] BF16
  nv_bfloat16* B_smem = A_smem + BLOCK_M * BLOCK_K;
  
  // mbarrier array (one per stage for TMA)
  __shared__ uint64_t mbars[NUM_STAGES];
  
  // Tensor memory address pointer
  __shared__ uint32_t tmem_addr_ptr;
  
  //===========================================================================
  // Initialization (single thread)
  //===========================================================================
  if (tid == 0) {
    // Initialize mbarriers
    for (int i = 0; i < NUM_STAGES; i++) {
      uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbars[i]));
      mbarrier_init(mbar_addr, 1);
    }
    fence_mbarrier_init();
    
    // Allocate tensor memory
    // We need BLOCK_N columns for accumulator
    uint32_t addr_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_addr_ptr));
    tmem_alloc(addr_ptr, BLOCK_N);
  }
  
  __syncthreads();
  
  uint32_t tmem_addr = tmem_addr_ptr;
  
  //===========================================================================
  // Main GEMM loop
  //===========================================================================
  int phase = 0;
  const int num_iters = K / BLOCK_K;
  
  for (int k_iter = 0; k_iter < num_iters; k_iter++) {
    const int stage = k_iter % NUM_STAGES;
    
    //-------------------------------------------------------------------------
    // TMA load
    //-------------------------------------------------------------------------
    if (warp_id == 0 && elect_sync()) {
      uint32_t A_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(A_smem));
      uint32_t B_smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(B_smem));
      uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbars[stage]));
      
      // K offset for this iteration
      const int off_k = k_iter * BLOCK_K;
      
      // Issue TMA loads
      // A: [M, K] -> load [BLOCK_M, BLOCK_K] at (off_k, off_m)
      tma_load_2d(A_smem_addr, &A_tmap, off_k, off_m, mbar_addr);
      
      // B: [N, K] -> load [BLOCK_N, BLOCK_K] at (off_k, off_n)
      tma_load_2d(B_smem_addr, &B_tmap, off_k, off_n, mbar_addr);
      
      // Signal expected transfer size
      const uint32_t tx_count = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
      mbarrier_arrive_expect_tx(mbar_addr, tx_count);
    }
    
    // Wait for TMA
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbars[stage]));
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;
    
    // tcgen05 fence after sync
    tcgen05_fence();
    
    //-------------------------------------------------------------------------
    // Tensor Core MMA
    //-------------------------------------------------------------------------
    if (warp_id == 0 && elect_sync()) {
      // Encode shared memory descriptors
      uint32_t A_addr = static_cast<uint32_t>(__cvta_generic_to_shared(A_smem));
      uint32_t B_addr = static_cast<uint32_t>(__cvta_generic_to_shared(B_smem));
      
      // LBO = leading byte offset = BLOCK_M * 16 (for A) or BLOCK_N * 16 (for B)
      // SBO = stride byte offset = 8 * BLOCK_K * sizeof(nv_bfloat16) = 8 * 64 * 2 = 1024
      const uint32_t LBO_A = BLOCK_M * 16;
      const uint32_t LBO_B = BLOCK_N * 16;
      const uint32_t SBO = 8 * 16; // 8 rows * 16 bytes
      
      uint64_t desc_a = encode_smem_desc(A_addr, LBO_A, SBO, 0);
      uint64_t desc_b = encode_smem_desc(B_addr, LBO_B, SBO, 0);
      
      // Encode instruction descriptor
      uint32_t inst_desc = encode_inst_desc_bf16(MMA_M, MMA_N);
      
      // Accumulation predicate: 0 for first iteration, 1 for others
      uint32_t pred_acc = (k_iter == 0) ? 0 : 1;
      
      // Issue MMA operations for each K slice
      for (int mma_k = 0; mma_k < MMA_PER_K; mma_k++) {
        // Select K slice of shared memory
        uint32_t A_slice = A_addr + mma_k * BLOCK_M * MMA_K * sizeof(nv_bfloat16);
        uint32_t B_slice = B_addr + mma_k * BLOCK_N * MMA_K * sizeof(nv_bfloat16);
        
        uint64_t desc_a_slice = encode_smem_desc(A_slice, LBO_A, SBO, 0);
        uint64_t desc_b_slice = encode_smem_desc(B_slice, LBO_B, SBO, 0);
        
        // Accumulation for first MMA iteration
        uint32_t pred = (k_iter == 0 && mma_k == 0) ? 0 : 1;
        
        tcgen05_mma(tmem_addr, desc_a_slice, desc_b_slice, inst_desc, pred);
      }
      
      // Commit MMA operations
      tcgen05_commit(mbar_addr);
    }
    
    // Wait for MMA completion
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;
  }
  
  //===========================================================================
  // Epilogue: Write results to global memory
  //===========================================================================
  // tcgen05 fence before reading tensor memory
  tcgen05_fence();
  
  // Each thread reads 4 FP32 values from tensor memory
  // and writes to global memory
  // Layout D: each warp has access to 32 consecutive rows
  
  for (int n = 0; n < BLOCK_N; n += 8) {
    for (int m = 0; m < BLOCK_M; m += TB_SIZE) {
      const int row = m + tid;
      if (row < BLOCK_M) {
        // Encode tensor memory address
        // Row in bits [21:16], column in bits [15:0]
        const uint32_t tmem_row = warp_id * 32 + (lane_id % 32);
        const uint32_t tmem_col = n;
        const uint32_t tmem_ptr = (tmem_row << 16) | tmem_col;
        
        // Load from tensor memory
        float data[4];
        tmem_load(data, tmem_ptr);
        tmem_wait_ld();
        
        // Convert to BF16 and store
        for (int i = 0; i < 4; i++) {
          const int g_row = off_m + row;
          const int g_col = off_n + n + i;
          if (g_row < M && g_col < N) {
            C[g_row * N + g_col] = __float2bfloat16(data[i]);
          }
        }
      }
    }
  }
  
  //===========================================================================
  // Cleanup
  //===========================================================================
  __syncthreads();
  if (warp_id == 0) {
    tmem_dealloc(tmem_addr, BLOCK_N);
  }
}

//=============================================================================
// Benchmark wrapper
//=============================================================================
extern "C" void run_tc_gemm_full(int M, int N, int K, int repeats, int warmup) {
  // Allocate memory
  nv_bfloat16 *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, M * K * sizeof(nv_bfloat16));
  cudaMalloc(&d_B, K * N * sizeof(nv_bfloat16));
  cudaMalloc(&d_C, M * N * sizeof(nv_bfloat16));
  
  // Initialize
  cudaMemset(d_A, 1, M * K * sizeof(nv_bfloat16));
  cudaMemset(d_B, 1, K * N * sizeof(nv_bfloat16));
  cudaMemset(d_C, 0, M * N * sizeof(nv_bfloat16));
  
  // Create TMA descriptors
  CUtensorMap A_tmap, B_tmap;
  create_tma_descriptor_2d(&A_tmap, d_A, BLOCK_K, M, BLOCK_K, BLOCK_M);
  create_tma_descriptor_2d(&B_tmap, d_B, BLOCK_K, N, BLOCK_K, BLOCK_N);
  
  // Launch configuration
  dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  dim3 block(TB_SIZE);
  
  // Shared memory size
  size_t smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
  
  // Warmup
  for (int w = 0; w < warmup; ++w) {
    tc_gemm_kernel<<<grid, block, smem_size>>>(A_tmap, B_tmap, d_C, M, N, K);
  }
  cudaDeviceSynchronize();
  
  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    return;
  }
  
  // Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  float total_ms = 0.0f;
  for (int r = 0; r < repeats; ++r) {
    cudaEventRecord(start);
    tc_gemm_kernel<<<grid, block, smem_size>>>(A_tmap, B_tmap, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    total_ms += ms;
  }
  
  double avg_ms = total_ms / repeats;
  double tflops = 2.0 * M * N * K / 1e12 / (avg_ms / 1000.0);
  
  std::fprintf(stdout, "CONFIG kernel=tcgen05_full m=%d n=%d k=%d gpu=\"%s\"\n",
               M, N, K, gpu_name().c_str());
  std::fprintf(stdout, "RESULT kernel=tcgen05_full elapsed_ms=%.6f tflops=%.3f\n",
               avg_ms, tflops);
  
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
