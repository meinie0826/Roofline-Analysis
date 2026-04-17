# Blackwell DSMEM 2SM GEMM Experiments

## Overview
This experiment investigates whether Distributed Shared Memory (DSMEM) can provide Tensor Core-like performance for matrix B sharing across thread blocks in a cluster.

## Key Findings

### 1. CUTLASS 1SM vs 2SM Comparison
| Matrix Size | 1SM TFLOPS | 2SM TFLOPS | Relative |
|-------------|------------|------------|----------|
| 2048×2048×8192 | 449.4 | 444.7 | 99.0% |
| 8192×8192×8192 | 563.4 | 558.9 | 99.2% |
| 2048×4096×8192 | 481.4 | 478.1 | 99.3% |

**Discovery**: 2SM mode is NOT faster than 1SM for these matrix sizes on B300.

### 2. Software DSMEM Sharing (bench_3way)
| Mode | Time (ms) | GFLOPS | Improvement |
|------|-----------|--------|--------------|
| baseline (independent load) | 4.76 | 14,444 | 1.0x |
| D1 (DSMEM copy) | 3.59 | 19,150 | **1.33x** |

**Note**: This kernel does NOT use Tensor Core. GFLOPS ~19 indicates FP32 compute.

### 3. Hardware Analysis
- B300 Tensor Core peak: ~1000+ TFLOPS (BF16)
- Our TC kernels should achieve 500+ TFLOPS
- Current bench_3way uses only 0.003% of TC capacity

## Implementation Status

### Completed
- ✅ CUTLASS 1SM/2SM comparison
- ✅ Software DSMEM sharing verification
- ✅ TC kernel skeleton (bench_tc_simple)
- ✅ Full TC kernel implementation (bench_tc_full)

### In Progress
- ⏳ Testing bench_tc_full on server
- ⏳ Comparing TC kernel with/without DSMEM

### Pending
- 📋 Performance analysis at different matrix sizes
- 📋 DSMEM bandwidth measurement

## Files

| File | Description |
|------|-------------|
| `bench_cutlass_2sm_gemm.cu` | CUTLASS 1SM/2SM comparison |
| `bench_3way.cu` | Software DSMEM sharing (no TC) |
| `bench_tc_simple.cu` | Simple MMA kernel skeleton |
| `bench_tc_full.cu` | Complete tcgen05.mma implementation |
| `common.h` | Shared utilities |

## Usage

```bash
# Build all
make all

# Build specific targets
make hand-written  # bench_3way, bench_tc_*
make cutlass      # CUTLASS benchmarks

# Run benchmarks
./bench_3way --mode=all --m=2048 --n=2048 --k=8192
./bench_cutlass_2sm_gemm --mode=1sm --m=8192 --n=8192 --k=8192
./bench_tc_full --m=2048 --n=2048 --k=8192
```

## Technical Details

### tcgen05.mma Components
1. **TMA Descriptor**: Encodes global memory layout
2. **mbarrier**: Synchronizes TMA and TC operations
3. **Tensor Memory**: Stores MMA accumulators
4. **Shared Memory Descriptor**: Encodes smem layout for MMA
5. **Instruction Descriptor**: Encodes MMA parameters

### DSMEM Considerations
- DSMEM bandwidth: ~20 GB/s (estimated)
- For 64×64 tile: 512 KB per K iteration
- At 500K TFLOPS: 128 iterations/ms = 64 MB/ms DSMEM needed
- DSMEM appears sufficient for most workloads

## References
- [tcgen05 for dummies](https://gau-nernst.github.io/tcgen05/)
- [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [PTX ISA 9.2](https://docs.nvidia.com/cuda/parallel-thread-execution/)
