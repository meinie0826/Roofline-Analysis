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

### 2. Current Focus
- B200/B300 CUTLASS Blackwell GEMM bring-up
- `mma.1sm` vs `mma.2sm` controlled comparison
- isolating the cost/benefit of B-operand sharing in CTA-pair execution

## Implementation Status

### Completed
- ✅ CUTLASS 1SM/2SM comparison
- ✅ CUTLASS `mma.2sm` cost benchmark with normalized summary output

### In Progress
- ⏳ Validating CUTLASS configs across B200 (`sm_100a`) and B300 (`sm_103a`)
- ⏳ Comparing TC kernel with/without DSMEM

### Pending
- 📋 Performance analysis at different matrix sizes
- 📋 DSMEM bandwidth measurement

## Files

| File | Description |
|------|-------------|
| `bench_cutlass_2sm_gemm.cu` | CUTLASS 1SM/2SM comparison |
| `bench_cutlass_mma2sm_cost.cu` | Controlled 1SM vs 2SM benchmark with summary ratios |
| `bench_d1_only.cu` | CUTLASS-based baseline around D1 / DSMEM-related flow |
| `bench_umma_1sm_2sm_cost.cu` | Instruction-level BF16 UMMA 1SM/2SM cost baseline |
| `common.h` | Shared utilities |

## Usage

```bash
# Build all
make all

# Build specific targets
make cutlass      # CUTLASS benchmarks
make bench_cutlass_2sm_gemm
make bench_cutlass_mma2sm_cost
make bench_umma_1sm_2sm_cost

# Override architecture when needed
make ARCH=sm_100a bench_cutlass_2sm_gemm      # B200
make ARCH=sm_103a bench_cutlass_2sm_gemm      # B300

# Run benchmarks
./bench_cutlass_2sm_gemm --mode=1sm --m=8192 --n=8192 --k=8192
./bench_cutlass_mma2sm_cost --mode=compare --m=8192 --n=8192 --k=8192 --tile-n=128 --stages=2
./bench_umma_1sm_2sm_cost --mode=compare --tile-n=128 --depth=256 --iters=1000
```

## `bench_cutlass_mma2sm_cost`

This benchmark focuses on the indirect question:

- how much faster or slower is `mma.2sm` than a comparable `mma.1sm` baseline
- how much of that difference remains after normalizing by total FLOPs
- what B-operand traffic reduction does the 2SM kernel ideally get from sharing

The benchmark runs CUTLASS `1sm` and `2sm` kernels on the same `(M, N, K)` problem and prints:

- `RESULT ... mode=1sm ...`
- `RESULT ... mode=2sm ...`
- `SUMMARY ... compare=2sm_vs_1sm ...`

Important summary fields:

- `speedup`: `time_1sm / time_2sm`
- `tflops_ratio`: `tflops_2sm / tflops_1sm`
- `ns_per_flop_ratio`: normalized cost ratio
- `est_b_bytes_ratio`: idealized B-tile GMEM traffic ratio implied by tile geometry
- `est_b_bw_ratio`: implied B-load bandwidth ratio using those estimates

Example:

```bash
make ARCH=sm_100a bench_cutlass_mma2sm_cost
./bench_cutlass_mma2sm_cost --mode=compare --m=8192 --n=8192 --k=8192 --tile-n=128 --stages=2
./bench_cutlass_mma2sm_cost --mode=1sm --m=8192 --n=8192 --k=4096 --tile-n=64 --stages=4
./bench_cutlass_mma2sm_cost --mode=2sm --m=8192 --n=8192 --k=4096 --tile-n=64 --stages=4
```

Notes:

- `ARCH` defaults to `sm_100a` so the build matches B200 by default. Override with
  `make ARCH=sm_103a ...` on B300.
- `tile_n=64` or `128` is supported in the current compare path.
- `stages=2` or `4` is supported.
- The reported `est_b_*` values are analytic estimates derived from tile geometry, not hardware counter reads.

## `bench_umma_1sm_2sm_cost`

This benchmark strips the comparison down to the BF16 UMMA instruction itself:

- `1sm`: `M=128, N=tile_n, K=16`
- `2sm`: `M=256, N=tile_n, K=16`
- SS layout only

It is meant to answer the instruction-level version of the same question as the CUTLASS benchmark:

- does `2sm` cost more or less cycles per MMA issue
- after normalizing by total FLOPs, is `2sm` actually more efficient

Important summary fields:

- `cycles_per_mma_ratio`: raw instruction cost ratio
- `cycles_per_flop_ratio`: normalized cost ratio
- `flops_per_cycle_ratio`: normalized throughput ratio

Example:

```bash
make bench_umma_1sm_2sm_cost
./bench_umma_1sm_2sm_cost --mode=compare --tile-n=128 --depth=256 --iters=1000
./bench_umma_1sm_2sm_cost --mode=1sm --tile-n=64 --depth=128 --iters=2000
./bench_umma_1sm_2sm_cost --mode=2sm --tile-n=64 --depth=128 --iters=2000
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
