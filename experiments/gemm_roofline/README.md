# GEMM Roofline Analysis

This experiment analyzes the performance characteristics of GEMM (General Matrix Multiply) operations across different matrix shapes, visualizing the transition from memory-bound to compute-bound regions using the Roofline model.

## Background

The Roofline model is a performance model that provides a visual representation of performance limitations:

- **Memory-Bound Region**: When arithmetic intensity is low, performance is limited by memory bandwidth
- **Compute-Bound Region**: When arithmetic intensity is high, performance is limited by peak compute throughput
- **Ridge Point**: The arithmetic intensity where memory-bound transitions to compute-bound

For GEMM `C = A @ B` where A is M×K and B is K×N:
- **FLOPs** = 2 × M × N × K (multiply-add operations)
- **Bytes accessed** = M×K + K×N + M×N (reading A, B and writing C)
- **Arithmetic Intensity** = (2 × M × N × K) / (M×K + K×N + M×N) FLOPs/Byte

## Key Observations

1. **Small matrices** (especially small K): Low arithmetic intensity → Memory-bound
2. **Large matrices**: High arithmetic intensity → Compute-bound
3. **Square matrices** (M=N=K): Arithmetic intensity ≈ 2K/3

### Transition Analysis

As matrix size increases:
- Small shapes (e.g., 64×64×32): AI ≈ 7.5 → Memory-bound
- Medium shapes (e.g., 512×512×512): AI ≈ 170 → Near ridge point
- Large shapes (e.g., 4096×4096×4096): AI ≈ 1365 → Compute-bound

## Files

- `benchmark_roofline.py` - Benchmark script for testing various GEMM shapes
- `plot_roofline.py` - Roofline plotting script
- `gemm_kernel.py` - CuTeDSL GEMM kernel (optional, can also use PyTorch matmul)
- `run_benchmark.sh` - Runner script for the complete analysis

## Usage

### On B300 Server

```bash
# Set GPU specs for B300
export PEAK_TFLOPS=2500  # FP16 Tensor Core peak
export PEAK_BANDWIDTH=8000  # HBM3e bandwidth

# Run with default balanced shapes
./run_benchmark.sh

# Or run comprehensive sweep
SHAPE_TYPE=comprehensive ./run_benchmark.sh

# Or focus on memory-heavy shapes
SHAPE_TYPE=memory_heavy ./run_benchmark.sh
```

### Direct Python Usage

```bash
# Run benchmark
python3 benchmark_roofline.py \
    --shape-type balanced \
    --warmup 5 \
    --iterations 20 \
    --output-dir results \
    --peak-tflops 2500 \
    --peak-bandwidth 8000

# Generate plots
python3 plot_roofline.py results/roofline_results_*.json \
    --peak-tflops 2500 \
    --peak-bandwidth 8000
```

## Shape Types

| Type | Description | Use Case |
|------|-------------|----------|
| `balanced` | Square matrices (M=N=K), doubling from 64 to 8192 | Basic roofline curve |
| `memory_heavy` | Small K values, varying M×N | Memory-bound region |
| `compute_heavy` | Large K values, small M×N | Compute-bound region |
| `layer_like` | Transformer layer shapes | Real-world applications |
| `comprehensive` | Full sweep across all dimensions | Complete analysis |

## Expected Results

### B300 GPU (Blackwell)

- **Peak FP16 TC**: 2500 TFLOPS (sparse) / 1250 TFLOPS (dense)
- **Peak Bandwidth**: ~8000 GB/s (HBM3e)
- **Ridge Point**: AI ≈ 312 FLOPs/Byte (dense) or 156 FLOPs/Byte (sparse)

### Performance Characteristics

For PyTorch `torch.matmul` (uses cuBLAS Tensor Cores):

| Shape (M×N×K) | AI | Expected GFLOPS | Region |
|--------------|-----|-----------------|--------|
| 64×64×32 | 7.5 | ~60,000 | Memory-bound |
| 256×256×256 | 85 | ~340,000 | Near ridge |
| 1024×1024×1024 | 341 | ~800,000 | Compute-bound |
| 4096×4096×4096 | 1365 | ~1,200,000 | Compute-bound |

## Customization

### Adding Custom Shapes

Modify `generate_shapes()` in `benchmark_roofline.py`:

```python
# Add your custom shape progression
shapes.append((your_M, your_N, your_K))
```

### Different Data Types

Change `dtype` parameter:

```python
# FP16 (default) - uses Tensor Cores
dtype=torch.float16

# FP32 - uses CUDA cores
dtype=torch.float32

# BF16 - uses Tensor Cores (if supported)
dtype=torch.bfloat16
```

## Dependencies

```bash
pip install torch matplotlib numpy
```

For CuTe DSL kernel (optional):
```bash
pip install nvidia-cutlass-dsl
```

## Output

The benchmark produces:
1. JSON results file with all metrics
2. PNG roofline plot
3. PNG transition analysis plot
4. Console summary statistics

Results are saved in `results/roofline_results_YYYYMMDD_HHMMSS.json`

## References

- [Roofline Model Paper](https://dl.acm.org/doi/10.1145/1498765.1498785) - Williams et al., CACM 2009
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) - Profiling tool
- [CUTLASS](https://github.com/NVIDIA/cutlass) - High-performance GEMM kernels
