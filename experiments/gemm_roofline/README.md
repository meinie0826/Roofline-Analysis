# GEMM Roofline Analysis

This experiment analyzes GEMM performance across different matrix shapes and implementations, visualizing the transition from memory-bound to compute-bound regions using the Roofline model.

## Background

The Roofline model provides a visual representation of performance limitations:

- **Memory-Bound Region**: When arithmetic intensity is low, performance is limited by memory bandwidth
- **Compute-Bound Region**: When arithmetic intensity is high, performance is limited by peak compute throughput
- **Ridge Point**: The arithmetic intensity where memory-bound transitions to compute-bound

For GEMM `C = A @ B` where A is M×K and B is K×N:
- **FLOPs** = 2 × M × N × K (multiply-add operations)
- **Bytes accessed** = M×K + K×N + M×N (reading A, B and writing C)
- **Arithmetic Intensity** = (2 × M × N × K) / (M×K + K×N + M×N) FLOPs/Byte

## Implementations Tested

| Backend | Precision | Peak TFLOPS | Notes |
|---------|-----------|-------------|-------|
| **cuBLAS** (torch.matmul) | BF16 | 1250 | PyTorch default, highly optimized |
| **DeepGEMM** | FP8 | 5000 | DeepSeek's optimized kernel, SM90+ |

## Key Observations

1. **Small matrices**: Low arithmetic intensity → Memory-bound
2. **Large matrices**: High arithmetic intensity → Compute-bound
3. **FP8 vs BF16**: FP8 has 4x higher peak TFLOPS but same memory bandwidth

### Arithmetic Intensity by Shape

| Shape (M=N=K) | AI (FP8) | AI (BF16) | Region |
|---------------|----------|-----------|--------|
| 64 | 21 | 42 | Memory-bound |
| 512 | 170 | 341 | Near ridge |
| 4096 | 1365 | 2731 | Compute-bound |

## Files

```
gemm_roofline/
├── README.md                  # This file
├── benchmark_deepgemm.py      # Main benchmark script
├── test_deepgemm.py           # DeepGEMM installation test
├── analyze_roofline.py        # Plotting and analysis script
├── run_deepgemm.sh            # Runner script
├── requirements.txt           # Dependencies
└── results/                   # Output directory
```

## Usage

### Quick Start

```bash
# Run full analysis
./run_deepgemm.sh
```

### Manual Steps

```bash
# 1. Test DeepGEMM installation
python3 test_deepgemm.py

# 2. Run benchmark
python3 benchmark_deepgemm.py --shape-type balanced --output-dir results

# 3. Analyze and plot
python3 analyze_roofline.py results/deepgemm_roofline_*.json
```

### On B300 Server

```bash
# Set GPU specs for B300
export PEAK_FP8_TFLOPS=5000
export PEAK_BF16_TFLOPS=1250
export PEAK_BANDWIDTH_GBPS=8000

# Run comprehensive sweep
python3 benchmark_deepgemm.py \
    --shape-type balanced \
    --warmup 5 \
    --iterations 20 \
    --output-dir results

# Generate plots
python3 analyze_roofline.py results/deepgemm_roofline_*.json
```

## Shape Types

| Type | Description | Use Case |
|------|-------------|----------|
| `balanced` | Square matrices (M=N=K) | Basic roofline curve |
| `memory_heavy` | Small K values | Memory-bound region detail |
| `compute_heavy` | Large K values | Compute-bound region detail |
| `inference_like` | Transformer layer shapes | Real-world performance |

## Installing DeepGEMM

DeepGEMM requires SM90 (Hopper) or SM100 (Blackwell) GPU:

```bash
# Clone with submodules
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM

# Development install
./develop.sh

# Or regular install
./install.sh
```

### Requirements

- NVIDIA SM90 or SM100 GPU (H100/H800/B100/B200/B300)
- Python 3.8+
- CUDA 12.3+ (12.9+ recommended for best performance)
- PyTorch 2.1+
- CUTLASS 4.0+ (included as submodule)

## Expected Results

### B300 GPU (Blackwell)

- **Peak FP8**: 5000 TFLOPS
- **Peak BF16**: 1250 TFLOPS  
- **Peak BW**: 8000 GB/s
- **Ridge Point FP8**: AI = 625
- **Ridge Point BF16**: AI = 156

### Performance Comparison

For large shapes (M=N=K=4096):

| Backend | Precision | Expected TFLOPS | Efficiency |
|---------|-----------|-----------------|------------|
| cuBLAS | BF16 | ~1000 | ~80% |
| DeepGEMM | FP8 | ~3500-4000 | ~70-80% |

## Dependencies

```bash
pip install torch matplotlib numpy
```

For DeepGEMM:
```bash
# See DeepGEMM repository for detailed installation
```

## Output

The benchmark produces:

1. **JSON results file**: Contains all metrics for each shape/backend
2. **Roofline plot**: Shows performance vs arithmetic intensity
3. **Comparison plot**: Bar chart comparing backends for same shapes

Results are saved in `results/deepgemm_roofline_YYYYMMDD_HHMMSS.json`

## Troubleshooting

### DeepGEMM not found

```
Error: No module named 'deep_gemm'
```

Solution: Install DeepGEMM following the instructions above.

### CUDA out of memory

```
RuntimeError: CUDA out of memory
```

Solution: Reduce matrix sizes or use `--shape-type memory_heavy` for smaller K values.

### Compute capability mismatch

```
RuntimeError: DeepGEMM requires SM90 or SM100
```

Solution: DeepGEMM only works on Hopper/Blackwell GPUs. For other GPUs, use the cuBLAS BF16 benchmark.

## References

- [Roofline Model Paper](https://dl.acm.org/doi/10.1145/1498765.1498785) - Williams et al., CACM 2009
- [DeepGEMM Repository](https://github.com/deepseek-ai/DeepGEMM)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [PyTorch Roofline Blog](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
