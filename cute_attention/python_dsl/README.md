# FlashAttention Implementation Study

## Project Structure

```
cute_attention/python_dsl/
├── kernels/          # Kernel implementations
│   └── stage0_attention.py   # Naive attention (baseline)
├── tests/            # Correctness tests
│   └── test_correctness.py
├── benchmark.py      # Performance benchmark
└── README.md
```

## Usage

```bash
cd cute_attention/python_dsl

# Run correctness tests
python benchmark.py --test

# Run benchmark
python benchmark.py --bench

# Run both
python benchmark.py --test --bench
```

## Optimization Stages

- **Stage 0**: Naive attention (baseline)
  - Each CTA processes one query row
  - Global memory access for each K, V element
  - Expected: ~0.1 TFLOPs

- **Stage 1**: Tiled attention (FA2 algorithm)
  - Block-based computation
  - Online softmax
  - Expected: ~600 TFLOPs

- **Stage 2**: +TMEM accumulator
  - Use Tensor Memory for accumulation
  - Avoid SMEM roundtrip
  - Expected: ~650 TFLOPs

- **Stage 3+**: Advanced optimizations...
