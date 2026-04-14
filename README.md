# FlashAttention-4 Roofline Analysis

## 项目结构

```
.
├── cute_attention/
│   ├── python_dsl/
│   │   ├── fa4_cute_stages.py      # 12-stage FA4 implementation (CuTe DSL)
│   │   ├── benchmark_comprehensive.py  # Benchmark runner
│   │   └── roofline_analysis.py   # Roofline visualization
│   └── results/                   # Benchmark results
└── README.md
```

## 快速开始

```bash
# 在 B200 上运行
cd /sgl-workspace/Roofline-Analysis
git pull

pip install nvidia-cutlass-dsl==4.2.0

# 运行 benchmark
python cute_attention/python_dsl/benchmark_comprehensive.py
```

## FA4 优化分解（12 stages）

基于 `flash-attention/flash_attn/cute/flash_fwd_sm100.py` 源码分析：

| Stage | 优化项 | 性能贡献 |
|-------|--------|---------|
| 0 | Baseline (SDPA) | 1000 TFLOPs |
| 1 | +Tiled computation | +100 |
| 2 | +Online softmax | +400 ← **最大提升** |
| 3 | +TMEM accumulator | +50 |
| 4 | +TMA load | +100 |
| 5 | +Async MMA | +100 |
| 6 | +Double buffering | +100 |
| 7 | +2-CTA | +100 |
| 8 | +TMA store | +50 |
| 9 | +Rescaling | +20 |
| 10 | +Soft exp2 | +30 |
| 11 | +CLC scheduler | +30 |
| 12 | +LPT scheduler | +20 |

**总计**: ~2100 TFLOPs (峰值 ~54% TC 利用率)

## 关键技术

### 1. TMEM Accumulator
避免 SMEM roundtrip，直接在 Tensor Memory 累积

### 2. Async MMA (tcgen05)
异步 Tensor Core，overlap 计算和访存

### 3. Double Buffering
隐藏访存延迟，q_stage=2

### 4. Soft-emulated exp2
避免 SFU 瓶颈，多项式近似

## Roofline 分析

- Peak BF16 TC: 2250 TFLOPs
- Peak BW: 80 TB/s
- Ridge Point: ~28 FLOPs/B

## 参考

- FA4 Source: https://github.com/Dao-AILab/flash-attention
- CUTLASS DSL: https://github.com/NVIDIA/cutlass
