# FlashAttention 综合性能分析

完整的研究级 benchmark，包含：
- **Baseline**: PyTorch SDPA, FlashAttention 2/3/4
- **我们的实现**: Stage 0-4 渐进优化
- **正确性验证**: 所有 kernel 与 SDPA 对比
- **Roofline 分析**: 可视化优化在 roofline 上的位置
- **消融实验**: 量化每个优化的贡献

## 模型配置

基于最新开源模型的常见 shape：

| 模型 | Seqlen | Batch | Heads | Head Dim | GQA | 说明 |
|------|--------|-------|-------|----------|-----|------|
| GPT-3-short | 1024 | 自动 | 32 | 128 | No | 短序列推理 |
| LLaMA-2-7B | 2048 | 自动 | 32 | 128 | No | 标准训练 |
| LLaMA-3-8B | 4096 | 自动 | 32 | 128 | Yes (8) | GQA 训练 |
| Qwen-72B | 8192 | 自动 | 64 | 128 | Yes (8) | 长序列 |
| GLM-4-9B | 16384 | 自动 | 32 | 128 | Yes (4) | 超长序列 |
| Kimi-long | 32768 | 自动 | 32 | 128 | Yes (4) | 32K 上下文 |

## 快速开始

### 在 B200 服务器上运行

```bash
cd /sgl-workspace/Roofline-Analysis
git pull

# 完整 benchmark（约 10 分钟）
bash cute_attention/python_dsl/run_comprehensive.sh

# 快速测试（约 2 分钟）
bash cute_attention/python_dsl/run_comprehensive.sh --quick
```

### 在本地分析结果

```bash
git pull origin main

# 运行分析
cd cute_attention/python_dsl
python3 roofline_analysis.py --file ../results/benchmark_comprehensive_*.json

# 查看结果
open ../results/roofline_analysis.png
open ../results/ablation_analysis.png
```

## 输出文件

每次运行会生成以下文件：

```
cute_attention/results/
├── benchmark_comprehensive_20260414T102548Z.json  # 原始 benchmark 数据
├── efficiency_analysis.json                        # 效率分析
├── roofline_analysis.png                           # Roofline 图
├── ablation_analysis.png                            # 消融分析图
├── run_comprehensive_20260414T102548Z.log          # 完整日志
└── gpu_info.txt                                    # GPU 信息
```

所有结果自动提交到 git。

## Roofline 分析

### 理论背景

**Roofline 模型**描述了性能上限：

```
Performance = min(Peak FLOPs, AI × Peak Bandwidth)
```

其中：
- **Arithmetic Intensity (AI)**: FLOPs/Byte，算术强度
- **Ridge Point**: AI 峰值 = Peak FLOPs / Peak Bandwidth
- **Memory Bound**: AI < Ridge Point，性能受限于带宽
- **Compute Bound**: AI > Ridge Point，性能受限于计算

### B200 参数

| 参数 | 值 |
|------|-----|
| Peak BF16 TFLOPs | 2250 |
| Peak HBM Bandwidth | 80 TB/s |
| Ridge Point | ~28 FLOPs/Byte |
| Theoretical TC Util | 50% |

### Attention 的 Arithmetic Intensity

```
AI = FLOPs / Bytes
   = (2 × S × S × H × D) / (4 × B × S × H × D)
   ≈ S / (2 × B)
```

典型值：
- S=1024, B=32: AI ≈ 16 FLOPs/B (Memory Bound)
- S=4096, B=8: AI ≈ 256 FLOPs/B (Compute Bound)
- S=32768, B=1: AI ≈ 16384 FLOPs/B (Deep Compute Bound)

### 优化方法在 Roofline 上的体现

1. **Tiling (Stage 1)**: 
   - 提高 data reuse，不改变 AI
   - 主要减少带宽开销

2. **Memory Optimization (Stage 2)**:
   - 优化 SMEM 布局，减少 bank conflict
   - 不改变 AI，提高实际带宽利用率

3. **Tensor Core (Stage 3)**:
   - 使用 GMMA 指令，接近峰值计算
   - 在 compute bound 区域效果显著

4. **Online Softmax + Pipeline (Stage 4)**:
   - 减少 HBM 访问（只存 O 和 LSE）
   - AI 基本不变，但更接近 roofline

## 消融实验

每个优化的贡献：

| 优化 | 加速比 | 主要改进 |
|------|--------|----------|
| Stage 0 → 1 (Tiling) | ~2.5x | 数据复用 |
| Stage 1 → 2 (SMEM Opt) | ~1.5x | Bank conflict free |
| Stage 2 → 3 (TC MMA) | ~1.2x | Tensor Core |
| Stage 3 → 4 (Pipeline) | ~1.2x | 隐藏延迟 |

累积加速比：**~5x**

## 正确性验证

所有 kernel 与 PyTorch SDPA 对比：

```python
max_abs_error = (output - reference).abs().max()
relative_error = max_abs_error / reference.abs().max()

assert relative_error < 1e-2  # BF16 精度
```

## 性能目标

在 B200 上：

| Kernel | 目标 TFLOPs | 目标 TC Util |
|--------|------------|--------------|
| SDPA | 200+ | 10%+ |
| FA2 | 600+ | 30%+ |
| FA3 | 900+ | 40%+ |
| FA4 | 1100+ | 50%+ |
| Stage 4 | 1000+ | 45%+ |

## 依赖

```bash
pip install torch matplotlib numpy
pip install nvidia-cutlass-dsl  # 可选，用于 CuTe 实现
pip install flash-attn          # 可选，用于 FA2/3/4 baseline
```

## 参考资料

- [Roofline Model Paper](https://dl.acm.org/doi/10.1145/1498765.1498785)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FA2 Paper](https://arxiv.org/abs/2307.08691)
- [FA3 Paper](https://arxiv.org/abs/2408.04268)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
