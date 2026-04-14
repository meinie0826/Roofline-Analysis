# FA4 Ablation Experiment

通过渐进式禁用优化，分析 FlashAttention-4 (Blackwell/SM100) 各优化对性能的贡献。

## Kernel 演进阶段

```
Stage 0: Baseline     → 无 Blackwell 特定优化 (naive)
Stage 1: +PingPong    → q_stage=2 (异步 TMEM 流水线)
Stage 2: +CondRescale → 条件重缩放 (跳过 ~10x 重缩放)
Stage 3: +Ex2Emu      → FMA exp2 仿真 (~2x exp 吞吐)
Stage 4: +LPT         → LPT 调度器 (causal +4-14%)
```

## 快速开始

**重要**：pip 安装的 `flash-attn-4` 不支持 ablation 接口，必须从源码编译。

### 方案一：一键脚本（推荐）

在 B200 服务器上：
```bash
git clone git@github.com:meinie0826/Roofline-Analysis.git
cd Roofline-Analysis
bash fa4/setup_and_run.sh
```

### 方案二：手动步骤

详见 [QUICKSTART.md](QUICKSTART.md)

### 本地拉取数据

```bash
git pull origin main
ls fa4/results/
```

## 输出文件

运行后在 `fa4/results/` 生成：

| 文件 | 内容 |
|------|------|
| `ablation_*_D128.csv` | 各阶段 TFLOPs/s、TC利用率、正确性验证 |
| `fa4_ablation_*.png` | 屋顶线图 (各资源 ceiling + 实测性能) |
| `ncu_counters_*.csv` | NSight 硬件计数器 (MUFU、FMA、SMEM traffic) |
| `run_*.log` | 完整运行日志 |

## 数据自动推送

脚本运行结束后会自动：
```bash
git add fa4/results/
git commit -m "fa4: add experiment results ..."
git push origin HEAD
```

本地拉取数据：
```bash
git pull origin
```

## 单独运行 benchmark

```bash
# 仅打印屋顶线理论 (无需 GPU)
python fa4/benchmark_ablation_sm100.py --roofline-only

# 指定 seqlen
python fa4/benchmark_ablation_sm100.py --seqlen 2048,4096,8192 --causal-only

# 保存 CSV
python fa4/benchmark_ablation_sm100.py --csv results.csv --plot --plot-dir ./
```

## 依赖

- `torch` (CUDA 12.x)
- `flash-attn-4` (pip 或源码)
- `matplotlib` (绘图)
- `numpy`
- `nsight-compute` (可选，硬件计数器 profiling)
