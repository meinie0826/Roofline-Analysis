# FA4 Ablation 快速运行指南

## 问题：pip 安装的 FA4 不支持 Ablation

```
Ablation parameters support:
  ✗ _ablation_q_stage
  ✗ _ablation_no_lpt
  ✗ _ablation_rescale_threshold
  ✗ _ablation_ex2_emu_freq
```

**解决方案**：需要从源码编译 FA4。

## 在 B200 服务器上一键运行

```bash
# 1. Clone 你的仓库
git clone git@github.com:meinie0826/Roofline-Analysis.git
cd Roofline-Analysis

# 2. 运行一键脚本（会自动 clone FA4、编译、运行、推送结果）
bash fa4/setup_and_run.sh
```

脚本会自动：
- Clone flash-attention 源码
- 复制 benchmark 脚本到 FA4 目录
- 从源码编译安装 FA4（5-10分钟）
- 运行 ablation 实验
- 将结果推送到 git

## 本地拉取结果

```bash
git pull origin main
ls fa4/results/
```

## 输出文件

```
fa4/results/
├── ablation_noncausal_D128_20260414T081234Z.csv  # 完整数据
├── ablation_causal_D128_20260414T081234Z.csv
├── fa4_ablation_fwd_hdim128_sl4096_causal0.png   # 屋顶线图
├── fa4_ablation_fwd_hdim128_sl4096_causal1.png
└── run_20260414T081234Z.log                      # 运行日志
```

## 手动步骤（如果自动脚本失败）

```bash
# 1. Clone FA4
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# 2. 复制 benchmark
cp /path/to/Roofline-Analysis/fa4/benchmark_ablation_sm100.py benchmarks/

# 3. 编译安装
pip install -e . --no-build-isolation

# 4. 运行实验
cd /path/to/Roofline-Analysis
bash fa4/run_experiment.sh /path/to/flash-attention
```

## 服务器环境要求

- CUDA 12.x
- GCC 11+ 或 12+
- Python 3.10+
- 至少 32GB 内存（编译需要）
- B200/B100 GPU（否则无法运行 FA4 kernel）

## 预期运行时间

- 编译：5-10 分钟
- 非因果实验：~5 分钟
- 因果实验：~5 分钟
- 总计：15-20 分钟
