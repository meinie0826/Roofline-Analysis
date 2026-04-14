# FlashAttention Kernel 实验运行指南

## 项目概述

自己实现的 FlashAttention kernel，从 naive 到优化的渐进式演进，展示每个优化的性能贡献。

## Kernel 演进路径

| Stage | 名称 | 主要优化 | 预期性能 |
|-------|------|---------|---------|
| 0 | Naive | 无优化（baseline） | 0.5-1 TFLOPs/s |
| 1 | Tiled | 分块计算 | 2-4 TFLOPs/s |
| 2 | Shared Memory | SMEM 缓存 tiles | 8-15 TFLOPs/s |
| 3 | Tensor Core | WMMA MMA 指令 | 50-80 TFLOPs/s |
| 4 | Final | 在线 softmax + 软件流水线 | 100-150 TFLOPs/s |

## 在 B200 服务器上运行

### 方式一：一键运行（推荐）

```bash
# 1. Clone 仓库
git clone git@github.com:meinie0826/Roofline-Analysis.git
cd Roofline-Analysis

# 2. 运行脚本（会自动编译、测试、推送结果）
bash attention_kernels/scripts/run_experiment.sh
```

### 方式二：手动步骤

```bash
# 1. 编译 kernels
cd attention_kernels
mkdir build && cd build
cmake ..
make -j$(nproc)

# 2. 设置环境
cd ../..
export LD_LIBRARY_PATH=$PWD/build:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD:$PYTHONPATH

# 3. 运行 benchmark
python3 attention_kernels/python/benchmark.py \
    --seqlen 1024,2048,4096,8192 \
    --csv results/benchmark.csv \
    --plot \
    --plot-dir results/

# 4. 推送结果
git add results/
git commit -m "attention_kernels: add results"
git push origin main
```

## 本地获取数据

```bash
git pull origin main
ls attention_kernels/results/
```

## 输出文件

```
results/
├── benchmark_20260414T081234Z.csv     # 完整性能数据
├── roofline_sl1024_causal0.png        # 屋顶线图
├── roofline_sl2048_causal0.png
├── roofline_sl4096_causal0.png
├── roofline_sl1024_causal1.png        # 因果注意力图
└── gpu_info.txt                       # GPU 信息
```

## CSV 数据格式

| 列 | 说明 |
|----|------|
| stage | 优化阶段 (0-4) |
| stage_name | 阶段名称 |
| batch_size | 批大小 |
| seq_len | 序列长度 |
| n_heads | 注意力头数 |
| head_dim | 头维度 |
| causal | 是否因果注意力 |
| avg_ms | 平均运行时间 |
| tflops | TFLOPs/s |
| tc_utilization_pct | Tensor Core 利用率 |
| max_diff | 与参考实现的误差 |

## 环境要求

- NVIDIA GPU (Ampere+, 推荐 A100/B200)
- CUDA 12.x
- GCC 11+ 或 compatible
- Python 3.10+
- PyTorch 2.0+
- matplotlib (绘图用)

## 单独测试某个 Stage

```bash
# 只测试 Stage 4
python3 attention_kernels/python/benchmark.py --stage 4 --seqlen 4096

# 只测试 causal
python3 attention_kernels/python/benchmark.py --causal-only --seqlen 4096,8192

# 减少迭代次数（快速测试）
python3 attention_kernels/python/benchmark.py --warmup 3 --rep 10
```

## 代码结构

```
attention_kernels/
├── kernels/
│   ├── stage0_naive.cu          # 最基础实现
│   ├── stage1_tiled.cu          # 分块计算
│   ├── stage2_shared_mem.cu     # SMEM 优化
│   ├── stage3_tensor_core.cu    # Tensor Core MMA
│   └── stage4_final.cu          # 在线 softmax + 流水线
├── python/
│   ├── attention_wrapper.py     # Python binding
│   └── benchmark.py             # Benchmark 框架
├── scripts/
│   └── run_experiment.sh        # 自动运行脚本
├── CMakeLists.txt               # 编译配置
└── README.md                    # 本文档
```

## 性能预期

根据 GPU 不同，预期性能：

### A100 (80GB)
- Peak BF16: 312 TFLOPs/s
- Stage 4 预期: ~150 TFLOPs/s (~48% TC util)

### B200
- Peak BF16: 2250 TFLOPs/s (密集) 或 ~1000 TFLOPs/s (稀疏)
- Stage 4 预期: ~300-500 TFLOPs/s

## 故障排查

### 编译错误
```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GCC 版本
gcc --version

# 需要至少 GCC 11 和 CUDA 12.0
```

### 运行时错误
```bash
# 检查 GPU 内存
nvidia-smi

# 检查库是否加载
ldd build/libattention_kernels.so
```

### 导入错误
```python
# 检查 PYTHONPATH
import sys
sys.path.insert(0, "/path/to/Roofline-Analysis")

# 检查 LD_LIBRARY_PATH
import os
os.environ["LD_LIBRARY_PATH"] = "/path/to/build:" + os.environ.get("LD_LIBRARY_PATH", "")
```
