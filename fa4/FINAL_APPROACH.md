# FlashAttention-4 性能分析方案

## 当前状态

由于以下原因，我们不自己写 kernel，而是直接使用 FA4 源码：

1. **FA4 使用 CuTe 框架**：这是 NVIDIA 的高级抽象，比纯 CUDA 更高效
2. **Ablation 接口不公开**：`_ablation_*` 参数是内部调试接口
3. **时间有限**：重写 kernel 工作量大且容易出错

## 方案

### 方案 1：直接使用 FA4 benchmark（推荐）

FA4 源码自带 benchmark，直接运行：

```bash
cd /sgl-workspace/Roofline-Analysis/flash-attention/hopper

# 编译 FA4（如果还没编译）
cd ..
pip install -e . --no-build-isolation
cd hopper

# 运行 benchmark
python3 benchmark_attn.py \
    --seqlens 512,1024,2048,4096,8192 \
    --causal \
    --output results.json
```

### 方案 2：对比 FA4 vs SDPA

```bash
cd /sgl-workspace/Roofline-Analysis

python3 fa4/benchmark_comparison.py \
    --seqlen 512,1024,2048,4096,8192 \
    --output fa4/results/comparison.json
```

## 理论分析（不需要运行 kernel）

FA4 的关键优化已经在理论层面可以分析：

### 优化点及性能贡献（估计）

| 优化 | 描述 | 预期提升 | 依据 |
|------|------|---------|------|
| **q_stage=2 (Ping-pong)** | 异步 TMEM 流水线 | +30-50% | Hopper 新特性 |
| **Conditional Rescale** | 跳过不必要的 rescale | +5-10% | 减少 exp 操作 |
| **Exp2 Emulation** | FMA + MUFU 混合 | +10-20% | EXP 瓶颈缓解 |
| **LPT Scheduler** | Causal 专用调度 | +5-15% | 负载均衡 |

### Roofline 分析（理论）

```
B200 Peak BF16: 2250 TFLOPs/s
MUFU EXP:       562  GFLOPs/s (2250/4)

TC:EXP 比例 = 4:1 (理论)
实际应用中 softmax 的 exp 成为瓶颈

FA4 通过以下方式缓解：
- 软件 exp 仿真（利用 FP32 FMA 单元）
- 与 TC MMA 流水线重叠
```

## 在 B200 服务器上运行

```bash
# 1. Clone 仓库
git clone git@github.com:meinie0826/Roofline-Analysis.git
cd Roofline-Analysis

# 2. FA4 已在 flash-attention 目录

# 3. 编译 FA4（如果需要）
cd flash-attention
pip install -e . --no-build-isolation

# 4. 运行 benchmark
cd ../fa4
python3 benchmark_comparison.py --seqlen 512,1024,2048,4096,8192

# 5. 或使用 FA4 自带 benchmark
cd ../flash-attention/hopper
python3 benchmark_attn.py --causal
```

## 输出

- `fa4/results/comparison.json`：FA4 vs SDPA 性能数据
- `fa4/results/gpu_info.txt`：GPU 信息
- 控制台输出：实时性能对比

## 注意

由于 FA4 需要 SM90+ (Hopper/Blackwell)，如果运行在非兼容 GPU 上会报错。
