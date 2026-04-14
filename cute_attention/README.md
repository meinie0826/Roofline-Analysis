# FlashAttention CuTe DSL Implementation

完整的 FlashAttention 实现，使用 Python CuTe DSL（FA4 方式），展示从 naive 到最终优化的完整过程。

## 项目结构

```
cute_attention/
├── python_dsl/           # Python CuTe DSL 实现 (推荐)
│   ├── flash_attention_dsl.py      # 核心实现 (Stage 0-4)
│   ├── benchmark_all_stages.py     # 自动化 benchmark
│   ├── run_benchmark.sh            # 完整执行脚本
│   └── analyze_results.py          # 结果分析
├── results/              # Benchmark 结果 (自动生成)
└── kernels/              # C++ CuTe 实现 (可选)
```

## 快速开始

### 在 B200 服务器上运行

```bash
# 1. Clone 仓库
cd /sgl-workspace
git clone git@github.com:meinie0826/Roofline-Analysis.git
cd Roofline-Analysis

# 2. 安装依赖
pip install nvidia-cutlass-dsl==4.2.0

# 3. 运行完整 benchmark (自动提交结果)
cd cute_attention/python_dsl
bash run_benchmark.sh
```

### 在本地分析结果

```bash
# 1. 拉取结果
git pull origin main

# 2. 分析最新结果
cd cute_attention/python_dsl
python analyze_results.py --latest

# 3. 或分析特定文件
python analyze_results.py --file benchmark_20260414T093000Z.json
```

## 输出示例

### B200 运行输出

```
================================================================
 FlashAttention CuTe DSL Benchmark
================================================================
 Timestamp: 20260414T093000Z
 Project:   /sgl-workspace/Roofline-Analysis
 Results:    cute_attention/results
================================================================

[1/4] Saving GPU information...
NVIDIA B200, 580.82.07, 183359 MiB

[2/4] Running benchmarks...

  Seqlen=1024, Batch=32
  --------------------------------------------------------------------------
  Stage     Time(ms)     TFLOPs      TC Util%     Speedup    
  --------------------------------------------------------------------------
  0         125.340      0.8         0.04%        1.0x       
  1         32.100       3.2         0.14%        4.0x       
  2         10.250       10.0        0.44%        12.5x      
  3         1.820        56.2        2.50%        70.3x      
  4         0.850        120.3       5.35%        150.4x     

[4/4] Committing results to git...
  ✓ Results committed and pushed

================================================================
 DONE!
================================================================
 Results: cute_attention/results/benchmark_20260414T093000Z.json
 Pull locally with: git pull origin main
================================================================
```

### 本地分析输出

```
================================================================
  FlashAttention CuTe DSL Benchmark Analysis
================================================================
  GPU: NVIDIA B200
  Compute Capability: SM_10.0
  Peak TFLOPs: 2250
================================================================

  Summary by Stage
================================================================

  Stage   Avg TFLOPs      Max TFLOPs      Avg TC Util%    
  ------------------------------------------------------
  0        0.9             1.1             0.04            
  1        3.5             4.2             0.16            
  2        11.2            12.8            0.50            
  3        62.3            73.8            2.77            
  4        115.6           122.1           5.14            

  Overall Speedup (Stage 4 vs Stage 0): 128.4x
```

## 优化阶段

| Stage | 描述 | 关键技术 | 性能 |
|-------|------|---------|------|
| 0 | Naive | 单线程/query | ~1 TF |
| 1 | Tiled | SMEM 缓存 | ~4 TF |
| 2 | Memory Opt | Bank conflict free | ~12 TF |
| 3 | Tensor Core | GMMA MMA | ~70 TF |
| 4 | Final | Online softmax + Pipe | ~120 TF |

## 自动化流程

`run_benchmark.sh` 会：

1. 检测 GPU 信息
2. 运行所有 stage 的 benchmark
3. 保存结果到 JSON
4. 自动提交到 git

你在本地只需要：

```bash
git pull
python cute_attention/python_dsl/analyze_results.py --latest
```

## 数据格式

结果保存在 `cute_attention/results/benchmark_*.json`:

```json
{
  "timestamp": "2026-04-14T09:30:00Z",
  "device": {
    "name": "NVIDIA B200",
    "compute_capability": "10.0",
    "memory_gb": 183.359,
    "peak_tflops": 2250.0
  },
  "results": [
    {
      "stage": 0,
      "seqlen": 1024,
      "batch": 32,
      "avg_ms": 125.34,
      "tflops": 0.8,
      "tc_util_pct": 0.04
    }
  ]
}
```

## 参考

- [FA4 Python CuTe Interface](../flash-attention/flash_attn/cute/interface.py)
- [CUTLASS DSL Documentation](https://github.com/NVIDIA/cutlass/tree/main/python)
