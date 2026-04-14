# FlashAttention Kernel 演进分析

从 Naive 到优化的渐进式实现，分析每个优化的性能贡献。

## Kernel 演进路径

### Stage 0: Naive
**特点**：最基础的实现
- 每个线程计算一个输出元素 O[i,j]
- 直接从全局内存读取 Q, K, V
- 完整的 N×N attention matrix
- 无任何优化

**瓶颈**：
- 全局内存带宽受限
- O(N²) 的中间结果存储
- 无数据复用

### Stage 1: Tiled (Block-wise)
**优化点**：分块计算
- 将 Q, K, V 分成 128×128 的 tiles
- 每个 CTA (thread block) 计算一个 output tile
- 减少全局内存访问次数

**性能提升**：~2-3x
- 每个元素只需加载一次（vs N 次在 naive）

### Stage 2: Shared Memory
**优化点**：利用 shared memory 缓存
- Q, K, V tiles 先加载到 shared memory
- block 内所有线程复用
- 减少 global memory traffic

**性能提升**：~2-4x
- Global memory 访问减少 ~128x（tile size）

### Stage 3: Tensor Core (WMMA)
**优化点**：使用 Tensor Core
- warp-level MMA 指令 (ldmatrix, mma)
- BF16 输入，FP32 累加
- 矩阵乘法吞吐量提升 16x

**性能提升**：~4-8x
- 接近峰值 FLOPs

### Stage 4: Final Optimized
**优化点**：
- 在线 Softmax（避免存储 N×N attention）
- 软件流水线（重叠计算和内存访问）
- 寄存器分配优化
- 因果注意力专用调度（可选）

**性能提升**：~1.5-2x over Stage 3
- 内存占用从 O(N²) → O(N)
- 隐藏内存延迟

## Roofline 分析目标

```
TFLOPs/s
  ^
  |          Stage 4 ●─── TC Roofline
  |         /
  |    Stage 3 ●
  |       /
  |  Stage 2 ●
  |     /
  |Stage 1 ●
  |   /
  |Stage 0 ●
  |_________________________________>
```

每个阶段会清晰展示：
- 当前瓶颈资源（TC/SMEM/BW）
- 优化后瓶颈转移
- 理论 vs 实测性能差距

## 项目结构

```
attention_kernels/
├── kernels/
│   ├── stage0_naive.cu         # 最基础实现
│   ├── stage1_tiled.cu         # 分块计算
│   ├── stage2_shared_mem.cu    # Shared memory 优化
│   ├── stage3_tensor_core.cu   # Tensor Core MMA
│   └── stage4_final.cu         # 最终优化版本
├── python/
│   ├── attention_wrapper.py    # Python binding
│   ├── benchmark.py            # 性能测试框架
│   └── roofline_analysis.py    # 屋顶线分析
├── results/                    # 实验数据输出
└── scripts/
    └── run_experiment.sh       # 运行脚本
```

## 编译和运行

```bash
# 编译所有 kernel
mkdir build && cd build
cmake ..
make -j$(nproc)

# 运行 benchmark
python python/benchmark.py --seqlen 1024,2048,4096,8192

# 生成屋顶线图
python python/roofline_analysis.py --plot
```

## 硬件要求

- NVIDIA GPU (Ampere+ 推荐，用于 Tensor Core BF16)
- CUDA 12.x
- GCC 11+

## 预期结果

| Stage | TFLOPs/s (估计) | 主要瓶颈 | 优化贡献 |
|-------|----------------|----------|---------|
| 0 | 0.5-1 | Global BW | Baseline |
| 1 | 2-4 | Global BW | 2-4x (tiling) |
| 2 | 8-15 | Compute | 3-4x (SMEM) |
| 3 | 50-80 | TC/EXP | 6-8x (Tensor Core) |
| 4 | 100-150 | TC Roofline | 1.5-2x (online softmax) |
