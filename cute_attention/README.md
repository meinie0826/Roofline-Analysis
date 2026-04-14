# FlashAttention CuTe Implementation

从零开始实现的 FlashAttention，使用 NVIDIA CuTe 框架，展示从 naive 到最终优化的完整过程。

## 优化阶段

### Stage 0: Naive Baseline
- **实现**: 每个线程处理一个 query position
- **特点**: 直接从全局内存读取，无数据复用
- **性能**: ~0.5-1 TFLOPs/s
- **代码**: [kernels/stage0_naive.cuh](kernels/stage0_naive.cuh)

### Stage 1: Tiled Computation
- **优化**: 每个 CTA 处理一个 query tile (kBlockM positions)
- **特点**: Shared memory 缓存 Q, K tiles
- **性能**: ~2-4 TFLOPs/s
- **提升**: 3-4x vs Stage 0
- **代码**: [kernels/stage1_tiled.cuh](kernels/stage1_tiled.cuh)

### Stage 2: Optimized SMEM Layout
- **优化**: Padding 避免 bank conflict，向量化加载
- **特点**: 高效的 smem -> register 数据传输
- **性能**: ~8-15 TFLOPs/s
- **提升**: 3-4x vs Stage 1
- **代码**: [kernels/stage2_smem.cuh](kernels/stage2_smem.cuh)

### Stage 3: Tensor Core MMA
- **优化**: 使用 warp-level GMMA 指令
- **特点**: Softmax 与 MMA 流水线
- **性能**: ~50-80 TFLOPs/s
- **提升**: 5-7x vs Stage 2
- **代码**: [kernels/stage3_mma.cuh](kernels/stage3_mma.cuh)

### Stage 4: Final Optimized
- **优化**: 在线 Softmax (Flash Attention 算法) + 软件流水线
- **特点**: TMA + MMA 重叠，优化的寄存器分配
- **性能**: ~100-150 TFLOPs/s
- **提升**: 1.5-2x vs Stage 3
- **代码**: [kernels/stage4_final.cuh](kernels/stage4_final.cuh)

## 编译和运行

### 依赖
- CUDA 12.0+
- CUTLASS 3.5+
- Python 3.8+ (PyTorch 2.0+)

### 编译
```bash
# 安装 CUTLASS
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j install

# 编译 CuTe Attention
cd cute_attention
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90;100
make -j
```

### 运行 Benchmark
```bash
cd cute_attention/scripts
bash run_all.sh
```

## 性能对比

预期性能对比（B200 GPU，BF16，Causal=True）：

| Stage | Seqlen=1K | Seqlen=4K | Seqlen=8K | TC Util |
|-------|-----------|-----------|-----------|---------|
| 0     | 1 TF      | 0.8 TF    | 0.6 TF    | <1%     |
| 1     | 3 TF      | 2.5 TF    | 2 TF      | <1%     |
| 2     | 10 TF     | 9 TF      | 8 TF      | <1%     |
| 3     | 60 TF     | 50 TF     | 45 TF     | 3-4%    |
| 4     | 120 TF    | 100 TF    | 90 TF     | 5-6%    |

## 项目结构

```
cute_attention/
├── kernels/
│   ├── stage0_naive.cuh      # Baseline
│   ├── stage1_tiled.cuh      # +Tiling
│   ├── stage2_smem.cuh       # +SMEM optimization
│   ├── stage3_mma.cuh        # +Tensor Core
│   ├── stage4_final.cuh      # +Online softmax
│   └── flash_attention.cuh  # Interface
├── python/
│   ├── cute_attention/       # Python module
│   └── benchmark.py          # Benchmark script
├── scripts/
│   └── run_all.sh           # Run everything
└── CMakeLists.txt
```

## 学习要点

1. **CuTe 框架**: 学习 NVIDIA 的高级 CUDA 抽象
2. **Tensor Core 编程**: 理解 GMMA 指令的使用
3. **内存优化**: SMEM 布局、bank conflict 避免
4. **算法优化**: 在线 Softmax (Flash Attention 的核心)
5. **流水线技术**: TMA + MMA 重叠

## 参考

- [FlashAttention-3 Paper](https://arxiv.org/abs/2312.03519)
- [CuTe Tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
