# FlashAttention Python CuTe DSL Implementation

使用 **Python CuTe DSL**（FA4 的方式）从零实现 FlashAttention，展示从 naive 到最终优化的完整过程。

## 为什么用 Python DSL？

FA4 使用 `nvidia-cutlass-dsl` Python 包，相比 CUDA C++ 有以下优势：

1. **更简单的语法**：不需要处理复杂的模板元编程
2. **自动内存管理**：CuTe DSL 自动处理 shared memory 布局
3. **JIT 编译**：运行时编译，更灵活
4. **易于调试**：Python 代码更容易理解和修改

## 优化阶段

| Stage | 优化点 | 预期性能 (B200) | 提升 |
|-------|--------|----------------|------|
| 0 | Naive baseline | ~1 TFLOPs/s | 1x |
| 1 | Tiled computation | ~4 TFLOPs/s | 4x |
| 2 | Bank-conflict free SMEM | ~12 TFLOPs/s | 12x |
| 3 | Tensor Core MMA | ~70 TFLOPs/s | 70x |
| 4 | Online softmax + pipelining | ~120 TFLOPs/s | 120x |

## 安装

```bash
# 安装 CUTLASS DSL
pip install nvidia-cutlass-dsl==4.2.0

# 需要 CUDA 12.0+
```

## 使用方法

```python
from flash_attention_dsl import flash_attention_forward

# Stage 4 (最优)
out, lse = flash_attention_forward(q, k, v, stage=4, causal=True)

# 测试不同阶段
for stage in range(5):
    out, lse = flash_attention_forward(q, k, v, stage=stage)
```

## 实现细节

### Stage 0: Naive
- 每个线程处理一个 query position
- 直接从全局内存读取，无数据复用
- 性能受限：带宽瓶颈

### Stage 1: Tiling
- 每个 CTA 处理一个 query tile
- Shared memory 缓存 Q, K, V tiles
- 减少 global memory 访问

### Stage 2: Memory Optimization
- Swizzled shared memory layout（避免 bank conflict）
- 向量化加载（128-bit）
- 高效 SMEM -> register 传输

### Stage 3: Tensor Core
- Warp-level GMMA 指令
- Tiled MMA（16×16×16 tiles）
- Softmax 与 MMA 流水线

### Stage 4: Final (FA4 Style)
- **Online softmax**（Flash Attention 核心算法）
- **Software pipelining**（TMA + MMA 重叠）
- **Persistent kernels**（causal 专用调度）
- Optimal register allocation

## 参考 FA4 实现

FA4 的核心代码在 `flash-attention/flash_attn/cute/interface.py`：

```python
# FA4 的 Python CuTe 实现方式
@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool

def _flash_attn_fwd(q, k, v, ...):
    # 使用 cute.compile JIT 编译
    _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
        fa_fwd,
        q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor,
        softmax_scale,
        ...
    )
```

## 在 B200 服务器上运行

```bash
cd /sgl-workspace/Roofline-Analysis
git pull

cd cute_attention/python_dsl
python benchmark.py
```

## 输出示例

```
================================================================
 FlashAttention Python CuTe DSL Benchmark
================================================================
 GPU: NVIDIA B200
 Peak: 2250 TFLOPs/s

 Seqlen=4096, Batch=8
 Stage    Time(ms)   TFLOPs    TC Util   Speedup
 -------------------------------------------------
 0        120.45     1.1       0.05%     1.0x
 1        32.10      4.2       0.19%     3.8x
 2        10.50      12.8      0.57%     11.5x
 3        1.82       73.8      3.28%     66.2x
 4        1.10       122.1     5.43%     109.5x
================================================================
```

## 学习要点

1. **CuTe DSL 基础**：理解 `cute.kernel`, `cute.shared_memory`, `cute.gemm` 等 API
2. **Tiling 策略**：如何选择合适的 block size
3. **Tensor Core 编程**：GMMA 指令的使用
4. **Pipeline 技术**：如何隐藏内存延迟
5. **Online Softmax**：Flash Attention 的数学原理

## 参考

- [FA4 Paper](https://arxiv.org/abs/2312.03519)
- [CUTLASS DSL Documentation](https://github.com/NVIDIA/cutlass/tree/main/python)
- [CuTe Tutorial](https://github.com/NVIDIA/cutlass/tree/main/examples/cute)
