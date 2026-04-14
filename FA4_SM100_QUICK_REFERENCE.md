# FA4 SM100 快速索引与开发清单

## 一键查找：按功能分类

### 内存与布局
| 功能 | 文件 | 代码行 | 关键类/函数 |
|-----|-----|-------|----------|
| SMEM 配置 | flash_fwd_sm100.py | 255-297 | `_setup_attributes()` |
| TMEM 偏移 | flash_fwd_sm100.py | 216-230 | `self.tmem_*_offset` |
| Shared Storage 定义 | flash_fwd_sm100.py | 768-812 | `@cute.struct SharedStorage` |
| K/V 不对称布局 | softmax.py | 262-297 | `uneven_kv_smem` 处理 |
| 张量转置规则 | flash_fwd_sm100.py | 340-367 | Q/K/V/O layout transformation |

### TMA 与数据加载
| 功能 | 文件 | 代码行 | 关键类/函数 |
|-----|-----|-------|----------|
| TMA Descriptor 生成 | flash_fwd_sm100.py | 688-728 | `make_tiled_tma_atom_*` |
| 异步复制 Fallback | flash_fwd_sm100.py | 703-707, 748-758 | `copy_utils.tiled_copy_2d()` |
| Load Warp 主逻辑 | flash_fwd_sm100.py | 1250-1410 | `def load(...)` |
| PagedKV 管理 | paged_kv.py | - | `PagedKVManager` 类 |
| Paged KV 特例 | flash_fwd_sm100.py | 2859-2900 | `load_KV()` 中的 paged 处理 |

### Pipeline 与同步
| 功能 | 文件 | 代码行 | 关键对象 |
|-----|-----|-------|---------|
| Pipeline 创建 | flash_fwd_sm100.py | 1182-1241 | 各 `.create()` 调用 |
| Named Barriers | flash_fwd_sm100.py | 1243-1248 | `NamedBarrier(barrier_id=...)` |
| TMEM 分配/释放 | flash_fwd_sm100.py | 1970-1973, 1979-1986 | `tmem.allocate()`, `tmem.free()` |
| 状态机进度 | pipeline.py | - | `.advance()`, `.producer_tail()` |
| 集群同步 | flash_fwd_sm100.py | 1250, 1317 | `pipeline_init_arrive/wait()` |

### MMA 与计算
| 功能 | 文件 | 代码行 | 关键函数 |
|-----|-----|-------|---------|
| MMA 对象创建 | flash_fwd_sm100.py | 423-436 | `make_trivial_tiled_mma()` |
| Q*K^T MMA 循环 | flash_fwd_sm100.py | 1469+ | `def mma(...)` 前半部分 |
| P*V MMA 循环 | flash_fwd_sm100.py | 1469+ | `def mma(...)` 后半部分 |
| 2CTA 指令 | blackwell_helpers.py | - | `tcgen05.CtaGroup.TWO` 模式 |
| MMA 描述符 | mma_sm100_desc.py | - | TCGEN05 ↔ MMA 映射 |
| 累加器管理 | flash_fwd_sm100.py | 1215-1230 | TMEM 指针重定向 |

### Softmax 与统计
| 功能 | 文件 | 代码行 | 关键类/函数 |
|-----|-----|-------|----------|
| Online Softmax | softmax.py | - | `Softmax.online_softmax()` |
| SM100 Softmax 特例 | softmax.py | - | `SoftmaxSm100` 类 |
| Softmax Loop | flash_fwd_sm100.py | 1758+ | `def softmax_loop(...)` |
| 分段 P 写入 | flash_fwd_sm100.py | 202-204 | `self.split_P_arrive` |
| Exp2 仿真 | flash_fwd_sm100.py | 77-88 | `_TUNING_CONFIG` 中的 `ex2_emu_*` |
| 统计 Barrier | flash_fwd_sm100.py | 1248 | `sm_stats_barrier` |

### Correction 与 Epilogue
| 功能 | 文件 | 代码行 | 关键函数 |
|-----|-----|-------|---------|
| Correction Loop | flash_fwd_sm100.py | 2064+ | `def correction_loop(...)` |
| Rescale 因子 | flash_fwd_sm100.py | 2064+ | correction 的核心计算 |
| Epilogue 存储 | flash_fwd_sm100.py | 2711+ | `def epilogue_s2g(...)` |
| 变长 Epilogue | flash_fwd_sm100.py | 2027, 298 | `use_correction_warps_for_epi` |
| LSE 输出 | flash_fwd_sm100.py | 2711+ | `mLSE` 的计算与存储 |

### 调度与分工
| 功能 | 文件 | 代码行 | 关键对象 |
|-----|-----|-------|---------|
| 调度器选择 | flash_fwd_sm100.py | 208-216 | `self.TileScheduler = ...` |
| Persistent Scheduler | tile_scheduler.py | - | `StaticPersistentTileScheduler` |
| LPT Scheduler（Causal） | tile_scheduler.py | - | `SingleTileLPTScheduler` |
| Varlen Scheduler | tile_scheduler.py | - | `SingleTileVarlenScheduler` |
| CLC 动态调度 | flash_fwd_sm100.py | 1297-1342 | `ClcState.create()` 和 scheduler warp |

---

## 快速参考：常见配置

### 最简单配置（baseline）
```python
fa = FlashAttentionForwardSm100(
    head_dim=128,
    is_causal=False,
    use_2cta_instrs=False,
    is_persistent=True,
)
# 预期：
# - q_stage=2, kv_stage=2, s_stage=2
# - TMA 启用（若支持）
# - 16 个 warp（标准配置）
# - Grid = (num_seqs / 2, num_heads, num_batches)
```

### 2CTA 模式（性能优化）
```python
fa = FlashAttentionForwardSm100(
    head_dim=128,
    use_2cta_instrs=True,
    ...
)
# 变化：
# - cluster_shape_mn = (2, 1)
# - mma_tiler_qk = (256, 128, 128)  # M 翻倍
# - 两个 CTA 共用单条 TCGEN05 指令
```

### Causal Attention
```python
fa = FlashAttentionForwardSm100(
    head_dim=128,
    is_causal=True,
    use_clc_scheduler=True,  # 推荐用 CLC
)
# 变化：
# - TileScheduler = SingleTileLPTScheduler
# - 动态调度（左→右）
# - CLC scheduler warp 活跃
```

### GQA/MQA
```python
fa = FlashAttentionForwardSm100(
    head_dim=128,
    qhead_per_kvhead=8,  # 8:1 GQA
    ...
)
# 变化：
# - head 维度的分组处理
# - pack_gqa 选项影响 head 打包
```

### 变长 Attention
```python
fa = FlashAttentionForwardSm100(
    head_dim=128,
    is_varlen_q=True,
    ...
)
# 变化：
# - TileScheduler = SingleTileVarlenScheduler
# - use_correction_warps_for_epi = True（Correction 兼 Epilogue）
# - cu_seqlens_q 输入变为强制
```

---

## 调试与性能分析清单

### 正确性检查

- [ ] **阶段 1-2**：内存布局
  - [ ] `self.head_dim_padded` 是 16 的倍数
  - [ ] `self.kv_stage` 的计算：SMEM 不超过 224KB
  - [ ] `self.tmem_total` 不超过 `get_max_tmem_alloc_cols`

- [ ] **阶段 3**：TMA 配置
  - [ ] `tma_atom_Q/K/V/O` 不为空（或有 fallback）
  - [ ] `tma_copy_bytes` 合理（不超过 TMA 带宽）
  - [ ] SMEM layout 与 TMA layout 对齐

- [ ] **阶段 4**：Pipeline 同步
  - [ ] Barrier count 正确（2 per stage）
  - [ ] Producer/Consumer 包含的线程数一致
  - [ ] `defer_sync=True` 避免过早竞争

- [ ] **阶段 5-6**：Warp 分工与调度
  - [ ] Warp ID 无重叠、无空洞
  - [ ] `threads_per_cta` 正确（应为 512）
  - [ ] Grid 尺寸合理（不过大或过小）

- [ ] **阶段 7**：MMA 配置
  - [ ] `tiled_mma_qk` 与 `tiled_mma_pv` 创建成功
  - [ ] Thread partitioning 与 SMEM 分割一致
  - [ ] 2CTA 模式下 MMA tiler M 维翻倍

- [ ] **阶段 8-11**：计算正确性
  - [ ] Softmax 结果范围 [0, 1)
  - [ ] O 的值与标准 attention 接近（相对误差 < 1%）
  - [ ] LSE 值为正（或特定情况下合理）
  - [ ] Global memory no out-of-bounds writes

### 性能分析

| 指标 | 期望范围 | 检查方法 |
|-----|---------|---------|
| **SMEM 占用** | < 224 KB | 运行前打印计算值 |
| **TMEM 占用** | < 512 cols | `self.tmem_total` |
| **寄存器占用** | < 255 per thread | nvcc `-Xptxas=-v` 输出 |
| **指令数** | < 2000 per warp | PTX 反汇编检查 |
| **L1 缓存缺失** | < 5% | nsys profiling |
| **SMEM 冲突** | < 1% | profiler 详细 |
| **Bank conflict** | 避免（除设计） | SMEM layout swizzle 检查 |

### 常见错误与检查

| 错误 | 症状 | 修复 |
|-----|------|------|
| SMEM 溢出 | Kernel 崩溃或输出错误 | 减少 `kv_stage` 或 `q_stage` |
| TMA 对齐错误 | CUDA error 或数据错乱 | 确保地址 128B 对齐 |
| Barrier 数量不匹配 | 死锁 | 检查 `MemRange[Int64, ...]` 的第二个参数 |
| Warp divergence | 性能下降 | 加 `make_warp_uniform()` |
| 双 CTA 同步错误 | 结果正确但稀奇古怪 | 检查 `cluster_shape_mnk` 和 `cta_layout_vmnk` |

---

## 实验工作流

### 验证 Baseline（一周）
1. **创建最小测试**：4 个 head, 128 head_dim, 512 seq_len
2. **运行标准 attention**：获得 ground truth
3. **运行 FA4-SM100**：比对输出
4. **精度分析**：计算相对误差分布

### 优化迭代
1. **阶段 N 的增量测试**：仅修改当前阶段
2. **回归检查**：验证之前的阶段不被破坏
3. **性能 profiling**：nsys, ncu, cutlass profiler
4. **调参**：根据 `_TUNING_CONFIG` 调整参数

### Benchmark
```bash
# 运行 flash-attention 的官方 benchmark
python cute_attention/python_dsl/benchmark_comprehensive.py

# 对比 SM90 vs SM100
python cute_attention/python_dsl/roofline_analysis.py
```

---

## 关键代码片段查阅

### 获取当前 warp 的 work_tile
```python
# 在 load 或 mma 中
work_tile = tile_scheduler.initial_work_tile_info()
while work_tile.is_valid_tile:
    m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
    # 处理...
    work_tile = tile_scheduler.advance_to_next_work()
```

### TMEM 指针管理
```python
# MMA warp 中
tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
tmem.wait_for_alloc()
tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)

# 使用 tmem_ptr 创建累加器
tStS = thr_mma_qk.make_fragment_C(...)
tOtO = thr_mma_pv.make_fragment_C(...)
tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)

# 完成后释放
tmem.relinquish_alloc_permit()
tmem_alloc_barrier.arrive_and_wait()
tmem.free(tmem_ptr)
```

### Pipeline 等待与信号
```python
# Producer（如 MMA warp）信号
pipeline.producer_acquire()  # 等待 empty 信号
# ... do work ...
pipeline.producer_release()  # 释放 full 信号

# Consumer（如 Softmax warp）等待
pipeline.consumer_wait()  # 等待 full 信号
# ... consume ...
pipeline.consumer_release()  # 释放 empty 信号
```

### Softmax 计算
```python
softmax = SoftmaxSm100.create(scale_log2, q_stage * m_block_size)
# 循环不同的 K block
for n_block in ...:
    # Load S from TMEM at tmem_s_offset
    acc_S = ...  # 从 TMEM 读取
    row_scale = softmax.online_softmax(acc_S, is_first=(n_block==0), check_inf=True)
    # 使用 row_scale 计算 P...
```

---

## 文件导航速查表

| 问题 | 查看文件 | 主要类/函数 |
|-----|---------|----------|
| SM100 vs SM90 的区别 | flash_fwd_sm100.py vs flash_fwd_sm90.py | 整体结构对比 |
| Barrier/Pipeline 用法 | pipeline.py, pipeline_sm100.py | Pipeline 类定义 |
| Softmax 实现细节 | softmax.py | `Softmax`, `SoftmaxSm100` |
| Memory layout 工具 | blackwell_helpers.py, hopper_helpers.py | `make_smem_layout_*` |
| TMA/Copy 操作 | copy_utils.py（quack） | `tiled_copy_2d`, TMA 函数 |
| 块稀疏处理 | block_sparse_utils.py | `produce_block_sparse_loads_sm100` |
| Paged KV | paged_kv.py | `PagedKVManager` |
| 调度器逻辑 | tile_scheduler.py | 各类 Scheduler |
| GQA/Pack 处理 | pack_gqa.py | `pack_gqa_layout`, `PackGQA` |
| 变长处理 | seqlen_info.py | `SeqlenInfoQK` |
| 多 head/batch 处理 | block_info.py | `BlockInfo` |
| Attention mask | mask.py | `AttentionMask` |

---

## 单元测试建议

### 单阶段测试模板

```python
# test_stage_N.py
import unittest
import torch
import cutlass
from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

class TestStageN(unittest.TestCase):
    def setUp(self):
        self.B, self.H, self.D, self.S = 1, 1, 128, 512
        self.fa = FlashAttentionForwardSm100(head_dim=self.D)
    
    def test_config_validity(self):
        """Check all config parameters are within valid ranges"""
        self.assertGreaterEqual(self.fa.head_dim_padded, self.fa.head_dim)
        self.assertEqual(self.fa.head_dim_padded % 16, 0)
        self.assertLessEqual(self.fa.tmem_total, 512)
    
    def test_smem_layout(self):
        """Verify SMEM sizes don't overflow"""
        # 计算 SMEM 占用
        smem_total = ...  # 根据 _setup_attributes 计算
        self.assertLessEqual(smem_total, 224 * 1024)
    
    def test_kernel_launch(self):
        """Run kernel and check for CUDA errors"""
        Q = torch.randn(self.B, self.S, self.H, self.D, dtype=torch.float16, device='cuda')
        K = torch.randn(self.B, self.S, self.H, self.D, dtype=torch.float16, device='cuda')
        V = torch.randn(self.B, self.S, self.H, self.D, dtype=torch.float16, device='cuda')
        O = torch.empty_like(Q)
        
        # Launch
        self.fa(Q, K, V, O, None, softmax_scale=1/math.sqrt(self.D))
        
        # Basic checks
        self.assertFalse(torch.isnan(O).any())
        self.assertFalse(torch.isinf(O).any())

if __name__ == '__main__':
    unittest.main()
```

### 回归测试

```python
# test_integration.py，在完成所有 12 阶段后
def test_accuracy_vs_reference():
    """Compare FA4-SM100 output against PyTorch attention"""
    Q, K, V = ..., ..., ...  # 测试输入
    
    # Reference
    O_ref = F.scaled_dot_product_attention(Q, K, V, scale=1/math.sqrt(D))
    
    # FA4
    O_fa4 = fa(Q, K, V, ...)
    
    # Check
    rel_error = torch.abs(O_fa4 - O_ref).max() / (torch.abs(O_ref).max() + 1e-6)
    self.assertLess(rel_error, 0.01, f"Relative error too large: {rel_error}")
```

---

## 版本控制建议

### 分支策略

```
main
├── dev/stage-1-2 (基础框架)
├── dev/stage-3-4 (TMA + Pipeline)
├── dev/stage-5-7 (Warp + MMA)
├── dev/stage-8-11 (Softmax + Epilogue)
└── dev/stage-12 (Load optimization)
```

### Commit 信息规范

```
[Stage N] Brief description

Detailed explanation:
- 实现的功能
- 依赖的前置阶段
- 已通过的测试

Related to: <issue/pr>
```

---

## 支持与资源

### 官方文档
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [CuTe](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL)
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention)

### 相关论文
- Flash-Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- Blackwell Architecture (NVIDIA whitepaper)
- Online Softmax in Attention

### 常用命令

```bash
# 编译检查
cd flash-attention && python setup.py build_ext --inplace

# 运行测试
python -m pytest tests/ -v

# 性能分析
nsys profile --trace cuda,osrt -o profile_sm100.nsys-rep python run_fa4.py
ncu --set=full -o profile_sm100.ncu-rep python run_fa4.py

# 代码生成查看
cuobjdump -sass kernel.cubin | head -100
```

---

## FAQ

**Q: 为什么 SM100 使用 TMEM 而不是 SMEM？**

A: TMEM 有以下优势：
- 更大容量（512 列 vs 224KB）
- 低延迟访问
- 无 bank conflict（设计上）
- 避免 SMEM 的 roundtrip（S 和 O 不需要回 SMEM）

**Q: 2CTA 指令是什么？为什么需要它？**

A: TCGEN05 允许两个 CTA 用单条指令完成 MMA，性能约提升 1.8-2x。代价是需要同步两个 CTA 并调整内存分配。

**Q: 为什么有两组 Softmax Warps？**

A: 在 `q_stage=2` 时，可以用不同的 warp 组处理不同 Q 的阶段，实现更好的 pipeline 重叠。

**Q: 如何调试 Barrier 死锁？**

A: 
1. 确认 producer/consumer 的线程数一致
2. 加 debug printf 检查何处卡住
3. 使用 Barrier ID 验证是否用错了
4. 在 `pipeline_init_wait` 前后插 `__syncthreads()`

**Q: Split-KV 是什么？**

A: 当 K/V 超大时，分成多个块分别处理，最后通过 max/sum 的结合规则得到最终结果。

