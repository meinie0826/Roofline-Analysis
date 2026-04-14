# FA4 (FlashAttention-4) SM100 前向实现分阶段计划

> 基于真实代码分析：`flash-attention/flash_attn/cute/flash_fwd_sm100.py`

## 概述

本计划根据 SM100 实现的真实代码依赖关系制定，而非按 README 中的顺序。SM100 前向计算采用**高度分工的 warp 特化架构**，将计算分解为 7 个并行的 warp 组，通过精心设计的 pipeline 同步机制协调。

### 关键创新
1. **TMEM（Tensor Memory）**：存储中间结果 S、P、O（避免 SMEM roundtrip）
2. **2CTA Instructions**：两个 CTA 可用单条 TCGEN05 指令完成MMA
3. **Warp 特化**：16 个 warp 的精细分工
4. **Pipeline 重叠**：异步加载、计算、softmax、存储完全重叠

---

## 阶段 1: 基础框架与输入验证

**依赖**：无

**实现内容**：
- 类 `FlashAttentionForwardSm100` 初始化框架
- 张量输入转置与布局（Q、K、V、O 的维度重排）
- 配置参数验证：head_dim 对齐、dtype 一致性检查
- 计算内存需求（SMEM、TMEM 布局）

**关键代码位置**：
- `__init__` 方法（行 106-253）：参数存储与架构判断
- `_setup_attributes` 方法（行 255-297）：SMEM 分配计算，pipeline stage 配置
- `__call__` 方法（行 326-344）：张量转置

**参考对应**：
- SM90：`flash_fwd_sm90.py` 的 `__init__` 和 `__call__`，但 SM100 有额外的 TMEM 配置

**输出**：
- 配置好的内存布局
- SharedStorage 结构定义（SMEM 分配）
- TMA 描述符（如果启用）

---

## 阶段 2: 内存布局与 Shared Memory 配置

**依赖**：阶段 1

**实现内容**：
- **SMEM 布局定义**：Q、K、V、O、Scale 的 shared memory 映射
- **TMEM 偏移计算**：S（attention scores）、P（softmax 结果）、O 的 TMEM 地址
- **Shared Storage 结构**：barrier、scale buffer 定义
- **Warp 分工映射**：16 个 warp 的标号分配

**关键代码位置**：
- `_setup_attributes` (行 255-297)：SMEM stage 计算
  - `kv_stage`：K、V 有多少 stage（通常 2-3）
  - `s_stage`：S 的 stage（通常 2）
  - `uneven_kv_smem`：不对称 KV 分配（hdim 192, dv 128 特殊情况）
- `__call__` 中的 `SharedStorage` struct（行 769-812）：
  ```python
  @cute.struct
  class SharedStorage:
      mbar_load_Q: MemRange[Int64, q_stage * 2]
      mbar_load_KV: MemRange[Int64, kv_stage * 2]
      ...
      sQ: Align[MemRange[q_dtype, sQ_size], 1024]
      sK: Align[MemRange[k_dtype, ...], 1024]
  ```
- `kernel` 开头（行 1166-1176）：Warp 分工标号

**参考对应**：
- SM90 在 hopper/flash_fwd_kernel_sm90.h 中的 SharedStorage
- SM100 多了 TMEM 相关字段

**输出**：
- SMEM 的具体地址分配
- Warp ↦ 任务 的对应关系
- Pipeline 同步用的 mbarrier 集合

---

## 阶段 3: TMA (Tensor Memory Access) 配置

**依赖**：阶段 2

**实现内容**：
- **TMA 描述符生成**：Q、K、V、O 的 TMA ATOM
- **TMA 与 SMEM 的细粒度映射**（CuTe）
- **备选方案**：不支持 TMA 时的 async copy fallback
- **Copy 操作**：gmem_tiled_copy_Q/O（异步复制）

**关键代码位置**：
- `__call__` 中 TMA 部分（行 688-742）：
  ```python
  if self.use_tma_Q:
      tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(...)
  else:
      gmem_tiled_copy_Q = copy_utils.tiled_copy_2d(...)
  ```
- TMA load op 定义（行 681-682）：
  ```python
  tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
  tma_store_op = cpasync.CopyBulkTensorTileS2GOp()
  ```

**参考对应**：
- SM90 在 `flash_fwd_sm90.py` 中类似的 TMA 配置
- CuTe 库的 `make_tiled_tma_atom_*` 函数

**输出**：
- `tma_atom_Q/K/V/O`：GPU kernel 中使用的 TMA 描述符
- `gmem_tiled_copy_Q/O`：fallback 用的 copy atom

---

## 阶段 4: Pipeline 初始化与同步机制

**依赖**：阶段 2、3

**实现内容**：
- **Pipeline 对象创建**：
  - `pipeline_q`：Q 加载管道（TMA UMMA 或 Async UMMA）
  - `pipeline_kv`：K、V 加载管道
  - `pipeline_s_p_o`：S 计算 + P 写入 + O rescale 同步
  - `pipeline_p_lastsplit`：最后一个 K block 的 P 写入
  - `pipeline_o_acc`：O 累加完成信号
  - `pipeline_sm_stats`：Softmax 统计信息同步
- **Named Barrier 初始化**：
  - Tensor Memory 分配/释放 barrier
  - Softmax 统计 barrier

**关键代码位置**：
- `kernel` 中的 pipeline 创建（行 1182-1241）：
  ```python
  pipeline_q = pipeline_custom.PipelineTmaUmma.create(
      barrier_storage=storage.mbar_load_Q.data_ptr(),
      num_stages=self.q_stage,
      producer_group=tma_warp,
      consumer_group=mma_warp,
      ...
  )
  ```
- Named Barrier 定义（行 1243-1248）：
  ```python
  tmem_alloc_barrier = pipeline.NamedBarrier(
      barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
      num_threads=...
  )
  ```

**参考对应**：
- SM90 的 pipeline 设计（但 SM100 增加了 TMEM-related 的同步）
- `flash_attn/cute/pipeline.py` 中的 Pipeline 类

**输出**：
- 6 个 Pipeline 对象及其同步状态机
- 3 个 Named Barrier 对象

---

## 阶段 5: Warp 角色初始化与控制流

**依赖**：阶段 4

**实现内容**：
- **Warp 分配逻辑**：根据 `warp_idx` 分派任务
- **Empty/CLC Scheduler Warp**：用于动态调度或空闲
- **Load Warp**：处理 TMA/异步加载逻辑
- **MMA Warp**：管理 TMEM 分配与释放
- **Softmax/Correction/Epilogue Warps**：分工处理后处理

**关键代码位置**：
- Warp 分工（行 1176-1180）：
  ```python
  self.softmax0_warp_ids = (0, 1, 2, 3)
  self.softmax1_warp_ids = (4, 5, 6, 7)
  self.correction_warp_ids = (8, 9, 10, 11)
  self.mma_warp_id = 12
  self.epilogue_warp_ids = (13,)
  self.load_warp_ids = (14,)
  self.empty_warp_ids = (15,)
  ```
- Warp 条件分派（行 1298 起）：
  ```python
  if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
      self.load(...)
  if warp_idx == self.mma_warp_id:
      self.mma(...)
  ```

**参考对应**：
- SM90 的 warp 分工相似但数量少
- 本质是 cooperative group 的编程模式

**输出**：
- 每个 warp 的任务清晰
- 控制流的分支（if-conditions）

---

## 阶段 6: Tile 调度器与工作分配

**依赖**：阶段 2

**实现内容**：
- **Tile Scheduler 选择**：根据模式（causal、local、varlen、persistent）选择调度器
- **TileSchedulerArguments 构建**：问题规模、head 数、batch 数、split 数
- **Grid 尺寸计算**：根据 tile 大小和问题规模计算
- **可选 CLC 调度**：动态持久化调度器（用于 Causal）

**关键代码位置**：
- 调度器选择（行 208-216）：
  ```python
  if is_varlen_q:
      self.TileScheduler = SingleTileVarlenScheduler
  elif self.is_causal or self.is_local or self.use_clc_scheduler:
      self.TileScheduler = SingleTileLPTScheduler
  elif self.is_persistent:
      self.TileScheduler = StaticPersistentTileScheduler
  ```
- TileSchedulerArguments（行 734-759）：
  ```python
  tile_sched_args = TileSchedulerArguments(
      num_tiles_m=cute.ceil_div(...),
      num_heads=...,
      num_batches=...,
      ...
  )
  ```

**参考对应**：
- `flash_attn/cute/tile_scheduler.py` 中多个调度器实现
- 比 SM90 增加了 CLC 调度器支持

**输出**：
- 调度器对象
- Grid 尺寸（用于 kernel launch）

---

## 阶段 7: MMA 操作配置（Q*K^T 和 P*V）

**依赖**：阶段 2、3

**实现内容**：
- **TCGEN05 MMA 配置**：
  - Q*K^T MMA：operand major modes（K×K）
  - P*V MMA：（K×MN），P 来自 TMEM
- **与 SMEM/TMEM 的布局绑定**
- **CTA Group（2CTA）支持**：
  - 2CTA 时 MMA tiler 在 M 维翻倍
  - 单 CTA 时 cluster shape 是 (1, 1)

**关键代码位置**：
- CTA group 决策（行 414-415）：
  ```python
  cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
  ```
- MMA 对象创建（行 423-436）：
  ```python
  tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
      self.q_dtype, q_major_mode, k_major_mode,
      self.qk_acc_dtype, cta_group, self.mma_tiler_qk[:2]
  )
  tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
      self.v_dtype, p_major_mode, v_major_mode,
      self.pv_acc_dtype, cta_group, self.mma_tiler_pv[:2],
      p_source  # TMEM
  )
  ```

**参考对应**：
- `flash_attn/cute/blackwell_helpers.py`：SM100 特定的 MMA 配置
- `flash_attn/cute/mma_sm100_desc.py`：MMA 描述符与 TCGEN05 的映射

**输出**：
- `tiled_mma_qk`：Q*K^T MMA 对象
- `tiled_mma_pv`：P*V MMA 对象
- 线程分区信息

---

## 阶段 8: Softmax 计算（在线 softmax 与两两交叉）

**依赖**：阶段 5、7（warp 分工 + MMA）

**实现内容**：
- **两个 Softmax Warp 组**：
  - 第一组（0-3）：处理 m_block 的前半部分 K blocks
  - 第二组（4-7）：处理后半部分（S_stage=2 时）或空闲
- **Online Softmax**：
  - 对每一行 S（attention score）计算 max、exp、sum
  - 依次更新 row_max、row_sum
  - 应用分层求和以减少 warp divergence
- **S_stage 管理**：不同阶段的 softmax 结果写入 TMEM 的不同位置
- **分段 P 写入**（split_P_arrive）：
  - 计算到 P 的 3/4 位置时信号 MMA warp 开始 P*V
  - 完成剩余 P 再信号 MMA warp

**关键代码位置**：
- Softmax loop（行 1758 起）：
  ```python
  def softmax_loop(
      self, stage: Int32, tStS: cute.Tensor, ...
  ):
      # Online softmax implementation
  ```
- 进入条件（行 2009-2019）：
  ```python
  if (const_expr(self.q_stage == 2) and warp_idx <= self.softmax1_warp_ids[-1]) or ...
  ```
- Softmax 类（`softmax.py` 中 SoftmaxSm100）：
  ```python
  class Softmax:
      def online_softmax(self, acc_S, is_first=False, check_inf=True)
  ```

**参考对应**：
- SM90（hopper/softmax.h）的 online softmax 算法
- SoftmaxSm100 vs Softmax 的差异（exp2 emulation 配置）

**输出**：
- TMEM 中的 P（softmax 后得分）
- row_max、row_sum statistics（用于 rescale）

---

## 阶段 9: MMA 主循环（Q*K^T + P*V）

**依赖**：阶段 7、8

**实现内容**：
- **MMA Warp 职责**：
  1. 分配 TMEM 用于存储 S、P、O
  2. 循环处理每个 K block：
     - 等待 Q ready（pipeline_q）
     - 等待 K ready（pipeline_kv）
     - 执行 Q*K^T MMA → S（存入 TMEM）
     - 信号 S ready（pipeline_s_p_o）
  3. 等待 P ready（来自 softmax）
  4. 执行 P*V MMA → O（写入 TMEM）
  5. 信号 O ready（pipeline_o_acc）
- **TMEM Offset 管理**：
  - S：offset 0 和 128（两个 s_stage）
  - P：offset 64（S 的一半）和 192
  - O：offset 256 和 384
- **Blocking 等待与 fence**

**关键代码位置**：
- `mma` 方法（行 1469 起）：
  ```python
  def mma(self, tiled_mma_qk, tiled_mma_pv, sQ, sK, sV, 
          tStS, tOtO, tOrP, pipeline_q, pipeline_kv, ...):
      # Q*K^T MMA loop
      # P*V MMA loop
  ```
- TMEM 分配（行 1970-1973）：
  ```python
  tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
  tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
  ```
- MMA 操作（`mma_sm100_desc.py`）

**参考对应**：
- SM90 的 MMA loop（但 SM100 用 TMEM 而非 SMEM 存储 S）
- TCGEN05 指令的参数与同步

**输出**：
- TMEM 中的 O（未 rescale 的输出）
- O 的数据布局与后续 rescale 的对应

---

## 阶段 10: Correction Loop（O 的 Rescale）

**依赖**：阶段 8、9（softmax stats + O accumulation）

**实现内容**：
- **Correction Warp 组（8-11）**：
  1. 等待 softmax stats ready（来自 softmax warps）
  2. 等待 O 累加完成（来自 MMA warp）
  3. 对每一行计算 rescale 因子 $\frac{\text{row\_sum\_prev}}{\text{row\_sum\_cur}}$
  4. 将 O 乘以该因子并写入结果
- **变长 Attention 特殊情况**：
  - 若 `use_correction_warps_for_epi`，correction warp 兼任 epilogue
  - 否则 correction warp 只负责 rescale，epilogue warp 负责 G2S 写入

**关键代码位置**：
- 进入条件（行 2064）：
  ```python
  if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
  ```
- `correction_loop` 方法（行 XXX）：
  ```python
  def correction_loop(self, thr_mma_qk, thr_mma_pv, tStS, tOtO, 
                      sScale, mO, mLSE, sO, ...):
  ```

**参考对应**：
- SM90 中的 rescale 逻辑（但 SM100 可能做法复新）
- 变长处理的特殊路径

**输出**：
- O 在 SMEM 中的 rescaled 版本（若存储到 SMEM）
- O 的 LSE（log-sum-exp）统计（用于训练反向传播）

---

## 阶段 11: Epilogue（O 写回 Global Memory）

**依赖**：阶段 10（O rescale）

**实现内容**：
- **Epilogue Warp（13）**：
  1. 等待 O 在 SMEM 中 ready（或来自 correction）
  2. 将 O 从 SMEM 写出到 global memory
  3. 可选：TMA 写出（如果 `use_tma_O`）
  4. 否则使用 universal copy 操作
- **LSE 写出**（若 `mLSE is not None`）：
  - 每个 tile 计算 3 个统计：row_max、row_sum、split 信息
  - 写入预分配的 LSE 张量

**关键代码位置**：
- 进入条件（行 2027）：
  ```python
  if const_expr(not self.use_correction_warps_for_epi):
      if warp_idx >= self.epilogue_warp_ids[0] and ...
  ```
- `epilogue_s2g` 方法（行 2711 起）：
  ```python
  def epilogue_s2g(self, mO, sO, gmem_tiled_copy_O, 
                   tma_atom_O, pipeline_o_epi, ...):
  ```
- TMA 写出与 fallback copy（行 706-725）

**参考对应**：
- SM90 的 epilogue 实现
- CuTe 的 TMA 写出 vs universal copy

**输出**：
- GPU 全局内存中最终的 O 张量
- 可选的 LSE 统计

---

## 阶段 12: Load Warp 细节（Q、K、V 加载）

**依赖**：阶段 4、5（pipeline + 工作分配）

**实现内容**：
- **加载调度**：
  1. 遍历 tile_scheduler 分配的工作块
  2. 对每个 tile（m_block, head_idx, batch_idx）：
     - 计算 Q、K、V 的地址与形状
     - 发起 TMA 加载或异步复制
- **TMA 路径**：
  - K、V 首次加载触发 pipeline_kv 同步
- **异步复制路径**：
  - 使用线程块的协作分配线程
  - 处理 paged KV（page_table）情况
- **块稀疏支持**（block sparsity）：
  - 判断哪些 K blocks 需要加载
  - 调用 `produce_block_sparse_loads_sm100`

**关键代码位置**：
- `load` 方法（行 1250 起）：
  ```python
  def load(self, thr_mma_qk, thr_mma_pv, mQ, mK, mV, sQ, sK, sV, ...):
      work_tile = tile_scheduler.initial_work_tile_info()
      while work_tile.is_valid_tile:
          # Process m_block, head_idx, batch_idx, split_idx
          load_Q(...)
          load_K(...)
          load_V(...)
          work_tile = tile_scheduler.advance_to_next_work()
  ```
- `load_Q` 与 `load_Q_non_tma`（行 2814 起）
- `load_KV`（行 2859 起）

**参考对应**：
- SM90 的 load 实现（hopper/flash_fwd.py）
- Paged KV 管理（CUDA 推理优化）

**输出**：
- 所有需要的 Q、K、V 数据加载到 SMEM
- Pipeline 状态推进

---

## 实现优先级与关键依赖图

```
阶段 1 (基础框架)
  ↓
阶段 2 (SMEM 配置) ─ → 阶段 3 (TMA) ─ ┐
  ↓                                    ├→ 阶段 4 (Pipeline)
阶段 5 (Warp 分工)                      ↓
  ↓                                阶段 7 (MMA 配置)
阶段 6 (Tile 调度)                      ↓
                                  阶段 9 (MMA 主循环)
                                  ↓
                            阶段 8 (Softmax) ← 竞争 TMEM
                            ↓
                            阶段 10 (Correction)
                            ↓
                            阶段 11 (Epilogue)

补充（可与主流程并行）：
  阶段 12 (Load Warp，依赖阶段 4、5)
```

### 关键并行机会
- **阶段 3 与 阶段 2 的一部分可并行**：TMA 配置不依赖 warp 分工
- **阶段 6 可与 阶段 4、5 并行**：tile scheduler 独立于 pipeline 细节
- **阶段 8、9、10 形成紧密的流水**：无法真正并行，必须按序

---

## 最自然的分阶段执行顺序

基于真实代码依赖关系（**非顺序组织**），推荐顺序：

1. **第一周**：阶段 1-2（框架与内存布局）
2. **第二周**：阶段 3、4、6（TMA、Pipeline、调度器）
3. **第三周**：阶段 5、12（Warp 分工、Load 逻辑）
4. **第四周**：阶段 7（MMA 配置）
5. **第五周**：阶段 9（MMA 主循环）
6. **第六周**：阶段 8（Softmax）
7. **第七周**：阶段 10-11（Correction + Epilogue）

---

## 每个阶段的测试要点

| 阶段 | 测试方法 |
|------|---------|
| 1-2 | 打印内存映射、检查对齐 |
| 3-4 | TMA 描述符有效性、barrier 初始化 |
| 5 | Warp 标号无冲突 |
| 6 | Grid 尺寸是否合理 |
| 7 | MMA thread partitioning 正确性 |
| 8 | Softmax 结果范围 [0, 1] |
| 9 | O 的数值正确性（vs 标准 attention） |
| 10 | Rescaling 因子的计算 |
| 11 | Global memory 写入正确性、无越界 |
| 12 | 加载数据与预期对齐 |

---

## 与 SM90 实现的关键对比表

| 特性 | SM90 (Hopper) | SM100 (Blackwell) |
|-----|--------------|------------------|
| **中间结果存储** | SMEM（S、P） | TMEM（S、P、O） |
| **Warp 数** | 8 | 16 |
| **MMA 指令集** | Warpgroup MMA | TCGEN05 MMA |
| **CTA 协作** | 不支持 | 2CTA instructions |
| **Pipeline 类型** | PipelineAsync | PipelineTmaUmma + PipelineAsyncUmma |
| **Softmax 实现** | Softmax 类 | SoftmaxSm100 类 |
| **Cluster 形状** | (1, 1) | (1, 1) 或 (2, 1)（2CTA） |
| **Exp2 优化** | 原生 FP32 exp2 | 硬件 exp2 + 仿真混合 |

---

## 代码组织建议

将 FA4 实现拆分为对应 12 个阶段的模块：

```
fa4_sm100/
├── 01_framework.py        # 阶段 1
├── 02_memory_layout.py     # 阶段 2
├── 03_tma_config.py        # 阶段 3
├── 04_pipeline_sync.py     # 阶段 4
├── 05_warp_roles.py        # 阶段 5
├── 06_tile_scheduler.py    # 阶段 6
├── 07_mma_setup.py         # 阶段 7
├── 08_softmax.py           # 阶段 8
├── 09_mma_loop.py          # 阶段 9
├── 10_correction.py        # 阶段 10
├── 11_epilogue.py          # 阶段 11
├── 12_load_warp.py         # 阶段 12
├── tests/
│   ├── test_01_02.py
│   ├── test_03_04.py
│   ├── test_05_06.py
│   ├── test_07_09.py
│   ├── test_08_10_11.py
│   └── test_12.py
└── README.md               # 本计划文档
```

---

## 代码片段示例：关键 API

### 初始化（阶段 1）
```python
fa = FlashAttentionForwardSm100(
    head_dim=128,
    head_dim_v=128,
    qhead_per_kvhead=1,
    is_causal=False,
    use_2cta_instrs=False,
)
```

### 执行（阶段 1-12 集成）
```python
fa(mQ, mK, mV, mO, mLSE,
   softmax_scale=1/math.sqrt(head_dim),
   stream=stream)
```

### 关键数据结构
- `SharedStorage`：SMEM 布局（阶段 2）
- `BlockInfo`：Tile 信息（阶段 6）
- `SoftmaxSm100`：Online softmax（阶段 8）
- `NamedBarrier`：线程同步（阶段 4）

