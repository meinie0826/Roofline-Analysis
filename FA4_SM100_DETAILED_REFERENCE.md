# FA4 SM100 实现阶段详细参考表

## 阶段 1：基础框架与输入验证

### 1.1 参数初始化

| 参数 | 类型 | 默认值 | 含义 |
|-----|------|--------|------|
| `head_dim` | int | 128 | Query/Key 的特征维度 |
| `head_dim_v` | int | head_dim | Value 的特征维度（可不同） |
| `qhead_per_kvhead` | int | 1 | GQA/MQA 的倍数 |
| `is_causal` | bool | False | 是否 causal mask |
| `is_local` | bool | False | 是否 local（sliding window）attention |
| `is_split_kv` | bool | False | 是否 split-KV 分支 |
| `pack_gqa` | bool | False | 是否 pack GQA 的 head 维度 |
| `m_block_size` | int | 128 | Query tile 的行数 |
| `n_block_size` | int | 128 | Key tile 的行数 |
| `q_stage` | int | 2 | Q 在 SMEM 的 pipeline stage数 |
| `use_2cta_instrs` | bool | False | 是否使用 2CTA MMA 指令 |
| `use_clc_scheduler` | bool | False | 是否动态 CLC 调度（causal 时） |

### 1.2 计算的关键配置

```python
# 从输入参数衍生
self.head_dim_padded = ceil(head_dim / 16) * 16  # 必须 16 对齐
self.head_dim_v_padded = ceil(head_dim_v / 16) * 16
self.cta_group_size = 2 if use_2cta_instrs else 1

# MMA tiler
self.mma_tiler_qk = (cta_group_size * m_block_size, n_block_size, head_dim_padded)
self.mma_tiler_pv = (cta_group_size * m_block_size, head_dim_v_padded, n_block_size)

# Cluster shape（影响网格分组）
self.cluster_shape_mn = (2, 1) if use_2cta_instrs else (1, 1)
```

### 1.3 张量转置规则

```python
# 输入的默认形状（假设 varlen=False）
# mQ: (batch, seq_len_q, heads, head_dim)
# mK: (batch, seq_len_k, heads_kv, head_dim)
# mV: (batch, seq_len_k, heads_kv, head_dim_v)
# mO: (batch, seq_len_q, heads, head_dim_v)

# 转置后（GPU kernel 中期望的形状）
# mQ: (seq_len_q, head_dim, heads, batch)
# mK: (seq_len_k, head_dim, heads_kv, batch)
# mV: (head_dim_v, seq_len_k, heads_kv, batch)  ← 注意 mV 特殊！
# mO: (seq_len_q, head_dim_v, heads, batch)

Q_layout_transpose = [1, 3, 2, 0]  # (b,s,h,d) → (s,d,h,b)
KV_layout_transpose = [1, 3, 2, 0]
V_layout_transpose = [1, 0, 2, 3]  # (b,s,h,dv) → (dv,s,h,b)  ← 特殊
```

### 1.4 Warp 分工（16 个 warp）

```
Warp 0-3:   Softmax 第一组（处理前半部分 K blocks）
Warp 4-7:   Softmax 第二组（处理后半部分或 stage=1）
Warp 8-11:  Correction 校正 O 的 rescale
Warp 12:    MMA（Q*K^T 和 P*V）
Warp 13:    Epilogue（输出写回）
Warp 14:    Load（加载 Q、K、V）
Warp 15:    Empty（未来扩展或 CLC scheduler）
```

### 1.5 代码位置与关键变量

| 任务 | 代码行 | 变量/函数 | 输出 |
|-----|-------|---------|------|
| 参数存储 | 116-200 | `__init__` | `self.m_block_size`, `self.head_dim_padded` 等 |
| SMEM 计算 | 255-297 | `_setup_attributes` | `self.kv_stage`, `self.s_stage` |
| 张量转置 | 340-367 | `__call__` | `mQ, mK, mV, mO` 的转后形状 |
| dtype 检查 | 375-378 | `__call__` | 确保 Q, K, V dtype 一致 |

---

## 阶段 2：内存布局与 Shared Memory 配置

### 2.1 SMEM 尺寸计算

```python
# Q 和 O 可能重叠存储（overlap_sO_sQ）
smem_size_q = q_stage * m_block_size * head_dim_padded * (dtype_width / 8)
smem_size_o = q_stage * m_block_size * head_dim_v_padded * (dtype_width / 8)

# K 和 V 共享 SMEM（但用不同的 stride）
smem_size_kv_per_stage = max(
    n_block_size * head_dim_padded,
    n_block_size * head_dim_v_padded
) // cta_group_size  # 分摊到两个 CTA（若 2CTA 模式）

# 自动计算能容纳多少 stage
total_smem = 224 * 1024  # SM100 per CTA 的 SMEM
kv_stage = (total_smem - smem_size_q_o) / smem_size_kv_per_stage

# 特殊情况：hdim=192, dv=128 时可用 uneven_kv_smem
if head_dim_padded == 192 and head_dim_v_padded == 128 and kv_stage == 2:
    kv_stage = 3  # 利用空隙，3 stages + 特殊 stride 跳过
    uneven_kv_smem = True
    uneven_kv_smem_offset = n_block_size * (head_dim_padded - head_dim_v_padded) // 2
```

### 2.2 SMEM 结构定义

```python
@cute.struct
class SharedStorage:
    # Barriers（每个 stage 有 2 个 mbarrier：full + empty）
    mbar_load_Q: MemRange[Int64, q_stage * 2]
    mbar_load_KV: MemRange[Int64, kv_stage * 2]
    mbar_S_full_P_full_O_rescaled: MemRange[Int64, q_stage * 2]
    mbar_P_full_lastsplit: MemRange[Int64, q_stage * 2]
    mbar_O_full: MemRange[Int64, q_stage * 2]
    mbar_softmax_stats: MemRange[Int64, q_stage * 2]
    mbar_O_epi: MemRange[Int64, q_stage * 2]
    mbar_s0_s1_sequence: MemRange[Int64, 2 * 2]
    
    # TMEM 管理
    tmem_dealloc_mbar_ptr: Int64
    tmem_holding_buf: Int32
    
    # 数据缓冲
    sScale: MemRange[Float32, q_stage * m_block_size * 2]  # row_max + row_sum
    
    # CLC 相关（若启用）
    clc_mbar_ptr: MemRange[Int64, clc_mbar_size]
    clc_response: MemRange[Int32, clc_response_size]
    
    # 大对齐缓冲（1024 字节对齐）
    sO: Align[MemRange[o_dtype, ...], 1024]
    sQ: Align[MemRange[q_dtype, ...], 1024]
    sK: Align[MemRange[k_dtype, ...], 1024]
```

### 2.3 TMEM 偏移（Tensor Memory）

```python
# TMEM 用于存放 S（attention scores）、P（softmax）、O（输出）
tmem_s_offset = [0, n_block_size]  # S 的两个 stage：0, 128
tmem_o_offset = [
    n_block_size + n_block_size + 0 * head_dim_v_padded,  # 256
    n_block_size + n_block_size + 1 * head_dim_v_padded,  # 384（q_stage=2）
]
tmem_total = tmem_o_offset[-1] + head_dim_v_padded  # 一般 512

# P 在 TMEM 中的位置（从 S 的一半开始）
tmem_s_to_p_offset = n_block_size // 2  # 64
tmem_p_offset = [0 + 64, 128 + 64]  # 64, 192
```

### 2.4 Scale 缓冲（Softmax 统计）

```python
# sScale 存储每行的 row_max 和 row_sum
# 布局：[row_max_stage0..., row_sum_stage0..., row_max_stage1..., row_sum_stage1...]
# 大小：q_stage * m_block_size * 2 个 Float32

# Softmax 的两个 warp 组共享这个缓冲
# Softmax 0 处理 stage 0（若 q_stage==2）
# Softmax 1 处理 stage 1
```

### 2.5 代码位置

| 任务 | 代码行 | 变量/函数 |
|-----|-------|---------|
| SMEM 大小计算 | 261-297 | `_setup_attributes` 内的计算 |
| SharedStorage 定义 | 768-812 | `@cute.struct class SharedStorage` |
| TMEM 偏移 | 216-230 | `self.tmem_s_offset`, `self.tmem_o_offset` 等 |
| Scale 缓冲 | 214-215 | `self.tmem_vec_offset` |

### 2.6 重要的 Layout 对象

```python
# SMEM 布局（CuTe 的 composed layout）
sQ_layout = make_smem_layout_a(tiled_mma_qk, mma_tiler_qk, q_dtype, q_stage)
sK_layout = make_smem_layout_b(tiled_mma_qk, mma_tiler_qk, k_dtype, kv_stage)
tP_layout = make_smem_layout_a(tiled_mma_pv, mma_tiler_pv, q_dtype, s_stage)
sV_layout = make_smem_layout_b(tiled_mma_pv, mma_tiler_pv, v_dtype, kv_stage)
sO_layout = make_smem_layout_epi(o_dtype, o_layout, epi_tile, q_stage)

# 实际 SMEM 张量（kernel 中）
sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
sV = cute.make_tensor(cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer)
sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
    # 或（overlap 时）
sO = cute.make_tensor(cute.recast_ptr(sQ.iterator, sO_layout.inner, o_dtype), sO_layout.outer)
```

---

## 阶段 3：TMA (Tensor Memory Access) 配置

### 3.1 TMA Load/Store Operators

```python
# CTA 群体（单 CTA 或 2CTA）
cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

# 定义 TMA 操作
tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)   # Global → Shared
tma_store_op = cpasync.CopyBulkTensorTileS2GOp()          # Shared → Global
```

### 3.2 TMA Descriptor 生成

```python
# Q 的 TMA（若启用 use_tma_Q）
if use_tma_Q:
    tma_atom_Q, mQ_transformed = cute.nvgpu.make_tiled_tma_atom_A(
        tma_load_op,
        mQ,                          # Global memory tensor
        cute.select(sQ_layout, mode=[0, 1, 2]),  # SMEM 的前 3 维（不含 stage）
        mma_tiler_qk,               # MMA tiling
        tiled_mma_qk,               # Tiled MMA 对象
        cta_layout_vmnk.shape,      # CTA 布局
    )
else:
    # Fallback 到异步复制
    async_copy_elems = 128 // q_dtype.width
    gmem_tiled_copy_Q = copy_utils.tiled_copy_2d(
        q_dtype, threads_per_row, num_load_threads, async_copy_elems, is_async=True
    )

# K 的 TMA（若启用 use_tma_KV）
if use_tma_KV:
    tma_atom_K, mK_transformed = cute.nvgpu.make_tiled_tma_atom_B(...)
    tma_atom_V, mV_transformed = cute.nvgpu.make_tiled_tma_atom_B(...)

# O 的 TMA（若启用 use_tma_O，仅适用 SM90+）
if use_tma_O:  # 检查：SM90+, varlen=False, 无 pack_gqa 冲突
    tma_atom_O, mO_transformed = cpasync.make_tiled_tma_atom(
        tma_store_op, mO, cute.select(sO_layout, mode=[0, 1]), epi_tile
    )
else:
    gmem_tiled_copy_O = cute.make_tiled_copy_tv(
        atom_universal_copy, tO_layout, vO_layout
    )
```

### 3.3 TMA 拷贝字节数（影响带宽分配）

```python
# 计算每个 TMA 操作的数据量（用于管道费用估计）
tma_copy_bytes = {}
for name, mX, layout in [("Q", mQ, sQ_layout), ("K", mK, sK_layout), ("V", mV, sV_layout)]:
    tma_copy_bytes[name] = cute.size_in_bytes(mX.element_type, cute.select(layout, [0, 1, 2]))

# 乘以 CTA 组大小（若 2CTA）
for name in ("Q", "K", "V"):
    tma_copy_bytes[name] *= cta_group_size

# 用于 pipeline 初始化
tx_count_Q = tma_copy_bytes["Q"]
tx_count_KV = tma_copy_bytes["K"]
```

### 3.4 代码位置

| 任务 | 代码行 | 函数/变量 |
|-----|-------|---------|
| TMA 操作定义 | 681-682 | `tma_load_op`, `tma_store_op` |
| Q 的 TMA | 688-703 | `make_tiled_tma_atom_A` 或 `tiled_copy_2d` |
| K、V 的 TMA | 708-728 | `make_tiled_tma_atom_B` |
| O 的 TMA | 730-758 | `make_tiled_tma_atom` 或 fallback |
| 拷贝字节数 | 677-685 | `self.tma_copy_bytes` 字典 |

---

## 阶段 4：Pipeline 初始化与同步机制

### 4.1 Pipeline 对象概览

| Pipeline 名 | 方向 | producer_group | consumer_group | 作用 |
|-----------|-----|----------------|----------------|------|
| `pipeline_q` | 加载 | TMA warp 或 Load threads | MMA warp | Q 的加载同步 |
| `pipeline_kv` | 加载 | TMA warp 或 Load threads | MMA warp | K、V 的加载同步 |
| `pipeline_s_p_o` | UMMA | MMA warp | Softmax+Correction threads | S ready + P consumed + O rescaled |
| `pipeline_p_lastsplit` | 异步 | Softmax warps | MMA warp | 最后一个 K 块的 P ready |
| `pipeline_o_acc` | UMMA | MMA warp | Correction threads | O 累加完成 |
| `pipeline_sm_stats` | 异步 | Softmax threads | Correction threads | 统计 ready（row_max, row_sum） |
| `pipeline_o_epi` | 异步 | Correction threads | Epilogue threads | O rescaled ready（若不用 correction 做 epilogue） |

### 4.2 Pipeline 创建代码

```python
# Q Pipeline（TMA/Async UMMA）
if use_tma_Q:
    pipeline_q = pipeline_custom.PipelineTmaUmma.create(
        barrier_storage=storage.mbar_load_Q.data_ptr(),
        num_stages=q_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # 单个 TMA warp
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),  # MMA warp
        tx_count=tma_copy_bytes["Q"],
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    )
else:
    pipeline_q = pipeline_custom.PipelineAsyncUmma.create(
        barrier_storage=storage.mbar_load_Q.data_ptr(),
        num_stages=q_stage,
        producer_group=load_threads,
        consumer_group=mma_warp,
        cta_layout_vmnk=cta_layout_vmnk,
        defer_sync=True,
    )

# S-P-O Pipeline（MMA → Softmax → Correction）
pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
    barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
    num_stages=q_stage,
    producer_group=mma_warp,                      # MMA 生产 S
    consumer_group=softmax_correction_threads_cluster,  # Softmax+Correction 消费
    cta_layout_vmnk=cta_layout_vmnk,
    defer_sync=True,
)

# Softmax 统计 Pipeline
pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
    barrier_storage=storage.mbar_softmax_stats.data_ptr(),
    num_stages=q_stage,
    producer_group=softmax_threads,
    consumer_group=correction_threads,
    defer_sync=True,
)
```

### 4.3 Named Barrier（线程间同步）

```python
# TMEM 分配/释放 barrier（协调 MMA 和 Softmax/Correction）
tmem_alloc_barrier = pipeline.NamedBarrier(
    barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
    num_threads=WARP_SIZE * (1 + 4 + 4 + 4),  # 1 MMA + 2 Softmax groups + 1 Correction
)

# Softmax 统计 barrier（Softmax warps 0 之间的同步）
sm_stats_barrier = pipeline_custom.NamedBarrier(
    barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
    num_threads=WARP_SIZE * 2,  # 仅 Softmax0 和 Softmax1？或另外的组织
)
```

### 4.4 Pipeline 状态机

```python
# 每个 stage 的状态机
q_producer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Producer, q_stage
)
kv_producer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Producer, kv_stage
)
softmax_consumer_state = pipeline.make_pipeline_state(
    pipeline.PipelineUserType.Consumer, q_stage
)

# 状态转移
state.advance()  # 已生产完当前 stage，准备下一个
state.acquire_w_index_phase(idx, phase)  # 获得特定 stage（带 phase）
state.wait_acquire()  # 等待可用
state.release()  # 消费完成
```

### 4.5 代码位置

| 任务 | 代码行 | 函数/对象 |
|-----|-------|---------|
| Pipeline 创建 | 1182-1241 | 各 `pipeline_*.create()` 调用 |
| Barrier 初始化 | 1243-1248 | `tmem_alloc_barrier`, `sm_stats_barrier` |
| Pipeline 状态机 | kernel 各处 | `.producer_tail()`, `.advance()` 等 |

---

## 阶段 5：Warp 角色初始化与控制流

### 5.1 Warp 分配常量

```python
self.softmax0_warp_ids = (0, 1, 2, 3)      # 第一个 Softmax 组
self.softmax1_warp_ids = (4, 5, 6, 7)      # 第二个（或可能为空）
self.correction_warp_ids = (8, 9, 10, 11)  # Correction/Rescale
self.mma_warp_id = 12                       # 单个 MMA warp
self.epilogue_warp_ids = (13,)              # Epilogue
self.load_warp_ids = (14,)                  # Load（TMA 或 async）
self.empty_warp_ids = (15,)                 # 未来用

self.threads_per_cta = WARP_SIZE * 16       # 总共 512 个线程
```

### 5.2 Warp 分配的动态调整

根据 `q_stage` 和是否启用 TMA/Async：

```python
# 若 q_stage == 1（仅一个 Q stage）
if q_stage == 1:
    if not use_tma_KV or not use_tma_Q:
        # 没有 TMA，用 Load warp 做异步复制
        empty_warp_ids = empty_warp_ids + load_warp_ids
        load_warp_ids = softmax1_warp_ids  # Softmax1 不用，改成 Load
    else:
        # 用 TMA，Softmax1 空着
        empty_warp_ids = empty_warp_ids + softmax1_warp_ids
    softmax1_warp_ids = ()

# 若 q_stage == 2 且无 TMA KV
elif not use_tma_KV:
    load_warp_ids = (14, 15)  # 两个 Load warp
    empty_warp_ids = ()

# 若启用 correction_warps_for_epi（varlen 模式）
if use_correction_warps_for_epi:
    empty_warp_ids = empty_warp_ids + epilogue_warp_ids
    epilogue_warp_ids = correction_warp_ids  # Correction 兼任 Epilogue
```

### 5.3 Kernel 中的 Warp 分派

```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

# Empty / CLC Scheduler Warp
if use_clc_scheduler:
    if warp_idx == clc_scheduler_warp_id:
        # 调度器逻辑
        self.clc_scheduler_warp(tile_scheduler)

# Load Warp
if warp_idx >= load_warp_ids[0] and warp_idx <= load_warp_ids[-1]:
    self.load(...)

# MMA Warp
if warp_idx == mma_warp_id:
    tmem.allocate(...)
    self.mma(...)
    tmem.free(...)

# Epilogue Warp
if not use_correction_warps_for_epi:
    if warp_idx >= epilogue_warp_ids[0] and warp_idx <= epilogue_warp_ids[-1]:
        self.epilogue_s2g(...)

# Softmax Warp
if (q_stage == 2 and warp_idx <= softmax1_warp_ids[-1]) or \
   (q_stage == 1 and warp_idx <= softmax0_warp_ids[-1]):
    # Softmax 状态=0 或 1 取决于 warp_idx
    stage = ... # 计算 stage
    self.softmax_loop(stage=stage, ...)

# Correction Warp
if warp_idx >= correction_warp_ids[0] and warp_idx < mma_warp_id:
    self.correction_loop(...)
```

### 5.4 代码位置

| 任务 | 代码行 | 变量/逻辑 |
|-----|-------|---------|
| Warp 标号定义 | 1176-1180 | `self.softmax0_warp_ids` 等 |
| 动态调整 | 1181-1197 | 条件判断后修改 warp_ids |
| SoftMax 进入条件 | 2009-2019 | `if (q_stage==2 and ...) or (q_stage==1 and ...)` |
| Correction 进入条件 | 2064 | `if warp_idx >= correction_warp_ids[0] ...` |
| Epilogue 进入条件 | 2027 | `if const_expr(not use_correction_warps_for_epi) and ...` |

---

## 阶段 6：Tile 调度器与工作分配

### 6.1 调度器类型选择

```python
# 根据模式选择调度器
if is_varlen_q:
    TileScheduler = SingleTileVarlenScheduler
elif is_causal or is_local or use_clc_scheduler:
    TileScheduler = SingleTileLPTScheduler  # 左到右（causal）
elif is_persistent:
    TileScheduler = StaticPersistentTileScheduler  # 持久化（所有 block 循环）
else:
    TileScheduler = SingleTileScheduler  # 单个 tile（一般性）
```

### 6.2 TileSchedulerArguments 构建

```python
tile_sched_args = TileSchedulerArguments(
    m = cute.ceil_div(mQ.shape[0], _num_block_divisor),  # Q 行数 / m_block_size
    h = cute.size(mQ.shape[2]),                           # head 数
    b = cute.size(mQ.shape[3]) or ...,                    # batch 或 num_seqs
    s = num_splits,                                       # split-kv 分支数
    k = cute.size(mK.shape[0]) or ...,                    # K 行数或 paged 对应
    sK = mQ.shape[1],                                     # Q 行数（已 shape[0]）
    dV = mV.shape[0],                                     # V 的 head_dim_v
    total_q = ...,                                        # 总 Q 元素数
    tile_shape_mn = self.cta_tiler[:2],                  # (m_block_size * q_stage, n_block_size)
    mCuSeqlensQ = mCuSeqlensQ,                            # 变长相关
    qhead_per_kvhead_packgqa = qhead_per_kvhead or 1,     # GQA 倍数
    is_persistent = self.is_persistent,
    lpt = self.is_causal or self.is_local,                # left-to-right
    cluster_shape_mn = self.cluster_shape_mn,
    use_cluster_idx = not is_persistent and cta_group_size > 1,
)

tile_sched_params = TileScheduler.to_underlying_arguments(
    tile_sched_args, scheduling_mode=self.scheduling_mode
)
```

### 6.3 Grid 尺寸计算

```python
grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
# 返回 (num_ctas_m, num_ctas_n, ...)
# 用于 kernel launch：
kernel.launch(
    grid=grid_dim,
    block=[threads_per_cta, 1, 1],
    cluster=cluster_shape_mnk if size(cluster_shape_mnk) > 1 else None,
    ...
)
```

### 6.4 Tile 信息结构

在 load 和 mma 中反复使用：

```python
work_tile = tile_scheduler.initial_work_tile_info()
while work_tile.is_valid_tile:
    m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
    
    # 处理当前 tile...
    
    work_tile = tile_scheduler.advance_to_next_work()
```

### 6.5 代码位置

| 任务 | 代码行 | 函数/变量 |
|-----|-------|---------|
| 调度器选择 | 208-216 | `self.TileScheduler = ...` |
| Arguments 构建 | 734-759 | `TileSchedulerArguments(...)` |
| 参数转换 | 760-761 | `.to_underlying_arguments()` |
| Grid 计算 | 762 | `.get_grid_shape()` |

---

## 阶段 7：MMA 操作配置（Q*K^T 和 P*V）

### 7.1 MMA Operator 创建

```python
# 使用 TCGEN05（Blackwell 特定）
cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
q_major_mode = tcgen05.OperandMajorMode.K      # Q 是 K-major（行优先）
k_major_mode = tcgen05.OperandMajorMode.K      # K 也是 K-major
v_major_mode = tcgen05.OperandMajorMode.MN     # V 是 M-major / N-major 混合

# 构建 MMA 对象
tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
    q_dtype,       # 输入 A (Q) 的 dtype
    q_major_mode,  # K-major
    k_major_mode,  # K-major
    qk_acc_dtype,  # 累加器 dtype（Float32）
    cta_group,     # 单/双 CTA
    mma_tiler_qk[:2],  # (256 或 128, 128) for (m, n)（示例）
)

tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
    v_dtype,
    p_major_mode,
    v_major_mode,
    pv_acc_dtype,
    cta_group,
    mma_tiler_pv[:2],
    p_source=tcgen05.OperandSource.TMEM,  # P 来自 TMEM！
)
```

### 7.2 Tiling 与线程分区

```python
# 从 tiled_mma_qk 获取单线程的 MMA 对象
thr_mma_qk = tiled_mma_qk.get_slice(mma_tile_coord_v)
thr_mma_pv = tiled_mma_pv.get_slice(mma_tile_coord_v)

# 获取累加器的形状
qk_acc_shape = thr_mma_qk.partition_shape_C(mma_tiler_qk[:2])
pv_acc_shape = thr_mma_pv.partition_shape_C(mma_tiler_pv[:2])

# 创建累加器张量（存储在 TMEM 中）
tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, s_stage))
tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, q_stage))

# 重新指向 TMEM 的正确偏移
tOtO = cute.make_tensor(tOtO.iterator + tmem_o_offset[0], tOtO.layout)
```

### 7.3 SMEM 输入分区

```python
# Q、K、V 在 SMEM 中的分区（由 MMA 对象做）
tSrQ = thr_mma_qk.make_fragment_A(sQ)  # Q 分片
tSrK = thr_mma_qk.make_fragment_B(sK)  # K 分片
tOrV = thr_mma_pv.make_fragment_B(sV)  # V 分片

# 对于 q_stage=2，需要 Q 的两个阶段
tSrQs = (tSrQ[None, None, None, 0], tSrQ[None, None, None, 1])
```

### 7.4 2CTA 指令的特殊性

当 `use_2cta_instrs=True`：
- MMA tiler 在 M 维翻倍：`(2*m_block_size, n_block_size, d)`
- 两个 CTA 的线程共用一条 TCGEN05 指令
- Cluster shape 为 (2, 1)
- SMEM/TMEM 的 stage stride 需调整以容纳两个 CTA 的数据

### 7.5 代码位置

| 任务 | 代码行 | 函数/变量 |
|-----|-------|---------|
| MMA 对象创建 | 423-436 | `make_trivial_tiled_mma(...)` |
| 线程 MMA 获取 | 1213-1214 | `thr_mma_qk = tiled_mma_qk.get_slice(...)` |
| 累加器初始化 | 1215-1225 | `thr_mma_qk.make_fragment_C(...)` |
| SMEM 分区 | 1593-1596 | `thr_mma_qk.make_fragment_A(sQ)` 等 |

---

## 参考资源

- **SM100 工具**：`flash_attn/cute/blackwell_helpers.py`
- **MMA 描述**：`flash_attn/cute/mma_sm100_desc.py`
- **Pipeline 库**：`flash_attn/cute/pipeline.py`
- **Softmax 实现**：`flash_attn/cute/softmax.py` 中的 `SoftmaxSm100` 类
- **CuTe 库**：cutlass/cute 模块

