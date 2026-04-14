# CuTe DSL Causal Attention Roadmap

目标限定为 `q, k, v -> o` 的 `causal attention` forward，不处理 dropout、alibi、window、varlen、paged KV、GQA/MQA、backward，也不引入额外分支条件。

工程分成 6 个实现阶段，统一输入输出接口，方便逐阶段验证正确性和性能。
另外保留一个 `baseline_fa4` 对照入口，只用于 correctness / benchmark，不作为最终实现路径。

## Stage 0: Naive

- 计算图: `S = Q @ K^T -> causal mask -> softmax(S) -> O = P @ V`
- 特点: 不做融合，显式 materialize score，CTA-per-row
- 目的: 建立最直观的正确性基线

## Stage 1: Online Softmax

- 优化点: 去掉整行 `score` 的全量保存，改成 running max / running sum
- 收益: 降低中间存储压力，接近 FlashAttention 的核心数值形式

## Stage 2: KV Blocking

- 优化点: 按 `N` 维分块遍历 KV
- 收益: 为 shared memory staging 和主循环流水化做准备

## Stage 3: Shared Memory + Warp MMA

- 优化点: Q/K/V tile 化，shared memory staging，warp / tensor core MMA
- 收益: 从“正确”切到“高吞吐”

## Stage 4: Pipeline + Specialized Schedule

- 优化点: cp.async / TMA、warp specialization、persistent tile scheduling
- 收益: 进入 FA4 的主优化区间

## Stage 5: Full Self-Hosted Kernel

- 不调用 `flash-attention/flash_attn/cute` 的执行实现
- 目标: 用我们自己的 CuTe DSL kernel 覆盖 full causal forward 路径
- `flash-attention` 和 `cutlass/examples/77_blackwell_fmha` 只作为设计和性能参考

## 当前代码状态

- 已实现:
  - `reference`
  - `stage0`
  - `stage1` 的自研 CuTe FA2 风格前向版（causal/qkv-only）
  - `stage1_ref` 的 PyTorch online-softmax 参考版
  - `stage2` 的 PyTorch blocked 参考版
  - `stage3` 的 CuTe blocked + online softmax 前向版
  - `stage4` 的 CuTe K-tile shared-memory staging 前向版
  - `stage5` 的 CuTe K/V-tile shared-memory staging 前向版
  - `stage6` 的 CuTe Q/K/V-tile shared-memory staging（Q 使用 fp16 缓存）前向版
  - `stage7` 的 CuTe Q/K/V/score-tile shared-memory staging（score/prob 使用 fp16 缓存）前向版
  - `baseline_fa4` 对照入口
  - `baseline_sdpa`（PyTorch SDPA 对照）

## 建议推进顺序

1. 先用 `reference` 和 `stage0` 把 causal correctness 固定住。
2. 再把 `stage1` 的 online softmax 数学形式迁移进 CuTe kernel。
3. 然后做 `stage2 -> stage3`，把 KV blocking 和 shared memory / MMA 结合起来。
4. 最后把 `stage4` 收敛成我们自己的 `stage5`，做到不依赖外部 FA4 执行实现。
5. 全程可以用 `baseline_fa4` 做 correctness 和性能对照，但不能让实现依赖它。

## 当前默认行为

- `stage0` 是纯 CuTe DSL 路径，不提供 PyTorch fallback。
- `stage1` 是自研 CuTe FA2 风格实现。
- `stage1_ref / stage2` 是 PyTorch 参考实现，用于验证数学形式。
- `stage3` 是我们自己的 CuTe blocked + online softmax 内核（causal/qkv-only/fwd）。
- `stage4` 是我们自己的 CuTe K 缓存（SMEM staging）内核。
- `stage5` 是我们自己的 CuTe K/V 缓存（SMEM staging）内核。
- `stage6` 是我们自己的 CuTe Q/K/V 缓存（Q 用 fp16，K/V 用 fp16）内核。
- `stage7` 是我们自己的 CuTe Q/K/V/score 缓存（score/prob 用 fp16）内核。
- 真正的自研 CuTe 主线会继续落在 `stage3 -> stage5`。

## Benchmark

对比 `stage0/stage1/stage2/stage3/stage4/stage5/stage6/stage7/baseline_fa4/baseline_sdpa`：

```bash
python benchmark.py \
  --stages all \
  --batch 1 --heads 16 --seqlen 1024 --headdim 128 \
  --dtype float16 --block-m 64 --block-n 128 --num-threads 128 \
  --warmup 5 --repeat 20
```
