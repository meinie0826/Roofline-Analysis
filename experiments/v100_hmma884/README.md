# V100 HMMA.884 Latency Study

这个目录用于在 Volta V100 (`sm_70`) 上系统测试 `mma.sync.aligned.m8n8k4` 的 lowering 形式，以及尽量逼近“一条高层 PTX `mma` 指令”的程序可见周期。

## 目标

这套实验重点回答三类问题：

1. 不同 dtype 路线在 V100 上会 lower 成几条 `HMMA.884`
2. dependent chain 与 multiple independent streams 的 `cycles_per_mma` 差多少
3. 实验结果如何稳定保存，便于后续复盘和横向对比

## 当前 benchmark matrix

| binary | PTX form | mode | streams | 目的 |
|---|---|---|---:|---|
| `bench_empty` | none | `empty` | 0 | 匹配 loop skeleton 的空基准 |
| `bench_empty_matched_dep` | none | `empty_matched_dep` | 1 | 与 `bench_mma_f16_dep` 同骨架的 matched-empty（显式标量 ALU 指令，用于 subtraction） |
| `bench_empty_matched_f32acc` | none | `empty_matched_f32acc` | 1/2/4/8 | 与 `bench_mma_f32acc` 同骨架的 matched-empty（显式标量 ALU 指令，用于 subtraction） |
| `bench_mma_f16_dep` | `m8n8k4 row.col f16,f16,f16,f16` | `dep` | 1 | 单依赖链 latency 代理 |
| `bench_mma_f16_indep` | `m8n8k4 row.col f16,f16,f16,f16` | `indep` | 2/4/8 | 吞吐上限近似（编译期固定 streams） |
| `bench_mma_f32acc` | `m8n8k4 row.col f32,f16,f16,f16` | `fixedc` / `fixedc_indep` | 1/2/4/8 | mixed path lowering 与多 stream 对比（编译期固定 streams） |

## 文件说明

- [common.h](common.h)：共享参数解析、结果结构、打印格式、CUDA 检查逻辑
- [bench_empty.cu](bench_empty.cu)：matched empty baseline
- [bench_empty_matched_dep.cu](bench_empty_matched_dep.cu)：与 `bench_mma_f16_dep` 同构、且不会被优化为空的 subtraction baseline
- [bench_empty_matched_f32acc.cu](bench_empty_matched_f32acc.cu)：与 `bench_mma_f32acc` 同构、且不会被优化为空的 subtraction baseline
- [bench_mma_f16_dep.cu](bench_mma_f16_dep.cu)：FP16 path 单链依赖
- [bench_mma_f16_indep.cu](bench_mma_f16_indep.cu)：FP16 path 多独立链
- [bench_mma_f32acc.cu](bench_mma_f32acc.cu)：mixed path，多 stream 版本
- [run.sh](run.sh)：统一编译、反汇编、运行、写 raw/summary/metadata

## 方法说明

### 1. 为什么要分 dep / indep / empty

- `dep`：让同一组 accumulator 被连续复用，尽量逼近链式依赖 latency
- `indep`：在一个 loop body 里交错多组独立 accumulator，观察吞吐极限
- `empty`：保留相似 loop 结构，用来识别 `clock64()`、loop control、寄存器活动本身的影响

### 2. 为什么要调 `streams`

对于 `indep` 类 benchmark（`bench_mma_f16_indep` 与 `bench_mma_f32acc`），当前实现会在 host 侧按 `streams` 分派到模板实例化的 kernel，计时区内不再包含运行时 `for (s < streams)` 分支：

- `streams=2`：轻度打散依赖
- `streams=4`：更接近 steady-state issue
- `streams=8`：进一步观察是否继续下降到吞吐平台

如果 `dep` 显著大于 `indep8`，通常说明你确实看到了链式依赖代价与吞吐上限的差异。

### 3. 关于 mixed path 的 caveat

`bench_mma_f32acc` 使用的是：

```ptx
mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16
```

这条 Volta PTX 形式里，`D` 是 `f32`，但 `C` 仍是 4 个 packed `f16x2` 寄存器，所以它不能像纯 `f16 -> f16` 路线那样直接把上一条输出原样回灌成下一条输入。
因此这里的 `streams=1` 更准确地表示 **single-stream fixed-C**，`streams>1` 表示 **multi-stream fixed-C throughput-ish**；它主要用于：

- 验证 lowering 是否出现 `HMMA.884.F32.F16.STEP0/1/2/3`
- 比较 single-stream 与 multi-stream 的 steady-state 差异

不要把它和 `bench_mma_f16_dep` 当成完全同口径的“真依赖链 latency”结果。

## 运行方式

默认运行：

```bash
cd experiments/v100_hmma884
bash run.sh
```

默认会 sweep 多个 `unroll`：`UNROLL_LIST=1,2,4,8,16`，并对 `f16/f32` 的 stream 组合做全矩阵运行。

显式指定工具：

```bash
NVCC=/usr/local/corex/bin/nvcc CUOBJDUMP=cuobjdump bash run.sh
```

指定参数：

```bash
REPEATS=20 LOOP_ITERS=8192 WARMUP_ITERS=128 WARMUP_LAUNCHES=5 \
UNROLL_LIST=1,2,4,8,16 F16_STREAMS_LIST=2,4,8 F32_STREAMS_LIST=1,2,4,8 \
bash run.sh
```

指定结果目录：

```bash
OUTDIR=results/manual_run_01 bash run.sh
```

## 输出文件

每次运行默认写入：

```text
results/<timestamp>/
```

其中包含：

- `results_raw.csv`：每次 repeat 一行原始结果
- `results_summary.csv`：按 benchmark 聚合后的均值/中位数/stddev/min/max
- `results_derived.csv`：自动派生 `f16 dep` 与 `f32acc` 相对各自 matched-empty 的差值
- `results_baseline_compare.csv`：同时比较 `raw empty` 与 `matched empty` 两种减法口径
- `metadata.json`：GPU / nvcc / git commit / 参数等元数据
- `run.log`：整次运行的终端日志
- `sass_*.txt`：每个 binary 的完整 SASS dump
- `bench_*_u*_s*.txt`：每个 benchmark 每组参数的单独 stdout 记录

### `results_raw.csv` 字段

- `benchmark`
- `dtype`
- `mode`
- `streams`
- `repeat`
- `loop_iters`
- `unroll`
- `total_mma`
- `cycles`
- `cycles_per_mma`
- `sink_lo`
- `sink_hi`

### `results_summary.csv` 字段

- `benchmark`
- `dtype`
- `mode`
- `streams`
- `loop_iters`
- `unroll`
- `repeats`
- `mean_cycles`
- `median_cycles`
- `stddev_cycles`
- `hmma_steps_per_mma`
- `mean_cycles_per_mma`
- `median_cycles_per_mma`
- `stddev_cycles_per_mma`
- `min_cycles_per_mma`
- `max_cycles_per_mma`
- `mean_cycles_per_hmma_step`：按 dtype 自动归一化后的均值（`f16` 用 2 step，`f32acc` 用 4 step）

### `results_derived.csv` 字段

- `benchmark`
- `mode`
- `loop_iters`
- `unroll`
- `streams`
- `mean_cycles`
- `matched_empty_mean_cycles`
- `delta_cycles`
- `dep_mean_cycles_per_mma`
- `matched_empty_mean_cycles_per_mma`
- `delta_cycles_per_mma`
- `delta_cycles_per_hmma_step`

### `results_baseline_compare.csv` 字段

- `benchmark`
- `mode`
- `loop_iters`
- `unroll`
- `streams`
- `mean_cycles`
- `raw_empty_mean_cycles`
- `delta_vs_raw_empty_cycles`
- `delta_vs_raw_empty_cycles_per_mma`
- `delta_vs_raw_empty_cycles_per_hmma_step`
- `matched_empty_mean_cycles`
- `delta_vs_matched_empty_cycles`
- `delta_vs_matched_empty_cycles_per_mma`
- `delta_vs_matched_empty_cycles_per_hmma_step`

## 建议检查顺序

### 1. 先看 lowering

```bash
grep HMMA results/<timestamp>/sass_bench_mma_f16_dep.txt
grep HMMA results/<timestamp>/sass_bench_mma_f16_indep.txt
grep HMMA results/<timestamp>/sass_bench_mma_f32acc.txt
```

优先验证：

- `f16,f16,f16,f16` 是否主要出现 `HMMA.884.F16.F16.STEP0/1`
- `f32,f16,f16,f16` 是否出现 `HMMA.884.F32.F16.STEP0/1/2/3`

### 2. 再看 summary

重点看：

- `bench_mma_f16_dep` vs `bench_empty_matched_dep`（是否得到稳定、正值的 subtraction）
- `bench_mma_f16_dep` vs `bench_mma_f16_indep streams=8`
- `bench_mma_f32acc streams=1` vs `streams=8`
- `bench_empty` 是否异常接近 dependent case

## 如何解释结果

推荐口径：

- `dep`：程序可见的链式 latency 代理
- `dep - empty_matched_dep`：更接近“去除 loop/control 后”的链式 latency 代理
- `indep8`：更接近 steady-state throughput 代理
- `empty`：控制开销基线

如果你看到：

- `dep` 明显高于 `indep4/8`
- 且不同 repeat 的 stddev 不大

那这组结果通常是有解释价值的。

如果你看到：

- `empty` 和 `dep` 差不多
- 或者 `indep` 低得离谱

那说明当前结构仍被 loop / compiler scheduling 明显污染，应该优先相信 lowering 结论，而不是把 `cycles_per_mma` 当成“单条 HMMA step 的真实硬件周期”。

## 后续可扩展方向

- 增加整数 tensor core 路径（如 `s8/s8 -> s32`）
- 做更严格的 matched-empty control
- 系统枚举 `row/col` 与 dtype 组合
- 增加 host 侧自动做 empty subtraction 的派生 summary
- 对不同 CUDA / ptxas 版本重复实验，比较 lowering 差异
