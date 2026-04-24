# `cute_gemm`

最小版 CuTeDSL GEMM 实验目录。

当前内容：
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_1cta_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_1cta_cutedsl.py): 只用 `tcgen05.mma` 的 `1cta` CuTeDSL GEMM
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_pipeline_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_pipeline_cutedsl.py): 用 `PipelineUmmaAsync` 做 `2cta` 同步的版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_commit_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_commit_cutedsl.py): 用底层 `commit` 协议做 `2cta` 同步的版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_nopipeline_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_nopipeline_cutedsl.py): `2cta + TMA load A/B + 单 stage` 对照版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_2stage_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_2stage_cutedsl.py): `2cta + TMA load A/B + 2-stage AB pipeline` 版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_3stage_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_3stage_cutedsl.py): `2cta + TMA load A/B + 3-stage AB pipeline` 版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_pipeline_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_pipeline_cutedsl.py): `2cta + TMA load A/B + AB pipeline` 版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_6stage_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_6stage_cutedsl.py): `2cta + TMA load A/B + 6-stage AB pipeline` 版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_pipeline_tma_store_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_pipeline_tma_store_cutedsl.py): `2cta + TMA load A/B + AB pipeline + TMA store C` 版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/ref.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/ref.py): `torch` reference
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/benchmark.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/benchmark.py): 正确性验证 + 性能对比，输出 Torch 分配版和预分配 cuBLAS/cuBLASLt baseline
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/configs.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/configs.py): autotune candidate 配置
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/autotune.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/autotune.py): 逐 shape 测候选 kernel 并选择最快配置
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/cublaslt_benchmark.cu](/Users/meiziyuan/Roofline-Analysis/cute_gemm/cublaslt_benchmark.cu): cuBLASLt C++ algo heuristic baseline
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/run.sh](/Users/meiziyuan/Roofline-Analysis/cute_gemm/run.sh): 统一运行入口

当前约束：
- 不用 TMA
- 不做 AB pipeline
- tile 固定为 `(128, 256, 64)`
- `M/N/K` 需要分别整除 `(128, 256, 64)`
- `A/B` 是 `fp16`
- 输出 `C` 是 `fp32`

`2cta` 版本当前约束：
- 使用官方 tutorial 风格的 `2cta + tma + pipeline`
- tile 固定为 `(256, 256, 64)`
- `M/N/K` 需要分别整除 `(256, 256, 64)`
- `A/B` 是 `fp16`
- 输出 `C` 是 `fp16`

优化拆解顺序：
1. `2cta_pipeline`: baseline，普通线程搬 A/B 到 SMEM，`PipelineUmmaAsync` 只同步 MMA 完成。
2. `2cta_tma_nopipeline`: A/B load 改 TMA multicast，但 `ab_stages=1`，隔离 TMA load 收益。
3. `2cta_tma_2stage`: A/B TMA load 加 2-stage AB pipeline，观察浅 pipeline 对隐藏 TMA latency 的收益。
4. `2cta_tma_3stage`: A/B TMA load 加 3-stage AB pipeline，用于 AB stage sweep。
5. `2cta_tma_pipeline`: A/B TMA load 加 4-stage AB pipeline，作为当前深 pipeline 版本。
6. `2cta_tma_6stage`: A/B TMA load 加 6-stage AB pipeline，确认更深 pipeline 是否被 SMEM/同步开销抵消。
7. `2cta_tma_pipeline_tma_store`: 保持 4-stage AB pipeline，把 C 写回改成 SMEM staging + TMA store。
8. 后续候选：2-stage/4-stage epilogue、只 multicast A/B 单边对照、tile shape sweep、persistent scheduler。

Autotune 参数说明：
- `ab_stages` 是 compile-time tuning knob，会影响 shared storage、SMEM layout 和 `PipelineTmaUmma` stage 数。
- `tma_store` 也是 compile-time tuning knob，会影响 epilogue SMEM 分配和 C 的 TMA store atom。
- `tile_shape` 是 compile-time tuning knob，当前 autotune metadata 已记录，后续需要新增对应 kernel candidate 才能真正 sweep。
- warp specialization 数量也应该调优，目前候选固定为 `1 TMA warp + 1 MMA warp + 4 epilogue warps = 192 threads`；后续可以加入 `160/192/224 threads`、`epilogue_warps=3/4/5` 等候选。

单个 shape 正确性：

```bash
cd /Users/meiziyuan/Roofline-Analysis
bash cute_gemm/run.sh --mnk 128,256,64
```

`2cta` 学习版单独运行：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/mma_gemm_2cta_pipeline_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_commit_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_nopipeline_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_2stage_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_3stage_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_pipeline_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_6stage_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_pipeline_tma_store_cutedsl.py --mnk 256,256,64
```

小 shape 对比：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/benchmark.py --variant 1cta --shape-set small
```

大 shape 对比：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/benchmark.py --variant 1cta --shape-set large
```

`2cta` 小 shape：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/benchmark.py --variant 2cta --shape-set small
```

`2cta` 大 shape：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/benchmark.py --variant 2cta --shape-set large
```

两个版本一起跑：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/benchmark.py --variant all --shape-set all
```

手动指定 shape：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/benchmark.py --variant 2cta --shapes 256,256,64 1024,1024,256
```

`2cta + TMA pipeline` 正确性和 benchmark：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/mma_gemm_2cta_tma_nopipeline_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_2stage_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_pipeline_cutedsl.py --mnk 256,256,64
python3 cute_gemm/mma_gemm_2cta_tma_pipeline_tma_store_cutedsl.py --mnk 256,256,64
python3 cute_gemm/benchmark.py --variant 2cta_tma_nopipeline --shape-set small
python3 cute_gemm/benchmark.py --variant 2cta_tma_2stage --shape-set small
python3 cute_gemm/benchmark.py --variant 2cta_tma_3stage --shape-set small
python3 cute_gemm/benchmark.py --variant 2cta_tma_pipeline --shape-set small
python3 cute_gemm/benchmark.py --variant 2cta_tma_6stage --shape-set small
python3 cute_gemm/benchmark.py --variant 2cta_tma_pipeline_tma_store --shape-set small
python3 cute_gemm/benchmark.py --variant 2cta_tma_pipeline --shape-set large
```

benchmark 输出包含 `flops`、各 backend 的 `*_tflops`，并可通过 `--cublaslt-bin cute_gemm/cublaslt_benchmark` 追加 C++ cuBLASLt baseline。

AB stage sweep：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/benchmark.py --variant 2cta_tma_nopipeline --shape-set all
python3 cute_gemm/benchmark.py --variant 2cta_tma_2stage --shape-set all
python3 cute_gemm/benchmark.py --variant 2cta_tma_3stage --shape-set all
python3 cute_gemm/benchmark.py --variant 2cta_tma_pipeline --shape-set all
python3 cute_gemm/benchmark.py --variant 2cta_tma_6stage --shape-set all
```

cuBLASLt C++ baseline：

```bash
cd /Users/meiziyuan/Roofline-Analysis
nvcc -O3 -std=c++17 cute_gemm/cublaslt_benchmark.cu -lcublasLt -lcublas -o cute_gemm/cublaslt_benchmark
./cute_gemm/cublaslt_benchmark --mnk 4096,2048,512 --warmup 20 --iters 100 --algos 32 --workspace-mb 64
```

Autotune：

```bash
cd /Users/meiziyuan/Roofline-Analysis
python3 cute_gemm/autotune.py --group ab-stage --shape-set all
python3 cute_gemm/autotune.py --group tma-store --shape-set all
python3 cute_gemm/autotune.py --group default --shape-set all --warmup 10 --iters 50
python3 cute_gemm/autotune.py --group default --shapes 4096,2048,512 --cublaslt-bin cute_gemm/cublaslt_benchmark
```

autotune 结果会写到 `cute_gemm/autotune_results/latest.json`，同时保留带时间戳的 JSON。
