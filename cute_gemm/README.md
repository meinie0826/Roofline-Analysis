# `cute_gemm`

最小版 CuTeDSL GEMM 实验目录。

当前内容：
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_1cta_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_1cta_cutedsl.py): 只用 `tcgen05.mma` 的 `1cta` CuTeDSL GEMM
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_pipeline_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_pipeline_cutedsl.py): 用 `PipelineUmmaAsync` 做 `2cta` 同步的版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_commit_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_commit_cutedsl.py): 用底层 `commit` 协议做 `2cta` 同步的版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_pipeline_cutedsl.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_2cta_tma_pipeline_cutedsl.py): `2cta + TMA load A/B + AB pipeline` 版本
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/ref.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/ref.py): `torch` reference
- [/Users/meiziyuan/Roofline-Analysis/cute_gemm/benchmark.py](/Users/meiziyuan/Roofline-Analysis/cute_gemm/benchmark.py): 正确性验证 + 性能对比
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
python3 cute_gemm/mma_gemm_2cta_tma_pipeline_cutedsl.py --mnk 256,256,64
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
python3 cute_gemm/mma_gemm_2cta_tma_pipeline_cutedsl.py --mnk 256,256,64
python3 cute_gemm/benchmark.py --variant 2cta_tma_pipeline --shape-set small
python3 cute_gemm/benchmark.py --variant 2cta_tma_pipeline --shape-set large
```
