# GEMM Reference Harness

这个目录是一个独立的 GEMM 基础设施目录，现在已经把 driver 和 backend 文件拆开了，方便后面逐步往里加 kernel：

- 一个最朴素的 row-major CUDA naive GEMM
- 一个 cuBLAS `sgemm` reference
- 一个步骤 1 的 `SIMT GEMM`
  - `cp.async` 读取 A/B tile 到 shared memory
  - double buffer
  - 先保证正确和结构清楚，不追求复杂 thread mapping
- 正确性校验
- benchmark loop
- 批量 shape 运行和结果落盘

后面如果你要继续塞 CUTLASS kernel、自写 kernel 或别的调度，只需要沿着这里的输入初始化、校验和结果输出往里接就行。

## 文件

```text
gemm_reference/
├── gemm_reference_common.h   # 公共选项、检查宏、统计/输出工具
├── bench_gemm_reference.cu   # benchmark driver / correctness driver
├── naive_gemm.cu             # 暴力 naive GEMM
├── cublas_reference.cu       # cuBLAS reference
├── simt_gemm_cp_async.cu     # 步骤 1: SIMT GEMM with cp.async + double buffer
├── step2_tcgen05_mma_gemm.cu # 步骤 2: tcgen05.mma warp-level GEMM
├── Makefile                  # 编译
├── run.sh                    # 批量 shape 跑 benchmark 并落盘
├── README.md                 # 说明
└── results/                  # 输出目录
```

## 主程序行为

现在有三个 backend：

- `naive`
  - 每个线程算一个 `C[row, col]`
  - 完全暴力三重循环，没有 tiling、shared memory、vectorize
- `simt`
  - 单独的 `simt_gemm_cp_async.cu`
  - `16x16` 线程块
  - `16x16x16` tile
  - `cp.async` 读 A/B tile 到 shared memory
  - 两级 double buffer
  - 每个线程负责一个输出元素
- `cublas`
  - 用 `cublasSgemm`
  - 通过 row-major / column-major 转置关系直接复用同一份 row-major 输入

另外有一个独立的步骤 2 程序：

- `step2_tcgen05_mma_gemm.cu`
  - 基于 Blackwell `tcgen05.mma + tcgen05.ld`
  - 参考 CUTLASS tutorial 01 的最小路径
  - 用 `half` 输入、`float` 输出
  - 初始化成小整数值，和 `cublasGemmEx(..., CUBLAS_COMPUTE_32F_PEDANTIC)` 做精确对拍
  - 目标是 `max_abs=0`

正确性默认拿 `cuBLAS` 输出当 reference，和自定义 kernel 做逐元素比较，输出：

- `max_abs`
- `max_rel`
- `l2_rel`
- `fail_count`

benchmark 默认输出每个 backend 的：

- `avg_ms`
- `median_ms`
- `min_ms`
- `gflops`
- `checksum`

## 编译

默认架构按这个 repo 的 Blackwell 机器习惯设成 `sm_100a`：

```bash
cd /Users/meiziyuan/Roofline-Analysis/experiments/gemm_reference
make
```

如果你在 B300 上跑：

```bash
cd /Users/meiziyuan/Roofline-Analysis/experiments/gemm_reference
make ARCH=sm_103a
```

## 单次运行

```bash
cd /Users/meiziyuan/Roofline-Analysis/experiments/gemm_reference
./bench_gemm_reference --m=512 --n=512 --k=512 --warmup=5 --iters=20
```

只测某一个 backend：

```bash
./bench_gemm_reference --m=512 --n=512 --k=512 --backend=naive
./bench_gemm_reference --m=512 --n=512 --k=512 --backend=simt
./bench_gemm_reference --m=512 --n=512 --k=512 --backend=cublas
```

关掉 correctness：

```bash
./bench_gemm_reference --m=512 --n=512 --k=512 --check=0
```

`simt` backend 现在要求：

```bash
--block-m=16 --block-n=16
```

## 步骤 2 运行

编译：

```bash
cd /Users/meiziyuan/Roofline-Analysis/experiments/gemm_reference
make bench_step2_tcgen05_mma ARCH=sm_100a
```

运行：

```bash
cd /Users/meiziyuan/Roofline-Analysis/experiments/gemm_reference
./bench_step2_tcgen05_mma --m=128 --n=256 --k=64 --warmup=5 --iters=20
```

当前步骤 2 约束：

```bash
m % 128 == 0
n % 256 == 0
k % 64 == 0
```

## 批量跑

```bash
cd /Users/meiziyuan/Roofline-Analysis/experiments/gemm_reference
bash run.sh
```

默认会跑：

```text
128x128x128
256x256x256
512x512x512
```

如果你想自定义 shape：

```bash
cd /Users/meiziyuan/Roofline-Analysis/experiments/gemm_reference
ARCH=sm_103a \
SHAPES=128x128x128,512x512x512,1024x1024x1024 \
WARMUP=10 \
ITERS=50 \
bash run.sh
```

## 输出格式

`run.sh` 会在 `results/<timestamp>/` 下生成：

- `run.log`
- `metadata.json`
- `benchmark.csv`
- `correctness.csv`
- `summary.json`
- 每个 shape 一份 `shape_<M>x<N>x<K>.txt`

主程序 stdout 里的关键行是：

- `CHECK ...`
- `RESULT ...`

runner 就是靠这两种行来抽取 CSV/JSON 的，所以后面你要加新 backend 的话，尽量继续沿用这个格式。

## 备注

- 现在先固定是 `float` / `sgemm`，目的是先把设施搭稳。
- `beta != 0` 也支持，benchmark 时会在计时外恢复初始 `C`。
- `naive` 很慢是预期行为。
- `simt` 是步骤 1 的结构化版本，重点是 `cp.async + shared memory + double buffer` 路线先立住。
