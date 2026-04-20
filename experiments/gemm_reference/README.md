# GEMM Reference Harness

这个目录是一个独立的“步骤 0”测试设施，先把最基础的 reference 和 benchmark 骨架搭好：

- 一个最朴素的 row-major CUDA naive GEMM
- 一个 cuBLAS `sgemm` reference
- 正确性校验
- benchmark loop
- 批量 shape 运行和结果落盘

后面如果你要继续塞 CUTLASS kernel、自写 kernel 或别的调度，只需要沿着这里的输入初始化、校验和结果输出往里接就行。

## 文件

```text
gemm_reference/
├── bench_gemm_reference.cu   # naive + cuBLAS + correctness + benchmark
├── Makefile                  # 单目标编译
├── run.sh                    # 批量 shape 跑 benchmark 并落盘
├── README.md                 # 说明
└── results/                  # 输出目录
```

## 主程序行为

`bench_gemm_reference.cu` 里现在有两个 backend：

- `naive`
  - 每个线程算一个 `C[row, col]`
  - 完全暴力三重循环，没有 tiling、shared memory、vectorize
- `cublas`
  - 用 `cublasSgemm`
  - 通过 row-major / column-major 转置关系直接复用同一份 row-major 输入

正确性默认拿 `cuBLAS` 输出当 reference，和 `naive` 做逐元素比较，输出：

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
./bench_gemm_reference --m=512 --n=512 --k=512 --backend=cublas
```

关掉 correctness：

```bash
./bench_gemm_reference --m=512 --n=512 --k=512 --check=0
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
- 这个目录更偏“测试骨架”，不是性能实现；`naive` 很慢是预期行为。
