# Blackwell DSMEM And 2SM Experiments

这个目录用于把问题拆成两层来测：

1. 通用 cluster DSMEM 的读 / 写 / ping-pong 延迟
2. Blackwell 1SM / 2SM GEMM 的端到端行为，以及一个“软件显式 DSMEM 共享 B”代理实现

目标不是一次性证明某条“未公开的内部链路”存在，而是先把可以直接测出来的量分离出来，再做强推断。

## 目录结构

- `common.h`
  通用参数解析、结果打印、CUDA 检查
- `bench_dsmem_read.cu`
  local SMEM vs remote DSMEM 流式读
- `bench_dsmem_write.cu`
  local SMEM vs remote DSMEM 流式写
- `bench_dsmem_pingpong.cu`
  两个 CTA 之间的 remote shared flag 往返延迟
- `bench_software_dsmem_gemm.cu`
  两个 CTA 的 cluster 中，显式让 CTA0 装载 B tile，CTA1 通过 DSMEM 读 B，再用 WMMA 做 1SM 级 tensor-core GEMM 代理
- `bench_cutlass_2sm_gemm.cu`
  用 CUTLASS Blackwell builder 分别实例化 1SM / 2SM kernel，并记录运行时间
- `run.sh`
  统一编译、运行、落盘原始 CSV

## 环境要求

- CUDA 12.8 或更新
- Blackwell GPU，建议 `sm_100a`
- 仓库自带 CUTLASS 子目录可用

## 设计说明

### 1. DSMEM 微基准

`bench_dsmem_read.cu` / `bench_dsmem_write.cu` 都使用：

- `clusterDim.x = 2`
- `blockIdx.x % 2 == 0` 的 CTA 作为 producer / owner
- `blockIdx.x % 2 == 1` 的 CTA 作为 consumer / timed CTA
- 访问模式为 unit-stride、warp-coalesced
- `--align-bytes=32|64|128`
- `--vec-bytes=4|8|16`

`mode=local` 时，timed CTA 访问自己的 shared memory。

`mode=remote` 时，timed CTA 通过 `cluster.map_shared_rank()` 访问另一 CTA 的 shared memory。

### 2. Ping-pong 延迟

`bench_dsmem_pingpong.cu` 用两个 CTA 在 cluster 内互相写 flag，并用 `__threadfence_cluster()` 推送可见性。

这不是“硬件内部链路带宽”，但能给出 DSMEM 控制面往返成本。

### 3. 软件 DSMEM 共享 B 代理

`bench_software_dsmem_gemm.cu` 的实现刻意不追求 CUTLASS 级最优，只做一件事：

- CTA0 装载共享的 B tile
- CTA1 通过 DSMEM 读取这块 B tile
- CTA1 使用 WMMA 做 1SM tensor-core GEMM 代理

这样它更像“显式 DSMEM 搬运 B”的替身实验，而不是峰值内核。

`--stages` 参数当前只控制 staging buffer 深度和 shared memory 占用，不会像 CUTLASS TMA pipeline 那样形成真正的异步流水。这一点要单独注意。

### 4. 硬件 1SM / 2SM 对照

`bench_cutlass_2sm_gemm.cu` 使用 CUTLASS builder 生成：

- `KernelTmaWarpSpecialized1SmSm100 + TmaWarpSpecialized1Sm`
- `KernelTmaWarpSpecialized2SmSm100 + TmaWarpSpecialized2Sm`

并支持：

- `--mode=1sm|2sm`
- `--tile-n=64|128|256`
- `--stages=1|2|4`

这样可以扫：

- `tile_n`
- `stages`
- `mode`

## 典型用法

编译并跑一轮默认 sweep：

```bash
cd experiments/blackwell_dsmem_2sm
bash run.sh
```

单独跑 DSMEM 读带宽：

```bash
./bench_dsmem_read --mode=remote --vec-bytes=16 --align-bytes=128 --buffer-bytes=65536 --iters=2048
```

单独跑软件 DSMEM 共享 B 代理：

```bash
./bench_software_dsmem_gemm --mode=remote --tile-n=128 --k-tiles=64 --stages=4 --repeats=20
```

单独跑 CUTLASS 2SM：

```bash
./bench_cutlass_2sm_gemm --mode=2sm --m=4096 --n=256 --k=4096 --tile-n=256 --stages=4 --repeats=50
```

## 输出

`run.sh` 默认写到：

```text
results/<timestamp>/
```

包含：

- `results_dsmem_read.csv`
- `results_dsmem_write.csv`
- `results_dsmem_pingpong.csv`
- `results_software_gemm.csv`
- `results_cutlass_gemm.csv`
- `metadata.json`
- `run.log`

## 建议先看什么

1. `results_dsmem_read.csv`
   对比 `mode=local` 和 `mode=remote` 的 `bandwidth_gbps`
2. `results_dsmem_pingpong.csv`
   看 `cycles_per_roundtrip`
3. `results_software_gemm.csv`
   看 `mode=local` vs `mode=remote` 的 `avg_ms`
4. `results_cutlass_gemm.csv`
   看 `mode=1sm` vs `mode=2sm` 在不同 `tile_n` / `stages` 下的趋势

## 当前实现边界

- 软件 DSMEM 共享 B 代理不是峰值实现，更适合做“显式 DSMEM 路径”的对照，而不是绝对峰值比较
- `bench_cutlass_2sm_gemm.cu` 依赖 CUDA / CUTLASS 版本与 Blackwell 工具链匹配
- 本机未接 Blackwell 开发环境时，代码主要按仓库已有 CUTLASS / CUDA API 约束编写，建议以服务器上的实际编译结果为准
