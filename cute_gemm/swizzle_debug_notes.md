# CuTeDSL Swizzle Debug Notes

文件位置：

- kernel: `/Users/meiziyuan/Roofline-Analysis/cute_gemm/mma_gemm_1cta_cutedsl.py`
- helper: `/Users/meiziyuan/cutlass/python/CuTeDSL/cutlass/utils/blackwell_helpers.py`

## 复现命令

```bash
cd /workspace/Roofline-Analysis
bash cute_gemm/run.sh --mnk 128,256,64 --debug-swizzle
```

## 一次实际输出

```text
[cute_gemm/run.sh] local CuTeDSL MLIR libs not found at /workspace/Roofline-Analysis/cutlass/python/CuTeDSL/cutlass/_mlir/_mlir_libs
[cute_gemm/run.sh] falling back to installed python package 'cutlass'
=== host swizzle debug begin ===
a_smem_layout       = S<3,4,3> o 0 o ((128,16),1,4,1):((64,1),0,16,0)
a_smem_layout.outer = ((128,16),1,4,1):((64,1),0,16,0)
a_smem_layout.inner = S<3,4,3>
b_smem_layout       = S<3,4,3> o 0 o ((256,16),1,4,1):((64,1),0,16,0)
b_smem_layout.outer = ((256,16),1,4,1):((64,1),0,16,0)
b_smem_layout.inner = S<3,4,3>
=== host swizzle debug end ===
=== A swizzle mapping ===
  i |              coord |    raw |    swz
  0 |  ((0, 0), 0, 0, 0) |      0 |      0
  1 |  ((0, 1), 0, 0, 0) |      1 |      1
  2 |  ((0, 2), 0, 0, 0) |      2 |      2
  3 |  ((0, 3), 0, 0, 0) |      3 |      3
  4 |  ((0, 4), 0, 0, 0) |      4 |      4
  5 |  ((0, 5), 0, 0, 0) |      5 |      5
  6 |  ((0, 6), 0, 0, 0) |      6 |      6
  7 |  ((0, 7), 0, 0, 0) |      7 |      7
  8 |  ((0, 8), 0, 0, 0) |      8 |      8
  9 |  ((0, 9), 0, 0, 0) |      9 |      9
 10 | ((0, 10), 0, 0, 0) |     10 |     10
 11 | ((0, 11), 0, 0, 0) |     11 |     11
 12 | ((0, 12), 0, 0, 0) |     12 |     12
 13 | ((0, 13), 0, 0, 0) |     13 |     13
 14 | ((0, 14), 0, 0, 0) |     14 |     14
 15 | ((0, 15), 0, 0, 0) |     15 |     15
first raw!=swz index: 128
=== A first swizzle differences ===
  i |              coord |    raw |    swz
128 |  ((2, 0), 0, 0, 0) |    128 |    144
129 |  ((2, 1), 0, 0, 0) |    129 |    145
130 |  ((2, 2), 0, 0, 0) |    130 |    146
131 |  ((2, 3), 0, 0, 0) |    131 |    147
132 |  ((2, 4), 0, 0, 0) |    132 |    148
133 |  ((2, 5), 0, 0, 0) |    133 |    149
134 |  ((2, 6), 0, 0, 0) |    134 |    150
135 |  ((2, 7), 0, 0, 0) |    135 |    151
136 |  ((2, 8), 0, 0, 0) |    136 |    152
137 |  ((2, 9), 0, 0, 0) |    137 |    153
138 | ((2, 10), 0, 0, 0) |    138 |    154
139 | ((2, 11), 0, 0, 0) |    139 |    155
140 | ((2, 12), 0, 0, 0) |    140 |    156
141 | ((2, 13), 0, 0, 0) |    141 |    157
142 | ((2, 14), 0, 0, 0) |    142 |    158
143 | ((2, 15), 0, 0, 0) |    143 |    159

=== B swizzle mapping ===
  i |              coord |    raw |    swz
  0 |  ((0, 0), 0, 0, 0) |      0 |      0
  1 |  ((0, 1), 0, 0, 0) |      1 |      1
  2 |  ((0, 2), 0, 0, 0) |      2 |      2
  3 |  ((0, 3), 0, 0, 0) |      3 |      3
  4 |  ((0, 4), 0, 0, 0) |      4 |      4
  5 |  ((0, 5), 0, 0, 0) |      5 |      5
  6 |  ((0, 6), 0, 0, 0) |      6 |      6
  7 |  ((0, 7), 0, 0, 0) |      7 |      7
  8 |  ((0, 8), 0, 0, 0) |      8 |      8
  9 |  ((0, 9), 0, 0, 0) |      9 |      9
 10 | ((0, 10), 0, 0, 0) |     10 |     10
 11 | ((0, 11), 0, 0, 0) |     11 |     11
 12 | ((0, 12), 0, 0, 0) |     12 |     12
 13 | ((0, 13), 0, 0, 0) |     13 |     13
 14 | ((0, 14), 0, 0, 0) |     14 |     14
 15 | ((0, 15), 0, 0, 0) |     15 |     15
first raw!=swz index: 128
=== B first swizzle differences ===
  i |              coord |    raw |    swz
128 |  ((2, 0), 0, 0, 0) |    128 |    144
129 |  ((2, 1), 0, 0, 0) |    129 |    145
130 |  ((2, 2), 0, 0, 0) |    130 |    146
131 |  ((2, 3), 0, 0, 0) |    131 |    147
132 |  ((2, 4), 0, 0, 0) |    132 |    148
133 |  ((2, 5), 0, 0, 0) |    133 |    149
134 |  ((2, 6), 0, 0, 0) |    134 |    150
135 |  ((2, 7), 0, 0, 0) |    135 |    151
136 |  ((2, 8), 0, 0, 0) |    136 |    152
137 |  ((2, 9), 0, 0, 0) |    137 |    153
138 | ((2, 10), 0, 0, 0) |    138 |    154
139 | ((2, 11), 0, 0, 0) |    139 |    155
140 | ((2, 12), 0, 0, 0) |    140 |    156
141 | ((2, 13), 0, 0, 0) |    141 |    157
142 | ((2, 14), 0, 0, 0) |    142 |    158
143 | ((2, 15), 0, 0, 0) |    143 |    159

=== swizzle debug begin ===
sA.layout       = ((128,16),1,4,1):((64,1),0,16,0)
sA_stage.layout = ((128,16),1,4):((64,1),0,16)
sA.iter.swz     = raw_ptr(0x0000000000000480: f16, smem, align<128>)
sA.iter.raw     = raw_ptr(0x0000000000000480: f16, smem, align<128>)
sB.layout       = ((256,16),1,4,1):((64,1),0,16,0)
sB_stage.layout = ((256,16),1,4):((64,1),0,16)
sB.iter.swz     = raw_ptr(0x0000000000004480: f16, smem, align<128>)
sB.iter.raw     = raw_ptr(0x0000000000004480: f16, smem, align<128>)
=== swizzle debug end ===
PASS {'mnk': (128, 256, 64), 'dtype': 'fp16->fp32'}
```

## 怎么看这几行

### 1. `a_smem_layout = inner o outer`

对 A 来说：

```text
a_smem_layout       = S<3,4,3> o 0 o ((128,16),1,4,1):((64,1),0,16,0)
a_smem_layout.outer = ((128,16),1,4,1):((64,1),0,16,0)
a_smem_layout.inner = S<3,4,3>
```

可以先理解成两步：

1. `outer` 先把逻辑坐标映射成 raw offset
2. `inner = S<3,4,3>` 再把 raw offset 改写成 swizzled offset

这里不用先纠结 `S<3,4,3>` 的符号形式，先盯住它对 offset 做了什么。

### 2. `outer` 怎么算 raw offset

`outer` 的 shape/stride 是：

```text
((128,16),1,4,1):((64,1),0,16,0)
```

可把一个坐标写成：

```text
((m, k_inner), mma, k_block, stage)
```

对应 raw offset 是：

```text
raw = m * 64 + k_inner * 1 + mma * 0 + k_block * 16 + stage * 0
```

当前这个 case 里 `mma=1`、`stage=1`，所以实际起作用的是：

```text
raw = m * 64 + k_inner + k_block * 16
```

例如：

- `((0, 0), 0, 0, 0) -> raw = 0`
- `((0, 15), 0, 0, 0) -> raw = 15`
- `((2, 0), 0, 0, 0) -> raw = 2 * 64 = 128`

这和日志是对上的。

### 3. 为什么前 128 个 `raw == swz`

日志里有：

```text
first raw!=swz index: 128
```

它表示：

- `i = 0..127` 时，swizzle 对这段 raw offset 没改动
- 从 `i = 128` 开始，swizzle 才开始真正改变地址

所以前面看到：

```text
0 -> raw 0 -> swz 0
1 -> raw 1 -> swz 1
...
15 -> raw 15 -> swz 15
```

并不代表没有 swizzle，只代表在这段地址范围里，swizzle 恰好是 identity。

### 4. 第一处变化到底发生了什么

第一条变化是：

```text
128 | ((2, 0), 0, 0, 0) | raw 128 | swz 144
```

接下来一整段：

```text
129 -> 145
130 -> 146
...
143 -> 159
```

也就是：

```text
128..143  ->  144..159
```

目前能直接看出的现象是：

- 这不是“数据先搬家”
- 而是同一个逻辑坐标算出的 raw offset，被 `S<3,4,3>` 改写成了另一个物理 offset
- 在这一段里，效果像是整块平移了 `+16`

### 5. `sA.iter.swz` 和 `sA.iter.raw` 为什么打印一样

日志里：

```text
sA.iter.swz     = raw_ptr(0x0000000000000480: f16, smem, align<128>)
sA.iter.raw     = raw_ptr(0x0000000000000480: f16, smem, align<128>)
```

看起来一样，不代表 swizzle 没挂上。原因是当前 `Pointer.__str__` 只打印：

- 基地址
- dtype
- memspace
- alignment

不会把 swizzle type 也打印出来。

所以：

- ptr 字符串一样
- 但 offset 的解释规则不一样

真正能证明 swizzle 生效的是上面的 `raw -> swz` mapping，而不是 ptr 的字符串。

## 这次日志最值得记住的结论

1. `outer` 决定逻辑坐标如何映射到 raw offset。
2. `inner` 决定 raw offset 如何再变成 swizzled offset。
3. swizzle 改的是“地址解释”，不是 tensor 的逻辑 shape。
4. 当前这个例子里，swizzle 从 raw offset `128` 开始第一次生效。
5. 第一段明显变化是 `128..143 -> 144..159`。

## 后续如果继续学，可以重点看

1. 为什么 `outer` 的 stride 恰好是 `((64,1),0,16,0)`。
2. `partition_shape_A(...)` 和这个 `outer` 的 shape/stride 是怎么对上的。
3. `S<3,4,3>` 到底是按哪几位地址 bit 做变换。
4. 为什么 `gA_tile[i]` 和 `sA_stage[i]` 可以共享同一个逻辑编号 `i`。
