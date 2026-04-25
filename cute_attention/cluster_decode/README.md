# Cluster Decode – ClusterFusion Megakernel (CuTeDSL / Blackwell)

Implements the ClusterFusion-style decoder layer in **CuTeDSL** targeting Blackwell (SM100 / SM103).

Mirrors the design of `3rd/clusterfusion/include/5090/llama/kernel.cuh` but expressed entirely
in Python/CuTeDSL instead of raw CUDA C++.

---

## Architecture

### Current implementation status

`cluster_megakernel.py` currently matches ClusterFusion's launch topology
(`cluster_size` CTAs per attention head). Stage 0 RMSNorm already uses the
ClusterFusion ownership model: each CTA scans its `DIM_PER_BLOCK` hidden slice
and DSM-reduces the partial L2 sums. Later stages are still correctness
baselines: each CTA computes the full per-head QKV/attention/W_o path and only
CTA rank 0 writes K/V and W_o partials.

`cluster_primitives.py` contains the CuTeDSL DSM helper sketches. The
megakernel currently uses the scalar sum helper for RMSNorm; vector reductions
and scalar max/sum for attention are not wired in yet.

### Target ClusterFusion ownership model

One **cluster** = one attention head.  Each cluster has `cluster_size` CTAs (2 or 4).

```
Cluster (cluster_size CTAs)
  ├── CTA 0  owns hidden[0 : D/C]   and KV cache[0 : S/C]
  ├── CTA 1  owns hidden[D/C : 2D/C] and KV cache[S/C : 2S/C]
  └── ...
```

### Pipeline stages per cluster

| Stage | Description | Cross-CTA comm |
|-------|-------------|---------------|
| 0 RMSNorm | Partial ‖x‖² over each CTA's hidden slice | `cluster_reduce_scalar_sum` |
| 1 W_qkv GEMM | Partial Q/K/V dot-products | `cluster_reduce_vector_add` (LINEAR) |
| 2 RoPE | In-register rotation, no comm | – |
| 3 Flash-decode | Online softmax over each CTA's KV slice | – |
| 4 Softmax reduce | Global max → global sum | `cluster_reduce_scalar_max`, `cluster_reduce_scalar_sum`, `cluster_reduce_vector_add` (ATTN) |
| 5 W_o GEMM | Each CTA writes its output slice | `atomic_add` to global output |

---

## File layout

```
cluster_decode/
├── common.py               ClusterDecodeConfig + MegakernelConfig
├── cluster_primitives.py   cluster_reduce_* helpers (mapa + atomic_add)
├── cluster_megakernel.py   Full CuTeDSL megakernel
├── megakernel_reference.py Pure-PyTorch reference for correctness verification
├── verify_correctness.py   CLI correctness runner
├── cluster_decode.py       Standalone attn baseline (v0, single CTA)
├── cluster_decode_split.py Standalone attn skeleton (cluster launch, leader fallback)
├── cluster_decode_reduce.py CPU-only split-KV reduce contract check
└── tests/
    ├── test_correctness.py  Original attention-only tests
    └── test_megakernel.py   Megakernel tests (reference + CuTeDSL vs ref)
```

---

## Quickstart

### 1. CPU-only reference sanity check (no GPU needed)

```bash
PYTHONPATH=/Users/meiziyuan/Roofline-Analysis/cute_attention \
python3 -m pytest -q \
  cute_attention/cluster_decode/tests/test_megakernel.py::TestReferenceForward \
  -v
```

### 2. Attention-only reduce contract (CPU)

```bash
PYTHONPATH=/Users/meiziyuan/Roofline-Analysis/cute_attention \
python3 -m cluster_decode.verify_correctness --stage reduce --cluster-size 2
```

### 3. Full megakernel CuTeDSL vs reference (requires Blackwell GPU)

```bash
PYTHONPATH=/Users/meiziyuan/Roofline-Analysis/cute_attention \
python3 -m cluster_decode.verify_correctness \
  --stage megakernel \
  --hidden-dim 256 --num-heads 4 --seq-len 128 \
  --cluster-size 2 --num-threads 128 --dtype float16
```

### 4. Full pytest suite

```bash
PYTHONPATH=/Users/meiziyuan/Roofline-Analysis/cute_attention \
python3 -m pytest -q cute_attention/cluster_decode/tests/ -v
```

---

## Key design decisions vs ClusterFusion CUDA

| ClusterFusion (CUDA) | This implementation (CuTeDSL) |
|---------------------|-------------------------------|
| `cp.async.bulk.shared::cluster` (inline PTX) | `cute.arch.mapa()` + element-wise `atomic_add` |
| `cluster.map_shared_rank()` + `atomicAdd/fmaxf` (scalars) | Same via `mapa` + `atomic_add` / `atomic_max` |
| TMA `cp_async_bulk_tensor_2d_global_to_shared` | Direct element access (TMA to be added) |
| `tcgen05.mma` (Blackwell tensor core) | Scalar loop accumulation (MMA to be added) |

The `mapa`-based element loop is functionally equivalent and will be replaced with
inline-PTX bulk copy in a subsequent pass once correctness is established.

---

## Scope

- MHA only (matching Q/K/V heads).
- Decode only (`q_len = 1`).
- Dense, non-paged KV cache.
- `cluster_size` ∈ {2, 4}.
- Default model dimensions: Llama-2-7B (`hidden_dim=4096, num_heads=32, head_dim=128`).
