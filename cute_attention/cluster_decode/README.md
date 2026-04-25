# Cluster Decode Experiments

This directory isolates the ClusterFusion-style decode attention line from the
main `python_dsl` causal-attention stages.

Current scope:

- MHA only: matching Q/K/V heads.
- Decode only: `q_len=1`.
- `head_dim=128`.
- Dense non-paged KV first.
- Cluster sizes 2 and 4.

The first implementation milestones are intentionally small:

1. `cluster_decode.py`: single-CTA decode correctness baseline.
2. `cluster_decode_split.py`: cluster launch plus split-KV ownership skeleton.
3. `cluster_decode_reduce.py`: fine-grained leader-only reduce contract for
   `(m_i, l_i, O_i[head_dim])`.

## Correctness

CPU-only reduce-contract check:

```bash
PYTHONPATH=/Users/meiziyuan/Roofline-Analysis/cute_attention \
python3 -m cluster_decode.verify_correctness \
  --stage reduce --cluster-size 2 --seq-len 129 --head-dim 128
```

CUDA + CuTeDSL kernel checks:

```bash
PYTHONPATH=/Users/meiziyuan/Roofline-Analysis/cute_attention \
python3 -m cluster_decode.verify_correctness \
  --stage all \
  --batch 1 --heads 2 --seq-len 129 --head-dim 128 \
  --cluster-size 2 --num-threads 128 --dtype float16
```

Pytest subset:

```bash
PYTHONPATH=/Users/meiziyuan/Roofline-Analysis/cute_attention \
python3 -m pytest -q /Users/meiziyuan/Roofline-Analysis/cute_attention/cluster_decode/tests
```

Every DSM / cluster-reduce iteration should pass the reduce-contract check
first, then the CUDA kernel checks on a B200/CuTeDSL environment.
