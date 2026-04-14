# FA4 Roofline Analysis

Performance analysis of FlashAttention-4 (FA4) on NVIDIA Blackwell B200 (SM100).
Decomposes each optimization's contribution by mapping it to a specific hardware bottleneck,
and visualizes how each technique raises a particular roofline ceiling.

## Key Insight

B200 doubled Tensor Core throughput (H100: 989T → B200: 2250T BF16) but left
MUFU (exp2) and SMEM bandwidth **unchanged**. This asymmetry creates new co-bottlenecks
that FA4 is specifically designed to address.

```
Per-tile cycle counts on B200 (M=N=D=128):
  TC  (BF16 MMA):      1024 cycles   ← halved from H100's 2048
  EXP (MUFU.EX2):      1024 cycles   ← UNCHANGED → now co-bottleneck!
  SMEM bandwidth:       768 cycles   ← UNCHANGED
```

## Optimization → Hardware Mapping

| Stage | Optimization | Hardware Target | Ceiling Effect |
|-------|-------------|-----------------|----------------|
| S0 | Baseline (serial) | Pipeline stall | 1125 TFLOPs/s (50% util) |
| S1 | +ping-pong (q_stage=2) | TC idle time | **2250T (100%) ← 2× gain** |
| S2 | +conditional rescaling | Correction warpgroup | +5-10% |
| S3 | +FMA exp2 emulation | MUFU throughput | EXP ceil: 2250→4091T |
| S4 | +LPT scheduler | SM utilization (causal) | +4-14% causal |
| S5 | +2-CTA MMA | SMEM bandwidth | SMEM ceil: 3000→4500T |

## Charts

| File | Description |
|------|-------------|
| `plots/fa4_hopper_vs_blackwell_asymmetry.png` | Hardware asymmetry: why B200 needs new opts |
| `plots/fa4_roofline_ladder_b200_D128_causal.png` | Roofline ladder (causal, hdim=128) |
| `plots/fa4_roofline_ladder_b200_D128_noncausal.png` | Roofline ladder (non-causal, +2-CTA) |
| `plots/fa4_per_resource_ceiling_b200_D128.png` | Per-resource ceiling progression |
| `plots/fa4_hdim_bottleneck_b200.png` | Bottleneck heatmap by head dimension |
| `plots/FA4_Roofline_Analysis_Report.md` | Full written analysis report |

## Usage

### Generate all charts (no GPU required)
```bash
pip install matplotlib numpy
python fa4_roofline_analysis.py --out-dir ./plots --report
```

### Run actual ablation benchmark (requires B200/SM100)
```bash
# Theory only (no GPU)
python benchmark_ablation_sm100.py --roofline-only

# Full ablation
python benchmark_ablation_sm100.py --seqlen 1024,2048,4096,8192 --causal-only --csv results.csv --plot

# NSight profiling
ncu --metrics sm__sass_inst_executed_op_mufu_ex2.sum,\
        sm__sass_inst_executed_op_ffma.sum,\
        l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum,\
        sm__cycles_elapsed.avg \
    python benchmark_ablation_sm100.py --seqlen 4096 --no-correctness --rep 3
```

## Requirements

- Python 3.10+
- `matplotlib`, `numpy` (for charts, no GPU needed)
- For `benchmark_ablation_sm100.py`: FA4 repository (`flash-attention`), CUDA, B200 GPU

## Source

Analysis based on FA4 paper and CuTe-DSL implementation:
- https://github.com/Dao-AILab/flash-attention (flash_attn/cute/flash_fwd_sm100.py)
- https://tridao.me/blog/2025/flash4/
