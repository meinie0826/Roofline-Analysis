# Roofline-Analysis

Per-kernel hardware roofline analysis. Each subdirectory analyzes one kernel family.

```
Roofline-Analysis/
├── fa4/          ← FlashAttention-4 (B200/SM100)
└── ...           ← future kernels (FlashMLA, GEMM, ...)
```

---

## FA4 — FlashAttention-4 on B200

### What this measures

5-stage ablation ladder: each optimization is enabled one at a time.
Measured TFLOPs/s is compared against theoretical hardware ceilings
derived purely from NVIDIA public datasheets.

| Stage | What changes |
|-------|-------------|
| S0 | Baseline: q_stage=1 (serial), MUFU-only exp, no cond-rescale, linear sched |
| S1 | +ping-pong: q_stage=2 (async TMEM MMA↔softmax overlap) |
| S2 | +conditional rescaling: skip O correction when Δmax < 8.0 |
| S3 | +exp2 FMA emulation: polynomial approx on FP32 units alongside MUFU |
| S4 | +LPT scheduler: load-balance causal tiles across SMs |

Theory ceilings (D=128, M=N=128, B200):

```
TC  cycles = 1024c  →  TC ceiling  = 2250 TFLOPs/s  (= B200 peak)
EXP cycles = 1024c  →  EXP ceiling = 2250 TFLOPs/s  ← co-bottleneck with TC!
SMEM cycles = 768c  →  SMEM ceiling = 3000 TFLOPs/s
Serial (TC+EXP)     →  Serial ceil  = 1125 TFLOPs/s  (50% utilization)
```

### How to run (on B200 machine)

**1. Clone and set up:**
```bash
git clone git@github.com:meinie0826/Roofline-Analysis.git
cd Roofline-Analysis

# Install FA4
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install -e . && cd ..

# Copy benchmark into FA4 repo
cp fa4/benchmark_ablation_sm100.py flash-attention/benchmarks/

# Dependencies
pip install matplotlib numpy
```

**2. Run everything (benchmark + NSight + push results):**
```bash
bash fa4/run_experiment.sh ./flash-attention ./fa4/results origin
```

This will:
- Run ablation benchmark (non-causal + causal, seqlen=512..8192, warmup=10, rep=50)
- Run NSight hardware counter profiling (seqlen=4096) if `ncu` is available
- Save all CSVs, plots, and a full stdout log to `fa4/results/`
- `git commit` and `git push` the results automatically

**3. Pull results on your local machine:**
```bash
git pull origin main
```

Then compare measured vs theory:
```bash
python3 fa4/fa4_compare_measured_vs_theory.py \
    --csv fa4/results/ablation_noncausal_D128.csv \
    --out-dir fa4/results

python3 fa4/fa4_compare_measured_vs_theory.py \
    --csv fa4/results/ablation_causal_D128.csv \
    --causal \
    --out-dir fa4/results
```

### Scripts

| File | Description | Needs GPU? |
|------|-------------|------------|
| `fa4/fa4_roofline_theory.py` | Theory ceilings from hardware specs only | No |
| `fa4/benchmark_ablation_sm100.py` | Actual TFLOPs/s measurement | Yes (B200) |
| `fa4/run_experiment.sh` | End-to-end runner + git push | Yes (B200) |
| `fa4/fa4_compare_measured_vs_theory.py` | Plot measured CSV vs theory | No |

### Output files (after running)

```
fa4/results/
├── run_<timestamp>.log              ← full stdout capture
├── ablation_noncausal_D128.csv      ← raw TFLOPs/s per stage × seqlen
├── ablation_causal_D128.csv
├── ncu_counters_<timestamp>.csv     ← NSight hardware counters
└── *.png                            ← roofline ladder plots
```
