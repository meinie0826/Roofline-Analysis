# Blackwell WS vs SW Pipeline Experiments

This directory keeps local, commit-friendly copies of the CUTLASS CuTeDSL
GEMM examples used to study warp specialization (WS) versus software pipeline
(SW pipeline) behavior. Do not edit the copies under `3rd/cutlass` for this
experiment; make changes here instead.

## Baselines

- `baselines/tutorial_fp16_gemm_1_sw_pipeline.py`
  - Copied from `3rd/cutlass/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_1.py`.
  - Tutorial SW pipeline baseline.

- `baselines/tutorial_fp16_gemm_2_warp_specialized.py`
  - Copied from `3rd/cutlass/examples/python/CuTeDSL/blackwell/tutorial_gemm/fp16_gemm_2.py`.
  - Tutorial warp-specialized baseline with dedicated TMA, MMA, and epilogue warps.

- `baselines/dense_gemm.py`
  - Copied from `3rd/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm.py`.
  - More configurable dense GEMM example for follow-up controlled experiments.

- `baselines/dense_gemm_software_pipeline.py`
  - Copied from `3rd/cutlass/examples/python/CuTeDSL/blackwell/dense_gemm_software_pipeline.py`.
  - More configurable software-pipeline dense GEMM example.

## Initial Commands

Run from the repository root:

```bash
python experiments/blackwell_ws_sw_pipeline/baselines/tutorial_fp16_gemm_1_sw_pipeline.py \
  --mnk 8192,8192,8192

python experiments/blackwell_ws_sw_pipeline/baselines/tutorial_fp16_gemm_2_warp_specialized.py \
  --mnk 8192,8192,8192
```

For profiler-driven analysis, start with K-sweeps and keep M/N fixed:

```bash
ncu --target-processes all \
  --section SpeedOfLight \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  python experiments/blackwell_ws_sw_pipeline/baselines/tutorial_fp16_gemm_1_sw_pipeline.py \
    --mnk 8192,8192,512
```

Save generated logs, CSV files, and profiler exports under `results/`.

## Timing Sweep

The easiest timing experiment is the local K-sweep runner:

```bash
python experiments/blackwell_ws_sw_pipeline/run_k_sweep.py \
  --m 8192 \
  --n 8192 \
  --k_values 64,128,256,512,1024,2048,4096,8192 \
  --variants sw,ws \
  --warmup_iterations 10 \
  --iterations 100 \
  --repeats 3 \
  --skip_ref_check
```

The runner launches each variant as a separate subprocess, records the
`RESULT,...` line printed by the benchmark, and writes:

- `results/k_sweep_<timestamp>.csv`
- `results/logs_<timestamp>/*.log`

Use smaller settings for a smoke test:

```bash
python experiments/blackwell_ws_sw_pipeline/run_k_sweep.py \
  --m 1024 \
  --n 1024 \
  --k_values 64,128 \
  --iterations 5 \
  --repeats 1 \
  --skip_ref_check
```

Both local tutorial baselines also support direct benchmarking:

```bash
python experiments/blackwell_ws_sw_pipeline/baselines/tutorial_fp16_gemm_1_sw_pipeline.py \
  --mnk 8192,8192,512 \
  --do_benchmark \
  --warmup_iterations 10 \
  --iterations 100 \
  --skip_ref_check

python experiments/blackwell_ws_sw_pipeline/baselines/tutorial_fp16_gemm_2_warp_specialized.py \
  --mnk 8192,8192,512 \
  --do_benchmark \
  --warmup_iterations 10 \
  --iterations 100 \
  --skip_ref_check
```

## Analysis Plan

Use K-sweeps to fit:

```text
T_sw(Ktiles) = A_sw + B_sw * Ktiles
T_ws(Ktiles) = A_ws + B_ws * Ktiles
```

`A` captures prologue, epilogue, launch, and tail effects. `B` captures the
steady-state mainloop cost per K tile. The useful decision boundary is whether
WS improves `B`, only improves `A`, or loses occupancy enough to increase both.

Recommended workflow:

1. Run one smoke test to confirm the copied baselines build and launch.
2. Run the full K-sweep with `--repeats 3`.
3. Fit `avg_ms` versus `K / 64` separately for `sw_pipeline` and
   `warp_specialized`.
4. Compare the fitted intercepts (`A`) and slopes (`B`).
5. Profile representative points with NCU:
   - small K where `A` dominates, for example `K=64` or `K=128`
   - large K where `B` dominates, for example `K=4096` or `K=8192`

Look at scheduler stalls, tensor pipe utilization, barrier stalls, and memory
latency metrics to decide whether the timing difference is from pipeline state
transitions, TMA latency hiding, occupancy, or epilogue behavior.
