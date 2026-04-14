#!/usr/bin/env bash
# =============================================================================
# FA4 Ablation Experiment Runner (FA4 source build)
# =============================================================================
# Use this script when you have FA4 built from source.
# 
# IMPORTANT: The installed flash-attn-4 pip package may NOT support ablation
# interfaces. Run this check first:
#   python fa4/check_fa4_ablation_support.py
#
# If NOT supported, you need to:
#   1. Clone FA4: git clone https://github.com/Dao-AILab/flash-attention.git
#   2. Copy this benchmark: cp fa4/benchmark_ablation_sm100.py flash-attention/benchmarks/
#   3. Install from source: cd flash-attention && pip install -e . --no-build-isolation
#   4. Run this script: bash fa4/run_experiment.sh /path/to/flash-attention
#
# All results (CSV, NSight counters, plots, log) are committed and pushed to git.
# Pull from your local machine to see results.
#
# Usage:
#   bash fa4/run_experiment.sh <flash-attention-repo> [output-dir] [git-remote]
#
# Example:
#   bash fa4/run_experiment.sh ~/flash-attention ./fa4/results origin
# =============================================================================

set -euo pipefail

FA4_REPO="${1:-$HOME/flash-attention}"
OUT_DIR="${2:-./fa4/results}"
GIT_REMOTE="${3:-origin}"

BENCH="$FA4_REPO/benchmarks/benchmark_ablation_sm100.py"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "$BENCH" ]]; then
    echo "ERROR: benchmark not found at $BENCH"
    echo "  1. Copy fa4/benchmark_ablation_sm100.py into \$FA4_REPO/benchmarks/"
    echo "  2. Or set \$1 to the correct flash-attention path."
    exit 1
fi

mkdir -p "$OUT_DIR"

# Capture GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
LOG_FILE="$OUT_DIR/run_${TIMESTAMP}.log"

echo "================================================================" | tee "$LOG_FILE"
echo " FA4 Ablation Experiment" | tee -a "$LOG_FILE"
echo " Timestamp : $TIMESTAMP" | tee -a "$LOG_FILE"
echo " GPU       : $GPU_INFO" | tee -a "$LOG_FILE"
echo " FA4 repo  : $FA4_REPO" | tee -a "$LOG_FILE"
echo " Output    : $OUT_DIR" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# ── Step 1: Theory ceilings (no GPU) ──────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[1/4] Roofline theory (hardware specs only)" | tee -a "$LOG_FILE"
python3 "$SCRIPT_DIR/fa4_roofline_theory.py" --no-plots 2>&1 | tee -a "$LOG_FILE"

# ── Step 2: Non-causal ablation ───────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[2/4] Ablation: non-causal (seqlen=512,1024,2048,4096,8192, hdim=128)" | tee -a "$LOG_FILE"
python3 "$BENCH" \
    --non-causal-only \
    --seqlen 512,1024,2048,4096,8192 \
    --hdim 128 \
    --warmup 10 \
    --rep 50 \
    --csv "$OUT_DIR/ablation_noncausal_D128.csv" \
    --plot \
    --plot-dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# ── Step 3: Causal ablation ───────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[3/4] Ablation: causal" | tee -a "$LOG_FILE"
python3 "$BENCH" \
    --causal-only \
    --seqlen 512,1024,2048,4096,8192 \
    --hdim 128 \
    --warmup 10 \
    --rep 50 \
    --csv "$OUT_DIR/ablation_causal_D128.csv" \
    --plot \
    --plot-dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# ── Step 4: NSight hardware counters ──────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[4/4] NSight hardware counter profiling" | tee -a "$LOG_FILE"

NCU_CSV="$OUT_DIR/ncu_counters_${TIMESTAMP}.csv"
NCU_METRICS="sm__sass_inst_executed_op_mufu_ex2.sum,sm__sass_inst_executed_op_fadd.sum,sm__sass_inst_executed_op_ffma.sum,sm__sass_inst_executed_op_fmul.sum,l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_shared_op_st.sum,sm__cycles_elapsed.avg,sm__cycles_active.avg,sm__warps_active.avg.pct_of_peak_sustained_active"

if command -v ncu &>/dev/null; then
    echo "  Running ncu on seqlen=4096 (both causal and non-causal, rep=3)..." | tee -a "$LOG_FILE"
    ncu \
        --metrics "$NCU_METRICS" \
        --csv \
        --log-file "$NCU_CSV" \
        --target-processes all \
        python3 "$BENCH" \
            --seqlen 4096 \
            --no-correctness \
            --rep 3 \
            --warmup 1 \
        2>&1 | tee -a "$LOG_FILE"
    echo "  NSight CSV: $NCU_CSV" | tee -a "$LOG_FILE"
else
    echo "  WARNING: 'ncu' not in PATH — skipping hardware counter profiling." | tee -a "$LOG_FILE"
    echo "  To run manually after the fact:" | tee -a "$LOG_FILE"
    echo "    ncu --metrics '$NCU_METRICS' \\" | tee -a "$LOG_FILE"
    echo "        --csv --log-file $NCU_CSV \\" | tee -a "$LOG_FILE"
    echo "        python3 $BENCH --seqlen 4096 --no-correctness --rep 3" | tee -a "$LOG_FILE"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo " Files written:" | tee -a "$LOG_FILE"
ls -lh "$OUT_DIR/" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# ── Git commit + push ─────────────────────────────────────────────────────────
echo ""
echo "[push] Committing results and pushing to $GIT_REMOTE..."
cd "$REPO_ROOT"

git add fa4/results/
git commit -m "fa4: add experiment results $TIMESTAMP

GPU: $GPU_INFO
Files:
$(ls -1 "$OUT_DIR/")" \
    || echo "  (nothing new to commit)"

git push "$GIT_REMOTE" HEAD 2>&1
echo "[push] Done. Pull on your local machine with: git pull $GIT_REMOTE"
