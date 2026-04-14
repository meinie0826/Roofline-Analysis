#!/usr/bin/env bash
# =============================================================================
# FA4 Ablation Experiment (pip-installed FA4)
# =============================================================================
# Run this if check_fa4_ablation_support.py reports full support.
#
# Usage:
#   # 1. First check if ablation is supported
#   python fa4/check_fa4_ablation_support.py
#
#   # 2. If supported, run this script
#   bash fa4/run_experiment_with_pip.sh
#
# This script:
#   - Runs ablation benchmarks
#   - Saves CSV + plots to fa4/results/
#   - Commits and pushes to git automatically
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$REPO_ROOT/fa4/results"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

# ── Check ablation support ─────────────────────────────────────────────────────
echo "Checking FA4 ablation support..."
if ! python3 "$SCRIPT_DIR/check_fa4_ablation_support.py"; then
    echo ""
    echo "ERROR: Installed FA4 does not support ablation interfaces."
    echo "You need to clone FA4 source. See instructions above."
    exit 1
fi

# ── Setup ──────────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR"

GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
LOG_FILE="$OUT_DIR/run_${TIMESTAMP}.log"

echo "" | tee "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo " FA4 Ablation Experiment (pip-installed FA4)" | tee -a "$LOG_FILE"
echo " Timestamp : $TIMESTAMP" | tee -a "$LOG_FILE"
echo " GPU       : $GPU_INFO" | tee -a "$LOG_FILE"
echo " Output    : $OUT_DIR" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# ── Run benchmark ──────────────────────────────────────────────────────────────
BENCH="$SCRIPT_DIR/benchmark_ablation_sm100.py"

echo "" | tee -a "$LOG_FILE"
echo "Running full ablation (causal + non-causal)..." | tee -a "$LOG_FILE"

python3 "$BENCH" \
    --seqlen 512,1024,2048,4096,8192 \
    --hdim 128 \
    --warmup 10 \
    --rep 50 \
    --csv "$OUT_DIR/ablation_full_D128_${TIMESTAMP}.csv" \
    --plot \
    --plot-dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# ── Git commit + push ─────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[push] Committing results..." | tee -a "$LOG_FILE"
cd "$REPO_ROOT"

git add fa4/results/
git commit -m "fa4: add ablation experiment results $TIMESTAMP

GPU: $GPU_INFO
Run: pip-installed FA4" \
    || echo "  (nothing new to commit)"

git push origin HEAD 2>&1 | tee -a "$LOG_FILE"
echo "[push] Done." | tee -a "$LOG_FILE"
