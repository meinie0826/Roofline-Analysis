#!/usr/bin/env bash
# =============================================================================
# FA4 Ablation: Complete Setup + Run Script
# =============================================================================
# One-click setup for B200 server:
#   1. Clone FA4 source
#   2. Copy benchmark script
#   3. Build FA4 from source
#   4. Run ablation experiments
#   5. Push results to git
#
# Usage on B200 server:
#   git clone git@github.com:meinie0826/Roofline-Analysis.git
#   cd Roofline-Analysis
#   bash fa4/setup_and_run.sh
#
# Prerequisites:
#   - CUDA 12.x
#   - GCC/G++ compatible with CUDA
#   - git + GitHub SSH key configured
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FA4_DIR="$REPO_ROOT/flash-attention"
OUT_DIR="$REPO_ROOT/fa4/results"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

echo "================================================================"
echo " FA4 Ablation Setup + Run"
echo "================================================================"
echo " Timestamp: $TIMESTAMP"
echo " Repo root: $REPO_ROOT"
echo " FA4 dir  : $FA4_DIR"
echo "================================================================"

# ── Step 1: Clone FA4 source ──────────────────────────────────────────────────
echo ""
echo "[1/5] Checking FA4 source..."

if [[ ! -d "$FA4_DIR" ]]; then
    echo "  Cloning flash-attention repo..."
    git clone https://github.com/Dao-AILab/flash-attention.git "$FA4_DIR"
else
    echo "  ✓ FA4 source already exists at $FA4_DIR"
    cd "$FA4_DIR"
    git pull origin main || echo "  (git pull failed, using existing version)"
fi

# ── Step 2: Copy benchmark script ─────────────────────────────────────────────
echo ""
echo "[2/5] Copying benchmark script..."

mkdir -p "$FA4_DIR/benchmarks"
cp "$SCRIPT_DIR/benchmark_ablation_sm100.py" "$FA4_DIR/benchmarks/"
echo "  ✓ Copied to $FA4_DIR/benchmarks/benchmark_ablation_sm100.py"

# ── Step 3: Build FA4 from source ─────────────────────────────────────────────
echo ""
echo "[3/5] Building FA4 from source..."

cd "$FA4_DIR"

# Check if already installed
if python3 -c "import flash_attn; print(flash_attn.__file__)" 2>/dev/null | grep -q "$FA4_DIR"; then
    echo "  ✓ FA4 already installed from this source"
else
    echo "  Installing FA4 in editable mode..."
    echo "  (This may take 5-10 minutes for compilation)"
    pip install -e . --no-build-isolation
    echo "  ✓ Installation complete"
fi

# Verify installation
echo ""
echo "  Verifying installation..."
python3 -c "from flash_attn.cute.interface import _flash_attn_fwd; print('  ✓ _flash_attn_fwd available')" || {
    echo "  ✗ Installation failed"
    exit 1
}

# ── Step 4: Run experiments ───────────────────────────────────────────────────
echo ""
echo "[4/5] Running ablation experiments..."

mkdir -p "$OUT_DIR"
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
LOG_FILE="$OUT_DIR/run_${TIMESTAMP}.log"

echo "" | tee "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo " FA4 Ablation Experiment" | tee -a "$LOG_FILE"
echo " Timestamp : $TIMESTAMP" | tee -a "$LOG_FILE"
echo " GPU       : $GPU_INFO" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

BENCH="$FA4_DIR/benchmarks/benchmark_ablation_sm100.py"

# Non-causal
echo "" | tee -a "$LOG_FILE"
echo "[4a/5] Non-causal ablation..." | tee -a "$LOG_FILE"
python3 "$BENCH" \
    --non-causal-only \
    --seqlen 512,1024,2048,4096,8192 \
    --hdim 128 \
    --warmup 10 \
    --rep 50 \
    --csv "$OUT_DIR/ablation_noncausal_D128_${TIMESTAMP}.csv" \
    --plot \
    --plot-dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# Causal
echo "" | tee -a "$LOG_FILE"
echo "[4b/5] Causal ablation..." | tee -a "$LOG_FILE"
python3 "$BENCH" \
    --causal-only \
    --seqlen 512,1024,2048,4096,8192 \
    --hdim 128 \
    --warmup 10 \
    --rep 50 \
    --csv "$OUT_DIR/ablation_causal_D128_${TIMESTAMP}.csv" \
    --plot \
    --plot-dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# ── Step 5: Git commit + push ─────────────────────────────────────────────────
echo ""
echo "[5/5] Pushing results to git..."

cd "$REPO_ROOT"

# Add results
git add fa4/results/
git add fa4/flash-attention 2>/dev/null || true  # submodule if needed

# Commit
git commit -m "fa4: add ablation results $TIMESTAMP

GPU: $GPU_INFO
FA4 source: $FA4_DIR

Files:
$(ls -1 "$OUT_DIR" 2>/dev/null | grep -E '\.(csv|png|log)$' | head -20)" \
    || echo "  (nothing new to commit)"

# Push
echo "  Pushing to origin..."
git push origin HEAD 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "================================================================"
echo " DONE!"
echo "================================================================"
echo " Results saved to: $OUT_DIR"
echo " Pull locally with: git pull origin main"
echo "================================================================"
