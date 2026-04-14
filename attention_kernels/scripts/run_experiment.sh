#!/usr/bin/env bash
# =============================================================================
# FlashAttention Kernel Experiment Runner
# =============================================================================
# Runs all benchmarks and pushes results to git.
#
# Usage:
#   bash attention_kernels/scripts/run_experiment.sh
#
# This script:
#   1. Builds all kernel stages
#   2. Runs benchmarks for various configurations
#   3. Generates plots and CSV data
#   4. Commits and pushes results to git
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTN_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ATTN_ROOT/build"
OUT_DIR="$ATTN_ROOT/results"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

echo "================================================================"
echo " FlashAttention Kernel Experiment"
echo "================================================================"
echo " Timestamp: $TIMESTAMP"
echo " Repo root: $REPO_ROOT"
echo " Build dir: $BUILD_DIR"
echo " Output:    $OUT_DIR"
echo "================================================================"

# ── Step 1: Build kernels ─────────────────────────────────────────────────────
echo ""
echo "[1/4] Building CUDA kernels..."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [[ ! -f Makefile ]]; then
    cmake "$ATTN_ROOT"
fi

make -j$(nproc)
echo "  ✓ Build complete"

# Verify library exists
if [[ ! -f "$BUILD_DIR/libattention_kernels.so" ]]; then
    echo "ERROR: Shared library not found"
    exit 1
fi

# ── Step 2: Setup output directory ─────────────────────────────────────────────
echo ""
echo "[2/4] Setting up output directory..."

mkdir -p "$OUT_DIR"

# Save GPU info
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "$GPU_INFO" > "$OUT_DIR/gpu_info.txt"
echo "  GPU: $GPU_INFO"

# ── Step 3: Run benchmarks ─────────────────────────────────────────────────────
echo ""
echo "[3/4] Running benchmarks..."

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}"

# Run with plots (causal only)
python3 attention_kernels/python/benchmark.py \
    --seqlen 512,1024,2048,4096,8192 \
    --hdim 128 \
    --causal-only \
    --csv "$OUT_DIR/benchmark_${TIMESTAMP}.csv" \
    --plot \
    --plot-dir "$OUT_DIR" \
    --warmup 10 \
    --rep 50

echo "  ✓ Benchmarks complete"

# ── Step 4: Git commit + push ───────────────────────────────────────────────────
echo ""
echo "[4/4] Pushing results to git..."

cd "$REPO_ROOT"

# Add results
git add attention_kernels/results/ 2>/dev/null || true
git add attention_kernels/build/libattention_kernels.so 2>/dev/null || true

# Commit
git commit -m "attention_kernels: add benchmark results $TIMESTAMP

GPU: $GPU_INFO
Files:
$(ls -1 "$OUT_DIR" 2>/dev/null | head -10)" \
    || echo "  (nothing new to commit)"

# Push
git push origin HEAD

echo ""
echo "================================================================"
echo " DONE!"
echo "================================================================"
echo " Results saved to: $OUT_DIR"
echo " Pull locally with: git pull origin"
echo "================================================================"
