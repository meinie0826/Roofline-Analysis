#!/bin/bash
# =============================================================================
# Unified FlashAttention Stage Benchmark Runner
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "==============================================="
echo " Unified FlashAttention Stage Analysis"
echo "==============================================="

# Run quick comprehensive benchmark
python3 "$SCRIPT_DIR/benchmark_comprehensive.py" --quick --max-stage 3

echo ""
echo "==============================================="
echo " Run Full Benchmark"
echo "==============================================="

# Run full benchmark and roofline pipeline
bash "$SCRIPT_DIR/run_comprehensive.sh"
