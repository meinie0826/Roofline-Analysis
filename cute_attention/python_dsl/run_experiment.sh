#!/bin/bash
# =============================================================================
# FlashAttention-4 Benchmark Runner
# =============================================================================

set -e
cd "$(dirname "$0")/.."
git pull

echo "==============================================="
echo " FlashAttention-4 Optimization Analysis"
echo "==============================================="

# Run analysis
python3 cute_attention/python_dsl/fa4_optimization_analysis.py

echo ""
echo "==============================================="
echo " Run Benchmark (SDPA)"
echo "==============================================="

# Run benchmark
for seqlen in 1024 2048 4096 8192 16384; do
    python3 cute_attention/python_dsl/fa4_optimization_analysis.py --benchmark $seqlen
done
