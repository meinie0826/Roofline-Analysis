#!/bin/bash
# Run on B200 GPU server
# Requirements: pip install torch nvidia-cutlass-dsl==4.2.0

cd /workspace/Roofline-Analysis
git pull

echo "==============================================="
echo " FlashAttention-4 Stage Analysis"
echo "==============================================="
python3 cute_attention/python_dsl/flash_attention_stages.py --stages

echo ""
echo "==============================================="
echo " Running Benchmarks"
echo "==============================================="

# Test different stages
for stage in 0 1 8; do
    echo ""
    echo "--- Stage $stage ---"
    python3 cute_attention/python_dsl/flash_attention_stages.py --test $stage
done
