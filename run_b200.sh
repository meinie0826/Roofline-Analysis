#!/bin/bash
# Run on B200 GPU server

cd /workspace/Roofline-Analysis
git pull

echo "==============================================="
echo " FlashAttention - Stage 0 (Naive)"
echo "==============================================="

python3 cute_attention/python_dsl/benchmark.py --test --bench
