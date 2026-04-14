#!/bin/bash
# Run on B200 GPU server
# Requirements: pip install torch

cd /workspace/Roofline-Analysis
git pull

echo "==============================================="
echo " Testing Naive Attention Kernel"
echo "==============================================="

python3 cute_attention/python_dsl/naive_attention.py
