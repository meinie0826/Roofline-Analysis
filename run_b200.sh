#!/bin/bash
# FlashAttention Stage 0 Benchmark Script
# Run on B200 GPU server

set -e

# ============================================================
# Configuration
# ============================================================
WORKSPACE="/workspace/Roofline-Analysis"
REPO="git@github.com:meinie0826/Roofline-Analysis.git"

# ============================================================
# Setup
# ============================================================
echo "==============================================="
echo " FlashAttention Stage 0 Benchmark"
echo "==============================================="
echo ""

# Check if workspace exists, if not clone repo
if [ ! -d "$WORKSPACE" ]; then
    echo "Cloning repository..."
    git clone $REPO $WORKSPACE
fi

cd $WORKSPACE
echo "Pulling latest code..."
git pull

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA required."
    exit 1
fi

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo ""

# ============================================================
# Correctness Tests
# ============================================================
echo "==============================================="
echo " Running Correctness Tests"
echo "==============================================="

python3 cute_attention/python_dsl/benchmark.py --test

echo ""

# ============================================================
# Performance Tests
# ============================================================
echo "==============================================="
echo " Running Performance Benchmark"
echo "==============================================="

python3 cute_attention/python_dsl/benchmark.py --bench --seqlen 128 256 512 1024 2048

echo ""
echo "==============================================="
echo " Benchmark Complete"
echo "==============================================="
