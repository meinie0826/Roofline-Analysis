#!/bin/bash
# Simple GEMM Roofline Benchmark Runner
# Tests cuBLAS and DeepGEMM BF16 implementations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "GEMM Roofline Benchmark"
echo "=========================================="

# Check CUDA
python3 -c "import torch; assert torch.cuda.is_available()" || {
    echo "Error: CUDA not available"
    exit 1
}

echo "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
echo ""

# Check DeepGEMM
python3 -c "import deep_gemm" 2>/dev/null && {
    echo "✓ DeepGEMM is installed"
    python3 -c "import deep_gemm; print(f'  BF16 API available: {hasattr(deep_gemm, \"bf16_gemm_nt\")}')" 
} || {
    echo "✗ DeepGEMM not installed (BF16 comparison will be skipped)"
}

echo ""
echo "Running benchmark..."
python3 "${SCRIPT_DIR}/benchmark_simple.py" \
    --shape-type balanced \
    --warmup 5 \
    --iterations 20 \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Complete!"
echo "Results: ${RESULTS_DIR}"
echo ""
