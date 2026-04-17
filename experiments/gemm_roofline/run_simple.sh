#!/bin/bash
# Run GEMM Roofline Benchmark
# Compatible with DeepGEMM BF16 API (requires N == K)

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
    echo "✗ DeepGEMM not installed (only cuBLAS will be tested)"
}

echo ""
echo "Running benchmark..."
echo "Note: DeepGEMM BF16 requires N == K constraint"
echo ""

python3 "${SCRIPT_DIR}/benchmark_simple.py" \
    --warmup 5 \
    --iterations 20 \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo "Results: ${RESULTS_DIR}"
echo ""
