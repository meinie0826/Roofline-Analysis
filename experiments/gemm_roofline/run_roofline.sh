#!/bin/bash
# Run complete GEMM Roofline Analysis
# 1. Run benchmark
# 2. Generate roofline plots

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "GEMM Roofline Analysis"
echo "=========================================="

# Check dependencies
python3 -c "import torch; import matplotlib" 2>/dev/null || {
    echo "Error: Missing dependencies"
    echo "Install with: pip install torch matplotlib"
    exit 1
}

# Check CUDA
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "Error: CUDA not available"
    exit 1
}

echo ""
echo "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
echo ""

# Step 1: Run benchmark
echo "Step 1: Running GEMM benchmark..."
python3 "${SCRIPT_DIR}/benchmark_simple.py" \
    --shape-type balanced \
    --warmup 5 \
    --iterations 20 \
    --output-dir "${RESULTS_DIR}"

# Find latest results file
RESULTS_FILE=$(ls -t "${RESULTS_DIR}"/gemm_roofline_*.json 2>/dev/null | head -1)

if [ -z "${RESULTS_FILE}" ]; then
    echo "Error: No results file found"
    exit 1
fi

echo ""
echo "Step 2: Generating roofline plots..."
python3 "${SCRIPT_DIR}/plot_roofline.py" "${RESULTS_FILE}" \
    --output "${RESULTS_DIR}/roofline"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}"/*.png 2>/dev/null || true
echo ""
