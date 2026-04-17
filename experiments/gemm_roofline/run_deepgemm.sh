#!/bin/bash
# Run DeepGEMM Roofline Analysis
# This script tests GEMM performance using cuBLAS (BF16) and DeepGEMM (FP8)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "DeepGEMM Roofline Analysis"
echo "=========================================="

# Check Python
python3 -c "import torch; import matplotlib" 2>/dev/null || {
    echo "Error: Missing dependencies"
    echo "Please install: pip install torch matplotlib"
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

# Check DeepGEMM
echo "Checking DeepGEMM installation..."
python3 -c "import deep_gemm" 2>/dev/null && {
    echo "  ✓ DeepGEMM is installed"
} || {
    echo "  ✗ DeepGEMM not installed"
    echo ""
    echo "To install DeepGEMM:"
    echo "  git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git"
    echo "  cd DeepGEMM && ./develop.sh"
    echo ""
    echo "FP8 benchmarks will be skipped."
}

echo ""
echo "Step 1: Testing DeepGEMM..."
python3 "${SCRIPT_DIR}/test_deepgemm.py"

echo ""
echo "Step 2: Running benchmark..."
python3 "${SCRIPT_DIR}/benchmark_deepgemm.py" \
    --shape-type balanced \
    --warmup 5 \
    --iterations 20 \
    --output-dir "${RESULTS_DIR}"

# Find latest results file
RESULTS_FILE=$(ls -t "${RESULTS_DIR}"/deepgemm_roofline_*.json 2>/dev/null | head -1)

if [ -z "${RESULTS_FILE}" ]; then
    echo "Error: No results file found"
    exit 1
fi

echo ""
echo "Step 3: Analyzing results..."
python3 "${SCRIPT_DIR}/analyze_roofline.py" "${RESULTS_FILE}" \
    --output "${RESULTS_DIR}/roofline_plot"

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results: ${RESULTS_DIR}"
echo ""
