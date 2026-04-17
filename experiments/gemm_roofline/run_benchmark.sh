#!/bin/bash
# Roofline Analysis Runner Script
# This script runs the GEMM benchmark and generates roofline plots

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Default parameters
PEAK_TFLOPS=${PEAK_TFLOPS:-2500}
PEAK_BANDWIDTH=${PEAK_BANDWIDTH:-8000}
SHAPE_TYPE=${SHAPE_TYPE:-"balanced"}
ITERATIONS=${ITERATIONS:-20}
WARMUP=${WARMUP:-5}

echo "=========================================="
echo "GEMM Roofline Analysis"
echo "=========================================="
echo "Peak TFLOPS: ${PEAK_TFLOPS}"
echo "Peak Bandwidth: ${PEAK_BANDWIDTH} GB/s"
echo "Shape Type: ${SHAPE_TYPE}"
echo "Results Directory: ${RESULTS_DIR}"
echo ""

# Check for Python and required packages
python3 -c "import torch; import matplotlib" 2>/dev/null || {
    echo "Error: Required packages not found"
    echo "Please install: pip install torch matplotlib"
    exit 1
}

# Check for CUDA
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
    echo "Error: CUDA is not available"
    echo "This benchmark requires a CUDA-capable GPU"
    exit 1
}

echo "Running benchmark..."
python3 "${SCRIPT_DIR}/benchmark_roofline.py" \
    --shape-type "${SHAPE_TYPE}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --output-dir "${RESULTS_DIR}" \
    --peak-tflops "${PEAK_TFLOPS}" \
    --peak-bandwidth "${PEAK_BANDWIDTH}"

# Find the latest results file
RESULTS_FILE=$(ls -t "${RESULTS_DIR}"/roofline_results_*.json 2>/dev/null | head -1)

if [ -z "${RESULTS_FILE}" ]; then
    echo "Error: No results file found"
    exit 1
fi

echo ""
echo "Generating plots..."
python3 "${SCRIPT_DIR}/plot_roofline.py" "${RESULTS_FILE}" \
    --peak-tflops "${PEAK_TFLOPS}" \
    --peak-bandwidth "${PEAK_BANDWIDTH}" \
    --title "GEMM Roofline Analysis (${SHAPE_TYPE} shapes)"

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "Results saved to: ${RESULTS_DIR}"
echo "=========================================="
