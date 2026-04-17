#!/bin/bash
# Run Extended GEMM Roofline Analysis with LLM-Realistic Shapes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "LLM GEMM Roofline Analysis"
echo "=========================================="

# Check dependencies
python3 -c "import torch; import matplotlib" 2>/dev/null || {
    echo "Error: Missing dependencies"
    exit 1
}

echo ""
echo "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
echo ""

# Run benchmarks
echo "Running LLM shape benchmarks..."
python3 "${SCRIPT_DIR}/benchmark_llm_shapes.py" \
    --mode llm \
    --warmup 5 \
    --iterations 20 \
    --output-dir "${RESULTS_DIR}"

# Find latest results
RESULTS_FILE=$(ls -t "${RESULTS_DIR}"/llm_shapes_benchmark_*.json 2>/dev/null | head -1)

if [ -z "${RESULTS_FILE}" ]; then
    echo "Error: No results file found"
    exit 1
fi

echo ""
echo "Generating roofline plot..."
python3 "${SCRIPT_DIR}/plot_b300_roofline.py" \
    --results "${RESULTS_FILE}" \
    --output "${RESULTS_DIR}/llm_roofline.png"

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo "Results: ${RESULTS_DIR}"
ls -lh "${RESULTS_DIR}"/*.png 2>/dev/null || true
