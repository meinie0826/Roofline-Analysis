#!/bin/bash
# Complete Roofline Analysis Script
# Runs both PyTorch and Triton (if available) benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}   GEMM Roofline Analysis Suite${NC}"
echo -e "${YELLOW}============================================${NC}"

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

python3 -c "import torch" 2>/dev/null && \
    echo -e "  ${GREEN}✓${NC} PyTorch" || \
    { echo -e "  ${RED}✗${NC} PyTorch (required)"; exit 1; }

python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && \
    echo -e "  ${GREEN}✓${NC} CUDA available" || \
    { echo -e "  ${RED}✗${NC} CUDA not available (required)"; exit 1; }

python3 -c "import matplotlib" 2>/dev/null && \
    echo -e "  ${GREEN}✓${NC} matplotlib" || \
    { echo -e "  ${YELLOW}?${NC} matplotlib (optional, for plotting)"

python3 -c "import triton" 2>/dev/null && \
    echo -e "  ${GREEN}✓${NC} Triton" || \
    echo -e "  ${YELLOW}?${NC} Triton (optional)"

# GPU info
echo -e "\n${YELLOW}GPU Information:${NC}"
python3 -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0)}')"
python3 -c "import torch; print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Run benchmarks
echo -e "\n${YELLOW}============================================${NC}"
echo -e "${YELLOW}   Running Benchmarks${NC}"
echo -e "${YELLOW}============================================${NC}"

# Default B300 parameters
export PEAK_TFLOPS=${PEAK_TFLOPS:-2500}
export PEAK_BANDWIDTH=${PEAK_BANDWIDTH:-8000}

# Run balanced shapes first
echo -e "\n${YELLOW}[1/4] Balanced shapes (M=N=K)${NC}"
SHAPE_TYPE=balanced python3 "${SCRIPT_DIR}/benchmark_roofline.py" \
    --output-dir "${RESULTS_DIR}" \
    --warmup 5 \
    --iterations 20

# Run memory-heavy shapes
echo -e "\n${YELLOW}[2/4] Memory-heavy shapes (small K)${NC}"
SHAPE_TYPE=memory_heavy python3 "${SCRIPT_DIR}/benchmark_roofline.py" \
    --output-dir "${RESULTS_DIR}" \
    --warmup 3 \
    --iterations 10

# Run compute-heavy shapes
echo -e "\n${YELLOW}[3/4] Compute-heavy shapes (large K)${NC}"
SHAPE_TYPE=compute_heavy python3 "${SCRIPT_DIR}/benchmark_roofline.py" \
    --output-dir "${RESULTS_DIR}" \
    --warmup 3 \
    --iterations 10

# Run layer-like shapes (if applicable)
echo -e "\n${YELLOW}[4/4] Layer-like shapes (transformer layers)${NC}"
SHAPE_TYPE=layer_like python3 "${SCRIPT_DIR}/benchmark_roofline.py" \
    --output-dir "${RESULTS_DIR}" \
    --warmup 3 \
    --iterations 10

# Generate combined plot
echo -e "\n${YELLOW}============================================${NC}"
echo -e "${YELLOW}   Generating Plots${NC}"
echo -e "${YELLOW}============================================${NC}"

# Find all result files
RESULT_FILES=$(ls "${RESULTS_DIR}"/roofline_results_*.json 2>/dev/null | sort)

if [ -z "${RESULT_FILES}" ]; then
    echo -e "${RED}Error: No result files found${NC}"
    exit 1
fi

# Combine all results
python3 << EOF
import json
import sys
from pathlib import Path

results_dir = Path("${RESULTS_DIR}")
all_results = []
gpu_specs = None

for f in sorted(results_dir.glob("roofline_results_*.json")):
    with open(f) as fp:
        data = json.load(fp)
        all_results.extend(data["results"])
        if gpu_specs is None:
            gpu_specs = data.get("gpu_specs", {})

# Write combined results
combined_file = results_dir / "combined_results.json"
with open(combined_file, "w") as fp:
    json.dump({
        "gpu_specs": gpu_specs,
        "results": all_results
    }, fp, indent=2)

print(f"Combined {len(all_results)} results into {combined_file}")
EOF

# Generate plot
COMBINED_FILE="${RESULTS_DIR}/combined_results.json"
if [ -f "${COMBINED_FILE}" ]; then
    echo -e "\n${YELLOW}Generating roofline plot...${NC}"
    python3 "${SCRIPT_DIR}/plot_roofline.py" "${COMBINED_FILE}" \
        --peak-tflops "${PEAK_TFLOPS}" \
        --peak-bandwidth "${PEAK_BANDWIDTH}" \
        --title "GEMM Roofline Analysis (All Shapes)" \
        --output "${RESULTS_DIR}/roofline_complete.png"
fi

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}   Analysis Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "\nResults saved to: ${RESULTS_DIR}"
echo -e "\nKey files:"
echo -e "  - Combined data: ${RESULTS_DIR}/combined_results.json"
echo -e "  - Roofline plot: ${RESULTS_DIR}/roofline_complete.png"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "  1. Review the roofline plot to see performance characteristics"
echo -e "  2. Identify the ridge point (memory→compute transition)"
echo -e "  3. Analyze specific shapes of interest"
