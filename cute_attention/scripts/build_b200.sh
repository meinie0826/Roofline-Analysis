#!/usr/bin/env bash
# =============================================================================
# 在 B200 上快速构建 CuTe FlashAttention（不需要编译 CUTLASS）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUT_DIR="$PROJECT_ROOT/results"

echo "================================================================"
echo " CuTe FlashAttention Quick Build (B200)"
echo "================================================================"

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found"
    exit 1
fi

# 检查 CUTLASS 源码（只需要 headers，不需要编译）
if [[ ! -d "$HOME/cutlass/include" ]]; then
    echo ""
    echo "Downloading CUTLASS (header-only, no build needed)..."
    cd "$HOME"
    git clone https://github.com/NVIDIA/cutlass.git --depth 1 --branch v3.5.0
fi

# 设置 CUTLASS 路径
export CUTLASS_PATH="$HOME/cutlass"

echo "CUTLASS path: $CUTLASS_PATH"

# 编译
echo ""
echo "[1/3] Building CuTe Attention..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DCUTLASS_INCLUDE_DIRS="$CUTLASS_PATH/include"

make -j$(nproc)

if [[ ! -f "$BUILD_DIR/libcute_attention.so" ]]; then
    echo "ERROR: Library not found after build"
    exit 1
fi

echo "✓ Build complete"

# 运行 benchmark
echo ""
echo "[2/3] Running benchmark..."
mkdir -p "$OUT_DIR"

cd "$PROJECT_ROOT/python"
python3 benchmark.py \
    --seqlen 512,1024,2048,4096 \
    --output "$OUT_DIR/benchmark_$(date +%Y%m%dT%H%M%SZ).json"

echo ""
echo "[3/3] Saving GPU info..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT_DIR/gpu_info.txt"

echo ""
echo "================================================================"
echo " DONE!"
echo "================================================================"
