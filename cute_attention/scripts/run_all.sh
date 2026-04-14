#!/usr/bin/env bash
# =============================================================================
# Build and Run CuTe FlashAttention
# =============================================================================
# 需要先安装 CUTLASS:
#   git clone https://github.com/NVIDIA/cutlass.git
#   cd cutlass && mkdir build && cd build
#   cmake .. -DCUTLASS_ENABLE_TESTS=OFF
#   make -j install
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
OUT_DIR="$PROJECT_ROOT/results"

echo "================================================================"
echo " CuTe FlashAttention Build & Run"
echo "================================================================"
echo " Project : $PROJECT_ROOT"
echo " Build   : $BUILD_DIR"
echo " Output  : $OUT_DIR"
echo "================================================================"

# 检查 CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)
echo "CUDA version: $CUDA_VERSION"

# 设置环境变量
export CUTLASS_PATHS="${CUTLASS_PATHS:-/usr/local/cuda/include}"
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90;100}"

# 编译
echo ""
echo "[1/3] Building..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_CUDA_ARCHITECTURES" \
    -DCUTLASS_INCLUDE_DIRS="$CUTLASS_PATHS"

make -j$(nproc)

# 检查编译结果
if [[ ! -f "$BUILD_DIR/libcute_attention.so" ]]; then
    echo "ERROR: Library not found after build"
    exit 1
fi

echo "✓ Build complete"

# 运行 benchmark
echo ""
echo "[2/3] Running benchmark..."
cd "$PROJECT_ROOT/python"

mkdir -p "$OUT_DIR"

python3 benchmark.py \
    --seqlen 512,1024,2048,4096,8192 \
    --output "$OUT_DIR/benchmark_$(date +%Y%m%dT%H%M%SZ).json"

echo ""
echo "[3/3] Saving GPU info..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT_DIR/gpu_info.txt"
cat "$OUT_DIR/gpu_info.txt"

echo ""
echo "================================================================"
echo " DONE!"
echo "================================================================"
echo " Results: $OUT_DIR"
echo "================================================================"
