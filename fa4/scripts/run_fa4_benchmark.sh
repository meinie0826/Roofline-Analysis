#!/usr/bin/env bash
# =============================================================================
# FlashAttention-4 性能测试 (简化版)
# =============================================================================
# 使用 FA4 源码直接测试性能
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FA4_DIR="$REPO_ROOT/flash-attention/hopper"
OUT_DIR="$REPO_ROOT/fa4/results"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

echo "================================================================"
echo " FlashAttention-4 Performance Benchmark"
echo "================================================================"
echo " Timestamp: $TIMESTAMP"
echo " FA4 dir  : $FA4_DIR"
echo " Output   : $OUT_DIR"
echo "================================================================"

# 检查 FA4 是否存在
if [[ ! -d "$FA4_DIR" ]]; then
    echo "ERROR: FA4 directory not found at $FA4_DIR"
    echo "Please clone flash-attention repo first:"
    echo "  git clone https://github.com/Dao-AILab/flash-attention.git"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_DIR"

# 保存 GPU 信息
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "$GPU_INFO" > "$OUT_DIR/gpu_info.txt"
echo "GPU: $GPU_INFO"

# 检查是否已编译
if [[ ! -f "$FA4_DIR/../flash_attn_3/_C.so" ]]; then
    echo ""
    echo "FA4 not compiled. Compiling..."
    cd "$FA4_DIR/.."
    pip install -e . --no-build-isolation
fi

# 运行 benchmark
cd "$FA4_DIR"

echo ""
echo "Running benchmark..."
python3 benchmark_attn.py \
    --seqlens 512,1024,2048,4096,8192 \
    --causal \
    --output "$OUT_DIR/benchmark_${TIMESTAMP}.json" \
    2>&1 | tee "$OUT_DIR/run_${TIMESTAMP}.log"

echo ""
echo "================================================================"
echo " DONE!"
echo "================================================================"
echo " Results saved to: $OUT_DIR"
echo "================================================================"
