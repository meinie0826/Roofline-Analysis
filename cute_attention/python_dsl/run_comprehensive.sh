#!/usr/bin/env bash
# =============================================================================
# FlashAttention 完整 Benchmark 流程
# =============================================================================
#
# 功能：
#   1. 运行 benchmark（包含 SDPA, FA2, FA3, FA4 baseline 和我们的实现）
#   2. 验证正确性
#   3. 生成 Roofline 分析
#   4. 自动提交到 git
#
# Usage:
#   cd /sgl-workspace/Roofline-Analysis
#   bash cute_attention/python_dsl/run_comprehensive.sh [--quick]
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/cute_attention/results"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

# 解析参数
QUICK_MODE=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=true
    shift
fi

echo "================================================================"
echo " FlashAttention Comprehensive Benchmark"
echo "================================================================"
echo " Timestamp: $TIMESTAMP"
echo " Project:   $PROJECT_ROOT"
echo " Results:   $RESULTS_DIR"
echo " Quick:     $QUICK_MODE"
echo "================================================================"

# 检查环境
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'" || {
    echo "ERROR: CUDA not available"
    exit 1
}

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 保存 GPU 信息
echo ""
echo "[1/5] Saving GPU information..."
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > "$RESULTS_DIR/gpu_info.txt"
cat "$RESULTS_DIR/gpu_info.txt"

# 运行 comprehensive benchmark
echo ""
echo "[2/5] Running comprehensive benchmark..."

BENCHMARK_ARGS="--config models"
if $QUICK_MODE; then
    BENCHMARK_ARGS="$BENCHMARK_ARGS --quick"
fi

cd "$SCRIPT_DIR"
python3 benchmark_comprehensive.py \
    $BENCHMARK_ARGS \
    --output "$RESULTS_DIR/benchmark_comprehensive_${TIMESTAMP}.json"

# 检查结果
BENCHMARK_FILE="$RESULTS_DIR/benchmark_comprehensive_${TIMESTAMP}.json"
if [[ ! -f "$BENCHMARK_FILE" ]]; then
    echo "ERROR: Benchmark results not found"
    exit 1
fi

# Roofline 分析
echo ""
echo "[3/5] Running Roofline analysis..."
python3 roofline_analysis.py \
    --file "$BENCHMARK_FILE" \
    --out-dir "$RESULTS_DIR"

# 保存完整日志
echo ""
echo "[4/5] Saving complete log..."
{
    echo "FlashAttention Comprehensive Benchmark Log"
    echo "Timestamp: $TIMESTAMP"
    echo "GPU: $(cat $RESULTS_DIR/gpu_info.txt)"
    echo ""
    echo "=== Benchmark Results ==="
    cat "$BENCHMARK_FILE"
    echo ""
    echo "=== Efficiency Analysis ==="
    if [[ -f "$RESULTS_DIR/efficiency_analysis.json" ]]; then
        cat "$RESULTS_DIR/efficiency_analysis.json"
    fi
} > "$RESULTS_DIR/run_comprehensive_${TIMESTAMP}.log"

# Git 提交
echo ""
echo "[5/5] Committing results to git..."
cd "$PROJECT_ROOT"

git add cute_attention/results/ 2>/dev/null || true

if git diff --cached --quiet; then
    echo "  No new results to commit"
else
    git commit -m "bench: add comprehensive benchmark results $TIMESTAMP" || true
    
    # Push with retry
    for i in {1..3}; do
        if git push origin main; then
            echo "  ✓ Results committed and pushed"
            break
        else
            echo "  ⚠ Push attempt $i failed, retrying..."
            git pull --rebase origin main
            sleep 1
        fi
    done
fi

echo ""
echo "================================================================"
echo " DONE!"
echo "================================================================"
echo " Benchmark: $BENCHMARK_FILE"
echo " Analysis:  $RESULTS_DIR/efficiency_analysis.json"
echo " Plots:     $RESULTS_DIR/roofline_analysis.png"
echo "            $RESULTS_DIR/ablation_analysis.png"
echo ""
echo " Pull locally: git pull origin main"
echo "================================================================"
