#!/usr/bin/env bash
# =============================================================================
# FlashAttention Benchmark Runner - 完整自动化流程
# =============================================================================
# 
# 功能：
#   1. 运行所有 stage 的 benchmark (0-4)
#   2. 保存结果到 JSON 文件
#   3. 自动提交到 git
#   4. 用户拉取后可分析数据
#
# Usage:
#   cd /sgl-workspace/Roofline-Analysis/cute_attention/python_dsl
#   bash run_benchmark.sh
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/cute_attention/results"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

echo "================================================================"
echo " FlashAttention CuTe DSL Benchmark"
echo "================================================================"
echo " Timestamp: $TIMESTAMP"
echo " Project:   $PROJECT_ROOT"
echo " Results:    $RESULTS_DIR"
echo "================================================================"

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

# 检查 PyTorch CUDA
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA required'" || {
    echo "ERROR: CUDA not available"
    exit 1
}

# 创建结果目录
mkdir -p "$RESULTS_DIR"

# 保存 GPU 信息
echo ""
echo "[1/4] Saving GPU information..."
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader > "$RESULTS_DIR/gpu_info.txt"
cat "$RESULTS_DIR/gpu_info.txt"

# 运行 benchmark
echo ""
echo "[2/4] Running benchmarks..."
cd "$SCRIPT_DIR"

python3 benchmark_all_stages.py \
    --seqlen 512,1024,2048,4096,8192 \
    --output "benchmark_${TIMESTAMP}.json" \
    --no-commit

# 检查结果
if [[ ! -f "$RESULTS_DIR/benchmark_${TIMESTAMP}.json" ]]; then
    echo "ERROR: Benchmark results not found"
    exit 1
fi

# 保存 benchmark 输出日志
echo ""
echo "[3/4] Saving logs..."
{
    echo "FlashAttention CuTe DSL Benchmark Log"
    echo "Timestamp: $TIMESTAMP"
    echo "GPU: $(cat $RESULTS_DIR/gpu_info.txt)"
    echo ""
    cat "$RESULTS_DIR/benchmark_${TIMESTAMP}.json"
} > "$RESULTS_DIR/run_${TIMESTAMP}.log"

# 提交到 git
echo ""
echo "[4/4] Committing results to git..."
cd "$PROJECT_ROOT"

git add cute_attention/results/ 2>/dev/null || true

if git diff --cached --quiet; then
    echo "  No new results to commit"
else
    git commit -m "bench: add CuTe DSL benchmark results $TIMESTAMP" || true
    git push origin main || {
        echo "  ⚠ Push failed, pulling changes..."
        git pull --rebase origin main
        git push origin main
    }
    echo "  ✓ Results committed and pushed"
fi

echo ""
echo "================================================================"
echo " DONE!"
echo "================================================================"
echo " Results: $RESULTS_DIR/benchmark_${TIMESTAMP}.json"
echo " Pull locally: git pull origin main"
echo "================================================================"
