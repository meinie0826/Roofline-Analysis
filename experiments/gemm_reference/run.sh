#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_100a}
SHAPES=${SHAPES:-128x128x128,256x256x256,512x512x512}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}
BLOCK_M=${BLOCK_M:-16}
BLOCK_N=${BLOCK_N:-16}
SEED=${SEED:-2026}
ALPHA=${ALPHA:-1.0}
BETA=${BETA:-0.0}
ATOL=${ATOL:-1e-3}
RTOL=${RTOL:-1e-3}
BACKEND=${BACKEND:-all}
CHECK=${CHECK:-1}

if [[ -z "${OUTDIR:-}" ]]; then
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTDIR="$ROOT_DIR/results/$TIMESTAMP"
fi
mkdir -p "$OUTDIR"

BENCHMARK_CSV="$OUTDIR/benchmark.csv"
CORRECTNESS_CSV="$OUTDIR/correctness.csv"
SUMMARY_JSON="$OUTDIR/summary.json"
METADATA_JSON="$OUTDIR/metadata.json"
RUN_LOG="$OUTDIR/run.log"

exec > >(tee "$RUN_LOG") 2>&1

require_tool() {
  local tool_name="$1"
  local hint="$2"
  if ! command -v "$tool_name" >/dev/null 2>&1; then
    echo "ERROR: required tool '$tool_name' not found."
    echo "Hint: $hint"
    exit 1
  fi
}

require_tool "$NVCC" "Set NVCC=/path/to/nvcc"
require_tool python3 "python3 is required to write metadata and summaries"

git_commit=$(git -C "$ROOT_DIR/../.." rev-parse HEAD 2>/dev/null || echo unknown)
git_status=$(git -C "$ROOT_DIR/../.." status --short 2>/dev/null || true)
gpu_info=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo unknown)
nvcc_version=$("$NVCC" --version 2>&1 | tr '\n' ' ' | sed 's/"/\\"/g')

python3 - <<PY
import json

metadata = {
  "timestamp": "$(date -Iseconds)",
  "root_dir": "$ROOT_DIR",
  "outdir": "$OUTDIR",
  "git_commit": "$git_commit",
  "git_status": """$git_status""",
  "arch": "$ARCH",
  "nvcc": "$NVCC",
  "nvcc_version": """$nvcc_version""",
  "gpu_info": """$gpu_info""",
  "shapes": "$SHAPES",
  "warmup": $WARMUP,
  "iters": $ITERS,
  "block_m": $BLOCK_M,
  "block_n": $BLOCK_N,
  "seed": $SEED,
  "alpha": $ALPHA,
  "beta": $BETA,
  "atol": $ATOL,
  "rtol": $RTOL,
  "backend": "$BACKEND",
  "check": $CHECK,
}

with open("$METADATA_JSON", "w") as f:
  json.dump(metadata, f, indent=2)
PY

printf 'backend,m,n,k,warmup,iters,avg_ms,median_ms,min_ms,gflops,tflops,checksum\n' > "$BENCHMARK_CSV"
printf 'backend,m,n,k,pass,atol,rtol,max_abs,max_rel,l2_rel,fail_count\n' > "$CORRECTNESS_CSV"

echo "==============================================="
echo "Compiling bench_gemm_reference.cu"
echo "==============================================="
make clean
make NVCC="$NVCC" ARCH="$ARCH"

append_results() {
  local log_path="$1"
  python3 - "$log_path" "$BENCHMARK_CSV" "$CORRECTNESS_CSV" <<'PY'
import csv
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
benchmark_csv = Path(sys.argv[2])
correctness_csv = Path(sys.argv[3])

for line in log_path.read_text().splitlines():
  if line.startswith("RESULT "):
    fields = {}
    for token in line.split()[1:]:
      if "=" not in token:
        continue
      key, value = token.split("=", 1)
      fields[key] = value
    with benchmark_csv.open("a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
        fields["backend"],
        fields["m"],
        fields["n"],
        fields["k"],
        fields["warmup"],
        fields["iters"],
        fields["avg_ms"],
        fields["median_ms"],
        fields["min_ms"],
        fields["gflops"],
        fields["tflops"],
        fields["checksum"],
      ])
  elif line.startswith("CHECK "):
    fields = {}
    for token in line.split()[1:]:
      if "=" not in token:
        continue
      key, value = token.split("=", 1)
      fields[key] = value
    with correctness_csv.open("a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow([
        fields["backend"],
        fields["m"],
        fields["n"],
        fields["k"],
        fields["pass"],
        fields["atol"],
        fields["rtol"],
        fields["max_abs"],
        fields["max_rel"],
        fields["l2_rel"],
        fields["fail_count"],
      ])
PY
}

IFS=',' read -r -a SHAPE_VALUES <<< "$SHAPES"
for shape in "${SHAPE_VALUES[@]}"; do
  IFS='x' read -r M N K <<< "$shape"
  LOG_PATH="$OUTDIR/shape_${M}x${N}x${K}.txt"

  echo "==============================================="
  echo "Running shape ${M}x${N}x${K}"
  echo "==============================================="
  ./bench_gemm_reference \
    --m="$M" \
    --n="$N" \
    --k="$K" \
    --warmup="$WARMUP" \
    --iters="$ITERS" \
    --block-m="$BLOCK_M" \
    --block-n="$BLOCK_N" \
    --seed="$SEED" \
    --alpha="$ALPHA" \
    --beta="$BETA" \
    --atol="$ATOL" \
    --rtol="$RTOL" \
    --backend="$BACKEND" \
    --check="$CHECK" | tee "$LOG_PATH"

  append_results "$LOG_PATH"
done

python3 - "$BENCHMARK_CSV" "$CORRECTNESS_CSV" "$SUMMARY_JSON" <<'PY'
import csv
import json
import sys

benchmark_csv, correctness_csv, summary_json = sys.argv[1:4]

with open(benchmark_csv, newline="") as f:
  benchmark_rows = list(csv.DictReader(f))

with open(correctness_csv, newline="") as f:
  correctness_rows = list(csv.DictReader(f))

summary = {
  "benchmark_rows": benchmark_rows,
  "correctness_rows": correctness_rows,
}

with open(summary_json, "w") as f:
  json.dump(summary, f, indent=2)
PY

echo "==============================================="
echo "Complete"
echo "==============================================="
echo "Output directory: $OUTDIR"
echo "Benchmark CSV:    $BENCHMARK_CSV"
echo "Correctness CSV:  $CORRECTNESS_CSV"
echo "Summary JSON:     $SUMMARY_JSON"
