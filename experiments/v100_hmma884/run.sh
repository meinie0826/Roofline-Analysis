#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

NVCC=${NVCC:-nvcc}
CUOBJDUMP=${CUOBJDUMP:-cuobjdump}
ARCH=${ARCH:-sm_70}
REPEATS=${REPEATS:-10}
LOOP_ITERS=${LOOP_ITERS:-4096}
WARMUP_ITERS=${WARMUP_ITERS:-64}
WARMUP_LAUNCHES=${WARMUP_LAUNCHES:-3}
UNROLL_LIST=${UNROLL_LIST:-1,2,4,8,16}
F16_STREAMS_LIST=${F16_STREAMS_LIST:-2,4,8}
F32_STREAMS_LIST=${F32_STREAMS_LIST:-1,2,4,8}

if [[ -z "${OUTDIR:-}" ]]; then
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTDIR="$ROOT_DIR/results/$TIMESTAMP"
fi
mkdir -p "$OUTDIR"

RAW_CSV="$OUTDIR/results_raw.csv"
SUMMARY_CSV="$OUTDIR/results_summary.csv"
DERIVED_CSV="$OUTDIR/results_derived.csv"
METADATA_JSON="$OUTDIR/metadata.json"
RUN_LOG="$OUTDIR/run.log"

exec > >(tee "$RUN_LOG") 2>&1

echo "Output directory: $OUTDIR"

require_tool() {
  local tool_name="$1"
  local hint="$2"
  if ! command -v "$tool_name" >/dev/null 2>&1; then
    echo "ERROR: required tool '$tool_name' not found in PATH."
    echo "Hint: $hint"
    exit 1
  fi
}

require_tool "$NVCC" "Set NVCC to your compiler path, e.g. NVCC=/usr/local/cuda/bin/nvcc bash run.sh"
require_tool "$CUOBJDUMP" "Set CUOBJDUMP to your cuobjdump path, e.g. CUOBJDUMP=/usr/local/cuda/bin/cuobjdump bash run.sh"

git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
git_status=$(git -C "$ROOT_DIR" status --short 2>/dev/null || true)
hostname_value=$(hostname)
nvcc_version=$($NVCC --version 2>&1 | tr '\n' ' ' | sed 's/"/\\"/g')
gpu_info=$(nvidia-smi --query-gpu=name,compute_cap,clocks.sm,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo unknown)

python3 - <<PY
import json
metadata = {
  "timestamp": "$(date -Iseconds)",
  "root_dir": "$ROOT_DIR",
  "outdir": "$OUTDIR",
  "hostname": "$hostname_value",
  "git_commit": "$git_commit",
  "git_status": """$git_status""",
  "arch": "$ARCH",
  "nvcc": "$NVCC",
  "nvcc_version": """$nvcc_version""",
  "cuobjdump": "$CUOBJDUMP",
  "gpu_info": """$gpu_info""",
  "loop_iters": $LOOP_ITERS,
  "warmup_iters": $WARMUP_ITERS,
  "warmup_launches": $WARMUP_LAUNCHES,
  "repeats": $REPEATS,
  "unroll_list": "$UNROLL_LIST",
  "f16_streams_list": "$F16_STREAMS_LIST",
  "f32_streams_list": "$F32_STREAMS_LIST",
}
with open("$METADATA_JSON", "w") as f:
    json.dump(metadata, f, indent=2)
PY

printf 'benchmark,dtype,mode,streams,repeat,loop_iters,unroll,total_mma,cycles,cycles_per_mma,sink_lo,sink_hi\n' > "$RAW_CSV"

compile() {
  local src="$1"
  local out="$2"
  echo "==============================================="
  echo "Compiling $src"
  echo "==============================================="
  "$NVCC" -arch="$ARCH" -O3 "$src" -o "$out"
}

dump_sass() {
  local bin="$1"
  local out="$OUTDIR/sass_${bin}.txt"
  echo "==============================================="
  echo "Dumping SASS for $bin"
  echo "==============================================="
  "$CUOBJDUMP" --dump-sass "./$bin" > "$out"
  grep HMMA "$out" || true
  echo "Saved SASS to $out"
}

append_results() {
  local txt="$1"
  python3 - "$txt" "$RAW_CSV" <<'PY'
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text().splitlines()
out = Path(sys.argv[2])
for line in text:
    if not line.startswith("RESULT "):
        continue
    fields = {}
    for token in line.split()[1:]:
        key, value = token.split("=", 1)
        fields[key] = value
    out.write_text(out.read_text() + ",".join([
        fields["benchmark"],
        fields["dtype"],
        fields["mode"],
        fields["streams"],
        fields["repeat"],
        fields["loop_iters"],
        fields["unroll"],
        fields["total_mma"],
        fields["cycles"],
        fields["cycles_per_mma"],
        fields["sink_lo"],
        fields["sink_hi"],
    ]) + "\n")
PY
}

run_case() {
  local name="$1"
  local unroll="$2"
  shift
  shift
  local suffix="u${unroll}"
  for arg in "$@"; do
    if [[ "$arg" == --streams=* ]]; then
      suffix="${suffix}_s${arg#--streams=}"
    fi
  done
  local log="$OUTDIR/${name}_${suffix}.txt"
  echo "==============================================="
  echo "Running $name --unroll=$unroll $*"
  echo "==============================================="
  "./$name" --loop-iters="$LOOP_ITERS" --warmup-iters="$WARMUP_ITERS" --warmup-launches="$WARMUP_LAUNCHES" --repeats="$REPEATS" --unroll="$unroll" "$@" | tee "$log"
  append_results "$log"
}

compile bench_empty.cu bench_empty
compile bench_empty_matched_dep.cu bench_empty_matched_dep
compile bench_empty_matched_f32acc.cu bench_empty_matched_f32acc
compile bench_mma_f16_dep.cu bench_mma_f16_dep
compile bench_mma_f16_indep.cu bench_mma_f16_indep
compile bench_mma_f32acc.cu bench_mma_f32acc

dump_sass bench_mma_f16_dep
dump_sass bench_empty_matched_dep
dump_sass bench_mma_f16_indep
dump_sass bench_empty_matched_f32acc
dump_sass bench_mma_f32acc

IFS=',' read -r -a UNROLL_VALUES <<< "$UNROLL_LIST"
IFS=',' read -r -a F16_STREAM_VALUES <<< "$F16_STREAMS_LIST"
IFS=',' read -r -a F32_STREAM_VALUES <<< "$F32_STREAMS_LIST"

for unroll in "${UNROLL_VALUES[@]}"; do
  run_case bench_empty "$unroll"
  run_case bench_empty_matched_dep "$unroll"
  run_case bench_mma_f16_dep "$unroll"
  for streams in "${F32_STREAM_VALUES[@]}"; do
    run_case bench_empty_matched_f32acc "$unroll" --streams="$streams"
  done
  for streams in "${F16_STREAM_VALUES[@]}"; do
    run_case bench_mma_f16_indep "$unroll" --streams="$streams"
  done
  for streams in "${F32_STREAM_VALUES[@]}"; do
    run_case bench_mma_f32acc "$unroll" --streams="$streams"
  done
done

python3 - "$RAW_CSV" "$SUMMARY_CSV" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

raw_path, summary_path = sys.argv[1], sys.argv[2]

def hmma_steps_per_mma(dtype):
    if dtype == "f16_f16_f16_f16":
        return 2
    if dtype == "f32_f16_f16_f16":
        return 4
    return 0

groups = defaultdict(list)
groups_cycles = defaultdict(list)
with open(raw_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['benchmark'], row['dtype'], row['mode'], row['streams'], row['loop_iters'], row['unroll'])
        groups[key].append(float(row['cycles_per_mma']))
        groups_cycles[key].append(float(row['cycles']))

with open(summary_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'benchmark', 'dtype', 'mode', 'streams', 'loop_iters', 'unroll', 'repeats',
        'mean_cycles', 'median_cycles', 'stddev_cycles',
        'hmma_steps_per_mma',
        'mean_cycles_per_mma', 'median_cycles_per_mma', 'stddev_cycles_per_mma',
        'min_cycles_per_mma', 'max_cycles_per_mma',
        'mean_cycles_per_hmma_step'
    ])
    for key in sorted(groups):
        values = groups[key]
        cycles_values = groups_cycles[key]
        stddev = statistics.pstdev(values) if len(values) > 1 else 0.0
        cycles_stddev = statistics.pstdev(cycles_values) if len(cycles_values) > 1 else 0.0
        steps = hmma_steps_per_mma(key[1])
        mean_cycles = statistics.mean(values)
        mean_per_step = (mean_cycles / steps) if steps else 0.0
        writer.writerow([
            *key,
            len(values),
            f"{statistics.mean(cycles_values):.2f}",
            f"{statistics.median(cycles_values):.2f}",
            f"{cycles_stddev:.2f}",
            steps,
            f"{mean_cycles:.8f}",
            f"{statistics.median(values):.8f}",
            f"{stddev:.8f}",
            f"{min(values):.8f}",
            f"{max(values):.8f}",
            f"{mean_per_step:.8f}",
        ])
PY

python3 - "$SUMMARY_CSV" "$DERIVED_CSV" <<'PY'
import csv
import sys

summary_path, derived_path = sys.argv[1], sys.argv[2]
matched_f16 = {}
dep_f16 = {}
matched_f32 = {}
mma_f32 = {}
with open(summary_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["benchmark"] == "bench_empty_matched_dep":
            key = (row["loop_iters"], row["unroll"], row["streams"])
            matched_f16[key] = row
        elif row["benchmark"] == "bench_mma_f16_dep":
            key = (row["loop_iters"], row["unroll"], row["streams"])
            dep_f16[key] = row
        elif row["benchmark"] == "bench_empty_matched_f32acc":
            key = (row["loop_iters"], row["unroll"], row["streams"])
            matched_f32[key] = row
        elif row["benchmark"] == "bench_mma_f32acc":
            key = (row["loop_iters"], row["unroll"], row["streams"])
            mma_f32[key] = row

with open(derived_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "benchmark",
        "mode",
        "loop_iters",
        "unroll",
        "streams",
        "mean_cycles",
        "matched_empty_mean_cycles",
        "delta_cycles",
        "dep_mean_cycles_per_mma",
        "matched_empty_mean_cycles_per_mma",
        "delta_cycles_per_mma",
        "delta_cycles_per_hmma_step",
    ])
    for key in sorted(dep_f16):
        if key not in matched_f16:
            continue
        dep_row = dep_f16[key]
        empty_row = matched_f16[key]
        dep_mean = float(dep_row["mean_cycles_per_mma"])
        empty_mean = float(empty_row["mean_cycles_per_mma"])
        dep_cycles = float(dep_row["mean_cycles"])
        empty_cycles = float(empty_row["mean_cycles"])
        delta = dep_mean - empty_mean
        writer.writerow([
            dep_row["benchmark"],
            dep_row["mode"],
            key[0],
            key[1],
            key[2],
            f"{dep_cycles:.2f}",
            f"{empty_cycles:.2f}",
            f"{(dep_cycles - empty_cycles):.2f}",
            f"{dep_mean:.8f}",
            f"{empty_mean:.8f}",
            f"{delta:.8f}",
            f"{(delta / 2.0):.8f}",
        ])
    for key in sorted(mma_f32):
        if key not in matched_f32:
            continue
        mma_row = mma_f32[key]
        empty_row = matched_f32[key]
        mma_mean = float(mma_row["mean_cycles_per_mma"])
        empty_mean = float(empty_row["mean_cycles_per_mma"])
        mma_cycles = float(mma_row["mean_cycles"])
        empty_cycles = float(empty_row["mean_cycles"])
        delta = mma_mean - empty_mean
        writer.writerow([
            mma_row["benchmark"],
            mma_row["mode"],
            key[0],
            key[1],
            key[2],
            f"{mma_cycles:.2f}",
            f"{empty_cycles:.2f}",
            f"{(mma_cycles - empty_cycles):.2f}",
            f"{mma_mean:.8f}",
            f"{empty_mean:.8f}",
            f"{delta:.8f}",
            f"{(delta / 4.0):.8f}",
        ])
PY

echo "Saved raw results to $RAW_CSV"
echo "Saved summary to $SUMMARY_CSV"
echo "Saved derived results to $DERIVED_CSV"
echo "Saved metadata to $METADATA_JSON"
