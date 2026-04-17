#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_100a}
CUTLASS_DIR=${CUTLASS_DIR:-$ROOT_DIR/../../cutlass}
REPEATS=${REPEATS:-10}
WARMUP_REPEATS=${WARMUP_REPEATS:-3}
ITERS=${ITERS:-2048}
BUFFER_BYTES=${BUFFER_BYTES:-65536}
ALIGN_LIST=${ALIGN_LIST:-32,64,128}
VEC_LIST=${VEC_LIST:-4,8,16}
SOFT_TILE_N_LIST=${SOFT_TILE_N_LIST:-64,128,256}
STAGES_LIST=${STAGES_LIST:-1,2,4}
CUTLASS_STAGES_LIST=${CUTLASS_STAGES_LIST:-2,4}
CUTLASS_TILE_N_LIST=${CUTLASS_TILE_N_LIST:-64,128,256}
GEMM_M=${GEMM_M:-4096}
GEMM_K=${GEMM_K:-4096}
PINGPONG_ITERS=${PINGPONG_ITERS:-10000}

if [[ -z "${OUTDIR:-}" ]]; then
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTDIR="$ROOT_DIR/results/$TIMESTAMP"
fi
mkdir -p "$OUTDIR"

READ_CSV="$OUTDIR/results_dsmem_read.csv"
WRITE_CSV="$OUTDIR/results_dsmem_write.csv"
PINGPONG_CSV="$OUTDIR/results_dsmem_pingpong.csv"
SOFT_GEMM_CSV="$OUTDIR/results_software_gemm.csv"
CUTLASS_GEMM_CSV="$OUTDIR/results_cutlass_gemm.csv"
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

git_commit=$(git -C "$ROOT_DIR/../.." rev-parse HEAD 2>/dev/null || echo unknown)
gpu_info=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || echo unknown)

HAVE_CUTLASS=0
if [[ -f "$CUTLASS_DIR/include/cute/tensor.hpp" && -d "$CUTLASS_DIR/tools/util/include" ]]; then
  HAVE_CUTLASS=1
fi

python3 - <<PY
import json
metadata = {
  "timestamp": "$(date -Iseconds)",
  "root_dir": "$ROOT_DIR",
  "git_commit": "$git_commit",
  "arch": "$ARCH",
  "nvcc": "$NVCC",
  "cutlass_dir": "$CUTLASS_DIR",
  "have_cutlass": $HAVE_CUTLASS,
  "gpu_info": """$gpu_info""",
  "repeats": $REPEATS,
  "warmup_repeats": $WARMUP_REPEATS,
  "iters": $ITERS,
  "buffer_bytes": $BUFFER_BYTES,
  "align_list": "$ALIGN_LIST",
  "vec_list": "$VEC_LIST",
  "soft_tile_n_list": "$SOFT_TILE_N_LIST",
  "cutlass_tile_n_list": "$CUTLASS_TILE_N_LIST",
  "cutlass_stages_list": "$CUTLASS_STAGES_LIST",
  "stages_list": "$STAGES_LIST",
  "gemm_m": $GEMM_M,
  "gemm_k": $GEMM_K,
  "pingpong_iters": $PINGPONG_ITERS,
}
with open("$METADATA_JSON", "w") as f:
    json.dump(metadata, f, indent=2)
PY

printf 'benchmark,mode,repeat,iters,buffer_bytes,align_bytes,vec_bytes,cycles,bytes,elapsed_ns,bandwidth_gbps,checksum\n' > "$READ_CSV"
printf 'benchmark,mode,repeat,iters,buffer_bytes,align_bytes,vec_bytes,cycles,bytes,elapsed_ns,bandwidth_gbps,checksum\n' > "$WRITE_CSV"
printf 'benchmark,repeat,iters,cycles,elapsed_ns,cycles_per_roundtrip,ns_per_roundtrip\n' > "$PINGPONG_CSV"
printf 'benchmark,mode,repeat,m,n,k,tile_n,stages,elapsed_ms,gflops,bytes_b_share,checksum\n' > "$SOFT_GEMM_CSV"
printf 'benchmark,mode,m,n,k,tile_n,stages,avg_ms,gflops,bytes_b_share\n' > "$CUTLASS_GEMM_CSV"

compile() {
  local src="$1"
  local out="$2"
  shift 2
  echo "==============================================="
  echo "Compiling $src"
  echo "==============================================="
  "$NVCC" -std=c++17 -O3 -arch="$ARCH" "$src" -o "$out" "$@"
}

append_results() {
  local txt="$1"
  local csv="$2"
  python3 - "$txt" "$csv" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text().splitlines()
csv_path = Path(sys.argv[2])

for line in text:
    if not line.startswith("RESULT "):
        continue
    fields = {}
    for token in line.split()[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    existing = csv_path.read_text()
    if "results_dsmem_read.csv" in str(csv_path) or "results_dsmem_write.csv" in str(csv_path):
        row = [
            fields["benchmark"], fields["mode"], fields["repeat"], fields["iters"],
            fields["buffer_bytes"], fields["align_bytes"], fields["vec_bytes"],
            fields["cycles"], fields["bytes"], fields["elapsed_ns"], fields["bandwidth_gbps"],
            fields["checksum"],
        ]
    elif "results_dsmem_pingpong.csv" in str(csv_path):
        row = [
            fields["benchmark"], fields["repeat"], fields["iters"], fields["cycles"],
            fields["elapsed_ns"], fields["cycles_per_roundtrip"], fields["ns_per_roundtrip"],
        ]
    elif "results_software_gemm.csv" in str(csv_path):
        row = [
            fields["benchmark"], fields["mode"], fields["repeat"], fields["m"], fields["n"], fields["k"],
            fields["tile_n"], fields["stages"], fields["elapsed_ms"], fields["gflops"],
            fields["bytes_b_share"], fields["checksum"],
        ]
    else:
        row = [
            fields["benchmark"], fields["mode"], fields["m"], fields["n"], fields["k"],
            fields["tile_n"], fields["stages"], fields["avg_ms"], fields["gflops"],
            fields["bytes_b_share"],
        ]
    csv_path.write_text(existing + ",".join(row) + "\n")
PY
}

compile bench_dsmem_read.cu bench_dsmem_read
compile bench_dsmem_write.cu bench_dsmem_write
compile bench_dsmem_pingpong.cu bench_dsmem_pingpong
compile bench_software_dsmem_gemm.cu bench_software_dsmem_gemm
if [[ "$HAVE_CUTLASS" == "1" ]]; then
  compile bench_cutlass_2sm_gemm.cu bench_cutlass_2sm_gemm \
    --expt-relaxed-constexpr \
    -I"$CUTLASS_DIR/include" \
    -I"$CUTLASS_DIR/tools/util/include"
else
  echo "WARNING: CUTLASS not found at $CUTLASS_DIR"
  echo "Skipping bench_cutlass_2sm_gemm compilation and runtime sweep."
  echo "Set CUTLASS_DIR=/path/to/cutlass to enable the hardware 1SM/2SM comparison."
fi

IFS=',' read -r -a ALIGN_VALUES <<< "$ALIGN_LIST"
IFS=',' read -r -a VEC_VALUES <<< "$VEC_LIST"
IFS=',' read -r -a SOFT_TILE_VALUES <<< "$SOFT_TILE_N_LIST"
IFS=',' read -r -a CUTLASS_TILE_VALUES <<< "$CUTLASS_TILE_N_LIST"
IFS=',' read -r -a STAGES_VALUES <<< "$STAGES_LIST"
IFS=',' read -r -a CUTLASS_STAGES_VALUES <<< "$CUTLASS_STAGES_LIST"

for align in "${ALIGN_VALUES[@]}"; do
  for vec in "${VEC_VALUES[@]}"; do
    for mode in local remote; do
      log="$OUTDIR/read_${mode}_a${align}_v${vec}.txt"
      ./bench_dsmem_read --mode="$mode" --repeats="$REPEATS" --warmup-repeats="$WARMUP_REPEATS" \
        --iters="$ITERS" --buffer-bytes="$BUFFER_BYTES" --align-bytes="$align" --vec-bytes="$vec" | tee "$log"
      append_results "$log" "$READ_CSV"

      log="$OUTDIR/write_${mode}_a${align}_v${vec}.txt"
      ./bench_dsmem_write --mode="$mode" --repeats="$REPEATS" --warmup-repeats="$WARMUP_REPEATS" \
        --iters="$ITERS" --buffer-bytes="$BUFFER_BYTES" --align-bytes="$align" --vec-bytes="$vec" | tee "$log"
      append_results "$log" "$WRITE_CSV"
    done
  done
done

pingpong_log="$OUTDIR/pingpong.txt"
./bench_dsmem_pingpong --repeats="$REPEATS" --warmup-repeats="$WARMUP_REPEATS" --iters="$PINGPONG_ITERS" | tee "$pingpong_log"
append_results "$pingpong_log" "$PINGPONG_CSV"

for tile_n in "${SOFT_TILE_VALUES[@]}"; do
  for stages in "${STAGES_VALUES[@]}"; do
    for mode in local remote; do
      log="$OUTDIR/software_${mode}_n${tile_n}_s${stages}.txt"
      ./bench_software_dsmem_gemm --mode="$mode" --m="$GEMM_M" --n="$tile_n" --k="$GEMM_K" \
        --tile-n="$tile_n" --stages="$stages" --repeats="$REPEATS" --warmup-repeats="$WARMUP_REPEATS" | tee "$log"
      append_results "$log" "$SOFT_GEMM_CSV"
    done
  done
done

if [[ "$HAVE_CUTLASS" == "1" ]]; then
  for tile_n in "${CUTLASS_TILE_VALUES[@]}"; do
    for stages in "${CUTLASS_STAGES_VALUES[@]}"; do
      for mode in 1sm 2sm; do
        if [[ "$mode" == "1sm" && "$tile_n" == "256" ]]; then
          echo "Skipping unsupported CUTLASS case mode=1sm tile_n=256"
          continue
        fi
        log="$OUTDIR/cutlass_${mode}_n${tile_n}_s${stages}.txt"
        ./bench_cutlass_2sm_gemm --mode="$mode" --m="$GEMM_M" --n="$tile_n" --k="$GEMM_K" \
          --tile-n="$tile_n" --stages="$stages" --repeats="$REPEATS" --warmup-repeats="$WARMUP_REPEATS" | tee "$log"
        append_results "$log" "$CUTLASS_GEMM_CSV"
      done
    done
  done
fi

echo "Results written to $OUTDIR"
