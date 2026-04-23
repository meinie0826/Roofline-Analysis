#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DSL_SRC="${ROOT}/cutlass/python/CuTeDSL"
CUTLASS_DSL_LIBS="${CUTLASS_DSL_SRC}/cutlass/_mlir/_mlir_libs"

if [[ -d "${CUTLASS_DSL_LIBS}" ]]; then
  export PYTHONPATH="${CUTLASS_DSL_SRC}:${PYTHONPATH:-}"
else
  echo "[cute_gemm/run.sh] local CuTeDSL MLIR libs not found at ${CUTLASS_DSL_LIBS}" >&2
  echo "[cute_gemm/run.sh] falling back to installed python package 'cutlass'" >&2
fi

python3 "${ROOT}/cute_gemm/mma_gemm_1cta_cutedsl.py" "$@"
