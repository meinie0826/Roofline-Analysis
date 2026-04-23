#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DSL_SRC="${ROOT}/cutlass/python/CuTeDSL"
CUTLASS_DSL_LIBS="${CUTLASS_DSL_SRC}/cutlass/_mlir/_mlir_libs"
DEBUG_LOG_DIR="${ROOT}/cute_gemm/debug_logs"

debug_swizzle_enabled=0
for arg in "$@"; do
  if [[ "${arg}" == "--debug-swizzle" ]]; then
    debug_swizzle_enabled=1
    break
  fi
done

if [[ -d "${CUTLASS_DSL_LIBS}" ]]; then
  export PYTHONPATH="${CUTLASS_DSL_SRC}:${PYTHONPATH:-}"
else
  echo "[cute_gemm/run.sh] local CuTeDSL MLIR libs not found at ${CUTLASS_DSL_LIBS}" >&2
  echo "[cute_gemm/run.sh] falling back to installed python package 'cutlass'" >&2
fi

if [[ "${debug_swizzle_enabled}" -eq 1 ]]; then
  mkdir -p "${DEBUG_LOG_DIR}"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_path="${DEBUG_LOG_DIR}/swizzle_debug_${ts}.log"
  latest_path="${DEBUG_LOG_DIR}/latest_swizzle_debug.log"
  echo "[cute_gemm/run.sh] writing debug log to ${log_path}" >&2
  python3 "${ROOT}/cute_gemm/mma_gemm_1cta_cutedsl.py" "$@" 2>&1 | tee "${log_path}"
  cp "${log_path}" "${latest_path}"
else
  python3 "${ROOT}/cute_gemm/mma_gemm_1cta_cutedsl.py" "$@"
fi
