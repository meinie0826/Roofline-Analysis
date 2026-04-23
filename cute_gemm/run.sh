#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/cutlass/python/CuTeDSL:${PYTHONPATH:-}"

python3 "${ROOT}/cute_gemm/mma_gemm_1cta_cutedsl.py" "$@"
