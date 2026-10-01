#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/eval_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
ARGS=(--devices "$CUDA_VISIBLE_DEVICES" --session "${EVAL_SESSION:-posttrain_20261001}" --idle-seconds "${QUEUE_IDLE_SECONDS:-120}")
if [[ ${CHECK_ONLY:-0} == 1 ]]; then ARGS+=(--check-only); fi
if [[ ${SUMMARY_ONLY:-0} == 1 ]]; then ARGS+=(--summary); fi
if [[ ${ONCE:-0} == 1 ]]; then ARGS+=(--once); fi
printf 'Evaluation only; shared 24-hour window. Live log: %s\n' "$LOG"
"$PYTHON_BIN" -u "$ROOT/eval_worker.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
