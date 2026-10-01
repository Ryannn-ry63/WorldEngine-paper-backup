#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${1:?Usage: bash start_recovery_h100.sh train_B_s2|eval}
PYTHON_BIN=${PYTHON_BIN:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${MODE}_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
ARGS=("$MODE" --devices "$CUDA_VISIBLE_DEVICES" --session "${EVAL_SESSION:-disk_recovery_20261001}" --idle-seconds "${QUEUE_IDLE_SECONDS:-120}")
printf 'Infrastructure recovery: %s\nLive log: %s\n' "$MODE" "$LOG"
if [[ ${CHECK_ONLY:-0} == 1 ]]; then
  "$PYTHON_BIN" -u "$ROOT/recover.py" "${ARGS[@]}" --check-only 2>&1 | tee "$LOG"
else
  "$PYTHON_BIN" -u "$ROOT/recover.py" "${ARGS[@]}" --check-only 2>&1 | tee "$LOG"
  "$PYTHON_BIN" -u "$ROOT/recover.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
fi
