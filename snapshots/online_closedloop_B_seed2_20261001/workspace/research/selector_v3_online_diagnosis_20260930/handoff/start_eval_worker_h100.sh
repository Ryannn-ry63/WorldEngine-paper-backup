#!/usr/bin/env bash
set -euo pipefail
WORK=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$WORK/registry/launcher_logs"
LOG="$WORK/registry/launcher_logs/queue_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
ARGS=(--group "${QUEUE_GROUP:-all}" --devices "$CUDA_VISIBLE_DEVICES")
if [[ ${CHECK_ONLY:-0} == 1 ]]; then ARGS+=(--check-only); fi
printf 'Independent evaluation/diagnostic queue. Live log: %s\n' "$LOG"
"$PYTHON_BIN" -u "$WORK/handoff/run_queue.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
