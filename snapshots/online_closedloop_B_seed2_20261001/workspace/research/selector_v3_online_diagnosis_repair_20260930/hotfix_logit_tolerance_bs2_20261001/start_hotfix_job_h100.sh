#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
JOB=${1:?Usage: bash start_hotfix_job_h100.sh train_B_s2}
PYTHON_BIN=${PYTHON_BIN:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/${JOB}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
ARGS=("$JOB" --devices "$CUDA_VISIBLE_DEVICES")
if [[ ${CHECK_ONLY:-0} == 1 ]]; then ARGS+=(--check-only); fi
printf 'Hotfix resume: %s\nLive log: %s\n' "$JOB" "$LOG"
"$PYTHON_BIN" -u "$ROOT/resume_train_h100.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
