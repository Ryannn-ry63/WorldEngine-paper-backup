#!/usr/bin/env bash
set -euo pipefail
REPAIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$REPAIR/logs"
JOB=${1:?Usage: bash start_job_h100.sh train_A_s0 or roots48}
[[ "$JOB" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "Invalid job ID" >&2; exit 2; }
LOG="$REPAIR/logs/${JOB}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
ARGS=("$JOB" --devices "$CUDA_VISIBLE_DEVICES")
if [[ ${CHECK_ONLY:-0} == 1 ]]; then ARGS+=(--check-only); fi
printf 'Repaired launcher, original training checkpoint. Job: %s\nLog: %s\n' "$JOB" "$LOG"
"$PYTHON_BIN" -u "$REPAIR/repair_run.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
