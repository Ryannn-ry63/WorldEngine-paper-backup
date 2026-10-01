#!/usr/bin/env bash
set -euo pipefail
WORK=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
JOB=${1:?Usage: bash start_job_h100.sh admit_0 or train_A_s0 or dev_parent_R}
[[ "$JOB" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "Invalid job ID" >&2; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$WORK/registry/launcher_logs"
LOG="$WORK/registry/launcher_logs/${JOB}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
ARGS=("$JOB" --devices "$CUDA_VISIBLE_DEVICES")
if [[ ${CHECK_ONLY:-0} == 1 ]]; then ARGS+=(--check-only); fi
printf 'Job: %s\nLive log: %s\n' "$JOB" "$LOG"
"$PYTHON_BIN" -u "$WORK/handoff/run_job.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
