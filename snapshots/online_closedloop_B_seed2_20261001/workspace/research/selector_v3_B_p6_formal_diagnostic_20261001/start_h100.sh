#!/usr/bin/env bash
set -euo pipefail
TASK_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ALG_PY=/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python
export PYTHONDONTWRITEBYTECODE=1
export EVAL_SCRATCH_ROOT=${EVAL_SCRATCH_ROOT:-/tmp/worldengine_B_p6_formal}
mkdir -p "$TASK_DIR/logs"
LOG_FILE="$TASK_DIR/logs/$(hostname)_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
echo "Worker log: $LOG_FILE"
"$ALG_PY" -u "$TASK_DIR/run.py" "$@" 2>&1 | tee "$LOG_FILE"
