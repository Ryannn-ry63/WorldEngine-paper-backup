#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
TASK_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$TASK_CODE_DIR/../../../miniconda3/envs/algengine/bin/python" "$TASK_CODE_DIR/migration/owned_task_runner.py" "$@"
