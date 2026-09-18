#!/usr/bin/env bash
set -euo pipefail
CHECK_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECK_PYTHON="${STUDY_PYTHON:-$CHECK_CODE_DIR/../../../miniconda3/envs/algengine/bin/python}"
if [[ ! -x "$CHECK_PYTHON" ]]; then
    echo "Shared Python is inaccessible: $CHECK_PYTHON" >&2
    exit 2
fi
PYTHONDONTWRITEBYTECODE=1 "$CHECK_PYTHON" "$CHECK_CODE_DIR/scripts/check_paper_project_mounts.py"
