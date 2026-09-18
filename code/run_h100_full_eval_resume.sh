#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
RESUME_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$RESUME_CODE_DIR"
if [[ -z "${STUDY_PYTHON:-}" ]]; then
    RESUME_BOOTSTRAP="$RESUME_CODE_DIR/../../../miniconda3/envs/algengine/bin/python"
    [[ -x "$RESUME_BOOTSTRAP" ]] || { echo 'Shared algengine Python is unavailable; set STUDY_PYTHON.' >&2; exit 2; }
    STUDY_PYTHON="$("$RESUME_BOOTSTRAP" -c 'import json; from pathlib import Path; print(json.loads((Path.cwd().parent / "registry/local_paths.json").read_text())["training_python"])')"
fi
[[ -x "$STUDY_PYTHON" ]] || { echo 'STUDY_PYTHON must be executable.' >&2; exit 2; }
export STUDY_PYTHON PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PATH="$(dirname -- "$STUDY_PYTHON"):${PATH:-/usr/bin:/bin}"
exec "$STUDY_PYTHON" -u scripts/resume_h100_full_eval.py "$@"
