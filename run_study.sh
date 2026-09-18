#!/usr/bin/env bash
set -euo pipefail
STUDY_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$STUDY_CODE_DIR"
if [[ -z "${STUDY_PYTHON:-}" ]]; then
    STUDY_PYTHON="$(python -c 'import json; from pathlib import Path; print(json.loads((Path.cwd().parent / "registry/local_paths.json").read_text())["training_python"])')"
fi
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec "$STUDY_PYTHON" -m selector_study "$@"
