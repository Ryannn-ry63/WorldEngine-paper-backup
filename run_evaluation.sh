#!/usr/bin/env bash
set -euo pipefail
EVAL_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$EVAL_CODE_DIR"
if [[ -z "${STUDY_PYTHON:-}" ]]; then
    STUDY_PYTHON="$(python -c 'import json; from pathlib import Path; print(json.loads((Path.cwd().parent / "registry/local_paths.json").read_text())["training_python"])')"
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
if [[ "${1:-}" == prepare ]]; then
    shift
    exec "$STUDY_PYTHON" scripts/prepare_study_evaluation.py "$@"
fi
if [[ "${1:-}" == status ]]; then
    shift
    exec "$STUDY_PYTHON" scripts/evaluation_status.py "$@"
fi
exec "$STUDY_PYTHON" scripts/run_study_evaluation.py "$@"
