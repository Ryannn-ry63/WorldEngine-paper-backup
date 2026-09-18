#!/usr/bin/env bash
set -euo pipefail
# Simulator crashes must not fill the shared output volume with core dumps.
ulimit -c 0
PAPER_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PAPER_CODE_DIR"
if [[ -z "${STUDY_PYTHON:-}" ]]; then
    # Batch workers may have no Python or activated conda environment in PATH.
    # Bootstrap from the shared project environment before reading the registry.
    PAPER_BOOTSTRAP_PYTHON="$PAPER_CODE_DIR/../../../miniconda3/envs/algengine/bin/python"
    if [[ ! -x "$PAPER_BOOTSTRAP_PYTHON" ]]; then
        echo "Shared Python is missing: $PAPER_BOOTSTRAP_PYTHON. Set STUDY_PYTHON to an executable Python path." >&2
        exit 2
    fi
    STUDY_PYTHON="$("$PAPER_BOOTSTRAP_PYTHON" -c 'import json; from pathlib import Path; print(json.loads((Path.cwd().parent / "registry/local_paths.json").read_text())["training_python"])')"
fi
if [[ ! -x "$STUDY_PYTHON" ]]; then
    echo "Configured STUDY_PYTHON is not executable: $STUDY_PYTHON" >&2
    exit 2
fi
export STUDY_PYTHON
export PATH="$(dirname -- "$STUDY_PYTHON"):${PATH:-/usr/bin:/bin}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec "$STUDY_PYTHON" -m paper_study "$@"
