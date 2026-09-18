#!/usr/bin/env bash
set -euo pipefail
FOUR_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$FOUR_CODE_DIR"
if (( $# > 2 )); then
    echo 'Usage: bash run_h100_bridges_4gpu.sh [new_run_id] [four_physical_gpu_ids]' >&2
    exit 2
fi
FOUR_RUN="${1:-20260916_factorial_bridges_4h100_v1}"
FOUR_DEVICES="${2:-0,1,2,3}"
if [[ ! "$FOUR_RUN" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Invalid run ID' >&2
    exit 2
fi
if [[ -z "${STUDY_PYTHON:-}" ]]; then
    STUDY_PYTHON="$(python -c 'import json; from pathlib import Path; print(json.loads((Path.cwd().parent / "registry/local_paths.json").read_text())["training_python"])')"
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$STUDY_PYTHON" -u scripts/run_study_bridges_4h100.py --run "$FOUR_RUN" --devices "$FOUR_DEVICES" \
    2>&1 | tee -a "../runs/${FOUR_RUN}.launch.log"
