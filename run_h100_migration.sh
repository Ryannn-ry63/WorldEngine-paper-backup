#!/usr/bin/env bash
# A bounded single-H100 calibration. It does not launch training or the full table.
set -euo pipefail
H100_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$H100_CODE_DIR"
H100_RUN="${1:-20260916_factorial_h100_migration_v1}"
H100_DEVICE="${2:-0}"
if (( $# > 2 )); then
    echo 'Usage: bash run_h100_migration.sh [new_run_id] [one_physical_gpu_id]' >&2
    exit 2
fi
if [[ ! "$H100_RUN" =~ ^[A-Za-z0-9_-]+$ ]] || [[ ! "$H100_DEVICE" =~ ^[0-9]+$ ]]; then
    echo 'Provide a safe run ID and a single numeric physical GPU ID.' >&2
    exit 2
fi
if [[ -z "${STUDY_PYTHON:-}" ]]; then
    STUDY_PYTHON="$(python -c 'import json; from pathlib import Path; print(json.loads((Path.cwd().parent / "registry/local_paths.json").read_text())["training_python"])')"
fi
export STUDY_PYTHON PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
# Check the physical device without touching another user's processes.
H100_NAME="$(nvidia-smi --id="$H100_DEVICE" --query-gpu=name --format=csv,noheader)"
if [[ "$H100_NAME" != *H100* ]]; then
    echo "Requires the selected H100; found: $H100_NAME" >&2
    exit 2
fi
bash run_evaluation.sh prepare --run "$H100_RUN"
bash run_evaluation.sh preflight --run "$H100_RUN" --devices "$H100_DEVICE"
for H100_MODE in NR R; do
    bash run_evaluation.sh collect --run "$H100_RUN" --policy scalar_v3 \
        --cohort bridge --mode "$H100_MODE" --devices "$H100_DEVICE" --timeout 3600
done
bash run_evaluation.sh status --run "$H100_RUN"
echo 'Both fixed-V3 bridge conditions passed historical parity. Learned-model bridges and full evaluation remain separate.'
