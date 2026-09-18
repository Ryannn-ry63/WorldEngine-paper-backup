#!/usr/bin/env bash
# Four-H100 full/common closed-loop queue; no training or open-loop jobs.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# != 0 )); then
    echo 'Usage: bash run_h100_full_eval.sh (uses GPUs 0,1,2,3)' >&2
    exit 2
fi
FORMAL_RUN=20260916_factorial_bridges_4h100_v1
for FORMAL_GPU in 0 1 2 3; do
    FORMAL_NAME="$(nvidia-smi --id="$FORMAL_GPU" --query-gpu=name --format=csv,noheader)"
    if [[ "$FORMAL_NAME" != *H100* ]]; then
        echo "Requires four H100 GPUs; device $FORMAL_GPU is $FORMAL_NAME" >&2
        exit 2
    fi
done
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != 0,1,2,3 ]]; then
    echo 'CUDA_VISIBLE_DEVICES must be unset or 0,1,2,3 for this launcher.' >&2
    exit 2
fi
if [[ -z "${STUDY_PYTHON:-}" ]]; then
    STUDY_PYTHON="$(python -c 'import json; from pathlib import Path; print(json.loads((Path.cwd().parent / "registry/local_paths.json").read_text())["training_python"])')"
fi
export STUDY_PYTHON PYTHONDONTWRITEBYTECODE=1
exec 9>"../runs/${FORMAL_RUN}/queue.lock"
flock -n 9 || { echo 'Another queue is active; stop it before restarting.' >&2; exit 2; }
exec > >(tee -a "../runs/${FORMAL_RUN}.formal_closed.log") 2>&1
# Preserve the misconfigured partial run only after its driver has released its lock.
"$STUDY_PYTHON" - "$FORMAL_RUN" <<'PY'
import datetime,fcntl,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
from evaluation_status import status
from selector_study.common import artifact, read, write
run=Path.cwd().parent/'runs'/sys.argv[1]
with (run/'active.lock').open('a') as lock:
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('The previous evaluation is still running. Press Ctrl+C in its H100 terminal, wait for cleanup, then rerun this launcher.')
    current=status(sys.argv[1])
    if current['groups']['bridge']['accepted']!=26:
        raise SystemExit('All 26 accepted bridge conditions are required')
    contract=read(run/'run_contract.json')
    if contract.get('workers')!=4 or contract['ray_cpu_threads']<4*contract['simulation_cpu_threads']:
        raise SystemExit('Contract cannot schedule four concurrent simulations')
    partial=run/'collections/full_NR_scalar_v3'
    if (partial/'collection.json').exists():
        co=read(partial/'collection.json')
        if co['workers']!=4:
            if (co['workers']!=1 or co['id']!='full_NR_scalar_v3'
                    or co['contract']!=artifact(run/'run_contract.json')
                    or (partial/'accepted.json').exists() or (partial/'audit.json').exists()):
                raise SystemExit('Unexpected existing collection; preserve it for manual review')
            stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            archive=run/'archive/incomplete_single_gpu'/('full_NR_scalar_v3_'+stamp)
            archive.parent.mkdir(parents=True,exist_ok=True)
            partial.rename(archive)
            write(archive/'archive_reason.json',dict(reason='Stopped incomplete single-worker attempt; restart with explicit four GPUs',original=str(partial),workers=1))
            print('Preserved incomplete single-GPU outputs at '+str(archive),flush=True)
print('Four-GPU closed-loop queue: 13 policies x 2 cohorts x NR/R = 52 conditions.',flush=True)
PY
for FORMAL_COHORT in full common; do
    for FORMAL_POLICY in scalar_v3 D{1..4}_seed{0..2}; do
        for FORMAL_MODE in NR R; do
            echo "EVALUATING ${FORMAL_COHORT} ${FORMAL_POLICY} ${FORMAL_MODE} on GPUs 0,1,2,3"
            bash run_evaluation.sh collect --run "$FORMAL_RUN" --policy "$FORMAL_POLICY" --cohort "$FORMAL_COHORT" --mode "$FORMAL_MODE" --devices 0,1,2,3 --timeout 10800
        done
    done
done
bash run_evaluation.sh status --run "$FORMAL_RUN"
