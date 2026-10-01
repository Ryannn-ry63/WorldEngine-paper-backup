#!/usr/bin/env python3
"""Evaluation-only scheduling window; reuse registered jobs, locks and acceptance."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
REPAIR = ROOT.parent
sys.path.insert(0, str(REPAIR))
import repair_run as runner
import repair_queue as queue
import summarize_experiment as summary_module


def verify_entry():
    manifest = runner.read(ROOT / 'manifest.json')
    for name, expected in manifest['files'].items():
        if runner.sha(ROOT / name) != expected:
            raise RuntimeError('Evaluation entry changed: ' + name)


def evaluation_clock(session):
    path = ROOT / 'sessions' / session / 'clock.json'
    if not path.exists():
        now = time.time()
        value = dict(scope='evaluation_only', session=session,
                     started_unix=now, deadline_unix=now + 86400, wall_seconds=86400,
                     original_training_clock=runner.read(runner.REGISTRY / 'campaign_clock.json'),
                     original_training_clock_sha256=runner.sha(runner.REGISTRY / 'campaign_clock.json'),
                     experiment_sha256=runner.sha(runner.REGISTRY / 'experiment.json'),
                     evaluation_entry_sha256=runner.sha(ROOT / 'manifest.json'))
        try:
            runner.write(path, value, immutable=True)
        except FileExistsError:
            pass
    value = runner.read(path)
    if (value.get('scope') != 'evaluation_only' or
            value['experiment_sha256'] != runner.sha(runner.REGISTRY / 'experiment.json') or
            value['evaluation_entry_sha256'] != runner.sha(ROOT / 'manifest.json')):
        raise RuntimeError('Evaluation window registration differs')
    return value


def inventory(spec):
    rows = []
    for job in spec['jobs']:
        if job['kind'] != 'evaluation':
            continue
        status_path = runner.REGISTRY / 'jobs' / job['id'] / 'status.json'
        state = runner.read(status_path).get('status') if status_path.exists() else 'NOT_STARTED'
        rows.append(dict(id=job['id'], dataset=job['dataset'], status=state,
                         ready=state not in ('COMPLETE', 'FAILED') and runner.ready(job)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devices', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7'))
    p.add_argument('--session', default='posttrain_20261001')
    p.add_argument('--idle-seconds', type=float, default=120)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', a.session):
        p.error('Invalid session ID')
    if a.idle_seconds < 0:
        p.error('idle-seconds must be nonnegative')
    devices = a.devices.split(',')
    if len(devices) != 8 or len(set(devices)) != 8 or not all(x.isdigit() for x in devices):
        p.error('Exactly eight distinct GPU IDs required')
    verify_entry()
    spec = runner.registration()
    if not (REPAIR / 'evidence/recovery_release.json').exists():
        raise RuntimeError('Validated repair release is missing')
    if (runner.REGISTRY / 'correctness_failure.json').exists():
        raise RuntimeError('Correctness stop blocks evaluation')
    rows = inventory(spec)
    print(json.dumps(dict(event='EVALUATION_INVENTORY', jobs=rows,
                          ready=sum(r['ready'] for r in rows),
                          complete=sum(r['status']=='COMPLETE' for r in rows)), ensure_ascii=False), flush=True)
    if a.check_only:
        print('CHECK_ONLY: registration and job readiness passed; no clock, export or GPU job started.', flush=True)
        return 0

    clock_path = ROOT / 'sessions' / a.session / 'clock.json'
    clock = evaluation_clock(a.session)
    print(json.dumps(dict(event='EVALUATION_WINDOW', path=str(clock_path), **clock)), flush=True)
    if time.time() >= clock['deadline_unix']:
        raise RuntimeError('This evaluation window expired; explicitly choose a new EVAL_SESSION')
    original_registration = runner.registration
    original_write = runner.write
    original_summary_read = summary_module.read

    def registered():
        verify_entry()
        return original_registration()

    def clock_for_summary(path):
        if Path(path) == runner.REGISTRY / 'campaign_clock.json':
            return evaluation_clock(a.session)
        return original_summary_read(path)

    def annotated_write(path, value, **kwargs):
        path = Path(path)
        if path.name == 'status.json' and path.parent.parent == runner.REGISTRY / 'jobs':
            value = dict(value, evaluation_window=dict(path=str(clock_path), sha256=runner.sha(clock_path)))
        return original_write(path, value, **kwargs)

    # These are process-local bindings. No training file, deadline or rule is edited.
    runner.registration = queue.registration = registered
    runner.campaign_clock = queue.campaign_clock = lambda: evaluation_clock(a.session)
    runner.write = annotated_write
    summary_module.read = clock_for_summary
    if a.summary:
        summary_module.summarize()
        return 0
    sys.argv = ['repair_queue.py', '--group', 'evaluation', '--devices', a.devices,
                '--idle-seconds', str(a.idle_seconds)] + (['--once'] if a.once else [])
    queue.main()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
