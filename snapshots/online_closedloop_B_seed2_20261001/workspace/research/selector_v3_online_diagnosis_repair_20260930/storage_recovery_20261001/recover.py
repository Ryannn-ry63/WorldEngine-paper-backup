#!/usr/bin/env python3
"""Infrastructure-only recovery: committed B_s2 + exclusive, local-scratch eval."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
REPAIR = ROOT.parent
OLD = REPAIR.with_name('selector_v3_online_diagnosis_20260930')
REG = OLD / 'registry'
GIB = 2 ** 30
TRANSIENT = {'sensor_blobs', 'frames', 'merged_ann_files', 'distributed_collect'}


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value, immutable=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if immutable:
            os.link(tmp, p)
        else:
            os.replace(tmp, p)
        fd = os.open(str(p.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def directory_lock(path):
    """Atomic namespace lock, independent of per-client flock mount behavior.

    Never steal another host's lock based on its PID or elapsed time. A killed
    owner leaves evidence; explicit inspection is required before removal.
    """
    lockdir = Path(str(path) + '.mkdir')
    lockdir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lockdir.mkdir()
    except FileExistsError:
        raise BlockingIOError('Already claimed: ' + str(lockdir))
    token = uuid.uuid4().hex
    owner = dict(token=token, host=socket.gethostname(), pid=os.getpid(), started=time.time())
    try:
        write(lockdir / 'owner.json', owner, immutable=True)
        yield owner
    finally:
        p = lockdir / 'owner.json'
        if p.exists() and read(p).get('token') == token:
            p.unlink()
            lockdir.rmdir()


def verify_entry():
    manifest = read(ROOT / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Recovery entry changed: ' + name)
    for name, expected in manifest['dependencies'].items():
        if sha(name) != expected:
            raise RuntimeError('Frozen dependency changed: ' + name)


def setup():
    sys.path.insert(0, str(REPAIR))
    import repair_run as runner
    import experiment_common as common
    import evaluation_artifacts as artifacts
    import summarize_experiment as summary
    for module in (runner, common, artifacts, summary):
        module.lock = directory_lock
        module.write = write
    verify_entry()
    runner.registration()
    if (REG / 'correctness_failure.json').exists():
        raise RuntimeError('Correctness stop remains active')
    if not (REPAIR / 'evidence/recovery_release.json').exists():
        raise RuntimeError('Validated repair release is missing')
    return runner, artifacts, summary


def budget_window(previous_wall, now):
    remaining = 43200.0 - previous_wall
    if remaining <= 0:
        raise RuntimeError('Original cumulative training budget exhausted')
    return dict(started_unix=now, deadline_unix=now + remaining + 600,
                wall_seconds=remaining + 600, cumulative_training_cap_seconds=43200,
                already_used_training_seconds=previous_wall,
                scope='train_B_s2_infrastructure_recovery_only')


def audit_boundary(runner):
    from innovation3.ddp_session_boundary import validate_round, receipt
    from innovation3.live_learning import state_digest
    import torch
    p = REG / 'jobs/train_B_s2/training.progress.json'
    before = sha(p)
    state = read(p)
    spec = state['contract']
    if spec['episodes'] != 904 or spec['seed'] != 2 or spec['world_size'] != 8 or spec['online_update_timing'] != 'after_rollout8':
        raise ValueError('Unexpected training contract')
    if spec['settings_sha256'] != sha(runner.SETTINGS) or spec['manifest_sha256'] != sha(REG / 'scene_manifest.json'):
        raise ValueError('Training inputs changed')
    if len(state['rounds']) * 8 != state['completed']:
        raise ValueError('Incomplete committed round')
    for i, records in enumerate(state['rounds']):
        validate_round(records, spec, i)
    hashes = []
    i = len(state['rounds']) - 1
    for rank, row in enumerate(state['rounds'][-1]):
        if sha(row['report']) != row['report_sha256'] or sha(row['checkpoint']['path']) != row['checkpoint']['sha256']:
            raise ValueError('Committed report/checkpoint changed')
        previous = state['rounds'][i - 1][rank] if i else None
        if receipt(row['report'], spec, i, rank, previous) != row:
            raise ValueError('Committed receipt mismatch')
        data = torch.load(row['checkpoint']['path'], map_location='cpu')
        if data['source_sha256'] != spec['source_sha256'] or data['learner']['attempts'] != state['completed']:
            raise ValueError('Checkpoint provenance mismatch')
        hashes.append({k: state_digest(data['learner'][k]) for k in ('selector', 'reference', 'optimizer')})
    if any(h != hashes[0] for h in hashes) or sha(p) != before:
        raise ValueError('All-rank state mismatch or concurrent progress change')
    return dict(status='PASS_COMMITTED_BOUNDARY', completed=state['completed'], target=904,
                remaining_episodes=904-state['completed'], progress_sha256=before,
                training_wall_seconds=state['training_wall_seconds'], final_round_state=hashes[0],
                last_checkpoints=[r['checkpoint'] for r in state['rounds'][-1]])


def load_hotfix():
    path = REPAIR / 'hotfix_logit_tolerance_bs2_20261001/resume_train_h100.py'
    spec = importlib.util.spec_from_file_location('bs2_registered_hotfix', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_hotfix()
    return module


def train(runner, devices, check_only):
    hotfix = load_hotfix()
    audit = audit_boundary(runner)
    print(json.dumps(audit), flush=True)
    if audit['completed'] == 904:
        print('ALREADY_COMPLETE: train_B_s2', flush=True)
        return 0
    if not check_only:
        with directory_lock(ROOT / 'training_window.lock'):
            path = ROOT / 'training_window.json'
            old_clock_path = REG / 'campaign_clock.json'
            if path.exists():
                clock = read(path)
                if time.time() >= clock['deadline_unix']:
                    raise RuntimeError('Recovery window expired; inspect before extending again')
            else:
                old_clock = read(old_clock_path)
                clock = budget_window(audit['training_wall_seconds'], time.time())
                clock.update(original_clock=old_clock, original_clock_sha256=sha(old_clock_path),
                             boundary_before=audit, recovery_manifest_sha256=sha(ROOT / 'manifest.json'))
                write(path, clock, immutable=True)
            # Original children read this exact clock path at committed boundaries.
            # Preserve original clock in the immutable recovery receipt; do not
            # edit runtime settings, source contracts, episode/update budgets.
            active = read(old_clock_path)
            if active.get('recovery_window') != str(path):
                if sha(old_clock_path) != clock['original_clock_sha256']:
                    raise RuntimeError('Campaign clock changed by another process')
                updated = dict(active, deadline_unix=clock['deadline_unix'],
                               wall_seconds=clock['deadline_unix'] - active['started_unix'],
                               recovery_window=str(path), recovery_reason='disk_quota_interruption',
                               original_wall_seconds=active['wall_seconds'],
                               cumulative_training_cap_seconds=43200)
                write(old_clock_path, updated)
            runner.campaign_clock = lambda: read(path)
    sys.argv = ['resume_train_h100.py', 'train_B_s2', '--devices', devices] + (['--check-only'] if check_only else [])
    return hotfix.main()


def scratch_base():
    p = Path(os.environ.get('EVAL_SCRATCH_ROOT', '/tmp/worldengine_eval_recovery')).absolute()
    p.mkdir(parents=True, exist_ok=True)
    if p.is_symlink() or p.stat().st_dev == REG.stat().st_dev:
        raise RuntimeError('EVAL_SCRATCH_ROOT must be on node-local disk, not the shared quota filesystem')
    if shutil.disk_usage(p).free < 80 * GIB:
        raise RuntimeError('Need at least 80 GiB free local scratch')
    if shutil.disk_usage(REG).free < 64 * GIB:
        raise RuntimeError('Need at least 64 GiB free shared storage before starting evaluation')
    return p


def archive_tree(src, dst, failures_only=False):
    """Retain metrics, plans, metadata and logs; disposable images/IPC stay local."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return []
    copied = []
    for folder, dirs, files in os.walk(src, followlinks=False):
        dirs[:] = [d for d in dirs if d not in TRANSIENT and not (Path(folder) / d).is_symlink()]
        for name in files:
            p = Path(folder) / name
            if p.is_symlink():
                continue
            if failures_only and p.suffix not in ('.log', '.json', '.yaml', '.yml', '.csv', '.txt'):
                continue
            rel = p.relative_to(src)
            q = dst / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            # Metrics, metadata and executed plans are evidence; byte verify.
            h = sha(p)
            if sha(q) != h:
                raise RuntimeError('Evidence copy hash mismatch: ' + str(rel))
            copied.append(dict(path=str(q), sha256=h, size=p.stat().st_size))
    return copied


def scratch_evaluation(runner):
    original_eval = runner.evaluation_job
    original_processes = runner.run_processes

    def wrapped(job, folder, devices, deadline, dry=False):
        local = scratch_base() / (job['id'] + '_' + uuid.uuid4().hex)
        local.mkdir()
        shared = Path(folder) / 'evaluation'
        local_out = local / 'evaluation'
        write(Path(folder) / 'scratch.json', dict(host=socket.gethostname(), path=str(local),
              transient_directories=sorted(TRANSIENT), metrics_and_plans_archived=True), immutable=True)

        def relocated(commands, process_folder, process_deadline, **kwargs):
            converted = []
            for argv, cwd, env in commands:
                argv = list(map(str, argv))
                if '--output' not in argv or Path(argv[argv.index('--output')+1]) != shared:
                    raise ValueError('Unexpected evaluation command; no output relocation performed')
                argv[argv.index('--output')+1] = str(local_out)
                converted.append((argv, cwd, env))
            try:
                original_processes(converted, process_folder, process_deadline, **kwargs)
            except BaseException:
                archive_tree(local_out, shared, failures_only=True)
                raise
            files = archive_tree(local_out, shared)
            files += archive_tree(local_out.with_name('evaluation_inputs'), shared.with_name('evaluation_inputs'))
            write(Path(folder) / 'evidence_archive.json', dict(files=files, omitted=sorted(TRANSIENT)), immutable=True)

        runner.run_processes = relocated
        try:
            result = original_eval(job, folder, devices, deadline, dry)
            if result['status'] == 'COMPLETE':
                # Original exact-coverage acceptance runs on the shared copy.
                shutil.rmtree(local)
                write(Path(folder) / 'scratch_released.json', dict(path=str(local), after='PASS_EXACT_COVERAGE'))
            return result
        finally:
            runner.run_processes = original_processes

    runner.evaluation_job = wrapped


def retry_allowed(status, window):
    if status.get('status') == 'COMPLETE':
        return False
    # Never guess that a remotely RUNNING process is dead.
    if status.get('status') == 'RUNNING':
        return False
    return not (status.get('status') == 'FAILED' and status.get('infrastructure_recovery', {}).get('window') == window)


def evaluation(runner, summary, devices, check_only, session, idle_seconds):
    spec = runner.registration()
    jobs = [j for j in spec['jobs'] if j['kind'] == 'evaluation']
    window = ROOT / 'evaluation_sessions' / session / 'clock.json'
    key = str(window)
    statuses = {j['id']: read(REG / 'jobs' / j['id'] / 'status.json') if (REG / 'jobs' / j['id'] / 'status.json').exists() else {} for j in jobs}
    ready = [j['id'] for j in jobs if retry_allowed(statuses[j['id']], key) and runner.ready(j)]
    print(json.dumps(dict(event='RECOVERY_INVENTORY', ready=ready,
                         complete=sum(s.get('status') == 'COMPLETE' for s in statuses.values()),
                         scratch=str(scratch_base()))), flush=True)
    if check_only:
        runner.preflight(jobs[0], devices)
        print('CHECK_ONLY: no evaluation clock or GPU jobs started.', flush=True)
        return 0
    if not window.exists():
        now = time.time()
        try:
            write(window, dict(scope='evaluation_only', started_unix=now, deadline_unix=now+86400,
                               wall_seconds=86400, experiment_sha256=sha(REG / 'experiment.json'),
                               recovery_manifest_sha256=sha(ROOT / 'manifest.json')), immutable=True)
        except FileExistsError:
            pass  # atomic link chose another node's fully written window
    clock = read(window)
    if clock['experiment_sha256'] != sha(REG / 'experiment.json') or clock['recovery_manifest_sha256'] != sha(ROOT / 'manifest.json'):
        raise RuntimeError('Evaluation session contract changed')
    if time.time() >= clock['deadline_unix']:
        raise RuntimeError('Evaluation window expired; use a new EVAL_SESSION after inspection')
    runner.campaign_clock = lambda: read(window)
    old_summary_read = summary.read
    summary.read = lambda p: read(window) if Path(p) == REG / 'campaign_clock.json' else old_summary_read(p)
    original_write = runner.write
    def annotated(p, value, **kwargs):
        if Path(p).name == 'status.json' and Path(p).parent.parent == REG / 'jobs':
            value = dict(value, infrastructure_recovery=dict(window=key, manifest_sha256=sha(ROOT / 'manifest.json')))
        return original_write(p, value, **kwargs)
    runner.write = annotated
    scratch_evaluation(runner)
    idle = time.monotonic()
    while time.time() < clock['deadline_unix']:
        if (REG / 'correctness_failure.json').exists():
            raise RuntimeError('Correctness stop blocks evaluation')
        try:
            summary.summarize()
        except BlockingIOError:
            pass
        ran = False
        for job in jobs:
            status_path = REG / 'jobs' / job['id'] / 'status.json'
            s = read(status_path) if status_path.exists() else {}
            if not retry_allowed(s, key) or not runner.ready(job):
                continue
            try:
                # Claim before reading status again. This lock covers the whole
                # job; runner/exports/preparation also use namespace locks.
                with directory_lock(REG / 'jobs' / job['id'] / 'recovery_claim.lock'):
                    s = read(status_path) if status_path.exists() else {}
                    if not retry_allowed(s, key) or not runner.ready(job):
                        continue
                    attempt_note = ROOT / 'evaluation_sessions' / session / 'before_retry' / (job['id'] + '_' + uuid.uuid4().hex + '.json')
                    write(attempt_note, dict(previous_status=s, job=job), immutable=True)
                    print('CLAIMED: ' + job['id'], flush=True)
                    # Explicit retry: FAILED jobs from the previous window are
                    # eligible. A new failure is never silently loop-retried.
                    runner.run(job, devices, wait=True)
                    ran = True
            except BlockingIOError:
                continue
            except Exception as exc:
                print('RECOVERY_JOB_FAILED: ' + job['id'] + ' ' + repr(exc), flush=True)
                raise
            if ran:
                idle = time.monotonic()
                break
        if not ran:
            failures = [j['id'] for j in jobs if (REG/'jobs'/j['id']/'status.json').exists()
                        and (lambda s: s.get('status')=='FAILED' and s.get('infrastructure_recovery',{}).get('window')==key)(read(REG/'jobs'/j['id']/'status.json'))]
            if failures:
                raise RuntimeError('This recovery window has failures: ' + repr(failures))
            if time.monotonic() - idle >= idle_seconds:
                print('QUEUE_IDLE_EXIT: no ready unclaimed work; relaunch when B_s2 p6 or selection is ready.', flush=True)
                return 0
            time.sleep(10)
    raise TimeoutError('Evaluation window ended')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train_B_s2', 'eval'))
    p.add_argument('--devices', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7'))
    p.add_argument('--session', default=os.environ.get('EVAL_SESSION', 'disk_recovery_20261001'))
    p.add_argument('--idle-seconds', type=float, default=120)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    import re
    ids = a.devices.split(',')
    if len(ids)!=8 or len(set(ids))!=8 or not all(i.isdigit() for i in ids):
        p.error('Exactly eight distinct devices are required')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.session) or a.idle_seconds < 0:
        p.error('Invalid session or idle timeout')
    def stop(signum, frame):
        raise InterruptedError('Recovery stopped by signal ' + str(signum))
    signal.signal(signal.SIGTERM, stop)
    runner, artifacts, summary = setup()
    if a.mode == 'train_B_s2':
        with directory_lock(REG / 'jobs/train_B_s2/recovery_claim.lock'):
            return train(runner, a.devices, a.check_only)
    return evaluation(runner, summary, a.devices, a.check_only, a.session, a.idle_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
