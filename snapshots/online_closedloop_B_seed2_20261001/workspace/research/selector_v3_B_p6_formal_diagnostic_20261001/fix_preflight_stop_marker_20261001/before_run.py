#!/usr/bin/env python3
"""Independent fixed-checkpoint four-table diagnosis; never writes old selection."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent
OLD = ROOT.with_name('selector_v3_online_diagnosis_20260930')
RECOVERY = ROOT.with_name('selector_v3_online_diagnosis_repair_20260930') / 'storage_recovery_20261001'
sys.path.insert(0, str(RECOVERY))
import recover

read, write, sha, lock = recover.read, recover.write, recover.sha, recover.directory_lock
MODES = ('cl_R', 'cl_NR', 'ol_rare', 'ol_navtest')
POLICIES = ('parent', 'train_B_s0_p6', 'train_B_s1_p6', 'train_B_s2_p6')


def jobs(scope):
    modes = MODES if scope == 'all' else MODES[:2] if scope == 'closedloop' else (scope,)
    return [dict(id=f'{mode}__{policy}', kind='evaluation', dataset='formal',
                 mode=mode, policy=policy, interval_seconds=0.5, record_plans=True)
            for policy in POLICIES for mode in modes]


def verify_artifact(artifact):
    for field, digest in [('path', 'sha256'), ('selector', 'selector_sha256'),
                          ('marker', 'marker_sha256'), ('checkpoint', 'checkpoint_sha256')]:
        if field in artifact and sha(artifact[field]) != artifact[digest]:
            raise ValueError('Changed model artifact: ' + artifact[field])
    return artifact


def setup():
    manifest = read(ROOT / 'manifest.json')
    for path, digest in manifest['files'].items():
        if sha(path) != digest:
            raise ValueError('Frozen diagnostic input changed: ' + path)
    runner, artifacts, _ = recover.setup()
    prepared = read(OLD / 'registry/development/prepared.json')
    artifacts.verify_prepared(prepared)
    # All models already exist. Avoid old registry preparation/export locks when
    # multiple table workers start simultaneously. No new model is materialized.
    runner.prepare_development = lambda: prepared
    runner.materialize_policy = lambda policy: verify_artifact(manifest['policies'][policy])
    for artifact in manifest['policies'].values():
        verify_artifact(artifact)
    contract = read(OLD / 'registry/fulltable_data_contract.json')
    if sha(contract['scenarios']) != contract['scenarios_sha256']:
        raise ValueError('Formal scenarios changed')
    for mode, tokens in contract['expected_tokens'].items():
        if len(tokens) != manifest['expected_counts'][mode] or len(set(tokens)) != len(tokens):
            raise ValueError('Invalid formal token set: ' + mode)
    return runner, artifacts, manifest


def validate_complete(status, artifacts):
    if sha(status['result']) != status['result_sha256']:
        raise ValueError('Accepted result changed')
    result = read(status['result'])
    for key, path in result['files'].items():
        if sha(path) != result['files_sha256'][key]:
            raise ValueError('Accepted metrics changed: ' + path)
    contract = read(OLD / 'registry/fulltable_data_contract.json')
    accepted = artifacts.accept_stage(result['output'], result['job']['mode'],
                                     contract['expected_tokens'][result['job']['mode']])
    if accepted['rows'] != result['rows'] or accepted['means'] != result['means']:
        raise ValueError('Result differs from exact-coverage acceptance')
    return result


def inventory(artifacts, save=False):
    rows = []
    for job in jobs('all'):
        p = ROOT / 'jobs' / job['id'] / 'status.json'
        status = read(p) if p.exists() else dict(status='PENDING')
        row = dict(job=job['id'], mode=job['mode'], policy=job['policy'], status=status['status'])
        if status['status'] == 'COMPLETE':
            result = validate_complete(status, artifacts)
            row.update(count=result['counts'], **{k:100*v for k,v in result['means'].items()})
        rows.append(row)
    groups = []
    for mode in MODES:
        group = [r for r in rows if r['mode'] == mode and r['status'] == 'COMPLETE']
        if len(group) == 4:
            parent = next(r for r in group if r['policy'] == 'parent')
            seeds = [r for r in group if r['policy'] != 'parent']
            groups.append(dict(mode=mode, seeds=3, metrics={key:dict(
                parent=parent[key], B_mean=statistics.mean(r[key] for r in seeds),
                B_seed_sd=statistics.stdev(r[key] for r in seeds),
                delta=statistics.mean(r[key] for r in seeds)-parent[key])
                for key in artifacts.METRICS}))
    report = dict(scope='diagnostic_formal_fixed_B_p6_not_selection', rows=rows,
                  aggregate=groups, metric_scale=100, complete=sum(r['status']=='COMPLETE' for r in rows))
    if save:
        # Snapshot only; use --status again after all workers finish.
        with lock(ROOT / 'summary.lock'):
            write(ROOT / 'summary.json', report)
            fields = ['job','mode','policy','status','count',*artifacts.METRICS]
            with tempfile.NamedTemporaryFile('w', dir=ROOT, delete=False, newline='') as f:
                writer=csv.DictWriter(f, fieldnames=fields);writer.writeheader();writer.writerows(rows)
                temp=f.name
            os.replace(temp, ROOT / 'summary.csv')
    print(json.dumps(report, indent=2), flush=True)


def preflight(runner, devices):
    cfg, env = runner.configured_environment(devices)
    for key, value in cfg.items():
        if isinstance(value, str) and value.startswith('/') and not Path(value).exists():
            raise FileNotFoundError(f'{key}: {value}')
    recover.scratch_base()
    subprocess.run([cfg['algengine_python'], '-c',
        'import torch; assert torch.cuda.device_count()==8, "Need eight visible GPUs"; '
        'names=[torch.cuda.get_device_name(i) for i in range(8)]; print(names); '
        'assert all("H100" in n for n in names), "Expected 8 H100 GPUs"'],
        env=env, cwd=runner.CODE, check=True)
    runner.preflight(jobs('all')[0], devices)


def run_queue(args, runner, artifacts):
    preflight(runner, args.devices)
    recover.scratch_evaluation(runner)
    selected = jobs(args.scope)
    if args.job:
        selected = [j for j in selected if j['id'] == args.job]
        if not selected:
            raise ValueError('--job does not belong to scope')
    # Cross-node job claims protect duplicate submissions. Per-host lock protects
    # this entry point from oversubscribing the same eight GPUs.
    with lock(ROOT / 'nodes' / (socket.gethostname() + '.lock')):
        for job in selected:
            base = ROOT / 'jobs' / job['id']
            claim = lock(base / 'job.lock')
            try:
                claim.__enter__()
            except BlockingIOError:
                print('OWNED_BY_ANOTHER_WORKER:', job['id'], flush=True)
                continue
            try:
                p = base / 'status.json'
                status = read(p) if p.exists() else {}
                if status.get('status') == 'COMPLETE':
                    validate_complete(status, artifacts)
                    print('ALREADY_COMPLETE:', job['id'], flush=True)
                    continue
                if status.get('status') == 'RUNNING':
                    raise RuntimeError('Unresolved RUNNING status; inspect owning host: ' + job['id'])
                if status.get('status') == 'FAILED' and not args.retry_failed:
                    print('FAILED_REQUIRES_EXPLICIT_RETRY:', job['id'], flush=True)
                    continue
                attempt = base / ('attempt_' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8])
                attempt.mkdir(parents=True)
                status = dict(status='RUNNING', job=job, attempt=str(attempt),
                              host=socket.gethostname(), pid=os.getpid(), devices=args.devices,
                              started_unix=time.time(), manifest_sha256=sha(ROOT/'manifest.json'))
                write(p, status)
                try:
                    result=runner.evaluation_job(job, attempt, args.devices, time.time()+86400)
                    status.update(result, finished_unix=time.time())
                    write(p, status)
                    print('COMPLETE:', job['id'], flush=True)
                except BaseException as e:
                    status.update(status='FAILED', error=repr(e), finished_unix=time.time())
                    write(p, status)
                    raise
            finally:
                claim.__exit__(None, None, None)
    inventory(artifacts, save=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=('all','closedloop',*MODES), nargs='?', default='all')
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--job')
    parser.add_argument('--retry-failed', action='store_true')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--cpu-check', action='store_true')
    action.add_argument('--check-only', action='store_true')
    action.add_argument('--dry-run', action='store_true')
    action.add_argument('--status', action='store_true')
    args=parser.parse_args()
    ids=args.devices.split(',')
    if len(ids)!=8 or len(set(ids))!=8 or not all(x.isdigit() for x in ids):
        parser.error('Supply exactly eight distinct numeric devices')
    runner, artifacts, manifest=setup()
    if args.status:
        return inventory(artifacts, save=True)
    if args.cpu_check:
        print(json.dumps(dict(status='PASS_CPU_CONTRACT_CHECK', jobs=len(jobs(args.scope)),
                              expected_counts=manifest['expected_counts'], gpu_verified=False)))
        return
    if args.check_only:
        preflight(runner,args.devices)
        print('PASS_H100_PREFLIGHT: no evaluation started')
        return
    if args.dry_run:
        def dry_commands(commands, folder, deadline, **kwargs):
            for argv, cwd, env in commands:
                argv=list(map(str,argv))
                if '--dry-run' not in argv:
                    raise ValueError('Refusing to execute a real evaluation in command check')
                # The runtime command builder uses only the standard library.
                # Skip sitecustomize/CUDA bootstrap during this CPU-only check.
                subprocess.run([argv[0], '-S', *argv[1:]], cwd=cwd, env=env,
                               check=True, timeout=120)
        runner.run_processes=dry_commands
        with tempfile.TemporaryDirectory(prefix='we_formal_command_check_') as temp:
            for job in jobs(args.scope):
                folder=Path(temp)/job['id'];folder.mkdir()
                runner.evaluation_job(job,folder,args.devices,time.time()+86400,dry=True)
        print('PASS_DRY_RUN: no GPU workers or registry jobs started')
        return
    def stop(signum, frame):
        raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    run_queue(args,runner,artifacts)


if __name__=='__main__':
    main()
