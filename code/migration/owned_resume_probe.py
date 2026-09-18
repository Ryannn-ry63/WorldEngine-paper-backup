"""Bounded migration diagnostic. Formal queues and their admission gate stay unchanged."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

M = Path(__file__).resolve().parent
C = M.parent
sys.path.insert(0, str(C))
import owned_runtime as owned

A = '20260917_paper_A_v3'
POLICY = 'D4_seed0'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def prepare():
    cert = M / 'priority_copy_20260918' / (A + '.accepted.json')
    accepted = read(cert)
    if accepted['status'] != 'PASS_UNIT_BYTES_NOT_RESUME' or accepted['complete_run'] != A:
        raise RuntimeError('Complete copied A directory must first pass checksum verification')
    if read(M / 'owned_environment_ready.json')['status'] != 'PASS_ENVIRONMENT_FILES':
        raise RuntimeError('Owned environment verification is incomplete')
    source = owned.OUT / 'runs' / A
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + str(os.getpid())
    name = 'owned_resume_probe_' + stamp
    bundle = owned.O / 'runtime/resume_probes' / name
    run = bundle / 'run'
    runtime = run / 'runtime'
    runtime.mkdir(parents=True, exist_ok=False)
    contract = read(source / 'run_contract.json')
    original_hash = sha(source / 'run_contract.json')
    inventory = contract['implementation']
    for relative, expected in inventory.items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid runtime inventory path')
        src = source / 'runtime' / rel
        dst = runtime / rel
        if sha(src) != expected:
            raise RuntimeError('Copied runtime differs: ' + relative)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha(dst) != expected:
            raise RuntimeError('Probe runtime copy differs: ' + relative)
    data = source / 'runtime/data'
    if not data.is_symlink():
        raise RuntimeError('Expected public data symlink')
    (runtime / 'data').symlink_to(os.readlink(data), target_is_directory=True)
    shutil.copy2(source / 'inputs.json', run / 'inputs.json')
    logical = owned.W / 'runs' / name
    contract.update(runtime=str(logical / 'runtime'), workers=1, gpu_count=1,
                    ray_cpu_threads=8, simulation_cpu_threads=8,
                    diagnostic_only=True, formal_table_ready=False,
                    migration_probe_source=dict(path=str(source / 'run_contract.json'), sha256=original_hash),
                    scheduling='Single GPU migration diagnostic: eight bridge scenes, NR then R')
    write(run / 'run_contract.json', contract)
    manifest = dict(name=name, bundle=str(bundle), run=str(run), logical_run=str(logical),
                    source_run=A, policy=POLICY, source_contract_sha256=original_hash,
                    source_inputs_sha256=sha(source / 'inputs.json'),
                    probe_contract_sha256=sha(run / 'run_contract.json'),
                    copied_runtime_files=len(inventory), certificate=str(cert), certificate_sha256=sha(cert),
                    diagnostic_only=True, formal_resume_authorized=False, created=time.time())
    write(bundle / 'probe.json', manifest)
    print('PROBE_DIRECTORY=' + str(bundle), flush=True)
    return bundle


def launch(bundle, mode, device):
    manifest = read(bundle / 'probe.json')
    pairs = owned.mappings() + [(Path(manifest['run']), manifest['logical_run'])]
    for backing, logical in pairs:
        if not backing.exists():
            raise FileNotFoundError(str(backing))
    # Preserve original path identity within this process only. The new run is
    # outside the background copier's destination tree and contains no hardlinks.
    write(owned.O / 'runtime/path_map.json', [dict(backing=str(b), logical=l) for b, l in pairs])
    exe = owned.O / 'tools/proot/usr/bin/proot'
    cmd = [str(exe), '-v', '-1', '-w', str(C)]
    for backing, logical in pairs:
        cmd += ['-b', str(backing) + ':' + logical + '!']
    cmd += [str(owned.ALG / 'bin/python'), str(Path(__file__).resolve()), '_child',
            '--bundle', str(bundle), '--scope', mode, '--device', device]
    env = owned.environment()
    env['CUDA_VISIBLE_DEVICES'] = device if mode == 'run' else ''
    os.execve(cmd[0], cmd, env)


def child(bundle, scope, device):
    manifest = read(bundle / 'probe.json')
    logical = Path(manifest['logical_run'])
    physical = Path(manifest['run'])
    if logical.stat().st_ino != physical.stat().st_ino or logical.stat().st_dev != physical.stat().st_dev:
        raise RuntimeError('Probe output mapping is not isolated')
    source = owned.OUT / 'runs' / A
    if sha(source / 'run_contract.json') != manifest['source_contract_sha256']:
        raise RuntimeError('Source contract changed')
    if sha(logical / 'run_contract.json') != manifest['probe_contract_sha256']:
        raise RuntimeError('Probe contract changed')
    result = dict(status='RUNNING', scope=scope, hostname=socket.gethostname(),
                  started=time.time(), diagnostic_only=True, formal_resume_authorized=False, stages=[])
    def stage(name):
        result['current_stage'] = name
        write(bundle / 'summary.json', result)
        print('CHECK ' + name, flush=True)
    def passed(name):
        result['stages'].append(name)
        write(bundle / 'summary.json', result)
    def interrupted(*_):
        raise KeyboardInterrupt('Migration probe interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if scope == 'run':
            from paper_study.hardware import hardware
            hardware([device])
            stage('OWNED_GPU_RUNTIME')
            subprocess.run([str(owned.ALG / 'bin/python'), str(M / 'check_owned_runtime.py'),
                            '--gpu', device], check=True, timeout=1200)
            passed('OWNED_GPU_RUNTIME')
        from paper_study import evaluation as evaluation
        from paper_study.common import read as eread, verify
        stage('REUSE_MIGRATED_OPEN_RARE')
        previous = evaluation.check_open(owned.W / 'runs' / A, POLICY, 'open_rare')
        result['reused_open_scenes'] = previous['count']
        passed('REUSE_MIGRATED_OPEN_RARE')
        stage('FROZEN_INPUTS_AND_CPU_PREFLIGHT')
        engine, run, contract, inputs, runtime = evaluation.load(logical)
        env = evaluation.environment(engine, run, contract, runtime, device, POLICY)
        if scope != 'run':
            env['CUDA_VISIBLE_DEVICES'] = ''
        # Reuses the frozen scientific implementation, with the single-GPU
        # diagnostic resource contract. No four-GPU admission is weakened.
        engine.preflight(run, contract, inputs, runtime, env)
        for mode in ('NR', 'R'):
            historic = owned.W / 'runs' / evaluation.SOURCE / 'collections' / ('bridge_' + mode + '_' + POLICY)
            accepted = eread(historic / 'accepted.json')
            if accepted['status'] != 'PASS':
                raise RuntimeError('Historical learned bridge is not accepted')
            verify(accepted['audit'])
        passed('FROZEN_INPUTS_AND_CPU_PREFLIGHT')
        if scope == 'run':
            from run_study_bridges_4h100 import cpu_capacity
            if cpu_capacity() < 10:
                raise RuntimeError('This diagnostic requires at least 10 allocated CPU cores')
            for mode in ('NR', 'R'):
                stage('NEW_BRIDGE_' + mode)
                engine.collection(run, contract, inputs, runtime, env, POLICY, 'bridge', mode, 7200)
                group = run / 'collections' / ('bridge_' + mode + '_' + POLICY)
                historic = owned.W / 'runs' / evaluation.SOURCE / 'collections' / group.name
                # Same eight scenes/noise/model: compare metrics, inputs, logits,
                # selected candidates and actions under the existing tolerances.
                engine.compare_scalar(eread(group / 'audit.json'), eread(verify(eread(historic / 'accepted.json')['audit'])))
                passed('NEW_BRIDGE_' + mode + '_HISTORICAL_PARITY')
                stage('REUSE_NEW_BRIDGE_' + mode)
                before = sha(group / 'audit.json')
                engine.collection(run, contract, inputs, runtime, env, POLICY, 'bridge', mode, 7200)
                if sha(group / 'audit.json') != before:
                    raise RuntimeError('Accepted audit changed during reuse')
                passed('REUSE_NEW_BRIDGE_' + mode)
        if sha(source / 'run_contract.json') != manifest['source_contract_sha256'] or sha(source / 'inputs.json') != manifest['source_inputs_sha256']:
            raise RuntimeError('Formal source metadata changed')
        result['status'] = 'PASS_ISOLATED_RESUME_PROBE' if scope == 'run' else 'PASS_OLD_RESULT_REUSE_AND_CPU_PREFLIGHT'
        result['single_gpu_closed_loop_verified'] = scope == 'run'
        result['four_gpu_verified'] = False
        result['training_optimizer_resume_verified'] = False
    except BaseException as exc:
        result.update(status='FAILED_PROBE', error=repr(exc))
        raise
    finally:
        result['finished'] = time.time()
        write(bundle / 'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'audit', 'run', '_child'])
    p.add_argument('--device', default='0')
    p.add_argument('--bundle', type=Path)
    p.add_argument('--scope', choices=['audit', 'run'])
    args = p.parse_args()
    if not args.device.isdigit():
        p.error('One numeric GPU ID is required')
    if args.mode == '_child':
        if os.environ.get('WORLDENGINE_OWNED_RUNTIME') != '1' or args.bundle is None or args.scope is None:
            p.error('Internal invocation requires the owned runtime and a prepared bundle')
        return child(args.bundle, args.scope, args.device)
    bundle = prepare()
    if args.mode != 'prepare':
        launch(bundle, args.mode, args.device)


if __name__ == '__main__':
    main()
