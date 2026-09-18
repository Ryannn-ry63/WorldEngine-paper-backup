"""Four-H100 bridge queue using the already verified, frozen evaluation engine."""
import argparse
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selector_study.common import CODE, artifact, read, verify, write
import run_study_evaluation as engine
from evaluation_status import status


def cpu_capacity():
    limits = [float(len(os.sched_getaffinity(0)))]
    # cgroup v2 (including enclosing quotas exposed by this mount).
    base = Path('/sys/fs/cgroup')
    relative = next((line.split(':', 2)[2] for line in
                     Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::')), '/')
    current = base / relative.lstrip('/')
    for directory in (current, *current.parents):
        if directory != base and base not in directory.parents:
            continue
        path = directory / 'cpu.max'
        if path.exists():
            quota, period = path.read_text().split()
            if quota != 'max':
                limits.append(int(quota) / int(period))
    # Common cgroup v1 mounts.
    for folder in ('cpu', 'cpu,cpuacct'):
        parent = base / folder
        quota, period = parent/'cpu.cfs_quota_us', parent/'cpu.cfs_period_us'
        if quota.exists() and period.exists() and int(quota.read_text()) > 0:
            limits.append(int(quota.read_text()) / int(period.read_text()))
    return min(limits)


def resources(capacity):
    if capacity < 8:
        raise RuntimeError('Four GPUs require at least 8 allocated CPU cores for this queue')
    per_sim = min(8, (math.floor(capacity) - 4) // 4)
    return dict(ray_cpu_threads=4 * per_sim, simulation_cpu_threads=per_sim)


def devices(value):
    ids = value.split(',')
    if len(ids) != 4 or len(set(ids)) != 4 or not all(x.isdigit() for x in ids):
        raise ValueError('Provide four distinct physical GPU IDs, for example 0,1,2,3')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible and not set(ids).issubset(visible.split(',')):
        raise RuntimeError('Requested GPUs are outside CUDA_VISIBLE_DEVICES')
    return ids


def hardware(ids):
    result = []
    for device in ids:
        name = subprocess.check_output(['nvidia-smi', '--id='+device,
                    '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
        if 'H100' not in name:
            raise RuntimeError('Selected GPU '+device+' is not H100: '+name)
        result.append(dict(device=device, name=name))
    return result


@contextlib.contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another queue is active: '+str(path)) from None
        yield


def source_run(name):
    source, contract, inputs, runtime = engine.load_run(name)
    checked = status(name)
    for mode in ('NR', 'R'):
        key = 'bridge_'+mode+'_scalar_v3'
        if checked['groups']['bridge']['conditions'][key] != 'ACCEPTED':
            raise RuntimeError('Accepted single-H100 source bridge required: '+key)
    return source, contract, inputs, runtime


def prepare(source, contract, inputs, runtime, run, allocation):
    """Create a new resource contract; never rewrite the single-GPU run."""
    run.mkdir(parents=True, exist_ok=True)
    target = run/'runtime'
    if target.exists() or target.is_symlink():
        if target.resolve() != runtime.resolve():
            raise RuntimeError('Foreign runtime mapping')
    else:
        target.symlink_to(runtime.resolve(), target_is_directory=True)
    drivers = {}
    for name, entry in contract['driver'].items():
        original = verify(entry)
        dest = run/'archive/drivers'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.read_bytes() != original.read_bytes():
                raise RuntimeError('Frozen driver collision')
        else:
            shutil.copy2(original, dest)
        drivers[name] = artifact(dest)
    updated = dict(contract, **allocation)
    updated.update(runtime=str(target), driver=drivers, workers=4,
        source_migration_contract=artifact(source/'run_contract.json'),
        bridge_queue=artifact(Path(__file__)),
        source_scalar_acceptances={mode:artifact(source/'collections'/
            ('bridge_'+mode+'_scalar_v3')/'accepted.json') for mode in ('NR', 'R')},
        scheduling='Four concurrent scenario shards per condition; conditions run sequentially')
    write(run/'inputs.json', inputs)
    write(run/'run_contract.json', updated)
    return updated


def cuda_preflight(run, contract, inputs, runtime, device_list):
    env = engine.environment(run, contract, runtime, device_list)
    engine.preflight(run, contract, inputs, runtime, env)
    support = Path(env['SIMENGINE_ROOT'])/'scripts/diffusiondrive'
    # Reuse production probes. Bound Ray explicitly instead of using host CPU count.
    program = ('import functools,runpy,sys,ray; '
        'ray.init=functools.partial(ray.init,num_cpus='+str(contract['ray_cpu_threads'])+'); '
        'sys.path.insert(0,sys.argv[1]); sys.argv=sys.argv[2:]; '
        'runpy.run_path(sys.argv[0],run_name="__main__")')
    jobs = [([env['ALGENGINE_PYTHON'], support/'preflight_mmcv_cuda.py',
        '--extension', env['WORLDENGINE_MMCV_EXTENSION'], '--all-visible',
        '--expected-capability', 'sm_90'], Path(env['ALGENGINE_ROOT']), env,
        run/'preflight/mmcv_cuda.log'),
        ([env['SIMENGINE_PYTHON'], '-c', program, support,
        support/'preflight_gsplat_cuda.py', '--extension', env['WORLDENGINE_GSPLAT_EXTENSION'],
        '--expected-capability', 'sm_90', '--ray-workers', '4'],
        Path(env['SIMENGINE_ROOT']), env, run/'preflight/gsplat_cuda.log')]
    for job in jobs:
        engine.execute([job], run/'preflight', 600)
    write(run/'preflight/cuda_report.json', dict(status='PASS', devices=device_list,
        contract=artifact(run/'run_contract.json'),
        logs=[artifact(job[3]) for job in jobs]), immutable=False)


def collect_condition(arguments):
    # Let the driver clean up the process groups that it owns on interruption.
    child = subprocess.Popen(arguments)
    try:
        code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, arguments)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='20260916_factorial_bridges_4h100_v1')
    parser.add_argument('--source', default='20260916_factorial_h100_migration_v1')
    parser.add_argument('--devices', default='0,1,2,3')
    parser.add_argument('--plan', action='store_true', help='Read-only plan; no GPU allocation or output creation')
    args = parser.parse_args()
    if not args.run or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                           for c in args.run) or args.run == args.source:
        parser.error('Use a distinct safe run ID')
    ids = devices(args.devices)
    gpu_names = None if args.plan else hardware(ids)
    print('Verifying frozen source models, runtime and accepted scalar bridges...', flush=True)
    source, contract, inputs, runtime = source_run(args.source)
    conditions = [(policy, mode) for policy in engine.POLICIES for mode in ('NR', 'R')]
    capacity = cpu_capacity()
    if args.plan:
        print(json.dumps(dict(run=args.run, source=str(source), conditions=conditions,
            count=len(conditions), devices=ids, current_host_cpu_capacity=capacity,
            proposed_resources=resources(capacity),
            hardware='NOT_CHECKED_PLAN_ONLY', no_training=True, no_full_evaluation=True), indent=2))
        return
    # Fail before creating a run or starting any GPU work on a wrong allocation.
    run = CODE.parent/'runs'/args.run
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Four-H100 queue interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    with lock(source/'active.lock'), lock(run/'queue.lock'):
        allocation = resources(capacity)
        if (run/'run_contract.json').exists():
            frozen = read(run/'run_contract.json')
            allocation = {k:frozen[k] for k in allocation}
            if allocation['ray_cpu_threads'] != 4 * allocation['simulation_cpu_threads']:
                raise RuntimeError('Frozen resources do not schedule four concurrent simulations')
            if allocation['ray_cpu_threads'] + 4 > math.floor(capacity):
                raise RuntimeError('New instance has insufficient CPUs for the frozen resource contract')
        prepare(source, contract, inputs, runtime, run, allocation)
        # The inherited module resolves its paths at import time to the source's frozen runtime.
        # Both runs share exactly that immutable runtime; only output paths/resources differ.
        _, fresh, fresh_inputs, fresh_runtime = engine.load_run(args.run)
        state = dict(status='RUNNING', started=time.time(), devices=gpu_names,
                     cpu_capacity=capacity, resources=allocation, completed_conditions=0)
        write(run/'attempt_status.json', state, immutable=False)
        try:
            cuda_preflight(run, fresh, fresh_inputs, fresh_runtime, args.devices)
            for index, (policy, mode) in enumerate(conditions, 1):
                print('BRIDGE '+str(index)+'/26: '+policy+' '+mode+' on '+args.devices, flush=True)
                # The proven driver checks/reuses complete audits and rejects partial outputs.
                collect_condition([sys.executable, str(CODE/'scripts/run_study_evaluation.py'),
                    'collect', '--run', args.run, '--policy', policy, '--cohort', 'bridge',
                    '--mode', mode, '--devices', args.devices, '--timeout', '3600'])
                state['completed_conditions'] = index
                write(run/'attempt_status.json', state, immutable=False)
            result = status(args.run)
            if result['groups']['bridge']['accepted'] != 26:
                raise RuntimeError('All 26 accepted bridges required')
            state['status'] = 'PASS_BRIDGES_26_OF_26'
            print('PASS: four-GPU scalar parity and all 24 learned-model bridges. Full evaluation remains separate.', flush=True)
        except BaseException as error:
            state.update(status='STOPPED_NOT_COMPLETE', error=str(error))
            raise
        finally:
            state['finished'] = time.time()
            write(run/'attempt_status.json', state, immutable=False)


if __name__ == '__main__':
    main()
