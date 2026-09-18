"""Resume the frozen factorial evaluation, storing only new collections on HDD2."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from selector_study.common import artifact, read, verify, write
from paper_study.hardware import hardware
from paper_study.storage import writable_probe
import run_study_evaluation as engine
import run_study_bridges_4h100 as four
from evaluation_status import status

SOURCE = '20260916_factorial_bridges_4h100_v1'
RESUME = '20260917_factorial_full_resume_v1'


def link_file(path, source):
    source = source.resolve()
    if path.is_symlink():
        if path.resolve() != source:
            raise RuntimeError('Unexpected mapping: '+str(path))
    elif path.exists():
        raise RuntimeError('Refusing to replace existing file: '+str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(source, target_is_directory=source.is_dir())


def attach_inputs(root, original):
    root.mkdir(parents=True, exist_ok=True)
    for name in ('run_contract.json', 'inputs.json'):
        link_file(root/name, original/name)


def archive_partial(path, archive, reason):
    if not path.exists():
        return
    if path.is_symlink() or (path/'accepted.json').exists() or (path/'audit.json').exists():
        raise RuntimeError('Refusing to archive an accepted/audited or mapped collection: '+str(path))
    # In addition to the outer queue/active locks, exclude direct collection writers.
    with four.lock(path/'run.lock'):
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        target = archive/(path.name+'_'+stamp)
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        write(target/'resume_archive_reason.json', dict(reason=reason, original=str(path),
              original_paths_retained_in_evidence=True, reusable_as_accepted_result=False))
        print('PRESERVED_PARTIAL '+str(target), flush=True)


def validate_result(folder, original, inputs, policy, cohort, mode):
    accepted = read(folder/'accepted.json')
    audit = read(verify(accepted['audit']))
    collection = read(verify(audit['collection']))
    if (accepted['status'] != 'PASS' or audit['status'] != 'PASS'
            or not accepted.get('actual_actions_verified')
            or collection['id'] != f'{cohort}_{mode}_{policy}'
            or collection['contract'] != artifact(original/'run_contract.json')
            or collection['checkpoint'] != inputs['models'][policy]
            or collection['rows'] != inputs['cohorts'][cohort]
            or audit['incomplete_or_dropped_scenes'] != 0):
        raise RuntimeError('Resumed result provenance mismatch: '+str(folder))


def publish(original, root, key):
    destination = root/'collections'/key
    if not (destination/'accepted.json').exists():
        raise RuntimeError('Only accepted results can be published')
    public = original/'collections'/key
    if public.is_symlink():
        if public.resolve() != destination.resolve():
            raise RuntimeError('Foreign collection mapping: '+str(public))
        return
    if public.exists():
        archive_partial(public, original/'archive/quota_interrupted',
                        'Quota-interrupted attempt retained; replacement accepted on HDD2')
    link_file(public, destination)


def inspect():
    original, contract, inputs, runtime = engine.load_run(SOURCE)
    progress = status(SOURCE)
    if progress['groups']['bridge']['accepted'] != 26:
        raise RuntimeError('All original 26 bridge acceptances required')
    if contract.get('workers') != 4:
        raise RuntimeError('Expected frozen four-worker contract')
    pending = []
    for cohort in ('full', 'common'):
        for policy in inputs['models']:
            for mode in ('NR', 'R'):
                key = f'{cohort}_{mode}_{policy}'
                if progress['groups'][cohort]['conditions'][key] != 'ACCEPTED':
                    pending.append((policy, cohort, mode))
    storage = read(CODE.parent/'registry/paper_storage.json')
    root = (Path(storage['runs_root'])/RESUME).resolve()
    if not Path(storage['runs_root']).is_absolute() or root == original.resolve():
        raise RuntimeError('Separate absolute continuation storage required')
    return original, contract, inputs, runtime, progress, pending, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--devices', default='0,1,2,3')
    args = parser.parse_args()
    original, contract, inputs, runtime, progress, pending, root = inspect()
    overview = dict(source=str(original), new_output=str(root), remaining_conditions=len(pending),
        accepted={g:progress['groups'][g]['accepted'] for g in ('bridge','full','common')},
        first_pending=pending[0] if pending else None,
        existing_acceptances_preserved=True, partial_condition_restarts_from_beginning=True)
    print(json.dumps(overview, indent=2), flush=True)
    if args.plan or not pending:
        return
    devices = four.devices(args.devices)
    names = hardware(devices)
    capacity = four.cpu_capacity()
    if capacity < contract['ray_cpu_threads'] + 4:
        raise RuntimeError('Frozen 4-worker evaluation requires at least '+str(contract['ray_cpu_threads']+4)+' CPU cores')
    def interrupted(*_):
        raise KeyboardInterrupt('Full evaluation resume interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    # Exclude the original launcher/driver; never kill other jobs or GPU holders.
    with four.lock(original/'queue.lock'), four.lock(original/'active.lock'):
        # Re-read acceptances after acquiring both locks.
        original, contract, inputs, runtime, progress, pending, root = inspect()
        writable_probe(root)
        with four.lock(root/'resume.lock'):
            attach_inputs(root, original)
            link_file(CODE.parent/'runs'/RESUME, root)
            write(root/'resume_contract.json', dict(source=artifact(original/'run_contract.json'),
                inputs=artifact(original/'inputs.json'), implementation=artifact(Path(__file__)),
                launcher=artifact(CODE/'run_h100_full_eval_resume.sh'), output=str(root),
                protocol_unchanged=True, accepted_results_stay_in_original_locations=True))
            attempt=dict(status='RUNNING',started=time.time(),hardware=names,cpu_capacity=capacity,
                         pid=os.getpid(),hostname=os.uname().nodename,remaining_at_start=len(pending))
            write(root/'resume_status.json', attempt, immutable=False)
            try:
                env=engine.environment(root,contract,runtime,args.devices)
                env.update(WORLDENGINE_ROOT=str(runtime), ALGENGINE_ROOT=str(runtime/'projects/AlgEngine'),
                           SIMENGINE_ROOT=str(runtime/'projects/SimEngine'))
                four.cuda_preflight(root,contract,inputs,runtime,args.devices)
                # Validate this instance against history, including H200, without rewriting old bridges.
                stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                gate=root/'hardware_validation'/stamp
                attach_inputs(gate,original)
                for mode in ('NR','R'):
                    engine.collection(gate,contract,inputs,runtime,env,'scalar_v3','bridge',mode,10800)
                completed=0
                for policy,cohort,mode in pending:
                    key=f'{cohort}_{mode}_{policy}';folder=root/'collections'/key
                    # A stopped continuation can be retried without discarding evidence.
                    if folder.exists() and not (folder/'accepted.json').exists() and not (folder/'audit.json').exists():
                        archive_partial(folder,root/'archive/incomplete', 'Stopped continuation; complete condition restarted')
                    print('RESUMING '+key,flush=True)
                    engine.collection(root,contract,inputs,runtime,env,policy,cohort,mode,10800)
                    validate_result(folder,original,inputs,policy,cohort,mode)
                    publish(original,root,key)
                    completed+=1
                    write(root/'resume_status.json',dict(attempt,completed_this_attempt=completed,last_accepted=key),immutable=False)
                final=status(SOURCE)
                if any(final['groups'][g]['accepted']!=26 for g in ('bridge','full','common')):
                    raise RuntimeError('Final complete-cohort verification failed')
                write(root/'resume_status.json',dict(attempt,status='PASS_FULL_AND_COMMON',finished=time.time(),
                    completed_this_attempt=completed,accepted={g:final['groups'][g]['accepted'] for g in ('bridge','full','common')}),immutable=False)
                print('PASS_FULL_AND_COMMON: all 52 formal closed-loop conditions accepted',flush=True)
            except BaseException as exc:
                try:write(root/'resume_status.json',dict(attempt,status='STOPPED',finished=time.time(),reason=str(exc)),immutable=False)
                except OSError as error:print('Unable to persist stop status: '+str(error),file=sys.stderr)
                raise


if __name__ == '__main__':
    main()
