#!/usr/bin/env python3
"""Foreground jobs with immutable inputs, private logs/tmp, locks and dependencies."""
import sys
from pathlib import Path
REPAIR = Path(__file__).resolve().parent
ORIGINAL = REPAIR.with_name("selector_v3_online_diagnosis_20260930")
sys.path.insert(0, str(ORIGINAL / "handoff"))
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import uuid
from experiment_common import WORK,CODE,TOOLS,REGISTRY,SETTINGS,read,write,sha,lock,source_hashes,configured_environment,command
from evaluation_artifacts import prepare_development,verify_prepared,materialize_policy,accept_stage,frozen_policies


def registration():
    from repair_contract import verify_repair
    verify_repair()
    spec=read(REGISTRY/'experiment.json')
    if spec['source_sha256']!=source_hashes() or spec['settings_sha256']!=sha(SETTINGS):
        raise ValueError('Frozen experiment code/settings changed; use a new version, never mixed resume')
    if sha(REGISTRY/'scene_candidates.json')!=spec['candidate_manifest_sha256']:
        raise ValueError('Candidate registration changed')
    return spec


def campaign_clock():
    path=REGISTRY/'campaign_clock.json'
    if not path.exists():
        now=time.time()
        try:write(path,dict(started_unix=now,deadline_unix=now+64800,wall_seconds=64800),immutable=True)
        except FileExistsError:pass
    return read(path)


def ready(job):
    if job['kind']=='admission':return True
    if not (REGISTRY/'scene_manifest.json').exists():return False
    if job.get('depends_on_marker') and not (REGISTRY/job['depends_on_marker']).exists():return False
    if job.get('depends_on_old64_dev'):
        from summarize_experiment import accepted
        if not all(accepted('dev_'+p+'_'+m) for p in ('parent','online64') for m in ('R','NR')):
            return False
    if job.get('depends_on_selection'):
        path=REGISTRY/'selection.json'
        return path.exists() and read(path).get('status')=='QUALIFIED_FINAL_DEV_SELECTION'
    return True


def run_processes(commands,folder,deadline,envs=None,live_single=False):
    """Own each process group, expose failures, stop all children on interruption."""
    children=[];logs=[];start=time.monotonic();previous={}
    def stop(signum,frame):raise InterruptedError('Signal '+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM):previous[sig]=signal.signal(sig,stop)
    try:
        for i,(argv,cwd,env) in enumerate(commands):
            print(json.dumps(dict(command=list(map(str,argv)),cwd=str(cwd))),flush=True)
            if live_single and len(commands)==1:out=None
            else:out=(folder/('lane%d.log'%i)).open('x');logs.append(out)
            children.append(subprocess.Popen(list(map(str,argv)),cwd=cwd,env=env,stdout=out,
                stderr=subprocess.STDOUT if out else None,start_new_session=True))
        while True:
            codes=[c.poll() for c in children]
            if any(x is not None and x!=0 for x in codes):raise RuntimeError('Child failure: '+repr(codes))
            if all(x==0 for x in codes):break
            if time.time()>=deadline:raise TimeoutError('Registered 18h campaign deadline reached')
            print(json.dumps(dict(event='job_running',elapsed_seconds=round(time.monotonic()-start),
                                  child_codes=codes,logs=str(folder))),flush=True)
            time.sleep(20)
    except BaseException:
        for path in folder.glob('lane*.log'):
            print('Failure log: '+str(path),flush=True)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-60:]),flush=True)
        raise
    finally:
        for child in children:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        for log in logs:log.close()
        for sig,handler in previous.items():signal.signal(sig,handler)


def maybe_finalize():
    if (REGISTRY/'scene_manifest.json').exists():return
    candidates=read(REGISTRY/'scene_candidates.json')
    if not all((REGISTRY/'admission'/(r['scene_id']+'.json')).exists() for r in candidates['scenes']):return
    try:handle=lock(REGISTRY/'finalize.lock')
    except BlockingIOError:return
    with handle:
        if not (REGISTRY/'scene_manifest.json').exists():
            from prepare_scene_pool import finalize
            finalize(REGISTRY/'scene_candidates.json',REGISTRY/'admission',REGISTRY/'scene_manifest.json')
            prepare_development()


def probe_job(job,folder,devices,deadline):
    cfg,base_env=configured_environment(devices,folder)
    if job['kind']=='admission':
        allrows=read(REGISTRY/'scene_candidates.json')['scenes']
        rows=allrows[job['shard']::job['shards']]
    elif job['kind']=='diagnostic_extra':
        from summarize_experiment import accepted
        manifest=read(REGISTRY/'scene_manifest.json')
        byid={r['scene_id']:r for r in manifest['scenes'] if r['split']=='development'}
        cases=[]
        for reaction in ('R','NR'):
            parent=accepted('dev_parent_'+reaction)['rows'];online=accepted('dev_online64_'+reaction)['rows']
            for sid in sorted(parent):
                delta={k:online[sid][k]-parent[sid][k] for k in parent[sid]}
                if any(delta[k]<-1e-6 for k in ('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')):
                    cases.append(dict(byid[sid],probe_reaction=reaction,extra_trigger=delta))
        rows=cases[job['shard']::job['shards']]
        write(folder/'failure_followup_sampling.json',dict(all_cases=len(cases),scenes=rows,
            inclusion='all dev clips with negative score or safety delta; separate from representative 48'))
    else:
        rows=[];bylog={}
        for row in read(REGISTRY/'scene_manifest.json')['scenes']:
            if row['split']=='development':bylog.setdefault(row['origin_log'],[]).append(row)
        import hashlib
        for name,group in sorted(bylog.items()):
            chosen=sorted(group,key=lambda r:hashlib.sha256(('diagnosis:'+r['scene_id']).encode()).hexdigest())[:2]
            if len(chosen)!=2:raise ValueError('48-root diagnosis needs two admitted clips per development log')
            rows+=chosen
        if len(rows)!=24:raise ValueError('48-root diagnosis requires 12 independent development logs')
        write(folder/'root_sampling.json',dict(policy='two_hash_selected_clips_per_log',roots_per_clip=[0,4],
            scenes=rows,continuation='same frozen parent after the selected first action'))
    cmds=[]
    for rank,gpu in enumerate(devices.split(',')):
        lane_rows=rows[rank::8]
        if not lane_rows:continue
        lane=folder/('rank%d'%rank);lane.mkdir()
        assignments=lane/'assignments.json';write(assignments,lane_rows)
        _,env=configured_environment(gpu,lane)
        argv=[cfg['algengine_python'],'-u',str(REPAIR/'diagnostic_probe.py'),'--settings',SETTINGS,
            '--assignments',assignments,'--output',lane/'probe','--workers','4',
            '--purpose',{'admission':'admission','diagnostic':'diagnostic','diagnostic_extra':'extra'}[job['kind']]]
        if job['kind']=='admission':argv+=['--candidate-manifest',REGISTRY/'scene_candidates.json','--receipt-dir',REGISTRY/'admission']
        else:argv+=['--comparison-selector',frozen_policies()['online64']['selector']]
        cmds.append((argv,CODE,env))
    run_processes(cmds,folder,deadline)
    if job['kind']=='admission':maybe_finalize()
    return dict(status='COMPLETE',scenes=len(rows),output=str(folder))


def training_job(job,folder,devices,deadline,check_only=False):
    cfg,env=configured_environment(devices,folder)
    manifest=read(REGISTRY/'scene_manifest.json');episodes=manifest['episode_budgets']['6']
    base=REGISTRY/'jobs'/job['id'];output=base/'training'
    resolved=dict(job=job,manifest_sha256=sha(REGISTRY/'scene_manifest.json'),episodes=episodes,
                  budgets=manifest['episode_budgets'],settings_sha256=sha(SETTINGS))
    if (base/'resolved.json').exists():
        if read(base/'resolved.json')!=resolved:raise ValueError('Training resume inputs changed')
    elif not check_only:write(base/'resolved.json',resolved,immutable=True)
    argv=[cfg['algengine_python'],'-u',TOOLS/'innovation3_runtime.py','--settings',SETTINGS,
        '--scene-manifest',REGISTRY/'scene_manifest.json','--output',output,'--devices',devices,
        '--episodes',str(episodes),'--steps','8','--seed',str(job['seed']),'--split','train',
        '--branch-workers','4','--fixed-segment','--wall-seconds','43200','--update-timing',job['update_timing']]
    if check_only:argv.append('--check-only')
    elif output.with_suffix('.progress.json').exists():argv.append('--resume')
    argv.append('ddp-online-session-gate')
    if check_only:command(argv,env);return dict(status='CHECK_ONLY')
    # Leave time to finish/commit the current round after the campaign stop flag.
    run_processes([(argv,CODE,env)],folder,deadline,live_single=True)
    progress=read(output.with_suffix('.progress.json'))
    status='COMPLETE' if progress['completed']==episodes else progress['status']
    return dict(status=status,output=str(output),completed=progress['completed'],coverage=progress.get('coverage'))


def evaluation_job(job,folder,devices,deadline,dry=False):
    prepared=prepare_development();verify_prepared(prepared)
    policy=job['policy']
    if policy.startswith('selected_s'):
        selection=read(REGISTRY/'selection.json');policy=selection['policies'][int(policy[-1])]
    artifact=materialize_policy(policy)
    if artifact is None:raise BlockingIOError('Checkpoint is being materialized by another evaluation job')
    cfg,env=configured_environment(devices,folder)
    # The TCP rendezvous is scoped to this job; independent nodes need not share it.
    with socket.socket() as sock:sock.bind(('',0));env['MASTER_PORT']=str(sock.getsockname()[1])
    output=folder/'evaluation'
    if job['dataset']=='formal':
        contract=read(REGISTRY/'fulltable_data_contract.json')
        if sha(contract['scenarios'])!=contract['scenarios_sha256']:raise ValueError('Formal scenes changed')
        scenarios,assets=contract['scenarios'],contract['assets'];expected=contract['expected_tokens'][job['mode']]
    else:
        scenarios,assets=prepared['scenarios'],prepared['assets'];expected=[r['scene_id'] for r in prepared['scenes']]
    argv=[cfg['algengine_python'],'-u',WORK/'handoff/selector_eval_runtime_20260930.py',
        '--settings',SETTINGS,'--devices',devices]
    if dry:argv.append('--dry-run')
    mode=job['mode']
    argv+=['openloop' if mode.startswith('ol_') else 'closedloop','--config',prepared['config'],
        '--checkpoint',artifact['path'],'--output',output,'--eval-seed','0',
        '--timeout',str(max(1,int(deadline-time.time()))),'--diagnostic-interval',str(job['interval_seconds'])]
    if job.get('record_plans'):argv.append('--record-plans')
    if mode.startswith('ol_'):argv+=['--split','navtest' if mode=='ol_navtest' else 'rare']
    else:argv+=['--reaction',mode[3:],'--scenarios',scenarios,'--assets',assets]
    inputs=dict(job=job,policy=policy,artifact=artifact,manifest_sha256=prepared['manifest_sha256'],
        expected=expected,configuration_sha256=prepared['config_sha256'])
    write(folder/'evaluation_inputs.json',inputs,immutable=True)
    run_processes([(argv,CODE,env)],folder,deadline,live_single=True)
    if dry:return dict(status='DRY_RUN')
    result=accept_stage(output,mode,expected)
    result.update(job=job,policy=policy,artifact=artifact,output=str(output),inputs_sha256=sha(folder/'evaluation_inputs.json'))
    write(folder/'result.json',result,immutable=True)
    return dict(status='COMPLETE',result=str(folder/'result.json'),result_sha256=sha(folder/'result.json'))


def preflight(job,devices):
    cfg,env=configured_environment(devices)
    if len(devices.split(','))!=8:raise ValueError('Exactly 8 devices required')
    command([cfg['algengine_python'],'-c',
        'from innovation3 import coverage_probe, ddp_online_session, online_probe; '
        'import worldengine; print("PASS_ALG_IMPORTS", worldengine.__file__)'],env)
    command([cfg['simengine_python'],'-c',
        'from innovation3 import fixed_worker; from worldengine.online import scorer_parity; '
        'print("PASS_SIM_IMPORTS", fixed_worker.__file__)'],env)
    if job['kind']=='train' and (REGISTRY/'scene_manifest.json').exists():
        folder=REGISTRY/'checks'/job['id'];folder.mkdir(parents=True,exist_ok=True)
        training_job(job,folder,devices,time.time()+43200,True)
    else:print('CHECK_ONLY: imports/registration passed; no GPU workers launched. Admission-dependent inputs remain pending.',flush=True)


def run(job,devices,check_only=False,wait=True):
    if len(devices.split(','))!=8 or len(set(devices.split(',')))!=8:raise ValueError('Use 8 distinct GPUs')
    spec=registration()
    if check_only:return preflight(job,devices)
    if job['kind'] in ('train','evaluation') and not (REPAIR/'evidence/recovery_release.json').exists():
        raise RuntimeError('Validated recovery has not been released')
    if job['kind']=='train':
        root_status=REGISTRY/'jobs/roots48/status.json'
        root=read(root_status) if root_status.exists() else {}
        if root.get('status')!='COMPLETE' or root.get('repair_manifest_sha256')!=sha(REPAIR/'repair_manifest.json'):
            raise RuntimeError('Run repaired roots48 successfully before resuming training; GPU wait avoided')
    deadline=campaign_clock()['deadline_unix']
    base=REGISTRY/'jobs'/job['id'];base.mkdir(parents=True,exist_ok=True)
    with lock(base/'job.lock'):
        if (base/'status.json').exists() and read(base/'status.json')['status']=='COMPLETE':
            status=read(base/'status.json')
            if status.get('result') and sha(status['result'])!=status['result_sha256']:raise ValueError('Accepted result changed')
            print('ALREADY_COMPLETE:',job['id'],flush=True);return status
        # Queue state can change while another node owns the job lock.
        if not wait and (base/'status.json').exists() and read(base/'status.json')['status']=='FAILED':
            raise BlockingIOError('Failed job requires explicit retry; stale queue observation')
        while not ready(job):
            if not wait:raise BlockingIOError('Dependencies are not ready')
            if time.time()>=deadline:raise TimeoutError('Campaign deadline while waiting for dependencies')
            if (REGISTRY/'correctness_failure.json').exists():raise RuntimeError('Correctness failure blocks downstream jobs')
            print('WAITING_FOR_DEPENDENCIES:',job['id'],flush=True);time.sleep(20)
            maybe_finalize()
        if time.time()>=deadline:
            raise TimeoutError('Campaign deadline; no new process started')
        if (REGISTRY/'correctness_failure.json').exists():
            failure=read(REGISTRY/'correctness_failure.json')
            if (job['id']!='roots48' or failure.get('job')!='roots48' or
                    failure.get('matched')!=['failed to reproduce the exact root']):
                raise RuntimeError('Unexpected correctness failure blocks repaired replay')
        attempt=base/('attempt_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8])
        attempt.mkdir()
        status=dict(status='RUNNING',job=job,host=socket.gethostname(),pid=os.getpid(),attempt=str(attempt),
            started_unix=time.time(),experiment_sha256=sha(REGISTRY/'experiment.json'),devices=devices,
            repair_manifest_sha256=sha(REPAIR/'repair_manifest.json'))
        write(base/'status.json',status)
        try:
            if job['kind'] in ('admission','diagnostic','diagnostic_extra'):result=probe_job(job,attempt,devices,deadline)
            elif job['kind']=='train':result=training_job(job,attempt,devices,deadline)
            else:result=evaluation_job(job,attempt,devices,deadline)
            registration()  # reject mid-run source changes
            status.update(result,finished_unix=time.time())
        except BlockingIOError:
            status.update(status='WAITING_FOR_EXPORT',finished_unix=time.time());write(base/'status.json',status);raise
        except BaseException as error:
            status.update(status='FAILED',error=repr(error),traceback=traceback.format_exc(),finished_unix=time.time())
            write(base/'status.json',status)
            critical=('parity failed','parity mismatch','changed inside','context mutated',
                'failed to reproduce the exact root','before first different action',
                'scorer mismatch','not explained by comfort','Canonical state/observation continuity failed')
            text='\n'.join(p.read_text(errors='replace')[-100000:] for p in attempt.glob('lane*.log'))
            if any(token in text for token in critical):
                stop=REGISTRY/'correctness_failure.json'
                if not stop.exists():
                    try:write(stop,dict(job=job['id'],attempt=str(attempt),reason='hard correctness check failed',
                        matched=[t for t in critical if t in text]),immutable=True)
                    except FileExistsError:pass
            raise
        write(base/'status.json',status)
        print(json.dumps(status),flush=True)
        return status


def result_exit_code(result, check_only=False):
    if check_only or result.get("status") == "COMPLETE": return 0
    return 3 if result.get("status") == "CORRECTNESS_STOPPED_AT_BOUNDARY" else 4

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('job');p.add_argument('--devices',default=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7'))
    p.add_argument('--check-only',action='store_true');p.add_argument('--no-wait',action='store_true')
    a=p.parse_args();spec=registration();matches=[j for j in spec['jobs'] if j['id']==a.job]
    if len(matches)!=1:p.error('Unknown job; inspect registry/experiment.json')
    result=run(matches[0],a.devices,a.check_only,not a.no_wait)
    raise SystemExit(result_exit_code(result, a.check_only))
if __name__=='__main__':main()
