"""Native factorial evaluation: immutable inputs, action audits and explicit GPU lease."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selector_study.common import CODE, artifact, read, sha, verify, write

METHOD='selector_prefeedback_factorial_eval_v1'
POLICIES=['scalar_v3']+[f'{a}_seed{s}' for s in (0,1,2) for a in ('D1','D2','D3','D4')]


def load_run(name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Invalid run ID')
    run=CODE.parent/'runs'/name;contract=read(run/'run_contract.json');inputs=read(run/'inputs.json')
    if contract['method']!=METHOD or inputs['method']!=METHOD:raise RuntimeError('Wrong protocol')
    runtime=run/'runtime'
    if contract['runtime']!=str(runtime):raise RuntimeError('Runtime moved')
    for relative,expected in contract['implementation'].items():
        if sha(runtime/relative)!=expected:raise RuntimeError('Runtime changed: '+relative)
    for name,item in contract['driver'].items():
        verify(item)
        if sha(CODE/'scripts'/name)!=item['sha256']:raise RuntimeError('Current evaluation driver differs from frozen archive')
    for item in inputs['artifacts'].values():verify(item)
    for item in inputs['models'].values():verify(item)
    script=runtime/'projects/AlgEngine/scripts/diffusiondrive'
    sys.path.insert(0,str(script))
    return run,contract,inputs,runtime


def environment(run,contract,runtime,devices):
    from run_selector_cfpi import environment as inherited
    legacy=Path(contract['source_worldengine_root']);env=inherited(legacy)
    exp=legacy.parent/'exp'
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        CUDA_VISIBLE_DEVICES=devices,WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP=devices,
        DIFFUSIONDRIVE_ROLLOUT_CODE_SHA=contract['code_sha'],NAVSIM_EXP_ROOT=str(exp),
        NAVSIM_METRIC_CACHE_PATH=str(exp/'metric_cache_navtest'),NAVSIM_METRIC_CACHE_PATH_EVAL=str(exp/'metric_cache_navtest'),
        NAVSIM_METRIC_CACHE_PATH_TRAIN=str(exp/'metric_cache_trainval'),NAVSIM_DEVKIT_ROOT=str(legacy.parent/'DiffusionDrive'),
        NAVSIM_OFFICIAL_RESCORE='always',SELECTOR_SNAPSHOT_OPEN='0',NAVSIM_RESCORE_SPLIT='navtest',
        NAVSIM_RESCORE_SHARDS=str(len(devices.split(','))))
    parent=read(legacy/'experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json')
    env.update(DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT=parent['baseline'],DIFFUSIONDRIVE_SELECTOR_INIT_SHA256=parent['baseline_sha256'])
    return env


def preflight(run,contract,inputs,runtime,env):
    config=runtime/'projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_selector_factorial_native.py'
    folder=run/'preflight';folder.mkdir(exist_ok=True)
    failures=[];probes={}
    for key in ('ALGENGINE_PYTHON','SIMENGINE_PYTHON','WORLDENGINE_MMCV_EXTENSION','WORLDENGINE_GSPLAT_EXTENSION','DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT'):
        if not Path(env[key]).is_file():failures.append('Missing '+key+': '+env[key])
    for key,path in inputs['assets'].items():
        if not Path(path).is_dir():failures.append('Missing '+key+' assets: '+path)
    config_env=dict(env,SELECTOR_SNAPSHOT_NOISE='formal_navtest_seed0',SELECTOR_SNAPSHOT_MODE='NR',SELECTOR_SNAPSHOT_CONDITION='preflight')
    snippets={
        'planner':("from mmcv import Config; import mmcv.ops; import mmdet3d_plugin; "
            "c=Config.fromfile("+repr(str(config))+"); "
            "assert c.selector_rollout_contract.experiment=="+repr(METHOD)+"; "
            "assert c.model.planning_head.online_reward is None; assert c.model.planning_head.export_rollout_context; "
            "assert c.model.planning_head.scene_selector_score_mode=='residual'; _=c.pretty_text; print('PLANNER_IMPORT_CONFIG_PASS')"),
        'simulator':"import torch,hydra,ray; from worldengine.runner.builders.env_builder import build_envs; from worldengine.manager.selector_snapshot_manager import SelectorSnapshotManager; print('SIMULATOR_IMPORT_PASS')"}
    for key,program in snippets.items():
        python=env['ALGENGINE_PYTHON' if key=='planner' else 'SIMENGINE_PYTHON']
        if not Path(python).is_file():continue
        result=subprocess.run([python,'-c',program],cwd=env['ALGENGINE_ROOT' if key=='planner' else 'SIMENGINE_ROOT'],
            env=config_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
        log=folder/(key+'.log');log.write_text(result.stdout)
        probes[key]=dict(exit_code=result.returncode,log=artifact(log))
        if result.returncode:failures.append(key+' import/config failed; see '+str(log))
    result=dict(status='FAIL' if failures else 'PASS_CPU_IMPORTS',failures=failures,probes=probes,
        contract=artifact(run/'run_contract.json'),gpu_kernels_validated=False,physical_simulator_bridge_completed=False)
    write(folder/'report.json',result,immutable=False)
    print(json.dumps(result),flush=True)
    if failures:raise RuntimeError('Evaluation preflight failed')
    return result


@contextlib.contextmanager
def gpu_lease(run,legacy,pid,pane):
    if pid is None:
        yield
        return
    def query(field):return subprocess.check_output(['tmux','display-message','-p','-t',pane,field],text=True).strip()
    proc=Path('/proc')/str(pid)
    if proc.joinpath('cmdline').read_bytes()!=b'python\0occupy_GPU.py\0' or proc.joinpath('cwd').resolve()!=legacy:
        raise RuntimeError('Not the authorized GPU holder')
    parent=next(int(x.split()[1]) for x in proc.joinpath('status').read_text().splitlines() if x.startswith('PPid:'))
    if int(query('#{pane_pid}'))!=parent or query('#{pane_current_path}')!=str(legacy):
        raise RuntimeError('Tmux holder identity differs')
    lease=dict(status='VERIFIED',pid=pid,pane=pane,started=time.time())
    path=run/'gpu_lease.json';write(path,lease,immutable=False)
    released=False
    try:
        os.kill(pid,signal.SIGINT);released=True
        for _ in range(100):
            if not proc.exists():break
            time.sleep(.2)
        if proc.exists():raise RuntimeError('Holder did not exit')
        lease['status']='RELEASED';write(path,lease,immutable=False)
        yield
    finally:
        if released and not proc.exists():
            if query('#{pane_current_command}') not in ('bash','sh','zsh'):
                lease['status']='RESTORE_REQUIRES_IDLE_SHELL';write(path,lease,immutable=False)
                raise RuntimeError('Refusing to type into non-idle pane')
            subprocess.run(['tmux','send-keys','-t',pane,'-l','python occupy_GPU.py'],check=True)
            subprocess.run(['tmux','send-keys','-t',pane,'Enter'],check=True)
            lease['status']='RESTORE_SENT'
            for _ in range(100):
                if query('#{pane_current_command}')=='python':lease['status']='RESTORED';break
                time.sleep(.2)
        elif proc.exists():lease['status']='ORIGINAL_HOLDER_STILL_RUNNING'
        lease['finished']=time.time();write(path,lease,immutable=False)
        print(json.dumps(lease),flush=True)


def execute(jobs,root,timeout):
    children=[];started=time.monotonic()
    try:
        for command,cwd,env,log in jobs:
            log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('w')
            try:p=subprocess.Popen(list(map(str,command)),cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException:stream.close();raise
            children.append((p,stream))
        tick=0.
        while True:
            codes=[p.poll() for p,_ in children]
            if any(x not in (None,0) for x in codes):raise RuntimeError('Evaluation child failed: '+str(codes)+'; logs: '+str(root/'logs'))
            if all(x==0 for x in codes):break
            elapsed=time.monotonic()-started
            if elapsed>timeout:raise TimeoutError('Collection timeout')
            if elapsed-tick>30:
                print(json.dumps(dict(collection=root.name,elapsed_seconds=round(elapsed),child_codes=codes)),flush=True);tick=elapsed
            time.sleep(1)
    finally:
        # Signal only process groups this invocation created, including surviving descendants.
        for p,_ in children:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        deadline=time.monotonic()+10
        for p,_ in children:
            try:p.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
        for _,stream in children:stream.close()


def compare_scalar(audit,old):
    import numpy as np
    import cfpi_common as c
    import selector_cfpi_deployment_common as d
    from selector_feedback_common import FINGERPRINT_FIELDS
    from audit_selector_cfpi_deployment import action_error
    if set(audit['metrics'])!=set(old['metrics']):raise RuntimeError('Scalar bridge membership differs')
    for scene,values in audit['metrics'].items():
        if any(abs(values[k]-old['metrics'][scene][k])>c.OUTCOME_TOLERANCE for k in d.METRICS):
            raise RuntimeError('Scalar bridge outcome changed: '+scene)
        for step in c.DECISION_STEPS:
            left=read(verify(audit['records'][scene][str(step)]['record']))
            right=read(verify(old['records'][scene][str(step)]['record']))
            x=c.load_pickle(verify(left['sidecar']));y=c.load_pickle(verify(right['sidecar']))
            for key in FINGERPRINT_FIELDS:
                if np.max(np.abs(np.asarray(x[key])-np.asarray(y[key])))>c.ARRAY_TOLERANCE:
                    raise RuntimeError('Scalar bridge input differs: '+key)
            if (x['selected_index']!=y['selected_index'] or action_error(left['actual_action'],right['actual_action'])>c.ACTION_TOLERANCE
                    or np.max(np.abs(x['current_logits']-y['current_logits']))>c.MODEL_RECOMPUTE_TOLERANCE):
                raise RuntimeError('Scalar bridge action/logits differ')


def collection(run,contract,inputs,runtime,env,policy,cohort,mode,timeout):
    import cfpi_common as c
    import selector_snapshot_audit as audit
    root=run/'collections'/f'{cohort}_{mode}_{policy}'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'audit.json').exists():
            result=audit.checked(root,full=True)
        else:
            if any(root.glob('split_*')) or (root/'logs').exists():
                raise RuntimeError('Partial collection retained; inspect/archive it explicitly before retry')
            rows=inputs['cohorts'][cohort];prefixes={}
            for row in rows:
                for prefix in (row['scene_id'],row['scene_id'].rsplit('-',1)[1]):
                    if prefix in prefixes and prefixes[prefix]!=row['scene_id']:raise RuntimeError('Ambiguous prefix')
                    prefixes[prefix]=row['scene_id']
            devices=env['CUDA_VISIBLE_DEVICES'].split(',')
            co=dict(method=METHOD,id=root.name,policy=policy,cohort=cohort,mode=mode,
                checkpoint=inputs['models'][policy],scenario=inputs['artifacts'][cohort+'_scenario'],
                rows=rows,prefixes=prefixes,asset_folder=inputs['assets'][cohort],
                noise=c.NOISE_NAMESPACE if cohort=='bridge' else 'formal_navtest_seed0',
                contract=artifact(run/'run_contract.json'),inputs=artifact(run/'inputs.json'),workers=len(devices))
            path=root/'collection.json';write(path,co)
            env=dict(env,SELECTOR_SNAPSHOT_NOISE=co['noise'],SELECTOR_SNAPSHOT_MODE=mode,SELECTOR_SNAPSHOT_CONDITION=co['id'])
            alg=Path(env['ALGENGINE_ROOT']);sim=Path(env['SIMENGINE_ROOT'])
            cmd=[env['SIMENGINE_PYTHON'],sim/'worldengine/runner/run_simulation.py',
                'debug_mode=True','debug_scene_name=null',f"data_file_path={co['scenario']['path']}",
                f"asset_folder_path={co['asset_folder']}",f'output_dir={root}/__WORKER_ID__/WE_output',
                f'job_name=factorial_{root.name}','use_planner_actions=true','ego_policy=env_input_policy',
                'ego_client=navformer_client','ego_controller=log_play_controller','ego_navigation=trajectory_navigation',
                'agent_policy='+('idm_policy' if mode=='R' else 'trajectory_policy'),
                'agent_navigation='+('idm_navigation' if mode=='R' else 'trajectory_navigation'),
                f'planner_data_path={root}/__WORKER_ID__/plan_traj',f'planner_client_folder={root}/__WORKER_ID__/frames',
                'with_metric_manager=true','with_dense_reward_manager=false','selector_snapshot=true',
                f'selector_snapshot_manifest={path}',f'selector_snapshot_manifest_sha256={sha(path)}',
                f'diffusiondrive_candidate_sidecar_path={root}/__WORKER_ID__/diffusiondrive_candidate_sidecars',
                'diffusiondrive_sidecar_timeout_s=300','distributed_mode=SCENARIO_BASED','worker=ray_distributed',
                'worker_id_prefix=split_','enable_resume=false',f'completed_scenarios_dir={root}/__WORKER_ID__/completed_scenarios',
                f"worker.threads_per_node={contract['ray_cpu_threads']}",
                f"number_of_cpus_allocated_per_simulation={contract['simulation_cpu_threads']}"]
            jobs=[(cmd,sim,env,root/'logs/simulator.log')]
            config=alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_factorial_native.py'
            for index,device in enumerate(devices):
                worker=root/f'split_{index}'
                for name in ('frames','plan_traj','merged_ann_files','diffusiondrive_candidate_sidecars'):(worker/name).mkdir(parents=True,exist_ok=True)
                cmd=[env['ALGENGINE_PYTHON'],alg/'closed_loop/sim_test.py',config,verify(co['checkpoint']),
                    '--seed','0','--log-dir',worker,'--cfg-options',f'sim.monitored_folder={worker}/frames',
                    f'sim.plan_save_path={worker}/plan_traj',f'sim.merged_ann_save_dir={worker}/merged_ann_files',
                    f'sim.diffusiondrive_rollout_sidecar_path={worker}/diffusiondrive_candidate_sidecars',
                    'sim.clean_temp_files=False','sim.clean_record_data=False',f'data_root={worker}/WE_output/openscene_format/']
                jobs.append((cmd,alg,dict(env,CUDA_VISIBLE_DEVICES=device),root/f'logs/planner{index}.log'))
            execute(jobs,root,timeout);result=audit.audit(path)
        if cohort=='bridge' and policy=='scalar_v3':
            compare_scalar(result,read(verify(inputs['artifacts']['scalar_bridge_'+mode])))
        write(root/'accepted.json',dict(status='PASS',audit=artifact(root/'audit.json'),
            scalar_historical_parity=(policy=='scalar_v3' and cohort=='bridge'),
            actual_actions_verified=True,independent_blind_test=False))
        print(json.dumps(dict(status='PASS',policy=policy,cohort=cohort,mode=mode,scenes=result['scenes'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','collect'])
    p.add_argument('--run',default='20260916_factorial_eval_v2')
    p.add_argument('--policy',choices=POLICIES,default='scalar_v3')
    p.add_argument('--cohort',choices=['bridge','full','common'],default='bridge')
    p.add_argument('--mode',choices=['NR','R'],default='NR')
    p.add_argument('--devices',default='0');p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--holder-pid',type=int);p.add_argument('--pane',default='0:0.0')
    args=p.parse_args()
    if len(set(args.devices.split(',')))!=len(args.devices.split(',')) or any(not x for x in args.devices.split(',')):
        p.error('Distinct explicit devices required')
    if args.timeout<=0:p.error('Positive timeout required')
    run,contract,inputs,runtime=load_run(args.run);env=environment(run,contract,runtime,args.devices)
    if args.command=='preflight':return preflight(run,contract,inputs,runtime,env)
    pre=read(run/'preflight/report.json')
    if pre['status']!='PASS_CPU_IMPORTS' or pre['contract']!=artifact(run/'run_contract.json'):raise RuntimeError('Current preflight required')
    if args.cohort!='bridge':
        for policy in POLICIES:
            for mode in ('NR','R'):
                entry=run/'collections'/f'bridge_{mode}_{policy}'/'accepted.json'
                accepted=read(entry)
                if accepted['status']!='PASS':raise RuntimeError('Every bridge condition required')
                verify(accepted['audit'])
    def interrupt(signum,frame):raise KeyboardInterrupt('Evaluation interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    with (run/'active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with gpu_lease(run,Path(contract['source_worldengine_root']),args.holder_pid,args.pane):
            collection(run,contract,inputs,runtime,env,args.policy,args.cohort,args.mode,args.timeout)


if __name__=='__main__':main()
