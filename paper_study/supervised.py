"""Prepare actual expert labels and distinct planner L2/original-BC baselines."""
import os
from pathlib import Path
import subprocess
import sys
from .common import CODE,artifact,legacy_imports,lock,paths,read,verify,write

def worker_env():
    legacy_imports()
    from run_selector_cfpi import environment
    source=Path(paths()['legacy_root'])
    env=environment(source)
    # Config builds absolute dataset/image paths from WORLDENGINE_ROOT. The clean
    # code checkout intentionally has no data directory; use the registered assets.
    env['WORLDENGINE_ROOT']=str(source)
    env['NAVFORMER_DISABLE_EFFICIENT_SDP']='1'
    env['PYTHONPATH']=str(CODE)+os.pathsep+env['PYTHONPATH']
    return env

def prepare(run):
    from .data import checked
    m=checked(run)
    if m['queue']!='D':raise ValueError('Supervised queue required')
    env=worker_env()
    subprocess.run([sys.executable,'-m','paper_study.supervised_worker','--run',str(run)],cwd=CODE,env=env,check=True)
    return read(run/'supervised_inputs.json')

def train(run,arm,seed,devices,smoke=False):
    if arm not in ('planner_l2','planner_bc'):raise ValueError('Unknown planner baseline')
    from .data import checked
    m=checked(run);inputs=read(run/'supervised_inputs.json')
    if inputs['manifest']!=artifact(run/'manifest.json'):raise RuntimeError('Supervised inputs drifted')
    env=worker_env();env['CUDA_VISIBLE_DEVICES']=devices
    config=verify(inputs['configs'][arm]);folder=run/('planner_smoke' if smoke else 'planner')/f'{arm}_seed{seed}'
    with lock(folder/'train.lock'):
        if (folder/'report.json').exists():
            old=read(folder/'report.json');verify(old['checkpoint']);return old
        from .evaluation import available_port
        env['MASTER_PORT']=str(available_port())
        alg=CODE/'projects/AlgEngine'
        cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=4','--master_port',env['MASTER_PORT'],
             str(alg/'scripts/train.py'),str(config),'--launcher','pytorch','--seed',str(seed),'--work-dir',str(folder),'--no-validate']
        if smoke:cmd+=['--cfg-options','runner.max_iters=8','checkpoint_config.interval=8']
        latest=folder/'latest.pth'
        if latest.exists():cmd+=['--resume-from',str(latest)]
        subprocess.run(cmd,cwd=alg,env=env,check=True)
        checkpoint=folder/f"iter_{8 if smoke else m['training']['steps']*inputs['accumulation']}.pth"
        if not checkpoint.exists():raise RuntimeError('Expected fixed supervised endpoint missing')
        import torch
        from materialize_grpo_selector_v3 import state_dict
        state=state_dict(torch.load(checkpoint,map_location='cpu'))
        base=state_dict(torch.load(verify(inputs['base']),map_location='cpu'))
        if any(not torch.equal(v,state[k]) for k,v in base.items() if not k.startswith('planning_head.')):
            raise RuntimeError('Frozen perception changed in exported planner')
        if smoke:
            result=dict(status='PASS',stage='PLANNER_SMOKE_NOT_FORMAL',checkpoint=artifact(checkpoint),frozen_perception_verified=True)
            write(folder/'report.json',result);return result
        # Reuse native observer with scene selector disabled; generator changes are explicitly declared.
        native=run/'native'/f'{arm}_seed{seed}';native.mkdir(parents=True,exist_ok=True)
        training=folder/'report.json'
        result=dict(status='PASS',stage='TRAINED_PLANNER_NOT_EVALUATED',checkpoint=artifact(checkpoint),
                    provenance=dict(manifest=artifact(run/'manifest.json'),arm=arm,seed=seed,config=inputs['configs'][arm]),
                    frozen_perception_verified=True,generator_changed=True)
        write(training,result)
        write(native/'report.json',dict(status='PASS',policy=f'{arm}_seed{seed}',checkpoint=artifact(checkpoint),
            config=dict(paper_original_selector=True,paper_generator_changed=True),
            provenance=dict(training=artifact(training),base=inputs['base']),generator_changed=True))
        return result
