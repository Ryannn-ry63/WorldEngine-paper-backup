"""Use the audited factorial native ABI in a separately frozen evaluation run."""
from .hardware import hardware as check_hardware
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import numpy as np
from .common import CODE, WORKSPACE, artifact, code_identity, legacy_imports, lock, read, run_path, sha, verify, write

SOURCE='20260916_factorial_bridges_4h100_v1'
METHOD='selector_prefeedback_factorial_eval_v1'
CONFIG='projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_selector_factorial_native.py'

def prepare(run,training_run=None,allocation=None):
    from .storage import ensure_run,snapshot
    ensure_run(run)
    legacy_imports()
    import run_study_evaluation as engine
    if allocation is None:
        import run_study_bridges_4h100 as four
        allocation=four.resources(four.cpu_capacity())
    source,contract,inputs,runtime=engine.load_run(SOURCE)
    for mode in ('NR','R'):
        p=source/'collections'/f'bridge_{mode}_scalar_v3'/'accepted.json'
        a=read(p)
        if not a.get('scalar_historical_parity'):raise RuntimeError('Accepted H100 scalar bridge required')
        verify(a['audit'])
    models=dict(inputs['models']) if training_run is None else {'scalar_v3':inputs['models']['scalar_v3']}
    configs={k:{} for k in models};extra={}
    if training_run is not None:
        for p in sorted((training_run/'native').glob('*/report.json')):
            r=read(p)
            if r['status']!='PASS':raise RuntimeError('Invalid native export')
            verify(r['checkpoint']);models[r['policy']]=r['checkpoint'];configs[r['policy']]=r['config'];extra[r['policy']]=artifact(p)
        if len(models)==1:raise RuntimeError('No native exports; train/export first')
    with lock(run/'prepare.lock'):
        target=run/'runtime'
        snapshot(runtime,target,contract['implementation'])
        config=target/CONFIG
        text=(runtime/CONFIG).read_text()+'''\n# Per-model scientific configuration; method name retains the validated native ABI.\nimport json\n_paper = json.loads(os.environ.get('PAPER_SELECTOR_CONFIG', '{}'))\nif _paper:\n    if _paper.get('paper_original_selector'):\n        model['planning_head']['scene_selector'] = None\n    else:\n        _paper = dict(_paper)\n        _unary = _paper.pop('paper_unary_matched', False)\n        model['planning_head']['scene_selector'] = dict(_delete_=True, **_paper)\n        if _unary:\n            custom_imports = dict(imports=['mmdet3d_plugin', 'paper_study.runtime_plugin'])\n            model['planning_head']['type'] = 'PaperUnaryPlanningHead'\n            model['planning_head']['scene_selector']['use_set_attention'] = True\n'''
        if (run/'run_contract.json').exists() and config.read_text()!=text:raise RuntimeError('Frozen config changed')
        config.write_text(text)
        # Freeze the adapter package as well; subprocesses must never import changing research code.
        dst=target/'paper_study'
        for p in (CODE/'paper_study').glob('*.py'):
            q=dst/p.name;q.parent.mkdir(exist_ok=True)
            if q.exists() and q.read_bytes()!=p.read_bytes():raise RuntimeError('Runtime adapter changed')
            if not q.exists():shutil.copy2(p,q)
        inventory=dict(contract['implementation']);inventory[CONFIG]=sha(config)
        for p in dst.glob('*.py'):inventory[str(p.relative_to(target))]=sha(p)
        updated=dict(contract,runtime=str(target),implementation=inventory,
            code_sha=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest(),
            paper_code=code_identity(),scientific_study='paper_parallel_20260917_v1',model_configs=configs,
            source_evaluation=artifact(source/'run_contract.json'),native_exports=extra,
            workers=4,gpu_count=4,full_condition_count=len(models)*6,
            ray_cpu_threads=32,simulation_cpu_threads=8)
        if allocation:updated.update(allocation)
        values=dict(inputs,models=models)
        write(run/'inputs.json',values);write(run/'run_contract.json',updated)
    return dict(status='PREPARED_NOT_EVALUATED',models=list(models),run=str(run))

def load(run):
    legacy_imports()
    import run_study_evaluation as engine
    r,c,i,rt=engine.load_run(run.name)
    if c['paper_code']!=code_identity():raise RuntimeError('Paper driver changed; freeze a new evaluation run')
    # The simulator records the manifest path verbatim, while the auditor resolves
    # artifact paths. Submit the physical run path so symlinked HDD2 runs agree.
    return engine,r.resolve(),c,i,rt

def environment(engine,run,contract,runtime,devices,policy):
    env=engine.environment(run,contract,runtime,devices)
    # Explicit paths prevent accidentally executing the mutable clean checkout.
    alg=runtime/'projects/AlgEngine';sim=runtime/'projects/SimEngine'
    env.update(WORLDENGINE_ROOT=str(runtime),ALGENGINE_ROOT=str(alg),SIMENGINE_ROOT=str(sim),
        PAPER_SELECTOR_CONFIG=json.dumps(contract['model_configs'][policy],sort_keys=True),
        PYTHONPATH=os.pathsep.join([str(runtime),str(alg),str(sim),str(alg/'scripts/diffusiondrive'),env.get('PYTHONPATH','')]))
    return env

def preflight(run,devices):
    engine,run,c,i,rt=load(run)
    env=environment(engine,run,c,rt,devices,'scalar_v3')
    engine.preflight(run,c,i,rt,env)
    import run_study_bridges_4h100 as four
    check_hardware(four.devices(devices))
    four.cuda_preflight(run,c,i,rt,devices)
    return dict(status='PASS_PREFLIGHT')

def require_bridge(run,policy):
    for p in ('scalar_v3',policy):
        for mode in ('NR','R'):
            a=read(run/'collections'/f'bridge_{mode}_{p}'/'accepted.json')
            if a['status']!='PASS':raise RuntimeError('Both scalar and learned bridges required')
            verify(a['audit'])

def closed(run,policy,cohort,mode,devices,timeout=10800):
    engine,run,c,i,rt=load(run)
    if policy not in i['models'] or cohort not in ('bridge','full','common') or mode not in ('NR','R'):raise ValueError('Unknown condition')
    pre=read(run/'preflight/cuda_report.json')
    if pre['status']!='PASS' or pre['contract']!=artifact(run/'run_contract.json'):raise RuntimeError('Current CUDA preflight required')
    if cohort!='bridge':require_bridge(run,policy)
    with lock(run/'active.lock'):
        env=environment(engine,run,c,rt,devices,policy)
        engine.collection(run,c,i,rt,env,policy,cohort,mode,timeout)

def available_port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]

def openloop(run,policy,block,devices,timeout=14400):
    engine,run,c,i,rt=load(run)
    if policy not in i['models'] or block not in ('open_navtest','open_rare'):raise ValueError('Unknown condition')
    # Queue A can reuse the unchanged native models' accepted source bridges.
    source=Path(c['source_evaluation']['path']).parent
    if not c['native_exports']:
        require_bridge(source,policy)
    else:require_bridge(run,policy)
    import selector_snapshot_common as s
    folder=run/'openloop'/policy/block
    with lock(run/'active.lock'),lock(folder/'run.lock'):
        if (folder/'audit.json').exists():return check_open(run,policy,block)
        attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
        current=Path(tempfile.mkdtemp(prefix='attempt_',dir=attempts))
        checkpoint=s.preserved_copy(verify(i['models'][policy]),current/'checkpoint.pth')
        env=environment(engine,run,c,rt,devices,policy)
        env.update(SELECTOR_SNAPSHOT_NOISE='formal_navtest_seed0',SELECTOR_SNAPSHOT_MODE='OPEN',SELECTOR_SNAPSHOT_OPEN='1',
                   SELECTOR_SNAPSHOT_CONDITION=policy+'_'+block,NAVSIM_OFFICIAL_RESCORE='always',PYTHON_BIN=env['ALGENGINE_PYTHON'],
                   MASTER_PORT=str(available_port()))
        alg=rt/'projects/AlgEngine'
        script=alg/'scripts'/('e2e_dist_eval.sh' if block=='open_navtest' else 'e2e_dist_eval_navtest_failures.sh')
        command=['bash',str(script),str(rt/CONFIG),checkpoint['path'],'4']
        engine.execute([(command,alg,env,current/'logs/openloop.log')],current,timeout)
        files=list((current/'test').glob('*_official_pdms/pdm_scores_merged.csv'))
        if len(files)!=1:raise RuntimeError('Expected exactly one official PDM output')
        values=s.metrics(files[0],i['open_tokens'][block],valid=True)
        raw=Path(str(files[0].parent).removesuffix('_official_pdms')+'.csv')
        rows=s.real_rows(raw,i['open_tokens'][block]);ade={}
        if block=='open_navtest':
            for key in ('ade_4s','fde_4s'):
                x=np.asarray([float(r[key]) for r in rows])
                if not np.isfinite(x).all():raise RuntimeError('Nonfinite ADE/FDE')
                ade[key]=float(x.mean())
        result=dict(status='PASS',policy=policy,block=block,checkpoint=i['models'][policy],
            contract=artifact(run/'run_contract.json'),noise='formal_navtest_seed0',pdm=artifact(files[0]),raw=artifact(raw),
            metrics=values,ade_fde=ade,count=len(values),artifacts=[artifact(files[0]),artifact(raw),checkpoint])
        write(folder/'audit.json',result);return result

def check_open(run,policy,block):
    _,_,_,i,_=load(run)
    a=read(run/'openloop'/policy/block/'audit.json')
    if (a['status']!='PASS' or a['checkpoint']!=i['models'][policy] or a['contract']!=artifact(run/'run_contract.json')
        or a['count']!=len(i['open_tokens'][block]) or a['noise']!='formal_navtest_seed0'):raise RuntimeError('Open-loop provenance differs')
    for e in a['artifacts']:verify(e)
    import selector_snapshot_common as s
    if a['metrics']!=s.metrics(verify(a['pdm']),i['open_tokens'][block],True):raise RuntimeError('Open-loop metrics changed')
    return a
