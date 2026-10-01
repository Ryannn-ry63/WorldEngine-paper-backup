"""Checksummed exports and exact-coverage evaluation acceptance."""
import csv
import json
import math
from pathlib import Path
import pickle
import statistics
import time
from experiment_common import WORK,CODE,TOOLS,REGISTRY,SETTINGS,read,write,sha,lock,configured_environment,command

METRICS=('no_at_fault_collisions','drivable_area_compliance','ego_progress',
         'time_to_collision_within_bound','comfort','score')
SAFETY=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def coverage(path,expected,aggregate,valid=False,metrics=METRICS):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    actual=[r for r in rows if r['token']!=aggregate];mean=[r for r in rows if r['token']==aggregate]
    tokens=[r['token'] for r in actual]
    if len(mean)!=1 or len(tokens)!=len(set(tokens)) or set(tokens)!=set(expected) or len(expected)!=len(set(expected)):
        raise ValueError('Missing/duplicate/unexpected tokens: '+str(path))
    for row in rows:
        if valid and row.get('valid','').lower()!='true':raise ValueError('Invalid score row: '+row['token'])
        if not all(math.isfinite(float(row[k])) for k in metrics):raise ValueError('Non-finite score')
        if metrics==METRICS and not all(0<=float(row[k])<=1 for k in metrics):raise ValueError('Invalid PDM range')
    return {r['token']:{k:float(r[k]) for k in metrics} for r in actual}


def stage_files(out,mode):
    out=Path(out)
    if mode.startswith('ol_'):
        files=list(out.with_name(out.name+'_inputs').glob('test/*_official_pdms/pdm_scores_merged.csv'))
        if len(files)!=1:raise ValueError('Expected exactly one official PDM csv')
        result=dict(pdm=files[0])
        if mode=='ol_navtest':result['ade']=Path(str(files[0].parent).removesuffix('_official_pdms')+'.csv')
        return result
    return dict(pdm=out/'simulation/WE_output/openscene_format'/('all_scenes_pdm_averages_'+mode[3:]+'.csv'))


def accept_stage(out,mode,expected):
    if read(Path(out)/'process_status.json')['status']!='PASS_PROCESSES_ONLY':raise ValueError('Processes failed')
    files=stage_files(out,mode)
    rows=coverage(files['pdm'],expected,'average' if mode.startswith('ol_') else 'overall_average',mode.startswith('ol_'))
    if 'ade' in files:coverage(files['ade'],expected,'average',metrics=('ade_4s','fde_4s'))
    return dict(status='PASS_EXACT_COVERAGE',rows=rows,counts=len(rows),
        means={k:statistics.mean(r[k] for r in rows.values()) for k in METRICS},
        files={k:str(v) for k,v in files.items()},files_sha256={k:sha(v) for k,v in files.items()})


def prepare_development():
    output=REGISTRY/'development'
    with lock(REGISTRY/'development.prepare.lock'):
        if (output/'prepared.json').exists():
            result=read(output/'prepared.json');verify_prepared(result);return result
        manifest=read(REGISTRY/'scene_manifest.json');cfg,env=configured_environment()
        rows=[r for r in manifest['scenes'] if r['split']=='development']
        from innovation3.scene_payload import single_scene_mapping
        output.mkdir(exist_ok=True);assets=output/'assets';assets.mkdir(exist_ok=True)
        scenes={}
        for row in rows:
            with open(row['scene_path'],'rb') as f:mapping=single_scene_mapping(pickle.load(f),row['scene_id'])
            if set(scenes)&set(mapping):raise ValueError('Duplicate development scenes')
            scenes.update(mapping)
            root=Path(row['asset_path']).parent.parent
            link=assets/row['scene_id']
            if not link.exists():link.symlink_to(root,target_is_directory=True)
            if link.resolve()!=root.resolve():raise ValueError('Wrong development asset link')
        target=output/'scenarios.pkl'
        temp=target.with_suffix('.tmp')
        with temp.open('wb') as f:pickle.dump(scenes,f,protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(target)
        config=output/'frozen_eval_config.py'
        config.write_text('_base_ = [%r]\nmodel = dict(planning_head=dict(online_reward=None, export_rollout_context=False))\n' %
            str(CODE/'projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v3.py'))
        result=dict(status='PREPARED_NOT_EVALUATED',scenarios=str(target),scenarios_sha256=sha(target),
            assets=str(assets),config=str(config),config_sha256=sha(config),
            manifest_sha256=sha(REGISTRY/'scene_manifest.json'),scenes=rows,
            development_is_independent_final_test=False)
        write(output/'prepared.json',result,immutable=True)
        return result


def verify_prepared(prepared):
    if prepared['manifest_sha256']!=sha(REGISTRY/'scene_manifest.json'):raise ValueError('Development manifest changed')
    for k in ('scenarios','config'):
        if sha(prepared[k])!=prepared[k+'_sha256']:raise ValueError('Development artifact changed: '+k)


def frozen_policies():
    path=REGISTRY/'frozen_policies.json'
    if path.exists():
        result=read(path)
    else:
        origin=read(REGISTRY/'origin.json');old=Path(origin['old64_progress'])
        evaldir=old.with_name(old.name.removesuffix('.progress.json')+'.eval64')
        prepared=read(evaldir/'prepared.json')
        cfg=read(SETTINGS)
        result={'parent':dict(prepared['policies']['parent'],selector=cfg['selector_state']),
                'online64':dict(prepared['policies']['online64'],selector=str(evaldir/'online64_selector.pt'))}
        for policy in result.values():policy['selector_sha256']=sha(policy['selector'])
        write(path,result,immutable=True)
    for policy in result.values():
        if sha(policy['path'])!=policy['sha256'] or sha(policy['selector'])!=policy['selector_sha256']:
            raise ValueError('Frozen parent/online64 artifact changed')
    return result


def audit_marker(marker_path):
    from innovation3.ddp_session_boundary import receipt,validate_round
    marker=read(marker_path);spec=marker['contract'];completed=marker['completed'];world=spec['world_size']
    output=Path(str(marker_path).split('.coverage')[0]);progress=read(output.with_suffix('.progress.json'))
    if progress['contract']!=spec or progress['completed']<completed or world!=8:
        raise ValueError('Invalid committed coverage boundary')
    records=progress['rounds'][completed//world-1]
    if records!=marker['final_records'] or marker['coverage']['minimum_passes']<marker['passes']:
        raise ValueError('Incomplete coverage or wrong boundary records')
    validate_round(records,spec,completed//world-1)
    import torch
    from innovation3.live_learning import state_digest
    parities=[]
    for rank,record in enumerate(records):
        if sha(record['report'])!=record['report_sha256']:raise ValueError('Committed report changed')
        previous=progress['rounds'][completed//world-2][rank] if completed>world else None
        if receipt(record['report'],spec,completed//world-1,rank,previous)!=record:
            raise ValueError('Committed receipt failed')
        payload=torch.load(record['checkpoint']['path'],map_location='cpu')
        if payload['source_sha256']!=spec['source_sha256'] or payload['learner']['attempts']!=completed:
            raise ValueError('Checkpoint source or update count differs')
        if payload.get('online_update_timing','per_step')!=spec['online_update_timing']:
            raise ValueError('Checkpoint timing differs')
        parities.append({k:state_digest(payload['learner'][k]) for k in ('selector','reference','optimizer')})
    if any(v!=parities[0] for v in parities):raise ValueError('All-rank checkpoint parity failed')
    return records[0]['checkpoint']['path']


def materialize_policy(policy):
    if policy in ('parent','online64'):return frozen_policies()[policy]
    # Names are train_A_s0_p3 etc; there is no mutable "latest" checkpoint.
    prefix,passes=policy.rsplit('_p',1)
    marker=REGISTRY/'jobs'/prefix/('training.coverage'+passes+'.json')
    folder=REGISTRY/'policies'/policy;receipt=folder/'prepared.json'
    try:handle=lock(REGISTRY/'policies'/(policy+'.lock'))
    except BlockingIOError:return None
    with handle:
        if receipt.exists():
            result=read(receipt)
            if result['marker_sha256']!=sha(marker) or sha(result['path'])!=result['sha256']:
                raise ValueError('Prepared policy changed')
            return result
        checkpoint=audit_marker(marker)
        cfg,env=configured_environment();env['CUDA_VISIBLE_DEVICES']=''
        attempt=folder/('attempt_'+str(time.time_ns()));attempt.mkdir(parents=True)
        selector=attempt/'selector.pt'
        command([cfg['algengine_python'],'-u','-m','innovation3.export_online_selector_v3',
            '--checkpoint',checkpoint,'--offline-selector',cfg['selector_state'],
            '--output',selector,'--manifest',attempt/'export.json'],env)
        target=attempt/'checkpoint.pth'
        command([cfg['algengine_python'],'-u',TOOLS/'materialize_grpo_selector_v3.py',
            '--baseline',cfg['baseline'],'--scene-selector-state',selector,
            '--expected-selector-sha256',sha(selector),'--expected-method','scene_conditioned_exact_group_grpo',
            '--release-name',policy,'--output',target,'--manifest',attempt/'materialize.json'],env)
        command([cfg['algengine_python'],'-u',TOOLS/'audit_grpo_selector_v3_checkpoint.py',
            '--baseline',cfg['baseline'],'--checkpoint',target,'--output',attempt/'frozen_baseline_audit.json'],env)
        result=dict(path=str(target),sha256=sha(target),selector=str(selector),selector_sha256=sha(selector),
            marker=str(marker),marker_sha256=sha(marker),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
        write(receipt,result,immutable=True)
        return result
