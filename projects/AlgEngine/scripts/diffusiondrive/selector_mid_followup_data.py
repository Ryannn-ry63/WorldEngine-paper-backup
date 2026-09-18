"""Consume completed MID study read-only; freeze independently owned follow-up inputs."""
from pathlib import Path
import copy
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_balance_data as old
import selector_balance_collections as old_dev
import selector_mid_followup_common as b
import selector_snapshot_common as s
from selector_snapshot_data import export_model


def checked_inputs(run):
    value = d.read(Path(run)/'followup_inputs.json')
    b.require(value['method'] == b.METHOD and value['status'] == 'PASS', 'Wrong follow-up inputs')
    for group in ('artifacts','parents','selector_models','reused_dev','reused_formal'):
        for entry in value[group].values():
            d.verify(entry)
    for cohort in value['cohorts'].values():
        d.verify(cohort['scenario'])
    return value


def load_rows(run):
    inputs = checked_inputs(run)
    entry = inputs['artifacts']['training_cache']
    rows = c.load_pickle(d.verify(entry))['rows']
    from selector_attribution_data import validate_rows
    validate_rows(rows,inputs['excluded_logs'])
    return rows,entry


def model_entry(run, policy):
    inputs = checked_inputs(run)
    if policy in inputs['selector_models']:
        return inputs['selector_models'][policy]
    b.require(policy in {f'{a}_seed{s}' for a in b.NEW_ARMS for s in b.SEEDS}, 'Unknown model')
    result = d.read(Path(run)/'train'/policy/'step_500_report.json')
    arm,seed = policy.rsplit('_seed',1)
    expected = dict(method=b.METHOD,policy=policy,arm=arm,seed=int(seed),
        inputs=d.artifact(Path(run)/'followup_inputs.json'),parent=inputs['parents'][seed],
        training=b.TRAIN,code_sha=d.read(Path(run)/'run_contract.json')['code_sha'],components=b.specification(arm))
    b.require(result['status']=='PASS' and result['step']==500
              and all(result['provenance'].get(k)==v for k,v in expected.items()),'Model provenance drift')
    d.verify(result['selector']);d.verify(result['resume'])
    return dict(result['selector'],kind='cfpi',score_mode='residual',provenance=result['provenance'])


def freeze(run, source, root):
    run,source,root = map(lambda p:Path(p).resolve(),(run,source,root))
    b.disjoint(run,(source,))
    if (run/'followup_inputs.json').exists():
        return checked_inputs(run)
    report = d.read(source/'summary.json')
    b.require(report['status']=='PASS' and report['task']=='development'
              and report['complete_training_jobs']==6 and report['complete_dev_conditions']==48
              and not report['missing'] and d.read(source/'ledger.json')['active'] is None,
              'Completed and stopped MID study required')
    release = d.read(source/'archive/release_manifest.json')
    for item in (release['summary'],*release['outputs']):d.verify(item)
    inputs = old.checked_inputs(source)
    models = dict(inputs['selector_models'])
    for arm in ('MID','MID_SCALE_MATCH'):
        for seed in b.SEEDS:
            policy=f'{arm}_seed{seed}'
            models[policy]=old.model_entry(source,policy)
    reused = dict(inputs['reused_dev'])
    for cohort in ('development','common'):
        for mode in b.MODES:
            for arm in ('MID','MID_SCALE_MATCH'):
                for seed in b.SEEDS:
                    policy=f'{arm}_seed{seed}'
                    result=old_dev.checked(source,cohort,mode,policy)
                    b.require(result is not None,'Missing MID development condition')
                    name=f'eval_{cohort}_{mode}_{policy}'
                    reused[name]=d.artifact(source/'collections'/name/'deployment_collection.json')
    dense = d.read(source/'dense/manifest.json')
    for block in dense['sources'].values():
        for entry in block['arrays'].values():d.verify(entry)
    c.locked_json(run/'dense/manifest.json',dense)
    for name,entry in models.items():
        s.preserved_copy(d.verify(entry),run/'archive/selectors'/name/Path(entry['path']).name)
    for name,entry in inputs['parents'].items():
        s.preserved_copy(d.verify(entry),run/'archive/parents'/name/Path(entry['path']).name)
    # The simulator/model code is identical; only explicit method allowlists are extended.
    source_root=source.parents[4]
    snapshot_contract=d.read(Path(inputs['snapshot_run'])/'run_contract.json')
    verified={}
    for relative in snapshot_contract['implementation']:
        if relative.startswith(('projects/AlgEngine/mmdet3d_plugin/','projects/AlgEngine/closed_loop/',
                                'projects/SimEngine/worldengine/')):
            p=root/relative;q=source_root/relative
            b.require(c.sha256_file(p)==c.sha256_file(q),'Inference implementation differs: '+relative)
            verified[relative]=d.artifact(p)
    b.require(len(verified)>20,'Missing inference compatibility inventory')
    c.locked_json(run/'archive/inference_compatibility.json',verified)
    artifacts=dict(inputs['artifacts'],source_mid_report=d.artifact(source/'summary.json'),
        source_mid_contract=d.artifact(source/'run_contract.json'),source_mid_inputs=d.artifact(source/'balance_inputs.json'),
        source_mid_release=d.artifact(source/'archive/release_manifest.json'),
        compatibility=d.artifact(run/'archive/inference_compatibility.json'))
    b.require(c.sha256_file(root/'kmeans_navsim_traj_20.npy')==artifacts['candidate_anchor']['sha256'],'Anchor changed')
    native=s.checked_inputs(source)
    native_models={p:native['models'][p] for p in ('scalar_v3','P1_seed2')}
    native_artifacts={k:v for k,v in native['artifacts'].items() if not k.startswith('bridge_reference_')}
    if b.task_from_run(run)=='mid-formal':
        for policy in b.NEW_NATIVE:
            print(dict(stage='native_export',policy=policy),flush=True)
            result=export_model(d.verify(native_models['scalar_v3']),d.verify(models[policy]),
                run/'models'/policy/'checkpoint.pth',dict(source_selector=models[policy],exposure=b.EXPOSURE),
                expected_policy=policy,method=b.METHOD)
            c.locked_json(run/'archive'/f'{policy}_export.json',result)
            native_models[policy]=result['checkpoint']
            for mode in b.MODES:
                native_artifacts[f'bridge_reference_{policy}_{mode}']=old_dev.checked(source,'development',mode,policy)['audit']
    c.locked_json(run/'inputs.json',dict(method=b.METHOD,models=native_models,artifacts=native_artifacts,
        cohorts=native['cohorts'],open_tokens=native['open_tokens'],assets=native['assets'],exposure=b.EXPOSURE))
    value=dict(status='PASS',method=b.METHOD,source_run=str(source),attribution_run=inputs['attribution_run'],
        snapshot_run=inputs['snapshot_run'],artifacts=artifacts,parents=inputs['parents'],selector_models=models,
        cohorts=inputs['cohorts'],excluded_logs=inputs['excluded_logs'],reused_dev=reused,
        reused_formal={k:v for k,v in inputs['reused_formal'].items() if k.startswith('scalar_v3__')})
    c.locked_json(run/'followup_inputs.json',value)
    return value
