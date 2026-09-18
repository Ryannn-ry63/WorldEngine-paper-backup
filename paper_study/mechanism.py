"""Full20 counterfactual diagnostics using audited prefix replay and fixed continuation.

Default input is historical fresh-round training states. These are exposed diagnostics,
not a new holdout. A caller can freeze a different audited specification explicitly.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .common import artifact,code_identity,legacy_imports,lock,paths,read,verify,write


def default_spec():
    source=Path(paths()['attribution_run']);inputs=read(source/'attribution_inputs.json')
    targets=read(verify(inputs['artifacts']['targets']))['targets']
    feedback=read(verify(inputs['artifacts']['feedback_inputs']))
    return dict(schema=1,purpose='historical_training_diagnostic',independent_blind_test=False,
        source_inputs=artifact(source/'attribution_inputs.json'),
        targets=[t for t in targets if t['fresh']],
        cohort=inputs['cohorts']['train'],models={k:inputs['models'][k] for k in ('scalar_v3','shared_seed0')},
        checkpoint_manifest=feedback['artifacts']['scalar_manifest'],excluded_logs=inputs['excluded_logs'],
        labels_quarantined_from_training=True)


def prepare(run,manifest=None,probe=False):
    legacy_imports()
    import cfpi_common as c
    import selector_feedback_common as f
    spec=read(manifest) if manifest else default_spec()
    if spec['purpose']!='historical_training_diagnostic' or spec['independent_blind_test'] is not False:
        raise ValueError('This diagnostic adapter is not a blind-test acquisition pipeline')
    if not spec['labels_quarantined_from_training']:raise ValueError('Mechanism labels must remain diagnostic')
    if probe:
        selected=[]
        for mode in ('NR','R'):
            group=sorted([t for t in spec['targets'] if t['mode']==mode],key=lambda t:t['row_id'])[:4]
            if len(group)!=4:raise RuntimeError('Probe needs four states per mode')
            selected.extend(group)
        spec=dict(spec,targets=selected)
    members={r['scene_id']:r for r in spec['cohort']['rows']};seen=set()
    verify(spec['checkpoint_manifest']);verify(spec['cohort']['scenario'])
    for model in spec['models'].values():verify(model)
    if not spec['targets']:raise ValueError('No diagnostic states')
    for t in spec['targets']:
        key=t['row_id'];row=members[t['scene_id']]
        if key in seen or row['origin_log'] in spec['excluded_logs']:raise ValueError('Duplicate or excluded state')
        seen.add(key)
        if t['visiting_policy']!=t['continuation_policy']:raise ValueError('Default sentinel requires matching visiting/continuation policy')
        if t['visiting_policy'] not in spec['models'] or t['mode'] not in ('NR','R'):raise ValueError('Unknown policy/mode')
        context=c.load_pickle(verify(t['context']))
        if f.fingerprint(context)!=t['source_fingerprint'] or int(context['selected_index'])!=t['visiting_index']:
            raise RuntimeError('Source state/action changed')
        for entry in [t['source_collection'],t['source_audit'],*t['prefix'].values()]:verify(entry)
    from .evaluation import prepare as prepare_runtime
    prepare_runtime(run)
    with lock(run/'mechanism_prepare.lock'):
        write(run/'mechanism.json',dict(status='PREPARED_NOT_COLLECTED',code=code_identity(),spec=spec,
            specification=artifact(manifest) if manifest else None,candidates=[0] if probe else list(range(20)),transport_probe=probe,
            subset_sizes=[2,4,8,20],subset_seed=0,subset_repeats=100))
        groups={}
        for t in spec['targets']:
            key=(t['mode'],t['visiting_policy'],Path(t['source_collection']['path']).parent.name)
            groups.setdefault(key,[]).append(t)
        bundles=[]
        for slot,((mode,policy,_),targets) in enumerate(sorted(groups.items())):
            sentinel=make(run,spec,f'sentinel_{slot}_{mode}',targets,mode,policy,None)
            references={t['source_collection']['path'] for t in targets}
            if len(references)!=1:raise RuntimeError('Ambiguous replay reference')
            treatments=[make(run,spec,f'full20_{slot}_{mode}_a{j:02d}',targets,mode,policy,j) for j in ([0] if probe else range(20))]
            bundles.append(dict(sentinel=sentinel,reference=targets[0]['source_collection'],treatments=treatments,continuation=policy))
        write(run/'mechanism_collections.json',dict(bundles=bundles))
    return dict(status='PREPARED_NOT_COLLECTED',states=len(seen),queries=len(seen)*(1 if probe else 20),transport_probe=probe,independent_blind_test=False)


def make(run,spec,name,targets,mode,policy,index):
    import cfpi_common as c
    import selector_cfpi_deployment_common as d
    from prepare_selector_cfpi_deployment import save_subset
    folder=run/'collections'/name
    routes=[]
    for t in targets:
        target=dict(t,forced_index=t['visiting_index'] if index is None else index,
                    sentinel=index is None,no_intervention=False)
        routes.append(dict(scene_id=t['scene_id'],origin_token=c.scene_token(t['scene_id']),fold=None,
            start_decision=4,model_key=policy,feedback_target=target,continuation_model_key=policy))
    scenes=[r['scene_id'] for r in routes]
    if len(scenes)!=len(set(scenes)):raise ValueError('Repeated scene in collection; split states into separate slots')
    routing=dict(status='PASS',method=d.METHOD,research_method='selector_decision_feedback_v1',policy=name,
        models={policy:spec['models'][policy]},routes=routes,terminal_publication_decision=12,
        incumbent_selector_sha256=spec['models']['scalar_v3']['sha256'],inference_uses_reward_or_q=False)
    write(folder/'routing.json',routing)
    membership=hashlib.sha256(json.dumps(sorted(scenes)).encode()).hexdigest()[:20]
    scenario_path=run/'scenarios'/membership/'scenarios.pkl'
    scenario_path.parent.mkdir(parents=True,exist_ok=True)
    scenario=save_subset(scenario_path,spec['cohort']['scenario'],scenes)
    contract=dict(status='PASS',method=d.METHOD,research_method='selector_decision_feedback_v1',collection_id=name,
        cohort='train',react_type=mode,policy=policy,continuation_policy=policy,role='sentinel' if index is None else 'diagnostic',
        sentinel=False,audit_targets={},initial_contexts={},routing=artifact(folder/'routing.json'),scenario=scenario,
        asset_folder=str(Path(paths()['legacy_root'])/'data/sim_engine/assets'/spec['cohort']['asset_family']/'assets'),
        inputs=artifact(run/'mechanism.json'),run_contract=artifact(run/'run_contract.json'),
        checkpoint_sha256=c.CHECKPOINT_SHA256,noise_namespace=c.NOISE_NAMESPACE,
        expected_decisions=list(c.DECISION_STEPS),worker_count=min(4,len(scenes)))
    write(folder/'deployment_collection.json',contract)
    return artifact(folder/'deployment_collection.json')


def collect(run,devices,probe=False):
    from .evaluation import load,environment,preflight
    from .queue import child
    legacy_imports()
    import run_selector_cfpi_deployment as deployment
    from selector_decision_collections import compare_sentinel,outcomes
    engine,run,contract,inputs,runtime=load(run)
    preflight(run,devices)
    m=read(run/'mechanism.json')
    if bool(m.get('transport_probe',False))!=probe:raise RuntimeError('Probe/full collection identity differs')
    if m['code']!=code_identity():raise RuntimeError('Mechanism implementation changed')
    env=environment(engine,run,contract,runtime,devices,'scalar_v3')
    class Adapter:
        def __init__(self):
            self.env=env;self.args=SimpleNamespace(gpus=4,source_worldengine_root=Path(paths()['legacy_root']))
            self.config=runtime/'projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_selector_decision_feedback.py'
            self.alg=runtime/'projects/AlgEngine';self.sim=runtime/'projects/SimEngine'
            self.checkpoint_manifest=verify(m['spec']['checkpoint_manifest']);self.code_sha=contract['code_sha']
            self.collection_timeout=21600
        def record(self,*items):print(*items,flush=True)
        def python(self,script,arguments,stage,env=None,timeout=900):
            script=runtime/'projects/AlgEngine/scripts/diffusiondrive'/Path(script).name
            engine.execute([([self.env['ALGENGINE_PYTHON'],str(script),*map(str,arguments)],self.alg,env or self.env,
                            run/'logs'/f'{stage}.log')],run/'logs',timeout)
        def execute(self,jobs,stage,gpus,timeout):
            updated=[]
            for cmd,cwd,e,log in jobs:
                cmd=list(map(str,cmd))
                if 'run_simulation.py' in cmd[1]:
                    cmd += [f"worker.threads_per_node={contract['ray_cpu_threads']}",
                            f"number_of_cpus_allocated_per_simulation={contract['simulation_cpu_threads']}"]
                updated.append((cmd,cwd,e,log))
            engine.execute(updated,run/'collections'/stage,timeout)
    runner=Adapter();bundles=read(run/'mechanism_collections.json')['bundles'];labels={};evidence=[]
    with lock(run/'active.lock'):
        for b in bundles:
            deployment.Runner.collection(runner,b['sentinel'])
            evidence.append(compare_sentinel(dict(sentinel=b['sentinel'],reference=b['reference'])))
            for entry in b['treatments']:
                deployment.Runner.collection(runner,entry)
                values,audit=outcomes(entry,b['continuation']);evidence.append(audit)
                for key,value in values.items():
                    if value['source_fingerprint']!=next(t['source_fingerprint'] for t in m['spec']['targets'] if t['row_id']==key):
                        raise RuntimeError('Counterfactual source identity changed')
                    labels.setdefault(key,{})[value['candidate_index']]=value['outcome']
        if probe:
            if set(labels)!={t['row_id'] for t in m['spec']['targets']} or any(set(v)!={0} for v in labels.values()):raise RuntimeError('Incomplete transport probe')
            result=dict(status='PASS_TRANSPORT_PROBE_ONLY',states=len(labels),queried_candidate=0,full20_complete=False,evidence=evidence)
            write(run/'transport_probe_report.json',result)
            return result
        result=summarize(labels,m['spec']['targets'],m['subset_seed'],m['subset_repeats'])
        write(run/'full20_labels.json',dict(status='PASS',labels=labels,evidence=evidence,labels_quarantined_from_training=True))
        write(run/'mechanism_report.json',dict(result,labels=artifact(run/'full20_labels.json'),independent_blind_test=False))
    return result


def summarize(labels,targets,seed=0,repeats=100):
    if set(labels)!={t['row_id'] for t in targets}:raise ValueError('Missing diagnostic states')
    rng=np.random.default_rng(seed);detail=[]
    for t in targets:
        row={int(k):v for k,v in labels[t['row_id']].items()}
        if set(row)!=set(range(20)):raise ValueError('Full20 requires every actual branch outcome')
        scores=np.array([row[j]['score'] for j in range(20)])
        if not np.isfinite(scores).all():raise ValueError('Nonfinite branch outcomes')
        baseline=scores[t['visiting_index']];bounds={}
        for k in (2,4,8,20):
            choices=[rng.choice(20,k,replace=False) for _ in range(repeats)]
            bounds[str(k)]=dict(mean_best_return=float(np.mean([scores[x].max() for x in choices])),
                probability_cover_best=float(np.mean([scores[x].max()==scores.max() for x in choices])))
        pair=t['candidate_indices']
        detail.append(dict(row_id=t['row_id'],mode=t['mode'],scene_id=t['scene_id'],fixed_continuation=t['continuation_policy'],
            visiting_return=float(baseline),full20_best_return=float(scores.max()),
            original_pair_best_return=float(scores[pair].max()),unqueried_better_than_pair=bool(scores.max()>scores[pair].max()+1e-8),
            any_safe_candidate=any(row[j]['no_at_fault_collisions']==1 and row[j]['drivable_area_compliance']==1 for j in range(20)),subsets=bounds))
    return dict(status='PASS_DIAGNOSTIC',states=len(detail),details=detail,
        interpretation='Fixed-state, fixed-continuation branch oracle; not a realizable full-policy upper bound',
        unqueried_better_fraction=float(np.mean([r['unqueried_better_than_pair'] for r in detail])))
