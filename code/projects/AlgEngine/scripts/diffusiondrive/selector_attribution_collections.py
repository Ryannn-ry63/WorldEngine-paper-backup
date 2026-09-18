"""Training-only local observations, full replay parity and fixed evaluation bundle."""
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_decision_common as x
import selector_attribution_common as a
import selector_attribution_data as data
from build_selector_feedback import collection_data
from prepare_selector_cfpi_deployment import save_subset


def make(run, name, cohort, mode, policy, targets=None):
    inputs = data.checked_inputs(run)
    if mode not in a.MODES or cohort not in inputs['cohorts']:
        raise RuntimeError('Unregistered condition')
    if targets is not None and (cohort != 'train' or not targets):
        raise RuntimeError('Local observations are training-only')
    members = {r['scene_id'] for r in inputs['cohorts'][cohort]['rows']}
    scenes = sorted(t['scene_id'] for t in targets) if targets is not None else sorted(members)
    if not scenes or len(set(scenes)) != len(scenes) or not set(scenes) <= members:
        raise RuntimeError('Scene ownership or coverage mismatch')
    if targets and any(t['visiting_policy'] != policy or t['mode'] != mode for t in targets):
        raise RuntimeError('Original visiting policy/mode must be preserved')
    folder = run/'collections'/name
    scenario = inputs['cohorts'][cohort]['scenario']
    if set(scenes) != members:
        scenario = save_subset(folder/'scenarios.pkl', scenario, scenes)
    routing = dict(status='PASS', method=d.METHOD, research_method=a.METHOD, policy=policy,
                   models={policy: data.model_entry(run, policy)},
                   routes=[dict(scene_id=s, origin_token=c.scene_token(s), fold=None,
                                start_decision=4, model_key=policy) for s in scenes],
                   terminal_publication_decision=12,
                   incumbent_selector_sha256=inputs['models']['scalar_v3']['sha256'],
                   inference_uses_reward_or_q=False)
    c.locked_json(folder/'routing.json', routing)
    contract = d.read(run/'run_contract.json')
    result = dict(status='PASS', method=d.METHOD, research_method=a.METHOD, collection_id=name,
                  cohort=cohort, react_type=mode, policy=policy, continuation_policy=None,
                  role='local_observation' if targets is not None else 'evaluation',
                  sentinel=False, audit_targets={}, initial_contexts={},
                  routing=d.artifact(folder/'routing.json'), scenario=scenario,
                  asset_folder=f"{contract['source_worldengine_root']}/data/sim_engine/assets/{inputs['cohorts'][cohort]['asset_family']}/assets",
                  inputs=d.artifact(run/'attribution_inputs.json'), run_contract=d.artifact(run/'run_contract.json'),
                  checkpoint_sha256=c.CHECKPOINT_SHA256, noise_namespace=c.NOISE_NAMESPACE,
                  expected_decisions=list(c.DECISION_STEPS), worker_count=min(contract['gpu_count'], len(scenes)))
    if targets is not None:
        c.locked_json(folder/'local_targets.json', dict(status='PASS', targets=targets))
        result.update(local_targets=d.artifact(folder/'local_targets.json'), local_definition=a.LOCAL_DEFINITION)
    c.locked_json(folder/'deployment_collection.json', result)
    return d.artifact(folder/'deployment_collection.json')


def local_bundle(run):
    inputs = data.checked_inputs(run)
    targets = d.verified_read(inputs['artifacts']['targets'])['targets']
    entries = []
    for fresh, policy in ((False, 'scalar_v3'), (True, 'shared_seed0')):
        for mode in a.MODES:
            subset = [t for t in targets if t['fresh'] == fresh and t['mode'] == mode]
            if len(subset) != (32 if fresh else 96):
                raise RuntimeError('Local collection quota changed; never drop a target')
            entries.append(make(run, f'local_round{2 if fresh else 1}_{mode}', 'train', mode, policy, subset))
    value = dict(status='PASS', collections=entries, expected_states=256, no_intervention=True)
    c.locked_json(run/'labels/collections.json', value)
    return value


def check_replay(actual, original, scene):
    if (any(abs(actual['metrics'][scene][k]-original['metrics'][scene][k]) > c.OUTCOME_TOLERANCE
            for k in d.METRICS) or actual['metrics'][scene]['first_violation_step'] !=
            original['metrics'][scene]['first_violation_step']):
        raise RuntimeError('Observer changed scene outcome/timing')
    for step in c.DECISION_STEPS:
        left = c.load_pickle(d.verify(actual['records'][scene][step]))
        right = c.load_pickle(d.verify(original['records'][scene][step]))
        if (int(left['selected_index']) != int(right['selected_index']) or
                any(not np.array_equal(np.asarray(left[k]), np.asarray(right[k]))
                    for k in f.FINGERPRINT_FIELDS)):
            raise RuntimeError('Observer changed state/candidates/action at decision '+str(step))


def attach_local(row, record, target, entry):
    expected = dict(status='PASS', method=a.METHOD, row_id=row['row_id'], scene_id=row['scene_id'],
                    mode=row['mode'], decision_step=row['decision_step'],
                    source_fingerprint=row['source_fingerprint'], candidate_bank_sha=row['candidate_bank_sha'],
                    candidate_indices=row['candidate_indices'], selected_index=target['visiting_index'],
                    local_definition=a.LOCAL_DEFINITION, collection_sha256=entry['sha256'],
                    action_max_abs=0., inference_uses_labels=False)
    if any(record.get(k) != v for k,v in expected.items()):
        raise RuntimeError('Local label record identity differs')
    outcomes = a.local_outcomes(record['scores'], record['components'], row['candidate_indices'])
    if outcomes != record['pair_outcomes']:
        raise RuntimeError('Local pair labels do not match full20 scores')
    sidecar = c.load_pickle(d.verify(record['sidecar']))
    if (f.fingerprint(sidecar) != row['source_fingerprint']
            or x.candidate_bank_hash(sidecar) != row['candidate_bank_sha']):
        raise RuntimeError('Local sidecar fingerprint differs')
    return dict(row, local_pair_outcomes=outcomes)


def assemble_labels(run):
    if (run/'labels/audit.json').exists():
        data.load_rows(run)
        return d.read(run/'labels/audit.json')
    bundle = d.read(run/'labels/collections.json')
    rows = {r['row_id']: r for r in data.initial_rows(run)}
    attached, references, artifacts = {}, {}, [d.artifact(run/'labels/collections.json')]
    for entry in bundle['collections']:
        contract = d.verified_read(entry)
        root = Path(entry['path']).parent
        checked = collection_data(run, root.name)
        if (contract['research_method'] != a.METHOD or contract['role'] != 'local_observation'
                or contract['cohort'] != 'train' or contract['run_contract'] != d.artifact(run/'run_contract.json')):
            raise RuntimeError('Foreign local observation collection')
        targets = d.verified_read(contract['local_targets'])['targets']
        paths = sorted(root.glob('split_*/WE_output/openscene_format/attribution_local_records/*.json'))
        records = {}
        for path in paths:
            record = d.read(path)
            if record['row_id'] in records:
                raise RuntimeError('Duplicate local record')
            records[record['row_id']] = record
            artifacts.extend((d.artifact(path), record['sidecar']))
        if set(records) != {t['row_id'] for t in targets}:
            raise RuntimeError('Local record coverage differs; no target dropping allowed')
        for target in targets:
            key = target['row_id']
            if key in attached:
                raise RuntimeError('Local state counted twice')
            reference = target['source_collection']
            path = d.verify(reference)
            if str(path) not in references:
                references[str(path)] = collection_data(path.parents[2], path.parent.name)
                artifacts.extend((reference, references[str(path)]['audit']))
            original = references[str(path)]
            source_contract = d.verified_read(reference)
            source_models = d.verified_read(source_contract['routing'])['models']
            current_models = d.verified_read(contract['routing'])['models']
            policy = target['visiting_policy']
            if (source_contract['react_type'] != contract['react_type']
                    or source_contract['noise_namespace'] != contract['noise_namespace']
                    or source_models[policy]['sha256'] != current_models[policy]['sha256']):
                raise RuntimeError('Local/source policy, noise or mode differs')
            check_replay(checked, original, target['scene_id'])
            attached[key] = attach_local(rows[key], records[key], target, entry)
        artifacts.extend((entry, contract['local_targets'], checked['audit']))
    if set(attached) != set(rows):
        raise RuntimeError('Local labels incomplete')
    ordered = [attached[k] for k in rows]  # preserve the original sampler order
    path = run/'labels/paired_cache.pkl'
    c.atomic_pickle(path, dict(status='PASS', method=a.METHOD, rows=ordered))
    result = dict(status='PASS', inputs=d.artifact(run/'attribution_inputs.json'),
                  code_sha=d.read(run/'run_contract.json')['code_sha'], cache=d.artifact(path),
                  artifacts=artifacts, local_definition=a.LOCAL_DEFINITION,
                  summary=a.label_summary(ordered), states=256, replay_decisions_checked=256*8,
                  training_uses_original_pair_only=True, unknown_is_not_bad=True)
    c.locked_json(run/'labels/audit.json', result)
    return result


def evaluation_conditions():
    policies = ['scalar_v3', 'gate_v3']+[f'{arm}_seed{s}' for arm in a.ARMS for s in a.SEEDS]
    return [(cohort, mode, policy) for cohort in ('development','common') for mode in a.MODES for policy in policies]


def evaluation_bundle(run):
    data.load_rows(run)
    entries = []
    inputs = data.checked_inputs(run)
    for cohort, mode, policy in evaluation_conditions():
        name = f'eval_{cohort}_{mode}_{policy}'
        entries.append(inputs['reused_collections'][name] if name in inputs['reused_collections']
                       else make(run, name, cohort, mode, policy))
    value = dict(status='PASS', collections=entries, expected_conditions=80, historical_conditions=32,
                 new_conditions=48, full_rare288_authorized=False)
    c.locked_json(run/'evaluation_collections.json', value)
    return value

