"""Immutable stage-two inputs, shared optimizer parents and historical reuse."""
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_decision_common as x
import selector_decision_data as old_data
import selector_attribution_common as a


def checked_inputs(run):
    value = d.read(run/'attribution_inputs.json')
    if value['method'] != a.METHOD or value['status'] != 'PASS':
        raise RuntimeError('Invalid attribution inputs')
    for section in ('artifacts', 'models', 'parents', 'reused_collections'):
        for entry in value[section].values():
            d.verify(entry)
    for cohort in value['cohorts'].values():
        d.verify(cohort['scenario'])
    return value


def initial_rows(run):
    inputs = checked_inputs(run)
    rows = c.load_pickle(d.verify(inputs['artifacts']['initial_cache']))['rows']
    validate_rows(rows, inputs['excluded_logs'])
    return rows


def validate_rows(rows, excluded):
    if (len(rows) != 256 or sum(bool(r['fresh']) for r in rows) != 64
            or len({r['row_id'] for r in rows}) != 256
            or any(sum(r['mode'] == mode for r in rows) != 128 for mode in a.MODES)):
        raise RuntimeError('Fixed feedback coverage/order contract changed')
    for row in rows:
        x.check_row(row)
        if len(row['candidate_indices']) != 2 or row['origin_log'] in set(excluded):
            raise RuntimeError('Additional query labels or evaluation logs are forbidden')
        if f.fingerprint(row) != row['source_fingerprint']:
            raise RuntimeError('Frozen state features changed')
        if a.preference(row, 'closed') != row['preference']:
            raise RuntimeError('Original preference derivation changed')


def freeze(run, prior):
    from train_selector_decision import parent_entry
    from report_selector_cfpi_deployment import checked_collection
    run, prior = Path(run).resolve(), Path(prior).resolve()
    if run == prior or prior in run.parents or run in prior.parents:
        raise RuntimeError('Attribution run cannot overlap historical source')
    if (run/'attribution_inputs.json').exists():
        return checked_inputs(run)
    print('Verifying frozen decision-pilot inputs and original 256 training states', flush=True)
    old = old_data.checked_inputs(prior)
    contract = d.read(prior/'run_contract.json')
    report = d.read(prior/'pilot_report.json')
    if (contract['method'] != x.METHOD or report['status'] != 'PASS'
            or d.read(prior/'decision_ledger.json')['active'] is not None):
        raise RuntimeError('Require the complete stopped decision pilot, regardless of effect sign')
    for key in ('lr', 'weight_decay', 'grad_clip', 'temperature', 'dense_batch', 'feedback_batch', 'teacher_kl'):
        if contract['training'][key] != a.TRAIN[key]:
            raise RuntimeError('Historical optimization settings differ')
    if contract['training']['steps_per_round'] != 500 or contract['training']['winner_nll'] != 1.:
        raise RuntimeError('Historical step/NLL contract differs')
    rows = old_data.initial_rows(prior)
    validate_rows(rows, old['excluded_logs'])
    artifacts = {k: d.artifact(prior/p) for k,p in dict(
        source_inputs='decision_inputs.json', source_contract='run_contract.json',
        source_report='pilot_report.json', source_ledger='decision_ledger.json').items()}
    artifacts.update(initial_cache=old['artifacts']['initial_cache'],
                     dense_manifest=old['artifacts']['dense_manifest'],
                     feedback_inputs=old['artifacts']['source_inputs'])
    feedback_root = Path(old['prior_run'])
    target_entries = [d.artifact(feedback_root/p) for p in
                      ('feedback/round1/shared/targets.json', 'feedback/round2/T/targets.json')]
    targets = {}
    for index, entry in enumerate(target_entries):
        artifacts[f'source_targets_{index}'] = entry
        for target in d.verified_read(entry)['targets']:
            key = x.row_id(target)
            if key in targets:
                raise RuntimeError('Duplicate frozen target')
            targets[key] = dict(target, row_id=key, fresh=index == 1)
    if set(targets) != {r['row_id'] for r in rows}:
        raise RuntimeError('Targets do not exactly cover the original 256 states')
    for row in rows:
        target = targets[row['row_id']]
        for key in ('scene_id', 'mode', 'decision_step', 'candidate_indices',
                    'source_fingerprint', 'visiting_policy', 'continuation_policy', 'fresh'):
            if row[key] != target[key]:
                raise RuntimeError('Source target/cache identity mismatch: '+key)
        context = c.load_pickle(d.verify(target['context']))
        if (f.fingerprint(context) != row['source_fingerprint'] or
                any(not np.array_equal(np.asarray(row[k]), np.asarray(context[k]))
                    for k in f.FINGERPRINT_FIELDS)):
            raise RuntimeError('Source context does not match original cached state')
        for entry in (target['source_collection'], target['source_audit'], *target['prefix'].values()):
            d.verify(entry)
    c.locked_json(run/'data/targets.json', dict(status='PASS', targets=list(targets.values())))
    artifacts['targets'] = d.artifact(run/'data/targets.json')
    dense = d.verified_read(artifacts['dense_manifest'])
    if any(r['origin_log'] in set(old['excluded_logs']) for r in dense['records']):
        raise RuntimeError('Evaluation log in dense replay')
    for source in dense['sources'].values():
        for entry in source['arrays'].values():
            d.verify(entry)
    c.locked_json(run/'dense/manifest.json', dense)
    artifacts['local_dense_manifest'] = d.artifact(run/'dense/manifest.json')
    parents = {str(seed): parent_entry(prior, seed) for seed in a.SEEDS}
    models = {p: old_data.model_entry(prior, p) for p in ('scalar_v3', 'gate_v3', 'shared_seed0')}
    for arm in ('P0', 'P1'):
        for seed in a.SEEDS:
            policy = f'{arm}_seed{seed}'
            print('Verifying shared parent and historical optimizer lineage: '+policy, flush=True)
            previous = parents[str(seed)]
            for start, end in ((0,100), (100,300), (300,500)):
                path = prior/'train'/policy/f'step_{end}_report.json'
                result = d.read(path)
                p = result['provenance']
                if (result['status'] != 'PASS' or p['arm'] != arm or p['seed'] != seed
                        or p['start'] != start or p['end'] != end or p['generation'] != 0
                        or p['parent'] != previous or p['data'] != artifacts['initial_cache']
                        or p['training'] != contract['training'] or p['code_sha'] != contract['code_sha']):
                    raise RuntimeError('Cannot reuse historical P0/P1: parent/data/optimizer contract differs')
                d.verify(result['selector']); d.verify(result['resume'])
                artifacts[f'{policy}_segment_{end}'] = d.artifact(path)
                previous = result['resume']
            models[policy] = old_data.model_entry(prior, policy)
    reused = {}
    for cohort in ('development', 'common'):
        for mode in a.MODES:
            for policy in ('scalar_v3', 'gate_v3', *[f'{arm}_seed{s}' for arm in ('P0','P1') for s in a.SEEDS]):
                name = f'eval_{cohort}_{mode}_{policy}'
                print('Verifying historical evaluation: '+name, flush=True)
                entry = d.artifact(prior/'collections'/name/'deployment_collection.json')
                checked = checked_collection(entry)
                if checked is None:
                    raise RuntimeError('Historical evaluation incomplete: '+name)
                condition = checked['collection']
                routing = d.verified_read(condition['routing'])
                if (any(condition.get(k) != v for k,v in dict(cohort=cohort, react_type=mode,
                        policy=policy, research_method=x.METHOD, role='evaluation').items())
                        or condition['noise_namespace'] != c.NOISE_NAMESPACE
                        or condition['scenario'] != old['cohorts'][cohort]['scenario']
                        or routing['models'] != {policy: models[policy]}
                        or any('feedback_target' in r for r in routing['routes'])
                        or set(checked['metrics']) != {r['scene_id'] for r in old['cohorts'][cohort]['rows']}):
                    raise RuntimeError('Historical evaluation identity differs')
                reused[name] = entry
                artifacts[name+'_audit'] = checked['audit']
    value = dict(status='PASS', method=a.METHOD, prior_run=str(prior), artifacts=artifacts,
                 parents=parents, models=models, reused_collections=reused, cohorts=old['cohorts'],
                 excluded_logs=old['excluded_logs'], exposure=a.EXPOSURE, local_definition=a.LOCAL_DEFINITION,
                 optimization_seeds=list(a.SEEDS), independent_acquisition_replicates=False)
    c.locked_json(run/'attribution_inputs.json', value)
    return value


def model_entry(run, policy):
    inputs = checked_inputs(run)
    if policy in inputs['models']:
        return inputs['models'][policy]
    allowed = {f'{arm}_seed{s}' for arm in a.NEW_ARMS for s in a.SEEDS}
    if policy not in allowed:
        raise RuntimeError('Unregistered attribution model')
    report = d.read(run/'train'/policy/'step_500_report.json')
    p = report['provenance']
    if (report['status'] != 'PASS' or p['method'] != a.METHOD or p['policy'] != policy
            or p['code_sha'] != d.read(run/'run_contract.json')['code_sha']
            or p['inputs'] != d.artifact(run/'attribution_inputs.json')
            or p['parent'] != inputs['parents'][str(p['seed'])] or report['step'] != 500):
        raise RuntimeError('New model provenance mismatch')
    d.verify(report['selector'])
    return dict(report['selector'], kind='cfpi', score_mode='residual', provenance=p)


def load_rows(run):
    audit = d.read(run/'labels/audit.json')
    if (audit['status'] != 'PASS' or audit['inputs'] != d.artifact(run/'attribution_inputs.json')
            or audit['code_sha'] != d.read(run/'run_contract.json')['code_sha']
            or audit['local_definition'] != a.LOCAL_DEFINITION):
        raise RuntimeError('Same-state local label audit required before ALL new arms')
    for entry in audit['artifacts']:
        d.verify(entry)
    rows = c.load_pickle(d.verify(audit['cache']))['rows']
    original = initial_rows(run)
    validate_rows(rows, checked_inputs(run)['excluded_logs'])
    for old, new in zip(original, rows):
        if (any(new[k] != old[k] for k in ('row_id', 'candidate_indices', 'branch_outcomes',
                'visiting_policy', 'continuation_policy', 'fresh', 'origin_log'))
                or any(not np.array_equal(np.asarray(new[k]), np.asarray(old[k]))
                       for k in f.FINGERPRINT_FIELDS)):
            raise RuntimeError('Label attachment changed original order/features/closed-loop outcomes')
        a.preference(new, 'local')
    return rows, audit['cache']


def cpu_preflight(run):
    """Real cached forward/backward, objective parity; NO optimizer construction or steps."""
    import torch
    import selector_cfpi_model as m
    from train_selector_feedback import pair_loss
    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    print('Running real-cache CPU objective/gradient parity; optimizer steps remain zero', flush=True)
    inputs = checked_inputs(run)
    manifest = d.verified_read(inputs['artifacts']['feedback_inputs'])['artifacts']['scalar_manifest']
    model, _, _ = m.load_incumbent(d.verify(manifest))
    model.train().requires_grad_(True)
    rows = initial_rows(run)
    probe = [r for mode in a.MODES for r in [v for v in rows if v['mode']==mode][:8]]
    logits = m.score(model, probe, list(range(16)), 'cpu', 'residual')
    pair, nll = a.feedback_loss(logits, probe, 'P1')
    old_pair, old_nll = x.feedback_loss(logits, probe)
    if not (torch.equal(pair, pair_loss(logits, probe)) and torch.equal(pair, old_pair)
            and torch.equal(nll, old_nll)):
        raise RuntimeError('P0/P1 frozen objective operation parity failed')
    (pair+nll).backward()
    norm = float(m.v3.clip_grad_norm_cpu_(model.parameters(), 10.))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError('Actual-cache backward failed')
    result = dict(status='PASS', device='cpu', inputs=d.artifact(run/'attribution_inputs.json'),
                  code_sha=d.read(run/'run_contract.json')['code_sha'], objective_parity=True,
                  gradient_norm=norm, optimizer_steps=0, training_performed=False,
                  rollout_performed=False, local_labels_available=False,
                  note='Local scorer/runtime parity still requires the separately launched labels stage.')
    c.locked_json(run/'cached_preflight_cpu.json', result)
    return result
