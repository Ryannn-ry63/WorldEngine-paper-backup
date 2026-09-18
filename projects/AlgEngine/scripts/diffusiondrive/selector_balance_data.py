"""Read-only historical lineage; new immutable manifests and endpoint exports."""
import hashlib
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_attribution_common as a
import selector_attribution_data as ad
import selector_balance_common as b
import selector_snapshot_common as s
from report_selector_cfpi_deployment import checked_collection


def checked_inputs(run):
    run = Path(run)
    value = d.read(run/'balance_inputs.json')
    b.require(value['method'] == b.METHOD and value['status'] == 'PASS', 'Wrong balance manifest')
    for group in ('artifacts', 'parents', 'selector_models', 'reused_dev', 'reused_formal'):
        for entry in value[group].values():
            d.verify(entry)
    for cohort in value['cohorts'].values():
        d.verify(cohort['scenario'])
    return value


def load_rows(run):
    inputs = checked_inputs(run)
    rows, cache = ad.load_rows(Path(inputs['attribution_run']))
    b.require(cache == inputs['artifacts']['training_cache'], 'Training cache drift')
    return rows, cache


def model_entry(run, policy):
    inputs = checked_inputs(run)
    if policy in inputs['selector_models']:
        return inputs['selector_models'][policy]
    b.require(policy in {f'{arm}_seed{s}' for arm in b.NEW_ARMS for s in b.SEEDS}, 'Unknown policy')
    result = d.read(Path(run)/'train'/policy/'step_500_report.json')
    p = result['provenance']
    arm, seed = policy.rsplit('_seed', 1)
    expected = dict(method=b.METHOD, policy=policy, arm=arm, seed=int(seed),
                    inputs=d.artifact(Path(run)/'balance_inputs.json'),
                    parent=inputs['parents'][seed], training=b.TRAIN,
                    code_sha=d.read(Path(run)/'run_contract.json')['code_sha'],
                    components=b.specification(arm, scale(run, int(seed)) if arm == 'MID_SCALE_MATCH' else 1.))
    b.require(result['status'] == 'PASS' and result['step'] == 500
              and all(p.get(k) == v for k,v in expected.items()), 'Model provenance drift')
    d.verify(result['selector']); d.verify(result['resume'])
    return dict(result['selector'], kind='cfpi', score_mode='residual', provenance=p)


def scale(run, seed):
    value = d.read(Path(run)/'scale_audit.json')
    b.require(value['status'] == 'PASS' and value['inputs'] == d.artifact(Path(run)/'balance_inputs.json')
              and value['code_sha'] == d.read(Path(run)/'run_contract.json')['code_sha'], 'Scale audit drift')
    item = value['seeds'][str(seed)]
    b.require(item['alpha'] == b.matched_scale(item['numerator'], item['denominator']), 'Scale changed')
    return item['alpha']


def native_compatibility(snapshot, root):
    """Allow the already-audited router-factory change only in an inactive branch."""
    contract = d.read(snapshot/'run_contract.json')
    files = contract['implementation']
    verified = {}
    ignored = 'projects/SimEngine/worldengine/manager/diffusiondrive_cfpi_deployment_manager.py'
    prefixes = ('projects/AlgEngine/mmdet3d_plugin/', 'projects/AlgEngine/closed_loop/',
                'projects/SimEngine/worldengine/')
    for relative, expected in files.items():
        if not relative.startswith(prefixes) or relative == ignored:
            continue
        path = root/relative
        actual = c.sha256_file(path)
        if actual != expected and relative == 'projects/SimEngine/worldengine/envs/base_env.py':
            text = path.read_text().replace('deployment_manager,', 'DiffusionDriveCFPIDeploymentManager,')
            text = text.replace('reward_manager = deployment_manager()', 'reward_manager = DiffusionDriveCFPIDeploymentManager()')
            b.require(hashlib.sha256(text.encode()).hexdigest() == expected, 'Unreviewed engine change')
        else:
            b.require(actual == expected, 'Native inference/simulator drift: '+relative)
        verified[relative] = dict(historical=expected, current=actual)
    b.require(len(verified) > 20, 'Native compatibility inventory incomplete')
    d.verify(contract['scorer'])
    return dict(status='PASS', files=verified, scorer=contract['scorer'],
                inactive_router_branch_only=True, native_dense_reward_manager=False)


def freeze(run, attribution, snapshot, root):
    from selector_snapshot_audit import checked
    from selector_snapshot_tasks import checked_open
    from selector_snapshot_data import export_model
    run, attribution, snapshot, root = map(lambda p: Path(p).resolve(), (run, attribution, snapshot, root))
    b.disjoint(run, (attribution, snapshot))
    if (run/'balance_inputs.json').exists():
        return checked_inputs(run)
    ar = d.read(attribution/'attribution_report.json')
    sr = d.read(snapshot/'summary.json')
    b.require(ar['status'] == sr['status'] == 'PASS', 'Sources incomplete')
    for path in (attribution/'decision_ledger.json', snapshot/'ledger.json'):
        b.require(d.read(path)['active'] is None, 'Source is still active')
    print(dict(stage='freeze', detail='Verifying historical inputs, labels and endpoint models'), flush=True)
    old = ad.checked_inputs(attribution)
    rows, cache = ad.load_rows(attribution)
    native = s.checked_inputs(snapshot)
    b.require(len(rows) == 256 and len(native['cohorts']['full']) == 288
              and len(native['cohorts']['development']) == 58 and len(native['cohorts']['common']) == 64,
              'Cohort size drift')
    for cohort in ('development', 'common'):
        b.require(old['cohorts'][cohort]['rows'] == native['cohorts'][cohort], 'Formal/dev cohort mismatch')
    ac = d.read(attribution/'run_contract.json')
    b.require(ac['training'] == a.TRAIN and ac['seeds'] == list(b.SEEDS), 'Endpoint training protocol drift')
    models = {p: ad.model_entry(attribution, p) for p in b.FORMAL_POLICIES}
    b.require(models['P1_seed2']['sha256'] == s.SELECTOR_SHA, 'Snapshot selected model drift')
    feedback_inputs = d.verified_read(old['artifacts']['feedback_inputs'])
    scalar_manifest_entry = feedback_inputs['artifacts']['scalar_manifest']
    scalar_manifest = d.verified_read(scalar_manifest_entry)
    b.require(models['scalar_v3']['sha256'] == scalar_manifest['scene_selector_state_sha256']
              and native['models']['scalar_v3']['sha256'] == scalar_manifest['checkpoint_sha256'] == c.CHECKPOINT_SHA256,
              'Selector-only and full V3 must resolve to the same frozen manifest')
    artifacts = dict(attribution_inputs=d.artifact(attribution/'attribution_inputs.json'),
        attribution_report=d.artifact(attribution/'attribution_report.json'),
        attribution_contract=d.artifact(attribution/'run_contract.json'),
        attribution_ledger=d.artifact(attribution/'decision_ledger.json'),
        snapshot_inputs=d.artifact(snapshot/'inputs.json'), snapshot_report=d.artifact(snapshot/'summary.json'),
        snapshot_contract=d.artifact(snapshot/'run_contract.json'), snapshot_ledger=d.artifact(snapshot/'ledger.json'),
        training_cache=cache, feedback_inputs=old['artifacts']['feedback_inputs'],
        dense_manifest=old['artifacts']['dense_manifest'], scalar_manifest=scalar_manifest_entry)
    for key, entry in models.items():
        s.preserved_copy(d.verify(entry), run/'archive/selectors'/key/Path(entry['path']).name)
    for seed, entry in old['parents'].items():
        s.preserved_copy(d.verify(entry), run/'archive/parents'/seed/Path(entry['path']).name)
    dense = d.verified_read(artifacts['dense_manifest'])
    for source in dense['sources'].values():
        for entry in source['arrays'].values():
            d.verify(entry)
    c.locked_json(run/'dense/manifest.json', dense)
    anchor = root/'kmeans_navsim_traj_20.npy'
    old_anchor = attribution.parents[4]/'kmeans_navsim_traj_20.npy'
    b.require(c.sha256_file(anchor) == c.sha256_file(old_anchor), 'Candidate anchor changed')
    artifacts['candidate_anchor'] = d.artifact(anchor)
    reused = {}
    for cohort, mode, policy in b.dev_conditions():
        if policy.startswith(b.NEW_ARMS):
            continue
        name = f'eval_{cohort}_{mode}_{policy}'
        entry = old['reused_collections'].get(name)
        if entry is None:
            entry = d.artifact(attribution/'collections'/name/'deployment_collection.json')
        result = checked_collection(entry)
        b.require(result is not None, 'Missing endpoint development result')
        co = result['collection']
        routing = d.verified_read(co['routing'])
        b.require(co['cohort'] == cohort and co['react_type'] == mode and co['policy'] == policy
                  and co['role'] == 'evaluation' and co['noise_namespace'] == c.NOISE_NAMESPACE
                  and co['scenario'] == old['cohorts'][cohort]['scenario']
                  and routing['models'] == {policy: models[policy]}
                  and not any('feedback_target' in r for r in routing['routes']), 'Endpoint evaluation identity drift')
        b.validate_metrics(result['metrics'], old['cohorts'][cohort]['rows'])
        reused[name] = entry
    for cohort in ('development', 'common'):
        for mode in b.MODES:
            name = f'eval_{cohort}_{mode}_scalar_v3'
            reused[name] = old['reused_collections'][name]
            b.require(checked_collection(reused[name]) is not None, 'Missing fixed V3 reference')
    print(dict(stage='freeze', detail='Verifying 12 historical native conditions'), flush=True)
    reuse_formal = {}
    for policy in ('scalar_v3', 'P1_seed2'):
        for block in b.BLOCKS:
            if block.startswith('open'):
                result = checked_open(snapshot, policy, block)
                path = snapshot/'openloop'/policy/block/'audit.json'
            else:
                path = snapshot/'collections'/f'{block}_{policy}'/'audit.json'
                result = checked(path.parent, full=True)
            b.require(result is not None, 'Missing historical formal condition')
            reuse_formal[policy+'__'+block] = d.artifact(path)
    compatibility = native_compatibility(snapshot, root)
    c.locked_json(run/'archive/native_compatibility.json', compatibility)
    artifacts['compatibility'] = d.artifact(run/'archive/native_compatibility.json')
    exported = {p: native['models'][p] for p in ('scalar_v3', 'P1_seed2')}
    exports = {}
    for policy in b.NEW_NATIVE:
        print(dict(stage='native_export', policy=policy), flush=True)
        exports[policy] = export_model(d.verify(native['models']['scalar_v3']), d.verify(models[policy]),
            run/'models'/policy/'checkpoint.pth',
            dict(source_selector=models[policy], exposure=b.EXPOSURE), expected_policy=policy, method=b.METHOD)
        exported[policy] = exports[policy]['checkpoint']
    c.locked_json(run/'archive/native_exports.json', exports)
    native_artifacts = {k: native['artifacts'][k] for k in
                        ('full_scenario', 'common_scenario', 'bridge_scenario')}
    for policy in b.NEW_NATIVE:
        for mode in b.MODES:
            reference = checked_collection(reused[f'eval_development_{mode}_{policy}'])
            native_artifacts[f'bridge_reference_{policy}_{mode}'] = reference['audit']
    c.locked_json(run/'inputs.json', dict(method=b.METHOD, artifacts=native_artifacts, models=exported,
        cohorts=native['cohorts'], open_tokens=native['open_tokens'], assets=native['assets'], exposure=b.EXPOSURE))
    result = dict(status='PASS', method=b.METHOD, attribution_run=str(attribution), snapshot_run=str(snapshot),
        artifacts=artifacts, parents=old['parents'], selector_models=models, reused_dev=reused,
        reused_formal=reuse_formal, cohorts=old['cohorts'], excluded_logs=old['excluded_logs'], exposure=b.EXPOSURE)
    c.locked_json(run/'balance_inputs.json', result)
    return result


def scale_audit(run):
    import torch
    from selector_dual_domain_probes import state_hash
    inputs = checked_inputs(run)
    rows, cache = load_rows(run)
    contract = d.read(Path(run)/'run_contract.json')
    if (Path(run)/'scale_audit.json').exists():
        for seed in b.SEEDS:
            scale(run, seed)
        return d.read(Path(run)/'scale_audit.json')
    torch.set_num_threads(4)
    manifest = d.verified_read(inputs['artifacts']['feedback_inputs'])['artifacts']['scalar_manifest']
    results = {}
    with torch.random.fork_rng(devices=[]):
        teacher, _, _ = m.load_incumbent(d.verify(manifest))
        teacher.eval().requires_grad_(False)
        for seed in b.SEEDS:
            parent = torch.load(d.verify(inputs['parents'][str(seed)]), map_location='cpu')
            b.require(parent['step'] == 500 and parent['provenance']['arm'] == 'shared'
                      and parent['provenance']['seed'] == seed, 'Wrong scale parent')
            model = m.initialize(teacher, 'residual').eval().requires_grad_(True)
            model.load_state_dict(parent['model'], strict=True)
            before = state_hash(model)
            parameters = tuple(model.parameters())
            numerator = denominator = 0.
            batches = []
            for start in range(0, 256, 16):
                subset = rows[start:start+16]
                z = m.score(model, subset, list(range(16)), 'cpu', 'residual')
                pair, nll = a.feedback_loss(z, subset, 'P1')
                p = torch.autograd.grad(pair, parameters, retain_graph=True, allow_unused=True)
                n = torch.autograd.grad(nll, parameters, allow_unused=True)
                energy = [0., 0.]
                for param, gp, gn in zip(parameters, p, n):
                    gp = (torch.zeros_like(param) if gp is None else gp).detach().double()
                    gn = (torch.zeros_like(param) if gn is None else gn).detach().double()
                    energy[0] += float((gn+gp).square().sum())
                    energy[1] += float((gn+.5*gp).square().sum())
                numerator += energy[0]; denominator += energy[1]
                batches.append(dict(row_ids=[r['row_id'] for r in subset], numerator=energy[0], denominator=energy[1]))
            b.require(state_hash(model) == before and all(p.grad is None for p in parameters),
                      'Scale probe updated parameters or accumulated gradients')
            results[str(seed)] = dict(parent=inputs['parents'][str(seed)], numerator=numerator,
                denominator=denominator, alpha=b.matched_scale(numerator, denominator), batches=batches)
            print(dict(stage='scale_probe', seed=seed, alpha=results[str(seed)]['alpha']), flush=True)
    result = dict(status='PASS', inputs=d.artifact(Path(run)/'balance_inputs.json'),
        code_sha=contract['code_sha'], seeds=results, optimizer_steps=0, training_rng_unchanged=True,
        note='Initial eval-mode gradient RMS only, not full optimization-trajectory matching.')
    c.locked_json(Path(run)/'scale_audit.json', result)
    return result
