"""Three frozen MID exports, bridge parity, and native full evaluation."""
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_mid_followup_common as b
import selector_mid_followup_data as data
import selector_snapshot_common as s
import selector_snapshot_audit as audit
from selector_snapshot_tasks import checked_open


def make_collection(run, policy, cohort, mode):
    run = Path(run)
    b.require(policy in b.NEW_NATIVE and cohort in ('bridge', 'full', 'common') and mode in b.MODES,
              'Unregistered native condition')
    inputs = s.checked_inputs(run)
    name = f'{cohort}_{mode}_{policy}'
    rows = inputs['cohorts'][cohort]
    prefixes = {}
    for row in rows:
        for prefix in (row['scene_id'], row['scene_id'].rsplit('-', 1)[1]):
            b.require(prefix not in prefixes or prefixes[prefix] == row['scene_id'], 'Ambiguous scene prefix')
            prefixes[prefix] = row['scene_id']
    value = dict(method=b.METHOD, id=name, policy=policy, cohort=cohort, mode=mode,
        checkpoint=inputs['models'][policy], scenario=inputs['artifacts'][cohort+'_scenario'],
        rows=rows, prefixes=prefixes, asset_folder=inputs['assets'][cohort],
        noise=c.NOISE_NAMESPACE if cohort == 'bridge' else b.FORMAL_NOISE,
        contract=d.artifact(run/'run_contract.json'), inputs=d.artifact(run/'inputs.json'), workers=4)
    path = run/'collections'/name/'collection.json'
    c.locked_json(path, value)
    return path


def parity(run, device='cpu'):
    import torch
    import selector_cfpi_model as m
    from materialize_grpo_selector_v3 import state_dict
    torch.set_num_threads(4)
    source = data.checked_inputs(run)
    inputs = s.checked_inputs(run)
    rows, _ = data.load_rows(run)
    results = {}
    for policy in b.NEW_NATIVE:
        print(dict(stage='native_cached_parity', policy=policy, device=device), flush=True)
        model, payload = m.load_selector(d.verify(source['selector_models'][policy]))
        checkpoint = state_dict(torch.load(d.verify(inputs['models'][policy]), map_location='cpu'))
        restored = m.v3.model_from_config(payload['scene_selector_config'])
        restored.load_state_dict({k[len(s.PREFIX):]:v for k,v in checkpoint.items() if k.startswith(s.PREFIX)}, strict=True)
        del checkpoint
        cpu_reference = m.load_selector(d.verify(source['selector_models'][policy]))[0].eval() if device.startswith('cuda') else None
        model.to(device).eval(); restored.to(device).eval()
        error = 0.
        cross_device_error = 0.
        with torch.no_grad():
            for start in range(0, len(rows), 16):
                batch = rows[start:start+16]
                left = m.score(model, batch, list(range(len(batch))), device, 'residual')
                right = m.score(restored, batch, list(range(len(batch))), device, 'residual')
                current = float((left-right).abs().max())
                b.require(np.isfinite(current) and current <= c.MODEL_RECOMPUTE_TOLERANCE
                          and torch.equal(left.argmax(-1), right.argmax(-1)), 'Native/cache parity failed: '+policy)
                error = max(error, current)
                if cpu_reference is not None:
                    cpu = m.score(cpu_reference, batch, list(range(len(batch))), 'cpu', 'residual')
                    delta = float((cpu-right.cpu()).abs().max())
                    b.require(np.isfinite(delta) and delta <= c.MODEL_RECOMPUTE_TOLERANCE
                        and torch.equal(cpu.argmax(-1),right.cpu().argmax(-1)), 'CPU/CUDA parity failed: '+policy)
                    cross_device_error = max(cross_device_error,delta)
        results[policy] = dict(max_abs=error, argmax_mismatches=0, states=len(rows),
                               source=source['selector_models'][policy], native=inputs['models'][policy],
                               cpu_cuda_max_abs=cross_device_error if cpu_reference is not None else None)
    result = dict(status='PASS', device=device, models=results, optimizer_steps=0,
        inputs=d.artifact(Path(run)/'inputs.json'), code_sha=d.read(Path(run)/'run_contract.json')['code_sha'])
    c.locked_json(Path(run)/f'cached_parity_{device}.json', result)
    return result


def compare_bridge(native, old):
    from selector_feedback_common import FINGERPRINT_FIELDS
    from audit_selector_cfpi_deployment import action_error
    records = {}
    for entry in old['artifacts']:
        if '/cfpi_deployment_records/' in entry['path']:
            rec = d.verified_read(entry)
            if rec['scene_id'] in native['metrics']:
                key = (rec['scene_id'], rec['state_step']+1)
                b.require(key not in records, 'Duplicate historical bridge record')
                records[key] = rec
    for scene, values in native['metrics'].items():
        b.require(scene in old['metrics'] and all(abs(values[k]-old['metrics'][scene][k]) <= c.OUTCOME_TOLERANCE
                                                 for k in d.METRICS), 'Native bridge outcome changed')
        for step in c.DECISION_STEPS:
            n = d.verified_read(native['records'][scene][str(step)]['record'])
            o = records[(scene, step)]
            left = c.load_pickle(d.verify(n['sidecar']))
            right = c.load_pickle(d.verify(o['sidecar']))
            b.require(int(left['selected_index']) == int(right['selected_index']), 'Native bridge selection changed')
            for key in FINGERPRINT_FIELDS:
                b.require(np.max(np.abs(np.asarray(left[key])-np.asarray(right[key]))) <= c.ARRAY_TOLERANCE,
                          'Native bridge state/candidates changed: '+key)
            b.require(np.max(np.abs(left['current_logits']-right['current_logits'])) <= c.MODEL_RECOMPUTE_TOLERANCE
                      and action_error(n['actual_action'], o['actual_action']) <= c.ACTION_TOLERANCE
                      and np.max(np.abs(left['deployed_trajectory']-right['deployed_trajectory'])) <= c.ACTION_TOLERANCE,
                      'Native bridge scores/action changed')


def bridge_report(run):
    inputs = s.checked_inputs(run)
    artifacts = []
    for policy in b.NEW_NATIVE:
        for mode in b.MODES:
            root = Path(run)/'collections'/f'bridge_{mode}_{policy}'
            native = audit.checked(root, full=True)
            old = d.verified_read(inputs['artifacts'][f'bridge_reference_{policy}_{mode}'])
            compare_bridge(native, old)
            artifacts.append(d.artifact(root/'audit.json'))
    result = dict(status='PASS', conditions=artifacts, inputs=d.artifact(Path(run)/'inputs.json'),
                  noise=c.NOISE_NAMESPACE, scenes_per_condition=8, expected_conditions=6)
    c.locked_json(Path(run)/'bridge_report.json', result)
    return result


def require_bridge(run):
    result = d.read(Path(run)/'bridge_report.json')
    b.require(result['status'] == 'PASS' and result['inputs'] == d.artifact(Path(run)/'inputs.json')
              and len(result['conditions']) == 6, 'All three MID models require both bridge modes')
    for entry in result['conditions']:
        d.verify(entry)


def checked_formal(run, policy, block):
    b.require((policy, block) in b.formal_conditions(), 'Unknown formal evaluation')
    inputs = data.checked_inputs(run)
    reused = inputs['reused_formal'].get(policy+'__'+block)
    source = Path(inputs['snapshot_run']) if reused else Path(run)
    path = (source/'openloop'/policy/block/'audit.json' if block.startswith('open')
            else source/'collections'/f'{block}_{policy}'/'audit.json')
    if reused:
        b.require(reused == d.artifact(path), 'Historical formal artifact changed')
    if not path.exists():
        return None
    result = checked_open(source, policy, block) if block.startswith('open') else audit.checked(path.parent, full=True)
    if result is not None:
        result = dict(result, evidence=d.artifact(path), historical_reuse=bool(reused))
    return result


def openloop(run, policy, block):
    b.require(policy in b.NEW_NATIVE and block in ('open_navtest', 'open_rare'), 'Unauthorized open-loop job')
    require_bridge(run)
    previous = checked_open(run, policy, block)
    if previous is not None:
        return previous
    inputs = s.checked_inputs(run)
    alg = Path(os.environ['ALGENGINE_ROOT'])
    config = alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_mid_followup_native.py'
    script = alg/'scripts'/('e2e_dist_eval.sh' if block == 'open_navtest' else 'e2e_dist_eval_navtest_failures.sh')
    folder = Path(run)/'openloop'/policy/block
    attempts = folder/'attempts'
    attempts.mkdir(parents=True, exist_ok=True)
    current = Path(tempfile.mkdtemp(prefix='attempt_', dir=attempts))
    checkpoint = s.preserved_copy(d.verify(inputs['models'][policy]), current/'checkpoint.pth')
    env = dict(os.environ, SELECTOR_SNAPSHOT_NOISE=b.FORMAL_NOISE, SELECTOR_SNAPSHOT_MODE='OPEN',
        SELECTOR_SNAPSHOT_OPEN='1', SELECTOR_SNAPSHOT_CONDITION=policy+'_'+block,
        NAVSIM_OFFICIAL_RESCORE='always', PYTHON_BIN=os.environ['ALGENGINE_PYTHON'],
        MASTER_PORT='28620' if block == 'open_navtest' else '28621')
    subprocess.run(['bash', str(script), str(config), checkpoint['path'], '4'], cwd=alg, env=env, check=True)
    files = list((current/'test').glob('*_official_pdms/pdm_scores_merged.csv'))
    b.require(len(files) == 1, 'Expected exactly one official PDM output')
    values = s.metrics(files[0], inputs['open_tokens'][block], valid=True)
    raw = Path(str(files[0].parent).removesuffix('_official_pdms')+'.csv')
    rows = s.real_rows(raw, inputs['open_tokens'][block])
    ade = {}
    if block == 'open_navtest':
        for key in ('ade_4s', 'fde_4s'):
            array = np.asarray([float(r[key]) for r in rows])
            b.require(np.isfinite(array).all(), 'Nonfinite ADE/FDE')
            ade[key] = float(array.mean())
    result = dict(status='PASS', policy=policy, block=block, noise=b.FORMAL_NOISE,
        checkpoint=inputs['models'][policy], contract=d.artifact(Path(run)/'run_contract.json'),
        pdm=d.artifact(files[0]), raw=d.artifact(raw), metrics=values, ade_fde=ade, count=len(values),
        artifacts=[d.artifact(files[0]), d.artifact(raw), checkpoint])
    c.locked_json(folder/'audit.json', result)
    return result
