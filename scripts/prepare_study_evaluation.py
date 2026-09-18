"""Freeze an independently owned simulator runtime without changing trained code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selector_study.common import CODE, artifact, read, sha, verify, write
from selector_study.data import checked

METHOD = 'selector_prefeedback_factorial_eval_v1'
DEFAULT_ID = '20260916_factorial_eval_v2'


def prepare(name):
    if not name or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in name):
        raise ValueError('Invalid evaluation ID')
    workspace = CODE.parent
    run = workspace/'runs'/name
    train = workspace/'runs/20260916_prefeedback_factorial_v1'
    training = checked(train)
    accepted = read(train/'acceptance_training.json')
    if accepted['status'] != 'PASS_TRAINING_AND_NATIVE_CACHE' or accepted['completed_models'] != 12:
        raise RuntimeError('All twelve accepted models required')
    legacy = Path(read(workspace/'registry/local_paths.json')['legacy_root'])
    previous = legacy/'experiments/worktrees/WorldEngine-selector-mid-followup-v1/experiments/diffusiondrive/selector_mid_followup_v1/runs/mid_formal_20260915_v1'
    old = read(previous/'inputs.json')
    models = {'scalar_v3': old['models']['scalar_v3']}
    for policy, evidence in accepted['models'].items():
        for key in ('training','native','checkpoint','selector'):
            verify(evidence[key])
        models[policy] = evidence['checkpoint']
    verify(models['scalar_v3'])
    artifacts = {k:old['artifacts'][k] for k in ('bridge_scenario','full_scenario','common_scenario')}
    for value in artifacts.values():verify(value)
    # The fixed scalar native bridge is a migration check against historical native execution.
    scalar_run = Path(models['scalar_v3']['path']).parents[2]
    for mode in ('NR','R'):
        artifacts['scalar_bridge_'+mode] = artifact(scalar_run/'collections'/f'bridge_{mode}_scalar_v3'/'audit.json')
    artifacts['training_acceptance'] = artifact(train/'acceptance_training.json')
    artifacts['historical_inputs'] = artifact(previous/'inputs.json')
    inputs = dict(method=METHOD, models=models, artifacts=artifacts,
                  cohorts={k:old['cohorts'][k] for k in ('bridge','full','common')},
                  open_tokens=old['open_tokens'], assets=old['assets'],
                  independent_blind_test=False, independent_pipeline_replicates=False)
    runtime = run/'runtime'
    inventory = {}
    for relative, digest in training['code']['files'].items():
        if relative.startswith('projects/'):
            source = CODE/relative
            if sha(source) != digest:raise RuntimeError('Training source changed')
    # Archive runtime dependencies, including shell/YAML/assets tracked in the clean snapshot.
    sources = [p for folder in ('projects','research') for p in (CODE/folder).rglob('*')
               if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts]
    sources.append(CODE/'kmeans_navsim_traj_20.npy')
    changed = {}
    for source in sorted(sources):
        relative = str(source.relative_to(CODE));raw = source.read_bytes();before = hashlib.sha256(raw).hexdigest()
        if relative in ('projects/AlgEngine/scripts/diffusiondrive/selector_snapshot_common.py',
                        'projects/AlgEngine/scripts/diffusiondrive/selector_snapshot_audit.py'):
            text = raw.decode()
            needle = "'selector_p1_balance_v1', 'selector_mid_followup_v1')"
            if text.count(needle) != 1:raise RuntimeError('Unexpected observer allowlist')
            raw = text.replace(needle,"'selector_p1_balance_v1', 'selector_mid_followup_v1', '"+METHOD+"')").encode()
            changed[relative] = dict(source_sha256=before, change='Add explicit factorial evaluation method to observer protocol allowlist only')
        target = runtime/relative;target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != raw:raise RuntimeError('Runtime archive collision: '+relative)
        else:target.write_bytes(raw)
        inventory[relative] = sha(target)
    config = runtime/'projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_selector_factorial_native.py'
    text = (CODE/'projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_selector_mid_followup_native.py').read_text()
    text = text.replace('selector_mid_followup_v1', METHOD).replace('Balance native evaluation', 'Factorial native evaluation')
    if config.exists() and config.read_text() != text:raise RuntimeError('Configuration collision')
    config.write_text(text);inventory[str(config.relative_to(runtime))] = sha(config)
    target = runtime/'data';source = legacy/'data'
    if target.is_symlink() or target.exists():
        if target.resolve() != source.resolve():raise RuntimeError('Foreign data mapping')
    else:target.symlink_to(source.resolve(), target_is_directory=True)
    code_sha = hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    driver_archive = run/'archive/drivers';driver_archive.mkdir(parents=True,exist_ok=True)
    tools = {}
    for name in ('prepare_study_evaluation.py','run_study_evaluation.py'):
        source = CODE/'scripts'/name;target = driver_archive/name
        if target.exists():
            if target.read_bytes()!=source.read_bytes():raise RuntimeError('Frozen driver changed')
        else:shutil.copy2(source,target)
        tools[name] = artifact(target)
    contract = dict(method=METHOD, code_sha=code_sha, implementation=inventory,
        training_manifest=artifact(train/'manifest.json'), training_acceptance=artifact(train/'acceptance_training.json'),
        driver=tools, runtime=str(runtime), source_worldengine_root=str(legacy),
        runtime_changes=changed, no_training=True, inference_uses_reward_or_q=False,
        full_condition_count=78, evaluation_seed=0, independent_blind_test=False,
        ray_cpu_threads=8, simulation_cpu_threads=8,
        required_before_full='Both scalar NR/R bridge migrations and native action audits for every learned model in both bridge modes',
        hardware_and_worker_count='Recorded per collection; bridge results must be checked, not assumed portable')
    write(run/'inputs.json',inputs);write(run/'run_contract.json',contract)
    print(json.dumps(dict(status='PREPARED_NOT_EVALUATED',run=str(run),runtime_files=len(inventory),changes=changed)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default=DEFAULT_ID)
    prepare(p.parse_args().run)
