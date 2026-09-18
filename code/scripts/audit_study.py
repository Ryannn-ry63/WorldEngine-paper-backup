"""Verify all twelve formal models and native exports; never certify driving scores."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from selector_study.common import CODE, artifact, output_run, read, verify, write
from selector_study.data import checked


def audit(run):
    manifest = checked(run)
    expected_manifest = artifact(run/'manifest.json')
    validation = read(run/'validation.json')
    if (validation['status'] != 'PASS' or validation['manifest'] != expected_manifest
            or not validation['resume_model_and_optimizer_bitwise_equal']):
        raise RuntimeError('Validation/resume evidence missing')
    for entry in [validation['cpu'], validation['gpu'], *validation['smoke_reports'].values()]:
        verify(entry)
    models = {}
    for seed in (0,1,2):
        for arm in ('D1','D2','D3','D4'):
            name = f'{arm}_seed{seed}'
            train_path = run/'checkpoints'/name/'report.json'
            native_path = run/'native'/name/'report.json'
            t, n = read(train_path), read(native_path)
            p = t['provenance']
            if (t['status'] != 'PASS' or t['stage'] != 'TRAINED_NOT_EVALUATED'
                    or t['steps'] != 1000 or p['steps'] != 1000 or p['smoke']
                    or p['arm'] != arm or p['seed'] != seed
                    or p['manifest'] != expected_manifest or p['parent'] != manifest['parent']
                    or p['initial_optimizer_state'] != 'empty' or t['trainable_tensors'] != 54
                    or t['evaluation_complete'] or t['export_max_abs'] > 1e-5):
                raise RuntimeError('Invalid formal training report: '+name)
            if [v['step'] for v in t['history']] != list(range(25,1001,25)):
                raise RuntimeError('Missing training progress: '+name)
            for row in t['history']:
                if any(not math.isfinite(float(v)) for v in row.values()):
                    raise RuntimeError('Nonfinite training history: '+name)
                if arm in ('D1','D2') and (row['pair'] != 0 or row['winner_nll'] != 0):
                    raise RuntimeError('Preference labels in no-preference objective: '+name)
            if (n['status'] != 'PASS' or n['policy'] != name
                    or n['provenance']['training'] != artifact(train_path)
                    or n['provenance']['manifest'] != expected_manifest
                    or n['provenance']['selector'] != t['selector']
                    or n['unchanged_nonselector_tensors'] != 963 or n['selector_tensors'] != 54
                    or n['cached_states'] != 256 or n['argmax_mismatches'] != 0
                    or n['cached_max_abs'] > 1e-5 or n['evaluation_complete']
                    or n['physical_simulator_bridge_completed']):
                raise RuntimeError('Invalid native checkpoint report: '+name)
            for entry in [t['selector'], t['resume'], n['checkpoint'], n['provenance']['exporter']]:
                verify(entry)
            models[name] = dict(training=artifact(train_path), native=artifact(native_path),
                selector=t['selector'], checkpoint=n['checkpoint'], seconds=t['seconds'])
    result = dict(status='PASS_TRAINING_AND_NATIVE_CACHE', manifest=expected_manifest,
        acceptance_implementation=artifact(__file__), validation=artifact(run/'validation.json'),
        models=models, completed_models=len(models), total_optimizer_steps=12000,
        physical_simulator_bridge_completed=False, evaluation_complete=False,
        independent_pipeline_replicates=False,
        next_required='Physical simulator bridge, then separately audited closed/open evaluation; no driving-score conclusion yet.')
    write(run/'acceptance_training.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='20260916_prefeedback_factorial_v1')
    args = parser.parse_args()
    print(json.dumps(audit(output_run(CODE.parent,args.run)), ensure_ascii=False, indent=2))
