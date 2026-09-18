"""Export a completed factorial selector and verify the native checkpoint on CPU.

This checks checkpoint assembly and cached inference, not simulator deployment.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from selector_study.common import CODE, artifact, legacy_imports, output_run, read, verify, write
from selector_study.data import checked, Dataset

PREFIX = 'planning_head.scene_selector.'


def export_one(run, arm, seed, manifest):
    legacy_imports()
    import selector_cfpi_model as m
    from materialize_grpo_selector_v3 import state_dict

    name = f'{arm}_seed{seed}'
    report_path = run/'checkpoints'/name/'report.json'
    trained = read(report_path)
    p = trained['provenance']
    if (trained['status'] != 'PASS' or trained['stage'] != 'TRAINED_NOT_EVALUATED'
            or trained['steps'] != manifest['training']['steps']
            or p['arm'] != arm or p['seed'] != seed or p['smoke']
            or p['parent'] != manifest['parent'] or p['manifest'] != artifact(run/'manifest.json')):
        raise RuntimeError('Expected completed, contract-matched formal training: '+name)
    selector = verify(trained['selector'])
    incumbent = read(verify(manifest['incumbent_manifest']))
    base = dict(path=incumbent['checkpoint'], sha256=incumbent['checkpoint_sha256'])
    verify(base)
    provenance = dict(manifest=artifact(run/'manifest.json'), training=artifact(report_path),
                      selector=trained['selector'], base=base, exporter=artifact(__file__))
    folder = run/'native'/name
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder/'report.json').exists():
            previous = read(folder/'report.json')
            if previous['provenance'] != provenance or previous['status'] != 'PASS':
                raise RuntimeError('Native export contract differs')
            verify(previous['checkpoint'])
            return previous
        write(folder/'contract.json', provenance)
        model, payload = m.load_selector(selector)
        if (payload['provenance'] != p or payload['score_mode'] != 'residual'
                or payload.get('inference_uses_reward_or_q') is not False
                or payload['scene_selector_config'] != incumbent['selector_payload']['scene_selector_config']):
            raise RuntimeError('Selector inference configuration changed')
        checkpoint = torch.load(verify(base), map_location='cpu')
        original = state_dict(checkpoint)
        weights = payload['scene_selector_state']
        selector_keys = {k for k in original if k.startswith(PREFIX)}
        frozen_keys = set(original) - selector_keys
        if (selector_keys != {PREFIX+k for k in weights}
                or len(selector_keys) != 54 or len(frozen_keys) != 963):
            raise RuntimeError('Unexpected native tensor inventory')
        assembled = copy.deepcopy(checkpoint)
        target = state_dict(assembled)
        for key, value in weights.items():
            if value.shape != original[PREFIX+key].shape or not torch.isfinite(value).all():
                raise RuntimeError('Invalid selector tensor: '+key)
            target[PREFIX+key] = value.detach().cpu().clone()
        assembled.setdefault('meta', {}).pop('diffusiondrive_grpo_selector_v3', None)
        assembled['meta']['selector_study'] = dict(method=manifest['method'], policy=name,
            provenance=provenance, scene_selector_config=payload['scene_selector_config'],
            inference_uses_reward_or_q=False, score_mode='residual')
        output = folder/'checkpoint.pth'
        if output.exists():
            saved = torch.load(output, map_location='cpu')
            if saved.get('meta') != assembled.get('meta'):
                raise RuntimeError('Existing native metadata differs')
        else:
            temporary = output.with_suffix('.tmp')
            torch.save(assembled, temporary)
            os.replace(temporary, output)
            saved = torch.load(output, map_location='cpu')
        saved_state = state_dict(saved)
        if set(saved_state) != set(original):
            raise RuntimeError('Serialized native keys differ')
        if any(not torch.equal(original[k], saved_state[k]) for k in frozen_keys):
            raise RuntimeError('Frozen generator/perception/base tensors changed')
        if any(not torch.equal(v, saved_state[PREFIX+k]) for k, v in weights.items()):
            raise RuntimeError('Serialized selector differs from trained selector')
        restored = m.v3.model_from_config(payload['scene_selector_config'])
        restored.load_state_dict({k[len(PREFIX):]:v for k,v in saved_state.items() if k.startswith(PREFIX)}, strict=True)
        model.eval(); restored.eval()
        rows = Dataset(manifest).feedback
        error = 0.; mismatches = 0
        with torch.no_grad():
            for start in range(0, len(rows), 16):
                batch = rows[start:start+16]; indices = list(range(len(batch)))
                left = m.score(model, batch, indices, 'cpu', 'residual')
                right = m.score(restored, batch, indices, 'cpu', 'residual')
                if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                    raise RuntimeError('Nonfinite native inference')
                error = max(error, float((left-right).abs().max()))
                mismatches += int((left.argmax(-1) != right.argmax(-1)).sum())
        if error > 1e-5 or mismatches:
            raise RuntimeError('Native cached inference parity failed')
        result = dict(status='PASS', policy=name, provenance=provenance, checkpoint=artifact(output),
            unchanged_nonselector_tensors=len(frozen_keys), selector_tensors=len(weights),
            cached_states=len(rows), cached_max_abs=error, argmax_mismatches=mismatches,
            physical_simulator_bridge_completed=False, evaluation_complete=False)
        write(folder/'report.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='20260916_prefeedback_factorial_v1')
    parser.add_argument('--arm', choices=['D1','D2','D3','D4'])
    parser.add_argument('--seed', type=int, choices=[0,1,2], default=0)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.all == bool(args.arm):
        parser.error('Choose either --all or --arm with optional --seed')
    run = output_run(CODE.parent, args.run)
    manifest = checked(run)
    jobs = [(a,s) for s in (0,1,2) for a in ('D1','D2','D3','D4')] if args.all else [(args.arm,args.seed)]
    missing = [f'{a}_seed{s}' for a,s in jobs if not (run/'checkpoints'/f'{a}_seed{s}'/'report.json').exists()]
    if missing:
        raise RuntimeError('Training must finish before export: '+','.join(missing))
    torch.set_num_threads(4)
    for arm, seed in jobs:
        print(json.dumps(export_one(run,arm,seed,manifest), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
