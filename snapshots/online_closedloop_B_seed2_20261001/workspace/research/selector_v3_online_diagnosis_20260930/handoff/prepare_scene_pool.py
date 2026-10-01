#!/usr/bin/env python3
"""Freeze all statically valid clips inside the existing log-level split.

No metric, policy score, collision or offroad outcome participates in admission.
The original five holdout logs and all known protected overlaps are excluded.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from collections import Counter
from experiment_common import REGISTRY, SETTINGS, read, write, sha, source_hashes


def static_reasons(row):
    reasons = []
    for key in ('static_inspected', 'all_camera_literals_available', 'static_length_ok',
                'static_sample_rate_ok', 'literal_log_matches_filename'):
        if row.get(key) is not True:
            reasons.append(key)
    if row.get('known_protected_overlaps'):
        reasons.append('protected_overlap')
    if not any(not o['missing'] for o in row.get('asset_options', [])):
        reasons.append('missing_assets')
    return reasons


def build(output):
    origin = read(REGISTRY/'origin.json')
    prior = read(origin['split_manifest'])
    inventory = read(origin['inventory'])
    split_by_log = {r['origin_log']: ('development' if r['split'] in ('development','dev') else r['split'])
                    for r in prior['scenes']}
    previous = {r['scene_id']:r for r in prior['scenes']}
    candidates, excluded, discovery = [], [], []
    for row in inventory['scenes']:
        split = split_by_log.get(row['origin_log'])
        reasons = static_reasons(row)
        if split not in ('train','development'):
            if split is None and not reasons:
                discovery.append(dict(scene_id=row['scene_id'], origin_log=row['origin_log'],
                    prior_engineering_exposure=row.get('prior_engineering_exposure',False)))
            continue
        if reasons:
            excluded.append(dict(scene_id=row['scene_id'], split=split, reasons=reasons))
            continue
        candidates.append((row,split))
    def make(pair):
        row,split = pair
        sid = row['scene_id']
        old = previous.get(sid)
        option = sorted([o for o in row['asset_options'] if not o['missing']],
                        key=lambda o:(0 if old and old['asset_path'] in o['required_paths'] else 1,o['pool']))[0]
        # Existing accepted rows retain their exact source pool. For a newly
        # discovered clip, use the inventory's byte-certified source among the
        # duplicate public pools; never choose by an unchecked path ordering.
        if old:
            scene_path = Path(old['scene_path'])
            scene_hash = sha(scene_path)
            if scene_hash != old['scene_sha256']:
                raise ValueError('Registered scene bytes changed: '+sid)
        else:
            matching = [Path(path) for path in row['scene_paths'] if sha(path) == row['scene_sha256']]
            if not matching:
                raise ValueError('Inventory source is ambiguous or changed: '+sid)
            scene_path, scene_hash = sorted(matching)[0], row['scene_sha256']
        asset_path = Path(option['root'])/'background'/(sid+'.ckpt')
        assets = {str(Path(p)):sha(p) for p in option['required_paths']}
        if old and (scene_hash != old['scene_sha256'] or assets[str(asset_path)] != old['asset_sha256']):
            raise ValueError('Previously accepted source changed: '+sid)
        seed = int(hashlib.sha256(sid.encode()).hexdigest()[:8],16) % (2**31-1000)
        return dict(scene_id=sid, origin_log=row['origin_log'], split=split,
            scene_path=str(scene_path), scene_sha256=scene_hash,
            asset_path=str(asset_path), asset_sha256=assets[str(asset_path)],
            asset_files_sha256=assets, asset_config=option['config'],
            scene_seed=(old['scene_seed'] if old else seed),
            candidate_seed=(old['candidate_seed'] if old else seed+17),
            map=row['literal_fields']['map'][0],
            prior_engineering_exposure=row.get('prior_engineering_exposure',False))
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(pool.map(make,candidates),key=lambda r:(r['split'],r['scene_id']))
    counts = dict(Counter(r['split'] for r in rows))
    if counts != dict(train=150,development=83):
        raise ValueError('Static pool changed; inspect before registration: '+repr(counts))
    result = dict(schema=1, status='STATIC_CANDIDATES_NOT_ADMITTED',scenes=rows,counts=counts,
        source_manifest=origin['split_manifest'],source_manifest_sha256=sha(origin['split_manifest']),
        source_inventory=origin['inventory'],source_inventory_sha256=sha(origin['inventory']),
        settings_sha256=sha(SETTINGS), source_sha256=source_hashes(),
        protected_holdout_logs=sorted(k for k,v in split_by_log.items() if v=='holdout'),
        log_splits=split_by_log, excluded_static=excluded,
        development_is_independent_final_test=False,
        admission_policy='complete_render_and_frozen_full20_8steps_no_outcome_filter_v1')
    write(output,result,immutable=True)
    write(REGISTRY/'additional_log_discovery.json',dict(status='DISCOVERY_ONLY_NOT_IN_EXPERIMENT',
        source_inventory_sha256=sha(origin['inventory']),scenes=discovery),immutable=True)
    print('Static candidates:',counts, 'logs:',dict(Counter(split_by_log.values())),flush=True)
    return result


def finalize(candidates, receipt_dir, output):
    manifest = read(candidates)
    if manifest['settings_sha256'] != sha(SETTINGS) or manifest['source_sha256'] != source_hashes():
        raise ValueError('Runtime changed after candidate registration')
    admitted, rejected, receipts = [], [], {}
    for row in manifest['scenes']:
        path = Path(receipt_dir)/(row['scene_id']+'.json')
        receipt = read(path)
        if receipt.get('candidate_manifest_sha256') != sha(candidates) or receipt.get('scene') != row:
            raise ValueError('Wrong admission receipt: '+row['scene_id'])
        if receipt.get('source_sha256') != manifest['source_sha256']:
            raise ValueError('Mixed admission source versions')
        receipts[row['scene_id']] = dict(path=str(path.resolve()),sha256=sha(path))
        if receipt['status']=='PASS_TECHNICAL_ADMISSION':
            report = read(receipt['report'])
            if sha(receipt['report']) != receipt['report_sha256'] or report.get('status') != 'PASS_FROZEN_PROBE':
                raise ValueError('Admission report failed or changed')
            if report['groups'] != 8 or report['candidate_simulation_steps'] != 1280:
                raise ValueError('Incomplete actual full20 admission')
            admitted.append(row)
        elif receipt['status']=='FAIL_TECHNICAL_INPUT' and receipt.get('failure_category') in (
                'input_file_missing','input_pickle_corrupt','insufficient_window'):
            rejected.append(dict(scene=row,receipt=receipts[row['scene_id']],reason=receipt['failure_category']))
        else:
            raise ValueError('Runtime failure is not a scene exclusion; fix/retry '+str(path))
    counts = dict(Counter(r['split'] for r in admitted))
    if counts.get('train',0)<8 or counts.get('development',0)<12:
        raise ValueError('Insufficient admitted pool')
    n=counts['train']
    final=dict(schema=4,purpose='fixed_coverage_diagnosis_v3',formal_ready=False,
        used_for_formal_training=False,reaction='R',steps=8,
        parameterization='v3_initialized_selector_finetune',scenes=admitted,counts=counts,
        settings_sha256=sha(SETTINGS),source_manifest=manifest['source_manifest'],
        source_manifest_sha256=manifest['source_manifest_sha256'],
        candidate_manifest=str(Path(candidates).resolve()),candidate_manifest_sha256=sha(candidates),
        log_splits=manifest['log_splits'],protected_holdout_logs=manifest['protected_holdout_logs'],
        technical_admission_complete=True,admission_receipts=receipts,excluded_technical=rejected,
        development_is_independent_final_test=False,
        episode_budgets={str(k):8*((k*n+7)//8) for k in (3,6)},
        source_sha256=source_hashes())
    write(output,final,immutable=True)
    print('Frozen admitted manifest:',output,counts,final['episode_budgets'],flush=True)
    return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('candidates','finalize'))
    p.add_argument('--candidates',type=Path,default=REGISTRY/'scene_candidates.json')
    p.add_argument('--receipts',type=Path,default=REGISTRY/'admission')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.stage=='candidates': build(a.output or a.candidates)
    else: finalize(a.candidates,a.receipts,a.output or REGISTRY/'scene_manifest.json')
if __name__=='__main__': main()
