"""Explicit, checksummed engineering and formal scene assignments.

Engineering manifests deliberately expose only a fixed pilot prefix.  Formal
manifests are a separate schema and must carry an audited origin-log-disjoint
split before the training runner is allowed to consume them.
"""
import json
import re
from pathlib import Path
from .paths import checked_path, sha256_file


ENGINEERING_PURPOSES = {
    'distinct_scene_engineering_pilot',
    'distinct_scene_engineering_pilot_v3_initialized_selector',
}
FORMAL_PURPOSE = 'online_formal_train_dev_holdout'
FORMAL_SPLITS = ('train', 'dev', 'holdout')


def origin_log(name):
    """Conservatively group timestamp/vehicle logs across numbered clip windows."""
    if not isinstance(name, str) or not re.fullmatch(
            r'\d{4}(?:\.\d{2}){5}_veh-\d+(?:_\d+_\d+)?', name):
        raise ValueError('Unsupported origin log: ' + repr(name))
    return re.sub(r'_\d+_\d+$', '', name)


def _validate_row(row, ids, logs):
    sid = row['scene_id']
    if not isinstance(sid, str) or Path(sid).name != sid or sid in ('.', '..'):
        raise ValueError('Invalid scene ID')
    log = origin_log(row['origin_log'])
    if sid in ids or log in logs:
        raise ValueError('Duplicate scene or origin log in DDP manifest')
    if origin_log(sid.rsplit('-', 1)[0]) != log:
        raise ValueError('Scene ID and origin log disagree')
    ids.add(sid); logs.add(log)
    for key in ('scene_seed', 'candidate_seed'):
        if type(row.get(key)) is not int or not 0 <= row[key] < 2**31:
            raise ValueError('Invalid seed namespace: ' + key)
    for key in ('asset_sha256', 'scene_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', row.get(key, '')):
            raise ValueError('Missing checksum: ' + key)


def load_assignments(manifest_path, settings_path, world, verify_files=True, split=None):
    manifest_path = checked_path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    purpose = manifest.get('purpose')
    formal = purpose == FORMAL_PURPOSE
    if formal:
        if manifest.get('schema') != 2 or manifest.get('formal_ready') is not True:
            raise ValueError('Formal manifest must be schema 2 and formal_ready=true')
        if manifest.get('used_for_formal_training') is not True:
            raise ValueError('Formal manifest must declare used_for_formal_training=true')
        if split not in FORMAL_SPLITS:
            raise ValueError('Formal manifest requires --split train, dev, or holdout')
        audit = manifest.get('audit', {})
        if audit.get('complete') is not True or not audit.get('version'):
            raise ValueError('Formal manifest lacks a complete audit record')
        if manifest.get('split_policy') != 'origin_log_80_10_10_v1':
            raise ValueError('Formal manifest split policy is not frozen')
    elif purpose not in ENGINEERING_PURPOSES or manifest.get('schema') != 1 \
            or manifest.get('formal_ready') is not False:
        raise ValueError('Expected an engineering manifest or an audited formal manifest')
    if manifest.get('reaction') != 'R' or manifest.get('steps') != 8:
        raise ValueError('Only the registered R / 8-decision engineering protocol is supported')
    if not re.fullmatch('[0-9a-f]{64}', manifest.get('scenario_source_sha256', '')):
        raise ValueError('Missing scenario source SHA256')
    if sha256_file(settings_path) != manifest.get('settings_sha256'):
        raise ValueError('Scene manifest settings SHA256 mismatch')
    cfg = json.loads(checked_path(settings_path).read_text())
    if purpose == 'distinct_scene_engineering_pilot_v3_initialized_selector' or formal:
        if cfg.get('online_parameterization') != 'v3_initialized_selector_finetune':
            raise ValueError('Initialized-selector manifest requires initialized-selector settings')
        if manifest.get('parameterization') != cfg.get('online_parameterization'):
            raise ValueError('Scene manifest parameterization mismatch')
    elif cfg.get('online_parameterization') not in (None, 'frozen_v3_plus_zero_residual'):
        raise ValueError('Legacy scene manifest cannot use initialized-selector settings')
    all_rows = manifest.get('scenes', [])
    rows = all_rows
    if not isinstance(all_rows, list):
        raise ValueError('Manifest scenes must be a list')
    if formal:
        exposed = set(manifest.get('audit', {}).get('exposed_origin_logs', []))
        if not exposed:
            raise ValueError('Formal manifest must record exposed origin logs')
        rows = []
        for row in all_rows:
            if row.get('split') not in FORMAL_SPLITS:
                raise ValueError('Formal scene is missing a valid split')
            if not row.get('asset_path'):
                raise ValueError('Formal scene is missing its audited asset_path')
            if row.get('origin_log') in exposed and row.get('split') != 'train':
                raise ValueError('Exposed origin log entered dev/holdout split')
            if row.get('split') == split:
                rows.append(row)
    if not isinstance(rows, list) or not 2 <= world <= len(rows):
        raise ValueError('Not enough explicitly assigned scenes for world size in split')
    # Validate the whole manifest, then select the requested formal split or
    # fixed engineering prefix.  The formal path rejects duplicate origin logs
    # across all splits, preventing leakage hidden outside the current split.
    ids, logs = set(), set()
    for row in all_rows:
        _validate_row(row, ids, logs)
    source = checked_path(Path(cfg['scenario_root'])/'original/navtrain_failures_per1/all_scenarios.pkl')
    if verify_files:
        if sha256_file(source) != manifest.get('scenario_source_sha256'):
            raise ValueError('Registered scenario source changed')
        for row in rows if formal else rows[:world]:
            asset = (checked_path(row['asset_path']) if formal else
                     checked_path(Path(cfg['asset_root'])/row['scene_id']/'background' /
                                  (row['scene_id']+'.ckpt')))
            if sha256_file(asset) != row['asset_sha256']:
                raise ValueError('Assigned exact asset checksum mismatch: ' + row['scene_id'])
            scene = checked_path(row['scene_path'])
            if sha256_file(scene) != row['scene_sha256']:
                raise ValueError('Assigned scene checksum mismatch: ' + row['scene_id'])
    return manifest, rows if formal else rows[:world]


def validate_scene_reports(reports, rows, manifest):
    """Fail closed on wrong worker, seed, source, asset or incomplete execution."""
    if len(reports) != len(rows):
        raise RuntimeError('Missing rank reports')
    for report, row in zip(reports, rows):
        worker = report.get('worker', {})
        if worker.get('scene') != row['scene_id']:
            raise RuntimeError('Worker used a different scene than its assignment')
        source_ok = (
            worker.get('source', {}).get('path') == row['scene_path'] and
            worker.get('source', {}).get('sha256') == row['scene_sha256'])
        # Older reports loaded the combined source. Keep them readable while
        # requiring the new direct-scene path to prove its own checksum.
        legacy_source_ok = worker.get('source', {}).get('sha256') == manifest['scenario_source_sha256']
        if (worker.get('asset', {}).get('sha256') != row['asset_sha256'] or
                not (source_ok or legacy_source_ok)):
            raise RuntimeError('Worker source/asset provenance mismatch')
        seeds = report.get('seed_namespaces', {})
        if any(seeds.get(k) != row[k] for k in ('candidate_seed', 'scene_seed')):
            raise RuntimeError('Worker seed namespace mismatch')
    return True
