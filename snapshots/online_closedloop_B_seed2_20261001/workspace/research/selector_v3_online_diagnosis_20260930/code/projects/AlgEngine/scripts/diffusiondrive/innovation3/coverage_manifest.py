"""Multiple clips per log, immutable technical admission, no cross-split leakage."""
import json
from pathlib import Path
from collections import Counter
from .paths import sha256_file
from .scene_manifest import _validate_row


def load_assignments(manifest_path,settings_path,world,verify_files=True,split='train'):
    m=json.loads(Path(manifest_path).read_text())
    if m.get('schema')!=4 or m.get('purpose')!='fixed_coverage_diagnosis_v3' or m.get('technical_admission_complete') is not True:
        raise ValueError('Coverage manifest requires completed technical admission')
    if m['settings_sha256']!=sha256_file(settings_path):
        raise ValueError('Manifest settings changed')
    for key in ('source_manifest','candidate_manifest'):
        if sha256_file(m[key])!=m[key+'_sha256']:
            raise ValueError('Frozen input changed: '+key)
    if split not in ('train','development'):
        raise ValueError('Only registered train/development splits are consumable')
    ids, logs=set(),{}
    for row in m['scenes']:
        _validate_row(row,ids,set())  # duplicate clips forbidden; repeated logs allowed within one split
        log=row['origin_log']; group=row['split']
        if (group not in ('train','development') or log in m['protected_holdout_logs'] or
                group != m['log_splits'].get(log) or (log in logs and logs[log] != group)):
            raise ValueError('Origin log crosses split/protected boundary')
        logs[log]=group
        receipt=m['admission_receipts'][row['scene_id']]
        if sha256_file(receipt['path'])!=receipt['sha256']:
            raise ValueError('Technical admission receipt changed')
    counts=dict(Counter(r['split'] for r in m['scenes']))
    if counts!=m['counts']:
        raise ValueError('Coverage count mismatch')
    rows=[r for r in m['scenes'] if r['split']==split]
    if not 1<=world<=len(rows):
        raise ValueError('Insufficient distinct scenes')
    if verify_files:
        for row in rows:
            files={row['scene_path']:row['scene_sha256'],**row['asset_files_sha256']}
            for path,expected in files.items():
                if sha256_file(path)!=expected:
                    raise ValueError('Scene/asset content changed: '+str(path))
    return m,rows
