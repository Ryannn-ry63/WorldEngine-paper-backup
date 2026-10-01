"""Development split derived only from the registered 33 training logs."""
import copy
import hashlib
import json
from pathlib import Path
from .paths import sha256_file
from .scene_manifest import _validate_row

PURPOSE = 'fixed_segment_development_25_8_v1'
SPLIT_SEED = 20260929


def build(source_path, settings_path):
    source_path, settings_path = Path(source_path).resolve(), Path(settings_path).resolve()
    source = json.loads(source_path.read_text())
    if source.get('purpose') != 'online_formal_train_dev_holdout' or source.get('counts') != dict(train=33, dev=4, holdout=5):
        raise ValueError('Expected the audited 33/4/5 source manifest')
    if source.get('audit', {}).get('complete') is not True:
        raise ValueError('Source audit is incomplete')
    rows = copy.deepcopy(source['scenes'])
    train = [r for r in rows if r['split']=='train']
    ordered = sorted(train, key=lambda r: hashlib.sha256(
        ('%d:%s'%(SPLIT_SEED, r['origin_log'])).encode()).hexdigest())
    held = {r['origin_log'] for r in ordered[:8]}
    for row in rows:
        if row['split']=='train' and row['origin_log'] in held:
            row['split']='development'
    return dict(schema=3, purpose=PURPOSE, formal_ready=False, used_for_formal_training=False,
        source_manifest=str(source_path), source_manifest_sha256=sha256_file(source_path),
        settings_sha256=sha256_file(settings_path), split_seed=SPLIT_SEED,
        counts=dict(train=25, development=8, dev=4, holdout=5), reaction='R', steps=8,
        parameterization='v3_initialized_selector_finetune', scenes=rows,
        development_is_independent_final_test=False,
        note='Development logs come from historical training; original dev/holdout unchanged.')


def load_assignments(manifest_path, settings_path, world, verify_files=True, split='train'):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('schema') == 4:
        from .coverage_manifest import load_assignments as load_coverage
        return load_coverage(manifest_path, settings_path, world, verify_files, split)
    if manifest.get('purpose') != PURPOSE or manifest.get('schema') != 3:
        raise ValueError('Expected the fixed segment development manifest')
    source = Path(manifest['source_manifest'])
    if sha256_file(source) != manifest['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    expected = build(source, settings_path)
    if manifest != expected:
        raise ValueError('Development split/settings differ from the deterministic contract')
    if split not in ('train', 'development', 'dev', 'holdout'):
        raise ValueError('Unknown development split')
    ids, logs = set(), set()
    for row in manifest['scenes']:
        _validate_row(row, ids, logs)
    rows = [r for r in manifest['scenes'] if r['split']==split]
    if not 1 <= world <= len(rows):
        raise ValueError('Not enough distinct logs for requested ranks')
    if verify_files:
        for row in rows:
            for kind in ('scene', 'asset'):
                if sha256_file(row[kind+'_path']) != row[kind+'_sha256']:
                    raise ValueError('Changed '+kind+': '+row['scene_id'])
    return manifest, rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.settings)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
