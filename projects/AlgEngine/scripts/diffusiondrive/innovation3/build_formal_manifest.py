"""Build the audited Innovation 3 train/dev/holdout manifest.

The input is intentionally an explicit audit receipt.  This command refuses
the preliminary owned-scene inventory and refuses to infer eligibility from
file existence, so engineering pilots cannot accidentally become paper data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .paths import checked_path, sha256_file
from .scene_manifest import FORMAL_PURPOSE, origin_log


def _digest(path: Path) -> str:
    return sha256_file(checked_path(path))


def _split_logs(logs, seed, forced_train=()):
    """Return a deterministic 80/10/10 origin-log split."""
    keyed = []
    for log in sorted(logs):
        key = hashlib.sha256((str(seed) + ':' + log).encode()).hexdigest()
        keyed.append((key, log))
    forced_train = set(forced_train)
    ordered = sorted(forced_train) + [log for _, log in sorted(keyed)
                                      if log not in forced_train]
    n = len(ordered)
    train_n = max(int(math.floor(n * 0.8)), len(forced_train))
    dev_n = int(math.floor(n * 0.1))
    # A formal split must have all three partitions.  The caller enforces the
    # practical train-size requirement for eight-rank DDP.
    if train_n < 1 or dev_n < 1 or n - train_n - dev_n < 1:
        raise ValueError('Need at least ten audited origin logs for 80/10/10 split')
    return (set(ordered[:train_n]), set(ordered[train_n:train_n + dev_n]),
            set(ordered[train_n + dev_n:]))


def build(audit_path: Path, settings_path: Path, output: Path, seed: int,
          min_train_scenes: int = 8):
    audit_path = checked_path(audit_path)
    settings_path = checked_path(settings_path)
    audit = json.loads(audit_path.read_text())
    if audit.get('status') != 'PASS_FORMAL_SCENE_AUDIT' or audit.get('formal_ready') is not True:
        raise ValueError('Input must be a completed PASS_FORMAL_SCENE_AUDIT receipt')
    if audit.get('qualification_scope') != 'complete_c_d_certification_navtest_public':
        raise ValueError('Input PASS_FORMAL_SCENE_AUDIT receipt lacks complete C/D/certification/navtest/public scope')
    version = audit.get('audit_version')
    exposed = audit.get('exposed_origin_logs')
    if not isinstance(version, str) or not version or not isinstance(exposed, list) or not exposed:
        raise ValueError('Audit receipt lacks versioned exposed-origin-log record')
    candidates = audit.get('scenes')
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('Audit receipt has no scene candidates')

    settings = json.loads(settings_path.read_text())
    if settings.get('online_parameterization') != 'v3_initialized_selector_finetune':
        raise ValueError('Formal online manifest requires initialized-selector settings')
    source = checked_path(Path(settings['scenario_root']) /
                          'original/navtrain_failures_per1/all_scenarios.pkl')
    source_sha = _digest(source)

    rows = []
    logs = set()
    seen = set()
    exposed = set(exposed)
    for item in candidates:
        if item.get('eligible') is not True:
            raise ValueError('Audit candidates must be explicitly eligible=True')
        sid = item.get('scene_id')
        log = origin_log(item.get('origin_log', sid.rsplit('-', 1)[0] if isinstance(sid, str) else None))
        if sid in seen or log in logs:
            raise ValueError('Formal manifest requires one scene per origin log')
        scene_path = item.get('scene_path') or item.get('owned_pickle')
        asset_path = (item.get('asset_path') or item.get('owned_background') or
                      item.get('background_checkpoint'))
        if not scene_path or not asset_path:
            raise ValueError('Audit scene lacks scene and background paths: ' + repr(sid))
        row = dict(scene_id=sid, origin_log=log, scene_path=str(checked_path(scene_path)),
                   asset_path=str(checked_path(asset_path)),
                   scene_sha256=_digest(Path(scene_path)), asset_sha256=_digest(Path(asset_path)),
                   audit_record_id=item.get('audit_record_id', sid),
                   prior_engineering_exposure=bool(item.get('prior_engineering_exposure', False)))
        rows.append(row); seen.add(sid); logs.add(log)

    forced_train = {origin_log(item['origin_log']) for item in candidates
                    if item.get('prior_engineering_exposure') is True}
    train_logs, dev_logs, holdout_logs = _split_logs(logs, seed, forced_train)
    split_by_log = {**{x: 'train' for x in train_logs},
                    **{x: 'dev' for x in dev_logs},
                    **{x: 'holdout' for x in holdout_logs}}
    if sum(split_by_log[r['origin_log']].__eq__('train') for r in rows) < min_train_scenes:
        raise ValueError('Audited train split is smaller than min_train_scenes')
    for index, row in enumerate(sorted(rows, key=lambda x: (split_by_log[x['origin_log']], x['origin_log']))):
        row['split'] = split_by_log[row['origin_log']]
        row['scene_seed'] = 10000 + index
        row['candidate_seed'] = 20000 + index
    rows.sort(key=lambda x: (('train', 'dev', 'holdout').index(x['split']), x['origin_log']))
    manifest = dict(
        schema=2, purpose=FORMAL_PURPOSE, formal_ready=True,
        used_for_formal_training=True, reaction='R', steps=8,
        parameterization='v3_initialized_selector_finetune',
        settings_sha256=_digest(settings_path), scenario_source_sha256=source_sha,
        split_policy='origin_log_80_10_10_v1', split_seed=seed,
        audit=dict(complete=True, version=version, receipt=str(audit_path.resolve()),
                   receipt_sha256=_digest(audit_path), exposed_origin_logs=sorted(exposed)),
        scenes=rows,
        counts={name: sum(r['split'] == name for r in rows)
                for name in ('train', 'dev', 'holdout')},
        seed_contract=dict(training_seed='CLI --seed in {0,1,2}',
                           scene_seed='per-row plus global episode namespace',
                           candidate_seed='per-row plus global episode namespace',
                           action_seed='candidate_seed plus 17'),
    )
    output = checked_path(output.absolute(), must_exist=False)
    if output.exists():
        raise FileExistsError('Use a fresh formal manifest path: ' + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-seed', type=int, default=20260926)
    parser.add_argument('--min-train-scenes', type=int, default=8)
    args = parser.parse_args()
    manifest = build(args.audit, args.settings, args.output, args.split_seed,
                     args.min_train_scenes)
    print(json.dumps(dict(status='PASS_FORMAL_MANIFEST_BUILT', output=str(args.output),
                          counts=manifest['counts'], split_seed=args.split_seed), sort_keys=True))


if __name__ == '__main__':
    main()
