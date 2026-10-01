"""Explicit, checksummed migration of a frozen p6 boundary into a new p8 ledger.

A legacy checkpoint is accepted only on the first imported round. Ordinary
resume still requires an identical source/settings/manifest contract.
"""
import copy
import json
from pathlib import Path
from .paths import sha256_file


def read_verified(path, digest):
    if sha256_file(path) != digest:
        raise ValueError('Continuation provenance checksum mismatch: ' + str(path))
    return json.loads(Path(path).read_text())


def load_import(reference, current_sources=None):
    m = read_verified(reference['path'], reference['sha256'])
    if m.get('schema') != 1 or m.get('purpose') != 'B_p6_to_p8_explicit_continuation':
        raise ValueError('Unknown continuation manifest')
    if current_sources is not None and m['new_source_sha256'] != current_sources:
        raise ValueError('Continuation target source changed')
    parent = read_verified(m['parent_progress']['path'], m['parent_progress']['sha256'])
    if parent['completed'] != 904 or len(parent['rounds']) != 113:
        raise ValueError('Expected exactly the committed p6 boundary')
    old = parent['contract']
    if old['world_size'] != 8 or old['episodes'] != 904 or old['online_update_timing'] != 'after_rollout8':
        raise ValueError('Only the frozen B p6 all-rank boundary may be imported')
    if len(m['checkpoints']) != 8:
        raise ValueError('Missing rank checkpoint')
    for rank, checkpoint in enumerate(m['checkpoints']):
        if checkpoint['sha256'] != parent['rounds'][-1][rank]['checkpoint']['sha256']:
            raise ValueError('Imported checkpoint is not the final committed rank')
    return m, parent


def import_boundary(output, spec, reference):
    from .ddp_session_boundary import receipt, validate_round
    from .provisional_session import atomic_json
    output = Path(output)
    if output.exists() or output.with_suffix('.progress.json').exists():
        raise FileExistsError('Import requires a fresh continuation ledger')
    m, parent = load_import(reference, spec['source_sha256'])
    old = parent['contract']
    if spec['episodes'] != 1200 or len(spec['rows']) != 150:
        raise ValueError('Expected eight complete visits of the 150-scene pool')
    allowed = {'episodes', 'source_sha256', 'settings_sha256', 'manifest_sha256', 'wall_budget_seconds'}
    if {k:v for k,v in spec.items() if k not in allowed} != {k:v for k,v in old.items() if k not in allowed}:
        raise ValueError('Algorithm/data/schedule changed during continuation')
    for key in ('settings_sha256', 'manifest_sha256'):
        if spec[key] != m['new_' + key]:
            raise ValueError('Continuation settings or scene manifest changed')
    # The parent ledger hash authenticates its historical prefix; verify the
    # final receipt and independent rank copies before accepting that prefix.
    for index, rows in enumerate(parent['rounds']):
        validate_round(rows, old, index)
    for rank, row in enumerate(parent['rounds'][-1]):
        if sha256_file(row['report']) != row['report_sha256']:
            raise ValueError('Parent final report changed')
        if receipt(row['report'], old, 112, rank, parent['rounds'][-2][rank]) != row:
            raise ValueError('Parent final receipt invalid')
        cp = m['checkpoints'][rank]
        if sha256_file(cp['path']) != cp['sha256']:
            raise ValueError('Independent rank copy changed')
    state = copy.deepcopy(parent)
    state.update(contract=spec, status='IMPORTED_P6_NOT_YET_CONTINUED', resumed_segments=0,
                 parent_training_wall_seconds=parent.get('training_wall_seconds', 0),
                 training_wall_seconds=0., lineage=dict(reference=reference, inherited_rounds=113))
    atomic_json(output.with_suffix('.progress.json'), state)
    return state


def inherited_contract(state, spec):
    """Verify the prefix on every resume; never re-label old reports as new code."""
    line = state.get('lineage')
    if line is None:
        return 0
    _, parent = load_import(line['reference'], spec['source_sha256'])
    if line['inherited_rounds'] != 113 or state['rounds'][:113] != parent['rounds']:
        raise ValueError('Continuation historical prefix changed')
    return 113


def validate_legacy_checkpoint(payload, checkpoint_path, reference, current_sources):
    m, parent = load_import(reference, current_sources)
    matches = [c for c in m['checkpoints'] if Path(c['path']).resolve() == Path(checkpoint_path).resolve()]
    if len(matches) != 1 or sha256_file(checkpoint_path) != matches[0]['sha256']:
        raise ValueError('Checkpoint is not an approved independent p6 copy')
    if payload['source_sha256'] != parent['contract']['source_sha256']:
        raise ValueError('Legacy checkpoint source differs from its registered parent')
    if payload['learner']['attempts'] != 904 or payload['learner']['version'] != 904:
        raise ValueError('Legacy import must preserve the p6 optimizer cursor')
