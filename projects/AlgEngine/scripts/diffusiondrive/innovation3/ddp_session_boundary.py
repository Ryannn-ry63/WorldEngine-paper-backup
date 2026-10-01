"""Checksummed, atomic all-rank episode boundaries; no CUDA dependencies."""
import json
from pathlib import Path

from .paths import sha256_file
from .provisional_session import atomic_json, verify_episode


def assignment(spec, round_index, rank):
    return spec['rows'][(rank + round_index) % len(spec['rows'])]


def receipt(path, spec, round_index, rank, previous=None):
    path = Path(path)
    report = json.loads(path.read_text())
    row = assignment(spec, round_index, rank)
    version = previous['policy_version'] if previous else 0
    # Keep each formal seed in a disjoint deterministic RNG namespace while
    # preserving the legacy seed=0 contract used by engineering gates.
    seed_index = round_index * spec['world_size'] + rank
    seed_index += spec.get('seed', 0) * 1000003
    verified = verify_episode(report, row, round_index, version,
                              previous['learner_hash'] if previous else None, spec,
                              seed_index=seed_index)
    if (report.get('dynamics_contract') != spec['dynamics_contract'] or
            report.get('ego_controller') != spec['ego_controller']):
        raise ValueError('Execution contract changed')
    if previous and verified['initialization_provenance'] != previous['initialization_provenance']:
        raise ValueError('Initialization changed across episode boundary')
    events = [e for e in report['events'] if e['kind'] == 'online_update']
    if not all(e['update'].get('ddp_state_parity') is True for e in events):
        raise ValueError('Missing per-update DDP state parity')
    return dict(verified, rank=rank, global_episode=round_index * spec['world_size'] + rank,
                report=str(path), report_sha256=sha256_file(path),
                trace=[[e['version_before'], e['version_after'], e['optimized']] for e in events],
                learning_verified=report.get('real_closed_loop') is True,
                local_signal_groups=report.get('local_groups_with_signal', 0),
                elapsed_seconds=report['elapsed_seconds'])


def validate_round(records, spec, index):
    world = spec['world_size']
    if len(records) != world or [r['rank'] for r in records] != list(range(world)):
        raise ValueError('Incomplete or duplicate rank boundary')
    if any(r['global_episode'] != index * world + rank for rank, r in enumerate(records)):
        raise ValueError('Episode cursor mismatch')
    if len({r['scene'] for r in records}) != world:
        raise ValueError('Duplicate scenes in a DDP round')
    if any(r['trace'] != records[0]['trace'] for r in records):
        raise ValueError('Ranks have different optimizer schedules')


def open_boundary(output, spec, resume):
    output = Path(output)
    progress = output.with_suffix('.progress.json')
    if not resume:
        if output.exists() or progress.exists():
            raise FileExistsError('Existing run; use --resume or a fresh output')
        state = dict(schema=2, contract=spec, rounds=[], status='RUNNING',
                     completed=0, resumed_segments=0, formal_ready=False,
                     used_for_formal_training=bool(spec.get('formal_training')))
        atomic_json(progress, state)
        return state
    state = json.loads(progress.read_text())
    if state.get('schema') != 2 or state.get('contract') != spec:
        raise ValueError('Resume contract/source changed or legacy progress; use a fresh run')
    rounds = state['rounds']
    if (state['completed'] != len(rounds) * spec['world_size'] or
            state['completed'] > spec['episodes']):
        raise ValueError('Invalid completed episode count')
    for index, records in enumerate(rounds):
        validate_round(records, spec, index)
        for rank, record in enumerate(records):
            if sha256_file(record['report']) != record['report_sha256']:
                raise ValueError('Committed report checksum mismatch')
            checked = receipt(record['report'], spec, index, rank,
                              rounds[index-1][rank] if index else None)
            if checked != record:
                raise ValueError('Committed boundary record mismatch')
    return state


def commit_round(output, spec, state, records):
    validate_round(records, spec, len(state['rounds']))
    state['rounds'].append(records)
    state['completed'] = len(state['rounds']) * spec['world_size']
    state['status'] = 'RUNNING'
    atomic_json(Path(output).with_suffix('.progress.json'), state)


def finish(output, spec, state):
    if state['completed'] != spec['episodes']:
        raise ValueError('Cannot finish an incomplete session')
    learning = all(any(round_[rank]['learning_verified'] for round_ in state['rounds'])
                   for rank in range(spec['world_size']))
    prefix = 'PASS' if learning else 'INCOMPLETE'
    suffix = ('_DDP_FORMAL_ONLINE_TRAINING' if spec.get('formal_training')
              else '_DDP_REAL_ONLINE_SESSION_GATE')
    result = dict(status=prefix + suffix,
                  world_size=spec['world_size'], episodes_total=state['completed'],
                  episodes_per_rank=len(state['rounds']), steps=8,
                  policy_version=state['rounds'][-1][0]['policy_version'],
                  contract=spec, rounds=state['rounds'], ddp_verified=True,
                  real_closed_loop=learning, resumed_segments=state['resumed_segments'],
                  boundary_continuity_verified=True, full_resume_verified=False,
                  formal_ready=False, used_for_formal_training=bool(spec.get('formal_training')),
                  training_split=spec.get('split'))
    atomic_json(Path(output), result)
    state['status'] = ('COMPLETE_FORMAL_TRAINING' if spec.get('formal_training') else
                       'COMPLETE_PROVISIONAL') if learning else 'INCOMPLETE_LEARNING_EVIDENCE'
    atomic_json(Path(output).with_suffix('.progress.json'), state)
    return 0 if learning else 2
