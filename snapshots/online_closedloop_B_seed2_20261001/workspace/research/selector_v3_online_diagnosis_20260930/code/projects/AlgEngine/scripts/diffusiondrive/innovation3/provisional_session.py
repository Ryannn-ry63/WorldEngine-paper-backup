"""Bounded exploratory multi-scene learning, with immutable episode attempts.

No formal split or evaluation claim. Existing probe/session contracts remain
unchanged. Resume trusts only atomically committed, checksum-verified boundaries.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

from .paths import checked_path, sha256_file
from .scene_manifest import load_assignments
from .online_session_probe import _episode_settings
from .session_lifecycle import SessionLedger


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with tmp.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if tmp.exists(): tmp.unlink()


def episode_row(rows, episode):
    # Initial A -> B -> A is an explicit reset/reload smoke; then visit all rows.
    index = (0, 1, 0)[episode] if episode < 3 else (episode - 1) % len(rows)
    return rows[index]


def contract(settings, manifest, rows, episodes, seed, workers, sources):
    return dict(schema=1, purpose='PROVISIONAL_MULTI_SCENE_NOT_FORMAL',
                settings_sha256=sha256_file(settings), manifest_sha256=sha256_file(manifest),
                episodes=episodes, steps=8, seed=seed, branch_workers=workers,
                parameterization='v3_initialized_selector_finetune',
                reward_contract='causal_h1_pdm_components_v1', source_sha256=sources,
                schedule=[episode_row(rows, i) for i in range(episodes)],
                formal_ready=False, used_for_formal_training=False)


def verify_episode(report, row, index, version, previous_hash, spec, seed_index=None):
    if report.get('status') not in ('PASS_STRICT_ONLINE_THROUGHPUT_PROBE',
                                   'INCOMPLETE_ONLINE_THROUGHPUT_EVIDENCE'):
        raise ValueError('Episode execution failed')
    if report.get('source_sha256') != spec['source_sha256']:
        raise ValueError('Episode source mismatch')
    if report.get('parameterization') != spec['parameterization']:
        raise ValueError('Episode parameterization mismatch')
    if report.get('causal_reward_contract') != spec['reward_contract']:
        raise ValueError('Reward contract mismatch')
    if report.get('branch_workers') != spec['branch_workers']:
        raise ValueError('Branch worker count mismatch')
    worker = report.get('worker', {})
    if worker.get('scene') != row['scene_id']:
        raise ValueError('Wrong assigned scene')
    if (worker.get('source', {}).get('sha256') != row['scene_sha256'] or
            worker.get('asset', {}).get('sha256') != row['asset_sha256']):
        raise ValueError('Scene/asset provenance mismatch')
    namespace = index if seed_index is None else seed_index
    seeds = dict(training_seed=spec['seed'],
                 candidate_seed=(row['candidate_seed'] + namespace*1000003) % 2**31,
                 scene_seed=(row['scene_seed'] + namespace*1000003) % 2**31,
                 action_seed=(row['candidate_seed'] + namespace*1000003 + 17) % 2**31)
    if report.get('seed_namespaces') != seeds:
        raise ValueError('Episode RNG namespace mismatch')
    for key in ('frozen_generator_and_original_selector_unchanged', 'frozen_reference_unchanged',
                'online_selector_trainable'):
        if report.get(key) is not True: raise ValueError('Missing evidence: '+key)
    timing = spec.get('online_update_timing', 'per_step')
    if report.get('online_update_timing', 'per_step') != timing:
        raise ValueError('Update timing mismatch')
    if timing == 'per_step' and report.get('next_step_waited_for_update') is not True:
        raise ValueError('Step update evidence missing')
    if timing == 'after_rollout8' and (report.get('next_step_waited_for_feedback') is not True or
            report.get('updates_at_rollout_boundary') is not True):
        raise ValueError('Rollout update evidence missing')
    from .fixed_protocol import SUPPORTED_CONTRACTS
    fixed = spec.get('reward_contract') in SUPPORTED_CONTRACTS
    worker_ok = (report.get('worker_episode_closed') is True and
                 report.get('worker_process_resident') is True and
                 report.get('candidate_simulation_steps') == 1280) if fixed else (
                 report.get('worker_exit', {}).get('alive') is False and
                 report.get('worker_exit', {}).get('returncode') == 0)
    if (report.get('candidate_branches') != 160 or report.get('candidate_groups') != 8 or
            report.get('idm_fallbacks') != 0 or not worker_ok):
        raise ValueError('Branch completeness or cleanup failure')
    if (report.get('policy_version_initial') != version or report.get('update_attempts_initial') != index*8
            or report.get('update_attempts') != 8 or report.get('cumulative_update_attempts') != (index+1)*8):
        raise ValueError('Episode learning counters mismatch')
    if previous_hash is not None and report.get('learner_state_hash_initial') != previous_hash:
        raise ValueError('Learner/optimizer/RNG continuity mismatch')
    ledger = SessionLedger('provisional', version, index*8)
    token = ledger.start_episode(row['scene_id'], seeds['scene_seed'])
    updates = [e for e in report['events'] if e['kind']=='online_update']
    if len(updates) != 8: raise ValueError('Expected eight decisions')
    if timing == 'after_rollout8':
        if any(e['version_before'] != version for e in updates) or any(
                e['optimizer_updates'] or e['version_after'] != version or
                e['online_selector_hash_before'] != e['online_selector_hash_after'] for e in updates[:7]):
            raise ValueError('Actor must stay fixed throughout collection')
        records = updates[-1]['optimizer_updates']
        if len(records) != 8 or [u['collection_index'] for u in records] != list(range(8)):
            raise ValueError('Must consume each collected group once in chronological order')
        if any(u['context_hash'] != e['context_hash'] for u,e in zip(records, updates)):
            raise ValueError('Optimizer used a different collected context')
        for u in records:
            ledger.record_decision(token, u['version_before'], u['version_after'], u['optimized'])
    for event in updates:
        if timing == 'per_step':
            ledger.record_decision(token, event['version_before'], event['version_after'], event['optimized'])
        feedback = event['branch_feedback']
        if fixed:
            from .fixed_protocol import validate
            validate(event['observation_identity'], event['selected'], event['candidate_hash'],
                     event['main_receipt'], feedback, True, expected_contract=spec['reward_contract'])
        else:
            for key in ('canonical_restored', 'selected_parity', 'selected_reward_parity', 'selected_history_parity'):
                if feedback.get(key) is not True: raise ValueError('Branch parity failed: '+key)
        if len(feedback.get('rewards', [])) != 20: raise ValueError('Missing rewards')
    ledger.close_episode(8)
    if (report['policy_version'] != ledger.policy_version or report['actual_optimizer_steps'] != ledger.policy_version-version
            or report['cumulative_optimizer_steps'] != ledger.policy_version):
        raise ValueError('Optimizer counter mismatch')
    checkpoint = report['learner_checkpoint']
    expected_disposable = not bool(spec.get('formal_training', False))
    if (checkpoint.get('disposable') is not expected_disposable or
            sha256_file(checkpoint['path']) != checkpoint['sha256']):
        raise ValueError('Checkpoint checksum/disposable mismatch')
    return dict(episode=index, scene=row['scene_id'], policy_version=ledger.policy_version,
                attempts=ledger.attempts, learner_hash=report['learner_state_hash_final'],
                checkpoint=checkpoint, initialization_provenance=report['initialization_provenance'],
                optimizer_steps=report['actual_optimizer_steps'],
                signal_groups=report.get('groups_with_signal'),
                selected_rewards=[float(e['branch_feedback']['rewards'][e['selected']]['reward']) for e in updates],
                reward_std=[e['update']['reward_std'] for e in updates])


def load_boundary(run_dir, spec):
    saved = json.loads((run_dir/'contract.json').read_text())
    if saved != spec: raise ValueError('Resume contract/settings/source changed; use a fresh run')
    state = json.loads((run_dir/'progress.json').read_text())
    records = state['completed']
    if state.get('next_episode') != len(records) or len(records) > spec['episodes']:
        raise ValueError('Invalid scene cursor')
    version, previous_hash, provenance = 0, None, None
    for i, record in enumerate(records):
        path = checked_path(record['report'])
        if sha256_file(path) != record['report_sha256']: raise ValueError('Episode report checksum mismatch')
        report = json.loads(path.read_text())
        verified = verify_episode(report, spec['schedule'][i], i, version, previous_hash, spec)
        if any(record.get(k) != v for k,v in verified.items()): raise ValueError('Boundary record mismatch')
        if i and verified['initialization_provenance'] != provenance: raise ValueError('Initialization changed')
        provenance = verified['initialization_provenance']
        version, previous_hash = verified['policy_version'], verified['learner_hash']
    return state


def execute(run_dir, settings, spec, resume=False, stop_after=None, probe=None):
    from . import online_probe
    if probe is None: probe = online_probe.main
    base = json.loads(Path(settings).read_text())
    if resume:
        state = load_boundary(run_dir, spec)
    else:
        if (run_dir/'contract.json').exists(): raise FileExistsError('Existing run requires --resume')
        atomic_json(run_dir/'contract.json', spec)
        state = dict(status='RUNNING', completed=[], next_episode=0, formal_ready=False,
                     used_for_formal_training=False, resumed_segments=0)
        atomic_json(run_dir/'progress.json', state)
    if state['next_episode'] == spec['episodes']:
        # A prior interrupted attempt may have left diagnostic fields behind.
        # Once the boundary ledger is complete, expose a clean terminal state.
        state.update(status='COMPLETE_PROVISIONAL', active_episode=None,
                     active_report=None)
        state.pop('error', None)
        atomic_json(run_dir/'progress.json', state)
        print('All requested episodes already completed.', flush=True); return 0
    if resume: state['resumed_segments'] += 1
    # Resume always starts a new parent. Refuse accidental reuse from another run.
    if online_probe._RESIDENT_LEARNER is not None or online_probe._RESIDENT_MODEL is not None:
        raise RuntimeError('Provisional run requires a fresh parent process')
    start_index = state['next_episode']
    state.update(status='RUNNING', active_episode=start_index,
                 active_report=None)
    state.pop('error', None)
    atomic_json(run_dir/'progress.json', state)
    try:
        for i in range(start_index, spec['episodes']):
            attempt = run_dir/('episode%03d_%s' % (i, uuid.uuid4().hex[:12]))
            attempt.mkdir()
            output, cfg = attempt/'report.json', attempt/'settings.json'
            row = spec['schedule'][i]
            _episode_settings(base, row, cfg, i)
            prev = state['completed'][-1] if state['completed'] else None
            version = prev['policy_version'] if prev else 0
            argv = ['provisional-episode', '--settings', str(cfg), '--output', str(output),
                    '--steps', '8', '--seed', str(spec['seed']), '--policy-version', str(version),
                    '--throughput', '--resident', '--branch-workers', str(spec['branch_workers'])]
            if i == start_index and prev:
                argv += ['--resume', prev['checkpoint']['path']]
            state.update(active_episode=i, active_report=str(output))
            atomic_json(run_dir/'progress.json', state)
            print(json.dumps(dict(event='episode_start', episode=i+1, total=spec['episodes'],
                                  scene=row['scene_id'], log=str(attempt/'episode.log'))), flush=True)
            old_argv = sys.argv
            started = time.monotonic()
            with (attempt/'episode.log').open('x', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                sys.argv = argv
                try: code = probe()
                finally: sys.argv = old_argv
            if code not in (0,2): raise RuntimeError('Episode process failed: '+str(code))
            report = json.loads(output.read_text())
            record = verify_episode(report,row,i,version,prev['learner_hash'] if prev else None,spec)
            if prev and record['initialization_provenance'] != prev['initialization_provenance']:
                raise ValueError('Initialization changed')
            record.update(report=str(output), report_sha256=sha256_file(output),
                          wall_seconds=time.monotonic()-started)
            state['completed'].append(record)
            state['next_episode'] = i+1
            state['status'] = 'COMPLETE_PROVISIONAL' if i+1==spec['episodes'] else 'RUNNING'
            state.pop('error', None)
            if state['status'] == 'COMPLETE_PROVISIONAL':
                state.update(active_episode=None, active_report=None)
            atomic_json(run_dir/'progress.json', state)
            print(json.dumps(dict(event='episode_committed', episode=i+1, attempts=record['attempts'],
                                  optimizer_steps=record['policy_version'], checkpoint=record['checkpoint']['path'])), flush=True)
            if stop_after and i+1-start_index >= stop_after and i+1 < spec['episodes']:
                state['status']='STOPPED_AT_BOUNDARY'; atomic_json(run_dir/'progress.json',state); return 0
    except BaseException as error:
        state.update(status='INTERRUPTED_OR_FAILED', error=repr(error))
        atomic_json(run_dir/'progress.json',state)
        raise
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--scene-manifest',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=64)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--branch-workers',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if not 3<=a.episodes<=64 or a.seed<0 or not 0<=a.branch_workers<=20:
        p.error('Use 3..64 episodes, nonnegative seed, 0..20 workers')
    if a.stop_after is not None and a.stop_after<1: p.error('stop-after must be positive')
    from .online_probe import source_hashes
    count=len(json.loads(a.scene_manifest.read_text())['scenes'])
    _,rows=load_assignments(a.scene_manifest,a.settings,count,verify_files=True)
    if json.loads(a.settings.read_text()).get('online_parameterization')!='v3_initialized_selector_finetune':
        p.error('Expected initialized V3 selector')
    run=checked_path(a.run_dir.absolute(),must_exist=False)
    code=Path(__file__).resolve().parents[5]
    if run==code or code in run.parents: p.error('Output must be outside code')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'run.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This run is already active')
        spec=contract(a.settings,a.scene_manifest,rows,a.episodes,a.seed,a.branch_workers,source_hashes())
        return execute(run,a.settings,spec,a.resume,a.stop_after)


if __name__=='__main__':
    raise SystemExit(main())
