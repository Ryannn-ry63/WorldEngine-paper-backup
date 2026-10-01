"""Provisional synchronized sessions with verified all-rank boundaries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from datetime import timedelta
import uuid
import time
from .paths import checked_path, sha256_file
from .scene_manifest import load_assignments
from .online_session_probe import _episode_settings
from . import ddp_online_probe as ddp
from .ddp_session_boundary import (assignment, receipt, open_boundary,
                                   commit_round, finish, atomic_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--scene-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64, help='Total across all ranks')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split', choices=('train', 'dev', 'holdout'))
    parser.add_argument('--branch-workers', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--continuation-manifest', type=Path, help='Explicit p6 migration; only for a fresh ledger')
    parser.add_argument('--update-timing', choices=('per_step', 'after_rollout8'), default='per_step')
    parser.add_argument('--stop-after-rounds', type=int)
    parser.add_argument('--fixed-segment', action='store_true')
    parser.add_argument('--wall-seconds', type=float, default=43200.)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 43200:
        parser.error('Wall budget must be in (0, 43200] seconds')
    segment_started = time.monotonic()
    rank = int(os.environ.get('RANK', '-1'))
    world = int(os.environ.get('WORLD_SIZE', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if rank < 0 or world < 2 or not 0 <= local_rank < world or len(visible) < world:
        parser.error('Use single-node torchrun with one explicitly visible device per rank')
    if not 1 <= args.episodes <= 4096 or args.episodes % world or args.steps != 8:
        parser.error('Use 8 steps and 1..4096 total episodes divisible by world size')
    if not 0 <= args.seed <= 2 or not 0 <= args.branch_workers <= 20:
        parser.error('Formal runs use seed 0, 1, or 2; branch workers are 0..20')
    if args.stop_after_rounds is not None and args.stop_after_rounds < 1:
        parser.error('stop-after-rounds must be positive')
    settings = checked_path(args.settings)
    manifest_path = checked_path(args.scene_manifest)
    output = checked_path(args.output.absolute(), must_exist=False)
    code = Path(__file__).resolve().parents[5]
    if output == code or code in output.parents:
        parser.error('Output must be outside code')
    assignment_loader = load_assignments
    if args.fixed_segment:
        from .fixed_manifest import load_assignments as assignment_loader
    manifest, rows = assignment_loader(manifest_path, settings, world,
                                       verify_files=False, split=args.split)
    is_formal = manifest.get('purpose') == 'online_formal_train_dev_holdout'
    if is_formal and args.split != 'train':
        parser.error('The training runner may consume only the formal train split')
    if args.fixed_segment and args.split != 'train':
        parser.error('Fixed online training may consume only its development train split')
    if not is_formal and not args.fixed_segment and (args.seed != 0 or args.split is not None):
        parser.error('Engineering manifests are seed0-only and do not accept --split')
    base = json.loads(settings.read_text())
    if base.get('online_parameterization') != 'v3_initialized_selector_finetune':
        parser.error('Expected initialized V3 selector')
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    os.environ['WORLDENGINE_DDP_LOCAL_GPU'] = visible[local_rank]
    from . import online_probe
    from .learner import OnlineV3Learner
    from .live_learning import state_digest
    from worldengine.online.execution import DEFAULT_EGO_CONTROLLER, execution_contract
    controller = base.get('online_ego_controller', DEFAULT_EGO_CONTROLLER)
    from .fixed_protocol import configured_contract
    spec = dict(schema=2, episodes=args.episodes, world_size=world, steps=8, seed=args.seed,
                branch_workers=args.branch_workers, settings_sha256=sha256_file(settings),
                manifest_sha256=sha256_file(manifest_path), rows=rows,
                source_sha256=online_probe.source_hashes(),
                parameterization=base['online_parameterization'],
                reward_contract=(configured_contract(base) if args.fixed_segment
                                 else 'causal_h1_pdm_components_v1'), ego_controller=controller,
                dynamics_contract=execution_contract(controller),
                schedule=('seeded_cyclic_global_episode_v2' if manifest.get('schema') == 4 else 'rank_plus_round_rotation_global_episode_seed_v1'),
                online_update_timing=args.update_timing,
                formal_manifest=is_formal, split=args.split,
                formal_training=is_formal)
    if args.fixed_segment:
        spec.update(fixed_segment=True, simulator_backend=base.get('fixed_backend', 'cached'),
                    full_audit_every=base.get('fixed_full_audit_every', 32),
                    wall_budget_seconds=args.wall_seconds, scoring_frequency='all8')
    ddp.torch, ddp.dist, ddp.state_digest = torch, dist, state_digest
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=1800))
    original = online_probe.OnlineV3Learner
    online_probe.OnlineV3Learner = ddp.distributed_learner_class(OnlineV3Learner)
    lock = None
    try:
        properties = torch.cuda.get_device_properties(local_rank)
        binding = dict(rank=rank, local_rank=local_rank, current_device=torch.cuda.current_device(),
                       visible_devices=visible, cuda_uuid=getattr(properties, 'uuid', None))
        bindings = [None] * world
        dist.all_gather_object(bindings, binding)
        ddp.validate_bindings(bindings, world)
        specs = [None] * world
        dist.all_gather_object(specs, spec)
        if any(s != spec for s in specs):
            raise ValueError('Rank contracts differ')
        state_box = [None]
        if rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            lock = output.with_suffix('.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assignment_loader(manifest_path, settings, world, verify_files=True, split=args.split)
            if args.continuation_manifest:
                if args.resume:
                    raise ValueError('Use import or resume, never both')
                from .continuation import import_boundary
                reference = dict(path=str(args.continuation_manifest.resolve()), sha256=sha256_file(args.continuation_manifest))
                state_box[0] = import_boundary(output, spec, reference)
            else:
                state_box[0] = open_boundary(output, spec, args.resume)
            if args.resume and state_box[0]['completed'] < args.episodes:
                state_box[0]['resumed_segments'] += 1
                atomic_json(output.with_suffix('.progress.json'), state_box[0])
        dist.broadcast_object_list(state_box, src=0)
        state = state_box[0]
        previous_wall = state.get('training_wall_seconds', 0.)
        start = len(state['rounds'])
        for index in range(start, args.episodes // world):
            previous = state['rounds'][-1][rank] if state['rounds'] else None
            version = previous['policy_version'] if previous else 0
            # Fresh attempts preserve evidence from every interrupted round.
            attempt = output.parent / (output.stem + '.attempts') / (
                'round%03d_rank%d_%s' % (index, rank, uuid.uuid4().hex))
            attempt.mkdir(parents=True)
            cfg, report_path = attempt/'settings.json', attempt/'report.json'
            _episode_settings(base, assignment(spec, index, rank), cfg,
                              index * world + rank, seed=args.seed)
            # The episode checkpoint kind is part of the formal training
            # contract; engineering pilots remain disposable and resumable.
            cfg_data = json.loads(cfg.read_text())
            cfg_data['formal_training'] = is_formal
            cfg_data['online_update_timing'] = args.update_timing
            legacy = state.get('lineage') if index == 113 else None
            if legacy:
                cfg_data['legacy_continuation'] = legacy['reference']
            cfg.write_text(json.dumps(cfg_data, indent=2) + '\n')
            argv = ['online-probe', '--settings', str(cfg), '--output', str(report_path),
                    '--steps', '8', '--seed', str(args.seed), '--policy-version', str(version),
                    '--throughput', '--resident', '--branch-workers', str(args.branch_workers)]
            if args.fixed_segment:
                argv.append('--fixed-segment')
            if (args.resume or args.continuation_manifest) and index == start and previous:
                checkpoint = previous['checkpoint']['path']
                if legacy:
                    from .continuation import load_import
                    imported, _ = load_import(legacy['reference'], spec['source_sha256'])
                    checkpoint = imported['checkpoints'][rank]['path']
                argv += ['--resume', checkpoint]
            with ddp._argv(argv):
                result = online_probe.main()
            if result not in (0, 2):
                raise RuntimeError('Rank episode failed')
            record = receipt(report_path, spec, index, rank, previous)
            records = [None] * world
            dist.all_gather_object(records, record)
            if rank == 0:
                state['training_wall_seconds'] = previous_wall + time.monotonic()-segment_started
                commit_round(output, spec, state, records)
                print(json.dumps(dict(event='round_committed', total_episodes=state['completed'],
                                      policy_version=records[0]['policy_version'])), flush=True)
            else:
                state['rounds'].append(records)
                state['completed'] = (index + 1) * world
            dist.barrier()
            external_stop = [None]
            if rank == 0:
                stop_file = base.get('campaign_stop_file')
                clock_file = base.get('campaign_deadline_file')
                if stop_file and Path(stop_file).exists():
                    external_stop[0] = 'CORRECTNESS_STOPPED_AT_BOUNDARY'
                elif clock_file and Path(clock_file).exists():
                    deadline = json.loads(Path(clock_file).read_text())['deadline_unix']
                    if time.time() + max(r['elapsed_seconds'] for r in records) >= deadline - 60:
                        external_stop[0] = 'CAMPAIGN_DEADLINE_AT_BOUNDARY'
            dist.broadcast_object_list(external_stop, src=0)
            if external_stop[0] and state['completed'] < args.episodes:
                online_probe.close_fixed_runtime()
                if rank == 0:
                    state['status'] = external_stop[0]
                    atomic_json(output.with_suffix('.progress.json'), state)
                dist.barrier()
                return 0
            budget_stop = [False]
            if rank == 0:
                elapsed = previous_wall + time.monotonic()-segment_started
                # Reserve the slowest observed full round before starting another.
                estimate = max(r['elapsed_seconds'] for r in records)
                budget_stop[0] = elapsed + estimate >= args.wall_seconds
            dist.broadcast_object_list(budget_stop, src=0)
            if budget_stop[0] and state['completed'] < args.episodes:
                online_probe.close_fixed_runtime()
                if rank == 0:
                    state['status'] = 'BUDGET_EXHAUSTED_AT_BOUNDARY'
                    atomic_json(output.with_suffix('.progress.json'), state)
                dist.barrier()
                return 0
            if (args.stop_after_rounds and index + 1 - start >= args.stop_after_rounds
                    and state['completed'] < args.episodes):
                if rank == 0:
                    state['status'] = 'STOPPED_AT_BOUNDARY'
                    atomic_json(output.with_suffix('.progress.json'), state)
                dist.barrier()
                online_probe.close_fixed_runtime()
                return 0
        online_probe.close_fixed_runtime()
        result_box = [finish(output, spec, state) if rank == 0 else None]
        dist.broadcast_object_list(result_box, src=0)
        return result_box[0]
    finally:
        online_probe.close_fixed_runtime(abort=True)
        online_probe.OnlineV3Learner = original
        if lock is not None:
            lock.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
