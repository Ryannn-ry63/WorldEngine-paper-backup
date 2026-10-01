#!/usr/bin/env python3
"""Run bounded Innovation 3 checks in the explicitly configured environment."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import os
from selector_runtime import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--scene-manifest', type=Path, help='explicit distinct-scene DDP engineering manifest')
    parser.add_argument('--scene-count', type=int, default=2, help='snapshot/protocol probes')
    parser.add_argument('--steps', type=int, default=None, help='default 8 for snapshot-probe, 2 for h1-protocol-probe')
    parser.add_argument('--seed', type=int, default=0, help='snapshot/protocol probes')
    parser.add_argument('--branch-workers', type=int, default=0,
                        help='persistent CPU SimEngine workers per online rank (efficiency pilot)')
    parser.add_argument('--episodes', type=int, default=16,
                        help='continuous online-session episodes')
    parser.add_argument('--session-mode', choices=('rebuild', 'resident'), default='rebuild',
                        help='online session resource lifecycle')
    parser.add_argument('--resume', action='store_true', help='resume a completed episode boundary when supported')
    parser.add_argument('--split', choices=('train', 'dev', 'holdout'),
                        help='formal manifest split; the training runner accepts train only')
    parser.add_argument('stage', choices=['preflight', 'learner-smoke', 'snapshot-probe', 'h1-protocol-probe', 'visual-probe', 'live-reward-probe', 'online-probe', 'online-throughput-probe', 'online-session-probe', 'ddp-h1-probe', 'ddp-online-throughput-probe', 'ddp-online-session-gate', 'reward-probe'])
    parser.add_argument('--stop-after-rounds', type=int, help='DDP session: stop at a committed round')
    parser.add_argument('--fixed-segment', action='store_true', help='Use exact four-second R reward')
    parser.add_argument('--wall-seconds', type=float, default=43200.)
    parser.add_argument('--update-timing', choices=('per_step', 'after_rollout8'), default='per_step')
    parser.add_argument('--check-only', action='store_true', help='Fixed session: validate imports and manifest without training')
    args = parser.parse_args()
    if args.fixed_segment and args.stage != 'ddp-online-session-gate':
        parser.error('--fixed-segment requires ddp-online-session-gate')
    if args.check_only and not args.fixed_segment:
        parser.error('--check-only requires --fixed-segment')
    if args.fixed_segment and (args.split != 'train' or not 0 < args.wall_seconds <= 43200):
        parser.error('Fixed session requires --split train and wall-seconds in (0, 43200]')
    if (args.resume or args.stop_after_rounds is not None) and args.stage != 'ddp-online-session-gate':
        parser.error('resume/stop-after-rounds require ddp-online-session-gate')
    if args.steps is None:
        args.steps = 2 if args.stage in ('h1-protocol-probe', 'ddp-h1-probe') else (1 if args.stage in ('visual-probe', 'live-reward-probe') else 8)
    if not 0 <= args.branch_workers <= 20:
        parser.error('--branch-workers must be in [0, 20]')
    if args.branch_workers and args.stage not in ('online-probe', 'online-throughput-probe',
                                                   'online-session-probe', 'ddp-online-throughput-probe', 'ddp-online-session-gate'):
        parser.error('--branch-workers is only supported by online throughput stages')
    if args.scene_manifest and args.stage not in ('ddp-online-throughput-probe', 'ddp-online-session-gate', 'online-session-probe'):
        parser.error('--scene-manifest is only supported for real DDP throughput')
    if args.split and args.stage != 'ddp-online-session-gate':
        parser.error('--split is only supported for formal DDP session training')
    cfg, env = environment(args.settings, args.devices)
    # Keep the immutable experiment paths, but prepend this isolated hotfix
    # so every DDP child imports the patched learner modules.
    overlay_root = Path(__file__).resolve().parent
    env['PYTHONPATH'] = str(overlay_root) + os.pathsep + env['PYTHONPATH']
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    modules = {'preflight': 'innovation3.preflight', 'learner-smoke': 'innovation3.smoke',
               'snapshot-probe': 'innovation3.snapshot_probe'}
    modules['h1-protocol-probe'] = 'innovation3.h1_protocol_probe'
    modules['visual-probe'] = 'innovation3.visual_online_probe'
    modules['live-reward-probe'] = 'innovation3.visual_online_probe'
    modules['online-probe'] = 'innovation3.online_probe'
    modules['online-throughput-probe'] = 'innovation3.online_probe'
    modules['online-session-probe'] = 'innovation3.online_session_probe'
    modules['ddp-online-throughput-probe'] = 'innovation3.ddp_online_probe'
    modules['ddp-online-session-gate'] = 'innovation3.ddp_online_session'
    modules['ddp-h1-probe'] = 'innovation3.ddp_h1_probe'
    modules['reward-probe'] = 'innovation3.reward_probe'
    python = cfg['simengine_python' if args.stage in ('snapshot-probe', 'reward-probe') else 'algengine_python']
    command = [python, '-m', modules[args.stage],
               '--settings', str(args.settings.absolute()), '--output', str(args.output.absolute())]
    if args.stage == 'preflight':
        command += ['--probe', '--required-gpus', str(len(args.devices.split(',')))]
    elif args.stage == 'learner-smoke':
        if len(args.devices.split(',')) != 1:
            parser.error('Learner smoke is single-rank; it is not a DDP test')
        command += ['--device', 'cuda:0']
    elif args.stage in ('visual-probe', 'live-reward-probe', 'online-probe', 'online-throughput-probe'):
        if len(args.devices.split(',')) != 1:
            parser.error('Visual/online probe is single-GPU; use one H100 first')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command += ['--steps', str(args.steps), '--seed', str(args.seed)]
        if args.stage == 'live-reward-probe':
            command.append('--reward-adapter')
        if args.stage == 'online-throughput-probe':
            command.append('--throughput')
        if args.branch_workers:
            command += ['--branch-workers', str(args.branch_workers)]
    elif args.stage == 'online-session-probe':
        if len(args.devices.split(',')) != 1:
            parser.error('Online session probe is single-GPU; use one rank first')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command += ['--steps', str(args.steps), '--seed', str(args.seed),
                    '--episodes', str(args.episodes), '--mode', args.session_mode,
                    '--throughput']
        if args.branch_workers:
            command += ['--branch-workers', str(args.branch_workers)]
    elif args.stage == 'ddp-h1-probe':
        count = len(args.devices.split(','))
        if count < 2:
            parser.error('DDP H1 pilot requires at least two explicit GPU IDs')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [python, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node', str(count), '-m', modules[args.stage],
                   '--settings', str(args.settings.absolute()),
                   '--output', str(args.output.absolute()), '--steps', str(args.steps),
                   '--resume-after', str(max(1, args.steps // 2)), '--seed', str(args.seed)]
    elif args.stage == 'ddp-online-throughput-probe':
        count = len(args.devices.split(','))
        if count < 2:
            parser.error('Real DDP online pilot requires at least two explicit GPU IDs')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [python, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node', str(count), '-m', modules[args.stage],
                   '--settings', str(args.settings.absolute()),
                   '--output', str(args.output.absolute()), '--steps', str(args.steps),
                   '--seed', str(args.seed)]
        if args.branch_workers:
            command += ['--branch-workers', str(args.branch_workers)]
    elif args.stage == 'ddp-online-session-gate':
        count = len(args.devices.split(','))
        if count < 2:
            parser.error('DDP online session gate requires at least two explicit GPU IDs')
        if not args.scene_manifest:
            parser.error('DDP online session gate requires --scene-manifest')
        if args.episodes < 1:
            parser.error('DDP online session gate requires positive --episodes')
        env.update(OPENBLAS_CORETYPE='Prescott', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        command = [python, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc_per_node', str(count), '-m', modules[args.stage],
                   '--settings', str(args.settings.absolute()),
                   '--output', str(args.output.absolute()), '--episodes', str(args.episodes),
                   '--steps', str(args.steps), '--seed', str(args.seed)]
        command += ['--branch-workers', str(args.branch_workers)]
        if args.split:
            command += ['--split', args.split]
        if args.resume:
            command.append('--resume')
        if args.stop_after_rounds is not None:
            command += ['--stop-after-rounds', str(args.stop_after_rounds)]
        if args.fixed_segment:
            command += ['--fixed-segment', '--wall-seconds', str(args.wall_seconds), '--update-timing', args.update_timing]
    elif args.stage == 'reward-probe':
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott')
        command += ['--steps', str(args.steps), '--seed', str(args.seed)]
    elif args.stage == 'snapshot-probe':
        # Dynamics workers use CPU only; never disturb an occupied GPU.
        # Pin before the child imports NumPy. The installed OpenBLAS 0.3.20
        # Cooperlake path fails SVD/pinv health checks on our target runtime.
        # Use this same numeric policy for future rendered-main/branch workers.
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott')
        command += ['--scene-count', str(args.scene_count), '--steps', str(args.steps), '--seed', str(args.seed)]
    else:
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott')
        command += ['--scene-count', str(args.scene_count), '--steps', str(args.steps), '--seed', str(args.seed)]
    if args.scene_manifest:
        from innovation3.scene_manifest import load_assignments
        if args.fixed_segment:
            from innovation3.fixed_manifest import load_assignments
            print('Checking fixed reward manifest and scene/asset checksums...', flush=True)
        manifest, _ = load_assignments(args.scene_manifest, args.settings,
                                       max(2, len(args.devices.split(','))), split=args.split)
        if args.steps != manifest['steps']:
            parser.error('Steps differ from the explicit scene manifest')
        command += ['--scene-manifest', str(args.scene_manifest.absolute())]
    with tempfile.TemporaryDirectory(prefix='innovation3-mpl-') as cache:
        env['MPLCONFIGDIR'] = cache
        if args.fixed_segment:
            # Exercise the actual import graph in a fresh interpreter with the
            # same bootstrap/environment as torchrun, before starting 8 ranks.
            print('Checking AlgEngine online imports...', flush=True)
            probe = (
                'from pathlib import Path; import os; import worldengine; '
                'from innovation3 import ddp_online_session, online_probe, fixed_learning, fixed_client; '
                'from innovation3 import learner; '
                'assert learner.LOGIT_DRIFT_ATOL == 5e-5 and fixed_learning.LOGIT_DRIFT_ATOL == 5e-5; '
                'assert Path(learner.__file__).resolve().parent.parent == Path(' + repr(str(overlay_root)) + '); '
                'print("PASS_NUMERIC_HOTFIX:", learner.__file__, flush=True); '

                'assert Path(worldengine.__file__).resolve().is_relative_to(Path(os.environ["SIMENGINE_ROOT"]).resolve()); '
                'print("PASS_FIXED_ALG_IMPORTS:", worldengine.__file__, flush=True)'
            )
            subprocess.run([python, '-u', '-c', probe], env=env, check=True)
            print('Checking SimEngine worker imports...', flush=True)
            subprocess.run([cfg['simengine_python'], '-u', '-c',
                            'from innovation3 import fixed_worker; print("PASS_FIXED_SIM_IMPORTS:", fixed_worker.__file__, flush=True)'],
                           env=env, check=True)
            print('Fixed reward manifest and startup imports passed.', flush=True)
            if args.check_only:
                print('CHECK_ONLY: no DDP workers or training started.', flush=True)
                return 0
            print('Starting foreground torchrun: ' + repr(command), flush=True)
        return subprocess.run(command, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
