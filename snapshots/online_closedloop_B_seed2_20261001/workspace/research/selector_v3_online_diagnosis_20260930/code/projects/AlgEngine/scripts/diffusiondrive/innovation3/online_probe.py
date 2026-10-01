"""Bounded single-GPU live full20/H1 episode used by pilots or formal runs."""
import argparse
import json
import os
import signal
from pathlib import Path
import socket
import subprocess
import time
import traceback

import numpy as np
import torch

from .candidate_inputs import from_export
from .image_transport import ImageBuffer, metadata_from_wire
from .learner import (FROZEN_V3_PLUS_ZERO_RESIDUAL, OnlineV3Learner,
                      PARAMETERIZATIONS)
from .v3_checkpoint_contract import is_registered_v3_selector_payload
from .live_audit import LiveWindow
from .live_learning import LiveLearning, learning_evidence, state_digest
from .paths import checked_path, sha256_file
from .transport import Channel
from worldengine.online.execution import (DEFAULT_EGO_CONTROLLER, execution_contract,
                                          validate_resume_execution)


_RESIDENT_MODEL = None
_RESIDENT_LEARNER = None
_RESIDENT_PARAMETERIZATION = None
_RESIDENT_SOURCE_SHA256 = None
_RESIDENT_CONFIG = None
_RESIDENT_FIXED_CLIENT = None


def close_fixed_runtime(abort=False):
    global _RESIDENT_FIXED_CLIENT
    if _RESIDENT_FIXED_CLIENT is not None:
        client, _RESIDENT_FIXED_CLIENT = _RESIDENT_FIXED_CLIENT, None
        client.close(abort=abort)


def source_hashes():
    code = Path(__file__).resolve().parents[5]
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((code/'projects/SimEngine/worldengine/online').glob('*.py'))
    paths += list((code/'projects/SimEngine/worldengine/components/agents/policy/pdm_planner').rglob('*.py'))
    paths += [code/name for name in (
        'projects/SimEngine/worldengine/manager/data_manager.py',
        'projects/SimEngine/worldengine/manager/metric_manager.py',
        'projects/SimEngine/worldengine/utils/math_utils.py',
        'projects/SimEngine/worldengine/components/agents/navigation/trajectory_navigation.py',
        'projects/SimEngine/worldengine/components/maps/road_networks/edge_road_network.py',
        'projects/SimEngine/worldengine/components/maps/lanes/base_center_lane.py',
        'projects/SimEngine/worldengine/components/agents/policy/idm_policy.py',
        'projects/SimEngine/worldengine/components/agents/navigation/idm_navigation.py',
        'projects/SimEngine/worldengine/manager/agent_manager.py',
        'projects/SimEngine/worldengine/components/agents/controller/log_play_controller.py',
        'projects/SimEngine/worldengine/components/agents/ego_agent.py',
        'projects/SimEngine/worldengine/components/agents/client/navformer_client.py',
        'projects/SimEngine/worldengine/components/agents/vehicle_model/vehicle_utils.py',
        'projects/SimEngine/worldengine/components/agents/controller/tracker/lqr_tracker.py',
        'projects/SimEngine/worldengine/components/agents/controller/two_stage_controller.py',
        'projects/SimEngine/worldengine/components/agents/controller/motion_model/kinematic_bicycle.py',
        'projects/AlgEngine/scripts/diffusiondrive/innovation3_runtime.py',
        'projects/AlgEngine/scripts/diffusiondrive/selector_runtime.py',
        'projects/AlgEngine/scripts/diffusiondrive/grpo_selector_v3_cached_common.py',
        'projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py')]
    return {str(p.relative_to(code)): sha256_file(p) for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--throughput', action='store_true',
                        help='measure production-like path without file oracle or duplicate forward')
    parser.add_argument('--resume', type=Path,
                        help='resume the learner/optimizer/RNG at a completed episode boundary')
    parser.add_argument('--policy-version', type=int, default=0,
                        help='learner version expected by the episode worker')
    parser.add_argument('--resident', action='store_true',
                        help='keep model and learner alive for the caller\'s next episode')
    parser.add_argument('--branch-workers', type=int, default=0,
                        help='experimental persistent CPU SimEngine branch workers (0 keeps serial H1)')
    parser.add_argument('--fixed-segment', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame): raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    if not 0 <= args.branch_workers <= 20:
        parser.error('--branch-workers must be in [0, 20]')
    if not 1 <= args.steps <= 8 or args.seed < 0 or args.policy_version < 0:
        parser.error('Use steps 1..8 and a nonnegative seed')
    output = checked_path(args.output.absolute(), must_exist=False)
    code = Path(__file__).resolve().parents[5]
    if code == output or code in output.parents:
        parser.error('Probe output must be outside code')
    if any(output.with_suffix(s).exists() for s in ('.json', '.worker.log', '.online.pt', '.failure')):
        raise FileExistsError('Use a fresh output basename: ' + str(output))
    cfg = json.loads(checked_path(args.settings).read_text())
    update_timing = cfg.get('online_update_timing', 'per_step')
    if update_timing not in ('per_step', 'after_rollout8') or (update_timing == 'after_rollout8' and
            (not args.fixed_segment or args.steps != 8)):
        raise ValueError('Delayed updates require a full fixed eight-step rollout')
    warmup_steps = 3 if args.fixed_segment else 13
    from .fixed_protocol import configured_contract
    reward_name = configured_contract(cfg) if args.fixed_segment else 'causal_h1_pdm_components_v1'
    formal_training = bool(cfg.get('formal_training', False))
    ego_controller = cfg.get('online_ego_controller', DEFAULT_EGO_CONTROLLER)
    dynamics_contract = execution_contract(ego_controller)
    parameterization = cfg.get('online_parameterization', FROZEN_V3_PLUS_ZERO_RESIDUAL)
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError('Unknown online_parameterization: ' + str(parameterization))
    initialization_source = None
    if all(key in cfg for key in ('selector_state', 'expected_sha256')):
        selector_path = checked_path(cfg['selector_state'])
        expected_selector_sha = cfg['expected_sha256'].get('selector_state')
        if expected_selector_sha and sha256_file(selector_path) != expected_selector_sha:
            raise ValueError('Registered V3 selector SHA256 mismatch before learner construction')
        payload = torch.load(selector_path, map_location='cpu')
        if not is_registered_v3_selector_payload(payload):
            raise ValueError('Unexpected offline V3 selector provenance')
        initialization_source = dict(kind='offline_selector_file', path=str(selector_path),
            sha256=expected_selector_sha, payload_schema=payload.get('schema_version'),
            method=payload.get('method'), scene_selector_config=payload.get('scene_selector_config'))
    candidate_seed = cfg.get('online_candidate_seed', args.seed)
    scene_seed = cfg.get('online_scene_seed', args.seed)
    action_seed = cfg.get('online_action_seed', args.seed)
    if any(type(x) is not int or not 0 <= x < 2**31 for x in (candidate_seed, scene_seed, action_seed)):
        raise ValueError('Invalid online seed namespace')
    report = dict(status='RUNNING', stage=('STRICT_ONLINE_THROUGHPUT' if args.throughput
                                           else 'STRICT_ONLINE_SELECTOR_ONLY'),
        steps=args.steps, seed=args.seed, seed_namespaces=dict(training_seed=args.seed,
            candidate_seed=candidate_seed, scene_seed=scene_seed, action_seed=action_seed), events=[], source_sha256=source_hashes(),
        code_head=(None if args.fixed_segment else subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip()),
        real_generator=False, live_render_verified=False, real_generated_reward=False,
        causal_reward_contract=reward_name,
        official_pdm_reward=False, online_update=False, real_closed_loop=False,
        formal_ready=False, full_resume_verified=False, ddp_verified=False,
        parameterization=parameterization, formal_training=formal_training,
        used_for_formal_training=formal_training,
        ego_controller=ego_controller, dynamics_contract=dynamics_contract,
        online_update_timing=update_timing,
        selection='categorical_T1_private_learner_rng',
        branch_execution=(('parallel_persistent_full20_workers_%d' % args.branch_workers)
                           if args.branch_workers else
                           ('serial_full20_production_like' if args.throughput
                            else 'serial_full20_correctness_reference')),
        branch_workers=args.branch_workers, throughput_mode=bool(args.throughput), profile_events=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
    child = live = images = channel = left = right = fixed_client = None
    episode_complete = False
    started = time.monotonic()
    worker_log = output.with_suffix('.worker.log')
    try:
        # Import heavy visual dependencies only inside the recorded failure boundary.
        from .live_inputs import LiveInputs
        from .visual_model import configuration, load_frozen, reset_temporal_state
        from .visual_parity import assert_same, file_oracle, result_parity
        global _RESIDENT_MODEL, _RESIDENT_LEARNER, _RESIDENT_PARAMETERIZATION, _RESIDENT_SOURCE_SHA256, _RESIDENT_CONFIG, _RESIDENT_FIXED_CLIENT
        # Episode-specific inputs may change; model/input contracts may not.
        resident_config = {k: v for k, v in cfg.items() if k not in (
            'visual_scene_id', 'visual_scene_path', 'visual_asset_path',
            'visual_asset_sha256', 'visual_scene_sha256', 'online_candidate_seed',
            'online_scene_seed', 'online_action_seed')}
        resident_reuse = bool(args.resident and _RESIDENT_MODEL is not None)
        if resident_reuse and (_RESIDENT_PARAMETERIZATION != parameterization or
                _RESIDENT_SOURCE_SHA256 != (initialization_source or {}).get('sha256') or
                _RESIDENT_CONFIG != resident_config):
            raise RuntimeError('Resident learner provenance/parameterization mismatch; rebuild required')
        if not resident_reuse:
            torch.manual_seed(candidate_seed)
            np.random.seed(candidate_seed)
        config = configuration(cfg, candidate_seed)
        if resident_reuse:
            model, learner = _RESIDENT_MODEL, _RESIDENT_LEARNER
            reset_temporal_state(model)
        else:
            model = load_frozen(config, cfg)
            selector = model.module.planning_head.scene_selector
            learner = OnlineV3Learner(selector, seed=action_seed,
                                       parameterization=parameterization,
                                       initialization_source=initialization_source)
            if args.resident:
                _RESIDENT_MODEL, _RESIDENT_LEARNER = model, learner
                _RESIDENT_PARAMETERIZATION = parameterization
                _RESIDENT_SOURCE_SHA256 = (initialization_source or {}).get('sha256')
                _RESIDENT_CONFIG = resident_config
        selector = model.module.planning_head.scene_selector
        # Rebind deterministic diffusion noise for each episode. Rebuild and
        # resident modes must see the same candidate bank under the same
        # episode seed even though resident keeps the model object alive.
        model.module.planning_head.set_candidate_noise_namespace(
            config.model.planning_head.candidate_noise_namespace)
        initial_model_hash = state_digest(model.state_dict())
        if args.resume:
            resume = checked_path(args.resume)
            payload = torch.load(resume, map_location='cpu')
            expected_kinds = {'DISPOSABLE_STRICT_ONLINE_PROBE_NOT_FORMAL_TRAINING',
                              'FORMAL_ONLINE_TRAINING_CHECKPOINT'}
            if payload.get('kind') not in expected_kinds or \
                    bool(payload.get('formal_training', False)) != formal_training:
                raise ValueError('Unexpected online learner checkpoint kind')
            if payload.get('parameterization', payload.get('learner', {}).get('parameterization')) != parameterization:
                raise ValueError('Resume checkpoint parameterization mismatch')
            validate_resume_execution(payload, dynamics_contract)
            if payload.get('source_sha256') != report['source_sha256']:
                raise ValueError('Resume source changed; use a fresh run')
            if args.fixed_segment and payload.get('reward_contract', {}).get('name') != reward_name:
                raise ValueError('Cannot resume a checkpoint from a different reward contract')
            if payload['learner'].get('version') != args.policy_version:
                raise ValueError('Resume checkpoint/version mismatch')
            if payload.get('online_update_timing', 'per_step') != update_timing:
                raise ValueError('Cannot resume a different update timing')
            learner.load_state_dict(payload['learner'])
            if learner.version != args.policy_version:
                raise ValueError('Resume checkpoint/version mismatch')
        if learner.version != args.policy_version:
            raise ValueError('Episode initial policy version mismatch')
        initial_policy_version = learner.version
        initial_attempts = learner.attempts
        initial_learner_hash = state_digest(learner.state_dict())
        if args.fixed_segment:
            from .fixed_learning import FixedLiveLearning
            online = FixedLiveLearning(learner, reward_contract=reward_name, update_timing=update_timing)
        else:
            online = LiveLearning(learner)
        initial_selector_hash = online.last_selector_hash
        live = LiveInputs(config.data.test)
        worker_env = dict(os.environ)
        worker_env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                          MKL_NUM_THREADS='1', OPENBLAS_CORETYPE='Prescott', PYTHONDONTWRITEBYTECODE='1')
        # A real DDP parent keeps all explicitly requested devices visible so
        # NCCL can address rank-local indices.  The renderer is a separate
        # process, so pin only that child to its rank's physical GPU.
        if worker_gpu := worker_env.get('WORLDENGINE_DDP_LOCAL_GPU'):
            worker_env['CUDA_VISIBLE_DEVICES'] = worker_gpu
        if not args.fixed_segment:
            left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
            channel = Channel(left, timeout=1800)
            images = ImageBuffer()
            command = [cfg['simengine_python'], '-u', '-m', 'innovation3.visual_worker',
                '--fd', str(right.fileno()), '--images-fd', str(images.fd),
                '--settings', str(args.settings.resolve()), '--output', str(output),
                '--steps', str(args.steps), '--seed', str(scene_seed),
                '--reward-adapter', '--online-updates', '--policy-version', str(learner.version)]
            if args.branch_workers:
                command += ['--branch-workers', str(args.branch_workers)]
        with worker_log.open('x') as log:
            if args.fixed_segment:
                from .fixed_client import FixedWorkerClient
                fixed_client = _RESIDENT_FIXED_CLIENT if args.resident else None
                backend = cfg.get('fixed_backend', 'cached')
                if fixed_client is None:
                    fixed_client = FixedWorkerClient(cfg, args.branch_workers, backend,
                                                    output.with_suffix('.resident-worker.log'), worker_env)
                    if args.resident:
                        _RESIDENT_FIXED_CLIENT = fixed_client
                if fixed_client.backend != backend or fixed_client.workers != args.branch_workers:
                    raise ValueError('Resident simulator contract changed')
                child, channel, images = fixed_client.child, fixed_client.channel, fixed_client.images
                ready = fixed_client.begin(cfg, output.with_suffix('.traces'), args.steps,
                    learner.version, learner.attempts, cfg.get('fixed_full_audit_every', 32))
                log.write('Resident worker PID %d; pool reused=%s\n' % (child.pid, ready.get('branch_pool_reused')))
            else:
                child = subprocess.Popen(command, env=worker_env, pass_fds=(right.fileno(), images.fd),
                                         stdout=log, stderr=subprocess.STDOUT)
                right.close()
                ready = channel.receive()
            if (ready.get('kind') != 'ready' or not ready.get('reward_adapter') or
                    not ready.get('online_updates') or ready.get('warmup_transitions') != warmup_steps or
                    ready.get('reward_contract', {}).get('name') != report['causal_reward_contract'] or
                    ready.get('dynamics_contract') != dynamics_contract or
                    ready.get('ego_controller') != ego_controller):
                raise RuntimeError('Worker online/reward/warmup contract mismatch')
            if cfg.get('visual_scene_id') and ready.get('scene') != cfg['visual_scene_id']:
                raise RuntimeError('Worker did not use the explicitly assigned scene')
            report['worker'] = ready
            report['gpu'] = dict(name=torch.cuda.get_device_name(0),
                                 total_bytes=torch.cuda.get_device_properties(0).total_memory)
            window = LiveWindow()
            previous_update_time = None
            for index in range(warmup_steps + args.steps):
                step_started_ns = time.monotonic_ns()
                message = channel.receive()
                if message.get('kind') != 'observation':
                    raise RuntimeError('Expected live observation')
                identity = message['identity']
                if identity['step'] != index or identity['scene'] != ready['scene']:
                    raise RuntimeError('Unexpected observation scene/step')
                frame = metadata_from_wire(message['frame'])
                window.append(frame, images.receive(message['images'], identity), identity)
                # The reader owns the shared camera buffer only until the
                # current observation has been copied into the resident
                # four-frame window. Release it before the next render.
                images.release(identity)
                observation_copied_ns = time.monotonic_ns()
                if index < warmup_steps:
                    if identity.get('policy_version') != learner.version:
                        raise RuntimeError('Warmup changed policy version')
                    channel.send(dict(kind='warmup', identity=identity))
                    report['events'].append(dict(kind='warmup', step=index, history_size=len(window.frames),
                                                 elapsed_seconds=(time.monotonic_ns()-step_started_ns)/1e9))
                    continue
                online.observe(identity)
                if previous_update_time is not None and online.times['observe'] <= previous_update_time:
                    raise RuntimeError('Observation did not follow completed update')
                data = live.prepare(list(window.frames), list(window.cameras))
                prepared_ns = time.monotonic_ns()
                if args.throughput:
                    # Correctness mode keeps a file oracle and duplicate
                    # forward. Throughput mode measures the resident path.
                    with torch.no_grad():
                        results = model(return_loss=False, rescale=True, **data)
                    oracle_results = None
                else:
                    oracle = file_oracle(config.data.test, list(window.frames), list(window.cameras))
                    assert_same(data, oracle)
                    with torch.no_grad():
                        results = model(return_loss=False, rescale=True, **data)
                        oracle_results = model(return_loss=False, rescale=True, **oracle)
                forwarded_ns = time.monotonic_ns()
                if not isinstance(results, list) or len(results) != 1:
                    raise RuntimeError('Unexpected resident model output')
                forward_errors = ({'mode': 'throughput_no_duplicate_oracle'} if args.throughput
                                  else result_parity(results[0], oracle_results[0]))
                context, base, export_error = from_export(results[0], identity['token'], selector)
                candidates = context['candidate_trajectories'][0].cpu().tolist()
                request = online.choose(context, base, candidates)
                channel.send(request)
                action_sent_ns = time.monotonic_ns()
                main_receipt = channel.receive()
                main_received_ns = time.monotonic_ns()
                online.main_executed(main_receipt)
                branches = channel.receive()
                feedback_received_ns = time.monotonic_ns()
                ack, audit = online.update(branches)
                update_complete_ns = time.monotonic_ns()
                if any(p.requires_grad or p.grad is not None for p in model.parameters()):
                    raise RuntimeError('Frozen generator or input V3 received gradients')
                record = dict(kind='online_update', step=index-warmup_steps, **audit,
                    selected=request['selected'], candidate_hash=request['candidate_hash'],
                    observation_identity=identity, input_frame_indices=[f['frame_idx'] for f in window.frames],
                    file_input_parity=True, forward_errors=forward_errors,
                    frozen_v3_export_max_abs_error=export_error,
                    main_receipt=main_receipt, branch_feedback=branches,
                    next_observation_parity=(index > warmup_steps))
                report['events'].append(record)
                if not args.throughput:
                    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                record['times_ns']['ack_sent'] = time.monotonic_ns()
                channel.send(ack)
                response = channel.receive()
                online.acknowledged(response)
                record['worker_update_receipt'] = response
                worker_ack_received_ns = time.monotonic_ns()
                record['times_ns']['worker_ack_received'] = worker_ack_received_ns
                previous_update_time = record['times_ns']['update_complete']
                record['profile'] = dict(
                    decision_wall_seconds=(worker_ack_received_ns-step_started_ns)/1e9,
                    observation_copy_seconds=(observation_copied_ns-step_started_ns)/1e9,
                    input_prepare_seconds=(prepared_ns-observation_copied_ns)/1e9,
                    forward_seconds=(forwarded_ns-prepared_ns)/1e9,
                    action_to_main_seconds=(main_received_ns-action_sent_ns)/1e9,
                    branch_feedback_seconds=(feedback_received_ns-main_received_ns)/1e9,
                    learner_update_seconds=(update_complete_ns-feedback_received_ns)/1e9,
                    update_ack_seconds=(worker_ack_received_ns-update_complete_ns)/1e9)
                report['profile_events'].append(record['profile'])
                print(json.dumps(dict(decision=index-warmup_steps, selected=request['selected'],
                    policy_version=learner.version, optimized=audit['optimized'],
                    reward_std=audit['update']['reward_std'],
                    logit_change=audit['same_context_logit_change_after_update'])), flush=True)
            channel.send(dict(kind='close'))
            closed = channel.receive()
            if closed.get('kind') != 'closed' or closed.get('idm_fallbacks') != 0:
                raise RuntimeError('Worker close/fallback check failed')
            if not args.fixed_segment and child.wait(timeout=30) != 0:
                raise RuntimeError('Worker exited unsuccessfully')
            if args.fixed_segment and (closed.get('episode_closed') is not True or
                    closed.get('reward_groups') != args.steps or closed.get('candidate_simulation_steps') != args.steps*160):
                raise RuntimeError('Incomplete segment episode boundary')
        if state_digest(model.state_dict()) != initial_model_hash:
            raise RuntimeError('Frozen generator/V3 state changed')
        if source_hashes() != report['source_sha256']:
            raise RuntimeError('Source changed during acceptance run')
        if getattr(online, 'collected', None):
            raise RuntimeError('Cannot checkpoint an unconsumed rollout')
        evidence = learning_evidence(report['events'])
        if (evidence['actual_optimizer_steps'] != learner.version - initial_policy_version or
                learner.attempts - initial_attempts != args.steps or
                evidence['update_attempts'] != args.steps):
            raise RuntimeError('Optimizer/attempt accounting mismatch')
        checkpoint = output.with_suffix('.online.pt')
        with checkpoint.open('xb') as stream:
            torch.save(dict(kind=('FORMAL_ONLINE_TRAINING_CHECKPOINT' if formal_training else
                                 'DISPOSABLE_STRICT_ONLINE_PROBE_NOT_FORMAL_TRAINING'),
                formal_training=formal_training, online_update_timing=update_timing,
                reward_contract=report['worker']['reward_contract'],
                simulator_backend=cfg.get('fixed_backend') if args.fixed_segment else 'reference',
                candidate_simulation_steps=(initial_attempts+args.steps)*20*(8 if args.fixed_segment else 1),
                dynamics_contract=dynamics_contract, ego_controller=ego_controller,
                source_sha256=report['source_sha256'],
                parameterization=parameterization, initialization=learner.provenance(),
                learner=learner.state_dict()), stream)
        verified = evidence['closed_loop_learning_verified']
        if args.throughput:
            decision_times = [x['decision_wall_seconds'] for x in report['profile_events']]
            report['throughput'] = dict(
                total_decision_seconds=sum(decision_times),
                mean_decision_seconds=sum(decision_times)/len(decision_times),
                decisions_per_second=len(decision_times)/sum(decision_times),
                min_decision_seconds=min(decision_times),
                max_decision_seconds=max(decision_times),
                warmup_seconds=sum(e['elapsed_seconds'] for e in report['events']
                                   if e['kind'] == 'warmup'))
        report.update(evidence,
            status=(('PASS_STRICT_ONLINE_THROUGHPUT_PROBE' if verified else
                     'INCOMPLETE_ONLINE_THROUGHPUT_EVIDENCE') if args.throughput else
                    ('PASS_STRICT_ONLINE_SELECTOR_ONLY_PROBE' if verified else
                     'INCOMPLETE_ONLINE_LEARNING_EVIDENCE')),
            real_generator=True, live_render_verified=True, real_generated_reward=True,
            online_update=evidence['actual_optimizer_steps'] > 0, real_closed_loop=verified,
            cumulative_optimizer_steps=learner.version, cumulative_update_attempts=learner.attempts,
            update_attempts_initial=initial_attempts,
            learner_state_hash_initial=initial_learner_hash,
            learner_state_hash_final=state_digest(learner.state_dict()),
            worker_exit=dict(pid=child.pid, returncode=child.returncode, alive=child.poll() is None),
            frozen_generator_and_v3_unchanged=True, frozen_generator_sha256=initial_model_hash,
            frozen_generator_and_original_selector_unchanged=True, frozen_reference_unchanged=True,
            online_selector_trainable=True, online_selector_changed=initial_selector_hash != online.last_selector_hash,
            residual_changed=initial_selector_hash != online.last_selector_hash,
            initialization_provenance=learner.provenance(),
            policy_version=learner.version, step0_v3_parity=(True if initial_policy_version == 0 else None), next_step_waited_for_update=(update_timing == "per_step"),
            next_step_waited_for_feedback=True, updates_at_rollout_boundary=(update_timing == "after_rollout8"),
            generated_steps=args.steps, candidate_groups=args.steps, candidate_branches=args.steps*20,
            warmup_frames=warmup_steps, history_frames=4, next_observation_checks=args.steps-1,
            worker_episode_closed=bool(closed.get('episode_closed')) if args.fixed_segment else True,
            worker_process_resident=bool(args.fixed_segment),
            candidate_simulation_steps=args.steps*20*(8 if args.fixed_segment else 1),
            branch_render_calls=closed.get('branch_render_calls', 0),
            branch_model_calls=closed.get('branch_model_calls', 0),
            learner_checkpoint=dict(path=str(checkpoint), sha256=sha256_file(checkpoint),
                                    disposable=not formal_training,
                                    formal_training=formal_training),
            idm_fallbacks=closed['idm_fallbacks'], reaction='R',
            policy_version_initial=initial_policy_version,
            resident_reuse=resident_reuse)
        episode_complete = True
    except BaseException as error:
        if args.resident:
            _RESIDENT_MODEL = _RESIDENT_LEARNER = _RESIDENT_CONFIG = None
            _RESIDENT_PARAMETERIZATION = _RESIDENT_SOURCE_SHA256 = None
        report.update(status='FAIL_STRICT_ONLINE_SELECTOR_ONLY', error=repr(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, worker_log=str(worker_log))
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(status=report['status'], report=str(output))), flush=True)
        if live is not None: live.close()
        if fixed_client is not None and not (episode_complete and args.resident):
            close_fixed_runtime(abort=not episode_complete)
            fixed_client.close(abort=not episode_complete)
        if fixed_client is None and child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        for resource in (left, right):
            if resource is not None: resource.close()
        if images is not None and fixed_client is None: images.close()
    return 0 if report['status'] in ('PASS_STRICT_ONLINE_SELECTOR_ONLY_PROBE',
                                     'PASS_STRICT_ONLINE_THROUGHPUT_PROBE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
