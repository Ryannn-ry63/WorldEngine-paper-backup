"""Resident visual main environment and exact CPU branches for online training."""
import argparse
import gc
import json
import logging
import os
import pickle
from pathlib import Path
import socket
import time
import traceback
import numpy as np
from .paths import sha256_file
from .scene_payload import single_scene_mapping
from .visual_assets import visual_asset
from .transport import Channel, digest
from .image_transport import ImageBuffer, metadata_to_wire
from .fixed_learning import FixedStepGate
from .fixed_protocol import configured_contract, EXPERT_CONTRACT_NAME
from worldengine.online.headless import HeadlessSimulator
from worldengine.online.observations import CanonicalObserver
from worldengine.online.actions import to_world_actions
from worldengine.online.fixed_rollout import FixedRewardSession
from worldengine.online.fixed_parallel import FixedBranchPool
from worldengine.online.fixed_diagnostics import compact_result, write_json, frame_record
from worldengine.online.segment_reward import CONTRACT
from worldengine.online.parallel_branches import _IDMFailures
from worldengine.online.state import structural_hash


def episode(request, channel, images, pool, workers, backend):
    cfg = request['settings']
    name = configured_contract(cfg)
    session_class, reward_contract = FixedRewardSession, CONTRACT
    if name == EXPERT_CONTRACT_NAME:
        from worldengine.online.expert_fixed_rollout import ExpertFixedRewardSession
        from worldengine.online.expert_segment_reward import CONTRACT as EXPERT_CONTRACT
        session_class, reward_contract = ExpertFixedRewardSession, EXPERT_CONTRACT
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=False)
    if request['mode'] not in ('train', 'frozen', 'diagnostic'):
        raise ValueError('Unknown fixed worker mode')
    training = request['mode']=='train'
    warmup = 13 if request['mode']=='diagnostic' else 3
    steps = request['steps']
    if type(steps) is not int or not 1 <= steps <= (8 if training else 16):
        raise ValueError('Expected 1..8 main decisions')
    if request['audit_every'] < 1:
        raise ValueError('Positive full-audit interval required')
    source = Path(cfg['visual_scene_path'])
    source_sha = sha256_file(source)
    if source_sha != cfg['visual_scene_sha256']:
        raise ValueError('Exact scene checksum mismatch')
    with source.open('rb') as stream:
        scenes = single_scene_mapping(pickle.load(stream), cfg['visual_scene_id'])
    scene_id, asset_root, asset = visual_asset(cfg, scenes)
    scene = scenes[scene_id]
    score_steps = cfg.get('probe_score_steps', list(range(steps))) if not training else list(range(steps))
    if not score_steps or any(type(x) is not int or not 0 <= x < steps for x in score_steps):
        raise ValueError('Invalid diagnostic scoring roots')
    reaction = cfg.get('probe_reaction', 'R') if not training else 'R'
    continuation_root = cfg.get('probe_continuation_root') if not training else None
    executed_frames, executed_route_ids = [], []
    gate = FixedStepGate(request['policy_version'], name, cfg.get('online_update_timing', 'per_step')) if training else None
    sim = observer = None
    failures = _IDMFailures()
    logging.getLogger().addHandler(failures)
    started = time.monotonic()
    try:
        max_steps = max(warmup + steps, warmup + max(score_steps) + 8)
        # Root snapshot hashes include the global configuration. A diagnostic
        # prefix and its shorter continuation must therefore use the SAME
        # horizon limit, even when fewer states actually need to be rendered.
        if request['mode'] == 'frozen':
            max_steps = max(18, max_steps)
        sim = HeadlessSimulator(scene_id, scene, reaction, max_steps, cfg['online_scene_seed'], backend=backend)
        observer = CanonicalObserver(sim, asset_root)
        session = session_class(sim, cfg['map_root'])
        reused = pool is not None
        if workers:
            if pool is None:
                pool = FixedBranchPool(scene_id, scene, reaction, max_steps, cfg['online_scene_seed'],
                    sim.codec.bundle, session.offset, session.actor_types, workers=workers,
                    timeout=900, global_config=sim.engine.global_config, backend=backend)
            else:
                pool.bind_scene(scene_id, sim.codec.bundle, session.offset,
                                session.actor_types, sim.engine.global_config)
        ready = dict(kind='ready', scene=scene_id, pid=os.getpid(), reward_adapter=True,
            online_updates=training, reward_contract=reward_contract, reaction=reaction,
            dynamics_contract=sim.dynamics_contract, ego_controller=sim.engine.global_config.ego_controller,
            warmup_transitions=warmup, steps=steps, backend=backend,
            source=dict(path=str(source), sha256=source_sha),
            asset=dict(path=str(asset), sha256=sha256_file(asset)),
            branch_workers=workers, branch_pool_reused=reused,
            branch_worker_receipts=pool.worker_receipts if pool else [],
            startup_seconds=time.monotonic()-started)
        channel.send(ready)
        groups = renders = 0
        for index in range(warmup+steps):
            frame, cameras, state_hash = observer.observe()
            renders += 1
            identity = dict(scene=scene_id, step=index, state_hash=state_hash, token=frame['token'],
                            policy_version=gate.gate.policy_version if gate else request['policy_version'])
            if gate is not None and index >= warmup:
                gate.observe(identity)
            channel.send(dict(kind='observation', identity=identity, frame=metadata_to_wire(frame),
                              images=images.publish(cameras, identity)))
            action = channel.receive()
            if action.get('identity') != identity:
                raise ValueError('Stale action identity')
            images.release(identity)
            if index < warmup:
                if action.get('kind') != 'warmup':
                    raise ValueError('Expected history initialization')
                session.warmup(session.warmup_action())
                continue
            if action.get('kind') != 'action' or digest(action['candidates']) != action['candidate_hash']:
                raise ValueError('Invalid candidate bank')
            if gate is not None:
                gate.choose(action)
            selected = action['selected']
            if type(selected) is not int or not 0 <= selected < 20:
                raise ValueError('Invalid selected candidate')
            if sim.snapshot().state_hash != state_hash:
                raise RuntimeError('Main root changed before execution')
            actions = to_world_actions(np.asarray(action['candidates'], dtype=np.float32), sim.engine.agent_manager.ego_agent)
            actual_start = session.capture()
            if continuation_root is not None and index-warmup >= continuation_root:
                if not executed_frames:
                    executed_frames.append(actual_start)
                for lane in sim.engine.agent_manager.ego_agent.navigation.checkpoint_lanes:
                    if lane.roadblock_id and lane.roadblock_id not in executed_route_ids:
                        executed_route_ids.append(lane.roadblock_id)
            world_action = actions[selected]
            executed_plan = dict(waypoints=world_action.waypoints.tolist(),
                headings=world_action.headings.tolist(), rear_pose=actual_start.ego[:3].tolist(),
                global_offset=session.offset.tolist())
            def canonical(snapshot, states):
                main = dict(kind='main', identity=identity, next_hash=snapshot.state_hash,
                            selected=selected, candidate_hash=action['candidate_hash'])
                if gate is not None:
                    gate.main_executed(main)
                channel.send(main)
            audit = ('full' if (request['attempts']+groups)%request['audit_every']==0 else 'boundary')
            history_hash = structural_hash((list(observer.history.raw), list(observer.history.frames)))
            scored = index-warmup in score_steps
            if scored:
                result = session.group(actions, selected, pool, canonical,
                    compare_h1=request['mode']=='diagnostic', audit=audit)
                record = compact_result(result)
                if cfg.get('probe_scorer_parity') and not training:
                    from worldengine.online.scorer_parity import compare
                    record['scorer_parity'] = [compare(session, b['frames'], b['route_roadblock_ids'])
                                               for b in result['traces']]
                groups += 1
            else:
                before_history = session.history.state_hash
                states = sim.step(actions[selected])
                canonical(sim.snapshot(), states)
                session.history.append(session.capture())
                record = dict(contract=name, main_hash=sim.snapshot().state_hash,
                    history_before_hash=before_history, history_after_hash=session.history.state_hash,
                    reference={}, candidate_simulation_steps=0)
            if continuation_root is not None and index-warmup >= continuation_root:
                executed_frames.append(session.capture())
                for lane in sim.engine.agent_manager.ego_agent.navigation.checkpoint_lanes:
                    if lane.roadblock_id and lane.roadblock_id not in executed_route_ids:
                        executed_route_ids.append(lane.roadblock_id)
            if structural_hash((list(observer.history.raw), list(observer.history.frames))) != history_hash:
                raise RuntimeError('Candidate branches modified visual history')
            record['executed_plan'] = executed_plan
            record.update(identity=identity, candidates=action['candidates'], candidate_hash=action['candidate_hash'],
                          candidate_source='live_DiffusionDrive', policy_version=identity['policy_version'])
            trace_path = output/('root_%02d.json'%index)
            write_json(trace_path, record)
            feedback = {k:v for k,v in record.items() if k not in ('traces','reference','candidates')}
            feedback['reference'] = {k:v for k,v in record['reference'].items() if k!='frames'}
            feedback.update(kind='feedback', scored=scored, selected=selected, trace_path=str(trace_path),
                            trace_sha256=sha256_file(trace_path))
            if gate is not None:
                gate.feedback(feedback)
            if failures.count:
                raise RuntimeError('IDM fallback invalidates the episode')
            channel.send(feedback)
            ack = channel.receive()
            if gate is not None:
                channel.send(gate.updated(ack))
            elif ack != dict(kind='ack', identity=identity, next_hash=feedback['main_hash']):
                raise ValueError('Frozen episode acknowledgement mismatch')
        if channel.receive() != dict(kind='close'):
            raise ValueError('Expected episode close')
        continuation = None
        if continuation_root is not None:
            if len(executed_frames) != 9:
                raise ValueError('Continuation must contain exactly eight executed transitions')
            from worldengine.online.scorer_parity import compare
            continuation = compare(session, executed_frames, executed_route_ids)
            write_json(output/'continuation.json', dict(continuation,
                frames=[frame_record(f) for f in executed_frames],route_roadblock_ids=executed_route_ids))
        # Release all renderer state before announcing a reusable episode boundary.
        observer.close(); observer = None
        sim.close(); sim = None
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        channel.send(dict(kind='closed', episode_closed=True, worker_resident=True,
                          render_calls=renders, reward_groups=groups, continuation=continuation, idm_fallbacks=failures.count,
                          branch_render_calls=0, branch_model_calls=0,
                          candidate_simulation_steps=groups*20*8,
                          elapsed_seconds=time.monotonic()-started))
        return pool
    except BaseException:
        if pool is not None:
            pool.close(abort=True)
        raise
    finally:
        if observer is not None:
            observer.close()
        if sim is not None:
            sim.close()
        logging.getLogger().removeHandler(failures)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--images-fd', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--backend', choices=('reference','cached'), default='cached')
    args = parser.parse_args()
    channel = Channel(socket.socket(fileno=args.fd), timeout=1800)
    images = ImageBuffer(args.images_fd)
    pool = None
    try:
        channel.send(dict(kind='resident_ready', pid=os.getpid(), backend=args.backend))
        while True:
            request = channel.receive()
            if request == dict(kind='shutdown'):
                if pool is not None:
                    pool.close(); pool = None
                channel.send(dict(kind='shutdown_complete'))
                break
            if request.get('kind') != 'episode':
                raise ValueError('Expected an episode assignment')
            pool = episode(request, channel, images, pool, args.workers, args.backend)
    except BaseException as error:
        try:
            channel.send(dict(kind='error', error=repr(error), traceback=traceback.format_exc()))
        except (OSError, EOFError):
            pass
        raise
    finally:
        if pool is not None:
            pool.close(abort=True)
        images.close(); channel.sock.close()


if __name__=='__main__':
    main()
