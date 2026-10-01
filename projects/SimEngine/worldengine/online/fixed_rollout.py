"""Fixed absolute candidate rollout; canonical history commits one step only."""
import time
import logging
from .parallel_branches import _IDMFailures
import numpy as np
from shapely.geometry import Point
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from worldengine.common.dataclasses import Trajectory
from worldengine.components.agents.policy.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from worldengine.components.agents.policy.pdm_planner.proposal.pdm_proposal import PDMProposalManager
from worldengine.components.agents.policy.pdm_planner.proposal.pdm_generator import PDMGenerator
from worldengine.components.agents.policy.pdm_planner.observation.pdm_observation import PDMObservation
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from .reward import capture_reward_frame, ego_state, H1Reward
from .reward_runtime import RewardSession
from .segment_reward import SegmentReward, CONTRACT
from .state import structural_hash

REFERENCE_PARAMETERS = dict(fallback_target_velocity=30., speed_limit_fraction=1.,
    min_gap_to_lead_agent=10., headway_time=1.5, accel_max=4.5, decel_max=3.)


def validate_action(action):
    if not isinstance(action, Trajectory):
        raise TypeError('Expected centered world-coordinate Trajectory')
    for name, shape in (('waypoints',(9,2)),('headings',(9,)),('velocities',(9,2)),('angular_velocities',(9,))):
        value = np.asarray(getattr(action,name), dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Fixed candidate requires finite 9-point ' + name)


def suffix(action, index):
    return Trajectory(**{name:np.asarray(getattr(action,name))[index:].copy()
        for name in ('waypoints','velocities','headings','angular_velocities')})


def validate_window(sim):
    end = int(sim.engine.episode_step)+8
    if sim.engine.global_config.ego_controller != 'log_play_controller':
        raise ValueError('Fixed candidate contract requires log_play_controller')
    if end > sim.engine.global_config.max_step or end >= sim.engine.current_scene['log_length']:
        raise ValueError('Insufficient real future window for all 8 candidate steps')


def rollout(sim, before, action, offset, actor_types, audit='full', audit_static=True):
    """One restore, eight transitions, compact detached actual frames; no render."""
    validate_action(action)
    if audit not in ('full', 'boundary'):
        raise ValueError('Unknown rollout audit mode')
    started = time.monotonic()
    sim.restore(before)
    restored = time.monotonic()
    validate_window(sim)
    frames = [capture_reward_frame(sim.engine, offset, actor_types)]
    route_ids = []
    def capture_route():
        # Same ordered checkpoint-roadblock union as MetricManager.step().
        for lane in sim.engine.agent_manager.ego_agent.navigation.checkpoint_lanes:
            roadblock_id = lane.roadblock_id
            if roadblock_id and roadblock_id not in route_ids:
                route_ids.append(roadblock_id)
    capture_route()
    hashes = []
    for i in range(8):
        sim.step(suffix(action,i), allow_two_point_logplay=True)
        snapshot = sim.snapshot() if audit == 'full' or i in (0, 7) else None
        frame = capture_reward_frame(sim.engine, offset, actor_types)
        if audit == 'full' and sim.snapshot().state_hash != snapshot.state_hash:
            raise RuntimeError('Branch frame capture mutated dynamics')
        frames.append(frame)
        capture_route()
        hashes.append(snapshot.state_hash if snapshot is not None else None)
    if audit_static:
        sim.codec.audit_static()
    return dict(before_hash=before.state_hash, first_hash=hashes[0], end_hash=hashes[-1],
                route_roadblock_ids=route_ids,
                state_hashes=hashes, frames=tuple(frames), audit=audit,
                restore_seconds=restored-started, rollout_seconds=time.monotonic()-restored)


def reference_action(session, frame):
    """PDM lead search/IDM on current live actors; no PDMPolicy future reader."""
    sampling = TrajectorySampling(num_poses=2, interval_length=.5)
    initial = ego_state(frame)
    observation = PDMObservation(sampling,sampling,200,observation_sample_res=1)
    observation.update(initial, DetectionsTracks(TrackedObjects(list(frame.actors))), [], session.route_lanes)
    manager = PDMProposalManager([session.centerline], BatchIDMPolicy(**REFERENCE_PARAMETERS))
    point = Point(*initial.center.array[:2])
    lanes = list(session.route_lanes.values())
    lanes.sort(key=lambda lane:lane.polygon.distance(point))
    speed_limit = lanes[0].speed_limit_mps
    if speed_limit is not None and (not np.isfinite(speed_limit) or speed_limit <= 0):
        raise ValueError('Invalid map speed limit')
    manager.update(speed_limit)
    generator = PDMGenerator(sampling,sampling,leading_agent_update_rate=1)
    states = generator.generate_proposals(initial,observation,manager)[0]
    if not np.isfinite(states).all() or not np.isfinite(generator._state_idm_array).all():
        raise ValueError('Non-finite map/IDM reference')
    poses = states[:,:3].copy()
    poses[0] = frame.ego[:3]
    speed = np.maximum(generator._state_idm_array[0,:,1],0.)
    unit = np.column_stack([np.cos(poses[:,2]),np.sin(poses[:,2])])
    points = poses[:,:2] - session.offset + initial.car_footprint.vehicle_parameters.rear_axle_to_center*unit
    angular = np.r_[0.,np.diff(np.unwrap(poses[:,2]))/.5]
    return Trajectory(points, speed[:,None]*unit, poses[:,2], angular)


def reference_rollout(session, before, adapter):
    sim = session.sim
    sim.restore(before)
    validate_window(sim)
    frames = [session.capture()]
    safe = frames[0]
    failure_step = None
    if min(adapter.safety(safe)) < 1:
        failure_step = 0
    else:
        for step in range(1,9):
            sim.step(reference_action(session,frames[-1]))
            frame = session.capture()
            frames.append(frame)
            if min(adapter.safety(frame)) < 1:
                failure_step = step
                break
            safe = frame
    progress = adapter.progress(frames[0],safe)
    if not np.isfinite(progress):
        raise ValueError('Invalid reference progress')
    return dict(progress_m=progress, failure_step=failure_step, frames=tuple(frames),
                parameters=REFERENCE_PARAMETERS, end_hash=sim.snapshot().state_hash)


class FixedRewardSession(RewardSession):
    def group(self, actions, selected, pool=None, on_main=None, compare_h1=False, audit='full'):
        if len(actions) != 20 or type(selected) is not int or not 0 <= selected < 20:
            raise ValueError('Expected 20 candidates and integral selected index')
        if audit not in ('full', 'boundary'):
            raise ValueError('Unknown group audit mode')
        sim = self.sim
        validate_window(sim)
        for action in actions:
            validate_action(action)
        history = self.history.snapshot()
        history_hash = self.history.state_hash
        if compare_h1 and len(history) < 14:
            raise ValueError('H1 comparison requires actual trailing history; padding forbidden')
        start = self.capture()
        if start.state_hash != history[-1].state_hash:
            raise ValueError('Reward history is not at the actual root state')
        bank_hash = structural_hash(actions)
        radius = 100.+max(float(np.linalg.norm(a.waypoints-a.waypoints[0],axis=1).max()) for a in actions)
        drivable = PDMDrivableMap.from_simulation(self.map_api, ego_state(start), radius)
        adapter = SegmentReward(self.centerline,self.route_lanes,drivable,self.map_api)
        sim.codec.audit_static()
        before = sim.snapshot()
        failures = _IDMFailures()
        logging.getLogger().addHandler(failures)
        main = None
        started = time.monotonic()
        try:
            main_states = sim.step(actions[selected])
            main = sim.snapshot()
            main_frame = self.capture()
            if on_main is not None:
                on_main(main,main_states)
            reference = reference_rollout(self,before,adapter)
            reference_end = time.monotonic()
            sim.restore(main)
            branches = (pool.run(before,actions,audit=audit) if pool else
                [rollout(sim,before,a,self.offset,self.actor_types,audit=audit,
                         audit_static=audit=='full') for a in actions])
            branch_end = time.monotonic()
            rewards = [adapter.score(b['frames'],reference['progress_m']) for b in branches]
            h1 = ([H1Reward(self.centerline,self.route_lanes,drivable,self.map_api).score(history,b['frames'][1])
                   for b in branches] if compare_h1 else None)
            if branches[selected]['first_hash'] != main.state_hash or branches[selected]['frames'][1].state_hash != main_frame.state_hash:
                raise RuntimeError('Selected candidate first step differs from canonical execution')
            if failures.count:
                raise RuntimeError('IDM fallback invalidates fixed segment group')
            if self.history.state_hash != history_hash or structural_hash(actions) != bank_hash:
                raise RuntimeError('Scoring mutated canonical history or candidate bank')
        finally:
            logging.getLogger().removeHandler(failures)
            sim.restore(main if main is not None else before)
            sim.codec.audit_static()
        self.history.append(main_frame)
        return dict(contract=CONTRACT['name'], backend=sim.backend, audit=audit,
            before_hash=before.state_hash, main_hash=main.state_hash,
            selected=selected, selected_first_state_parity=True, canonical_restored=True,
            history_before_hash=history_hash, history_after_hash=self.history.state_hash,
            first_hashes=[b['first_hash'] for b in branches], end_hashes=[b['end_hash'] for b in branches],
            rewards=rewards, h1_rewards=h1, reference=reference, traces=branches,
            profile=dict(reference_seconds=reference_end-started, branches_seconds=branch_end-reference_end,
                         score_seconds=time.monotonic()-branch_end, total_seconds=time.monotonic()-started,
                         worker_restore_seconds=[b['restore_seconds'] for b in branches],
                         worker_rollout_seconds=[b['rollout_seconds'] for b in branches],
                         pool_profile=pool.last_profile if pool else None),
            branch_render_calls=0, branch_model_calls=0)
