"""V2 expert-reference scoring of independently executed reactive candidates."""
import logging
import time
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from .fixed_rollout import validate_action, validate_window, rollout
from .parallel_branches import _IDMFailures
from .reward_runtime import RewardSession, RouteGeometry
from .reward import ego_state, H1Reward
from .expert_segment_reward import ExpertWindow, ExpertSegmentReward, CONTRACT
from .state import structural_hash


class ExpertFixedRewardSession(RewardSession):
    def scoring_reference(self, start):
        ego = self.sim.engine.agent_manager.ego_agent
        return ExpertWindow.from_track(ego.object_track, start.step,
            float(ego.rear_vehicle.rear_axle_to_center_dist), self.sim.engine.current_scene['id'])

    def branch_route_lanes(self, roadblock_ids):
        if not roadblock_ids:
            raise ValueError('No checkpoint roadblocks in actual branch')
        key = tuple(roadblock_ids)
        if not hasattr(self, '_scoring_route_cache'):
            self._scoring_route_cache = {}
        if key not in self._scoring_route_cache:
            route = RouteGeometry(100)
            route._map_api = self.map_api
            route._load_route_dicts(list(key))
            if not route._route_lane_dict:
                raise ValueError('Cannot resolve actual branch route')
            self._scoring_route_cache[key] = route._route_lane_dict
        return self._scoring_route_cache[key]

    def group(self, actions, selected, pool=None, on_main=None, compare_h1=False, audit='full'):
        if len(actions) != 20 or type(selected) is not int or not 0 <= selected < 20:
            raise ValueError('Expected twenty candidates and integral selected index')
        if audit not in ('full', 'boundary'):
            raise ValueError('Unknown audit mode')
        sim = self.sim
        validate_window(sim)
        for action in actions:
            validate_action(action)
        history, history_hash = self.history.snapshot(), self.history.state_hash
        if compare_h1 and len(history) < 14:
            raise ValueError('H1 comparison requires actual history')
        start = self.capture()
        if start.state_hash != history[-1].state_hash:
            raise ValueError('Reward history does not match the current root')
        bank_hash = structural_hash(actions)
        started = time.monotonic()
        reference = self.scoring_reference(start)
        drivable = PDMDrivableMap.from_simulation(self.map_api, ego_state(start), 100)
        adapter = ExpertSegmentReward(reference, start, drivable, self.map_api)
        reference_end = time.monotonic()
        sim.codec.audit_static()
        before = sim.snapshot()
        failures = _IDMFailures()
        logging.getLogger().addHandler(failures)
        main = None
        try:
            states = sim.step(actions[selected])
            main, main_frame = sim.snapshot(), self.capture()
            if on_main is not None:
                on_main(main, states)
            branches_started = time.monotonic()
            branches = pool.run(before, actions, audit=audit) if pool else [
                rollout(sim, before, action, self.offset, self.actor_types,
                        audit=audit, audit_static=audit=='full') for action in actions]
            branch_end = time.monotonic()
            rewards = [adapter.score(branch['frames'], self.branch_route_lanes(branch['route_roadblock_ids']))
                       for branch in branches]
            h1 = [H1Reward(self.centerline, self.route_lanes, drivable, self.map_api).score(history, b['frames'][1])
                  for b in branches] if compare_h1 else None
            if (branches[selected]['first_hash'] != main.state_hash or
                    branches[selected]['frames'][1].state_hash != main_frame.state_hash):
                raise RuntimeError('Selected branch first step differs from canonical execution')
            if failures.count:
                raise RuntimeError('IDM fallback invalidates the candidate group')
            if self.history.state_hash != history_hash or structural_hash(actions) != bank_hash:
                raise RuntimeError('Scoring mutated history or candidate bank')
        finally:
            logging.getLogger().removeHandler(failures)
            sim.restore(main if main is not None else before)
            sim.codec.audit_static()
        self.history.append(main_frame)
        return dict(contract=CONTRACT['name'], backend=sim.backend, audit=audit,
            before_hash=before.state_hash, main_hash=main.state_hash, selected=selected,
            selected_first_state_parity=True, canonical_restored=True,
            history_before_hash=history_hash, history_after_hash=self.history.state_hash,
            first_hashes=[b['first_hash'] for b in branches], end_hashes=[b['end_hash'] for b in branches],
            rewards=rewards, h1_rewards=h1, reference=adapter.receipt(rewards[0]['reference_progress_m']),
            traces=branches, branch_render_calls=0, branch_model_calls=0,
            profile=dict(reference_seconds=reference_end-started,
                branches_seconds=branch_end-branches_started, score_seconds=time.monotonic()-branch_end,
                total_seconds=time.monotonic()-started,
                worker_restore_seconds=[b['restore_seconds'] for b in branches],
                worker_rollout_seconds=[b['rollout_seconds'] for b in branches],
                pool_profile=pool.last_profile if pool else None))
