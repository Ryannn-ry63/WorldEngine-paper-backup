"""Actual reactive execution scored with the repository's expert-reference PDM.

ExpertWindow is detached scoring data. It must never enter a policy observation,
candidate generator, action controller or learner context.
"""
from dataclasses import dataclass
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from worldengine.common.dataclasses import NavsimTrajectory
from worldengine.utils import math_utils
from worldengine.components.agents.policy.pdm_planner.abstract_pdm_closed_planner import (
    transform_trajectory, get_trajectory_as_array)
from worldengine.components.agents.policy.pdm_planner.scoring.pdm_scorer import (
    PDMScorer, PROGRESS_DISTANCE_THRESHOLD)
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex as M, WeightedMetricIndex as W, EgoAreaIndex as A)
from .reward import ego_state, RewardFrame
from .segment_reward import actual_observation
from .state import structural_hash

CONTRACT = dict(name='fixed_candidate_4s_logplay_pdm_expert_v2',
    interval_seconds=.5, future_steps=8, samples=9, ego_controller='log_play_controller',
    nc_dac='whole_actual_segment_including_endpoint',
    ttc='repository_pdm_1s_projection_against_branch_actual_actors',
    comfort='actual_velocity_acceleration_9_samples',
    progress_reference='logged_ego_window_reanchored_as_metric_manager',
    progress_normalization='pair_max_raw_progress_threshold_0.1m_else_one',
    formula='repository_PDMScorer_aggregate; G*(5*G*EP+5*TTC+2*C)/12; G=NC*DAC',
    route='branch_checkpoint_roadblock_union', map_radius_m=100.,
    reference_rollout=False, expert_ego_future='scoring_only',
    official_full_closed_loop_pdms=False,
    evaluator_difference='training_keeps_actual_acceleration; legacy_evaluator_zero_fills')


@dataclass(frozen=True)
class ExpertWindow:
    root_step: int
    poses: np.ndarray  # relative rear-axle XY; ABSOLUTE heading, as MetricManager
    source_scene: str
    source_hash: str

    @classmethod
    def from_track(cls, track, root_step, rear_axle_to_center, scene_id):
        if type(root_step) is not int or root_step < 0:
            raise ValueError('Invalid expert root step')
        end = root_step + 9
        xy = np.asarray(track['position'], dtype=np.float64)[root_step:end, :2].copy()
        heading = np.asarray(track['heading'], dtype=np.float64)[root_step:end].copy()
        if xy.shape != (9, 2) or heading.shape != (9,) or not np.isfinite(xy).all() or not np.isfinite(heading).all():
            raise ValueError('Expert reference requires nine finite real samples; no padding')
        if 'valid' in track and not np.asarray(track['valid'])[root_step:end].all():
            raise ValueError('Invalid logged ego sample in expert window')
        if not np.isfinite(rear_axle_to_center) or rear_axle_to_center <= 0:
            raise ValueError('Invalid vehicle rear-axle offset')
        source_hash = structural_hash((scene_id, root_step, xy, heading, rear_axle_to_center))
        # Deliberately use the exact conversion/translation used by MetricManager.
        for i in range(9):
            xy[i] = math_utils.translate_longitudinally(xy[i], heading[i], -rear_axle_to_center).reshape(2)
        xy -= xy[0].copy()
        poses = np.column_stack([xy, heading])
        poses.setflags(write=False)
        return cls(root_step, poses, str(scene_id), source_hash)

    def materialize(self, start):
        if start.step != self.root_step or self.poses.shape != (9, 3) or not np.isfinite(self.poses).all():
            raise ValueError('Expert reference/root mismatch')
        initial = ego_state(start)
        trajectory = NavsimTrajectory(poses=self.poses.copy(),
            trajectory_sampling=TrajectorySampling(num_poses=9, interval_length=.5))
        transformed = transform_trajectory(trajectory, initial)
        states = get_trajectory_as_array(transformed, trajectory.trajectory_sampling, initial.time_point)
        if states.shape != (9, 11) or not np.isfinite(states).all():
            raise ValueError('Invalid transformed expert reference')
        line = PDMPath([state.rear_axle for state in transformed._trajectory])
        # Zero-length expert paths are valid; the original scorer handles them.
        return states, line


class ExpertSegmentReward:
    def __init__(self, reference, start, drivable_map, map_api):
        if drivable_map is None or len(drivable_map) == 0:
            raise ValueError('Expert PDM scoring requires drivable polygons')
        self.reference, self.start = reference, start
        self.expert_states, self.centerline = reference.materialize(start)
        self.drivable_map, self.map_api = drivable_map, map_api

    def score(self, frames, route_lanes):
        if len(frames) != 9 or frames[0].state_hash != self.start.state_hash or any(
                b.step != a.step + 1 for a, b in zip(frames, frames[1:])):
            raise ValueError('Scoring requires nine consecutive actual frames at the reference root')
        if not route_lanes:
            raise ValueError('Missing actual branch route lanes')
        for frame in frames:
            if frame.ego.shape != (11,) or not np.isfinite(frame.ego).all():
                raise ValueError('Invalid actual ego state')
            for actor in frame.actors:
                box = actor.box
                if (not box.geometry.is_valid or box.geometry.is_empty or
                    not np.isfinite([*box.center, box.length, box.width, box.height]).all()):
                    raise ValueError('Invalid actor geometry')
                if hasattr(actor, 'velocity') and not np.isfinite([actor.velocity.x, actor.velocity.y]).all():
                    raise ValueError('Invalid actor velocity')
        scorer = PDMScorer(TrajectorySampling(num_poses=8, interval_length=.5))
        states = np.stack([self.expert_states, np.stack([f.ego for f in frames])])
        scores = scorer.score_proposals(states, ego_state(frames[0]),
            actual_observation(frames, route_lanes, map_radius=100), self.centerline,
            route_lanes, self.drivable_map, self.map_api)
        nc, dac = (float(scorer._multi_metrics[k, 1]) for k in (M.NO_COLLISION, M.DRIVABLE_AREA))
        ttc, comfort, direction = (float(scorer._weighted_metrics[k, 1]) for k in (W.TTC, W.COMFORTABLE, W.DRIVING_DIRECTION))
        reference, progress = map(float, scorer._progress_raw)
        denominator = max(reference, progress)
        # Raw EP is diagnostic; the returned reward and gated EP come from scorer.
        ep = progress / denominator if denominator > PROGRESS_DISTANCE_THRESHOLD else 1.
        result = dict(reward=float(scores[1]), NC=nc, DAC=dac, EP=ep,
            gated_EP=float(scorer._weighted_metrics[W.PROGRESS, 1]), TTC=ttc,
            comfort=comfort, direction=direction, progress_m=progress, reference_progress_m=reference)
        if not np.isfinite(list(result.values())).all() or not 0 <= result['reward'] <= 1:
            raise ValueError('Invalid expert PDM reward')
        offroad = np.flatnonzero(scorer._ego_areas[1, :, A.NON_DRIVABLE_AREA])
        result['first_violation_seconds'] = dict(
            NC=float(scorer._collision_time_idcs[1]*.5) if np.isfinite(scorer._collision_time_idcs[1]) else None,
            DAC=float(offroad[0]*.5) if len(offroad) else None,
            TTC=float(scorer._ttc_time_idcs[1]*.5) if np.isfinite(scorer._ttc_time_idcs[1]) else None)
        return result

    def receipt(self, progress_m):
        return dict(kind='logged_ego_scoring_reference', root_step=self.reference.root_step,
            end_step=self.reference.root_step+8, source_scene=self.reference.source_scene,
            source_hash=self.reference.source_hash, states_hash=structural_hash(self.expert_states),
            progress_m=progress_m, simulated=False,
            frames=tuple(RewardFrame(self.reference.root_step+i, state.copy(), ())
                         for i, state in enumerate(self.expert_states)))
