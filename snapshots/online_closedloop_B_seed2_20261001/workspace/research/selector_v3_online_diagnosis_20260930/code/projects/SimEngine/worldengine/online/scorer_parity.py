"""Execute the real legacy MetricManager scorer on the identical actual trace.

The aligned path supplies measured dynamics; the legacy path retains its zero
acceleration/angular columns. These are diagnostics, never training rewards.
"""
from dataclasses import asdict
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from worldengine.manager.metric_manager import MetricManager
from worldengine.common.dataclasses import NavsimTrajectory
from worldengine.components.agents.policy.pdm_planner.scoring.pdm_scorer import PDMScorer
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from .reward_runtime import RouteGeometry
from .reward import ego_state
from .expert_segment_reward import ExpertSegmentReward

MAPPING=dict(reward='score',NC='no_at_fault_collisions',DAC='drivable_area_compliance',
             gated_EP='ego_progress',TTC='time_to_collision_within_bound',comfort='comfort',
             direction='driving_direction_compliance')


def compare(session, frames, route_ids, tolerance=1e-6):
    frames=tuple(frames)
    reference=session.scoring_reference(frames[0])
    lanes=session.branch_route_lanes(route_ids)
    drivable=PDMDrivableMap.from_simulation(session.map_api,ego_state(frames[0]),100)
    training=ExpertSegmentReward(reference,frames[0],drivable,session.map_api).score(frames,lanes)
    def evaluate(aligned):
        m=MetricManager.__new__(MetricManager)
        m._proposal_sampling=TrajectorySampling(num_poses=8,interval_length=.5)
        m._map_radius=100; m._map_api=session.map_api
        m.num_history=1; m.buffer_size=9; m.first_NC_DAC_step=None
        m._cached_roadblock_ids=list(route_ids)
        m.ego_states_list=[ego_state(f) for f in frames]
        m.detection_tracks_list=[DetectionsTracks(TrackedObjects(list(f.actors))) for f in frames]
        m.traffic_light_data_list=[[] for _ in frames]
        m._pdm_closed=RouteGeometry(100)
        m._pdm_closed._map_api=session.map_api
        m._pdm_closed._scorer=PDMScorer(m._proposal_sampling)
        if aligned:
            m.get_pred_states=lambda: np.stack([f.ego for f in frames]).copy()
        expert=NavsimTrajectory(poses=reference.poses.copy(),
            trajectory_sampling=TrajectorySampling(num_poses=9,interval_length=.5))
        return asdict(m.compute_pdm_scores(expert,np.stack([f.ego[3:5] for f in frames])))
    aligned=evaluate(True); legacy=evaluate(False)
    errors={k:abs(training[k]-aligned[v]) for k,v in MAPPING.items()}
    if max(errors.values())>tolerance:
        raise RuntimeError('Training vs aligned legacy scorer mismatch: '+repr(errors))
    if any(abs(training[k]-legacy[MAPPING[k]])>tolerance for k in ('NC','DAC','gated_EP','TTC','direction')):
        raise RuntimeError('Legacy dynamics difference changed a non-comfort component')
    expected_delta=training['NC']*training['DAC']*2/12*(training['comfort']-legacy['comfort'])
    if abs(training['reward']-legacy['score']-expected_delta)>tolerance:
        raise RuntimeError('Legacy score difference is not explained by comfort')
    return dict(training=training,legacy=legacy,aligned_legacy=aligned,max_abs_error=max(errors.values()),
                expected_legacy_comfort_score_delta=expected_delta,atol=tolerance,
                status='PASS_SAME_TRACE_PARITY')
