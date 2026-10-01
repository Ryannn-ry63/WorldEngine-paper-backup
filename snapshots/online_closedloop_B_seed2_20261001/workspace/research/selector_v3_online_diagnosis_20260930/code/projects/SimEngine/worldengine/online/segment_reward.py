"""Four-second actual-state PDM components with a map/IDM reference.

No scene or logged ego future enters this module. This is a training proxy,
not the expert-normalized official closed-loop PDMS evaluator.
"""
import numpy as np
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from worldengine.components.agents.policy.pdm_planner.observation.pdm_observation import PDMObservation
from worldengine.components.agents.policy.pdm_planner.utils.pdm_enums import MultiMetricIndex as M, WeightedMetricIndex as W, EgoAreaIndex as A
from .reward import H1Reward, CurrentObservation

CONTRACT = dict(name='fixed_candidate_4s_logplay_pdm_v1', interval_seconds=.5,
    future_steps=8, samples=9, ego_controller='log_play_controller',
    nc_dac='whole_actual_segment_including_endpoint',
    ttc='repository_pdm_1s_projection_against_branch_actual_actors',
    comfort='forward_actual_9_samples', progress_reference='shared_map_idm_4s',
    formula='G*(5*G*clip(d/max(d_ref,0.1),0,1)+5*TTC+2*C)/12; G=NC*DAC',
    official_full_closed_loop_pdms=False, expert_ego_future=False)


def actual_observation(frames, route_lanes, map_radius=0):
    sampling = TrajectorySampling(num_poses=len(frames)-1, interval_length=.5)
    observation = PDMObservation(sampling, sampling, map_radius, observation_sample_res=1)
    observation.update_detections_tracks(
        [DetectionsTracks(TrackedObjects(list(f.actors))) for f in frames],
        traffic_light_data=[[] for _ in frames], route_lane_dict=route_lanes,
        compute_traffic_light_data=True)
    return observation


class SegmentReward(H1Reward):
    def safety(self, frame):
        scorer = self._scorer([frame], CurrentObservation(frame))
        scorer._calculate_no_at_fault_collision()
        scorer._calculate_drivable_area_compliance()
        return (float(scorer._multi_metrics[M.NO_COLLISION, 0]),
                float(scorer._multi_metrics[M.DRIVABLE_AREA, 0]))

    def progress(self, start, end):
        scorer = self._scorer([start, end], CurrentObservation(end))
        scorer._calculate_progress()
        return float(scorer._progress_raw[0])

    def score(self, frames, reference_progress_m):
        if len(frames) != 9 or any(b.step != a.step+1 for a,b in zip(frames, frames[1:])):
            raise ValueError('Four-second reward requires 9 consecutive actual frames')
        if any(f.ego.shape != (11,) or not np.isfinite(f.ego).all() for f in frames):
            raise ValueError('Invalid actual ego state')
        if not np.isfinite(reference_progress_m) or reference_progress_m < 0:
            raise ValueError('Invalid reference progress')
        for f in frames:
            for actor in f.actors:
                box = actor.box
                if not box.geometry.is_valid or box.geometry.is_empty or not np.isfinite(
                        [*box.center, box.length, box.width, box.height]).all():
                    raise ValueError('Invalid actor geometry')
                if hasattr(actor,'velocity') and not np.isfinite([actor.velocity.x,actor.velocity.y]).all():
                    raise ValueError('Invalid actor velocity')
        scorer = self._scorer(frames, actual_observation(frames, self.route_lanes))
        scorer._calculate_no_at_fault_collision()
        scorer._calculate_drivable_area_compliance()
        scorer._calculate_ttc_optimized()
        scorer._calculate_progress()
        scorer._calculate_is_comfortable()
        scorer._calculate_driving_direction_compliance()
        nc, dac = (float(scorer._multi_metrics[k,0]) for k in (M.NO_COLLISION,M.DRIVABLE_AREA))
        ttc, comfort, direction = (float(scorer._weighted_metrics[k,0]) for k in (W.TTC,W.COMFORTABLE,W.DRIVING_DIRECTION))
        d = float(scorer._progress_raw[0])
        ep = float(np.clip(max(d,0.) / max(reference_progress_m,.1),0.,1.))
        gate = nc*dac
        result = dict(reward=gate*(5*gate*ep+5*ttc+2*comfort)/12,
                      NC=nc, DAC=dac, EP=ep, gated_EP=gate*ep, TTC=ttc,
                      comfort=comfort, direction=direction, progress_m=d,
                      reference_progress_m=float(reference_progress_m))
        if not np.isfinite(list(result.values())).all():
            raise ValueError('Non-finite segment reward')
        collision = scorer._collision_time_idcs[0]
        offroad = np.flatnonzero(scorer._ego_areas[0,:,A.NON_DRIVABLE_AREA])
        ttc_idx = scorer._ttc_time_idcs[0]
        result['first_violation_seconds'] = dict(
            NC=float(collision*.5) if np.isfinite(collision) else None,
            DAC=float(offroad[0]*.5) if len(offroad) else None,
            TTC=float(ttc_idx*.5) if np.isfinite(ttc_idx) else None)
        return result
