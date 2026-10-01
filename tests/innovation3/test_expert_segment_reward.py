"""V2 tests against the unchanged evaluator and real reactive engine fixtures."""
import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from test_causal_reward import frame, actor
from test_fixed_segment import geometry, fixture
from worldengine.online.reward import ego_state
from worldengine.online.expert_segment_reward import ExpertWindow, ExpertSegmentReward, CONTRACT
from worldengine.online.expert_fixed_rollout import ExpertFixedRewardSession
from worldengine.online.fixed_rollout import rollout
from worldengine.online.fixed_parallel import FixedBranchPool
from worldengine.online.fixed_diagnostics import compact_result
from worldengine.online.headless import diagnostic_candidates
from worldengine.online.segment_reward import actual_observation
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from worldengine.components.agents.policy.pdm_planner.scoring.pdm_scorer import PDMScorer
from worldengine.manager.metric_manager import MetricManager
from worldengine.common.dataclasses import NavsimTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from innovation3.fixed_protocol import validate


def reference(step=0, distance=16., heading=0.):
    return ExpertWindow.from_track(dict(position=np.column_stack([np.linspace(0, distance, 9), np.zeros(9)]),
        heading=np.full(9, heading)), step, 1.461, 'fixture')


def scorer(frames, ref=None):
    _, lanes, drivable = geometry()
    return ExpertSegmentReward(ref or reference(), frames[0], drivable, drivable), lanes


def test_final_collision_offroad_partial_gate_and_real_acceleration():
    frames = [frame(i) for i in range(9)]
    reward, lanes = scorer(frames)
    assert reward.score(frames, lanes)['reward'] == 1
    for which in ('collision', 'offroad', 'cone'):
        changed = copy.deepcopy(frames)
        obstacle = actor(19.)
        if which == 'cone':
            obstacle = StaticObject(TrackedObjectType.TRAFFIC_CONE, obstacle.box, obstacle.metadata)
        changed[-1] = frame(8, lateral=6.) if which == 'offroad' else frame(8, actors=[obstacle])
        result = reward.score(changed, lanes)
        key = 'DAC' if which == 'offroad' else 'NC'
        assert result['first_violation_seconds'][key] == 4.
        if which == 'cone':
            assert result['NC'] == .5 and result['gated_EP'] == .5
            assert result['reward'] == pytest.approx(.5*(5*.5+5*result['TTC']+2*result['comfort'])/12)
        else:
            assert result['reward'] == 0
    for f in frames:
        f.ego[5] = 8.
    reward, lanes = scorer(frames)
    assert reward.score(frames, lanes)['comfort'] == 0


def test_stationary_reference_uses_original_threshold_and_no_group_normalization():
    frames = [frame(i, speed=0.) for i in range(9)]
    for f in frames:
        f.ego[0] = 0.
    reward, lanes = scorer(frames, reference(distance=0.))
    assert reward.score(frames, lanes)['EP'] == 1.
    assert reward.score(frames, lanes)['reward'] == 1.
    moving = [frame(i) for i in range(9)]
    reward, lanes = scorer(moving, reference(distance=32.))
    original = reward.score(moving, lanes)
    faster = copy.deepcopy(moving)
    for f in faster[1:]:
        f.ego[0] *= 3
    assert reward.score(faster, lanes)['EP'] == 1.
    assert reward.score(moving, lanes) == original
    assert 0 < original['EP'] < 1


def test_reference_reanchoring_coordinates_heading_and_detachment():
    track = dict(position=np.column_stack([np.arange(9)*3., np.arange(9)*.2]),
                 heading=np.linspace(3., 3.4, 9))
    ref = ExpertWindow.from_track(track, 0, 1.461, 'scene')
    start = frame(0)
    start.ego[:3] = [700., -300., -3.1]
    states, line = ref.materialize(start)
    np.testing.assert_array_equal(states[0, :3], start.ego[:3])
    np.testing.assert_allclose(states[1:, :2], ref.poses[1:, :2]+start.ego[:2], atol=1e-9)
    np.testing.assert_allclose(np.unwrap(states[1:, 2]), np.unwrap(ref.poses[1:, 2]), atol=1e-9)
    saved = ref.poses.copy()
    track['position'][:] += 100
    np.testing.assert_array_equal(ref.poses, saved)
    with pytest.raises(ValueError):
        ref.poses[0, 0] = 4


def test_adapter_matches_unchanged_metric_manager_for_legacy_states():
    frames = [frame(i) for i in range(9)]
    frames[5] = frame(5, actors=[actor(30.)])
    ref = reference(distance=24.)
    adapter, lanes = scorer(frames, ref)
    result = adapter.score(frames, lanes)
    manager = MetricManager.__new__(MetricManager)
    manager.num_history, manager.num_future, manager.buffer_size = 1, 8, 9
    manager._proposal_sampling = TrajectorySampling(num_poses=8, interval_length=.5)
    manager._map_radius, manager._map_api = 100, adapter.map_api
    manager._cached_roadblock_ids = ['road']
    manager.ego_states_list = [ego_state(f) for f in frames]
    observation = actual_observation(frames, lanes)
    from nuplan.common.actor_state.tracked_objects import TrackedObjects
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    manager.detection_tracks_list = [DetectionsTracks(TrackedObjects(list(f.actors))) for f in frames]
    manager.traffic_light_data_list = [[] for _ in frames]
    manager.first_NC_DAC_step = None
    manager._pdm_closed = SimpleNamespace(_load_route_dicts=lambda ids: None,
        _route_lane_dict=lanes, _scorer=PDMScorer(manager._proposal_sampling))
    trajectory = NavsimTrajectory(ref.poses.copy(), TrajectorySampling(num_poses=9, interval_length=.5))
    with patch.object(PDMDrivableMap, 'from_simulation', return_value=adapter.drivable_map):
        expected = manager.compute_pdm_scores(trajectory, np.stack([f.ego[3:5] for f in frames]))
    assert result['reward'] == pytest.approx(expected.score, abs=1e-9, rel=1e-7)
    assert result['gated_EP'] == pytest.approx(expected.ego_progress, abs=1e-9)
    assert result['NC'] == expected.no_at_fault_collisions
    assert result['TTC'] == expected.time_to_collision_within_bound


def test_reference_validation_and_stale_receipts():
    frames = [frame(i) for i in range(9)]
    reward, lanes = scorer(frames)
    with pytest.raises(ValueError):
        reward.score(frames[:-1], lanes)
    with pytest.raises(ValueError):
        reference(step=1)
    bad = copy.deepcopy(frames)
    bad[3].ego[0] = np.nan
    with pytest.raises(ValueError):
        reward.score(bad, lanes)
    identity = dict(step=0)
    main = dict(kind='main', identity=identity, selected=0, candidate_hash='bank', next_hash='first')
    metadata = reward.receipt(0.)
    metadata.pop('frames')
    feedback = dict(kind='feedback', identity=identity, selected=0, candidate_hash='bank', main_hash='first',
        contract=CONTRACT['name'], scored=True, history_after_hash='next', reference=metadata,
        first_hashes=['first']*20, end_hashes=['end']*20, selected_first_state_parity=True,
        canonical_restored=True, rewards=[reward.score(frames, lanes)]*20)
    validate(identity, 0, 'bank', main, feedback, True, expected_contract=CONTRACT['name'])
    with pytest.raises(ValueError, match='contract'):
        validate(identity, 0, 'bank', main, feedback, True)
    feedback['reference']['frames'] = []
    with pytest.raises(ValueError, match='scoring-only'):
        validate(identity, 0, 'bank', main, feedback, True, expected_contract=CONTRACT['name'])


def test_reactive_all20_parallel_reverse_and_detached_expert_counterfactual():
    scene, sim, old_session = fixture('R')
    session = ExpertFixedRewardSession.__new__(ExpertFixedRewardSession)
    session.__dict__.update(old_session.__dict__)
    session.branch_route_lanes = lambda ids: session.route_lanes
    try:
        before, history = sim.snapshot(), session.history.snapshot()
        actions = diagnostic_candidates(sim)
        with patch.object(PDMDrivableMap, 'from_simulation', return_value=session.map_api):
            serial = session.group(actions, 12, audit='full')
            sim.restore(before); session.history.restore(history)
            with FixedBranchPool('fixture', scene, 'R', 8, 17, sim.codec.bundle, session.offset,
                    session.actor_types, workers=2, timeout=90, global_config=sim.engine.global_config) as pool:
                parallel = session.group(actions, 12, pool=pool, audit='boundary')
            for key in ('rewards', 'first_hashes', 'end_hashes', 'history_after_hash', 'main_hash'):
                assert serial[key] == parallel[key]
            assert compact_result(serial)['reference']['simulated'] is False
            for i in reversed(range(20)):
                trace = rollout(sim, before, actions[i], session.offset, session.actor_types, audit='boundary')
                assert trace['end_hash'] == serial['end_hashes'][i]
                assert trace['route_roadblock_ids'] == serial['traces'][i]['route_roadblock_ids']
            sim.restore(before); session.history.restore(history)
            source = session.scoring_reference(session.capture())
            changed = source.poses.copy(); changed[1:, :2] *= 4
            with patch.object(session, 'scoring_reference', return_value=replace(source, poses=changed)):
                counterfactual = session.group(actions, 12, audit='boundary')
            assert serial['main_hash'] == counterfactual['main_hash']
            assert serial['end_hashes'] == counterfactual['end_hashes']
            assert serial['rewards'] != counterfactual['rewards']
    finally:
        sim.close()
