"""Behavioral tests for whole-segment reward, restored branches and new wire."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from shapely.geometry import box
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'projects/SimEngine'))
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
sys.path.insert(0,str(Path(__file__).parent))
from test_headless_snapshot import scene_fixture
from test_causal_reward import frame,actor
from worldengine.online.headless import HeadlessSimulator,diagnostic_candidates
from worldengine.online.reward import RewardHistory,capture_reward_frame
from worldengine.online.fixed_rollout import FixedRewardSession,rollout,suffix,reference_rollout,reference_action
from worldengine.online.fixed_parallel import FixedBranchPool
from worldengine.online.segment_reward import SegmentReward
from worldengine.online.state import structural_hash
from worldengine.components.agents.policy.pdm_planner.utils.pdm_path import PDMPath
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from innovation3.fixed_protocol import validate


def geometry():
    polygon=box(-100,-6,400,6)
    drivable=PDMDrivableMap(['lane'],[SemanticMapLayer.LANE],[polygon])
    line=PDMPath([StateSE2(-100.,0.,0.),StateSE2(400.,0.,0.)])
    lanes={'lane':SimpleNamespace(polygon=polygon,speed_limit_mps=15.)}
    return line,lanes,drivable


def fixture(reaction='R',scene=None):
    scene=scene_fixture() if scene is None else scene
    sim=HeadlessSimulator('fixture',scene,reaction,8,17)
    session=FixedRewardSession.__new__(FixedRewardSession);session.sim=sim;session.offset=np.zeros(2)
    session.actor_types={k:v['type'] for k,v in scene['object_track'].items()}
    session.history=RewardHistory();session.history.append(session.capture())
    session.centerline,session.route_lanes,session.map_api=geometry()
    return scene,sim,session


def scorer():
    line,lanes,drivable=geometry()
    return SegmentReward(line,lanes,drivable,drivable)


def test_whole_segment_detects_final_frame_collision_and_offroad():
    reward=scorer();frames=[frame(i) for i in range(9)]
    assert reward.score(frames,16.)['reward']==1.
    collision=copy.deepcopy(frames);collision[8]=frame(8,actors=[actor(19.)])
    result=reward.score(collision,16.)
    assert result['NC']==0 and result['reward']==0 and result['first_violation_seconds']['NC']==4.
    offroad=copy.deepcopy(frames);offroad[8]=frame(8,lateral=6.)
    result=reward.score(offroad,16.)
    assert result['DAC']==0 and result['first_violation_seconds']['DAC']==4.


def test_progress_stopped_reference_and_double_gate_partial_collision():
    reward=scorer();frames=[frame(i) for i in range(9)]
    assert reward.score(frames,0.)['EP']==1.
    stopped=[frame(i,speed=0.) for i in range(9)]
    for f in stopped:f.ego[0]=0.
    result=reward.score(stopped,0.)
    assert result['EP']==0. and result['reward']==pytest.approx(7./12)
    a=actor(19.)
    obstacle=StaticObject(TrackedObjectType.TRAFFIC_CONE,a.box,a.metadata)
    frames[8]=frame(8,actors=[obstacle]);result=reward.score(frames,16.)
    assert result['NC']==.5 and result['gated_EP']==.5
    assert result['reward']==pytest.approx(.5*(5*.5+5*result['TTC']+2*result['comfort'])/12)


def test_invalid_windows_geometry_reference_rejected():
    reward=scorer();frames=[frame(i) for i in range(9)]
    with pytest.raises(ValueError):reward.score(frames[:-1],16.)
    bad=copy.deepcopy(frames);bad[-1]=frame(10)
    with pytest.raises(ValueError):reward.score(bad,16.)
    bad=copy.deepcopy(frames);bad[-1].ego[0]=np.nan
    with pytest.raises(ValueError):reward.score(bad,16.)
    with pytest.raises(ValueError):reward.score(frames,np.nan)
    with pytest.raises(ValueError):reward.score(frames,-1.)


@pytest.mark.parametrize('reaction',['NR','R'])
def test_all20_serial_parallel_reverse_spawn_exit_and_main_history(reaction):
    scene,sim,session=fixture(reaction)
    try:
        actions=diagnostic_candidates(sim);before=sim.snapshot();history=session.history.snapshot();bank=structural_hash(actions)
        with patch.object(PDMDrivableMap,'from_simulation',return_value=session.map_api):
            serial=session.group(actions,12)
            main=sim.snapshot();history_after=session.history.state_hash
            assert len(session.history.frames)==2 and session.history.frames[-1].step==1
            sim.restore(before);session.history.restore(history)
            with FixedBranchPool('fixture',scene,reaction,8,17,sim.codec.bundle,session.offset,session.actor_types,
                    workers=2,timeout=90,global_config=sim.engine.global_config) as pool:
                parallel=session.group(actions,12,pool=pool)
            for key in ('rewards','first_hashes','end_hashes','main_hash','history_after_hash'):
                assert parallel[key]==serial[key]
            assert sim.snapshot().state_hash==main.state_hash and session.history.state_hash==history_after
            for i in reversed(range(20)):
                trace=rollout(sim,before,actions[i],session.offset,session.actor_types)
                assert trace['end_hash']==serial['end_hashes'][i]
                assert [f.state_hash for f in trace['frames']]==[f.state_hash for f in serial['traces'][i]['frames']]
                expected=actions[i].waypoints[-1]
                actual=sim.engine.agent_manager.ego_agent.current_position[:2]
                np.testing.assert_array_equal(actual,expected)
            tokens=[{a.track_token for a in f.actors} for f in serial['traces'][0]['frames']]
            assert tokens[0]!=tokens[1] and tokens[1]!=tokens[2]
            assert structural_hash(actions)==bank
            if reaction=='R':
                actor_paths=[[[list(a.box.center) for a in f.actors] for f in b['frames']] for b in serial['traces']]
                assert len({structural_hash(x) for x in actor_paths})>1
    finally:sim.close()


def test_two_point_support_is_explicit_and_no_future_padding():
    _,sim,session=fixture()
    try:
        action=diagnostic_candidates(sim)[12]
        with pytest.raises(ValueError):sim.step(suffix(action,7))
        sim.step(suffix(action,7),allow_two_point_logplay=True)
        assert sim.engine.episode_step==1
        before=sim.snapshot()
        with pytest.raises(ValueError,match='future window'):session.group(diagnostic_candidates(sim),0)
        assert sim.snapshot().state_hash==before.state_hash
    finally:sim.close()


def test_failure_restores_main_but_does_not_commit_history():
    _,sim,session=fixture()
    try:
        actions=diagnostic_candidates(sim);before=sim.snapshot();history_hash=session.history.state_hash
        sim.step(actions[12]);main=sim.snapshot();sim.restore(before)
        with patch.object(PDMDrivableMap,'from_simulation',return_value=session.map_api),patch(
                'worldengine.online.fixed_rollout.reference_rollout',side_effect=RuntimeError('reference invalid')):
            with pytest.raises(RuntimeError,match='reference invalid'):session.group(actions,12)
        assert sim.snapshot().state_hash==main.state_hash and session.history.state_hash==history_hash
    finally:sim.close()


def test_reference_failure_uses_last_safe_progress():
    _,sim,session=fixture('NR')
    try:
        adapter=scorer();before=sim.snapshot();calls=[]
        original=adapter.safety
        def fail_third(frame):
            calls.append(frame)
            return (1.,0.) if len(calls)==3 else (1.,1.)
        with patch.object(adapter,'safety',side_effect=fail_third):ref=reference_rollout(session,before,adapter)
        assert ref['failure_step']==2
        assert ref['progress_m']==adapter.progress(calls[0],calls[1])
    finally:sim.close()


def test_changing_ego_expert_future_does_not_change_simulated_rewards():
    baseline=scene_fixture();changed=copy.deepcopy(baseline)
    ego_key=baseline['ego_id'] if 'ego_id' in baseline else 'ego'
    changed['object_track'][ego_key]['state']['position'][1:,:2]+=123.
    results=[]
    for scene in (baseline,changed):
        _,sim,session=fixture('R',scene)
        try:
            with patch.object(PDMDrivableMap,'from_simulation',return_value=session.map_api):
                result=session.group(diagnostic_candidates(sim),12)
            results.append((result['rewards'],[[f.state_hash for f in t['frames']] for t in result['traces']]))
        finally:sim.close()
    assert results[0]==results[1]


def test_new_wire_accepts_distinct_end_hash_and_rejects_stale_first_hash():
    identity=dict(step=3);rewards=[scorer().score([frame(i) for i in range(9)],16.)]*20
    main=dict(kind='main',identity=identity,selected=0,candidate_hash='bank',next_hash='first')
    feedback=dict(kind='feedback',identity=identity,selected=0,candidate_hash='bank',main_hash='first',
        contract='fixed_candidate_4s_logplay_pdm_v1',scored=True,history_before_hash='old',history_after_hash='new',
        first_hashes=['first']*20,end_hashes=['different_end']*20,rewards=rewards,
        selected_first_state_parity=True,canonical_restored=True)
    validate(identity,0,'bank',main,feedback,True,'old')
    feedback['first_hashes'][0]='different_end'
    with pytest.raises(ValueError):validate(identity,0,'bank',main,feedback,True,'old')


def test_angle_wrap_and_stopped_candidate_execute_last_point():
    from worldengine.online.actions import centered_pose_action
    _,sim,session=fixture('NR')
    try:
        root=sim.snapshot();points=np.column_stack([-np.arange(9)*2.,np.zeros(9)])
        headings=(np.linspace(3.,3.4,9)+np.pi)%(2*np.pi)-np.pi
        action=centered_pose_action(points,headings)
        trace=rollout(sim,root,action,session.offset,session.actor_types)
        np.testing.assert_array_equal(sim.engine.agent_manager.ego_agent.current_position[:2],points[-1])
        assert trace['frames'][-1].ego[2]==pytest.approx(headings[-1])
        sim.restore(root);center=sim.engine.agent_manager.ego_agent.current_position[:2]
        stopped=centered_pose_action(np.tile(center,(9,1)),np.zeros(9))
        trace=rollout(sim,root,stopped,session.offset,session.actor_types)
        assert np.linalg.norm(trace['frames'][-1].ego[3:5])==0.
    finally:sim.close()


def test_all_zero_candidates_are_valid_and_bad_actor_velocity_rejected():
    from nuplan.common.actor_state.state_representation import StateVector2D
    reward=scorer();frames=[frame(i,lateral=10.) for i in range(9)]
    assert [reward.score(frames,16.)['reward'] for _ in range(20)]==[0.]*20
    bad=actor(50.);bad._velocity=StateVector2D(np.nan,0.)
    with pytest.raises(ValueError,match='velocity'):reward.score([frame(i,actors=[bad]) for i in range(9)],16.)


def test_dead_fixed_worker_does_not_change_parent_and_closes_pool():
    scene,sim,session=fixture()
    try:
        with FixedBranchPool('fixture',scene,'R',8,17,sim.codec.bundle,session.offset,session.actor_types,
                workers=1,timeout=5,global_config=sim.engine.global_config) as pool:
            before=sim.snapshot();pool._processes[0].terminate();pool._processes[0].join(5)
            with pytest.raises((EOFError,OSError,RuntimeError)):
                pool.run(before,diagnostic_candidates(sim))
            assert sim.snapshot().state_hash==before.state_hash and pool._closed
    finally:sim.close()


def test_motionrft_fast_bridge_interpolates_without_importing_jax():
    from worldengine.online.motionrft_fast import _interpolate_candidates,CONTRACT
    candidates=np.zeros((20,8,3));candidates[:,:,0]=np.arange(8)*2.
    dense=_interpolate_candidates(candidates)
    assert dense.shape==(20,41,3)
    np.testing.assert_allclose(dense[:,::5],np.concatenate([np.zeros((20,1,3)),candidates],axis=1))
    assert CONTRACT['reaction']=='NR_screening_only' and CONTRACT['exact_R_reward'] is False
