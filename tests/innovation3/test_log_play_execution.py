"""Upstream ideal tracking endpoints, complete warmup, and protocol isolation."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'projects/SimEngine'))
sys.path.insert(0, str(Path(__file__).parent))
from test_headless_snapshot import scene_fixture
from worldengine.online.actions import to_world_actions, centered_pose_action
from worldengine.online.headless import HeadlessSimulator, dynamics_config
from worldengine.online.execution import execution_contract, validate_resume_execution
from worldengine.online.reward import capture_reward_frame


@pytest.mark.parametrize('reaction', ['NR', 'R'])
def test_generated_curve_executes_upstream_next_waypoint_and_velocity(reaction):
    scene = scene_fixture()
    scene['object_track']['ego']['state']['heading'][:] = .7
    sim = HeadlessSimulator('fixture', scene, reaction, 8, 17)
    try:
        assert sim.engine.global_config.ego_controller == 'log_play_controller'
        assert not sim.engine.global_config.online_corrected_bicycle
        assert sim.dynamics_contract == 'upstream_log_play_h1_v1'
        ego = sim.engine.agent_manager.ego_agent
        assert not hasattr(ego.controller, '_tracker')
        old_velocity = np.array(ego.current_velocity)
        bank = np.zeros((20, 8, 3))
        bank[:, :, 0] = np.arange(1, 9) * 2.
        bank[:, :, 1] = .01 * bank[:, :, 0] ** 2
        bank[:, :, 2] = np.arctan(.02 * bank[:, :, 0])
        action = to_world_actions(bank, ego)[0]
        before = sim.snapshot()
        sim.step(action)
        ego = sim.engine.agent_manager.ego_agent
        np.testing.assert_array_equal(ego.current_position, action.waypoints[1])
        np.testing.assert_array_equal(ego.current_velocity, action.velocities[1])
        assert ego.current_heading == pytest.approx(action.headings[1])
        assert ego.current_angular_velocity == action.angular_velocities[1]
        np.testing.assert_allclose(ego.current_acceleration, (action.velocities[1]-old_velocity)/.5)
        types = {k: v['type'] for k, v in scene['object_track'].items()}
        assert np.isfinite(capture_reward_frame(sim.engine, np.zeros(2), types).ego).all()
        expected = sim.snapshot()
        sim.restore(before)
        sim.step(action)
        assert sim.snapshot().state_hash == expected.state_hash
    finally:
        sim.close()


@pytest.mark.parametrize('field', ['velocities', 'angular_velocities'])
@pytest.mark.parametrize('invalid', [None, 'nonfinite', 'shape'])
def test_incomplete_action_is_rejected_before_mutating_environment(field, invalid):
    sim = HeadlessSimulator('fixture', scene_fixture(), 'NR', 8, 17)
    try:
        action = centered_pose_action(np.column_stack([np.arange(9), np.zeros(9)]), np.zeros(9))
        value = getattr(action, field)
        setattr(action, field, None if invalid is None else
                (np.full_like(value, np.nan) if invalid == 'nonfinite' else value[:-1]))
        before = sim.snapshot()
        with pytest.raises(ValueError, match='Ideal tracking requires'):
            sim.step(action)
        assert sim.snapshot().state_hash == before.state_hash
    finally:
        sim.close()


def test_complete_pose_action_handles_stationary_and_wrapped_heading():
    hold = centered_pose_action(np.tile([4., 3.], (9, 1)), np.full(9, .7))
    np.testing.assert_array_equal(hold.velocities, np.zeros((9, 2)))
    np.testing.assert_array_equal(hold.angular_velocities, np.zeros(9))
    curve = centered_pose_action(np.array([[0., 0.], [1., 0.], [2., 1.]]),
                                 np.array([3.1, -3.1, -3.]))
    assert np.max(np.abs(curve.angular_velocities)) < 1.
    np.testing.assert_array_equal(curve.velocities[1], [2., 2.])


def test_execution_contract_rejects_legacy_or_unknown_resume():
    ideal = execution_contract()
    legacy = execution_contract('two_stage_controller')
    assert ideal != legacy
    assert dynamics_config('R', 8, 'two_stage_controller').online_corrected_bicycle
    for payload in ({}, {'dynamics_contract': legacy}):
        with pytest.raises(ValueError, match='Resume execution contract'):
            validate_resume_execution(payload, ideal)
    validate_resume_execution({'dynamics_contract': ideal}, ideal)
    with pytest.raises(ValueError, match='Unsupported online ego controller'):
        dynamics_config('R', 8, 'unknown')
