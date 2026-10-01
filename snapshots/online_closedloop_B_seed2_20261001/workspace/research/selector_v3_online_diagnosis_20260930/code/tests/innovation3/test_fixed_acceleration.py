"""Semantic equivalence of packed geometry, audit schedules and worker reuse."""
import copy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from test_fixed_segment import fixture
from worldengine.components.maps.lanes.base_center_lane import BaseCenterLine
from worldengine.components.agents.policy.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from worldengine.online.geometry_cache import PackedLaneQueries, UnsupportedGeometry
from worldengine.online.headless import HeadlessSimulator, diagnostic_candidates
from worldengine.online.fixed_parallel import FixedBranchPool


def test_packed_geometry_preserves_every_distance_and_stable_ties():
    rng = np.random.RandomState(23)
    lines = [np.cumsum(rng.normal(0, 3, (25, 2)), axis=0) for _ in range(17)]
    lines += [lines[0].copy(), np.array([[0., 0.], [100., 0.]])]
    lanes = [BaseCenterLine(line) for line in lines]
    network = SimpleNamespace(graph={str(i): SimpleNamespace(lane=lane) for i, lane in enumerate(lanes)})
    cache = PackedLaneQueries(network)
    points = np.concatenate([rng.normal(0, 30, (100, 2)), *[x.lane_start_points for x in lanes]])
    for point in points:
        expected = np.array([lane.distance(point) for lane in lanes])
        actual = cache.distances(point)
        assert actual.tobytes() == expected.tobytes()
        order = sorted(range(len(lanes)), key=lambda i: expected[i])
        assert [row[1] for row in cache.query(point, True)] == [str(i) for i in order]
        assert cache.query(point)[1] == str(order[0])


def test_unreviewed_geometry_is_not_silently_approximated():
    class DifferentLane(BaseCenterLine):
        def distance(self, point):
            return 42.
    with pytest.raises(UnsupportedGeometry):
        PackedLaneQueries(SimpleNamespace(graph={'x': SimpleNamespace(lane=DifferentLane([[0., 0.], [5., 0.]]))}))


@pytest.mark.parametrize('reaction', ['R', 'NR'])
def test_cached_boundary_all20_equals_reference_full(reaction):
    _, sim, session = fixture(reaction)
    try:
        actions = diagnostic_candidates(sim)
        before, history = sim.snapshot(), session.history.snapshot()
        with patch.object(PDMDrivableMap, 'from_simulation', return_value=session.map_api):
            reference = session.group(actions, 12, audit='full')
            sim.restore(before)
            session.history.restore(history)
            sim.set_backend('cached')
            accelerated = session.group(actions, 12, audit='boundary')
        for key in ('first_hashes', 'end_hashes', 'rewards', 'main_hash', 'history_after_hash'):
            assert reference[key] == accelerated[key]
        for ref, fast in zip(reference['traces'], accelerated['traces']):
            assert [f.state_hash for f in ref['frames']] == [f.state_hash for f in fast['frames']]
            assert fast['state_hashes'][1:7] == [None]*6
        # Complete audits still check all intermediate hidden states.
        from worldengine.online.fixed_rollout import rollout
        audited = rollout(sim, before, actions[12], session.offset, session.actor_types, audit='full')
        assert audited['state_hashes'] == reference['traces'][12]['state_hashes']
    finally:
        sim.close()


def test_worker_rebinds_new_scene_without_restarting_and_rejects_old_root():
    scene, sim, session = fixture('R')
    pool = None
    try:
        root = sim.snapshot()
        pool = FixedBranchPool('fixture', scene, 'R', 8, 17, sim.codec.bundle,
            session.offset, session.actor_types, workers=2, timeout=60,
            global_config=sim.engine.global_config, backend='cached')
        pids = [p.pid for p in pool._processes]
        expected = pool.run(root, diagnostic_candidates(sim), audit='boundary')
        assert len(expected) == 20
        sim.close()
        changed = copy.deepcopy(scene)
        changed['object_track']['ego']['state']['position'][:, 1] += .25
        _, sim, session = fixture('NR', changed)
        new_root = sim.snapshot()
        pool.bind_scene('fixture-new', sim.codec.bundle, session.offset, session.actor_types, sim.engine.global_config)
        with pytest.raises(ValueError, match='bound scene'):
            pool.run(root, diagnostic_candidates(sim))
        result = pool.run(new_root, diagnostic_candidates(sim), audit='boundary')
        from worldengine.online.fixed_rollout import rollout
        serial = rollout(sim, new_root, diagnostic_candidates(sim)[12], session.offset, session.actor_types)
        assert result[12]['end_hash'] == serial['end_hash']
        assert [p.pid for p in pool._processes] == pids
    finally:
        if pool is not None:
            pool.close()
        sim.close()
