"""Exact packed lane queries, outside the simulator's mutable state graph.

Only the reviewed float64 BaseCenterLine implementation is accelerated. Other
lane implementations use their original query, never a different distance rule.
No vehicle update is batched or reordered by this module.
"""
import weakref
import numpy as np

BACKENDS = ('reference', 'cached')
BACKEND_VERSION = 'we_exact_r_geometry_v1'
_enabled = False
_networks = weakref.WeakKeyDictionary()


def configure(backend):
    global _enabled
    if backend not in BACKENDS:
        raise ValueError('Unknown exact simulator backend: ' + str(backend))
    _enabled = backend == 'cached'
    _networks.clear()


class UnsupportedGeometry(ValueError):
    pass


class PackedLaneQueries:
    def __init__(self, network):
        from worldengine.components.maps.lanes.base_center_lane import BaseCenterLine
        from worldengine.components.maps.lanes.base_lane import BaseLane
        self.keys = tuple(network.graph)
        self.lanes = tuple(info.lane for info in network.graph.values())
        if not self.lanes:
            raise UnsupportedGeometry('Empty road network')
        for lane in self.lanes:
            if (type(lane).distance is not BaseLane.distance or
                    type(lane).local_coordinates is not BaseCenterLine.local_coordinates or
                    type(lane).min_lineseg_dist is not BaseCenterLine.min_lineseg_dist):
                raise UnsupportedGeometry('Unreviewed lane projection')
            arrays = [lane.lane_start_points, lane.lane_end_points]
            arrays += [s[k] for s in lane.lane_segments for k in ('direction', 'lateral_direction')]
            if any(np.asarray(a).dtype != np.dtype('float64') or not np.isfinite(a).all() for a in arrays):
                raise UnsupportedGeometry('Only finite float64 geometry is accelerated')
        self.start = np.concatenate([x.lane_start_points for x in self.lanes])
        self.end = np.concatenate([x.lane_end_points for x in self.lanes])
        sizes = np.array([len(x.lane_segments) for x in self.lanes])
        self.offsets = np.r_[0, np.cumsum(sizes)[:-1]]
        self.lane_id = np.repeat(np.arange(len(self.lanes)), sizes)
        self.indices = np.arange(len(self.start))
        self.lengths = np.array([x.length for x in self.lanes])
        segments = [s for lane in self.lanes for s in lane.lane_segments]
        self.direction = np.array([s['direction'] for s in segments])
        self.lateral = np.array([s['lateral_direction'] for s in segments])
        prefix = []
        for lane in self.lanes:
            value = 0
            for segment in lane.lane_segments:
                prefix.append(value)
                value += segment['length']  # preserve the original sequential sum
        self.prefix = np.array(prefix)
        delta = self.end - self.start
        self.tangent = np.divide(delta, np.hypot(delta[:, 0], delta[:, 1]).reshape(-1, 1))
        if not np.isfinite(self.tangent).all():
            raise UnsupportedGeometry('Degenerate segment')
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False

    def distances(self, position):
        pos = np.asarray(position)
        if pos.shape != (2,) or not np.isfinite(pos).all():
            raise ValueError('Lane query requires a finite XY point')
        s = np.multiply(self.start - pos, self.tangent).sum(axis=1)
        t = np.multiply(pos - self.end, self.tangent).sum(axis=1)
        h = np.maximum.reduce([s, t, np.zeros(len(s))])
        pa = pos - self.start
        c = pa[:, 0] * self.tangent[:, 1] - pa[:, 1] * self.tangent[:, 0]
        distance = np.hypot(h, c)
        minimum = np.minimum.reduceat(distance, self.offsets)
        # np.argmin chooses the first tied segment, not an arbitrary reduction.
        nearest = np.minimum.reduceat(np.where(
            distance == minimum[self.lane_id], self.indices, len(self.indices)), self.offsets)
        dx = pos[0] - self.start[nearest, 0]
        dy = pos[1] - self.start[nearest, 1]
        longitudinal = self.prefix[nearest] + (
            dx * self.direction[nearest, 0] + dy * self.direction[nearest, 1])
        lateral = dx * self.lateral[nearest, 0] + dy * self.lateral[nearest, 1]
        a, b = longitudinal - self.lengths, 0 - longitudinal
        return abs(lateral) + np.where(a > 0, a, 0) + np.where(b > 0, b, 0)

    def query(self, position, return_all=False):
        distances = self.distances(position)
        order = np.argsort(distances, kind='stable') if return_all else [int(np.argmin(distances))]
        result = [(distances[i], self.keys[i], self.lanes[i]) for i in order]
        return result if return_all else result[0]


def cached_query(network, position, return_all=False):
    if not _enabled:
        return None
    if network not in _networks:
        try:
            _networks[network] = PackedLaneQueries(network)
        except UnsupportedGeometry:
            _networks[network] = None
    cache = _networks[network]
    return None if cache is None else cache.query(position, return_all)


def cache_status():
    return dict(enabled=_enabled, version=BACKEND_VERSION,
                accelerated_networks=sum(v is not None for v in _networks.values()),
                reference_networks=sum(v is None for v in _networks.values()))
