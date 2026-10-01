"""Frozen diagnostic replay using the identical initial snapshot and static bundle.

Training still uses its unmodified registered worker and snapshots.
"""
import copy
import json
from pathlib import Path
from innovation3 import fixed_worker as worker

OriginalSimulator = worker.HeadlessSimulator
OriginalObserver = worker.CanonicalObserver
OriginalEpisode = worker.episode
output = None
initial = None

def replay_simulator(scene_id, scene, reaction, max_steps, seed, backend):
    global initial
    key = (scene_id, reaction, max_steps, seed, backend)
    if initial is None or initial['key'] != key:
        sim = OriginalSimulator(scene_id, scene, reaction, max_steps, seed, backend=backend)
        initial = dict(key=key, config=copy.deepcopy(sim.engine.global_config),
                       bundle=sim.codec.bundle, snapshot=sim.snapshot())
        return sim
    # Rebuild the observer/history outside the snapshot, but restore the exact
    # initial simulator graph, RNGs, clock metadata and immutable map bundle.
    # Independently rebuilding a map gives it a fresh object name; comparing
    # those cross-reset full hashes was an invalid diagnostic assumption.
    sim = OriginalSimulator.from_snapshot(initial['config'], initial['bundle'],
                                           initial['snapshot'], backend=backend)
    if sim.snapshot().state_hash != initial['snapshot'].state_hash:
        sim.close()
        raise RuntimeError('Diagnostic initial snapshot restore mismatch')
    return sim

class AuditedObserver(OriginalObserver):
    def observe(self):
        result = super().observe()
        details = self.sim.codec.describe(self.sim.snapshot())
        path = output / ('replay_state_%02d.json' % self.sim.engine.episode_step)
        path.write_text(json.dumps(details, indent=2) + '\n')
        return result

def episode(request, *args, **kwargs):
    global output
    if request['mode'] != 'frozen':
        raise ValueError('Repair worker is restricted to frozen diagnostics')
    output = Path(request['output'])
    return OriginalEpisode(request, *args, **kwargs)

if __name__ == '__main__':
    worker.HeadlessSimulator = replay_simulator
    worker.CanonicalObserver = AuditedObserver
    worker.episode = episode
    worker.main()
