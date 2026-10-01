"""CPU branch workers reused across roots and scenes; exact R dynamics only."""
import hashlib
import logging
import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import wait
import numpy as np
from .headless import HeadlessSimulator, dynamics_config
from .state import structural_hash
from .parallel_branches import ParallelBranchPool, _IDMFailures
from .fixed_rollout import rollout, validate_action
from .geometry_cache import BACKENDS, BACKEND_VERSION, cache_status


def worker_main(conn, backend):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sim = None
    failures = _IDMFailures()
    logging.getLogger().addHandler(failures)
    try:
        conn.send(dict(kind='ready', pid=os.getpid(), cuda_visible_devices='',
                       backend=backend, backend_version=BACKEND_VERSION))
        scene = before = group = None
        while True:
            request = conn.recv()
            if request is None:
                break
            if request['kind'] == 'scene':
                if sim is not None:
                    sim.close()
                    sim = None
                scene = request
                before = group = None
                conn.send(dict(kind='scene_ready', scene_id=scene['scene_id'],
                    config_hash=structural_hash(scene['config']),
                    static_sha256=hashlib.sha256(scene['bundle']).hexdigest()))
                continue
            if request['kind'] == 'root':
                if scene is None:
                    raise RuntimeError('Scene must be bound before a root')
                started = time.monotonic()
                before, group = request['before'], request['group']
                if sim is None:
                    sim = HeadlessSimulator.from_snapshot(scene['config'], scene['bundle'], before, backend)
                else:
                    sim.restore(before)
                sim.codec.audit_static()
                conn.send(dict(kind='root_ready', group=group, before_hash=before.state_hash,
                               attach_seconds=time.monotonic()-started))
                continue
            if request.get('group') != group or before is None:
                raise RuntimeError('Stale branch root')
            if request['kind'] == 'finish':
                sim.codec.audit_static()
                conn.send(dict(kind='finished', group=group, geometry=cache_status()))
                continue
            if request['kind'] != 'candidate':
                raise ValueError('Unknown branch request')
            count = failures.count
            result = rollout(sim, before, request['action'], scene['offset'], scene['actor_types'],
                             audit=request['audit'], audit_static=request['audit']=='full')
            if failures.count != count:
                raise RuntimeError('IDM fallback invalidates branch')
            result.update(kind='result', index=request['index'], group=group, idm_fallbacks=0)
            conn.send(result)
    except BaseException as error:
        try:
            conn.send(dict(kind='error', error=repr(error), traceback=traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if sim is not None:
            sim.close()
        logging.getLogger().removeHandler(failures)
        conn.close()


class FixedBranchPool(ParallelBranchPool):
    def __init__(self, scene_id, scene, reaction='R', max_steps=8, seed=0,
                 static_bundle=None, offset=None, actor_types=None, workers=2, timeout=180.,
                 global_config=None, backend='reference'):
        if type(workers) is not int or not 1 <= workers <= 20:
            raise ValueError('workers must be 1..20')
        if backend not in BACKENDS or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('Invalid backend or timeout')
        self.timeout, self.backend = float(timeout), backend
        self._connections, self._processes = [], []
        self.worker_receipts = []
        self._group, self._closed = 0, False
        started = time.monotonic()
        ctx = mp.get_context('spawn')
        try:
            for worker in range(workers):
                parent, child = ctx.Pipe()
                process = ctx.Process(target=worker_main, args=(child, backend),
                                      name='worldengine-fixed-%02d' % worker)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._connections.append(parent)
                self._processes.append(process)
            for conn in self._connections:
                reply = self._receive(conn, 'ready')
                if reply['backend'] != backend or reply['cuda_visible_devices'] != '':
                    raise RuntimeError('Incorrect branch runtime')
                self.worker_receipts.append(reply)
            self.bind_scene(scene_id, static_bundle, offset, actor_types,
                            global_config if global_config is not None else dynamics_config(reaction, max_steps))
            self.startup_seconds = time.monotonic()-started
        except BaseException:
            self.close(abort=True)
            raise

    def _receive(self, conn, kind):
        if not conn.poll(self.timeout):
            raise TimeoutError('Fixed worker timed out: ' + kind)
        reply = conn.recv()
        if reply.get('kind') != kind:
            raise RuntimeError('Fixed worker failed: ' + repr(reply))
        return reply

    def bind_scene(self, scene_id, bundle, offset, actor_types, global_config):
        if self._closed or bundle is None or offset is None or actor_types is None:
            raise ValueError('Live pool and complete scene binding required')
        started = time.monotonic()
        request = dict(kind='scene', scene_id=scene_id, bundle=bundle,
                       offset=np.asarray(offset, dtype=float), actor_types=actor_types, config=global_config)
        expected = dict(kind='scene_ready', scene_id=scene_id,
                        static_sha256=hashlib.sha256(bundle).hexdigest(), config_hash=structural_hash(global_config))
        try:
            for conn in self._connections:
                conn.send(request)
            for conn in self._connections:
                if self._receive(conn, 'scene_ready') != expected:
                    raise RuntimeError('Worker scene binding mismatch')
            self.scene_binding = expected
            self.bind_seconds = time.monotonic()-started
        except BaseException:
            self.close(abort=True)
            raise

    def run(self, before, actions, audit='full'):
        if self._closed:
            raise RuntimeError('Branch pool is closed')
        if len(actions) != 20 or audit not in ('full', 'boundary'):
            raise ValueError('Expected 20 candidates and a valid audit mode')
        if before.static_sha256 != self.scene_binding['static_sha256'] or before.config_hash != self.scene_binding['config_hash']:
            raise ValueError('Snapshot does not belong to the bound scene')
        for action in actions:
            validate_action(action)
        self._group += 1
        group = self._group
        started = time.monotonic()
        self.last_profile = dict(pipe_send_seconds=0., pipe_receive_seconds=0.)
        def send(conn, message):
            t = time.monotonic()
            conn.send(message)
            self.last_profile['pipe_send_seconds'] += time.monotonic()-t
        def receive(conn):
            t = time.monotonic()
            message = conn.recv()
            self.last_profile['pipe_receive_seconds'] += time.monotonic()-t
            return message
        pending = iter(enumerate(actions))
        active, results = {}, {}
        def dispatch(worker):
            item = next(pending, None)
            if item is None:
                active.pop(worker, None)
                return
            index, action = item
            send(self._connections[worker], dict(kind='candidate', group=group, index=index,
                                                action=action, audit=audit))
            active[worker] = index
        try:
            for conn in self._connections:
                send(conn, dict(kind='root', group=group, before=before))
            attached = []
            for conn in self._connections:
                reply = self._receive(conn, 'root_ready')
                if reply['group'] != group or reply['before_hash'] != before.state_hash:
                    raise RuntimeError('Root cache identity mismatch')
                attached.append(reply['attach_seconds'])
            self.last_profile.update(root_cache_seconds=time.monotonic()-started,
                                     worker_attach_seconds=attached)
            for worker in range(len(self._connections)):
                dispatch(worker)
            while active:
                ready = wait([self._connections[w] for w in active], timeout=self.timeout)
                if not ready:
                    raise TimeoutError('Fixed branch group timed out')
                for conn in ready:
                    worker = self._connections.index(conn)
                    reply = receive(conn)
                    if reply.get('kind') != 'result':
                        raise RuntimeError('Fixed branch failed: ' + repr(reply))
                    index = reply['index']
                    if index != active[worker] or index in results or reply['group'] != group or reply['before_hash'] != before.state_hash:
                        raise RuntimeError('Mismatched fixed branch identity')
                    results[index] = reply
                    dispatch(worker)
            if sorted(results) != list(range(20)):
                raise RuntimeError('Incomplete candidate group')
            for conn in self._connections:
                send(conn, dict(kind='finish', group=group))
            self.last_profile['worker_geometry'] = []
            for conn in self._connections:
                reply = self._receive(conn, 'finished')
                if reply['group'] != group:
                    raise RuntimeError('Stale static audit')
                self.last_profile['worker_geometry'].append(reply['geometry'])
            self.last_profile['pool_seconds'] = time.monotonic()-started
            return [results[i] for i in range(20)]
        except BaseException:
            self.close(abort=True)
            raise
