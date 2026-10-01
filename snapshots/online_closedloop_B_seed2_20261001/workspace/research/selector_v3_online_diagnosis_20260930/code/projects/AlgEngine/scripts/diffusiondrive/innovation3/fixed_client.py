"""Persistent simulator process owner; render and CPU children share its lifetime."""
import os
import signal
import socket
import subprocess
from .image_transport import ImageBuffer
from .transport import Channel


class FixedWorkerClient:
    def __init__(self, cfg, workers, backend, log_path, env):
        self.backend = backend
        self.workers = workers
        self.child = self.log = self.images = self.channel = None
        self.closed = False
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.channel = Channel(left, timeout=1800)
            self.images = ImageBuffer()
            self.log = log_path.open('x')
            command = [cfg['simengine_python'], '-u', '-m', 'innovation3.fixed_worker',
                '--fd', str(right.fileno()), '--images-fd', str(self.images.fd),
                '--workers', str(workers), '--backend', backend]
            self.child = subprocess.Popen(command, env=env, pass_fds=(right.fileno(), self.images.fd),
                                          stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            right.close()
            ready = self.channel.receive()
            if ready.get('kind') != 'resident_ready' or ready['backend'] != backend:
                raise RuntimeError('Simulator startup failed: ' + repr(ready))
        except BaseException:
            self.close(abort=True)
            raise
        finally:
            right.close()

    def begin(self, cfg, output, steps, version, attempts, audit_every=32, mode='train'):
        if self.closed or self.child.poll() is not None:
            raise RuntimeError('Simulator process is not alive')
        self.channel.send(dict(kind='episode', settings=cfg, output=str(output), steps=steps,
                               policy_version=version, attempts=attempts, audit_every=audit_every, mode=mode))
        return self.channel.receive()

    def close(self, abort=False):
        if self.closed:
            return
        self.closed = True
        try:
            if self.child is not None and self.child.poll() is None and not abort:
                self.channel.send(dict(kind='shutdown'))
                if self.channel.receive() != dict(kind='shutdown_complete'):
                    raise RuntimeError('Simulator shutdown was not acknowledged')
                if self.child.wait(timeout=30) != 0:
                    raise RuntimeError('Simulator failed on shutdown')
        finally:
            # The renderer owns CPU spawn workers. Reap the entire private group
            # on exceptions instead of leaving orphan simulation processes.
            if self.child is not None:
                try:
                    os.killpg(self.child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.child.wait()
            if self.channel is not None:
                self.channel.sock.close()
            if self.images is not None:
                self.images.close()
            if self.log is not None:
                self.log.close()
