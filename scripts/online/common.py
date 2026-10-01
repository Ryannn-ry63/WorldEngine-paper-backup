"""Atomic records, namespace locks and checked local-scratch archiving."""
from contextlib import contextmanager
import hashlib,json,os,shutil,socket,time,uuid
from pathlib import Path
TRANSIENT = {"sensor_blobs", "frames", "merged_ann_files", "distributed_collect"}

def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value, immutable=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('x') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        if immutable:
            os.link(tmp, p)
        else:
            os.replace(tmp, p)
        fd = os.open(str(p.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def directory_lock(path):
    """Atomic namespace lock, independent of per-client flock mount behavior.

    Never steal another host's lock based on its PID or elapsed time. A killed
    owner leaves evidence; explicit inspection is required before removal.
    """
    lockdir = Path(str(path) + '.mkdir')
    lockdir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lockdir.mkdir()
    except FileExistsError:
        raise BlockingIOError('Already claimed: ' + str(lockdir))
    token = uuid.uuid4().hex
    owner = dict(token=token, host=socket.gethostname(), pid=os.getpid(), started=time.time())
    try:
        write(lockdir / 'owner.json', owner, immutable=True)
        yield owner
    finally:
        p = lockdir / 'owner.json'
        if p.exists() and read(p).get('token') == token:
            p.unlink()
            lockdir.rmdir()


def archive_tree(src, dst, failures_only=False):
    """Retain metrics, plans, metadata and logs; disposable images/IPC stay local."""
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return []
    copied = []
    for folder, dirs, files in os.walk(src, followlinks=False):
        dirs[:] = [d for d in dirs if d not in TRANSIENT and not (Path(folder) / d).is_symlink()]
        for name in files:
            p = Path(folder) / name
            if p.is_symlink():
                continue
            if failures_only and p.suffix not in ('.log', '.json', '.yaml', '.yml', '.csv', '.txt'):
                continue
            rel = p.relative_to(src)
            q = dst / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
            # Metrics, metadata and executed plans are evidence; byte verify.
            h = sha(p)
            if sha(q) != h:
                raise RuntimeError('Evidence copy hash mismatch: ' + str(rel))
            copied.append(dict(path=str(q), sha256=h, size=p.stat().st_size))
    return copied
