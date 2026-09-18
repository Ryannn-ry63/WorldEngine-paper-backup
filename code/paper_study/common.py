import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

CODE = Path(__file__).resolve().parents[1]
WORKSPACE = CODE.parent

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def artifact(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))

def verify(entry):
    p = Path(entry['path'])
    if sha(p) != entry['sha256']: raise RuntimeError('Artifact changed: '+str(p))
    return p

def write(path, value, immutable=True):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and immutable:
        if read(p) != value: raise RuntimeError('Frozen artifact collision: '+str(p))
        return
    fd, name = tempfile.mkstemp(prefix=p.name+'.', dir=p.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False); f.write('\n')
        os.replace(name, p)
    finally:
        if os.path.exists(name): os.unlink(name)

def safe_name(name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Run/policy ID must use letters, digits, underscore or hyphen')
    return name

def run_path(name): return WORKSPACE/'runs'/safe_name(name)

def paths(): return read(WORKSPACE/'registry/local_paths.json')

def legacy_imports():
    for p in (CODE/'projects/AlgEngine/scripts/diffusiondrive', CODE/'scripts'):
        if str(p) not in sys.path: sys.path.insert(0, str(p))

def code_identity():
    files = sorted((CODE/'paper_study').glob('*.py'))+[CODE/'run_paper_study.sh']
    return {str(p.relative_to(CODE)):sha(p) for p in files if p.exists()}

@contextlib.contextmanager
def lock(path):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Task already running: '+str(p)) from None
        yield

def torch_save(path, value):
    import torch
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=p.name+'.', dir=p.parent); os.close(fd)
    try: torch.save(value, name); os.replace(name, p)
    finally:
        if os.path.exists(name): os.unlink(name)
