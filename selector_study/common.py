import hashlib
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
LEGACY = CODE / 'projects/AlgEngine/scripts/diffusiondrive'

def legacy_imports():
    if str(LEGACY) not in sys.path:
        sys.path.insert(0, str(LEGACY))

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def artifact(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p))

def verify(item):
    p = Path(item['path'])
    if sha(p) != item['sha256']:
        raise RuntimeError('Artifact changed: ' + str(p))
    return p

def write(path, value, immutable=True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if immutable and p.exists():
        if read(p) != value:
            raise RuntimeError('Refusing to change frozen JSON: ' + str(p))
        return
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, p)

def code_identity():
    # Freeze both the new driver and inherited runtime dependencies.
    files = sorted(list((CODE/'selector_study').glob('*.py')) + list((CODE/'projects').rglob('*.py')))
    manifest = {str(p.relative_to(CODE)): sha(p) for p in files}
    return dict(sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(), files=manifest)

def inside(path, parent):
    return Path(parent).resolve() in Path(path).resolve().parents

def output_run(workspace, name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Run ID must contain only letters, numbers, underscore or hyphen')
    workspace = Path(workspace).resolve()
    if CODE != workspace/'code':
        raise RuntimeError('Workspace must own this independent code checkout')
    return workspace/'runs'/name
