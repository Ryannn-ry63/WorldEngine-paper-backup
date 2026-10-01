"""Immutable experiment artifacts and isolated subprocess environments."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

WORK = Path(__file__).resolve().parents[1]
CODE = WORK / 'code'
TOOLS = CODE / 'projects/AlgEngine/scripts/diffusiondrive'
REGISTRY = WORK / 'registry'
SETTINGS = REGISTRY / 'runtime.json'
sys.path.insert(0, str(TOOLS))
from selector_eval_runtime_20260930 import environment


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(text)
    try:
        if immutable:
            # link is an atomic create-if-absent; it never replaces a winner.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hashes():
    return {str(p.relative_to(WORK)):sha(p) for root in (CODE, WORK/'handoff')
            for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in
            ('.py', '.sh', '.yaml', '.yml', '.npy') and
            not any(x in p.parts for x in ('__pycache__', '.pytest_cache', 'tmp', 'tests'))}


def configured_environment(devices='0', job_dir=None):
    cfg, env = environment(SETTINGS, devices)
    if job_dir:
        temp = Path('/tmp')/('wev3_'+hashlib.sha256(str(Path(job_dir).resolve()).encode()).hexdigest()[:16])
        temp.mkdir(parents=True, exist_ok=True)
        env.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp),
                   RAY_TMPDIR=str(temp/'ray'), TORCH_EXTENSIONS_DIR=str(temp/'torch_extensions'))
    for key in ('RAY_ADDRESS', 'ip_head', 'redis_password', 'num_nodes'):
        env.pop(key, None)
    env['PYTHONPATH'] = str(WORK/'handoff') + os.pathsep + env['PYTHONPATH']
    return cfg, env


def command(argv, env=None, dry=False, cwd=CODE):
    argv = list(map(str, argv))
    print(json.dumps(dict(command=argv, cwd=str(cwd))), flush=True)
    if not dry:
        subprocess.run(argv, cwd=cwd, env=env, check=True)


def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a')
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        f.close()
        raise
    return f
