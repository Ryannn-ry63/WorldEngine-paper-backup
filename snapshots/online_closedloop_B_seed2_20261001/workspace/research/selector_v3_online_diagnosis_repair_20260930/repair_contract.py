"""Check the separately frozen repair without changing the training registration."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_repair():
    path=ROOT/'repair_manifest.json'
    if not path.exists():raise RuntimeError('Repair has not been validated and frozen')
    registration=json.loads(path.read_text())
    for name,expected in registration['files'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Repair source changed: '+name)
    return registration
