"""Build a private, byte-preserving B seed 2 research snapshot (no tensors)."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

SOURCE = Path(__file__).resolve().parents[2]
DEST = Path(sys.argv[1]).resolve()
SNAP = DEST / 'snapshots/online_closedloop_B_seed2_20261001'
WORK = SNAP / 'workspace'
EXP = SOURCE / 'research/selector_v3_online_diagnosis_20260930'
REPAIR = SOURCE / 'research/selector_v3_online_diagnosis_repair_20260930'
FORMAL = SOURCE / 'research/selector_v3_B_p6_formal_diagnostic_20261001'
ANALYSIS = SOURCE / 'research/online_closedloop_analysis_20261001'
records = {}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def copy(p):
    p = Path(p).absolute()
    if p.is_symlink() or not p.is_file():
        raise ValueError('Expected a regular source file: '+str(p))
    if p.suffix in {'.pt','.pth','.ckpt','.pkl','.so','.log'}:
        raise ValueError('Excluded payload: '+str(p))
    rel = p.relative_to(SOURCE)
    dst = WORK / rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    before = sha(p)
    shutil.copy2(p,dst)
    if sha(dst) != before or sha(p) != before:
        raise ValueError('Source changed during backup: '+str(p))
    records[str(rel)] = dict(sha256=before,bytes=p.stat().st_size)


def tree(root, allowed=None):
    for p in sorted(root.rglob('*')):
        if any(x in p.relative_to(root).parts for x in ('.git','__pycache__','.pytest_cache','logs')):
            continue
        if p.is_file() and (allowed is None or p.suffix in allowed):
            copy(p)


SNAP.mkdir(parents=True,exist_ok=False)
tree(EXP/'code')
tree(EXP/'handoff',{'.py','.sh','.md'})
for p in (EXP/'registry').glob('*.json'):
    copy(p)
for name in ('prepared.json','frozen_eval_config.py'):
    copy(EXP/'registry/development'/name)
for group in ('A','B'):
    for seed in range(3):
        job=EXP/'registry/jobs'/f'train_{group}_s{seed}'
        for name in ('status.json','training.coverage3.json','training.coverage6.json'):
            copy(job/name)
        if group=='B':
            copy(EXP/'registry/policies'/f'train_B_s{seed}_p6/prepared.json')
copy(EXP/'registry/jobs/train_B_s2/training.progress.json')
marker=read(EXP/'registry/jobs/train_B_s2/training.coverage6.json')
for row in marker['final_records']:
    p=Path(row['report'])
    if sha(p)!=row['report_sha256']:
        raise ValueError('Training endpoint report changed')
    copy(p)

# Repair source, registered overlays, manifests and compact release evidence.
tree(REPAIR,{'.py','.sh','.md'})
for folder in (REPAIR, REPAIR/'evidence', REPAIR/'storage_recovery_20261001',
               REPAIR/'hotfix_logit_tolerance_20261001',
               REPAIR/'hotfix_logit_tolerance_bs2_20261001'):
    for p in folder.glob('*.json'):
        copy(p)
for folder in (REPAIR/'hotfix_logit_tolerance_20261001/evidence',
               REPAIR/'hotfix_logit_tolerance_bs2_20261001/evidence'):
    for p in folder.glob('*.json'):
        copy(p)

for p in FORMAL.iterdir():
    if p.is_file() and p.suffix in {'.py','.sh','.md','.json','.csv'}:
        copy(p)
tree(FORMAL/'fix_preflight_stop_marker_20261001',{'.py','.json'})
statuses=[]
for p in sorted((FORMAL/'jobs').glob('*/status.json')):
    status=read(p)
    if status['status']!='COMPLETE':
        raise ValueError('Formal job incomplete')
    result=read(Path(status['result']))
    if sha(Path(status['result']))!=status['result_sha256']:
        raise ValueError('Formal result changed')
    copy(p);copy(Path(status['result']))
    attempt=Path(status['attempt'])
    for name in ('evaluation_inputs.json','scratch.json','scratch_released.json',
                 'evidence_archive.json','evaluation/process_status.json','evaluation/commands.json'):
        copy(attempt/name)
    for key,path in result['files'].items():
        if sha(Path(path))!=result['files_sha256'][key]:
            raise ValueError('Formal metric changed')
        copy(Path(path))
    statuses.append(status)
if len(statuses)!=16:
    raise ValueError('Expected sixteen completed jobs')

tree(ANALYSIS,{'.py','.md','.json','.csv','.png'})
for name in ('1001_online_closedloop_analysis_handoff.md',
             'docs/selector_v3_online_reward_optimization_20260928.md',
             'research/github_B_seed2_backup_20261001/package.py'):
    copy(SOURCE/name)

external={k:v for k,v in read(FORMAL/'manifest.json')['policies'].items()}
(SNAP/'MODEL_REFERENCES.json').write_text(json.dumps(dict(
    weights_included=False,policies=external,
    restore_note='Model tensors, training traces, datasets, maps, images, simulator assets and environments remain external; paths are original cluster paths.'),indent=2)+'\n')
(SNAP/'source_inventory.json').write_text(json.dumps(dict(
    source_workspace=str(SOURCE),file_count=len(records),
    total_bytes=sum(v['bytes'] for v in records.values()),files=records),indent=2)+'\n')
print(json.dumps(dict(snapshot=str(SNAP),files=len(records),
                      MiB=sum(v['bytes'] for v in records.values())/2**20)))
