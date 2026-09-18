"""Read-only dependency checks on a new platform project; no GPUs or data writes."""
from pathlib import Path
import json
import os
import socket

CODE=Path(__file__).resolve().parents[1]
registry=json.loads((CODE.parent/'registry/local_paths.json').read_text())
legacy=Path(registry['legacy_root'])
source=CODE.parent/'runs/20260916_factorial_bridges_4h100_v1'
items=[('shared_code','dir',CODE),('legacy_data','dir',legacy/'data'),
       ('training_python','file',Path(registry['training_python'])),
       ('simulator_python','file',Path(os.environ.get('DIFFUSIONDRIVE_SIMENGINE_PYTHON','/root/miniconda3/envs/simengine/bin/python'))),
       ('raw_images','dir',legacy/'data/raw/openscene-v1.1/sensor_blobs/trainval'),
       ('raw_test_images','dir',legacy/'data/raw/openscene-v1.1/sensor_blobs/test'),
       ('annotations','dir',legacy/'data/alg_engine/merged_infos_navformer'),
       ('metric_cache_navtest','dir',legacy.parent/'exp/metric_cache_navtest'),
       ('metric_cache_trainval','dir',legacy.parent/'exp/metric_cache_trainval'),
       ('mmcv_cuda','file',legacy/'artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so'),
       ('gsplat_cuda','file',legacy/'artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so')]
try:
    inputs=json.loads((source/'inputs.json').read_text())
    items += [('source_'+k,'file',Path(v['path'])) for k,v in inputs['artifacts'].items()]
    items += [('assets_'+k,'dir',Path(v)) for k,v in inputs['assets'].items()]
except (OSError,ValueError) as exc:
    print('SOURCE_INPUTS_UNREADABLE',str(exc));items.append(('source_inputs','file',source/'inputs.json'))
storage=json.loads((CODE.parent/'registry/paper_storage.json').read_text())
items += [('output_runs','dir',Path(storage['runs_root'])),('output_reports','dir',Path(storage['reports_root']))]
for q in 'BCDEF':items.append(('saved_run_'+q,'dir',CODE.parent/'runs'/f'20260917_paper_{q}_v3'))
print('HOST',socket.gethostname())
failed=[]
for label,kind,p in items:
    try:
        if kind=='file':
            with p.open('rb') as f:f.read(1)
        else:
            with os.scandir(p) as it:next(it,None)
        print('OK',label,str(p),'->',str(p.resolve()))
    except OSError as exc:
        failed.append(label)
        print('FAIL',label,str(p),'->',str(p.resolve()),repr(exc))
print('READ_ONLY_CHECK',json.dumps(dict(failed=failed,count=len(failed),note='No write tests, full-file hashes or GPU compatibility checks performed')))
raise SystemExit(1 if failed else 0)
