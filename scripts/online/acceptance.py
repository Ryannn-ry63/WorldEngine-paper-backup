"""Checksummed exports and exact-coverage evaluation acceptance."""
import csv
import json
import math
from pathlib import Path
import pickle
import statistics
import time
from common import read, sha

METRICS=('no_at_fault_collisions','drivable_area_compliance','ego_progress',
         'time_to_collision_within_bound','comfort','score')
SAFETY=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def coverage(path,expected,aggregate,valid=False,metrics=METRICS):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    actual=[r for r in rows if r['token']!=aggregate];mean=[r for r in rows if r['token']==aggregate]
    tokens=[r['token'] for r in actual]
    if len(mean)!=1 or len(tokens)!=len(set(tokens)) or set(tokens)!=set(expected) or len(expected)!=len(set(expected)):
        raise ValueError('Missing/duplicate/unexpected tokens: '+str(path))
    for row in rows:
        if valid and row.get('valid','').lower()!='true':raise ValueError('Invalid score row: '+row['token'])
        if not all(math.isfinite(float(row[k])) for k in metrics):raise ValueError('Non-finite score')
        if metrics==METRICS and not all(0<=float(row[k])<=1 for k in metrics):raise ValueError('Invalid PDM range')
    return {r['token']:{k:float(r[k]) for k in metrics} for r in actual}


def stage_files(out,mode):
    out=Path(out)
    if mode.startswith('ol_'):
        files=list(out.with_name(out.name+'_inputs').glob('test/*_official_pdms/pdm_scores_merged.csv'))
        if len(files)!=1:raise ValueError('Expected exactly one official PDM csv')
        result=dict(pdm=files[0])
        if mode=='ol_navtest':result['ade']=Path(str(files[0].parent).removesuffix('_official_pdms')+'.csv')
        return result
    return dict(pdm=out/'simulation/WE_output/openscene_format'/('all_scenes_pdm_averages_'+mode[3:]+'.csv'))


def accept_stage(out,mode,expected):
    if read(Path(out)/'process_status.json')['status']!='PASS_PROCESSES_ONLY':raise ValueError('Processes failed')
    files=stage_files(out,mode)
    rows=coverage(files['pdm'],expected,'average' if mode.startswith('ol_') else 'overall_average',mode.startswith('ol_'))
    if 'ade' in files:coverage(files['ade'],expected,'average',metrics=('ade_4s','fde_4s'))
    return dict(status='PASS_EXACT_COVERAGE',rows=rows,counts=len(rows),
        means={k:statistics.mean(r[k] for r in rows.values()) for k in METRICS},
        files={k:str(v) for k,v in files.items()},files_sha256={k:sha(v) for k,v in files.items()})
