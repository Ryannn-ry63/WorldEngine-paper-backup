"""Verify this archive using only Python's standard library, without cluster paths."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
WORK=ROOT/'workspace'
inventory=json.loads((ROOT/'source_inventory.json').read_text())
original=Path(inventory['source_workspace'])


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def archived(p):
    return WORK/Path(p).relative_to(original)


for name, record in inventory['files'].items():
    p=WORK/name
    assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],name

formal=WORK/'research/selector_v3_B_p6_formal_diagnostic_20261001'
contract=read(WORK/'research/selector_v3_online_diagnosis_20260930/registry/fulltable_data_contract.json')
summary=read(formal/'summary.json')
assert summary['complete']==16
statuses=list((formal/'jobs').glob('*/status.json'))
assert len(statuses)==16
for path in statuses:
    status=read(path)
    assert status['status']=='COMPLETE'
    p=archived(status['result'])
    assert sha(p)==status['result_sha256']
    result=read(p);mode=result['job']['mode']
    expected=set(contract['expected_tokens'][mode])
    assert result['counts']==len(expected)
    for kind, source in result['files'].items():
        p=archived(source)
        assert sha(p)==result['files_sha256'][kind]
        with p.open() as f:
            rows=list(csv.DictReader(f))
        aggregate='average' if mode.startswith('ol_') else 'overall_average'
        actual=[r for r in rows if r['token']!=aggregate]
        assert len(rows)==len(expected)+1 and len(actual)==len(expected)
        assert {r['token'] for r in actual}==expected
        metrics=list(result['means']) if kind=='pdm' else ['ade_4s','fde_4s']
        for r in actual:
            if kind=='pdm' and mode.startswith('ol_'):
                assert r['valid'].lower()=='true'
            for key in metrics:
                assert math.isfinite(float(r[key]))
                if kind=='pdm':
                    assert 0<=float(r[key])<=1
                    assert float(r[key])==result['rows'][r['token']][key]
        if kind=='pdm':
            for key in metrics:
                assert math.isclose(statistics.mean(float(r[key]) for r in actual),
                                    result['means'][key],rel_tol=0,abs_tol=1e-12)
            matching=[r for r in summary['rows'] if r['job']==result['job']['id']]
            assert len(matching)==1
            for key in metrics:
                assert math.isclose(matching[0][key],100*result['means'][key],abs_tol=1e-10)
print(f'PASS: {len(inventory["files"])} archived files; 16 formal jobs, exact coverage and metrics; no external assets needed for this verification.')
