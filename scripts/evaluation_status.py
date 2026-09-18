"""Read verified per-condition progress without treating training as evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selector_study.common import CODE, artifact, read, verify


def status(name):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Invalid run ID')
    run=CODE.parent/'runs'/name
    inputs=read(run/'inputs.json');contract=artifact(run/'run_contract.json')
    groups={}
    for cohort in ('bridge','full','common'):
        conditions={}
        for policy in inputs['models']:
            for mode in ('NR','R'):
                key=f'{cohort}_{mode}_{policy}';folder=run/'collections'/key
                if (folder/'accepted.json').exists():
                    accepted=read(folder/'accepted.json');a=read(verify(accepted['audit']))
                    co=read(verify(a['collection']))
                    if (accepted['status']!='PASS' or a['status']!='PASS' or co['id']!=key
                            or co['contract']!=contract or co['checkpoint']!=inputs['models'][policy]
                            or co['rows']!=inputs['cohorts'][cohort]
                            or (cohort=='bridge' and policy=='scalar_v3' and not accepted['scalar_historical_parity'])):
                        raise RuntimeError('Acceptance provenance drift: '+key)
                    state='ACCEPTED'
                elif (folder/'audit.json').exists():state='ACTION_AUDIT_EXISTS_ACCEPTANCE_PENDING'
                elif (folder/'collection.json').exists():state='STARTED_NOT_ACCEPTED'
                else:state='NOT_STARTED'
                conditions[key]=state
        groups[cohort]=dict(accepted=sum(v=='ACCEPTED' for v in conditions.values()),
            expected=len(conditions),conditions=conditions)
    pre=run/'preflight/report.json'
    preflight=read(pre)['status'] if pre.exists() else 'NOT_COMPLETED'
    attempt=read(run/'attempt_status.json') if (run/'attempt_status.json').exists() else None
    lease=read(run/'gpu_lease.json') if (run/'gpu_lease.json').exists() else None
    return dict(run=str(run),preflight=preflight,groups=groups,attempt_status=attempt,gpu_lease=lease,
        openloop='NOT_IMPLEMENTED_IN_THIS_DRIVER',
        formal_table_ready=False,independent_blind_test=False,
        note='STARTED_NOT_ACCEPTED does not imply a process is still running. Check logs and GPU lease; training status is separate.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default='20260916_factorial_eval_v2')
    print(json.dumps(status(p.parse_args().run),ensure_ascii=False,indent=2))
