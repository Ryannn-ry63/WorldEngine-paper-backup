#!/usr/bin/env python3
"""Run independent ready diagnostics/evaluations on this one eight-GPU node."""
import sys
from pathlib import Path
REPAIR = Path(__file__).resolve().parent
ORIGINAL = REPAIR.with_name("selector_v3_online_diagnosis_20260930")
sys.path.insert(0, str(ORIGINAL / "handoff"))
import argparse
import os
import time
from pathlib import Path
from experiment_common import REGISTRY,read
from repair_run import registration,campaign_clock,ready,run
from summarize_experiment import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',choices=('evaluation','diagnostic','all'),default='all')
    p.add_argument('--devices',default=os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3,4,5,6,7'))
    p.add_argument('--once',action='store_true');p.add_argument('--check-only',action='store_true')
    p.add_argument('--idle-seconds',type=float,default=120)
    a=p.parse_args();spec=registration()
    if a.idle_seconds<0:p.error('idle-seconds must be nonnegative')
    kinds={'evaluation'} if a.group=='evaluation' else {'diagnostic','diagnostic_extra'} if a.group=='diagnostic' else {'evaluation','diagnostic','diagnostic_extra'}
    jobs=[j for j in spec['jobs'] if j['kind'] in kinds]
    if a.check_only:
        print('CHECK_ONLY: eligible jobs:',[j['id'] for j in jobs]);return
    deadline=campaign_clock()['deadline_unix']
    idle_since=time.monotonic()
    while time.time()<deadline:
        if (REGISTRY/'correctness_failure.json').exists():raise RuntimeError('A correctness check stopped the campaign')
        ran=False
        try:summarize()
        except BlockingIOError:pass
        for job in jobs:
            status=REGISTRY/'jobs'/job['id']/'status.json'
            state=read(status).get('status') if status.exists() else 'NOT_STARTED'
            if state in ('COMPLETE','FAILED') or not ready(job):continue
            try:run(job,a.devices,wait=False)
            except BlockingIOError:continue
            except Exception as error:
                print('JOB_FAILED:',job['id'],repr(error),flush=True)
                if (REGISTRY/'correctness_failure.json').exists():raise
            ran=True;idle_since=time.monotonic();break
        if a.once and ran:return
        if not ran:
            if time.monotonic()-idle_since>=a.idle_seconds:
                print('QUEUE_IDLE_EXIT: no ready unclaimed work; relaunch when checkpoints are ready',flush=True)
                return
            print('QUEUE_WAITING: no unlocked ready job; finished/failed jobs will not be silently rerun',flush=True)
            time.sleep(20)
    try:summarize()
    except BlockingIOError:pass
    print('CAMPAIGN_DEADLINE: no further jobs started',flush=True)
if __name__=='__main__':main()
