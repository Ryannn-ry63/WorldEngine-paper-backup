import argparse
import json
from pathlib import Path
from .common import read,run_path
from .specs import QUEUES

def main():
    p=argparse.ArgumentParser(description='Independent paper experiments; explicit stages and three-seed queues')
    p.add_argument('command',choices=['plan','prepare','train','export','prepare-eval','preflight','openloop','closed','report','queue','status','oracle','prepare-supervised','planner-train','mechanism','mechanism-collect','debug-six','source-audit'])
    p.add_argument('--queue',choices=list(QUEUES),default='B')
    p.add_argument('--run',default='20260917_paper_feedback_v1')
    p.add_argument('--training-run');p.add_argument('--arm');p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--device',default='cuda');p.add_argument('--devices',default='0,1,2,3')
    p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--policy');p.add_argument('--block',choices=['open_navtest','open_rare'],default='open_navtest')
    p.add_argument('--cohort',choices=['bridge','full','common'],default='bridge');p.add_argument('--mode',choices=['NR','R'],default='NR')
    p.add_argument('--stage',choices=['train','eval','all'],default='all')
    p.add_argument('--manifest',type=Path)
    a=p.parse_args();run=run_path(a.run)
    if a.command=='plan':result=dict(queues=QUEUES,existing_task_unchanged=True,resource_limit=None)
    elif a.command=='prepare':
        from .data import prepare
        m=prepare(run,a.queue);result=dict(status='PREPARED',run=str(run),queue=m['queue'],arms=m['arms'])
    elif a.command=='train':
        if not a.arm:p.error('--arm required')
        from .train import feedback_training,cached_training
        fn=feedback_training if read(run/'manifest.json')['queue']=='B' else cached_training
        result=fn(run,a.arm,a.seed,a.device,a.smoke,a.stop_after)
    elif a.command=='export':
        from .export import export_one
        result=export_one(run,a.arm,a.seed)
    elif a.command=='prepare-eval':
        from .evaluation import prepare
        result=prepare(run,run_path(a.training_run) if a.training_run else None)
    elif a.command=='preflight':
        from .evaluation import preflight
        result=preflight(run,a.devices)
    elif a.command=='openloop':
        from .evaluation import openloop
        result=openloop(run,a.policy,a.block,a.devices)
    elif a.command=='closed':
        from .evaluation import closed
        result=closed(run,a.policy,a.cohort,a.mode,a.devices)
    elif a.command=='report':
        from .report import report
        result=report(run)
    elif a.command=='queue':
        from .queue import execute
        result=execute(a)
    elif a.command=='prepare-supervised':
        from .supervised import prepare
        result=prepare(run)
    elif a.command=='planner-train':
        from .supervised import train
        result=train(run,a.arm,a.seed,a.devices,a.smoke)
    elif a.command=='mechanism':
        from .mechanism import prepare
        result=prepare(run,a.manifest,a.smoke)
    elif a.command=='mechanism-collect':
        from .mechanism import collect
        result=collect(run,a.devices,a.smoke)
    elif a.command=='debug-six':
        from .debug import execute
        result=execute(a.devices)
    elif a.command=='source-audit':
        from .audit import sources
        result=sources(run)
    elif a.command=='oracle':
        from .oracle import analyze
        from .oracle import common_cache_entries
        m=read(run/'manifest.json') if (run/'manifest.json').exists() else dict(common_caches=common_cache_entries());result=[]
        for j,e in enumerate(m['common_caches']):result.append(analyze(e['cache'],run/'oracle'/f'noise{j}.json'))
    else:
        result=dict(run=str(run),training={str(x.parent.name):read(x) for x in run.glob('checkpoints/*/progress.json')},
                    accepted_closed=len(list(run.glob('collections/*/accepted.json'))),accepted_open=len(list(run.glob('openloop/*/*/audit.json'))))
    if result is not None:print(json.dumps(result,ensure_ascii=False,indent=2,default=str),flush=True)
if __name__=='__main__':main()
