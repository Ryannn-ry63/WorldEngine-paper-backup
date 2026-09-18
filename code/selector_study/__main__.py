import argparse
import json
from pathlib import Path
from .common import CODE,read,output_run

def main():
    p=argparse.ArgumentParser(description='Audited results and prefeedback factorial study')
    p.add_argument('command',choices=['export','prepare','validate','smoke','train','status'])
    p.add_argument('--workspace',type=Path,default=CODE.parent)
    p.add_argument('--run',default='20260916_prefeedback_factorial_v1')
    p.add_argument('--arm',choices=['D1','D2','D3','D4'])
    p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();workspace=args.workspace.resolve();run=output_run(workspace,args.run)
    if args.command=='export':
        from .results import export
        result=export(workspace,read(workspace/'registry/local_paths.json'))
    elif args.command=='prepare':
        from .data import prepare
        result=prepare(run,read(workspace/'registry/local_paths.json'))
        result={k:result[k] for k in ('status','method','training','membership')}
    elif args.command=='validate':
        from .validate import validate
        result=validate(run,args.device)
    elif args.command=='smoke':
        from .validate import smoke_suite
        result=smoke_suite(run,args.device)
    elif args.command=='train':
        if args.arm is None:p.error('train requires --arm')
        from .train import run_training
        result=run_training(run,args.arm,args.seed,args.device)
    else:
        entries={}
        for arm in ('D1','D2','D3','D4'):
            for seed in (0,1,2):
                path=run/'checkpoints'/f'{arm}_seed{seed}'/'progress.json'
                entries[f'{arm}_seed{seed}']=read(path) if path.exists() else dict(status='NOT_STARTED')
        result=dict(run=str(run),models=entries,evaluation='NOT_RUN')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
