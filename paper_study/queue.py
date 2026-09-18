"""One isolated four-card queue; subprocess boundaries also isolate runtime imports."""
from .hardware import hardware as check_hardware
import os
import signal
import subprocess
import sys
import time
from .common import CODE, legacy_imports, lock, read, run_path, write
from .specs import FEEDBACK, STRUCTURES

def child(arguments,log,env=None):
    print('START', ' '.join(arguments), 'log='+str(log),flush=True)
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        p=subprocess.Popen([sys.executable,'-m','paper_study',*arguments],cwd=CODE,env=env or os.environ.copy(),
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:time.sleep(1)
            if p.returncode:raise RuntimeError('Task failed; inspect '+str(log))
        except BaseException:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise

def parallel(jobs):
    children=[]
    try:
        for arguments,log,env in jobs:
            log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('a')
            try:p=subprocess.Popen([sys.executable,'-m','paper_study',*arguments],cwd=CODE,env=env,
                                   stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            except BaseException:stream.close();raise
            children.append((p,stream,log))
            print('START', ' '.join(arguments), 'gpu='+env.get('CUDA_VISIBLE_DEVICES',''), 'log='+str(log),flush=True)
        while any(p.poll() is None for p,_,_ in children):
            for p,_,log in children:
                if p.poll() not in (None,0):raise RuntimeError('Training failed; inspect '+str(log))
            time.sleep(1)
        for p,_,log in children:
            if p.returncode:raise RuntimeError('Training failed; inspect '+str(log))
    finally:
        for p,_,_ in children:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for p,stream,_ in children:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL);p.wait()
            stream.close()


def execute(args):
    legacy_imports()
    import run_study_bridges_4h100 as four
    ids=four.devices(args.devices);check_hardware(ids)
    allocation=four.resources(four.cpu_capacity())
    run=run_path(args.run)
    def interrupted(*_):raise KeyboardInterrupt('Paper queue interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    from .storage import ensure_run,stopped
    ensure_run(run)
    with lock(run/'queue.lock'):
        write(run/'attempt_status.json',dict(status='RUNNING',queue=args.queue,started=time.time(),devices=ids),False)
        try:
            if args.queue=='A':
                from .evaluation import prepare
                prepare(run,allocation=allocation)
                child(['source-audit','--run',run.name],run/'logs/source_audit.log')
                child(['oracle','--run',run.name],run/'logs/oracle.log')
                _evaluate(run,args.devices,open_only=True)
            elif args.queue in ('B','C','D','F'):
                from .data import prepare
                if args.stage!='eval':prepare(run,args.queue)
                arms=(list(FEEDBACK) if args.queue=='B' else list(STRUCTURES) if args.queue=='C' else
                      ['selector_sft','selector_grpo_matched','planner_l2','planner_bc'] if args.queue=='D' else ['paired_common','rarelog','rarerollout'])
                if args.stage!='eval':
                    if args.queue=='D':child(['prepare-supervised','--run',run.name],run/'logs/prepare_supervised.log')
                    tasks=[(arm,seed) for seed in (0,1,2) for arm in arms if not arm.startswith('planner_')]
                    # Four separate processes: each owns one GPU and its own optimizer/RNG/checkpoint.
                    for offset in range(0,len(tasks),4):
                        batch=tasks[offset:offset+4]
                        jobs=[]
                        for slot,(arm,seed) in enumerate(batch):
                            base=['--run',run.name,'--arm',arm,'--seed',str(seed),'--device','cuda']
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=ids[slot])
                            jobs.append((['train',*base,'--smoke'],run/'logs'/f'smoke_{arm}_seed{seed}.log',env))
                        parallel(jobs)
                        parallel([(arguments[:-1],run/'logs'/f'{arm}_seed{seed}.log',env)
                                  for (arguments,_,env),(arm,seed) in zip(jobs,batch)])
                        for arm,seed in batch:
                            child(['export','--run',run.name,'--arm',arm,'--seed',str(seed)],run/'logs'/f'export_{arm}_seed{seed}.log')
                    for seed in (0,1,2):
                        for arm in [a for a in arms if a.startswith('planner_')]:
                            child(['planner-train','--run',run.name,'--arm',arm,'--seed',str(seed),'--devices',args.devices,'--smoke'],
                                  run/'logs'/f'smoke_{arm}_seed{seed}.log')
                            child(['planner-train','--run',run.name,'--arm',arm,'--seed',str(seed),'--devices',args.devices],
                                  run/'logs'/f'{arm}_seed{seed}.log')
                if args.stage!='train':
                    from .evaluation import prepare as prepare_eval
                    ev=run_path(run.name+'_eval');prepare_eval(ev,run,allocation)
                    _evaluate(ev,args.devices)
            else:
                from .mechanism import prepare,collect
                prepare(run,args.manifest);collect(run,args.devices)
            write(run/'attempt_status.json',dict(status='COMPLETED_REQUESTED_STAGE',queue=args.queue,finished=time.time()),False)
        except BaseException as exc:
            stopped(run,exc)
            raise
    return dict(status='COMPLETED_REQUESTED_STAGE',run=str(run))

def _evaluate(run,devices,open_only=False):
    child(['preflight','--run',run.name,'--devices',devices],run/'logs/preflight.log')
    models=read(run/'inputs.json')['models']
    if not open_only:
        for policy in models:
            for mode in ('NR','R'):
                child(['closed','--run',run.name,'--policy',policy,'--cohort','bridge','--mode',mode,'--devices',devices],
                      run/'logs'/f'bridge_{mode}_{policy}.log')
    for policy in models:
        for block in ('open_navtest','open_rare'):
            child(['openloop','--run',run.name,'--policy',policy,'--block',block,'--devices',devices],run/'logs'/f'{block}_{policy}.log')
        if not open_only:
            for cohort in ('full','common'):
                for mode in ('NR','R'):
                    child(['closed','--run',run.name,'--policy',policy,'--cohort',cohort,'--mode',mode,'--devices',devices],
                          run/'logs'/f'{cohort}_{mode}_{policy}.log')
    child(['report','--run',run.name],run/'logs/report.log')
