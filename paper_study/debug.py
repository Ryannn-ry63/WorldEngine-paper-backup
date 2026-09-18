"""One four-H100 instance checks six queues; all outputs are isolated diagnostics."""
from .hardware import hardware as check_hardware
import hashlib
import json
import os
import signal
import socket
import time
from .common import artifact,code_identity,legacy_imports,lock,read,run_path,write
from .queue import child,parallel
from .specs import FEEDBACK,STRUCTURES
from .storage import ensure_run,stopped

def execute(devices):
    legacy_imports()
    import run_study_bridges_4h100 as four
    ids=four.devices(devices);hardware=check_hardware(ids)
    allocation=four.resources(four.cpu_capacity())
    identity=code_identity();version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:10]
    root=run_path('20260917_paper_debug_'+version);ensure_run(root)
    def interrupt(*_):raise KeyboardInterrupt('Debug queue interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    with lock(root/'debug.lock'):
        write(root/'contract.json',dict(code=identity,hardware=hardware,devices=devices,allocation=allocation,diagnostic_only=True))
        write(root/'attempt_status.json',dict(status='RUNNING',hostname=socket.gethostname(),pid=os.getpid(),started=time.time()),False)
        print('DEBUG_ROOT='+str(root),flush=True)
        results={}
        def operation(arguments,label):
            child(arguments,root/'logs'/(label+'.log'))
        def stage(name,function):
            print('CHECK_START '+name,flush=True)
            try:
                function();results[name]=dict(status='PASS_CHECKS',formal_experiments_complete=False)
            except Exception as exc:
                results[name]=dict(status='FAILED',error=str(exc))
                print('CHECK_FAILED '+name+': '+str(exc),flush=True)
            write(root/'summary.json',dict(status='IN_PROGRESS',checks=results,debug_only=True),False)
        def cached(queue,arms):
            run=run_path(root.name+'_'+queue)
            operation(['prepare','--run',run.name,'--queue',queue],queue+'_prepare')
            if queue=='D':operation(['prepare-supervised','--run',run.name],'D_expert_labels')
            for offset in range(0,len(arms),4):
                jobs=[]
                for slot,arm in enumerate(arms[offset:offset+4]):
                    arguments=['train','--run',run.name,'--arm',arm,'--seed','0','--device','cuda','--smoke']
                    jobs.append((arguments,root/'logs'/f'{queue}_{arm}.log',dict(os.environ,CUDA_VISIBLE_DEVICES=ids[slot])))
                parallel(jobs)
            if queue=='D':
                for arm in ('planner_l2','planner_bc'):
                    operation(['planner-train','--run',run.name,'--arm',arm,'--seed','0','--devices',devices,'--smoke'],'D_'+arm)
        try:
            stage('B',lambda:cached('B',list(FEEDBACK)))
            stage('C',lambda:cached('C',list(STRUCTURES)))
            stage('F',lambda:cached('F',['paired_common','rarelog','rarerollout']))
            stage('D',lambda:cached('D',['selector_sft','selector_grpo_matched']))
            ev=run_path(root.name+'_A')
            def evaluation_checks():
                operation(['prepare-eval','--run',ev.name],'A_prepare')
                operation(['preflight','--run',ev.name,'--devices',devices],'A_preflight')
                for policy in ('scalar_v3','D4_seed0'):
                    for mode in ('NR','R'):
                        operation(['closed','--run',ev.name,'--policy',policy,'--cohort','bridge','--mode',mode,'--devices',devices],f'A_bridge_{policy}_{mode}')
                # Actual official rare open-loop path, on the full registered 288 tokens.
                operation(['openloop','--run',ev.name,'--policy','scalar_v3','--block','open_rare','--devices',devices],'A_open_rare')
                operation(['source-audit','--run',ev.name],'A_source_audit')
            stage('A',evaluation_checks)
            def mechanism_checks():
                run=run_path(root.name+'_E')
                operation(['mechanism','--run',run.name,'--smoke'],'E_prepare_probe')
                operation(['mechanism-collect','--run',run.name,'--devices',devices,'--smoke'],'E_sentinel_and_branch')
            stage('E',mechanism_checks)
            result=dict(status='PASS_DEBUG_CHECKS' if all(v['status']=='PASS_CHECKS' for v in results.values()) else 'FAILED_CHECKS',
                        checks=results,debug_only=True,formal_experiments_complete=False,finished=time.time(),root=str(root))
            write(root/'summary.json',result,False);write(root/'attempt_status.json',result,False)
            print(json.dumps(result,indent=2),flush=True)
            if result['status']!='PASS_DEBUG_CHECKS':raise RuntimeError('Some checks failed; see '+str(root/'summary.json'))
            return result
        except BaseException as exc:
            stopped(root,exc);raise
