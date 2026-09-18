"""Temporarily release the user's verified tmux GPU holder and always restore it."""
import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

def query(pane,field):
    return subprocess.check_output(['tmux','display-message','-p','-t',pane,field],text=True).strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holder-pid',type=int,required=True)
    parser.add_argument('--pane',required=True)
    parser.add_argument('--run',default='20260916_prefeedback_factorial_v1')
    parser.add_argument('--train-all',action='store_true')
    args=parser.parse_args()
    code=Path(__file__).resolve().parents[1];workspace=code.parent
    paths=json.loads((workspace/'registry/local_paths.json').read_text())
    old=Path(paths['legacy_root']).resolve();proc=Path('/proc')/str(args.holder_pid)
    command=proc.joinpath('cmdline').read_bytes().split(b'\0')
    command=[x.decode() for x in command if x]
    if command!=['python','occupy_GPU.py'] or proc.joinpath('cwd').resolve()!=old:
        raise RuntimeError('PID is not the authorized GPU holder')
    parent=next(int(line.split()[1]) for line in proc.joinpath('status').read_text().splitlines() if line.startswith('PPid:'))
    if int(query(args.pane,'#{pane_pid}'))!=parent:
        raise RuntimeError('Holder does not belong to the specified tmux shell')
    if query(args.pane,'#{pane_current_path}')!=str(old):
        raise RuntimeError('Holder shell working directory changed')
    entry=workspace/'registry/gpu_holder_lease.json'
    journal=dict(holder_pid=args.holder_pid,pane=args.pane,cwd=str(old),command=command,
                 started=time.time(),status='VERIFIED',run=args.run)
    def save():entry.write_text(json.dumps(journal,indent=2)+'\n')
    save();released=False;child=None
    def interrupted(signum,frame):
        raise KeyboardInterrupt('GPU supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        os.kill(args.holder_pid,signal.SIGINT)
        released=True
        for _ in range(100):
            if not proc.exists():break
            time.sleep(.2)
        if proc.exists():raise RuntimeError('GPU holder did not exit on SIGINT; no SIGKILL attempted')
        journal['status']='RELEASED';save()
        commands=[['validate','--device','cuda'],['smoke','--device','cuda']]
        if args.train_all:
            commands.extend(['train','--arm',a,'--seed',str(s),'--device','cuda'] for s in (0,1,2) for a in ('D1','D2','D3','D4'))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        for command in commands:
            journal['active_command']=command;save()
            child=subprocess.Popen(['bash','run_study.sh',*command,'--run',args.run],cwd=code,env=env,start_new_session=True)
            status=child.wait();child=None
            if status:raise RuntimeError(f'Study command failed with exit {status}: {command}')
        journal['study_status']='PASS'
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                journal['status']='CHILD_STILL_RUNNING';save()
                raise RuntimeError('Training has not exited; holder cannot safely be restored concurrently')
        if released:
            if proc.exists():
                journal['status']='ORIGINAL_HOLDER_STILL_RUNNING'
            else:
                if query(args.pane,'#{pane_current_command}') not in ('bash','zsh','sh'):
                    journal['status']='RESTORE_REQUIRES_IDLE_SHELL';save()
                    raise RuntimeError('Tmux pane is no longer idle; refusing to type into another process')
                restart='python occupy_GPU.py'
                subprocess.run(['tmux','send-keys','-t',args.pane,'-l',restart],check=True)
                subprocess.run(['tmux','send-keys','-t',args.pane,'Enter'],check=True)
                journal['status']='RESTORE_SENT'
                for _ in range(100):
                    if query(args.pane,'#{pane_current_command}')=='python':
                        journal['status']='RESTORED';break
                    time.sleep(.2)
        journal['finished']=time.time();journal.pop('active_command',None);save()
        print(json.dumps(journal),flush=True)
        if journal['status']!='RESTORED':raise RuntimeError('GPU holder restoration needs verification')

if __name__=='__main__':main()
