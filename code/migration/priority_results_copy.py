"""Single-writer, resumable priority migration. Never delete source/destination files.
Priority verification is NOT permission to resume an incomplete experiment.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
M=Path(__file__).resolve().parent
DEFAULT_SOURCE=Path('/inspire/hdd2/project/roboticsystem2/wangcaojun-240208020180/nry/WorldEngine-paper')
DEFAULT_DEST=M.parents[2]/'WorldEngine-owned/project_outputs/WorldEngine-paper'
CORE_DIRS={'native','checkpoints','runtime','configs','data','scenarios','preflight'}
PRIORITY_RUNS=[f'20260917_paper_{q}_v3{s}' for q in 'ABCDEF' for s in ('','_eval')]

def atomic(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
 with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def priority_paths(source):
 # Exact roots first; no traversal of historical images merely to find checkpoints.
 result=[]
 for name in PRIORITY_RUNS:
  root=source/'runs'/name
  if not root.is_dir():continue
  for p in sorted(root.iterdir()):
   if not p.is_dir() or p.is_symlink() or p.name in CORE_DIRS:
    result.append(str(p.relative_to(source))+('/' if p.is_dir() and not p.is_symlink() else ''))
 return result

class Copier:
 def __init__(self,source,dest,audit):
  self.source=source;self.dest=dest;self.audit=audit;self.child=None;self.has_lock=False
  self.state=dict(status='STARTING',source=str(source),destination=str(dest),started=time.time(),single_writer=True,completed_units=[],phase=None)
 def save(self,**fields):
  self.state.update(fields,updated=time.time());atomic(self.audit/'priority_copy_status.json',self.state)
  (self.dest/'copy_status.txt').write_text(self.state['status']+' '+str(self.state.get('phase'))+' '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')
 def call(self,args,log):
  with log.open('a') as f:
   self.child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   rc=self.child.wait();self.child=None
  if rc:raise RuntimeError(f'rsync exit {rc}; see {log}')
 def unit(self,name,paths,complete_run=None):
  listing=self.audit/(name+'.files0');listing.write_bytes(b''.join(os.fsencode(p)+b'\0' for p in sorted(set(paths))))
  self.save(status='COPYING_PRIORITY' if name=='critical' else 'COPYING_UNIT',phase=name)
  log=self.audit/(name+'.copy.log');verify=self.audit/(name+'.verify.log')
  # -r is explicit because --files-from changes rsync -a recursion defaults.
  base=['rsync','-aHr','--from0','--files-from='+str(listing),'--partial','--partial-dir=.rsync-partial']
  self.call(base+['--stats',str(self.source)+'/',str(self.dest)+'/'],log)
  self.save(status='CHECKSUM_VERIFY_UNIT',phase=name)
  verify.write_text('')
  self.call(base+['--checksum','--dry-run','--itemize-changes','--out-format=%i %n%L',str(self.source)+'/',str(self.dest)+'/'],verify)
  if verify.stat().st_size:raise RuntimeError('Source/destination differences: '+str(verify))
  # Record exact selected roots and checkpoint byte hashes for critical tier.
  cert=dict(status='PASS_PRIORITY_BYTES_NOT_RESUME' if name=='critical' else 'PASS_UNIT_BYTES_NOT_RESUME',name=name,completed=time.time(),file_list=str(listing),file_list_sha256=sha(listing),copy_log=str(log),verify_log=str(verify),complete_run=complete_run,source_unchanged_required=True)
  if name=='critical':
   models=[]
   for run in PRIORITY_RUNS:
    for folder in ('native','checkpoints'):
     root=self.source/'runs'/run/folder
     if not root.exists():continue
     for parent,dirs,files in os.walk(root,followlinks=False):
      dirs[:]=[d for d in dirs if not Path(parent,d).is_symlink()]
      for fn in files:
       p=Path(parent,fn)
       if p.suffix not in ('.pt','.pth','.ckpt') or p.is_symlink():continue
       target=self.dest/p.relative_to(self.source);a=sha(p);b=sha(target)
       if a!=b:raise RuntimeError('Checkpoint SHA256 mismatch: '+str(p))
       models.append(dict(relative=str(p.relative_to(self.source)),bytes=p.stat().st_size,sha256=a))
   cert['checkpoint_files']=models;cert['checkpoint_bytes']=sum(x['bytes'] for x in models)
  atomic(self.audit/(name+'.accepted.json'),cert)
  self.state['completed_units'].append(name);self.save(status='UNIT_VERIFIED',phase=name)
  print(json.dumps(dict(event='UNIT_VERIFIED',name=name,complete_run=complete_run)),flush=True)
 def run(self):
  self.dest.mkdir(parents=True,exist_ok=True);self.audit.mkdir(parents=True,exist_ok=True)
  with (self.dest/'copy.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.has_lock=True
   self.unit('critical',priority_paths(self.source))
   # Whole queues are copied, including consumed images/frames and failure evidence.
   for name in PRIORITY_RUNS:
    if (self.source/'runs'/name).is_dir():self.unit(name,['runs/'+name+'/'],complete_run=name)
   if (self.source/'reports').is_dir():self.unit('reports',['reports/'])
   for p in sorted((self.source/'runs').iterdir()):
    if p.name not in PRIORITY_RUNS:self.unit('archive_'+p.name,['runs/'+p.name+('/' if p.is_dir() and not p.is_symlink() else '')],complete_run=p.name)
   # Final *read-only* checksum of all source entries. No second write pass can
   # overwrite a target after per-unit verification. No resumed jobs allowed yet.
   self.save(status='FINAL_CHECKSUM_VERIFY',phase='all_runs_reports')
   check=self.audit/'all.verify.log';check.write_text('')
   for part in ('runs','reports'):
    self.call(['rsync','-aHcn','--exclude=.rsync-partial/','--itemize-changes','--out-format=%i %n%L',str(self.source/part)+'/',str(self.dest/part)+'/'],check)
   if check.stat().st_size:raise RuntimeError('Final verification differs; source may have changed: '+str(check))
   self.save(status='PASS_BYTE_COPY_ONLY',phase='all_runs_reports',finished=time.time())

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=DEFAULT_SOURCE);p.add_argument('--dest',type=Path,default=DEFAULT_DEST);p.add_argument('--audit',type=Path,default=M/'priority_copy_20260918');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 if a.source.resolve()==a.dest.resolve() or a.source.resolve() in a.dest.resolve().parents:raise ValueError('Invalid destination overlap')
 if a.plan_only:
  print(json.dumps(dict(source=str(a.source),dest=str(a.dest),priority_roots=priority_paths(a.source),run_order=PRIORITY_RUNS),indent=2));return
 c=Copier(a.source,a.dest,a.audit)
 def stop(sig,frame):raise KeyboardInterrupt('Migration interrupted')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:c.run()
 except BaseException as exc:
  if c.child is not None:
   os.killpg(c.child.pid,signal.SIGTERM)
   try:c.child.wait(timeout=30)
   except subprocess.TimeoutExpired:
    os.killpg(c.child.pid,signal.SIGKILL);c.child.wait()
  # A lock conflict must not overwrite the active copier's status.
  if c.has_lock:c.save(status='STOPPED',reason=repr(exc))
  raise
if __name__=='__main__':main()
