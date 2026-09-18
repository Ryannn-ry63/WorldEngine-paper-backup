"""Copy small source dependencies; no environment or frozen-run mutations."""
from pathlib import Path
import json,subprocess,time,fcntl
M=Path(__file__).resolve().parent;N=M.parents[2]
OLD=Path('/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj')
DEST=N/'WorldEngine-owned/dependencies/legacy_sources_20260918'
DEST.mkdir(parents=True,exist_ok=True)
with (DEST/'backup.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 entries=[]
 try:
  for name,relative in [('nuplan-devkit','nuplan-devkit'),('navsim_v1','E2E/navsim_v1'),('mmcv','mmcv')]:
   src=OLD/relative;dst=DEST/name;dst.mkdir(exist_ok=True)
   with (M/f'backup_source_{name}.log').open('a') as out:
    subprocess.run(['rsync','-aH','--partial','--stats',str(src)+'/',str(dst)+'/'],stdout=out,stderr=subprocess.STDOUT,check=True)
   verify=subprocess.run(['rsync','-aHcn','--itemize-changes',str(src)+'/',str(dst)+'/'],capture_output=True,text=True,check=True)
   (M/f'backup_source_{name}_verify.log').write_text(verify.stdout+verify.stderr)
   if verify.stdout or verify.stderr:raise RuntimeError('Source/destination differ: '+name)
   entry=dict(name=name,source=str(src),destination=str(dst),status='PASS_BYTE_COPY_ONLY')
   entries.append(entry);print(json.dumps(entry),flush=True)
   (M/'runtime_source_backups_20260918.json').write_text(json.dumps(dict(status='COPYING',entries=entries),indent=2))
  result=dict(status='PASS_SOURCE_BACKUPS_ONLY',finished=time.time(),entries=entries,note='Source trees byte-checked. Symlinks preserved, environments and frozen paths unchanged; no claim of runnable migration.')
 except Exception as exc:
  result=dict(status='STOPPED',error=repr(exc),entries=entries);raise
 finally:
  (M/'runtime_source_backups_20260918.json').write_text(json.dumps(result,indent=2)+'\n')
