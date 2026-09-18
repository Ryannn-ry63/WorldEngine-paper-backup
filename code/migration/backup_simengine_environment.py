"""Archive local simulator environment without modifying or activating it."""
from pathlib import Path
import hashlib,json,os,subprocess,time,shutil,fcntl
M=Path(__file__).resolve().parent;N=M.parents[2]
D=N/'WorldEngine-owned/environments';D.mkdir(parents=True,exist_ok=True)
archive=D/'simengine_20260918.tar';partial=D/'simengine_20260918.tar.partial';status=M/'simengine_archive_status.json'
def save(value):status.write_text(json.dumps(value,indent=2)+'\n')
with (D/'simengine_backup.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert shutil.disk_usage(D).free>500*1024**3,'Keep reserve for results migration'
 assert not archive.exists() and not partial.exists(),'Inspect existing archive before retry'
 try:
  save(dict(status='ARCHIVING',source='/root/miniconda3/envs/simengine',destination=str(archive),started=time.time()))
  subprocess.run(['tar','-cpf',str(partial),'-C','/root/miniconda3/envs','simengine'],check=True)
  subprocess.run(['tar','-tf',str(partial)],stdout=subprocess.DEVNULL,check=True)
  h=hashlib.sha256()
  with partial.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
   os.fsync(f.fileno())
  partial.rename(archive)
  save(dict(status='PASS_ARCHIVE_CREATED_NOT_RESTORED',finished=time.time(),source='/root/miniconda3/envs/simengine',destination=str(archive),bytes=archive.stat().st_size,sha256=h.hexdigest(),note='Tar listing passed; archive hash recorded. Symlinks retained. Original prefix and editable dependencies require restore validation. Not conda-pack; no source/restored comparison or runtime validation yet.'))
 except Exception as exc:
  save(dict(status='STOPPED',reason=repr(exc),partial=str(partial)));raise
