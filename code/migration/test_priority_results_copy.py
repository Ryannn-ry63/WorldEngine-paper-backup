"""Real rsync fixtures: data retention, partial continuation, corruption and locking."""
import importlib.util,fcntl,json,os,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;SCRIPT=HERE/'priority_results_copy.py'
spec=importlib.util.spec_from_file_location('priority',SCRIPT);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
with tempfile.TemporaryDirectory(prefix='priority-copy-test-') as t:
 root=Path(t);src=root/'source';dst=root/'dest';audit=root/'audit'
 def put(rel,content):
  p=src/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content);return p
 model=put('runs/20260917_paper_B_v3/checkpoints/x/resume.pt',b'resume and optimizer state')
 put('runs/20260917_paper_B_v3/manifest.json',b'{"frozen":true}')
 put('runs/20260917_paper_B_v3_eval/runtime/code.py',b'pass\n')
 put('runs/20260917_paper_B_v3_eval/collections/c/split_0/frames/1.pkl',b'essential raw record')
 put('runs/20260917_paper_B_v3_eval/collections/c/split_0/WE_output/sensor_blobs/1.jpg',b'image evidence')
 put('runs/old_failed/debug.log',b'failed attempt preserved')
 put('reports/table.tsv',b'score\t1\n')
 link=src/'runs/20260917_paper_B_v3_eval/runtime/data';link.symlink_to('/missing/data/is-preserved')
 dst.mkdir();(dst/'extra-user-file').write_text('keep')
 # Seed interrupted-copy output; rerun must preserve all source and target extras.
 p=dst/model.relative_to(src);p.parent.mkdir(parents=True);p.write_bytes(model.read_bytes());os.utime(p,ns=(model.stat().st_atime_ns,model.stat().st_mtime_ns))
 cmd=[sys.executable,str(SCRIPT),'--source',str(src),'--dest',str(dst),'--audit',str(audit)]
 cp=subprocess.run(cmd,capture_output=True,text=True);assert cp.returncode==0,cp.stderr
 assert json.loads((audit/'priority_copy_status.json').read_text())['status']=='PASS_BYTE_COPY_ONLY'
 assert (dst/'extra-user-file').read_text()=='keep'
 for p in src.rglob('*'):
  q=dst/p.relative_to(src)
  if p.is_symlink():assert q.is_symlink() and os.readlink(p)==os.readlink(q)
  elif p.is_file():assert p.read_bytes()==q.read_bytes()
 critical=json.loads((audit/'critical.accepted.json').read_text());assert len(critical['checkpoint_files'])==1
 items=(audit/'critical.files0').read_bytes();assert b'collections' not in items
 # Competing writer must fail without changing status or logs of active owner.
 before=(audit/'priority_copy_status.json').read_bytes()
 with (dst/'copy.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  cp=subprocess.run(cmd,capture_output=True,text=True);assert cp.returncode!=0
 assert (audit/'priority_copy_status.json').read_bytes()==before
 # Same-size, same-mtime corruption must never be accepted by rsync quick check.
 target=dst/model.relative_to(src);st=target.stat();target.write_bytes(b'X'*st.st_size);os.utime(target,ns=(st.st_atime_ns,st.st_mtime_ns))
 cp=subprocess.run(cmd,capture_output=True,text=True);assert cp.returncode!=0
 assert json.loads((audit/'priority_copy_status.json').read_text())['status']=='STOPPED'
 assert model.read_bytes()==b'resume and optimizer state'
 print('PASS: priority ordering, all-file retention, partial continuation, symlink preservation, extra-file preservation, single-writer lock, same-size corruption rejection, source unchanged')
