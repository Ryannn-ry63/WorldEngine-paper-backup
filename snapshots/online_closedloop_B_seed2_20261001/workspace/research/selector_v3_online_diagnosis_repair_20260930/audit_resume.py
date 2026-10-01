"""Read-only original boundary/optimizer provenance verification for all six jobs."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.with_name('selector_v3_online_diagnosis_20260930')
sys.path.insert(0,str(OLD/'handoff'))
from experiment_common import sha,write
from run_job import registration
from innovation3.ddp_session_boundary import open_boundary
from innovation3.online_probe import source_hashes

def main():
 registration()
 sources=source_hashes();results=[]
 for path in sorted((OLD/'registry/jobs').glob('train_*/training.progress.json')):
  state=json.loads(path.read_text());spec=state['contract'];started=time.monotonic()
  assert spec['source_sha256']==sources
  assert spec['settings_sha256']==sha(OLD/'registry/runtime.json')
  assert spec['manifest_sha256']==sha(OLD/'registry/scene_manifest.json')
  assert state['completed']<456  # makes open_boundary strictly read-only: no marker publication
  before=sha(path)
  verified=open_boundary(path.parent/'training',spec,True)
  assert sha(path)==before
  value=dict(job=path.parent.name,completed=verified['completed'],target=spec['episodes'],
   rounds=len(verified['rounds']),checkpoint_count=verified['completed'],progress_sha256=before,
   all_rank_last_checkpoints=[x['checkpoint'] for x in verified['rounds'][-1]],
   status='PASS_ORIGINAL_RESUME_BOUNDARY',elapsed_seconds=time.monotonic()-started)
  results.append(value);print(json.dumps({k:v for k,v in value.items() if k!='all_rank_last_checkpoints'}),flush=True)
 write(ROOT/'evidence/resume_audit.json',dict(status='PASS',jobs=results,training_source_unchanged=True))
if __name__=='__main__':main()
