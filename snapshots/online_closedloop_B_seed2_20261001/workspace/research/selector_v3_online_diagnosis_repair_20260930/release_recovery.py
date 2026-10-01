"""Archive the diagnosed stop only after strict local replay and resume validation."""
import contextlib,json,os,shutil,sys,time
from pathlib import Path
from repair_contract import verify_repair,sha
ROOT=Path(__file__).resolve().parent
OLD=ROOT.with_name('selector_v3_online_diagnosis_20260930')
sys.path.insert(0,str(OLD/'handoff'))
from experiment_common import lock,write
from run_job import registration

def release():
 verify_repair();registration()
 receipt=ROOT/'evidence/recovery_release.json'
 if receipt.exists():
  print('Recovery already released; no stop flag changed');return
 validation=json.loads((ROOT/'evidence/validation.json').read_text())
 if validation['status']!='PASS':raise RuntimeError('Repair validation incomplete')
 for path,expected in validation['evidence_sha256'].items():
  if sha(ROOT/path)!=expected:raise RuntimeError('Validation evidence changed: '+path)
 audit=json.loads((ROOT/'evidence/resume_audit.json').read_text())
 if audit['status']!='PASS' or len(audit['jobs'])!=6:raise RuntimeError('Six verified jobs required')
 registry=OLD/'registry';stop=registry/'correctness_failure.json'
 archived=ROOT/'evidence/pre_recovery';archived.mkdir(exist_ok=True)
 with contextlib.ExitStack() as stack:
  stack.enter_context(lock(ROOT/'recovery.lock'))
  for name in sorted([x['job'] for x in audit['jobs']]+['roots48']):
   stack.enter_context(lock(registry/'jobs'/name/'job.lock'))
  root=json.loads((registry/'jobs/roots48/status.json').read_text())
  if root.get('status')!='COMPLETE' or root.get('repair_manifest_sha256')!=sha(ROOT/'repair_manifest.json'):
   raise RuntimeError('Repaired roots48 must complete on H100 before release')
  failure=json.loads(stop.read_text())
  if failure.get('job')!='roots48' or failure.get('matched')!=['failed to reproduce the exact root']:
   raise RuntimeError('An unexpected correctness stop must be investigated separately')
  for job in audit['jobs']:
   base=registry/'jobs'/job['job']
   if sha(base/'training.progress.json')!=job['progress_sha256']:raise RuntimeError('Progress changed after audit')
   state=json.loads((base/'status.json').read_text())
   if state['status']!='CORRECTNESS_STOPPED_AT_BOUNDARY':raise RuntimeError('Job not stopped')
   target=archived/job['job'];target.mkdir(exist_ok=True)
   for name in ('status.json','training.progress.json','resolved.json'):
    shutil.copy2(base/name,target/name)
  shutil.copy2(registry/'jobs/roots48/status.json',archived/'roots48_status.json')
  shutil.copy2(registry/'campaign_clock.json',archived/'campaign_clock.json')
  archived_stop=archived/'correctness_failure.json'
  if archived_stop.exists() and archived_stop.read_bytes()!=stop.read_bytes():raise RuntimeError('Conflicting stop archive')
  shutil.copy2(stop,archived_stop)
  value=dict(status='RELEASED_AFTER_VALIDATED_DIAGNOSTIC_REPAIR',released_unix=time.time(),
   original_experiment=str(OLD),repair_manifest_sha256=sha(ROOT/'repair_manifest.json'),
   archived_stop=str(archived_stop),archived_stop_sha256=sha(archived_stop),
   validation_sha256=sha(ROOT/'evidence/validation.json'),resume_audit_sha256=sha(ROOT/'evidence/resume_audit.json'),
   original_sources_unchanged=True,training_checkpoints_unchanged=True,
   deadline_extended=False,required_before_train='repaired roots48 COMPLETE on H100')
  write(receipt,value,immutable=True)
  # Only this confirmed diagnostic incident is cleared. Any future stop remains fatal.
  stop.unlink()
 print(json.dumps(value),flush=True)
if __name__=='__main__':release()
