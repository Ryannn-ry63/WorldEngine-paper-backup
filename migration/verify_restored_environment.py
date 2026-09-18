from pathlib import Path
import subprocess,json,time
M=Path(__file__).resolve().parent;D=M.parents[2]/'WorldEngine-owned/environments'
with (M/'simengine_restore_verified.log').open('w') as f:
 r=subprocess.run(['tar','--compare','--file',str(D/'simengine_20260918.tar'),'--directory',str(D),'--exclude=*.pyc','--exclude=*/__pycache__/*'],stdout=f,stderr=subprocess.STDOUT)
result=dict(status='PASS_RESTORED_ENVIRONMENT_FILES' if r.returncode==0 else 'FAIL_RESTORED_ENVIRONMENT_FILES',exit_code=r.returncode,finished=time.time(),excludes=['*.pyc','*/__pycache__/*'],note='Compare extracted environment with previously hash-verified archive; no GPU or data check implied')
(M/'simengine_restore_verified.json').write_text(json.dumps(result,indent=2)+'\n')
if r.returncode==0:
 p=M/'owned_environment_ready.json';d=json.loads(p.read_text());d.update(status='PASS_ENVIRONMENT_FILES',scope='Existing personal algengine retained; restored simengine compared to archived files; copied source dependencies already checksum-verified');p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(result),flush=True)
