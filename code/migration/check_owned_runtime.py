"""Non-data validation inside the owned path view. No training/evaluation output writes."""
from pathlib import Path
import argparse,hashlib,json,os,socket,subprocess,sys,time,signal
M=Path(__file__).resolve().parent;C=M.parent;N=C.parent.parent;O=N/'WorldEngine-owned';R=O/'runtime'
ALG=N.parent/'miniconda3/envs/algengine';SIM=Path('/root/miniconda3/envs/simengine')

def child_probe(kind,out):
 import importlib
 import torch
 if kind=='algengine':
  import mmcv,mmcv.ops,mmdet,mmdet3d,nuplan,navsim
  import mmdet3d_plugin
  from mmcv import Config
  config=Config.fromfile(str(C/'projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo_selector.py'))
  mods=['torch','mmcv','mmcv._ext','mmdet','mmdet3d','nuplan','navsim','mmdet3d_plugin']
 else:
  import ray,hydra,nuplan,navsim,gsplat
  from worldengine.runner.builders.env_builder import build_envs
  import gsplat.cuda._backend
  mods=['torch','ray','hydra','nuplan','navsim','gsplat','gsplat.csrc']
 origins={name:str(Path(importlib.import_module(name).__file__).resolve()) for name in mods}
 # Old private directories are hidden even while the original project remains mounted.
 for root in ('/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180','/inspire/hdd2/project/roboticsystem2/wangcaojun-240208020180'):
  assert not Path(root,'setup.sh').exists(),'Old private tree leaked into compatibility view'
 pairs=json.loads((R/'path_map.json').read_text())
 def backing(logical):
  for e in sorted(pairs,key=lambda x:len(x['logical']),reverse=True):
   if logical==e['logical'] or logical.startswith(e['logical']+'/'):return e['backing']+logical[len(e['logical']):]
  return logical
 checks={}
 for name,path in origins.items():
  physical=backing(path)
  assert physical.startswith(str(N.parent)) and '/project/' not in physical,(name,path,physical)
  s=Path(path).stat();t=Path(physical).stat()
  assert (s.st_ino,s.st_dev)==(t.st_ino,t.st_dev),(path,physical)
  checks[name]=dict(logical=path,backing=physical)
 executable=str(Path(sys.executable).absolute());physical=backing(executable)
 assert (Path(executable).stat().st_ino,Path(executable).stat().st_dev)==(Path(physical).stat().st_ino,Path(physical).stat().st_dev)
 # Basic Torch CPU execution and child inheritance. CUDA is separately tested.
 assert torch.equal(torch.arange(4).square(),torch.tensor([0,1,4,9]))
 p=subprocess.run([sys.executable,'-c','import torch;print(torch.arange(3).sum().item())'],capture_output=True,text=True,timeout=120)
 assert p.returncode==0 and p.stdout.strip()=='3',(p.stdout,p.stderr)
 Path(out).write_text(json.dumps(dict(status='PASS_IMPORTS_AND_BACKING',kind=kind,torch=torch.__version__,cuda=torch.version.cuda,prefix=sys.prefix,executable=executable,physical_executable=physical,modules=checks),indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',default='none');p.add_argument('--probe');p.add_argument('--output');a=p.parse_args()
 if a.probe:return child_probe(a.probe,a.output)
 stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid());folder=R/'checks'/stamp;folder.mkdir(parents=True)
 result=dict(status='RUNNING',hostname=socket.gethostname(),started=time.time(),gpu=a.gpu,steps=[],data_verified=False,old_results_resume_verified=False)
 def run(name,cmd,timeout=240):
  print('CHECK',name,flush=True)
  with (folder/(name+'.log')).open('w') as f:
   cp=subprocess.Popen(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:cp.wait(timeout=timeout)
   except BaseException:
    if cp.poll() is None:
     os.killpg(cp.pid,signal.SIGTERM)
     try:cp.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(cp.pid,signal.SIGKILL);cp.wait()
    raise
  result['steps'].append(dict(name=name,exit_code=cp.returncode,log=str(folder/(name+'.log'))))
  if cp.returncode:raise RuntimeError('Failed '+name+'; see '+str(folder/(name+'.log')))
 try:
  for name,prefix in [('algengine',ALG),('simengine',SIM)]:run(name,[prefix/'bin/python',__file__,'--probe',name,'--output',folder/(name+'.json')])
  baseline=O/'models/diffusiondrive/epoch_100.pth';old=Path('/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth')
  assert old.stat().st_ino==baseline.stat().st_ino
  assert str(old.resolve())==str(old),'Historical identity must remain stable'
  h=hashlib.sha256()
  with old.open('rb') as f:
   for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
  assert h.hexdigest()=='1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514'
  result['baseline_sha256']=h.hexdigest()
  if a.gpu!='none':
   support=C/'projects/SimEngine/scripts/diffusiondrive'
   run('mmcv_cuda',[ALG/'bin/python',support/'preflight_mmcv_cuda.py','--extension',os.environ['WORLDENGINE_MMCV_EXTENSION']])
   run('gsplat_cuda',[SIM/'bin/python',support/'preflight_gsplat_cuda.py','--extension',os.environ['WORLDENGINE_GSPLAT_EXTENSION']])
   run('ray_gpu',[SIM/'bin/python',M/'check_owned_ray.py'],timeout=240)
  result['status']='PASS_NONDATA_CPU' if a.gpu=='none' else 'PASS_NONDATA_GPU'
 except Exception as exc:
  result.update(status='FAIL_NONDATA',error=repr(exc));raise
 finally:
  result['finished']=time.time();result['directory']=str(folder)
  (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
  (R/('latest_cpu_check.json' if a.gpu=='none' else 'latest_gpu_check.json')).write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
