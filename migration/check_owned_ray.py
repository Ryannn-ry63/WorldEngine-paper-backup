"""One bounded Ray actor: checks worker spawning and gsplat without data."""
import os,tempfile
import ray

def worker():
 import sys
 from pathlib import Path
 from gsplat_multiarch import load_isolated_gsplat
 from preflight_gsplat_cuda import run_kernel
 extension=os.environ['WORLDENGINE_GSPLAT_EXTENSION']
 load_isolated_gsplat(extension)
 result=run_kernel(extension)
 assert not Path('/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/setup.sh').exists()
 result['python']=sys.executable
 return result

if __name__=='__main__':
 root=tempfile.mkdtemp(prefix='owned-ray-')
 ray.init(num_cpus=2,num_gpus=1,include_dashboard=False,object_store_memory=100*1024**2,_temp_dir=root)
 try:
  task=ray.remote(num_cpus=1,num_gpus=1)(worker)
  print('PASS_OWNED_RAY_GPU',ray.get(task.remote(),timeout=120),flush=True)
 finally:ray.shutdown()
