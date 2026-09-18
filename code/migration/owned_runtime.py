"""Process-local legacy-path compatibility; all non-data backing lives in global_user.
No host mounts or historical artifact edits. PRoot is not a security sandbox.
"""
from pathlib import Path
import argparse,json,os,sys,subprocess,time
M=Path(__file__).resolve().parent;C=M.parent;W=C.parent;N=W.parent;O=N/'WorldEngine-owned'
OLD='/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180'
OLD2='/inspire/hdd2/project/roboticsystem2/wangcaojun-240208020180'
ALG=N.parent/'miniconda3/envs/algengine';SIM=Path('/root/miniconda3/envs/simengine')
DEPS=O/'dependencies/legacy_sources_20260918';OUT=O/'project_outputs/WorldEngine-paper'

def mappings():
 empty=O/'runtime/hidden_legacy_private';empty.mkdir(parents=True,exist_ok=True)
 return [(empty,OLD),(empty,OLD2),
  (DEPS/'nuplan-devkit',OLD+'/repo-wcj/nuplan-devkit'),
  (DEPS/'navsim_v1',OLD+'/repo-wcj/E2E/navsim_v1'),
  (DEPS/'mmcv',OLD+'/repo-wcj/mmcv'),
  (O/'models/diffusiondrive/epoch_100.pth',OLD+'/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth'),
  (O/'runtime/legacy_bashrc',OLD+'/dotfiles/.bashrc'),
  (O/'environments/simengine',str(SIM)),
  (OUT,OLD2+'/nry/WorldEngine-paper')]

def environment():
 env=os.environ.copy()
 # Do not inherit invisible project paths or placeholder values from login shells.
 support=C/'projects/SimEngine/scripts/diffusiondrive'
 env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
  PYTHONPATH=os.pathsep.join(map(str,[support/'algengine_worker_bootstrap',support,C,C/'projects/AlgEngine',C/'projects/SimEngine',N/'DiffusionDrive',N/'mmcv',DEPS/'nuplan-devkit',DEPS/'navsim_v1'])),
  PATH=os.pathsep.join([str(ALG/'bin'),'/usr/local/sbin','/usr/local/bin','/usr/sbin','/usr/bin','/sbin','/bin']),
  WORLDENGINE_ROOT=str(N/'WorldEngine'),ALGENGINE_ROOT=str(C/'projects/AlgEngine'),SIMENGINE_ROOT=str(C/'projects/SimEngine'),
  DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE=str(ALG),DIFFUSIONDRIVE_SIMENGINE_PYTHON=str(SIM/'bin/python'),
  STUDY_PYTHON=str(ALG/'bin/python'),NUPLAN_DEVKIT_ROOT=str(DEPS/'nuplan-devkit'),
  NUPLAN_DEVKIT_PATH=str(DEPS/'nuplan-devkit'),NAVSIM_DEVKIT_ROOT=str(N/'DiffusionDrive'),
  NAVSIM_EXP_ROOT=str(N/'exp'),NAVSIM_METRIC_CACHE_PATH=str(N/'exp/metric_cache_navtest'),
  NAVSIM_METRIC_CACHE_PATH_TRAIN=str(N/'exp/metric_cache_trainval'),NAVSIM_METRIC_CACHE_PATH_EVAL=str(N/'exp/metric_cache_navtest'),
  OPENSCENE_DATA_ROOT=str(N/'WorldEngine/data/raw/openscene-v1.1'),
  NUPLAN_MAPS_ROOT=str(N/'WorldEngine/data/raw/nuplan/dataset/maps'),NUPLAN_MAP_VERSION='nuplan-maps-v1.0',
  WORLDENGINE_DIFFUSIONDRIVE_MMCV_BOOTSTRAP='1',WORLDENGINE_DIFFUSIONDRIVE_GSPLAT_BOOTSTRAP='1',
  WORLDENGINE_MMCV_EXTENSION=str(N/'WorldEngine/artifacts/toolchains/mmcv_sm89_sm90_v1/mmcv/_ext.so'),
  WORLDENGINE_GSPLAT_EXTENSION=str(N/'WorldEngine/artifacts/toolchains/gsplat_sm89_sm90_v1/gsplat/csrc.so'),
  LD_LIBRARY_PATH=str(O/'tools/proot/usr/lib/x86_64-linux-gnu'),
  DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT=str(O/'models/diffusiondrive/epoch_100.pth'),
  DIFFUSIONDRIVE_SELECTOR_INIT_SHA256='1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514',
  CUBLAS_WORKSPACE_CONFIG=':4096:8',WORLDENGINE_OWNED_RUNTIME='1')
 for k in ('PYTHONHOME','RAY_ADDRESS','CONDA_PREFIX','CUDA_HOME','CUDA_PATH','LD_PRELOAD'):env.pop(k,None)
 return env

def command(args):
 if not (M/'owned_environment_ready.json').exists():raise RuntimeError('Environment copy/validation not finished; inspect migration logs')
 ready=json.loads((M/'owned_environment_ready.json').read_text())
 if ready['status'] not in ('READY_FOR_PREFLIGHT','PASS_ENVIRONMENT_FILES'):raise RuntimeError('Environment files not ready')
 pairs=mappings()
 for host,guest in pairs:
  if not host.exists():raise FileNotFoundError(str(host))
 exe=O/'tools/proot/usr/bin/proot'
 cmd=[str(exe),'-v','-1','-w',str(C)]
 for host,guest in pairs:cmd+=['-b',str(host)+':'+guest+'!']
 (O/'runtime/path_map.json').write_text(json.dumps([dict(backing=str(h),logical=g) for h,g in pairs],indent=2)+'\n')
 if args.mode in ('check','check-gpu'):
  if args.mode=='check-gpu' and (not args.device.isdigit()):raise ValueError('Select one numeric GPU ID')
  return cmd+[str(ALG/'bin/python'),str(M/'check_owned_runtime.py'),'--gpu',args.device if args.mode=='check-gpu' else 'none']
 if args.mode=='run':
  if ready['status']!='PASS_ENVIRONMENT_FILES':raise RuntimeError('Full environment file verification is pending')
  status=(OUT/'copy_status.txt').read_text()
  if not status.startswith('PASS_BYTE_COPY_ONLY'):raise RuntimeError('Personal experiment outputs are not fully copied and checksum-verified; refusing resume')
  report=O/'runtime/latest_gpu_check.json'
  if not report.exists() or json.loads(report.read_text())['status']!='PASS_NONDATA_GPU':raise RuntimeError('Run check-gpu on this instance before resuming')
  import socket
  if json.loads(report.read_text())['hostname']!=socket.gethostname():raise RuntimeError('GPU verification belongs to another instance')
  rest=args.arguments
  if rest and rest[0]=='--':rest=rest[1:]
  if not rest:raise ValueError('Specify a command after run --')
  return cmd+rest
 raise ValueError(args.mode)

def main():
 p=argparse.ArgumentParser();subs=p.add_subparsers(dest='mode',required=True)
 subs.add_parser('check').set_defaults(device='0',arguments=[])
 g=subs.add_parser('check-gpu');g.add_argument('--device',default='0');g.set_defaults(arguments=[])
 r=subs.add_parser('run');r.add_argument('arguments',nargs=argparse.REMAINDER);r.set_defaults(device='0')
 a=p.parse_args()
 env=environment();cmd=command(a)
 if a.mode=='check':env['CUDA_VISIBLE_DEVICES']=''
 elif a.mode=='check-gpu':env['CUDA_VISIBLE_DEVICES']=a.device
 os.execve(cmd[0],cmd,env)
if __name__=='__main__':
 try:main()
 except (RuntimeError,FileNotFoundError,ValueError) as exc:
  print('OWNED_RUNTIME_NOT_READY: '+str(exc),file=sys.stderr);sys.exit(2)
