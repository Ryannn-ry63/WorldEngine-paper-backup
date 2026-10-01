from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import sys,importlib.util
h=Path('/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/hotfix_logit_tolerance_20261001')
o=h.parent.with_name('selector_v3_online_diagnosis_20260930')
sys.path[:0]=[str(h/'overlay'),str(o/'code/projects/AlgEngine/scripts/diffusiondrive')]
s=importlib.util.spec_from_file_location('hotfix_runtime',h/'overlay/innovation3_runtime.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
commands=[]
def capture(command,env,**kw):
 assert env['PYTHONPATH'].split(':')[0]==str(h/'overlay')
 if '-c' in command: compile(command[command.index('-c')+1],'<probe>','exec')
 commands.append(command)
 return SimpleNamespace(returncode=0)
for timing,seed in [('per_step',2),('after_rollout8',1)]:
 args=['runtime','--settings',str(o/'registry/runtime.json'),'--scene-manifest',str(o/'registry/scene_manifest.json'),'--output',str(h/'test_output'),'--devices','0,1,2,3,4,5,6,7','--episodes','904','--steps','8','--seed',str(seed),'--split','train','--branch-workers','4','--fixed-segment','--wall-seconds','43200','--update-timing',timing,'--resume','ddp-online-session-gate']
 with patch.object(sys,'argv',args),patch('innovation3.fixed_manifest.load_assignments',return_value=({'steps':8},[])),patch.object(m.subprocess,'run',side_effect=capture):
  assert m.main()==0
 cmd=commands[-1]
 assert '--resume' in cmd and cmd[cmd.index('--update-timing')+1]==timing
 assert cmd[cmd.index('--seed')+1]==str(seed)
 assert cmd[cmd.index('--nproc_per_node')+1]=='8'
 print('PASS_LAUNCHER_CONSTRUCTION',timing,flush=True)
