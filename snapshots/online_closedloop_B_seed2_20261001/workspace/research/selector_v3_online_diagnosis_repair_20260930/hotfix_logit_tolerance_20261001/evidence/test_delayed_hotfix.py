from pathlib import Path
from unittest.mock import patch
import sys
h=Path('/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/research/selector_v3_online_diagnosis_repair_20260930/hotfix_logit_tolerance_20261001')
c=h.parent.with_name('selector_v3_online_diagnosis_20260930')/'code'
sys.path[:0]=[str(h/'overlay'),str(c/'projects/AlgEngine/scripts/diffusiondrive'),str(c/'tests/innovation3')]
import innovation3.learner as l
from innovation3.fixed_learning import FixedLiveLearning
import torch
from test_online_learner import model,context
from test_live_learning import identity
from test_fixed_learning import _request,_feedback
torch.set_num_threads(1);torch.manual_seed(4)
a=l.OnlineV3Learner(model());online=FixedLiveLearning(a,update_timing='after_rollout8');x=context();base=torch.zeros(1,20)
f=a.current_logits

def kernels(base,inputs,reference_logits=None):
 out=f(base,inputs,reference_logits)
 return out+1.52587890625e-5 if torch.is_grad_enabled() else out
with patch.object(a,'current_logits',side_effect=kernels):
 for i in range(8):
  ident=identity(13+i,0,'state' if not i else 'next')
  req=_request(online,x,base,ident)
  main,feedback=_feedback(req,ident,[j/19 for j in range(20)],before='old' if not i else 'h'+str(i-1),after='h'+str(i))
  online.main_executed(main)
  ack,audit=online.update(feedback)
  online.acknowledged(online.awaiting_ack)
  assert a.version==(0 if i<7 else 8)
assert len(audit['optimizer_updates'])==8
print('PASS_DELAYED_ROLLOUT8_WITH_NUMERIC_DRIFT',a.version,flush=True)
