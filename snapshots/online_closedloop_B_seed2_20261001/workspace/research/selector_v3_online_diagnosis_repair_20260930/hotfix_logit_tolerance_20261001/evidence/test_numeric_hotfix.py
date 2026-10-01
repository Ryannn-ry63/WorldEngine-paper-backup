import sys, unittest, importlib.util
from pathlib import Path
from unittest.mock import patch
root=Path('/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine')
hot=root/'research/selector_v3_online_diagnosis_repair_20260930/hotfix_logit_tolerance_20261001'
code=root/'research/selector_v3_online_diagnosis_20260930/code'
sys.path[:0]=[str(hot/'overlay'),str(code/'projects/AlgEngine/scripts/diffusiondrive')]
import innovation3.learner as patched
import innovation3.fixed_learning as fixed
assert Path(patched.__file__).is_relative_to(hot)
assert Path(fixed.__file__).is_relative_to(hot)
import torch
spec=importlib.util.spec_from_file_location('test_online_learner', code/'tests/innovation3/test_online_learner.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class NumericHotfixTests(unittest.TestCase):
 def test_small_roundoff_preserves_behavior_and_training_graph(self):
  for drift in (1.52587890625e-5,1.1205673217773438e-5):
   torch.manual_seed(4)
   l=patched.OnlineV3Learner(m.model()); x=m.context(); b=torch.zeros(1,20)
   original=l.current_logits
   def different_kernel(base,inputs,ref=None):
    result=original(base,inputs,ref)
    return result+drift if torch.is_grad_enabled() else result
   with torch.no_grad():
    ref=l.reference_logits(b,x)
    expected=original(b,x,ref).softmax(-1)
   with patch.object(l,'current_logits',side_effect=different_kernel):
    _,p,_=l.choose(x,b,'s')
   self.assertTrue(torch.equal(p,expected))
   self.assertTrue(l.pending[1].requires_grad)
   self.assertTrue(l.update(torch.arange(20)[None,:],'s')['optimized'])
 def test_large_mismatch_still_rejected(self):
  l=patched.OnlineV3Learner(m.model());x=m.context();b=torch.zeros(1,20)
  def bad(base,inputs,ref=None):
   return torch.zeros(1,20)+(1e-2 if torch.is_grad_enabled() else 0)
  with patch.object(l,'current_logits',side_effect=bad):
   with self.assertRaisesRegex(RuntimeError,'logits diverged'):l.choose(x,b,'bad')
  self.assertIsNone(l.pending)
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m),unittest.defaultTestLoader.loadTestsFromTestCase(NumericHotfixTests)])
print('TESTING_PATCHED_LEARNER',patched.__file__,flush=True)
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
