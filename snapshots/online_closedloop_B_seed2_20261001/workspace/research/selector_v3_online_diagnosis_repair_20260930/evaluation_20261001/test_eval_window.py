import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ENTRY=Path(__file__).with_name('eval_worker.py')
spec=importlib.util.spec_from_file_location('eval_entry_test',ENTRY)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class EvaluationWindowTest(unittest.TestCase):
 def setup_paths(self, folder):
  root=Path(folder)/'evaluation';repair=Path(folder)/'repair';reg=Path(folder)/'registry'
  root.mkdir();reg.mkdir();(repair/'evidence').mkdir(parents=True)
  (repair/'evidence/recovery_release.json').write_text('{}')
  (reg/'campaign_clock.json').write_text('{"deadline_unix": 900, "scope": "training"}')
  (reg/'experiment.json').write_text('{}');(reg/'scene_manifest.json').write_text('{}')
  (root/'manifest.json').write_text('{}')
  return root,repair,reg
 def test_clock_is_idempotent_and_does_not_extend_training(self):
  with tempfile.TemporaryDirectory() as folder:
   root,repair,reg=self.setup_paths(folder);before=(reg/'campaign_clock.json').read_bytes()
   with patch.object(m,'ROOT',root),patch.object(m.runner,'REGISTRY',reg),patch.object(m.time,'time',return_value=1000):
    first=m.evaluation_clock('test')
    self.assertEqual(first['deadline_unix'],87400)
    with patch.object(m.time,'time',return_value=2000):
     self.assertEqual(m.evaluation_clock('test'),first)
   self.assertEqual((reg/'campaign_clock.json').read_bytes(),before)
 def test_queue_uses_only_evaluation_and_shared_clock(self):
  with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as s:
   root,repair,reg=self.setup_paths(folder);before=(reg/'campaign_clock.json').read_bytes()
   spec={'jobs':[{'id':'eval1','kind':'evaluation','dataset':'development'}, {'id':'train1','kind':'train'}]}
   for obj,key,value in [(m,'ROOT',root),(m,'REPAIR',repair),(m.runner,'REGISTRY',reg),
      (m,'verify_entry',lambda:None),(m.runner,'registration',lambda:spec),
      (m.queue,'registration',lambda:spec),(m.runner,'campaign_clock',lambda:{}),
      (m.queue,'campaign_clock',lambda:{}),(m.runner,'write',m.runner.write),
      (m.summary_module,'read',m.summary_module.read)]:
    s.enter_context(patch.object(obj,key,value))
   s.enter_context(patch.object(m.time,'time',return_value=1000))
   def queued():
    self.assertEqual(sys.argv[sys.argv.index('--group')+1],'evaluation')
    self.assertEqual(m.runner.campaign_clock(),m.queue.campaign_clock())
    self.assertEqual(m.summary_module.read(reg/'campaign_clock.json')['scope'],'evaluation_only')
    self.assertEqual(m.runner.read(reg/'campaign_clock.json')['scope'],'training')
    m.runner.write(reg/'jobs/eval1/status.json',{'status':'RUNNING'})
    self.assertIn('evaluation_window',m.runner.read(reg/'jobs/eval1/status.json'))
   q=s.enter_context(patch.object(m.queue,'main',side_effect=queued))
   s.enter_context(patch.object(sys,'argv',['entry','--session','test']))
   self.assertEqual(m.main(),0);q.assert_called_once()
   self.assertEqual((reg/'campaign_clock.json').read_bytes(),before)

if __name__=='__main__':unittest.main()
