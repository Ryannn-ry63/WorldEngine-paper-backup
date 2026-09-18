"""Synthetic integration tests. Never load a research model or launch a GPU job."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_mid_followup_common as b
import selector_mid_followup_data as data
import selector_mid_followup_collections as dev
import selector_mid_followup_native as native
import selector_snapshot_common as s
import train_selector_mid_followup as training
import report_selector_mid_followup as reporting
import run_selector_mid_followup as runner


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()


    def test_actual_training_loop_resume_matches_uninterrupted_toy_model(self):
        torch.manual_seed(12)
        teacher = torch.nn.Linear(3,20)
        parent_model = copy.deepcopy(teacher)
        optimizer = torch.optim.AdamW(parent_model.parameters(), lr=1e-4, weight_decay=1e-4)
        for _ in range(3):
            loss = parent_model(torch.randn(2,3)).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        parent = self.root/'parent.pt'
        torch.save(dict(step=500, provenance=dict(arm='shared', seed=0), model=parent_model.state_dict(),
            optimizer=optimizer.state_dict(), rng=np.random.default_rng(22).bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=None), parent)
        c.locked_json(self.root/'manifest.json', {'synthetic':True})
        c.locked_json(self.root/'feedback.json', {'artifacts':{'scalar_manifest':d.artifact(self.root/'manifest.json')}})
        inputs = dict(parents={'0':d.artifact(parent)}, artifacts={'feedback_inputs':d.artifact(self.root/'feedback.json')})
        rows = [dict(row_id=f'{mode}{i}', mode=mode, candidate_indices=[0,1],
                     feature=np.array([i/10,1.,-1.],np.float32)) for mode in b.MODES for i in range(16)]
        def prepare(name):
            run = self.root/name
            for filename, value in (('run_contract.json',{'code_sha':'toy','task':'sensitivity'}),
                    ('audit_report.json',{'status':'PASS','code_sha':'toy'}),
                    ('followup_inputs.json',{'synthetic':True}), ('scale_audit.json',{'synthetic':True})):
                c.locked_json(run/filename, value)
            return run
        full, interrupted = prepare('full'), prepare('interrupted')
        class Dense:
            def __init__(self, run):
                pass
            def sample(self, rng):
                return [dict(feature=rng.normal(size=3).astype(np.float32),
                    candidate_rewards=np.ones(20,np.float32), candidate_reward_valid_mask=np.ones(20,bool))
                    for _ in range(16)]
        def score(model, batch, indices, device, mode):
            return model(torch.as_tensor(np.stack([batch[i]['feature'] for i in indices])))
        def save(path, model, config, mode, provenance):
            torch.save(dict(weights=model.state_dict(), provenance=provenance), path)
        def load(path):
            model = torch.nn.Linear(3,20)
            model.load_state_dict(torch.load(path)['weights'])
            return model, {}
        with ExitStack() as stack:
            for obj, name, options in [
                (data,'checked_inputs',dict(return_value=inputs)), (data,'load_rows',dict(return_value=(rows,{}))),
                (training,'Dense',dict(new=Dense)),
                (training.m,'load_incumbent',dict(side_effect=lambda *args:(copy.deepcopy(teacher),{},{}))),
                (training.m,'initialize',dict(side_effect=lambda model,*args:copy.deepcopy(model))),
                (training.m,'score',dict(side_effect=score)), (training.m,'save_selector',dict(side_effect=save)),
                (training.m,'load_selector',dict(side_effect=load)),
                (training.a,'preference',dict(return_value=(0,1,.7))),
                (training.m.v3,'exact_group_loss',dict(side_effect=lambda z,t,*args:((z-t).square().mean(),)))]:
                stack.enter_context(patch.object(obj,name,**options))
            stack.enter_context(patch('builtins.print'))
            training.train(full,'LOW04',0,'cpu')
            clip = training.m.v3.clip_grad_norm_cpu_
            calls = [0]
            def fail_after_saved_checkpoint(parameters, maximum):
                calls[0] += 1
                if calls[0] == 38:
                    raise RuntimeError('synthetic interruption')
                return clip(parameters, maximum)
            with patch.object(training.m.v3,'clip_grad_norm_cpu_',side_effect=fail_after_saved_checkpoint):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    training.train(interrupted,'LOW04',0,'cpu')
            self.assertEqual(torch.load(interrupted/'train/LOW04_seed0/resume.pt')['step'],25)
            training.train(interrupted,'LOW04',0,'cpu')
        left = torch.load(full/'train/LOW04_seed0/resume.pt')
        right = torch.load(interrupted/'train/LOW04_seed0/resume.pt')
        for key,value in left['model'].items():
            self.assertTrue(torch.equal(value,right['model'][key]),key)
        self.assertEqual(left['history'],right['history'])
        self.assertEqual(left['clip_count'],right['clip_count'])
        self.assertEqual([p['step'] for p in right['gradient_probes']],[1,250,500])

    def test_endpoint_retraining_is_forbidden(self):
        with self.assertRaises(RuntimeError):
            training.train(self.root,'P1',0,'cpu')

    def test_training_scheduler_never_exceeds_four_devices(self):
        obj = runner.Runner.__new__(runner.Runner)
        obj.run = self.root
        obj.devices = ['3','5','6','7']
        obj.env = {'ALGENGINE_PYTHON':'python'}
        batches = []
        obj.capacity = lambda *args:None
        obj.execute = lambda jobs,*args,**kwargs:batches.append(jobs)
        with patch.object(data,'load_rows',return_value=([],{})):
            obj.training()
        self.assertEqual([len(batch) for batch in batches],[4,2])
        commands = [job[0] for batch in batches for job in batch]
        self.assertEqual(len(commands),6)
        self.assertEqual(len({tuple(map(str,cmd)) for cmd in commands}),6)
        for batch in batches:
            self.assertEqual(len({job[2]['CUDA_VISIBLE_DEVICES'] for job in batch}),len(batch))

    def test_incomplete_report_never_creates_complete_summary(self):
        c.locked_json(self.root/'run_contract.json', {'task':'sensitivity'})
        with patch.object(data,'checked_inputs',return_value={}), \
             patch.object(s,'checked_inputs',return_value={}), \
             patch.object(dev,'checked',return_value=None), \
             patch.object(native,'checked_formal',return_value=None):
            result = reporting.report(self.root)
        self.assertEqual(result['status'],'INCOMPLETE')
        self.assertEqual(len(result['missing']),78)
        self.assertFalse((self.root/'summary.json').exists())
        self.assertFalse((self.root/'archive/release_manifest.json').exists())

    def test_formal_requires_bridge_before_reading_or_launching_conditions(self):
        obj = runner.Runner.__new__(runner.Runner)
        obj.run = self.root
        with patch.object(native,'require_bridge',side_effect=RuntimeError('bridge missing')), \
             patch.object(native,'checked_formal') as checked:
            with self.assertRaisesRegex(RuntimeError,'bridge missing'):
                obj.formal_eval('MID_seed0','full_NR')
            checked.assert_not_called()


if __name__ == '__main__':
    unittest.main()
