"""Exercise the real loop/report with tiny CPU models and synthetic records only."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_attribution import a,c,d,row,metric,data,trainer,reporter,collections


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.zeros(20))


class EndToEndTests(unittest.TestCase):
    def test_real_loop_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            teacher=Toy()
            parent_model=Toy()
            optimizer=torch.optim.AdamW(parent_model.parameters(),lr=1e-4,weight_decay=1e-4)
            parent_model.logits.sum().backward();optimizer.step()
            torch.manual_seed(22)
            rng=np.random.default_rng(11);rng.integers(100,size=16)
            parent=root/'parent.pt'
            torch.save(dict(step=500,provenance=dict(seed=1,arm='shared'),model=parent_model.state_dict(),
                optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                cuda_rng=None),parent)
            marker=root/'marker.json';c.atomic_json(marker,dict(artifacts=dict(scalar_manifest={})))
            marker_entry=d.artifact(marker)
            rows=[row(i,'NR' if i%2==0 else 'R') for i in range(16)]
            dense_row=dict(candidate_rewards=np.arange(20,dtype=np.float32)/20,
                           candidate_reward_valid_mask=np.ones(20,dtype=bool))
            inputs=dict(parents={'1':d.artifact(parent)},artifacts=dict(feedback_inputs=marker_entry))
            interrupted=[False]
            class Dense:
                def __init__(self,run): self.calls=0
                def sample(self,rng):
                    self.calls+=1
                    if interrupted[0] and self.calls==251: raise RuntimeError('synthetic interruption')
                    rng.integers(100,size=16)
                    return [dense_row]*16
            def load_selector(path):
                state=torch.load(path,map_location='cpu');model=Toy()
                model.load_state_dict(state['scene_selector_state'])
                return model.eval(),state
            def make_run(name):
                run=root/name
                c.atomic_json(run/'run_contract.json',dict(code_sha='toy'))
                c.atomic_json(run/'audit_report.json',dict(status='PASS',code_sha='toy'))
                c.atomic_json(run/'attribution_inputs.json',dict(toy=True))
                return run
            with ExitStack() as stack:
                stack.enter_context(patch.object(data,'checked_inputs',return_value=inputs))
                stack.enter_context(patch.object(data,'load_rows',return_value=(rows,marker_entry)))
                stack.enter_context(patch.object(trainer,'Dense',Dense))
                stack.enter_context(patch.object(trainer.m,'load_incumbent',return_value=(teacher,{},{})))
                stack.enter_context(patch.object(trainer.m,'load_selector',side_effect=load_selector))
                stack.enter_context(patch.object(trainer.m,'score',
                    side_effect=lambda model,rows,ids,device,mode:model.logits[None].expand(len(ids),-1)))
                # The nested scalar manifest is synthetic but must be an existing artifact.
                c.atomic_json(marker,dict(artifacts=dict(scalar_manifest=d.artifact(parent))))
                inputs['artifacts']['feedback_inputs']=d.artifact(marker)
                full, resumed=make_run('full'),make_run('resumed')
                with redirect_stdout(io.StringIO()):
                    trainer.train(full,'CL_NLL',1,'cpu')
                    interrupted[0]=True
                    with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                        trainer.train(resumed,'CL_NLL',1,'cpu')
                    interrupted[0]=False
                    trainer.train(resumed,'CL_NLL',1,'cpu')
                    trainer.train(resumed,'CL_NLL',1,'cpu') # completed report is idempotent
                left=torch.load(full/'train/CL_NLL_seed1/resume.pt',map_location='cpu')
                right=torch.load(resumed/'train/CL_NLL_seed1/resume.pt',map_location='cpu')
                self.assertEqual(left['step'],500)
                self.assertTrue(torch.equal(left['model']['logits'],right['model']['logits']))
                self.assertTrue(torch.equal(left['torch_rng'],right['torch_rng']))
                self.assertEqual(left['rng'],right['rng'])
                self.assertEqual(left['history'],right['history'])
                self.assertEqual(left['gradient_probes'],right['gradient_probes'])
                for key in left['optimizer']['state'][0]:
                    self.assertTrue(torch.equal(left['optimizer']['state'][0][key],
                                                right['optimizer']['state'][0][key]))
                before_rng=torch.get_rng_state().clone()
                before_parent=d.artifact(parent)
                with patch.object(a,'SEEDS',(1,)):
                    probe=trainer.initial_gradient_audit(full)
                    self.assertEqual(len(probe['probes']),6)
                    self.assertEqual(probe['optimizer_steps'],0)
                    self.assertTrue(all(p['feedback_row_ids']==probe['probes'][0]['feedback_row_ids']
                                        for p in probe['probes']))
                    self.assertEqual(probe,trainer.initial_gradient_audit(full))
                self.assertTrue(torch.equal(before_rng,torch.get_rng_state()))
                self.assertEqual(before_parent,d.artifact(parent))

    def test_complete_report_repeats_and_cannot_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)
            c.atomic_json(run/'run_contract.json',dict(code_sha='toy'))
            c.atomic_json(run/'attribution_inputs.json',dict(toy=True))
            marker=run/'marker.json';c.atomic_json(marker,{})
            entry=d.artifact(marker)
            cohort_rows=[dict(scene_id='s0',origin_log='l0'),dict(scene_id='s1',origin_log='l1')]
            inputs=dict(cohorts={k:dict(rows=cohort_rows,scenario=entry) for k in ('development','common')},
                        reused_collections={})
            values={}
            for cohort,mode,policy in collections.evaluation_conditions():
                name=f'eval_{cohort}_{mode}_{policy}'
                folder=run/'collections'/name
                old=policy in ('scalar_v3','gate_v3') or policy.startswith(('P0_','P1_'))
                condition=dict(cohort=cohort,react_type=mode,policy=policy,role='evaluation',
                    noise_namespace=c.NOISE_NAMESPACE,scenario=entry,research_method='old' if old else a.METHOD,
                    run_contract=d.artifact(run/'run_contract.json'),
                    inputs=d.artifact(run/'attribution_inputs.json'))
                routing=dict(models={policy:entry},routes=[dict(scene_id=r['scene_id']) for r in cohort_rows])
                c.atomic_json(folder/'routing.json',routing)
                condition['routing']=d.artifact(folder/'routing.json')
                c.atomic_json(folder/'deployment_collection.json',condition)
                artifact=d.artifact(folder/'deployment_collection.json')
                if old: inputs['reused_collections'][name]=artifact
                gain=.01 if policy.startswith('P1_') else 0.
                values[artifact['path']]=dict(collection=condition,audit=entry,
                                             metrics={r['scene_id']:metric(.7+gain) for r in cohort_rows})
            with ExitStack() as stack:
                stack.enter_context(patch.object(data,'checked_inputs',return_value=inputs))
                stack.enter_context(patch.object(data,'model_entry',return_value=entry))
                stack.enter_context(patch.object(data,'load_rows',return_value=([row(),row(1,'R')],entry)))
                stack.enter_context(patch.object(reporter,'checked_collection',side_effect=lambda e:values[e['path']]))
                stack.enter_context(patch.object(reporter.m,'load_selector',return_value=(None,{})))
                stack.enter_context(patch.object(reporter,'scores',return_value=np.zeros((2,20))))
                stack.enter_context(patch.object(trainer,'initial_gradient_audit',return_value={}))
                c.atomic_json(run/'labels/initial_gradient_audit.json',dict(status='PASS',synthetic=True))
                result=reporter.report(run)
                self.assertEqual(result['decision'],'REVIEW_REQUIRED')
                self.assertFalse(result['automatic_new_method'])
                self.assertFalse(result['full_rare288_authorized'])
                self.assertAlmostEqual(result['contrasts']['development_NR']['nll_under_closed']['metrics']['score']['scene_mean'],.01)
                self.assertEqual(result,reporter.report(run))
                self.assertEqual(len(result['feedback']),6)
                self.assertTrue(all(len(v)==3 for v in result['feedback'].values()))
