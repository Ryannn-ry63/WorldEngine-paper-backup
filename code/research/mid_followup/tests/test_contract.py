"""Scope, loss, native export and matched-state invariants."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_attribution_common as a
import selector_mid_followup_common as b
import selector_snapshot_common as s
from selector_snapshot_data import export_model
from selector_mid_followup_cases import first_divergence
from selector_mid_followup_probes import decomposition,gradients,state_hash
from selector_dual_domain_state import NUMERIC,CATEGORICAL
from selector_dual_domain_analysis import Evidence,select_cases
import selector_feedback_common as f


class ContractTests(unittest.TestCase):
    def test_fixed_scope_and_cross_task_rejection(self):
        self.assertEqual(len(set(b.dev_conditions())),72)
        self.assertEqual(len(set(b.formal_conditions())),24)
        self.assertEqual(len(set(b.queue('sensitivity'))),24)
        self.assertEqual(len(set(b.queue('mid-formal'))),18)
        self.assertEqual(b.NEW_NATIVE,('MID_seed0','MID_seed1','MID_seed2'))
        self.assertEqual(b.task_spec('mid-formal')['expected_bridge_conditions'],6)
        self.assertTrue(all(q[1].startswith('MID_seed') for q in b.queue('mid-formal')))
        for task,stage in [('sensitivity','formal-eval'),('sensitivity','native-bridge'),('mid-formal','train'),('mid-formal','dev-eval')]:
            with self.assertRaises(RuntimeError):b.validate_stage(task,stage)
        for task in b.TASKS:
            for stage in ('audit','preflight','all','report'):b.validate_stage(task,stage)

    def test_fixed_loss_and_gradient_use_all_rows(self):
        rows=[dict(candidate_indices=[0,1]),dict(candidate_indices=[1,2])]
        with patch.object(a,'preference',side_effect=lambda r,source:(0,1,.7) if r is rows[0] else None):
            for arm,coefficient in [('LOW04',.4),('HIGH06',.6)]:
                z=torch.zeros(2,20,dtype=torch.float64,requires_grad=True)
                value,pair,nll=b.objective(z,rows,arm)
                expected=.7*(-z[0].log_softmax(0)[0]+coefficient*torch.nn.functional.softplus(z[0,1]-z[0,0]))/2
                self.assertTrue(torch.allclose(value,expected,rtol=0,atol=1e-14))
                self.assertTrue(torch.allclose(torch.autograd.grad(value,z,retain_graph=True)[0],torch.autograd.grad(expected,z)[0],rtol=0,atol=1e-14))
        for arm,alpha in [('MID',1),('P1',1),('LOW04',1.1),('HIGH06',float('nan'))]:
            with self.assertRaises(RuntimeError):b.specification(arm,alpha)

    def test_native_export_only_existing_three_mid_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);scalar=root/'scalar.pt';selector=root/'selector.pt'
            frozen={f'base{i}':torch.tensor([float(i)]) for i in range(963)}
            frozen.update({s.PREFIX+f'w{i}':torch.zeros(1) for i in range(54)})
            torch.save(dict(state_dict=frozen,meta={}),scalar)
            for policy in b.NEW_NATIVE:
                torch.save(dict(score_mode='residual',inference_uses_reward_or_q=False,provenance=dict(policy=policy),
                    scene_selector_config={},scene_selector_state={f'w{i}':torch.ones(1) for i in range(54)}),selector)
                result=export_model(scalar,selector,root/(policy+'.pt'),{},expected_policy=policy,method=b.METHOD)
                self.assertEqual(result['unchanged_nonselector_tensors'],963)
            for policy in ('P1_seed0','LOW04_seed0','MID_SCALE_MATCH_seed0','MID_seed3'):
                with self.assertRaises(RuntimeError):export_model(scalar,selector,root/'invalid.pt',{},expected_policy=policy,method=b.METHOD)

    def context(self):
        x={k:np.zeros(1) for k in f.FINGERPRINT_FIELDS}
        return dict(x,deployed_trajectory=np.zeros((40,3)),selected_index=0)

    def physical(self):
        return dict({k:np.zeros(1) for k in NUMERIC},**{k:'same' for k in CATEGORICAL})

    def test_physical_mismatch_cannot_rejoin_shared_prefix(self):
        left={s:self.context() for s in c.DECISION_STEPS};right=copy.deepcopy(left)
        lp={s:self.physical() for s in c.DECISION_STEPS};rp=copy.deepcopy(lp)
        right[5]['deployed_trajectory'][0,0]=1
        result=first_divergence(left,right,lp,rp)
        self.assertEqual(result['first_shared_context_action_divergence']['step'],5)
        self.assertTrue(all(not x['comparable'] for x in result['steps'] if x['step']>5))
        rp[4]['can_bus'][0]=1
        result=first_divergence(left,right,lp,rp)
        self.assertIsNone(result['first_shared_context_action_divergence'])
        self.assertTrue(all(not x['comparable'] for x in result['steps']))

    def test_gradient_probe_preserves_model_and_nll_identity(self):
        model=torch.nn.Linear(2,20);before=state_hash(model)
        rows=[dict(candidate_indices=[0,1],branch_outcomes=[dict(success=1,no_at_fault_collisions=1,drivable_area_compliance=1),dict(success=0,no_at_fault_collisions=0,drivable_area_compliance=1)])]*4
        with patch.object(a,'preference',return_value=(0,1,.8)):
            terms=decomposition(model(torch.randn(4,2)),rows)
            self.assertLess(float((terms['pair']+terms['set_mass']-terms['nll']).abs()),1e-6)
            result=gradients(terms,model)
        self.assertEqual(before,state_hash(model))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertIn('pair__nll',result['cosines'])

    def test_evidence_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'input';p.write_text('original');ev=Evidence();ev.use(p)
            p.write_text('changed')
            with self.assertRaises(RuntimeError):ev.finish()


if __name__=='__main__':unittest.main()
