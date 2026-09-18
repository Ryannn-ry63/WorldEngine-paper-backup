import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
SCRIPTS=ROOT/'projects/AlgEngine/scripts/diffusiondrive'
sys.path.insert(0,str(SCRIPTS))
import selector_dual_domain_analysis as h
from selector_dual_domain_io import sensitivity
from selector_dual_domain_probes import decomposition,gradients,state_hash
from selector_dual_domain_cases import first_divergence
import selector_attribution_common as a
import selector_feedback_common as f


def metric(score=.8,safe=True):
    v={k:1. for k in h.d.METRICS}
    v.update(score=score,success=float(safe),no_at_fault_collisions=float(safe))
    return v


def row():
    return dict(candidate_indices=[2,7],branch_outcomes=[metric(.8),metric(.2)],
                local_pair_outcomes=[metric(.8),metric(.2)])


class Objective(unittest.TestCase):
    def test_identity_and_gradients(self):
        for scale in (1.,1000.):
            z=(torch.randn(4,20,dtype=torch.float64)*scale).requires_grad_()
            rows=[row() for _ in range(4)]
            values=decomposition(z,rows)
            self.assertTrue(torch.allclose(values['pair']+values['set_mass'],values['nll'],atol=1e-10))
            pair,nll=a.feedback_loss(z,rows,'P1')
            expected=pair+nll;actual=2*values['pair']+values['set_mass']
            self.assertTrue(torch.allclose(expected,actual))
            self.assertTrue(torch.allclose(torch.autograd.grad(expected,z,retain_graph=True)[0],torch.autograd.grad(actual,z)[0]))

    def test_tie_denominator_and_unknowns(self):
        rows=[row(),row()];rows[1]['branch_outcomes']=[metric(.5),metric(.5)]
        z=torch.zeros(2,20,requires_grad=True);v=decomposition(z,rows)
        self.assertAlmostEqual(float(v['pair']),.6*np.log(2)/2,places=6)
        grad=torch.autograd.grad(v['pair'],z,retain_graph=True)[0]
        self.assertEqual(float(grad[0,0]),0.)
        self.assertEqual(float(grad[1].abs().sum()),0.)
        self.assertGreater(float(torch.autograd.grad(v['nll'],z)[0][0,0]),0.)

    def test_permutation(self):
        z=torch.randn(1,20);r=row();perm=torch.randperm(20)
        rr=copy.deepcopy(r);rr['candidate_indices']=[int((perm==k).nonzero()[0]) for k in r['candidate_indices']]
        for key in ('pair','nll','set_mass'):
            self.assertTrue(torch.allclose(decomposition(z,[r])[key],decomposition(z[:,perm],[rr])[key]))

    def test_no_preference(self):
        r=row();r['branch_outcomes']=[metric(.8,False),metric(.5,True)]
        z=torch.randn(1,20,requires_grad=True)
        self.assertEqual(float(decomposition(z,[r])['nll']),0.)

    def test_invalid_logits(self):
        for z in (torch.zeros(1,19),torch.full((1,20),float('nan'))):
            with self.assertRaises(RuntimeError):decomposition(z,[row()])

    def test_probe_does_not_update(self):
        model=torch.nn.Linear(2,20);before=state_hash(model)
        terms=decomposition(model(torch.ones(1,2)),[row()]);v=gradients(terms,model)
        self.assertEqual(before,state_hash(model));self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertGreater(v['norms']['pair'],0.)


class Contracts(unittest.TestCase):
    def test_output_overlap_and_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';root.mkdir()
            for dest in (root,root/'child',Path(tmp)):
                with self.assertRaises(RuntimeError):h.output_location(dest,[root])
            self.assertEqual(h.output_location(Path(tmp)/'new',[root]),Path(tmp)/'new')

    def test_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_text('a');ev=h.Evidence();ev.use(p,model=True)
            self.assertEqual(ev.finish()['optimizer_steps'],0)
            p.write_text('b')
            with self.assertRaises(RuntimeError):ev.finish()

    def test_seed_coverage(self):
        with self.assertRaises(RuntimeError):h.average([{'a':metric()},{'b':metric()}])

    def test_case_selection_includes_every_safety_difference(self):
        rows=[dict(scene_id=str(i),origin_log=str(i//2)) for i in range(10)]
        left=[{r['scene_id']:metric() for r in rows} for _ in range(3)];right=copy.deepcopy(left)
        right[2]['9']=metric(.8,False)
        selected=h.select_cases(left,right,rows)
        self.assertIn('9',[x['scene_id'] for x in selected])

    def test_sensitivity_does_not_drop_main_rows(self):
        rows=[dict(scene_id=str(i),origin_log=str(i)) for i in range(5)]
        delta={str(i):i/10 for i in range(5)};result=sensitivity(delta,rows)
        self.assertEqual(len(result['ranked']),5)
        with self.assertRaises(RuntimeError):sensitivity({'a':1},rows)

    def test_prefix_and_terminal(self):
        base={k:np.zeros(1) for k in f.FINGERPRINT_FIELDS}
        base.update(deployed_trajectory=np.zeros((40,3)),selected_index=0)
        left={s:copy.deepcopy(base) for s in range(4,12)};right=copy.deepcopy(left)
        right[6]['deployed_trajectory'][0,0]=1
        result=first_divergence(left,right)
        self.assertEqual(result['first_shared_context_action_divergence']['step'],6)
        self.assertFalse(result['steps'][3]['comparable'])
        right[12]=base
        with self.assertRaises(RuntimeError):first_divergence(left,right)

    def test_no_execution_or_optimizer_calls(self):
        forbidden={'step','backward','train','dispatch','execute','save_selector','audit_collection','freeze','subprocess','Popen','system'}
        for path in SCRIPTS.glob('*dual_domain*.py'):
            tree=ast.parse(path.read_text())
            calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
            self.assertFalse(calls&forbidden,(path,calls&forbidden))


if __name__=='__main__':unittest.main()
