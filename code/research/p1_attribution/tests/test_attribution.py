"""CPU-only regression and failure-injection tests. No real optimizer update/rollout."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
sys.path.insert(0,str(ROOT/'research/cfpi/tests'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_decision_common as x
import selector_attribution_common as a
import selector_attribution_data as data
import selector_attribution_collections as collections
import train_selector_attribution as trainer
import run_selector_attribution as runner
import report_selector_attribution as reporter
from train_selector_feedback import pair_loss
from selector_feedback_transport import validate_feedback
from run_selector_cfpi_deployment import charge, check_budget


def metric(score=.8, safe=True):
    return dict(score=score,no_at_fault_collisions=float(safe),drivable_area_compliance=1.,
                success=float(safe),ego_progress=.5,time_to_collision_within_bound=1.,
                comfort=1.,driving_direction_compliance=1.)


def row(index=0, mode='NR', fresh=False):
    r = dict(scene_id=f'log{index}-'+format(index,'016x'),origin_log=f'log{index}',
             mode=mode,decision_step=6,visiting_policy='shared_seed0',continuation_policy='shared_seed0',
             candidate_indices=[0,1],branch_outcomes=[metric(.2),metric(.8)],
             local_pair_outcomes=[metric(.8),metric(.2)],fresh=fresh)
    r.update({k:np.zeros((20,),np.float32) for k in f.FINGERPRINT_FIELDS})
    r['candidate_trajectories_8'] = np.zeros((20,8,3),np.float32)
    r.update(source_fingerprint=f.fingerprint(r),candidate_bank_sha=x.candidate_bank_hash(r))
    r.update(row_id=x.row_id(r),preference=f.pareto_pair(*r['branch_outcomes']))
    return r


class ObjectiveTests(unittest.TestCase):
    def test_six_components(self):
        expected = [('local',True,False),('local',True,True),('closed',True,False),
                    ('closed',True,True),('closed',False,False),('closed',False,True)]
        for arm,values in zip(a.ARMS,expected):
            self.assertEqual(tuple(a.components(arm).values()),values)
        with self.assertRaises(ValueError): a.components('best_seed')

    def test_p0_p1_exact_legacy_values_and_gradients(self):
        rows = [row(),row(1)]
        rows[1]['branch_outcomes']=[metric(.5),metric(.5)]
        rows[1]['preference']=None
        for arm in ('P0','P1'):
            z=torch.randn(2,20,requires_grad=True)
            other=z.detach().clone().requires_grad_()
            pair,nll=a.feedback_loss(z,rows,arm)
            if arm=='P0':
                expected=pair_loss(other,rows)
            else:
                p,n=x.feedback_loss(other,rows); expected=p+n
            self.assertTrue(torch.equal(pair+nll,expected))
            (pair+nll).backward();expected.backward()
            self.assertTrue(torch.equal(z.grad,other.grad))

    def test_local_labels_reverse_direction_without_changing_pairs(self):
        r=row();z=torch.zeros(1,20,requires_grad=True)
        pair,_=a.feedback_loss(z,[r],'L_Pair');pair.backward()
        self.assertLess(z.grad[0,0],0)
        self.assertEqual(float(z.grad[0,2:].abs().sum()),0.)
        self.assertEqual(r['candidate_indices'],[0,1])

    def test_full20_nll_not_two_way_nll(self):
        z=torch.zeros(1,20,requires_grad=True)
        pair,nll=a.feedback_loss(z,[row()],'CL_NLL')
        self.assertEqual(float(pair),0.)
        self.assertAlmostEqual(float(nll),.6*np.log(20),places=6)
        nll.backward()
        self.assertTrue(bool((z.grad[0,2:]>0).all()))

    def test_continue_has_no_label_read_or_feedback_gradient(self):
        r=row();del r['branch_outcomes'];del r['local_pair_outcomes']
        z=torch.randn(1,20,requires_grad=True)
        p,n=a.feedback_loss(z,[r],'Continue')
        (p+n).backward()
        self.assertEqual(float(z.grad.abs().sum()),0.)

    def test_ties_conflicts_no_fabricated_labels(self):
        for arm in a.ARMS:
            r=row();values=[metric(.8,False),metric(.5,True)]
            r.update(branch_outcomes=values,local_pair_outcomes=values)
            p,n=a.feedback_loss(torch.randn(1,20,requires_grad=True),[r],arm)
            self.assertEqual(float(p+n),0.)

    def test_all_rows_denominator(self):
        rows=[row(),row(1)]
        rows[1]['branch_outcomes']=[metric(.5),metric(.5)]
        p,n=a.feedback_loss(torch.zeros(2,20),rows,'P1')
        self.assertAlmostEqual(float(p),.6*np.log(2)/2,places=6)
        self.assertAlmostEqual(float(n),.6*np.log(20)/2,places=6)

    def test_unknown_added_query_not_allowed(self):
        r=row();r['candidate_indices']=[0,1,2]
        with self.assertRaises(RuntimeError):a.preference(r,'closed')

    def test_nonfinite_and_invalid_local(self):
        for scores,values,ids in [(np.ones(19),np.ones((20,6)),[0,1]),
            (np.full(20,np.nan),np.ones((20,6)),[0,1]),
            (np.ones(20),np.ones((20,6)),[1,1]),
            (np.ones(20),np.ones((20,6)),[0,20]),
            (np.ones(20),np.ones((20,6))*2,[0,1])]:
            with self.assertRaises(RuntimeError):a.local_outcomes(scores,values,ids)

    def test_local_pair_only_and_strict_success(self):
        values=np.ones((20,6));values[1,0]=.5
        outcomes=a.local_outcomes(np.linspace(0,1,20),values,[0,1])
        self.assertEqual(len(outcomes),2)
        self.assertEqual([v['success'] for v in outcomes],[1.,0.])
        self.assertAlmostEqual(outcomes[1]['score'],1/19)

    def test_gradient_probe_does_not_touch_rng_or_grad(self):
        parameter=torch.nn.Parameter(torch.tensor([2.,3.]))
        before=torch.get_rng_state().clone()
        norm=trainer.gradient_norm(parameter.square().sum(),(parameter,))
        self.assertAlmostEqual(norm,float(torch.linalg.vector_norm(parameter.double()*2)))
        self.assertIsNone(parameter.grad)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))


class DataTests(unittest.TestCase):
    def test_fixed_coverage_exclusions_and_fingerprint(self):
        rows=[row(i,'NR' if i%2==0 else 'R',i>=192) for i in range(256)]
        data.validate_rows(rows,[])
        with self.assertRaises(RuntimeError):data.validate_rows(rows[:255],[])
        with self.assertRaises(RuntimeError):data.validate_rows(rows,['log0'])
        changed=copy.deepcopy(rows);changed[0]['candidate_features'][0]=1
        with self.assertRaises(RuntimeError):data.validate_rows(changed,[])

    def test_shared_optimizer_and_rng_are_restored(self):
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
        # Populate a synthetic optimizer checkpoint, not a research model update.
        model(torch.ones(1,2)).sum().backward();optimizer.step()
        rng=np.random.default_rng(41);rng.integers(50,size=8)
        state=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                 rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=None))
        expected_np=rng.integers(100,size=16);expected_torch=torch.rand(4)
        new_model=torch.nn.Linear(2,1)
        new_opt=torch.optim.AdamW(new_model.parameters(),lr=.5)
        new_rng=np.random.default_rng(2)
        trainer.restore_state(state,new_model,new_opt,new_rng,'cpu')
        self.assertTrue(np.array_equal(expected_np,new_rng.integers(100,size=16)))
        self.assertTrue(torch.equal(expected_torch,torch.rand(4)))
        self.assertEqual(new_opt.param_groups[0]['lr'],1e-4)
        self.assertTrue(all(float(v['step'])==1 for v in new_opt.state.values()))
        for left,right in zip(model.parameters(),new_model.parameters()):self.assertTrue(torch.equal(left,right))

    def test_all_arms_same_feedback_sampling(self):
        rows=[row(i,'NR' if i%2==0 else 'R') for i in range(16)]
        results=[]
        for _ in a.ARMS:
            rng=np.random.default_rng(12)
            results.append([r['row_id'] for r in trainer.sample_feedback(rows,rng)])
        self.assertTrue(all(r==results[0] for r in results))
        self.assertEqual(len(results[0]),16)

    def test_local_attachment_fails_on_identity_and_pair_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            r=row();path=Path(folder)/'side.pkl';c.atomic_pickle(path,r)
            target=dict(visiting_index=1);entry=dict(sha256='collection')
            record=dict(status='PASS',method=a.METHOD,**{k:r[k] for k in (
                'row_id','scene_id','mode','decision_step','source_fingerprint','candidate_bank_sha','candidate_indices')},
                selected_index=1,local_definition=a.LOCAL_DEFINITION,collection_sha256='collection',
                action_max_abs=0.,inference_uses_labels=False,scores=[1.]*20,components=[[1.]*6]*20,
                pair_outcomes=a.local_outcomes(np.ones(20),np.ones((20,6)),[0,1]),sidecar=d.artifact(path))
            attached=collections.attach_local(r,record,target,entry)
            self.assertEqual(attached['branch_outcomes'],r['branch_outcomes'])
            for key,value in [('row_id','foreign'),('decision_step',12),('selected_index',0),('action_max_abs',.1)]:
                bad=dict(record,**{key:value})
                with self.assertRaises(RuntimeError):collections.attach_local(r,bad,target,entry)

    def test_observer_replay_rejects_outcome_change(self):
        current=dict(metrics={'scene':dict(metric(),first_violation_step=None)})
        source=dict(metrics={'scene':dict(metric(.3),first_violation_step=None)})
        with self.assertRaises(RuntimeError):collections.check_replay(current,source,'scene')


class ReportRunnerTests(unittest.TestCase):
    def test_exact_conditions_no_full_benchmark(self):
        conditions=collections.evaluation_conditions()
        self.assertEqual(len(conditions),80)
        self.assertEqual(len(set(conditions)),80)
        self.assertEqual(sum(p.startswith(tuple(a.NEW_ARMS)) for _,_,p in conditions),48)
        self.assertEqual({cohort for cohort,_,_ in conditions},{'development','common'})

    def test_contrast_interaction_and_paired_coverage(self):
        means={arm:{'s':{'score':v}} for arm,v in dict(P1=.8,P0=.7,L_Pair_NLL=.7,L_Pair=.65).items()}
        result=reporter.contrast(means,a.COMPARISONS['source_by_nll_interaction'],'score')
        self.assertAlmostEqual(result['s'],.05)
        means['P0']={'other':{'score':1.}}
        with self.assertRaises(RuntimeError):reporter.contrast(means,a.COMPARISONS['nll_under_closed'],'score')

    def test_interval_zero_is_not_significance(self):
        self.assertEqual(a.interval_verdict([-.01,.05]),'INCONCLUSIVE')
        self.assertEqual(a.interval_verdict([0,.05]),'INCONCLUSIVE')
        self.assertEqual(a.interval_verdict([.01,.05]),'POSITIVE_SIGNAL')
        with self.assertRaises(ValueError):a.interval_verdict([float('nan'),1.])

    def test_unknown_argmax_is_not_counted_as_bad(self):
        z=np.zeros((1,20));z[0,10]=10
        result=reporter.feedback_diagnostics(z,[row()])['NR']
        self.assertEqual(result['unknown_argmax'],1)
        self.assertEqual(result['sources']['closed']['known_dominated_argmax'],0)

    def test_local_label_reversals_and_mass(self):
        result=a.label_summary([row()])['NR']
        self.assertEqual(result['reversal'],1)
        self.assertAlmostEqual(result['closed']['gap_sum'],.6)

    def test_no_gpu_dispatch_from_audit_or_report(self):
        obj=runner.Runner.__new__(runner.Runner);obj.run=Path('/tmp/unused')
        obj.preflight=Mock(side_effect=AssertionError('GPU must not launch'))
        obj.audit=Mock()
        obj.args=SimpleNamespace(stage='audit')
        obj.dispatch();obj.audit.assert_called_once()
        obj.args.stage='report'
        with patch.object(reporter,'report',return_value=dict(status='INCOMPLETE',decision='REVIEW_REQUIRED')) as report:
            obj.dispatch();report.assert_called_once()
        obj.preflight.assert_not_called()

    def test_phase_requires_local_audit_before_cuda(self):
        obj=runner.Runner.__new__(runner.Runner);obj.run=Path('/tmp/unused');obj.code_sha='code'
        obj.args=SimpleNamespace(stage='train');obj.preflight=Mock()
        with patch.object(d,'read',return_value=dict(status='PASS',code_sha='code',
                  **{k:{} for k in ('inputs','cpu_preflight','forecast','local_bundle')})), patch.object(d,'verify'), \
                patch.object(data,'load_rows',side_effect=RuntimeError('labels missing')):
            with self.assertRaises(RuntimeError):obj.dispatch()
        obj.preflight.assert_not_called()

    def test_foreign_collection_cannot_execute(self):
        obj=runner.Runner.__new__(runner.Runner);obj.run=Path('/tmp/new')
        with patch.object(d,'verified_read',return_value=dict(research_method=a.METHOD)):
            with self.assertRaises(RuntimeError):obj.collection(dict(path='/tmp/old/collections/a/deployment_collection.json'))

    def test_budget_and_mapping(self):
        ledger=dict(gpu_hours_used=0.,phase_gpu_hours=dict(attribution=0.),phase_limits=dict(attribution=8.))
        charge(ledger,'attribution',3600,4)
        self.assertEqual(ledger['gpu_hours_used'],4.)
        with self.assertRaises(RuntimeError):check_budget(ledger,'attribution',8.,4.)
        self.assertEqual(f.gpu_devices(dict(CUDA_VISIBLE_DEVICES='4,5,6,7'),4),['4','5','6','7'])

    def test_transport_preserves_natural_actions(self):
        sidecar=dict(selector_rollout_contract=dict(research_method=a.METHOD,react_type='NR',
            source_data_split='train',development_consumed=False,test_consumed=False,
            legacy_exposed_benchmark=True,independent_unseen_test=False),
            planner_step=6,selected_index=3,cfpi_deployment={})
        condition=dict(research_method=a.METHOD,cohort='train',react_type='NR')
        self.assertFalse(validate_feedback(sidecar,condition,{}))
        sidecar['cfpi_deployment']['feedback_intervention']={}
        with self.assertRaises(RuntimeError):validate_feedback(sidecar,condition,{})


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
