import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_attribution_common as a
import selector_balance_common as b
import selector_balance_data as data
import selector_balance_collections as collections
import selector_balance_native as native
import selector_snapshot_common as s
import selector_snapshot_data as export
import selector_feedback_common as transport
import selector_feedback_transport as feedback
from train_selector_attribution import restore_state, sample_feedback
from selector_balance_forecast import durations
from report_selector_balance import average, compare
from run_selector_cfpi_deployment import charge, check_budget


def metrics(score=.8, success=1.):
    return dict(comfort=1., drivable_area_compliance=1., driving_direction_compliance=1.,
                ego_progress=.6, no_at_fault_collisions=success, score=score, success=success,
                time_to_collision_within_bound=1.)


class BalanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_condition_counts_and_unique_queue(self):
        self.assertEqual(len(set(b.dev_conditions())), 48)
        self.assertEqual(len(set(b.formal_conditions())), 42)
        queue = b.queue()
        self.assertEqual(len(queue), 54)
        self.assertEqual(len(set(queue)), 54)
        self.assertEqual(sum(t[0] == 'dev' for t in queue), 24)
        self.assertEqual(sum(t[0] == 'formal' for t in queue), 30)
        self.assertFalse(any(t[0] == 'formal' and t[1].startswith('MID') for t in queue))
        self.assertTrue(all(not t[-1].startswith('open') for t in queue[:-10]))

    def test_endpoint_value_and_gradient_identity(self):
        rows = [dict(candidate_indices=[0, 1]) for _ in range(16)]
        with patch.object(a, 'preference', return_value=(0, 1, .7)):
            for arm in ('P1', 'CL_NLL'):
                z = torch.randn(16, 20, dtype=torch.float64, requires_grad=True)
                pair, nll = a.feedback_loss(z, rows, arm)
                original = pair+nll
                actual, _, _ = b.objective(z, rows, arm)
                self.assertTrue(torch.equal(actual, original))
                g = torch.autograd.grad(actual, z, retain_graph=True)[0]
                expected = torch.autograd.grad(original, z)[0]
                self.assertTrue(torch.allclose(g, expected, atol=1e-14, rtol=0))

    def test_all_rows_denominator_no_information_and_scale(self):
        rows = [dict(candidate_indices=[0, 1]), dict(candidate_indices=[1, 2])]
        z = torch.zeros(2, 20, requires_grad=True)
        def preference(row, source):
            return (0, 1, 1.) if row is rows[0] else None
        with patch.object(a, 'preference', side_effect=preference):
            term, pair, nll = b.objective(z, rows, 'MID_SCALE_MATCH', 2.)
            self.assertAlmostEqual(float(pair), np.log(2)/2, places=6)
            self.assertAlmostEqual(float(nll), np.log(20)/2, places=6)
            self.assertTrue(torch.equal(term, 2*(nll+.5*pair)))
        with patch.object(a, 'preference', return_value=None):
            term, _, _ = b.objective(z, rows, 'MID')
            self.assertEqual(float(term), 0.)
            self.assertEqual(float(torch.autograd.grad(term, z)[0].abs().sum()), 0.)

    def test_scale_energy_and_invalid_inputs(self):
        p, n = torch.randn(50, dtype=torch.float64), torch.randn(50, dtype=torch.float64)
        numerator, denominator = float((n+p).square().sum()), float((n+.5*p).square().sum())
        alpha = b.matched_scale(numerator, denominator)
        self.assertAlmostEqual(float((alpha*(n+.5*p)).square().sum()), numerator, places=10)
        for num, den in ((1,0), (0,1), (-1,1), (float('nan'),1), (1,float('inf'))):
            with self.assertRaises(RuntimeError):
                b.matched_scale(num, den)
        for arm, alpha in (('unknown',1), ('MID',2), ('MID_SCALE_MATCH',0)):
            with self.assertRaises(RuntimeError):
                b.specification(arm, alpha)

    def test_restore_optimizer_and_numpy_torch_rng(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        rng = np.random.default_rng(8)
        rows = [dict(mode=mode, row_id=str(i)) for mode in b.MODES for i in range(20)]
        def step():
            selected = sample_feedback(rows, rng)
            x = torch.randn(4,3) + float(rng.normal())
            loss = model(x).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            return [r['row_id'] for r in selected]
        for _ in range(3):
            step()
        state = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=None))
        expected_rows = [step() for _ in range(5)]
        expected_model, expected_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        restore_state(state, model, optimizer, rng, 'cpu')
        self.assertEqual([step() for _ in range(5)], expected_rows)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected_model[key]))
        for key, value in optimizer.state_dict()['state'].items():
            for name, tensor in value.items():
                self.assertTrue(torch.equal(tensor, expected_optimizer['state'][key][name]))

    def native_fixture(self, method=b.METHOD):
        from audit_selector_cfpi_deployment import SHAPES
        co = dict(method=method, checkpoint={'sha256':'model'}, noise=b.FORMAL_NOISE,
                  id='condition', mode='NR', prefixes={'token':'scene'})
        side = {k:np.zeros(shape) for k,shape in SHAPES.items()}
        side.update(checkpoint_sha256='model', candidate_noise_namespace=b.FORMAL_NOISE,
            code_sha='code', deployed_trajectory=np.zeros((40,3)), scene_prefix='token',
            planner_step=4, selected_index=0, selected_indices=np.array(0),
            selector_rollout_contract=dict(experiment=method, condition='condition',
                react_type='NR', inference_uses_reward_or_q=False))
        return side, co

    def test_native_identity_and_protocol_separation(self):
        for method in (b.METHOD, s.METHOD):
            side, co = self.native_fixture(method)
            self.assertEqual(s.validate_native(side, co, 'code')['scene_id'], 'scene')
            for key, value in (('candidate_noise_namespace', c.NOISE_NAMESPACE),
                               ('checkpoint_sha256','wrong'), ('cfpi_deployment',{})):
                with self.assertRaises(RuntimeError):
                    s.validate_native(dict(side, **{key:value}), co, 'code')
        side, co = self.native_fixture('forged')
        with self.assertRaises(RuntimeError):
            s.validate_native(side, co, 'code')

    def test_native_export_five_registered_endpoints(self):
        frozen = {f'base{i}':torch.tensor([float(i)]) for i in range(963)}
        frozen.update({s.PREFIX+f'w{i}':torch.zeros(1) for i in range(54)})
        scalar = self.root/'scalar.pt'
        torch.save(dict(state_dict=frozen, meta={}), scalar)
        for policy in b.NEW_NATIVE:
            payload = dict(score_mode='residual', inference_uses_reward_or_q=False,
                provenance=dict(policy=policy), scene_selector_config={},
                scene_selector_state={f'w{i}':torch.ones(1) for i in range(54)})
            selector = self.root/'selector.pt'; torch.save(payload, selector)
            path = self.root/(policy+'.pt')
            result = export.export_model(scalar, selector, path, {}, expected_policy=policy, method=b.METHOD)
            self.assertEqual(result['unchanged_nonselector_tensors'], 963)
            self.assertEqual(result, export.export_model(scalar, selector, path, {}, expected_policy=policy, method=b.METHOD))
            checkpoint = torch.load(path)
            self.assertEqual(checkpoint['meta']['selector_snapshot']['method'], b.METHOD)
            for key in frozen:
                if not key.startswith(s.PREFIX):
                    self.assertTrue(torch.equal(frozen[key], checkpoint['state_dict'][key]))
            with self.assertRaises(RuntimeError):
                export.export_model(scalar, selector, self.root/'wrong.pt', {}, expected_policy='wrong', method=b.METHOD)

    def test_balance_transport_never_allows_training_intervention(self):
        self.assertIn(b.METHOD, transport.TRANSPORT_METHODS)
        co = dict(research_method=b.METHOD, react_type='NR', cohort='development', role='evaluation')
        side = dict(planner_step=4, selected_index=0, cfpi_deployment={},
            selector_rollout_contract=dict(research_method=b.METHOD, react_type='NR',
                source_data_split='development', development_consumed=True, test_consumed=False,
                legacy_exposed_benchmark=True, independent_unseen_test=False))
        self.assertFalse(feedback.validate_feedback(side, co, {}))
        with self.assertRaises(RuntimeError):
            feedback.validate_feedback(side, co, {'feedback_target':{'decision_step':4}})
        with self.assertRaises(RuntimeError):
            feedback.validate_feedback(side, dict(co, role='training'), {})

    def test_budget_not_reset_and_whole_allocation_charged(self):
        ledger = dict(gpu_hours_used=0., phase_gpu_hours={'balance':0.}, phase_limits={'balance':1.})
        charge(ledger, 'balance', 450, 4)
        self.assertEqual(ledger['gpu_hours_used'], .5)
        charge(ledger, 'balance', 450, 4)
        with self.assertRaises(RuntimeError):
            check_budget(ledger, 'balance', 1.)
        self.assertEqual(ledger['gpu_hours_used'], 1.)

    def test_output_overlap_and_immutable_json(self):
        for source in (self.root, self.root/'child', self.root.parent):
            with self.assertRaises(RuntimeError):
                b.disjoint(self.root, [source])
        path = self.root/'contract.json'
        c.locked_json(path, {'a':1})
        with self.assertRaises(RuntimeError):
            c.locked_json(path, {'a':2})
        self.assertEqual(d.read(path), {'a':1})

    def test_missing_duplicate_nonfinite_scenes(self):
        rows = [dict(scene_id='a', origin_log='log')]
        for values, subset in (({},rows), ({'a':metrics()},rows*2),
                               ({'a':metrics(float('nan'))}, rows)):
            with self.assertRaises(RuntimeError):
                b.validate_metrics(values, subset)
        b.validate_metrics({'a':metrics()}, rows)

    def test_joint_mean_not_mixed_average(self):
        values = dict(development_NR=.01, development_R=.01, common_NR=0., common_R=0.)
        self.assertTrue(b.joint_mean_candidate(values))
        self.assertFalse(b.joint_mean_candidate(dict(values, common_R=-.0001)))
        with self.assertRaises(RuntimeError):
            b.joint_mean_candidate({'development_NR':1.})

    def test_seed_average_and_paired_failure_counts(self):
        rows = [dict(scene_id='a', origin_log='l1'), dict(scene_id='b', origin_log='l2')]
        baseline = {r['scene_id']:metrics(.6, 1.) for r in rows}
        values = [dict(a=metrics(.7, float(seed != 1)), b=metrics(.8)) for seed in b.SEEDS]
        result = compare(values, [baseline]*3, rows)
        self.assertAlmostEqual(result['delta']['score'], .15)
        self.assertEqual([r['broken'] for r in result['rescued_broken']], [0,1,0])
        self.assertTrue(result['intervals']['score']['seed_average_first'])
        self.assertEqual(result['intervals']['score']['logs'], 2)
        with self.assertRaises(RuntimeError):
            average(values[:2])

    def test_duration_pairing(self):
        ledger = dict(events=[
            dict(stage='x', status='RUNNING', time='2026-09-14T00:00:00+00:00'),
            dict(stage='x', status='COMPLETE', time='2026-09-14T00:00:10+00:00'),
            dict(stage='y', status='RUNNING', time='2026-09-14T00:01:00+00:00')])
        self.assertEqual(durations(ledger), {'x':10.})

    def test_historical_dev_never_materialized_as_new(self):
        path = self.root/'historical.json'; c.locked_json(path, {'status':'PASS'})
        entry = d.artifact(path)
        inputs = dict(reused_dev={'eval_development_NR_P1_seed2':entry})
        with patch.object(data, 'checked_inputs', return_value=inputs):
            self.assertEqual(collections.make(self.root, 'development', 'NR', 'P1_seed2'), entry)
        self.assertFalse((self.root/'collections').exists())


if __name__ == '__main__':
    unittest.main()
