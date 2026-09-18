"""Complete-report integration on synthetic scenes, including negative outcomes."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_mid_followup_common as b
import selector_mid_followup_data as data
import selector_mid_followup_collections as dev
import selector_mid_followup_native as native
import selector_snapshot_common as s
import report_selector_mid_followup as reporting
import report_selector_balance as intervals


def metrics(score):
    return dict(comfort=1., drivable_area_compliance=1., driving_direction_compliance=1.,
        ego_progress=.6, no_at_fault_collisions=1., score=score, success=1.,
        time_to_collision_within_bound=1.)


class ReportTests(unittest.TestCase):
    def check_complete_report(self, task):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            for name in ('followup_inputs.json', 'inputs.json', 'run_contract.json', 'audit_report.json', 'bridge_report.json'):
                if name != 'bridge_report.json' or task != 'sensitivity':
                    c.locked_json(run/name, {'synthetic':True, 'task':task})
            for seed in b.SEEDS:
                for arm in (b.NEW_ARMS if task != 'mid-formal' else ()):
                    c.locked_json(run/'train'/f'{arm}_seed{seed}'/'step_500_report.json', {'synthetic':True})
            rows = {cohort:[dict(scene_id=cohort+str(i),origin_log='l'+str(i)) for i in range(2)]
                    for cohort in ('development','common','remainder')}
            full = rows['development']+rows['remainder']
            inputs = dict(cohorts={k:dict(rows=v) for k,v in rows.items()}, artifacts={})
            ninputs = dict(cohorts=dict(full=full, common=rows['common'], development=rows['development']))
            evidence = d.artifact(run/'inputs.json')
            def values(policy, subset):
                score = .8 if policy.startswith('P1') else .7 if policy.startswith('CL_NLL') else .6
                return {r['scene_id']:metrics(score) for r in subset}
            def dev_checked(run, cohort, mode, policy):
                self.assertNotEqual(task, 'mid-formal')
                return dict(metrics=values(policy,rows[cohort]),audit=evidence)
            def formal_checked(run, policy, block):
                self.assertNotEqual(task, 'sensitivity')
                subset = rows['common'] if block.startswith('common') else full
                return dict(metrics=values(policy,subset), evidence=evidence, ade_fde=dict(ade_4s=1.,fde_4s=2.))
            def interval(deltas, rows, samples=10000):
                mean = float(np.mean(list(deltas.values())))
                return dict(scene_mean=mean, scene_weighted_cluster_95=[mean,mean],
                    equal_log_mean=mean, equal_log_cluster_95=[mean,mean], seed_average_first=True)
            with patch.object(data,'checked_inputs',return_value=inputs), \
                 patch.object(data,'model_entry',return_value={}), \
                 patch.object(s,'checked_inputs',return_value=ninputs), \
                 patch.object(dev,'checked',side_effect=dev_checked), \
                 patch.object(native,'checked_formal',side_effect=formal_checked), \
                 patch.object(native,'require_bridge'), \
                 patch.object(intervals,'cluster_interval',side_effect=interval):
                first = reporting.report(run)
                second = reporting.report(run)
            self.assertEqual(first,second)
            self.assertEqual(first['status'],'PASS')
            self.assertEqual(first['decision'],'REVIEW_REQUIRED')
            self.assertEqual(first['complete_dev_conditions'], 72 if task != 'mid-formal' else 0)
            self.assertEqual(first['complete_formal_conditions'], 24 if task != 'sensitivity' else 0)
            self.assertEqual(first['complete_training_jobs'], 6 if task != 'mid-formal' else 0)
            if task != 'mid-formal':
                self.assertFalse(first['automatic_coefficient_selection'])
                self.assertTrue((run/'sensitivity.png').exists())
                self.assertEqual(len((run/'sensitivity_curves.csv').read_text().splitlines()),121)
            if task != 'sensitivity':
                self.assertEqual(first['comparisons']['full_NR']['MID__minus__scalar_v3']['delta']['score'],0)
                self.assertEqual(len((run/'summary.tsv').read_text().splitlines()),5)
                self.assertTrue(all(len(line.split('\t')) == 29 for line in (run/'summary.tsv').read_text().splitlines()))
            else:
                self.assertFalse((run/'summary.tsv').exists())
            self.assertTrue((run/'archive/release_manifest.json').exists())

    def test_complete_report_is_repeatable_retains_negative_results_and_29_columns(self):
        self.check_complete_report('sensitivity')

    def test_development_completes_without_formal_results_or_bridge(self):
        self.check_complete_report('sensitivity')

    def test_formal_completes_without_training_or_development_results(self):
        self.check_complete_report('mid-formal')


if __name__ == '__main__':
    unittest.main()
