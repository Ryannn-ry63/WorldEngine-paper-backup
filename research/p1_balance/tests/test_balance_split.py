"""Task isolation, scoped budget and unattended launcher regression checks."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_balance_common as b
import selector_balance_forecast as forecast
import run_selector_balance as runner


class SplitTests(unittest.TestCase):
    def test_scoped_queues_partition_original_without_changing_order(self):
        dev, formal = b.queue('development'), b.queue('formal')
        self.assertEqual(len(dev), 24)
        self.assertEqual(len(formal), 30)
        self.assertFalse(set(dev) & set(formal))
        self.assertEqual(set(dev) | set(formal), set(b.queue()))
        self.assertTrue(all(t[0] == 'dev' for t in dev))
        self.assertTrue(all(t[0] == 'formal' and not t[1].startswith('MID') for t in formal))
        self.assertEqual(b.task_spec('formal')['expected_training_jobs'], 0)
        self.assertEqual(b.task_spec('development')['expected_bridge_conditions'], 0)

    def test_wrong_stage_rejected_before_runner_can_mutate_run(self):
        for task, stages in [('formal', ('train','dev-eval')), ('development', ('native-bridge','formal-eval'))]:
            for stage in stages:
                with self.assertRaisesRegex(RuntimeError, 'outside'):
                    b.validate_stage(task, stage)
        with self.assertRaises(RuntimeError):
            b.task_spec('typo')

    def test_all_dispatch_runs_only_its_line(self):
        for task in ('development', 'formal'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                obj = runner.Runner.__new__(runner.Runner)
                obj.run = Path(directory)
                obj.args = SimpleNamespace(task=task, stage='all')
                obj.code_sha = 'test'
                obj.env = {}
                for name in ('preflight','training','bridge','dev_eval','formal_eval','helper','record'):
                    setattr(obj, name, Mock())
                def read(path):
                    return {'audit_report.json':dict(status='PASS', code_sha='test', inputs={},
                                native_inputs={}, parity={}, code_archive={}, forecast={},
                                **({'scale':{}} if task != 'formal' else {})),
                            'budget_forecast.json':dict(fits_budget=True),
                            'run_contract.json':dict(scorer={}),
                            'progress.json':dict(status='PASS')}[Path(path).name]
                with patch.object(runner.d, 'read', side_effect=read), patch.object(runner.d, 'verify'), \
                     patch.object(runner.data, 'checked_inputs'):
                    obj.dispatch()
                self.assertEqual(obj.training.call_count, int(task == 'development'))
                self.assertEqual(obj.bridge.call_count, int(task == 'formal'))
                self.assertEqual(obj.dev_eval.call_count, 24 if task == 'development' else 0)
                self.assertEqual(obj.formal_eval.call_count, 30 if task == 'formal' else 0)
                self.assertEqual(obj.helper.call_args.args, ('report',))

    def test_scoped_forecast_counts_only_relevant_gpu_work(self):
        ats = {f'eval_{cohort}_{mode}_CL_NLL_seed0':100. for cohort in ('development','common') for mode in b.MODES}
        ats['train_seed0'] = 10.
        sts = {('open_'+forecast.s.digest(['--policy','P1_seed2','--block',block])[:12]
                if block.startswith('open') else block+'_P1_seed2'):200. for block in b.BLOCKS}
        sts.update({f'bridge_{mode}_P1_seed2':20. for mode in b.MODES})
        for task, seconds, kinds in [('development',2420.,{'development','training'}),
                                     ('formal',6200.,{'formal','bridge'})]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                c.locked_json(run/'run_contract.json', dict(task=task, gpu_hour_limit=128))
                for folder in ('attribution','snapshot'):
                    c.locked_json(run/folder/('decision_ledger.json' if folder == 'attribution' else 'ledger.json'), {})
                with patch.object(forecast.data, 'checked_inputs', return_value=dict(attribution_run=str(run/'attribution'),snapshot_run=str(run/'snapshot'))), \
                     patch.object(forecast, 'durations', side_effect=[ats,sts]), \
                     patch.object(forecast, 'bytes_used', return_value=100):
                    result = forecast.forecast(run)
                self.assertEqual({t['kind'] for t in result['tasks']}, kinds)
                self.assertAlmostEqual(result['measured_component_wall_hours'], seconds/3600)
                self.assertAlmostEqual(result['planning_gpu_hours'], seconds*4/3600*1.35+4)

    def launcher(self, task, extra_env=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root/'job.sh'
            script.write_text((ROOT/'run_diffusiondrive_selector_p1_task_4h100.sh').read_text())
            (root/'run_diffusiondrive_selector_p1_balance_4h100.sh').write_text(
                'echo "CALL $*"\nif [[ "$1" == audit && "${FAIL_AUDIT:-0}" == 1 ]]; then exit 7; fi\n')
            env = dict(os.environ, WORLD_SIZE='1', LOCAL_WORLD_SIZE='1', SLURM_NTASKS='1')
            env.update(extra_env or {})
            return subprocess.run(['bash',str(script),task],env=env,capture_output=True,text=True)

    def test_platform_launch_audits_then_runs_scoped_all(self):
        for task,budget in [('development',64),('formal',128)]:
            result = self.launcher(task)
            self.assertEqual(result.returncode,0,result.stderr)
            calls = [line for line in result.stdout.splitlines() if line.startswith('CALL ')]
            self.assertEqual(len(calls),2)
            self.assertTrue(calls[0].startswith('CALL audit '))
            self.assertTrue(calls[1].startswith('CALL all '))
            self.assertTrue(all(f'--task {task} --gpus 4 --gpu-hours {budget}' in line for line in calls))

    def test_platform_launch_stops_on_audit_failure_or_multi_process_launch(self):
        failed = self.launcher('formal',{'FAIL_AUDIT':'1'})
        self.assertEqual(failed.returncode,7)
        self.assertNotIn('CALL all',failed.stdout)
        multi = self.launcher('formal',{'WORLD_SIZE':'4'})
        self.assertEqual(multi.returncode,2)
        self.assertNotIn('CALL',multi.stdout)


if __name__ == '__main__':
    unittest.main()
