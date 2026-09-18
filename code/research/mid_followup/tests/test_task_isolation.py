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
import selector_mid_followup_common as b
import selector_mid_followup_forecast as forecast
import run_selector_mid_followup as runner


class SplitTests(unittest.TestCase):

    def test_wrong_stage_rejected_before_runner_can_mutate_run(self):
        for task, stages in [('mid-formal', ('train','dev-eval')), ('sensitivity', ('native-bridge','formal-eval'))]:
            for stage in stages:
                with self.assertRaisesRegex(RuntimeError, 'outside'):
                    b.validate_stage(task, stage)
        with self.assertRaises(RuntimeError):
            b.task_spec('typo')

    def test_all_dispatch_runs_only_its_line(self):
        for task in ('sensitivity', 'mid-formal'):
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
                                **({'scale':{}} if task != 'mid-formal' else {})),
                            'budget_forecast.json':dict(fits_budget=True),
                            'run_contract.json':dict(scorer={}),
                            'progress.json':dict(status='PASS')}[Path(path).name]
                with patch.object(runner.d, 'read', side_effect=read), patch.object(runner.d, 'verify'), \
                     patch.object(runner.data, 'checked_inputs'):
                    obj.dispatch()
                self.assertEqual(obj.training.call_count, int(task == 'sensitivity'))
                self.assertEqual(obj.bridge.call_count, int(task == 'mid-formal'))
                self.assertEqual(obj.dev_eval.call_count, 24 if task == 'sensitivity' else 0)
                self.assertEqual(obj.formal_eval.call_count, 18 if task == 'mid-formal' else 0)
                self.assertEqual(obj.helper.call_args.args, ('report',))

    def test_scoped_forecast_counts_only_relevant_gpu_work(self):
        ats = {f'eval_{cohort}_{mode}_MID_seed0':100. for cohort in ('development','common') for mode in b.MODES}
        ats['train_batch_0'] = 10.
        sts = {('open_'+forecast.s.digest(['--policy','P1_seed2','--block',block])[:12]
                if block.startswith('open') else block+'_P1_seed2'):200. for block in b.BLOCKS}
        sts.update({f'bridge_{mode}_P1_seed2':20. for mode in b.MODES})
        for task, seconds, kinds in [('sensitivity',2420.,{'development','training'}),
                                     ('mid-formal',3720.,{'formal','bridge'})]:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                c.locked_json(run/'run_contract.json', dict(task=task, gpu_hour_limit=128))
                (run/'source/train/MID_seed0').mkdir(parents=True)
                for folder in ('source','snapshot'):
                    c.locked_json(run/folder/'ledger.json', {})
                with patch.object(forecast.data, 'checked_inputs', return_value=dict(source_run=str(run/'source'),snapshot_run=str(run/'snapshot'))), \
                     patch.object(forecast, 'durations', return_value=ats if task=='sensitivity' else sts), \
                     patch.object(forecast, 'bytes_used', return_value=100):
                    result = forecast.forecast(run)
                self.assertEqual({t['kind'] for t in result['tasks']}, kinds)
                self.assertAlmostEqual(result['measured_component_wall_hours'], seconds/3600)
                self.assertAlmostEqual(result['planning_gpu_hours'], seconds*4/3600*1.35+4)

    def launcher(self, task, extra_env=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root/'job.sh'
            script.write_text((ROOT/'run_diffusiondrive_selector_mid_followup_4h100.sh').read_text())
            (root/'env/bin').mkdir(parents=True)
            python=root/'env/bin/python'
            python.write_text('#!/usr/bin/env bash\necho "CALL $*"\nif [[ "$2" == audit && "${FAIL_AUDIT:-0}" == 1 ]]; then exit 7; fi\n')
            python.chmod(0o755)
            env = dict(os.environ, WORLD_SIZE='1', LOCAL_WORLD_SIZE='1', SLURM_NTASKS='1',DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE=str(root/'env'))
            env.update(extra_env or {})
            return subprocess.run(['bash',str(script),task],env=env,capture_output=True,text=True)

    def test_platform_launch_audits_then_runs_scoped_all(self):
        for task,budget in [('sensitivity',48),('mid-formal',72)]:
            result = self.launcher(task)
            self.assertEqual(result.returncode,0,result.stderr)
            calls = [line for line in result.stdout.splitlines() if line.startswith('CALL ')]
            self.assertEqual(len(calls),2)
            self.assertTrue(' audit ' in calls[0])
            self.assertTrue(' all ' in calls[1])
            self.assertTrue(all(f'--task {task} --gpus 4 --gpu-hours {budget}' in line for line in calls))

    def test_platform_launch_stops_on_audit_failure_or_multi_process_launch(self):
        failed = self.launcher('mid-formal',{'FAIL_AUDIT':'1'})
        self.assertEqual(failed.returncode,7)
        self.assertNotIn(' all ',failed.stdout)
        multi = self.launcher('mid-formal',{'WORLD_SIZE':'4'})
        self.assertEqual(multi.returncode,2)
        self.assertNotIn('CALL',multi.stdout)


if __name__ == '__main__':
    unittest.main()
