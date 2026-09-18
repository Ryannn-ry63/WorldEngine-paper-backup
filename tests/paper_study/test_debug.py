from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
from unittest import TestCase,mock
from paper_study import debug

class DebugChecks(TestCase):
    def test_failed_stage_does_not_hide_other_failures_or_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            def child(args,log):
                calls.append(args)
                if args[0]=='prepare' and args[-1]=='B':raise RuntimeError('simulated B input failure')
            four=SimpleNamespace(devices=lambda x:x.split(','),hardware=lambda x:['H100']*4,
                resources=lambda x:{},cpu_capacity=lambda:80)
            def ensure(p):p.mkdir(parents=True,exist_ok=True)
            with mock.patch.dict('sys.modules',{'run_study_bridges_4h100':four}),mock.patch.object(debug,'legacy_imports'),mock.patch.object(debug,'check_hardware',return_value=['H200']*4),mock.patch.object(debug,'code_identity',return_value={'test':'version'}),mock.patch.object(debug,'run_path',side_effect=lambda x:root/x),mock.patch.object(debug,'ensure_run',side_effect=ensure),mock.patch.object(debug,'child',side_effect=child),mock.patch.object(debug,'parallel'),mock.patch.object(debug.signal,'signal'):
                with self.assertRaisesRegex(RuntimeError,'Some checks failed'):debug.execute('0,1,2,3')
            path=next(root.glob('*/summary.json'));result=json.loads(path.read_text())
            self.assertEqual(set(result['checks']),set('ABCDEF'))
            self.assertEqual(result['status'],'FAILED_CHECKS');self.assertEqual(result['checks']['B']['status'],'FAILED')
            self.assertFalse(result['formal_experiments_complete'])
            self.assertTrue(any(a[0]=='planner-train' and '--smoke' in a for a in calls))
            self.assertTrue(any(a[0]=='mechanism-collect' and '--smoke' in a for a in calls))
            self.assertFalse(any(a[0]=='train' and '--smoke' not in a for a in calls))
