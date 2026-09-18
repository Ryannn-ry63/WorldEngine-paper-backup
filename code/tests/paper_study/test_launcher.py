"""Batch entry must work before conda activation, including subprocess PATH."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CODE=Path(__file__).resolve().parents[2]

class BatchLauncher(unittest.TestCase):
    def test_all_six_queues_without_python_in_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The wrapper may use POSIX dirname, but cannot discover any Python.
            bindir=Path(tmp)/'bin';bindir.mkdir()
            (bindir/'dirname').symlink_to(shutil.which('dirname'))
            env=dict(os.environ,PATH=str(bindir))
            env.pop('STUDY_PYTHON',None);env.pop('CONDA_PREFIX',None)
            self.assertIsNone(shutil.which('python',path=env['PATH']))
            self.assertIsNone(shutil.which('python3',path=env['PATH']))
            for queue in 'ABCDEF':
                with self.subTest(queue=queue):
                    result=subprocess.run(['/bin/bash',str(CODE/'run_paper_study.sh'),'plan','--queue',queue],
                        cwd=tmp,env=env,capture_output=True,text=True,timeout=30)
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn(queue,json.loads(result.stdout)['queues'])
    def test_explicit_interpreter_populates_child_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe=Path(tmp)/'probe';probe.mkdir()
            (probe/'python').write_text('#!/bin/bash\nprintf "%s\\n" "$PATH"\n')
            (probe/'python').chmod(0o755)
            env=dict(os.environ,STUDY_PYTHON=str(probe/'python'),PATH='/usr/bin:/bin')
            result=subprocess.run(['/bin/bash',str(CODE/'run_paper_study.sh'),'plan'],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip().split(':')[0],str(probe))
    def test_missing_explicit_interpreter_fails_clearly(self):
        env=dict(os.environ,STUDY_PYTHON='/nonexistent/paper-python',PATH='/usr/bin:/bin')
        result=subprocess.run(['/bin/bash',str(CODE/'run_paper_study.sh'),'plan'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('not executable',result.stderr)

if __name__=='__main__':unittest.main()
