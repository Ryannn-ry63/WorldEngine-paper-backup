"""Regression checks for optional control files in the H100 preflight."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import run


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.stop = root / 'correctness_failure.json'
        self.cfg = dict(algengine_python=str(root), data_root=str(root),
                        campaign_stop_file=str(self.stop),
                        campaign_deadline_file=str(root/'unused_old_clock.json'),
                        extra_python_paths=[str(root)])

    def test_missing_control_files_reaches_hardware_and_import_checks(self):
        runner = SimpleNamespace(configured_environment=lambda devices: (self.cfg, {}),
                                 CODE=self.temp.name, preflight=Mock())
        with patch.object(run.recover, 'scratch_base') as scratch, \
                patch.object(run.subprocess, 'run') as hardware:
            run.preflight(runner, '0,1,2,3,4,5,6,7')
        scratch.assert_called_once()
        hardware.assert_called_once()
        runner.preflight.assert_called_once()

    def test_present_stop_marker_blocks(self):
        self.stop.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'Correctness stop remains active'):
            run.validate_runtime_paths(self.cfg)

    def test_missing_real_input_still_blocks(self):
        self.cfg['data_root'] += '/missing_data'
        with self.assertRaisesRegex(FileNotFoundError, 'data_root'):
            run.validate_runtime_paths(self.cfg)

    def test_missing_extra_import_path_blocks(self):
        self.cfg['extra_python_paths'] = [self.temp.name + '/missing_library']
        with self.assertRaisesRegex(FileNotFoundError, 'extra_python_paths'):
            run.validate_runtime_paths(self.cfg)


if __name__ == '__main__':
    unittest.main()
