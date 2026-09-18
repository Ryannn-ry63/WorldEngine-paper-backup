from unittest import TestCase, mock
from paper_study.hardware import hardware

class HardwareChecks(TestCase):
    def test_hopper_models_are_admitted_and_actual_names_recorded(self):
        names=['NVIDIA H100 80GB HBM3', 'NVIDIA H200', 'NVIDIA H200 NVL', 'NVIDIA H100 NVL']
        with mock.patch('paper_study.hardware.subprocess.check_output', side_effect=names):
            result=hardware(['0','1','2','3'])
        self.assertEqual([row['name'] for row in result], names)
        self.assertEqual([row['device'] for row in result], ['0','1','2','3'])

    def test_other_architectures_still_rejected(self):
        with mock.patch('paper_study.hardware.subprocess.check_output', return_value='NVIDIA A100'):
            with self.assertRaisesRegex(RuntimeError, 'H100 or H200'):
                hardware(['0'])
