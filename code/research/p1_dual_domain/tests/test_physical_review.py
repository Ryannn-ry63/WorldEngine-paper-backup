import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'projects/AlgEngine/scripts/diffusiondrive'))
from selector_dual_domain_state import NUMERIC,CATEGORICAL,physical_equal,physical_state
from selector_dual_domain_analysis import Evidence
from selector_dual_domain_review import review
import cfpi_common as c


class PhysicalReviewTests(unittest.TestCase):
    def test_exact_and_state_drift(self):
        left={k:np.zeros(1) for k in (*NUMERIC,*CATEGORICAL)}
        self.assertTrue(physical_equal(left,copy.deepcopy(left)))
        for key in NUMERIC:
            right=copy.deepcopy(left);right[key][0]=1
            self.assertFalse(physical_equal(left,right),key)
        right=copy.deepcopy(left);right['gt_names']=np.array(['truck'])
        self.assertFalse(physical_equal(left,right))

    def test_nonfinite_or_wrong_shape_is_not_same_state(self):
        left={k:np.zeros(1) for k in (*NUMERIC,*CATEGORICAL)}
        for value in (np.array([float('nan')]),np.zeros(2)):
            right=copy.deepcopy(left);right['can_bus']=value
            self.assertFalse(physical_equal(left,right))

    def test_wrong_frame_or_foreign_root(self):
        class Reader:pass
        reader=Reader();reader.ev=Evidence()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'frame.pkl'
            record={k:np.zeros(1) for k in (*NUMERIC,*CATEGORICAL)}
            record.update(scene_name='scene',frame_idx=3)
            c.atomic_pickle(p,record)
            sidecar={'raw_observation_path':str(p)}
            self.assertEqual(physical_state(reader,sidecar,root,'scene',4)['frame_idx'],3)
            with self.assertRaises(RuntimeError):physical_state(reader,sidecar,root,'scene',5)
            with self.assertRaises(RuntimeError):physical_state(reader,sidecar,root/'foreign','scene',4)


if __name__=='__main__':unittest.main()
