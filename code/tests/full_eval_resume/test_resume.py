import importlib.util
from pathlib import Path
import sys
import tempfile
from unittest import TestCase

CODE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(CODE/'scripts'))
spec=importlib.util.spec_from_file_location('resume_under_test',CODE/'scripts/resume_h100_full_eval.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class ResumePaths(TestCase):
    def make_result(self, base):
        original=base/'original';destination=base/'hdd2';original.mkdir();destination.mkdir()
        r.write(original/'run_contract.json',{'frozen':True})
        rows=[{'scene_id':'s'}];model={'path':'checkpoint','sha256':'modelhash'}
        inputs={'models':{'D2_seed2':model},'cohorts':{'full':rows}}
        r.write(original/'inputs.json',inputs);r.attach_inputs(destination,original)
        key='full_R_D2_seed2';folder=destination/'collections'/key;folder.mkdir(parents=True)
        r.write(folder/'collection.json',dict(id=key,contract=r.artifact(destination/'run_contract.json'),checkpoint=model,rows=rows))
        r.write(folder/'audit.json',dict(status='PASS',collection=r.artifact(folder/'collection.json'),incomplete_or_dropped_scenes=0))
        r.write(folder/'accepted.json',dict(status='PASS',audit=r.artifact(folder/'audit.json'),actual_actions_verified=True))
        return original,destination,inputs,key

    def test_physical_output_keeps_original_contract_identity(self):
        with tempfile.TemporaryDirectory() as d:
            original,destination,inputs,key=self.make_result(Path(d))
            r.validate_result(destination/'collections'/key,original,inputs,'D2_seed2','full','R')
            self.assertEqual(r.artifact(destination/'run_contract.json'),r.artifact(original/'run_contract.json'))

    def test_partial_preserved_and_accepted_result_published_idempotently(self):
        with tempfile.TemporaryDirectory() as d:
            original,destination,inputs,key=self.make_result(Path(d))
            partial=original/'collections'/key;partial.mkdir(parents=True)
            (partial/'failed.log').write_bytes(b'original quota failure')
            r.publish(original,destination,key);r.publish(original,destination,key)
            self.assertEqual(partial.resolve(),(destination/'collections'/key).resolve())
            archives=list((original/'archive/quota_interrupted').glob('*/failed.log'))
            self.assertEqual(len(archives),1);self.assertEqual(archives[0].read_bytes(),b'original quota failure')

    def test_existing_accepted_result_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            original,destination,_,key=self.make_result(Path(d))
            existing=original/'collections'/key;existing.mkdir(parents=True)
            (existing/'accepted.json').write_text('original')
            with self.assertRaisesRegex(RuntimeError,'accepted/audited'):
                r.publish(original,destination,key)
            self.assertEqual((existing/'accepted.json').read_text(),'original')

    def test_incomplete_output_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(RuntimeError,'Only accepted'):
                r.publish(root/'original',root/'hdd2','partial')

    def test_wrong_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            original,destination,inputs,key=self.make_result(Path(d))
            inputs['models']['D2_seed2']={'path':'other','sha256':'other'}
            with self.assertRaisesRegex(RuntimeError,'provenance mismatch'):
                r.validate_result(destination/'collections'/key,original,inputs,'D2_seed2','full','R')
