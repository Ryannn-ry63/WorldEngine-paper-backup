"""Migration routing tests: no real project mutation or GPU use."""
from pathlib import Path
import tempfile
import unittest
import json
import sys
from unittest.mock import patch
import owned_task_runner as runner


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.personal = self.root / 'personal'; self.personal.mkdir()
        self.data = self.personal / 'data'
        self.v1 = self.root / 'data_v1'; self.v2 = self.root / 'data_v2'
        for root in (self.v1,self.v2):
            for relative,_ in runner.DATA_ALIASES:
                p=root/relative
                if p.suffix=='.pkl':p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'scenarios')
                else:p.mkdir(parents=True,exist_ok=True)
            for relative in runner.DATA_ENDPOINTS:(root/relative).mkdir(parents=True,exist_ok=True)
        self.data.symlink_to(self.v1,target_is_directory=True)
        self.code = self.personal/'dependency';self.code.mkdir()
        self.patches=[patch.object(runner,'DATA',self.data),
                      patch.object(runner,'PERSONAL',self.personal),
                      patch.object(runner.owned,'O',self.personal/'owned'),
                      patch.object(runner.owned,'mappings',return_value=[(self.code,'/legacy/code')])]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def test_relink_is_resolved_each_time_without_cached_old_target(self):
        first=runner.make_view()
        self.data.unlink();self.data.symlink_to(self.v2,target_is_directory=True)
        second=runner.make_view()
        self.assertEqual(runner.map_back(runner.OLD_DATA+'/alg_engine/example.pkl',first)[0],self.v1/'alg_engine/example.pkl')
        self.assertEqual(runner.map_back(runner.OLD_DATA+'/alg_engine/example.pkl',second)[0],self.v2/'alg_engine/example.pkl')
        self.assertTrue(self.v1.exists())

    def test_nested_broken_link_fails_without_old_project_fallback(self):
        p=self.v1/'raw/nuplan/dataset/maps';p.rmdir();p.symlink_to(self.root/'missing')
        with self.assertRaisesRegex(RuntimeError,'Required data endpoint missing'):runner.make_view()

    def test_external_nondata_dependency_is_rejected(self):
        self.code.rmdir();self.code.symlink_to(self.v1)
        with self.assertRaisesRegex(RuntimeError,'Non-data dependency outside'):runner.make_view()

    def test_specific_owned_alias_and_data_exception_under_hidden_project(self):
        with patch.object(runner.owned,'mappings',return_value=[(self.code,'/inspire/hdd/project/senior/code')]):
            entries=runner.make_view()
        self.assertEqual(runner.map_back('/inspire/hdd/project/senior/code/module.py',entries),(self.code/'module.py','owned'))
        self.assertEqual(runner.map_back('/inspire/hdd/project/unknown/secret',entries)[1],'hidden')
        self.assertEqual(runner.map_back(runner.OLD_DATA+'/alg_engine/ckpts/model.pth',entries)[1],'hidden')
        self.assertEqual(runner.map_back(runner.FULL_SCENARIO,entries)[0],self.v1/'sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl')

    def test_only_completed_training_queues_select_eval(self):
        for task in 'BCF':self.assertEqual(runner.command(task,'0,1,2,3')[-1],'eval')
        self.assertEqual(runner.command('D','0,1,2,3')[-1],'all')
        self.assertIn('run_h100_full_eval_resume.sh',runner.command('prior','0,1,2,3')[1])



class InterruptedCollections(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0,str(runner.C/'scripts'))
        self.tmp=tempfile.TemporaryDirectory();self.w=Path(self.tmp.name)
        self.root=self.w/'runs/20260917_paper_B_v3_eval';self.root.mkdir(parents=True)
        (self.root/'run_contract.json').write_text('{"frozen":true}')
        from paper_study.common import artifact
        self.model={'path':'checkpoint','sha256':'modelhash'};self.rows=[{'scene_id':'scene'}]
        (self.root/'inputs.json').write_text(json.dumps({'models':{'model':self.model},'cohorts':{'full':self.rows}}))
        self.condition=self.root/'collections/full_NR_model';(self.condition/'logs').mkdir(parents=True)
        (self.condition/'logs/simulator.log').write_bytes(b'preserve interrupted evidence')
        self.co=dict(id='full_NR_model',policy='model',cohort='full',mode='NR',contract=artifact(self.root/'run_contract.json'),checkpoint=self.model,rows=self.rows)
        (self.condition/'collection.json').write_text(json.dumps(self.co))
        self.override=patch.object(runner.owned,'W',self.w);self.override.start()
    def tearDown(self):
        self.override.stop();self.tmp.cleanup()
    def test_interrupted_payload_preserved_for_recovery(self):
        runner.archive_partial_evaluations('B')
        self.assertFalse(self.condition.exists())
        files=list((self.root/'archive/migration_interrupted').glob('*/logs/simulator.log'))
        self.assertEqual(len(files),1);self.assertEqual(files[0].read_bytes(),b'preserve interrupted evidence')
    def test_audited_condition_is_not_archived(self):
        (self.condition/'audit.json').write_text('{}')
        runner.archive_partial_evaluations('B')
        self.assertTrue((self.condition/'audit.json').exists())
        self.assertFalse((self.root/'archive').exists())
    def test_unknown_partial_identity_is_not_archived(self):
        self.co['checkpoint']={'path':'other','sha256':'wrong'}
        (self.condition/'collection.json').write_text(json.dumps(self.co))
        with self.assertRaisesRegex(RuntimeError,'Unrecognized partial'):runner.archive_partial_evaluations('B')
        self.assertTrue((self.condition/'logs/simulator.log').exists())

if __name__=='__main__':unittest.main()
