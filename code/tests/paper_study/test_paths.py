import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
from paper_study import evaluation, supervised
from paper_study.common import artifact

class SharedPaths(TestCase):
    def test_evaluation_submits_physical_manifest_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);physical=root/'hdd2'/'run';physical.mkdir(parents=True)
            alias=root/'shared_run';alias.symlink_to(physical, target_is_directory=True)
            manifest=physical/'collections'/'bridge'/'collection.json'
            manifest.parent.mkdir(parents=True);manifest.write_text('{}')
            contract={'paper_code':{'version':'test'}}
            engine=SimpleNamespace(load_run=lambda name:(alias,contract,{},alias/'runtime'))
            with mock.patch.dict('sys.modules', {'run_study_evaluation':engine}), mock.patch.object(evaluation,'legacy_imports'), mock.patch.object(evaluation,'code_identity',return_value=contract['paper_code']):
                _,run,_,_,runtime=evaluation.load(alias)
            self.assertEqual(str(run/'collections'/'bridge'/'collection.json'),artifact(manifest)['path'])
            self.assertEqual(runtime,alias/'runtime')

    def test_training_environment_uses_registered_data_and_clean_code(self):
        inherited={'WORLDENGINE_ROOT':'/clean/code','PYTHONPATH':'/clean/dependencies'}
        module=SimpleNamespace(environment=lambda source:dict(inherited))
        with mock.patch.dict('sys.modules',{'run_selector_cfpi':module}), mock.patch.object(supervised,'legacy_imports'), mock.patch.object(supervised,'paths',return_value={'legacy_root':'/registered/assets'}):
            env=supervised.worker_env()
        self.assertEqual(env['WORLDENGINE_ROOT'],'/registered/assets')
        self.assertTrue(env['PYTHONPATH'].startswith(str(supervised.CODE)))
