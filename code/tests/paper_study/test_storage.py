import contextlib
import errno
import io
import tempfile
from pathlib import Path
from unittest import TestCase,mock
from paper_study.common import read,sha
from paper_study import storage

class Storage(TestCase):
    def test_snapshot_excludes_untracked_core_and_archives_incomplete_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();run=root/'run';run.mkdir()
            (source/'code.py').write_text('pass\n');(source/'core.123').write_bytes(b'dump')
            (source/'data').symlink_to(root/'external',target_is_directory=True)
            target=run/'runtime';target.mkdir();(target/'partial').write_text('failed')
            storage.snapshot(source,target,{'code.py':sha(source/'code.py')})
            self.assertTrue((target/'code.py').is_file());self.assertFalse((target/'core.123').exists())
            self.assertTrue((target/'data').is_symlink())
            self.assertEqual(len(list((run/'failed_preparations').glob('*/runtime/partial'))),1)
    def test_snapshot_failed_hash_never_publishes_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'x').write_text('x')
            with self.assertRaises(RuntimeError):storage.snapshot(source,root/'runtime',{'x':'wrong'})
            self.assertFalse((root/'runtime').exists());self.assertFalse(list(root.glob('.runtime_build_*')))
    def test_new_shared_storage_and_existing_local_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'registry').mkdir();(root/'registry/paper_storage.json').write_text('{}')
            spec=dict(runs_root=str(root/'shared/runs'))
            with mock.patch.object(storage,'WORKSPACE',root),mock.patch.object(storage,'settings',return_value=spec):
                run=root/'runs/new';storage.ensure_run(run);storage.ensure_run(run)
                self.assertTrue(run.is_symlink());self.assertTrue((run/'storage.json').exists())
                local=root/'runs/existing';local.mkdir();(local/'evidence').write_text('preserve')
                with self.assertRaises(RuntimeError):storage.ensure_run(local)
                self.assertEqual((local/'evidence').read_text(),'preserve')
    def test_quota_failure_does_not_mask_original_exception(self):
        with mock.patch.object(storage,'write',side_effect=OSError(errno.EDQUOT,'Disk quota exceeded')):
            stream=io.StringIO()
            with contextlib.redirect_stderr(stream):storage.stopped(Path('/tmp/example'),ValueError('original failure'))
            self.assertIn('original failure',stream.getvalue());self.assertIn('Unable to persist',stream.getvalue())
