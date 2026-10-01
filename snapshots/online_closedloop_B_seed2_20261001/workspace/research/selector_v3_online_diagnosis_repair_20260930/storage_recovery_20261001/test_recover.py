import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import recover as r


def contend(path, event, out):
    event.wait()
    try:
        with r.directory_lock(path):
            out.put('won')
            time.sleep(0.5)
    except BlockingIOError:
        out.put('busy')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp', prefix='recovery-test-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_atomic_lock_has_one_winner(self):
        ctx = mp.get_context('spawn')
        start, out = ctx.Event(), ctx.Queue()
        children = [ctx.Process(target=contend, args=(str(self.root/'job.lock'), start, out)) for _ in range(8)]
        for p in children:p.start()
        start.set()
        results = [out.get(timeout=15) for _ in children]
        for p in children:
            p.join(15)
            self.assertEqual(p.exitcode, 0)
        self.assertEqual(results.count('won'), 1)
        self.assertEqual(results.count('busy'), 7)
        self.assertFalse((self.root/'job.lock.mkdir').exists())

    def test_exception_releases_owned_lock(self):
        with self.assertRaises(RuntimeError):
            with r.directory_lock(self.root/'lock'):
                raise RuntimeError('failure')
        with r.directory_lock(self.root/'lock'):
            pass

    def test_foreign_or_crashed_lock_never_stolen(self):
        lock = self.root/'lock.mkdir'
        lock.mkdir()
        with self.assertRaises(BlockingIOError):
            with r.directory_lock(self.root/'lock'):pass
        self.assertTrue(lock.exists())

    def test_budget_keeps_original_cumulative_limit(self):
        c = r.budget_window(26720.66689090361, 1000)
        self.assertAlmostEqual(c['deadline_unix'], 18079.33310909639)
        self.assertEqual(c['cumulative_training_cap_seconds'], 43200)
        with self.assertRaises(RuntimeError):r.budget_window(43200, 1000)

    def test_retry_never_reexecutes_complete_running_or_new_failure(self):
        self.assertTrue(r.retry_allowed({'status':'FAILED'}, 'new'))
        for s in [{'status':'COMPLETE'}, {'status':'RUNNING'},
                  {'status':'FAILED','infrastructure_recovery':{'window':'new'}}]:
            self.assertFalse(r.retry_allowed(s, 'new'))
        self.assertTrue(r.retry_allowed({'status':'WAITING_FOR_EXPORT'}, 'new'))

    def test_archive_preserves_numeric_plans_logs_not_images_ipc(self):
        src, dst = self.root/'src', self.root/'dst'
        for name in ['simulation/WE_output/openscene_format/all_scenes.csv',
                     'simulation/WE_output/openscene_format/snapshot_records/plan.json',
                     'simulation/plan_traj/plan.pkl', 'simulator.log', 'process_status.json',
                     'simulation/WE_output/openscene_format/sensor_blobs/cam/image.jpg',
                     'simulation/frames/frame.pkl', 'simulation/merged_ann_files/a.pkl']:
            p=src/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
        (src/'link').symlink_to(src/'process_status.json')
        files=r.archive_tree(src,dst)
        self.assertEqual(len(files),5)
        self.assertFalse((dst/'simulation/frames').exists())
        self.assertFalse((dst/'simulation/WE_output/openscene_format/sensor_blobs').exists())
        for f in files:self.assertEqual(r.sha(f['path']),f['sha256'])

    def test_scratch_bridge_accepts_shared_copy_before_removing_local(self):
        localroot=self.root/'local';localroot.mkdir()
        folder=self.root/'attempt';folder.mkdir()
        def process(commands,*args,**kw):
            cmd=commands[0][0];out=Path(cmd[cmd.index('--output')+1]);out.mkdir()
            self.assertTrue(localroot in out.parents)
            (out/'scores.csv').write_text('token,score\na,0.5\n')
        def evaluation(job,folder,devices,deadline,dry):
            runner.run_processes([(['runtime','--output',str(folder/'evaluation')], '.', {})],folder,deadline)
            self.assertTrue((folder/'evaluation/scores.csv').exists())
            self.assertEqual((folder/'evaluation/scores.csv').read_text(),'token,score\na,0.5\n')
            return {'status':'COMPLETE'}
        runner=SimpleNamespace(run_processes=process,evaluation_job=evaluation)
        r.scratch_evaluation(runner)
        with patch.object(r,'scratch_base',return_value=localroot):
            result=runner.evaluation_job({'id':'job'},folder,'0,1',999,False)
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(list(localroot.iterdir()),[])
        self.assertTrue((folder/'evidence_archive.json').exists())

    def test_scratch_bridge_failure_keeps_logs_and_never_claims_success(self):
        localroot=self.root/'local';localroot.mkdir()
        folder=self.root/'attempt';folder.mkdir()
        def process(commands,*args,**kw):
            cmd=commands[0][0];out=Path(cmd[cmd.index('--output')+1]);out.mkdir()
            (out/'simulator.log').write_text('actual failure')
            raise RuntimeError('child failed')
        def evaluation(job,folder,devices,deadline,dry):
            runner.run_processes([(['runtime','--output',str(folder/'evaluation')],'.',{})],folder,deadline)
            return {'status':'COMPLETE'}
        runner=SimpleNamespace(run_processes=process,evaluation_job=evaluation)
        r.scratch_evaluation(runner)
        with patch.object(r,'scratch_base',return_value=localroot), self.assertRaises(RuntimeError):
            runner.evaluation_job({'id':'job'},folder,'0,1',999,False)
        self.assertEqual((folder/'evaluation/simulator.log').read_text(),'actual failure')
        self.assertFalse((folder/'scratch_released.json').exists())
        self.assertEqual(len(list(localroot.iterdir())),1)


if __name__ == '__main__':unittest.main()
