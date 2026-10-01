import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import repair_run
import repair_queue

@pytest.mark.parametrize('status,expected', [('COMPLETE',0),('CORRECTNESS_STOPPED_AT_BOUNDARY',3),('BUDGET_EXHAUSTED_AT_BOUNDARY',4),('RUNNING',4)])
def test_exit_codes(status,expected):
    assert repair_run.result_exit_code({'status':status})==expected
    assert repair_run.result_exit_code({},True)==0

@pytest.mark.parametrize('status,expected',[('CORRECTNESS_STOPPED_AT_BOUNDARY',3),('COMPLETE',0)])
def test_cli_propagates_exit_to_platform(status,expected):
    code="import sys;sys.path.insert(0,%r);import repair_run as r;r.registration=lambda:{'jobs':[{'id':'test'}]};r.run=lambda *a,**k:{'status':%r};sys.argv=['repair_run','test'];r.main()"%(str(ROOT),status)
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==expected,result.stderr

def test_stale_failed_queue_cannot_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(repair_run,'REGISTRY',tmp_path)
    monkeypatch.setattr(repair_run,'registration',lambda:{})
    monkeypatch.setattr(repair_run,'REPAIR',tmp_path)
    (tmp_path/'evidence').mkdir()
    (tmp_path/'evidence/recovery_release.json').write_text('{}')
    monkeypatch.setattr(repair_run,'campaign_clock',lambda:{'deadline_unix':1e20})
    folder=tmp_path/'jobs/roots48';folder.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps({'status':'FAILED'}))
    with pytest.raises(BlockingIOError,match='explicit retry'):
        repair_run.run({'id':'roots48','kind':'diagnostic'},'0,1,2,3,4,5,6,7',wait=False)

def test_idle_queue_exits_without_sleep(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(repair_queue,'REGISTRY',tmp_path)
    monkeypatch.setattr(repair_queue,'registration',lambda:{'jobs':[]})
    monkeypatch.setattr(repair_queue,'campaign_clock',lambda:{'deadline_unix':1e20})
    monkeypatch.setattr(repair_queue,'summarize',lambda:None)
    monkeypatch.setattr(repair_queue.time,'sleep',lambda _:pytest.fail('Idle queue waited'))
    monkeypatch.setattr(sys,'argv',['queue','--idle-seconds','0'])
    repair_queue.main()
    assert 'QUEUE_IDLE_EXIT' in capsys.readouterr().out

def test_training_refuses_without_repaired_h100_diagnostic(tmp_path,monkeypatch):
    monkeypatch.setattr(repair_run,'REGISTRY',tmp_path)
    monkeypatch.setattr(repair_run,'registration',lambda:{})
    monkeypatch.setattr(repair_run,'REPAIR',tmp_path)
    (tmp_path/'evidence').mkdir()
    (tmp_path/'evidence/recovery_release.json').write_text('{}')
    with pytest.raises(RuntimeError,match='roots48 successfully'):
        repair_run.run({'id':'train_A_s0','kind':'train'},'0,1,2,3,4,5,6,7')
