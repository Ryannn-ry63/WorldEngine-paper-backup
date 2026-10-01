"""CPU fault-injection tests; not live H100 resume acceptance."""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3 import ddp_session_boundary as b
from innovation3.paths import sha256_file


def spec():
    return dict(world_size=2, episodes=4, seed=0, steps=8, branch_workers=8,
        source_sha256={'runner': 'source'}, parameterization='v3_initialized_selector_finetune',
        reward_contract='causal_h1_pdm_components_v1', dynamics_contract='upstream_log_play_h1_v1',
        ego_controller='log_play_controller', rows=[dict(scene_id='scene'+str(i),
            scene_sha256='s'+str(i), asset_sha256='a'+str(i),
            candidate_seed=20000+i, scene_seed=10000+i) for i in range(2)])


def report(tmp, s, index, rank):
    row = b.assignment(s, index, rank)
    namespace = (index*2+rank)*1000003
    path = tmp/('round%d_rank%d.json' % (index, rank))
    ckpt = path.with_suffix('.pt'); ckpt.write_bytes(b'checkpoint fixture')
    r = dict(status='PASS_STRICT_ONLINE_THROUGHPUT_PROBE', source_sha256=s['source_sha256'],
        parameterization=s['parameterization'], causal_reward_contract=s['reward_contract'],
        dynamics_contract=s['dynamics_contract'], ego_controller=s['ego_controller'], branch_workers=8,
        worker=dict(scene=row['scene_id'], source=dict(sha256=row['scene_sha256']),
                    asset=dict(sha256=row['asset_sha256'])),
        seed_namespaces=dict(training_seed=0, candidate_seed=row['candidate_seed']+namespace,
            scene_seed=row['scene_seed']+namespace, action_seed=row['candidate_seed']+namespace+17),
        candidate_branches=160, candidate_groups=8, idm_fallbacks=0,
        worker_exit=dict(alive=False, returncode=0), policy_version_initial=index*8,
        policy_version=(index+1)*8, update_attempts_initial=index*8, update_attempts=8,
        cumulative_update_attempts=(index+1)*8, actual_optimizer_steps=8,
        cumulative_optimizer_steps=(index+1)*8, initialization_provenance={'init':'offline'},
        learner_state_hash_initial='rank%d_boundary%d' % (rank,index),
        learner_state_hash_final='rank%d_boundary%d' % (rank,index+1), elapsed_seconds=1.,
        learner_checkpoint=dict(path=str(ckpt), sha256=sha256_file(ckpt), disposable=True),
        real_closed_loop=True, local_groups_with_signal=8,
        frozen_generator_and_original_selector_unchanged=True, frozen_reference_unchanged=True,
        online_selector_trainable=True, next_step_waited_for_update=True,
        events=[dict(kind='online_update', version_before=index*8+i, version_after=index*8+i+1,
          optimized=True, selected=0, update=dict(ddp_state_parity=True,reward_std=.2),
          branch_feedback=dict(canonical_restored=True,selected_parity=True,
            selected_reward_parity=True,selected_history_parity=True,rewards=[{'reward':.1}]*20))
          for i in range(8)])
    path.write_text(json.dumps(r)); return path


def commit_first(tmp):
    s=spec(); out=tmp/'summary.json'; state=b.open_boundary(out,s,False)
    records=[b.receipt(report(tmp,s,0,r),s,0,r) for r in range(2)]
    b.commit_round(out,s,state,records)
    return s,out,state


def test_resume_keeps_history_and_complete_resume_is_idempotent(tmp_path):
    s,out,state=commit_first(tmp_path)
    restored=b.open_boundary(out,s,True)
    assert restored==state
    records=[b.receipt(report(tmp_path,s,1,r),s,1,r,restored['rounds'][0][r]) for r in range(2)]
    b.commit_round(out,s,restored,records)
    assert b.finish(out,s,restored)==0
    saved=json.loads(out.read_text())
    assert len(saved['rounds'])==2 and saved['episodes_total']==4
    assert saved['full_resume_verified'] is False
    again=b.open_boundary(out,s,True)
    assert b.finish(out,s,again)==0
    assert json.loads(out.read_text())==saved


@pytest.mark.parametrize('change',['source','checkpoint','report','cursor','rank','legacy'])
def test_tampered_resume_fails_closed(tmp_path,change):
    s,out,state=commit_first(tmp_path)
    if change=='source': s['source_sha256']={'runner':'changed'}
    elif change=='checkpoint': Path(state['rounds'][0][0]['checkpoint']['path']).write_bytes(b'truncated')
    elif change=='report': Path(state['rounds'][0][0]['report']).write_text('{}')
    elif change=='cursor': state['completed']=4
    elif change=='rank': state['rounds'][0][1]['rank']=0
    elif change=='legacy': state['schema']=1
    if change in ('cursor','rank','legacy'): b.atomic_json(out.with_suffix('.progress.json'),state)
    with pytest.raises(ValueError): b.open_boundary(out,s,True)


@pytest.mark.parametrize('change',['continuity','ddp','history','namespace','attempts','execution'])
def test_episode_rejects_broken_contract(tmp_path,change):
    s,out,state=commit_first(tmp_path)
    path=report(tmp_path,s,1,0); r=json.loads(path.read_text())
    if change=='continuity': r['learner_state_hash_initial']='reset'
    elif change=='ddp': r['events'][0]['update']['ddp_state_parity']=False
    elif change=='history': r['events'][0]['branch_feedback']['selected_history_parity']=False
    elif change=='namespace': r['seed_namespaces']['action_seed']+=1
    elif change=='attempts': r['update_attempts_initial']=0
    elif change=='execution': r['dynamics_contract']='other'
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError): b.receipt(path,s,1,0,state['rounds'][0][0])


def test_no_boundary_lost_on_failed_atomic_replace(tmp_path):
    s,out,state=commit_first(tmp_path); progress=out.with_suffix('.progress.json')
    before=progress.read_bytes()
    records=[b.receipt(report(tmp_path,s,1,r),s,1,r,state['rounds'][0][r]) for r in range(2)]
    with patch('innovation3.provisional_session.os.replace',side_effect=OSError('injected failure')):
        with pytest.raises(OSError): b.commit_round(out,s,state,records)
    assert progress.read_bytes()==before
    assert b.open_boundary(out,s,True)['completed']==2


def test_missing_rank_and_different_schedule_rejected(tmp_path):
    s,out,state=commit_first(tmp_path); records=copy.deepcopy(state['rounds'][0])
    with pytest.raises(ValueError): b.validate_round(records[:1],s,0)
    records[1]['trace'][0][2]=False
    with pytest.raises(ValueError): b.validate_round(records,s,0)


def test_tied_session_not_promoted_to_learning_pass(tmp_path):
    s,out,state=commit_first(tmp_path)
    s['episodes']=2
    for r in state['rounds'][0]: r['learning_verified']=False
    assert b.finish(out,s,state)==2
    assert json.loads(out.read_text())['real_closed_loop'] is False


def test_fresh_run_refuses_existing_progress(tmp_path):
    s,out,state=commit_first(tmp_path)
    with pytest.raises(FileExistsError): b.open_boundary(out,s,False)


def test_runtime_routes_manifest_workers_zero_and_resume(tmp_path):
    import innovation3_runtime as runtime
    s=tmp_path/'s.json'; m=tmp_path/'m.json'; out=tmp_path/'out.json'
    argv=['runtime','--settings',str(s),'--scene-manifest',str(m),'--output',str(out),
          '--devices','0,1','--episodes','16','--branch-workers','0','--resume',
          '--stop-after-rounds','1','ddp-online-session-gate']
    with patch.object(sys,'argv',argv), patch.object(runtime,'environment',return_value=({'algengine_python':'python'},{})), \
            patch('innovation3.scene_manifest.load_assignments',return_value=({'steps':8},[])), \
            patch.object(runtime.subprocess,'run') as run:
        run.return_value.returncode=0
        assert runtime.main()==0
        cmd=run.call_args[0][0]
        assert cmd.count('--scene-manifest')==1
        assert cmd[cmd.index('--branch-workers')+1]=='0'
        assert '--resume' in cmd and '--stop-after-rounds' in cmd


def test_runner_stop_resume_and_completed_noop(tmp_path, monkeypatch):
    """Exercise the actual CLI loop with explicit CPU rank/visual doubles."""
    from types import SimpleNamespace
    import torch
    import torch.distributed as dist
    import innovation3
    from innovation3 import ddp_online_session as runner
    from innovation3 import online_probe
    s=spec()
    for row in s['rows']: row['scene_path']=str(tmp_path/'scene.pkl')
    settings=tmp_path/'settings.json'; settings.write_text(json.dumps({'online_parameterization':s['parameterization']}))
    manifest=tmp_path/'manifest.json'; manifest.write_text('{}')
    out=tmp_path/'result.json'; calls=[]
    def gather(target, value):
        target[0]=copy.deepcopy(value)
        if 'local_rank' in value:
            target[1]=dict(value,rank=1,local_rank=1,current_device=1)
        elif 'global_episode' in value:
            idx=value['global_episode']//2
            previous=None
            if idx: previous=json.loads(out.with_suffix('.progress.json').read_text())['rounds'][idx-1][1]
            target[1]=b.receipt(report(tmp_path,s,idx,1),s,idx,1,previous)
        else: target[1]=copy.deepcopy(value)
    def probe():
        argv=list(sys.argv); calls.append(argv)
        path=Path(argv[argv.index('--output')+1]); version=int(argv[argv.index('--policy-version')+1])
        fixture=report(tmp_path,s,version//8,0)
        path.write_text(fixture.read_text())
        return 0
    monkeypatch.setenv('RANK','0'); monkeypatch.setenv('WORLD_SIZE','2')
    monkeypatch.setenv('LOCAL_RANK','0'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    monkeypatch.setattr(runner,'load_assignments',lambda *a,**k: ({'steps':8},s['rows']))
    monkeypatch.setattr(online_probe,'source_hashes',lambda:s['source_sha256'])
    monkeypatch.setattr(online_probe,'main',probe)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:SimpleNamespace(uuid=None))
    for name in ('init_process_group','destroy_process_group','barrier','broadcast_object_list'):
        monkeypatch.setattr(dist,name,lambda *a,**k:None)
    monkeypatch.setattr(dist,'all_gather_object',gather)
    argv=['session','--settings',str(settings),'--scene-manifest',str(manifest),
          '--output',str(out),'--episodes','4']
    with patch.object(sys,'argv',argv+['--stop-after-rounds','1']): assert runner.main()==0
    assert not out.exists()
    assert json.loads(out.with_suffix('.progress.json').read_text())['completed']==2
    with patch.object(sys,'argv',argv+['--resume']): assert runner.main()==0
    assert len(calls)==2 and '--resume' in calls[1]
    assert json.loads(out.read_text())['episodes_total']==4
    before=out.read_bytes()
    with patch.object(sys,'argv',argv+['--resume']): assert runner.main()==0
    assert len(calls)==2 and out.read_bytes()==before
