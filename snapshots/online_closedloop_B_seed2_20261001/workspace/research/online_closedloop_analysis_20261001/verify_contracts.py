#!/usr/bin/env python3
"""Recheck saved scheduling, seed namespaces, feedback and update continuity.

Does not re-execute simulation or deserialize every historical checkpoint.
"""
from analyze import *
import importlib.util

def rank_corr(a,b):
    def ranks(x):
        _,inv,counts=np.unique(x,return_inverse=True,return_counts=True)
        return (np.cumsum(counts)-(counts+1)/2)[inv]
    a=ranks(a);b=ranks(b)
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None

def main():
    protocol_path=REG.parent/'code/projects/AlgEngine/scripts/diffusiondrive/innovation3/fixed_protocol.py'
    spec=importlib.util.spec_from_file_location('frozen_fixed_protocol',protocol_path);protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
    out={'protocol_path':str(protocol_path),'protocol_sha256':sha(protocol_path),'jobs':[]};assignments={}
    for method in ['A','B']:
      for seed in range(3):
        jid=f'train_{method}_s{seed}';p=read(REG/'jobs'/jid/'training.progress.json');c=p['contract'];prev=[None]*8;corr=defaultdict(list);counts=Counter();schedule=[]
        ordered=sorted(c['rows'],key=lambda r:hashlib.sha256(('%d:%s'%(seed,r['scene_id'])).encode()).hexdigest())
        for ri,records in enumerate(p['rounds']):
          assert len({str(r['trace']) for r in records})==1
          for rank,rec in enumerate(records):
            row=ordered[(ri*8+rank)%len(ordered)];assert rec['scene']==row['scene_id'];schedule.append(rec['scene'])
            report=read(rec['report']);assert sha(rec['report'])==rec['report_sha256']
            assert report['source_sha256']==c['source_sha256'];worker=report['worker']
            assert worker['scene']==row['scene_id'] and worker['source']['sha256']==row['scene_sha256'] and worker['asset']['sha256']==row['asset_sha256']
            ns=ri*8+rank+seed*1000003
            assert report['seed_namespaces']==dict(training_seed=seed,candidate_seed=(row['candidate_seed']+ns*1000003)%2**31,scene_seed=(row['scene_seed']+ns*1000003)%2**31,action_seed=(row['candidate_seed']+ns*1000003+17)%2**31)
            if prev[rank]:assert report['learner_state_hash_initial']==prev[rank]['learner_hash']
            assert report['policy_version_initial']==ri*8 and report['cumulative_optimizer_steps']==(ri+1)*8
            assert report['branch_render_calls']==report['branch_model_calls']==report['idm_fallbacks']==0
            for key in ['frozen_generator_and_original_selector_unchanged','frozen_reference_unchanged','online_selector_trainable']:assert report[key]
            es=[e for e in report['events'] if 'branch_feedback' in e];assert len(es)==8
            us=[e['update'] for e in es] if method=='A' else [u for e in es for u in e['optimizer_updates']]
            assert len(us)==8 and all(u['ddp_state_parity'] for u in us)
            if method=='A':
              assert report['next_step_waited_for_update']
              assert all(e['version_before']==ri*8+i and e['version_after']==ri*8+i+1 for i,e in enumerate(es))
            else:
              assert all(e['version_before']==ri*8 for e in es)
              assert all(not e['optimizer_updates'] and e['version_after']==ri*8 and e['online_selector_hash_before']==e['online_selector_hash_after'] for e in es[:7])
              assert [u['collection_index'] for u in us]==list(range(8))
              assert all(u['context_hash']==e['context_hash'] and u['version_before']==ri*8+i and u['version_after']==ri*8+i+1 for i,(u,e) in enumerate(zip(us,es)))
            previous_history=None
            for e in es:
              feedback=e['branch_feedback']
              protocol.validate(e['observation_identity'],e['selected'],e['candidate_hash'],e['main_receipt'],feedback,True,history_hash=previous_history,expected_contract=c['reward_contract'])
              previous_history=feedback['history_after_hash']
              assert np.max(np.abs(softmax(e['current_logits'])-e['probabilities']))<2e-6
              rw=[r['reward'] for r in feedback['rewards']]
              for pol,key in [('current','current_logits'),('reference','reference_logits')]:
                v=rank_corr(e[key],rw)
                if v is not None:corr[pol].append(v)
              counts['groups']+=1
            prev[rank]=rec;counts['reports']+=1
        assignments[jid]=schedule
        out['jobs'].append({'job':jid,**counts,'current_spearman_mean_nonconstant':mean(corr['current']),'reference_spearman_mean_nonconstant':mean(corr['reference']),'nonconstant_groups_current':len(corr['current']),'nonconstant_groups_reference':len(corr['reference']),'status':'PASS_SAVED_CONTRACTS'})
        print(jid,'PASS',flush=True)
    assert all(assignments[f'train_A_s{s}']==assignments[f'train_B_s{s}'] for s in range(3))
    out['paired_A_B_scene_schedules_identical']=True
    verify_evaluations(out)

def verify_evaluations(out):
    # Evaluate only the accepted set in the existing analysis snapshot.
    audit=read(OUT/'audit.json');prepared=read(REG/'development/prepared.json')
    for k in ['config','scenarios']:assert sha(prepared[k])==prepared[k+'_sha256']
    config_cases=[]
    for jid,item in audit['evaluations'].items():
      result=read(item['result']);inputs=read(item['inputs']);assert inputs['configuration_sha256']==prepared['config_sha256']
      commands_path=Path(result['output'])/'commands.json';commands=read(commands_path)
      simulators=[r['argv'] for r in commands if any(str(x).endswith('/run_simulation.py') for x in r['argv'])]
      planners=[r['argv'] for r in commands if any(str(x).endswith('/sim_test.py') for x in r['argv'])]
      assert len(simulators)==1 and len(planners)==8
      sim=simulators[0];assert 'data_file_path='+prepared['scenarios'] in sim and 'ego_controller=log_play_controller' in sim
      if jid.endswith('_R'):
        assert 'agent_policy=idm_policy' in sim and 'agent_navigation=idm_navigation' in sim
      else:
        # NR intentionally uses frozen default_runner.yaml trajectory_policy.
        assert not any(a.startswith('agent_policy=') for a in sim)
        default=REG.parent/'code/projects/SimEngine/worldengine/configs/default_runner.yaml'
        assert "agent_policy: 'trajectory_policy'" in default.read_text()
      for cmd in planners:
        assert prepared['config'] in cmd and result['artifact']['path'] in cmd
        assert cmd[cmd.index('--seed')+1]=='0'
      config_cases.append({'job':jid,'commands':str(commands_path),'commands_sha256':sha(commands_path),'config_sha256':prepared['config_sha256'],'scenarios_sha256':prepared['scenarios_sha256']})
    out['evaluation_config_checks']=config_cases
    out['evaluation_audit_time']=audit['created_utc']
    out['status']='PASS';dump('contract_verification.json',out)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations-only',action='store_true',help='Reuse previously completed training verification; refresh evaluation configuration checks.')
    args=parser.parse_args()
    if args.evaluations_only:
        previous=read(OUT/'contract_verification.json')
        assert previous['status']=='PASS' and len(previous['jobs'])==6
        assert sum(j['groups'] for j in previous['jobs'])==43392
        verify_evaluations(previous)
    else:main()
