#!/usr/bin/env python3
"""Audit only committed reports, excluding uncommitted retry episodes."""
from analyze import *

def main():
    jobs=[];roundrows=[];allgroups=[];provenance=[];violations=[]
    for method in ['A','B']:
      for seed in range(3):
        jid=f'train_{method}_s{seed}';folder=REG/'jobs'/jid;p=read(folder/'training.progress.json');c=p['contract']
        assert len(p['rounds'])==113 and p['completed']==904
        visits=Counter();seen=set();counts=Counter();groups=[];rounds=[];profiles=defaultdict(list);sources=Counter();updates=[]
        for ri,records in enumerate(p['rounds']):
          assert len(records)==8
          rg=[]
          for rank,rec in enumerate(records):
            assert rec['global_episode']==ri*8+rank and rec['rank']==rank
            assert rec['report'] not in seen;seen.add(rec['report']);visits[rec['scene']]+=1
            assert sha(rec['report'])==rec['report_sha256'],rec['report']
            report=read(rec['report']);sources[hashlib.sha256(json.dumps(report['source_sha256'],sort_keys=True).encode()).hexdigest()]+=1
            assert report['causal_reward_contract']==c['reward_contract'] and report['online_update_timing']==c['online_update_timing']
            assert report['initialization_provenance']==rec['initialization_provenance']
            events=[e for e in report['events'] if 'branch_feedback' in e];assert len(events)==8
            us=[e['update'] for e in events] if method=='A' else [u for e in events for u in e['optimizer_updates']]
            assert len(us)==8 and sum(u['optimized'] for u in us)==rec['optimizer_steps']
            updates.extend(us)
            counts['local_signal_groups']+=sum(u['local_group_signal'] for u in us)
            counts['rank_optimizer_steps']+=sum(u['optimized'] for u in us)
            counts['candidate_branches']+=report['candidate_branches'];counts['candidate_simulation_steps']+=report['candidate_simulation_steps']
            for e in events:
              f=e['branch_feedback'];rewards=f['rewards'];assert len(rewards)==20 and f['scored']
              # Feedback spans the committed main step: history must advance.
              # Branch restoration is a separate canonical_restored receipt.
              assert f['canonical_restored']
              rv=np.array([v['reward'] for v in rewards]);assert np.isfinite(rv).all() and ((rv>=0)&(rv<=1)).all()
              probs=np.array(e['probabilities']);ref=softmax(e['reference_logits']);logits=np.array(e['current_logits']);cur=int(logits.argmax());parent=int(np.argmax(e['reference_logits']));selected=e['selected']
              sf=np.array([v['NC']==1 and v['DAC']==1 for v in rewards]);rt=np.array([any(v['first_violation_seconds'][k]==0 for k in ['NC','DAC']) for v in rewards])
              counts['root_violation_disagreement']+=int(rt.any()!=rt.all())
              g={'job':jid,'method':method,'seed':seed,'round':ri,'global_episode':rec['global_episode'],'scene':rec['scene'],'step':e['step'],
                 'selected_reward':float(rv[selected]),'argmax_reward':float(rv[cur]),'reference_argmax_reward':float(rv[parent]),'oracle_reward':float(rv.max()),
                 'argmax_oracle_gap':float(rv.max()-rv[cur]),'reference_oracle_gap':float(rv.max()-rv[parent]),'same_candidates_argmax_gain':float(rv[cur]-rv[parent]),
                 'expected_reward':float(probs@rv),'reference_expected_reward':float(ref@rv),'entropy':float(-np.sum(probs*np.log(np.clip(probs,1e-300,1)))),
                 'reference_entropy':float(-np.sum(ref*np.log(np.clip(ref,1e-300,1)))),'max_probability':float(probs.max()),'top2_logit_margin':float(np.sort(logits)[-1]-np.sort(logits)[-2]),
                 'behavior_kl_to_reference':float(np.sum(probs*(np.log(np.clip(probs,1e-300,1))-np.log(np.clip(ref,1e-300,1))))),
                 'safe_candidate_fraction':float(sf.mean()),'no_safe_candidates':int(not sf.any()),'all_zero':int(np.all(rv==0)),'root_unsafe':int(rt.any()),
                 'argmax_unsafe_with_safe_available':int(sf.any() and not sf[cur]),'sampled_unsafe_with_safe_available':int(sf.any() and not sf[selected]),
                 'reference_unsafe_with_safe_available':int(sf.any() and not sf[parent]),
                 'argmax_changed_from_reference':int(cur!=parent),'raw_EP_selected':rewards[selected]['EP'],'gated_EP_selected':rewards[selected]['gated_EP'],
                 'local_signal':int(us[e['step']]['local_group_signal']), 'update_kl':us[e['step']].get('kl'), 'policy_loss':us[e['step']].get('policy_loss'),
                 'selected_DAC':rewards[selected]['DAC'],'selected_NC':rewards[selected]['NC'],'selected_TTC':rewards[selected]['TTC'],
                 'selected_comfort':rewards[selected]['comfort'],'candidate_comfort_mean':mean([r['comfort'] for r in rewards]),
                 'behavior_version':e['version_before'],'update_version_before':us[e['step']].get('version_before',e['version_before'])}
              for metric in ['DAC','NC','TTC']:
                tm=rewards[selected]['first_violation_seconds'][metric]
                g['selected_'+metric+'_first_seconds']=tm
              groups.append(g);rg.append(g)
              for k,v in e['profile'].items():profiles[k].append(v)
            counts['reports']+=1
          # DDP replicas share optimizer steps: count rank 0 once, verify all ranks agree.
          assert len({r['optimizer_steps'] for r in records})==1
          counts['global_optimizer_steps']+=records[0]['optimizer_steps']
          rr={'job':jid,'method':method,'seed':seed,'round':ri,'episodes_completed':(ri+1)*8,'max_rank_elapsed_seconds':max(r['elapsed_seconds'] for r in records)}
          for k in rg[0]:
            if k not in ['job','method','seed','round','scene','global_episode','step']:
              values=[g[k] for g in rg if g[k] is not None];rr[k]=mean(values)
          rounds.append(rr)
        assert dict(visits)==p['coverage']['scene_visits']
        markers=[]
        for passes in [3,6]:
          mp=folder/f'training.coverage{passes}.json';m=read(mp);assert m['contract']==c and m['final_records']==p['rounds'][m['completed']//8-1]
          cp=[]
          for r in m['final_records']:
            assert sha(r['checkpoint']['path'])==r['checkpoint']['sha256'];cp.append(r['checkpoint'])
          markers.append({'passes':passes,'marker':str(mp),'sha256':sha(mp),'completed':m['completed'],'checkpoints':cp})
        summary={'job':jid,'method':method,'seed':seed,**counts,'unique_scenes':len(visits),'min_visits':min(visits.values()),'max_visits':max(visits.values()),
          'committed_round_max_sum_seconds':sum(r['max_rank_elapsed_seconds'] for r in rounds),'registered_training_wall_seconds':p['training_wall_seconds'],'resumed_segments':p['resumed_segments'],
          'local_groups':len(groups),'source_variants':dict(sources),'markers':markers,'contract':{k:v for k,v in c.items() if k not in ['rows','source_sha256']},
          'profile_mean_seconds':{k:mean(v) for k,v in profiles.items()}}
        for k in groups[0]:
          if k not in ['job','method','seed','round','global_episode','scene','step']:
            summary[k]=mean([g[k] for g in groups if g[k] is not None])
        summary['zero_groups_root_unsafe']=sum(g['all_zero'] and g['root_unsafe'] for g in groups)
        summary['no_safe_groups_root_safe']=sum(g['no_safe_candidates'] and not g['root_unsafe'] for g in groups)
        summary['update_drift_max']=max(u.get('training_inference_max_abs_logit_drift',0) for u in updates)
        provenance.append({'job':jid,'initialization':rec['initialization_provenance'],'source_sha256':c['source_sha256'],'status':read(folder/'status.json')})
        jobs.append(summary);roundrows.extend(rounds);allgroups.extend(groups)
        print(jid,'reports',counts['reports'],'global updates',counts['global_optimizer_steps'],'zero',summary['all_zero'],'KL',summary['update_kl'],flush=True)
    dump('training_summary.json',jobs);dump('training_provenance.json',provenance);table('training_rounds.csv',roundrows);table('training_groups.csv',allgroups)
    # Fixed scheduling pass bins: comparable coverage, not identical visited states.
    bins=[]
    for j in jobs:
      for start,end in [(0,150),(150,300),(300,450),(450,600),(600,750),(750,904)]:
        gs=[g for g in allgroups if g['job']==j['job'] and start<=g['global_episode']<end]
        row={'job':j['job'],'episode_start':start,'episode_end_exclusive':end,'groups':len(gs)}
        for k in gs[0]:
          if k not in ['job','method','seed','round','global_episode','scene','step']:
            row[k]=mean([g[k] for g in gs if g[k] is not None])
        bins.append(row)
    table('training_pass_bins.csv',bins);dump('training_source_hashes.json',HASHES)

if __name__=='__main__': main()
