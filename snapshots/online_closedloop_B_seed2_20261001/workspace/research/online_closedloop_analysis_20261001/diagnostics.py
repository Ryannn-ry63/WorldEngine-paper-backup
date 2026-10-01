#!/usr/bin/env python3
from analyze import *

def main():
    s=read(REG/'jobs/roots48/status.json');assert s['status']=='COMPLETE'
    folder=Path(s['attempt']);roots=[];cont=[];parity=[]
    for path in sorted(folder.glob('rank*/probe/*/report.json')):
      r=read(path)
      if r.get('forced'):
        f=r['forced'];v=r['closed']['continuation']['training']
        cont.append({'scene':r['scene']['scene_id'],'decision':f['root_decision'],'choice':f['choice'],'fixed':f['fixed_reward'],'replanned':v['reward'],'delta':v['reward']-f['fixed_reward'],'NC':v['NC'],'DAC':v['DAC'],'raw_EP':v['EP'],'gated_EP':v['gated_EP'],'source':str(path)})
      else:
        for e in r['events']:
          if 'choices' not in e:continue
          rew=e['feedback']['rewards'];rv=np.array([x['reward'] for x in rew]);safev=np.array([x['NC']==1 and x['DAC']==1 for x in rew])
          legacy=np.array([p['legacy']['score'] for p in e['feedback']['scorer_parity']])
          row={'scene':r['scene']['scene_id'],'decision':e['decision'],'all_zero':bool((rv==0).all()),'safe_candidates':int(safev.sum()),'source':str(path),
               'legacy_vs_actual_dynamics_argmax_changed':bool(legacy.argmax()!=rv.argmax()),
               'legacy_oracle_loss_under_actual_dynamics':float(rv.max()-rv[legacy.argmax()])}
          for pol in ['parent','online64']:
            probs=softmax(e['logits'][pol]);sel=e['choices'][pol]
            row.update({pol+'_fixed_reward':float(rv[sel]),pol+'_oracle_gap':float(rv.max()-rv[sel]),pol+'_unsafe_with_safe_available':bool(safev.any() and not safev[sel]),pol+'_entropy':float(-sum(probs*np.log(np.clip(probs,1e-300,1))))})
          roots.append(row)
          for p in e['feedback']['scorer_parity']:parity.append(p)
    assert len(roots)==48 and len(cont)==144
    grouped=defaultdict(dict)
    for c in cont:grouped[(c['scene'],c['decision'])][c['choice']]=c
    comparisons=[]
    for key,cs in grouped.items():
      a,b=cs['oracle'],cs['parent'];gain=a['fixed']-b['fixed'];real=a['replanned']-b['replanned']
      comparisons.append({'scene':key[0],'decision':key[1],'oracle_minus_parent_fixed':gain,'oracle_minus_parent_replanned':real,'strict_fixed_advantage':gain>1e-8,'strict_reversal':gain>1e-8 and real< -1e-8})
    out={'scope':'old parent/online64; headless roots; common frozen-parent continuation; not A/B p6',
      'roots':len(roots),'continuations':len(cont),'no_safe_roots':sum(r['safe_candidates']==0 for r in roots),'all_zero_roots':sum(r['all_zero'] for r in roots),
      'aligned_parity_max':max(p['max_abs_error'] for p in parity),'parity_candidates':len(parity),
      'legacy_comfort_differences':sum(abs(p['training']['comfort']-p['legacy']['comfort'])>1e-8 for p in parity),
      'legacy_vs_actual_dynamics_argmax_changed_roots':sum(r['legacy_vs_actual_dynamics_argmax_changed'] for r in roots),
      'mean_training_minus_legacy_reward':mean([p['training']['reward']-p['legacy']['score'] for p in parity]),
      'oracle_strict_fixed_advantage_roots':sum(x['strict_fixed_advantage'] for x in comparisons),'oracle_strict_reversal_roots':sum(x['strict_reversal'] for x in comparisons),
      'oracle_minus_parent_fixed_mean':mean([r['oracle_minus_parent_fixed'] for r in comparisons]),'oracle_minus_parent_replanned_mean':mean([r['oracle_minus_parent_replanned'] for r in comparisons]),
      'by_choice':{pol:{'fixed_mean':mean([c['fixed'] for c in cont if c['choice']==pol]),'replanned_mean':mean([c['replanned'] for c in cont if c['choice']==pol]),'delta_mean':mean([c['delta'] for c in cont if c['choice']==pol])} for pol in ['parent','online64','oracle']}}
    table('roots48.csv',roots);table('continuations144.csv',cont);table('continuation_rank_reversals.csv',comparisons);dump('root_diagnostics.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
