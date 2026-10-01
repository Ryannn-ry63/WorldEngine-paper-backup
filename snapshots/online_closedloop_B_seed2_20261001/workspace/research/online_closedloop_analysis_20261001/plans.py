#!/usr/bin/env python3
"""Read archived plans, verify common-time geometry, and retain violation timing."""
from analyze import *

def main():
    audit=read(OUT/'audit.json');rows=[];cases={};steps=[];result={};checks=Counter()
    for jid,item in audit['evaluations'].items():
      d=read(item['result']);result[jid]=d;mode=jid.rsplit('_',1)[1]
      raw={r['token']:r for r in csv.DictReader(open(d['files']['pdm'])) if r['token']!='overall_average'}
      folder=Path(d['output']);archive=Path(item['result']).parent/'evidence_archive.json'
      archived={x['path']:x['sha256'] for x in read(archive)['files']} if archive.exists() else {}
      scenes=defaultdict(list)
      for p in sorted(folder.glob('simulation/split_*/executed_plans/*.json')):
        if str(p) in archived:assert sha(p)==archived[str(p)];checks['archived_plan_hashes']+=1
        else: checks['plans_without_archive_hash']+=1
        plan=read(p);scenes[plan['scene']].append(plan)
      assert set(scenes)==set(d['rows']),(jid,len(scenes))
      for scene,plans in scenes.items():
        plans.sort(key=lambda x:x['step']);assert [x['step'] for x in plans]==list(range(3,11))
        srow=raw[scene];v=srow.get('first_violation_step','');violation=float(v) if v and v.lower()!='nan' else None
        previous=None;vals=[]
        for plan in plans:
          if previous is not None:
            # Both arrays are in the same scene-world frame (common translation).
            old=np.asarray(previous['executed_world_waypoints'])[2:,:2];new=np.asarray(plan['executed_world_waypoints'])[1:,:2];count=min(len(old),len(new));dist=np.linalg.norm(old[:count]-new[:count],axis=1)
            geom={'overlap_mean_distance_m':float(dist.mean()),'next_step_distance_m':float(dist[0]),'max_distance_m':float(dist.max())}
            for k,value in geom.items():assert abs(value-plan['common_future'][k])<1e-9
            row={'job':jid,'scene':scene,'step':plan['step'],'first_NC_DAC_violation_step':violation,'strictly_before_violation':violation is None or plan['step']<violation,'reused':plan['reused'],**geom}
            steps.append(row);vals.append(row)
          previous=plan
        row={'job':jid,'scene':scene,'first_NC_DAC_violation_step':violation,**{SHORT[k]:d['rows'][scene][k] for k in METRICS}}
        for name,vs in [('all',vals),('before_violation',[x for x in vals if x['strictly_before_violation']]),('fixed_steps4_5',[x for x in vals if x['step'] in [4,5]])]:
          row[name+'_count']=len(vs)
          for k in ['overlap_mean_distance_m','next_step_distance_m','max_distance_m']:row[name+'_'+k]=mean([x[k] for x in vs])
        rows.append(row)
        cases[jid+'|'+scene]=plans
    table('plan_steps.csv',steps);table('plan_scenes.csv',rows)
    # Classify by parent->candidate joint safety. These are outcome-conditioned descriptions, not causal estimates.
    groups=[]
    for jid,d in result.items():
      mode=jid.rsplit('_',1)[1];base=result['dev_parent_'+mode]['rows'];rs=[r for r in rows if r['job']==jid]
      for name,cond in [('all',lambda t:True),('lost_safe',lambda t:safe(base[t]) and not safe(d['rows'][t])),('gained_safe',lambda t:not safe(base[t]) and safe(d['rows'][t])),('both_safe',lambda t:safe(base[t]) and safe(d['rows'][t])),('both_unsafe',lambda t:not safe(base[t]) and not safe(d['rows'][t]))]:
        subset=[r for r in rs if cond(r['scene'])];g={'job':jid,'category':name,'scene_count':len(subset)}
        for k in ['all_overlap_mean_distance_m','all_next_step_distance_m','before_violation_overlap_mean_distance_m','before_violation_next_step_distance_m','fixed_steps4_5_overlap_mean_distance_m']:
          g[k]=mean([r[k] for r in subset if r[k] is not None])
        groups.append(g)
    table('plan_categories.csv',groups);dump('plan_checks.json',dict(checks))
    # Deterministic case sampling: strongest repeated DAC loss, best safe score gain, median unchanged joint-safe scene.
    repeat=list(csv.DictReader(open(OUT/'repeated_failures.csv')));selected=[]
    for mode in ['R','NR']:
      rs=[r for r in repeat if r['method']=='A' and r['passes']=='6' and r['mode']==mode]
      bad=sorted([r for r in rs if int(r['DAC_loss_seeds'])>0],key=lambda r:(-int(r['DAC_loss_seeds']),float(r['PDMS_delta_mean_available']),r['scene']))[:3]
      bs=result['dev_parent_'+mode]['rows'];ids=[f'dev_train_A_s{s}_p6_{mode}' for s in range(3)]
      stable=[r for r in rs if safe(bs[r['scene']]) and all(safe(result[j]['rows'][r['scene']]) for j in ids)]
      good=sorted(stable,key=lambda r:(-float(r['PDMS_delta_mean_available']),r['scene']))[:2]
      unchanged=sorted(stable,key=lambda r:(abs(float(r['PDMS_delta_mean_available'])),r['scene']))[:1]
      for typ,rr in [('repeated_DAC_loss',bad),('safe_progress_gain',good),('near_unchanged_safe',unchanged)]:
        for r in rr:
          scene=r['scene'];entry={'mode':mode,'category':typ,'scene':scene,'selection':r,'plans':{j:cases[j+'|'+scene] for j in ['dev_parent_'+mode]+ids},'metrics':{j:result[j]['rows'][scene] for j in ['dev_parent_'+mode]+ids},'timing':{j:next(x for x in rows if x['job']==j and x['scene']==scene) for j in ['dev_parent_'+mode]+ids}}
          selected.append(entry)
    dump('selected_cases.json',selected)
    print(dict(checks), 'scenes',len(rows),'steps',len(steps),flush=True)

if __name__=='__main__':main()
