"""29-column results with explicit missing cells; never certify an incomplete table."""
import csv
import statistics
from .common import WORKSPACE, artifact, read, verify, write
from .evaluation import check_open
METRICS=('no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','score')
BLOCKS=('open_navtest','open_rare','full_NR','full_R')
SHORT=('NC','DAC','EP','TTC','Comfort','PDM')

def mean_metrics(values):
    if not values:raise ValueError('Empty metrics')
    return {k:statistics.mean(r[k] for r in values.values()) for k in (*METRICS,'success')}

def report(run):
    inputs=read(run/'inputs.json');contract=read(run/'run_contract.json')
    source=__import__('pathlib').Path(contract['source_evaluation']['path']).parent
    missing=[];evidence=[];rows=[];groups={};aux={};scene_metrics={}
    for policy,checkpoint in inputs['models'].items():
        values={};ade=None;aux[policy]={}
        for block in (*BLOCKS,'common_NR','common_R'):
            if block.startswith('open'):
                p=run/'openloop'/policy/block/'audit.json'
                if not p.exists():missing.append(policy+':'+block);continue
                a=check_open(run,policy,block)
                if block=='open_navtest':ade=a['ade_fde']
            else:
                folder=run/'collections'/f'{block}_{policy}'
                if not (folder/'accepted.json').exists() and not contract['native_exports']:
                    folder=source/'collections'/f'{block}_{policy}'
                p=folder/'audit.json'
                if not (folder/'accepted.json').exists():missing.append(policy+':'+block);continue
                accepted=read(folder/'accepted.json');verify(accepted['audit']);a=read(p);co=read(folder/'collection.json')
                if accepted['status']!='PASS' or co['checkpoint']!=checkpoint or co['noise']!='formal_navtest_seed0':
                    raise RuntimeError('Closed-loop identity differs')
                if set(a['metrics'])!={r['scene_id'] for r in inputs['cohorts'][block.rsplit('_',1)[0]]}:
                    raise RuntimeError('Closed-loop membership differs')
            scene_metrics.setdefault(policy,{})[block]=a['metrics']
            values[block]=mean_metrics(a['metrics']);evidence.append(artifact(p))
            if block.startswith('common') or block=='full_NR':aux[policy][block]=values[block]
        if ade is None or any(k not in values for k in BLOCKS):continue
        number=[ade['ade_4s'],ade['fde_4s']]+[values[b][m] for b in BLOCKS for m in METRICS]+[values['full_R']['success']]
        if len(number)!=27:raise RuntimeError('Wrong numeric table width')
        config=contract['model_configs'][policy]
        note='eval=formal_navtest_seed0; historical benchmarks; ADE/FDE=4s,m; independent_blind_test=false'
        if policy=='scalar_v3':note='train=V3 rarelog; fixed seed0 reference; '+note
        elif policy.startswith(('D1','D2','D3','D4')):note='init=V3 rarelog seed0; fixed common pool; hard='+('real' if policy[:2] in ('D1','D3') else 'real+synthetic')+'; fixed feedback-state KL; '+note
        else:
            native=read(verify(contract['native_exports'][policy]));training=read(verify(native['provenance']['training']))
            study=read(verify(training['provenance']['manifest']))
            sources={'B':'common+real rare+synthetic rollout dense replay; historical offline feedback', 'C':'common corrected navtrain', 'D':'common corrected navtrain expert trajectories', 'F':{'paired_common':'paired common only','rarelog':'paired common+real rare','rarerollout':'paired common+real rare+synthetic rollout'} .get(training['provenance']['arm'],'unknown')}
            note='train='+sources[study['queue']]+'; queue='+study['queue']+'; arm='+training['provenance']['arm']+'; init='+study['training']['initialization']+'; '+note
        rows.append([policy,note,*number])
        if '_seed' in policy:
            arm,seed=policy.rsplit('_seed',1);groups.setdefault(arm,{})[int(seed)]=number
    for arm,seeds in groups.items():
        if set(seeds)!={0,1,2}:continue
        for label,f in [('mean',statistics.mean),('std',statistics.stdev)]:
            rows.append([arm+'_3seed_'+label,'optimization seeds 0,1,2; sample SD; not independent acquisition repeats',
                         *[f(x) for x in zip(*(seeds[s] for s in (0,1,2)))]] )
    from .storage import reports_directory
    out=reports_directory(run.name)
    header=['Model Name','Note','ADE','FDE']+[b+':'+m for b in BLOCKS for m in SHORT]+['CL-R:Success Rate']
    for percent in (False,True):
        with (out/('formal_29cols'+('_percent' if percent else '')+'.tsv')).open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(header)
            for row in rows:w.writerow(row[:2]+[f'{v*(100 if percent and k>=2 else 1):.9f}' for k,v in enumerate(row[2:])])
    from .statistics import paired_log_interval
    contrasts={}
    for policy,blocks in scene_metrics.items():
        if policy=='scalar_v3':continue
        for block,metrics in blocks.items():
            if block.startswith('open') or block not in scene_metrics.get('scalar_v3',{}):continue
            cohort=block.rsplit('_',1)[0]
            logs={r['scene_id']:r['origin_log'] for r in inputs['cohorts'][cohort]}
            contrasts[policy+':'+block]={m:paired_log_interval(metrics,scene_metrics['scalar_v3'][block],logs,m) for m in (*METRICS,'success')}
    write(out/'paired_log_intervals.json',contrasts,False)
    write(out/'per_scene_metrics.json',scene_metrics,False)
    result=dict(status='COMPLETE' if not missing else 'INCOMPLETE',formal_table_ready=not missing,missing=missing,
                rows=len(rows),columns=29,auxiliary=aux,evidence=evidence)
    write(out/'summary.json',result,False);return result
