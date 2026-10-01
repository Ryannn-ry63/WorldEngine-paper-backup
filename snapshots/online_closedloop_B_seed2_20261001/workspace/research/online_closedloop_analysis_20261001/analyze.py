#!/usr/bin/env python3
"""Read-only source audit and paired analysis; writes only beside this script."""
import csv, hashlib, json, math, statistics, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

OUT = Path(__file__).resolve().parent
WORK = OUT.parent.parent
REG = WORK/'research/selector_v3_online_diagnosis_20260930/registry'
METRICS = ['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','score']
SHORT = dict(zip(METRICS,['NC','DAC','EP','TTC','C','PDMS']))
def read(p): return json.loads(Path(p).read_text())
HASHES = {}
def sha(p):
    p = str(p)
    if p not in HASHES:
        h=hashlib.sha256()
        with open(p,'rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        HASHES[p]=h.hexdigest()
    return HASHES[p]
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def table(name,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def mean(x): return float(np.mean(x)) if len(x) else None
def softmax(x):
    x=np.asarray(x,dtype=float);e=np.exp(x-x.max());return e/e.sum()
def safe(row): return row[METRICS[0]]==1 and row[METRICS[1]]==1

def main():
    spec=read(REG/'experiment.json');manifest=read(REG/'scene_manifest.json')
    prepared=read(REG/'development/prepared.json')
    for k in ['config','scenarios']:assert sha(prepared[k])==prepared[k+'_sha256']
    origins={r['scene_id']:r['origin_log'] for r in manifest['scenes'] if r['split']=='development'}
    tokens=sorted(origins);logs=sorted(set(origins.values()));n=len(tokens)
    rng=np.random.default_rng(0);draws=rng.integers(len(logs),size=(5000,len(logs)))
    indices=[np.array([i for i,t in enumerate(tokens) if origins[t]==log]) for log in logs]
    def ci(x):
        totals=np.array([sum(x[i] for i in ix) for ix in indices]);counts=np.array([len(ix) for ix in indices])
        return np.quantile(totals[draws].sum(1)/counts[draws].sum(1),[.025,.975]).tolist()
    audit={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'experiment_sha256':sha(REG/'experiment.json'),'jobs':{},'evaluations':{},'missing_development':[]}
    results={};absrows=[];pairedrows=[];transitions=[];bylogs=[];contracts={}
    for job in spec['jobs']:
        jid=job['id'];p=REG/'jobs'/jid/'status.json';s=read(p) if p.exists() else {'status':'NO_STATUS'}
        audit['jobs'][jid]={'status':s['status'],'status_path':str(p),'status_sha256':sha(p) if p.exists() else None}
        if job['kind']!='evaluation' or job.get('dataset')!='development':continue
        if s['status']!='COMPLETE':
            audit['missing_development'].append(jid);continue
        assert sha(s['result'])==s['result_sha256'],jid
        d=read(s['result']);assert d['job']==job and d['status']=='PASS_EXACT_COVERAGE'
        assert set(d['rows'])==set(tokens) and d['counts']==n
        for key,path in d['files'].items(): assert sha(path)==d['files_sha256'][key]
        raw=list(csv.DictReader(open(d['files']['pdm'])));actual=[x for x in raw if x['token']!='overall_average']
        assert len(actual)==n and len({x['token'] for x in actual})==n and len(raw)==n+1
        for row in actual:
            for k in METRICS:
                v=float(row[k]);assert math.isfinite(v) and 0<=v<=1 and v==d['rows'][row['token']][k]
        for k in METRICS: assert abs(mean([r[k] for r in d['rows'].values()])-d['means'][k])<1e-12
        inp=Path(s['attempt'])/'evaluation_inputs.json';assert sha(inp)==d['inputs_sha256']
        inputs=read(inp);assert inputs['job']==job and set(inputs['expected'])==set(tokens) and inputs['artifact']==d['artifact']
        assert inputs['configuration_sha256']==prepared['config_sha256']
        assert inputs['manifest_sha256']==sha(REG/'scene_manifest.json')
        artifact=d['artifact']
        for key in ['path','selector','checkpoint','marker']:
            if key in artifact: assert sha(artifact[key])==artifact['sha256' if key=='path' else key+'_sha256'],(jid,key)
        if 'marker' in artifact:
            marker=read(artifact['marker']);assert marker['final_records'][0]['checkpoint']['path']==artifact['checkpoint']
            assert marker['completed']==(456 if job['policy'].endswith('p3') else 904)
        audit['evaluations'][jid]={'result':s['result'],'result_sha256':sha(s['result']),'inputs':str(inp),'artifact':artifact,'checks':'PASS result/CSV/input/model/marker hashes, exact scenes, raw rows, finite range, means','elapsed_seconds':s['finished_unix']-s['started_unix']}
        results[jid]=d
        absrows.append({'job':jid,'status':'COMPLETE',**{SHORT[k]:d['means'][k] for k in METRICS},'joint_NC_DAC':mean([safe(r) for r in d['rows'].values()])})
    for jid in audit['missing_development']:absrows.append({'job':jid,'status':'MISSING'})
    dump('audit.json',audit);table('absolute_metrics.csv',absrows)
    def array(jid,k):return np.array([results[jid]['rows'][t][k] for t in tokens])
    comparisons=[]
    def compare(label,base,cand,complete=True):
        # base/cand are seed x scene x metric; average seeds within scene before log resampling.
        delta=cand-base;dv=delta.mean(0);out={'comparison':label,'seeds':len(delta),'complete_three_seed':complete}
        for z,k in enumerate(METRICS):
            x=dv[:,z];lo,hi=ci(x);out.update({SHORT[k]+'_delta':float(x.mean()),SHORT[k]+'_seed_sd':float(delta[:,:,z].mean(1).std(ddof=1)) if len(delta)>1 else None,SHORT[k]+'_ci_low':lo,SHORT[k]+'_ci_high':hi})
        comparisons.append(out)
        for log,ix in zip(logs,indices):bylogs.append({'comparison':label,'origin_log':log,'scenes':len(ix),**{SHORT[k]+'_delta':float(dv[ix,z].mean()) for z,k in enumerate(METRICS)},'PDMS_contribution':float(dv[ix,-1].sum()/n)})
        return out
    def matrix(jid):return np.array([[results[jid]['rows'][t][k] for k in METRICS] for t in tokens])
    for jid,d in results.items():
        mode=jid.rsplit('_',1)[1];parent=results['dev_parent_'+mode]
        for t in tokens:
            b=parent['rows'][t];v=d['rows'][t]
            pairedrows.append({'job':jid,'scene':t,'origin_log':origins[t],**{SHORT[k]+'_delta':v[k]-b[k] for k in METRICS},'parent_joint':safe(b),'candidate_joint':safe(v)})
        row={'job':jid}
        for k in [METRICS[0],METRICS[1],METRICS[3]]:
            b=array('dev_parent_'+mode,k);v=array(jid,k)
            row.update({SHORT[k]+'_pass_to_fail':int(((b==1)&(v<1)).sum()),SHORT[k]+'_fail_to_pass':int(((b<1)&(v==1)).sum())})
        b=np.array([safe(parent['rows'][t]) for t in tokens]);v=np.array([safe(d['rows'][t]) for t in tokens])
        for name,mask in [('both_safe',b&v),('lost_safe',b&~v),('gained_safe',~b&v),('both_unsafe',~b&~v),('parent_safe_fixed',b)]:
            row[name+'_n']=int(mask.sum())
            for k in ['score','ego_progress']:
                ds=array(jid,k)-array('dev_parent_'+mode,k)
                row[name+'_'+SHORT[k]+'_conditional_delta']=mean(ds[mask])
                row[name+'_'+SHORT[k]+'_population_contribution']=float(ds[mask].sum()/n)
        transitions.append(row)
    for p in [3,6]:
        for method in ['A','B']:
            for mode in ['R','NR']:
                ids=[f'dev_train_{method}_s{s}_p{p}_{mode}' for s in range(3)]
                for jid in ids:
                    if jid in results:compare(jid,np.array([matrix('dev_parent_'+mode)]),np.array([matrix(jid)]),False)
                if all(j in results for j in ids):compare(f'{method}_p{p}_{mode}',np.array([matrix('dev_parent_'+mode)]*3),np.array([matrix(j) for j in ids]))
        for mode in ['R','NR']:
            a=[f'dev_train_A_s{s}_p{p}_{mode}' for s in range(3)];b=[f'dev_train_B_s{s}_p{p}_{mode}' for s in range(3)]
            if all(j in results for j in a+b): compare(f'B_minus_A_p{p}_{mode}',np.array([matrix(j) for j in a]),np.array([matrix(j) for j in b]))
    for method in ['A','B']:
        for mode in ['R','NR']:
            a=[f'dev_train_{method}_s{s}_p3_{mode}' for s in range(3)];b=[f'dev_train_{method}_s{s}_p6_{mode}' for s in range(3)]
            if all(j in results for j in a+b): compare(f'{method}_p6_minus_p3_{mode}',np.array([matrix(j) for j in a]),np.array([matrix(j) for j in b]))
    for mode in ['R','NR']:
        compare('online64_'+mode,np.array([matrix('dev_parent_'+mode)]),np.array([matrix('dev_online64_'+mode)]),False)
        for pol in ['parent','online64']:compare('interval1_minus_half_'+pol+'_'+mode,np.array([matrix('dev_'+pol+'_'+mode)]),np.array([matrix('interval1_'+pol+'_'+mode)]),False)
        compare('interval_interaction_'+mode,np.array([matrix('interval1_parent_'+mode)-matrix('dev_parent_'+mode)]),np.array([matrix('interval1_online64_'+mode)-matrix('dev_online64_'+mode)]),False)
    repeat=[]
    for p in [3,6]:
        for method in ['A','B']:
            for mode in ['R','NR']:
                ids=[f'dev_train_{method}_s{s}_p{p}_{mode}' for s in range(3)];avail=[j for j in ids if j in results]
                for t in tokens:
                    b=results['dev_parent_'+mode]['rows'][t]
                    row={'method':method,'passes':p,'mode':mode,'scene':t,'origin_log':origins[t],'available_seeds':len(avail)}
                    for k in [METRICS[0],METRICS[1],METRICS[3]]:
                        row[SHORT[k]+'_loss_seeds']=sum(b[k]==1 and results[j]['rows'][t][k]<1 for j in avail)
                        row[SHORT[k]+'_gain_seeds']=sum(b[k]<1 and results[j]['rows'][t][k]==1 for j in avail)
                    row['PDMS_delta_mean_available']=mean([results[j]['rows'][t]['score']-b['score'] for j in avail]);repeat.append(row)
    table('paired_scenes.csv',pairedrows);table('paired_statistics.csv',comparisons);table('origin_log_effects.csv',bylogs);table('safety_transitions.csv',transitions);table('repeated_failures.csv',repeat)
    dump('source_hashes.json',HASHES)
    print(json.dumps({'accepted_evals':len(results),'missing':audit['missing_development'],'complete_method_rows':[r for r in comparisons if r['complete_three_seed'] and r['comparison'].startswith(('A_p','B_p'))]},indent=2),flush=True)

if __name__=='__main__':main()
