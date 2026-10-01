#!/usr/bin/env python3
"""Single-writer status, clustered paired development evidence, fixed selection rule."""
import argparse
import json
from pathlib import Path
import statistics
import time
from experiment_common import REGISTRY,read,write,sha,lock
from evaluation_artifacts import METRICS,SAFETY


def accepted(job_id):
    path=REGISTRY/'jobs'/job_id/'status.json'
    if not path.exists():return None
    status=read(path)
    if status.get('status')!='COMPLETE' or not status.get('result'):return None
    if sha(status['result'])!=status['result_sha256']:raise ValueError('Accepted result changed')
    result=read(status['result'])
    for key,path in result['files'].items():
        if sha(path)!=result['files_sha256'][key]:raise ValueError('Accepted CSV changed')
    if result['status']!='PASS_EXACT_COVERAGE':raise ValueError('Incomplete result')
    return result


def paired(parent,online,origins):
    if not parent or set(parent)!=set(online):raise ValueError('Unpaired evaluation rows')
    deltas={t:{k:online[t][k]-parent[t][k] for k in METRICS} for t in sorted(parent)}
    bylog={}
    for t,d in deltas.items():bylog.setdefault(origins[t],[]).append(d['score'])
    result=dict(delta={k:statistics.mean(d[k] for d in deltas.values()) for k in METRICS},
        per_scene=deltas,log_count=len(bylog),scene_count=len(deltas))
    import numpy as np
    values=list(bylog.values());totals=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
    rng=np.random.default_rng(0);indices=rng.integers(len(values),size=(5000,len(values)))
    ci=totals[indices].sum(1)/counts[indices].sum(1)
    result['paired_score_log_cluster_ci95']=np.quantile(ci,[.025,.975]).tolist()
    return result


def aggregate_methods(results,passes,origins):
    output={}
    for method in ('A','B'):
        seeds={}
        for seed in (0,1,2):
            modes={}
            for reaction in ('R','NR'):
                parent=results.get('dev_parent_'+reaction)
                candidate=results.get('dev_train_%s_s%d_p%d_%s'%(method,seed,passes,reaction))
                if parent and candidate:
                    modes[reaction]=paired(parent['rows'],candidate['rows'],origins)
            if len(modes)==2:seeds[str(seed)]=modes
        if len(seeds)==3:
            means={m:{k:statistics.mean(s[m]['delta'][k] for s in seeds.values()) for k in METRICS} for m in ('R','NR')}
            eligible=means['R']['score']>=.005 and means['NR']['score']>=-.005 and all(
                means[m][k]>=-.005 for m in ('R','NR') for k in SAFETY)
            output[method]=dict(seeds=seeds,mean_delta=means,eligible=eligible,
                score_delta_std={m:statistics.stdev(s[m]['delta']['score'] for s in seeds.values()) for m in ('R','NR')})
    return output


def diagnostic_summary(status):
    if not status or status.get('status')!='COMPLETE':return None
    folder=Path(status['attempt']);roots=[];continuations=[];geometric=[]
    import sys
    sys.path.insert(0,str(REGISTRY.parent/'code/projects/SimEngine'))
    from worldengine.online.plan_diagnostics import common_future_difference
    for path in sorted(folder.glob('rank*/probe/*/report.json')):
        report=read(path)
        if report.get('forced'):
            forced=report['forced'];scored=report['closed']['continuation']
            continuations.append(dict(scene=report['scene']['scene_id'],root_decision=forced['root_decision'],
                choice=forced['choice'],fixed_reward=forced['fixed_reward'],replanned_reward=scored['training']['reward'],
                delta=scored['training']['reward']-forced['fixed_reward'],components=scored['training']))
            continue
        previous=None
        for event in report['events']:
            plan=event['feedback']['executed_plan']['waypoints']
            if previous is not None:geometric.append(common_future_difference(previous,plan))
            previous=plan
            if 'choices' not in event:continue
            rewards=event['feedback']['rewards'];parity=event['feedback']['scorer_parity']
            roots.append(dict(scene=report['scene']['scene_id'],decision=event['decision'],choices=event['choices'],
                choice_rewards=event['choice_rewards'],oracle_gap=event['oracle_gap'],
                all_zero=all(r['reward']==0 for r in rewards),
                safe_candidates=sum(r['NC']==1 and r['DAC']==1 for r in rewards),
                unsafe_at_root=sum(any(r['first_violation_seconds'][k]==0 for k in ('NC','DAC')) for r in rewards),
                parity_max_abs=max(p['max_abs_error'] for p in parity),
                candidate_rewards=[r['reward'] for r in rewards],candidate_logits=event['logits']))
    if len(roots)!=48 or len(continuations)!=144:raise ValueError('Incomplete representative diagnostics')
    return dict(status='PASS_48_ROOT_DIAGNOSTICS',roots=roots,continuations=continuations,
        scope='24 clips / 12 origin logs; headless map-warmup roots; frozen parent continuation',
        mean_oracle_gap={p:statistics.mean(r['oracle_gap'][p] for r in roots) for p in ('parent','online64')},
        ranking_selection_disagreements=sum(r['choices']['parent']!=r['choices']['online64'] for r in roots),
        all_zero_roots=sum(r['all_zero'] for r in roots),
        common_future_mean_m=statistics.mean(g['overlap_mean_distance_m'] for g in geometric),
        next_step_mean_m=statistics.mean(g['next_step_distance_m'] for g in geometric),
        switching_penalty_added=False)


def summarize():
    with lock(REGISTRY/'summary.lock'):
        spec=read(REGISTRY/'experiment.json');statuses={};results={};jobs=spec['jobs']
        for job in jobs:
            path=REGISTRY/'jobs'/job['id']/'status.json'
            statuses[job['id']]=read(path) if path.exists() else dict(status='NOT_STARTED')
            if job['kind']=='evaluation':
                result=accepted(job['id'])
                if result:results[job['id']]=result
        manifest_path=REGISTRY/'scene_manifest.json'
        origins={r['scene_id']:r['origin_log'] for r in read(manifest_path)['scenes'] if r['split']=='development'} if manifest_path.exists() else {}
        methods={str(p):aggregate_methods(results,p,origins) for p in (3,6)}
        diagnostic=diagnostic_summary(statuses.get('roots48'))
        if diagnostic:write(REGISTRY/'diagnostic_summary.json',diagnostic)
        extra_followups=[]
        for key,status in statuses.items():
            if key.startswith('roots_extra_') and status.get('status')=='COMPLETE':
                for path in sorted(Path(status['attempt']).glob('rank*/probe/*/extra_roots.json')):
                    extra_followups.append(dict(read(path),record=str(path),record_sha256=sha(path)))
        old64={m:paired(results['dev_parent_'+m]['rows'],results['dev_online64_'+m]['rows'],origins)
               for m in ('R','NR') if 'dev_parent_'+m in results and 'dev_online64_'+m in results}
        intervals={}
        for policy in ('parent','online64'):
            for m in ('R','NR'):
                half=results.get('dev_'+policy+'_'+m);one=results.get('interval1_'+policy+'_'+m)
                if half and one:intervals[policy+'_'+m]=paired(half['rows'],one['rows'],origins)
        # Selection sees development results only. Formal files are never read
        # here to revise the method/seed choice.
        selection_path=REGISTRY/'selection.json'
        if not selection_path.exists() and len(methods['6'])==2:
            eligible=[m for m,v in methods['6'].items() if v['eligible']]
            if eligible:
                chosen=sorted(eligible,key=lambda m:(-methods['6'][m]['mean_delta']['R']['score'],m))[0]
                selection=dict(status='QUALIFIED_FINAL_DEV_SELECTION',method=chosen,budget_passes=6,
                    policies=['train_%s_s%d_p6'%(chosen,s) for s in (0,1,2)],rule=spec['selection'],
                    development_results={k:sha(REGISTRY/'jobs'/k/'status.json') for k in results if
                        k.startswith('dev_train_') and '_p6_' in k or k in ('dev_parent_R','dev_parent_NR')},
                    experiment_sha256=sha(REGISTRY/'experiment.json'),formal_test_used_for_selection=False)
            else:selection=dict(status='NO_METHOD_QUALIFIED',rule=spec['selection'],budget_passes=6)
            write(selection_path,selection,immutable=True)
        selection=read(selection_path) if selection_path.exists() else dict(status='PENDING_FULL_P6_THREE_SEED_DEV')
        clock=read(REGISTRY/'campaign_clock.json') if (REGISTRY/'campaign_clock.json').exists() else None
        exhausted=clock is not None and time.time()>=clock['deadline_unix']
        if exhausted and len(methods['6'])<2 and len(methods['3'])==2:
            selection=dict(status='INCOMPLETE_P6_BUDGET_P3_COMPARISON_ONLY',methods=methods['3'],
                formal_evaluation_allowed=False)
        summary=dict(status='IN_PROGRESS' if not exhausted else 'CAMPAIGN_TIME_REACHED',
            experiment_sha256=sha(REGISTRY/'experiment.json'),jobs=statuses,methods=methods,old64=old64,
            interval_diagnostics=intervals,extra_failure_followups=extra_followups,selection=selection,diagnostics_complete=diagnostic is not None,
            clock=clock,updated_unix=time.time(),official_improvement_verified=False)
        formal_ids=[j['id'] for j in jobs if j.get('dataset')=='formal']
        if selection.get('status')=='QUALIFIED_FINAL_DEV_SELECTION' and all(k in results for k in formal_ids):
            summary['formal_tables']={k:dict(means=results[k]['means'],count=results[k]['counts'],
                                              files=results[k]['files']) for k in formal_ids}
            summary['status']='COMPLETE_FULL_FORMAL_COVERAGE'
            comparison={}
            for mode in ('ol_navtest','ol_rare','cl_NR','cl_R'):
                parent=results['formal_parent_'+mode]['rows']
                per_seed={}
                for seed in (0,1,2):
                    online=results['formal_selected_s%d_%s'%(seed,mode)]['rows']
                    if set(parent)!=set(online):raise ValueError('Formal paired coverage differs')
                    per_seed[str(seed)]={k:statistics.mean(online[t][k]-parent[t][k] for t in parent) for k in METRICS}
                comparison[mode]=dict(per_seed_delta=per_seed,
                    mean_delta={k:statistics.mean(v[k] for v in per_seed.values()) for k in METRICS},
                    score_delta_std=statistics.stdev(v['score'] for v in per_seed.values()))
            summary['formal_comparison']=comparison
            summary['observed_closedloop_mean_score_improved_both_modes']=all(
                comparison[m]['mean_delta']['score']>0 for m in ('cl_R','cl_NR'))
            # A measured increase is reported separately from a significance claim.
            summary['official_improvement_verified']=False
        write(REGISTRY/'summary.json',summary)
        lines=['# 场景覆盖与更新时序实验','', '状态：'+summary['status'], '',
            'A：每步更新。B：冻结采集 8 步，按时间顺序重新前向、逐组更新。奖励和每组损失保持 v2。', '',
            '开发集含历史训练来源日志，只用于方法选择；正式测试保持独立。', '',
            '| 作业 | 状态 |','|---|---|']
        lines += ['| '+k+' | '+v['status']+' |' for k,v in statuses.items()]
        for passes,value in methods.items():
            lines+=['', '## '+passes+' 遍覆盖', '', '| 方法 | ΔR PDMS | ΔNR PDMS | 通过门槛 |','|---|---:|---:|---|']
            for method,v in value.items():lines.append('| %s | %.6f | %.6f | %s |'%(method,v['mean_delta']['R']['score'],v['mean_delta']['NR']['score'],v['eligible']))
            if not value:lines.append('| 等待完整三种子结果 | — | — | — |')
        if summary.get('formal_comparison'):
            lines+=['','## 正式四表（三种子）','','| 表 | 平均分数变化 | 种子间标准差 |','|---|---:|---:|']
            lines+=['| %s | %.6f | %.6f |'%(mode,v['mean_delta']['score'],v['score_delta_std'])
                    for mode,v in summary['formal_comparison'].items()]
        lines+=['','选择状态：'+selection['status'],'',
            '训练覆盖、奖励诊断、重规划间隔诊断与正式四表分别记录。未完成的项目不据此宣称效果提升。']
        (REGISTRY/'REPORT.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(status=summary['status'],selection=selection['status'],
            complete=sum(v['status']=='COMPLETE' for v in statuses.values()),jobs=len(jobs))),flush=True)
        return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        try:summary=summarize()
        except BlockingIOError:
            if not a.watch:raise
            summary={}
        if not a.watch or summary.get('status') in ('COMPLETE_FULL_FORMAL_COVERAGE','CAMPAIGN_TIME_REACHED'):break
        time.sleep(30)
if __name__=='__main__':main()
