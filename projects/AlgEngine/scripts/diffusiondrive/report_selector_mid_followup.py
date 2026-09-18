"""Scope-complete MID tables and lambda sensitivity, never automatic selection."""
import csv
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_snapshot_common as s
import selector_mid_followup_common as b
import selector_mid_followup_data as data
import selector_mid_followup_collections as dev
import selector_mid_followup_native as native
from report_selector_balance import summary,compare


def write_csv(path, rows):
    columns=sorted({k for r in rows for k in r})
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=columns);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def plot_sensitivity(run, table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=('CL_NLL','LOW04','MID','HIGH06','P1')
    xs=[b.COEFFICIENTS[a] for a in arms]
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True)
    curve=[]
    for col,(block,groups) in enumerate(sorted(table.items())):
        for row,metric in enumerate(('score','success')):
            ax=axes[row,col]
            for seed in b.SEEDS:
                ys=[groups[a]['seeds'][seed]['mean'][metric] for a in arms]
                ax.plot(xs,ys,'o--',alpha=.45,label=f'seed{seed}')
                curve.extend(dict(block=block,metric=metric,seed=seed,coefficient=x,value=y) for x,y in zip(xs,ys))
            means=[groups[a]['mean'][metric] for a in arms]
            std=[groups[a]['seed_std'][metric] for a in arms]
            ax.errorbar(xs,means,yerr=std,color='black',marker='o',capsize=3,label='mean ± seed std')
            ax.set_title(block+' / '+metric);ax.set_xlabel('lambda');ax.set_xticks(xs);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Fixed lambda=0.5 primary; exploratory sensitivity; error bars are seed std, not confidence intervals')
    fig.tight_layout();fig.savefig(run/'sensitivity.png',dpi=150);plt.close(fig)
    write_csv(run/'sensitivity_curves.csv',curve)
    return [d.artifact(run/'sensitivity.png'),d.artifact(run/'sensitivity_curves.csv')]


def report(run):
    run=Path(run);task=b.task_from_run(run);spec=b.task_spec(task)
    inputs=data.checked_inputs(run);ninputs=s.checked_inputs(run)
    missing=[];artifacts=[];train={};tables={};statistics={};comparisons={};paired=[];extra=[]
    if task=='sensitivity':
        for seed in b.SEEDS:
            for arm in b.NEW_ARMS:
                policy=f'{arm}_seed{seed}';p=run/'train'/policy/'step_500_report.json'
                if not p.exists():missing.append('train/'+policy)
                else:data.model_entry(run,policy);train[policy]=d.artifact(p)
        for cohort,mode,policy in b.dev_conditions():
            x=dev.checked(run,cohort,mode,policy)
            if x is None:missing.append(f'dev/{cohort}/{mode}/{policy}')
            else:tables[(cohort,mode,policy)]=x['metrics'];artifacts.append(x['audit'])
        complete=len(tables)==spec['expected_dev_conditions']
        if complete:
            for cohort in ('development','common'):
                rows=inputs['cohorts'][cohort]['rows']
                for mode in b.MODES:
                    key=cohort+'_'+mode
                    scalar=dev.checked(run,cohort,mode,'scalar_v3')['metrics']
                    values={arm:[tables[(cohort,mode,f'{arm}_seed{seed}')] for seed in b.SEEDS] for arm in b.ARMS}
                    statistics[key]={arm:summary(v) for arm,v in values.items()}
                    comparisons[key]=analyze_block(values,scalar,rows)
                    paired.extend(paired_rows(values,scalar,rows,'development',cohort,mode))
            extra+=plot_sensitivity(run,statistics)
    else:
        for policy,block in b.formal_conditions():
            x=native.checked_formal(run,policy,block)
            if x is None:missing.append(f'formal/{policy}/{block}')
            else:tables[(policy,block)]=x;artifacts.append(x['evidence'])
        if not (run/'bridge_report.json').exists():missing.append('native_bridge')
        else:native.require_bridge(run)
        if len(tables)==spec['expected_formal_conditions']:
            for block in b.BLOCKS:
                scalar=tables[('scalar_v3',block)]['metrics']
                values={'MID':[tables[(f'MID_seed{seed}',block)]['metrics'] for seed in b.SEEDS]}
                statistics[block]=dict(MID=summary(values['MID']),scalar_v3=dict(mean=s.mean(scalar),fixed_reference=True))
                if block=='open_navtest':
                    statistics[block]['MID']['ade_fde_seeds']=[tables[(f'MID_seed{seed}',block)]['ade_fde'] for seed in b.SEEDS]
                if block.startswith('open'):continue
                cohort,mode=block.rsplit('_',1);rows=ninputs['cohorts'][cohort]
                comparisons[block]=analyze_block(values,scalar,rows)
                if cohort=='full':
                    exposed={r['scene_id'] for r in ninputs['cohorts']['development']}
                    for label,subset in [('development58',[r for r in rows if r['scene_id'] in exposed]),
                                         ('remainder230',[r for r in rows if r['scene_id'] not in exposed])]:
                        comparisons[label+'_'+mode]=analyze_block(values,scalar,subset)
                paired.extend(paired_rows(values,scalar,rows,'formal',cohort,mode))
            write_formal_tsv(run,tables)
            extra.append(d.artifact(run/'summary.tsv'))
    result=dict(status='INCOMPLETE' if missing else 'PASS',decision='REVIEW_REQUIRED',**spec,missing=missing,
        complete_training_jobs=len(train),complete_dev_conditions=len(tables) if task=='sensitivity' else 0,
        complete_formal_conditions=len(tables) if task=='mid-formal' else 0,training_reports=train,
        primary_coefficient=.5,automatic_coefficient_selection=False,automatic_expansion=False,
        exposure=b.EXPOSURE,statistics=statistics,comparisons=comparisons,artifacts=artifacts,
        interval_note='Seed average first; 10000 log bootstraps; exploratory unadjusted intervals, not noninferiority proof.',
        reference_comparison='Run independent analysis after P1/CL_NLL formal report is complete; MID task does not wait.')
    if paired:
        write_csv(run/'paired_scenes.csv',paired);extra.append(d.artifact(run/'paired_scenes.csv'))
    result['outputs']=extra
    c.atomic_json(run/'progress.json',result)
    lines=['# MID follow-up / '+task+': '+result['status'],'','Decision: REVIEW_REQUIRED; fixed MID lambda=0.5.',
        '','| Block | Arm | PDM | Success | Seed PDM std |','|---|---|---:|---:|---:|']
    for block,groups in statistics.items():
        for arm,x in groups.items():
            std=x.get('seed_std',{}).get('score',0.)
            lines.append(f"| {block} | {arm} | {x['mean']['score']:.6f} | {x['mean']['success']:.6f} | {std:.6f} |")
    lines+=['','All seeds and negative results retained. No blind-test or automatic promotion claim.',
        'Common64 is not full-common. Remainder230 is historically exposed.','Missing: '+str(len(missing))]
    (run/'summary.md').write_text('\n'.join(lines)+'\n')
    if not missing:
        c.locked_json(run/'summary.json',result)
        c.locked_json(run/'archive/release_manifest.json',dict(status='PASS',method=b.METHOD,task=task,
            contract=d.artifact(run/'run_contract.json'),inputs=d.artifact(run/'followup_inputs.json'),
            summary=d.artifact(run/'summary.json'),outputs=[*extra,d.artifact(run/'summary.md')]))
    return result


def analyze_block(values,scalar,rows):
    result={}
    for arm,v in values.items():
        for reference in ('MID','P1','CL_NLL','scalar_v3'):
            if reference==arm or reference!='scalar_v3' and reference not in values:continue
            result[arm+'__minus__'+reference]=compare(v,[scalar]*3 if reference=='scalar_v3' else values[reference],rows)
    strata=dict(failed=[r for r in rows if scalar[r['scene_id']]['success']==0],
        safe_slow=[r for r in rows if scalar[r['scene_id']]['success']==1 and scalar[r['scene_id']]['ego_progress']<.5],
        safe_other=[r for r in rows if scalar[r['scene_id']]['success']==1 and scalar[r['scene_id']]['ego_progress']>=.5])
    result['baseline_strata']={name:{arm:compare(v,[scalar]*3,rs) for arm,v in values.items()} if rs else None for name,rs in strata.items()}
    return result


def paired_rows(values,scalar,rows,protocol,cohort,mode):
    return [dict(protocol=protocol,cohort=cohort,mode=mode,arm=arm,seed=seed,**r,
            **v[r['scene_id']],**{'scalar_'+k:scalar[r['scene_id']][k] for k in d.METRICS})
            for arm,vs in values.items() for seed,v in zip(b.SEEDS,vs) for r in rows]


def write_formal_tsv(run,tables):
    keys=('no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','score')
    blocks=('open_navtest','open_rare','full_NR','full_R')
    header=['model','note','ADE','FDE',*[block+'_'+k for block in blocks for k in keys],'CL_R_success']
    lines=['\t'.join(header)]
    for policy in b.FORMAL_POLICIES:
        means={block:s.mean(tables[(policy,block)]['metrics']) for block in blocks}
        ade=tables[(policy,'open_navtest')]['ade_fde']
        row=[policy,'fixed MID; three optimization seeds; historically exposed',str(ade['ade_4s']),str(ade['fde_4s']),
             *[str(means[block][k]) for block in blocks for k in keys],str(means['full_R']['success'])]
        lines.append('\t'.join(row))
    (run/'summary.tsv').write_text('\n'.join(lines)+'\n')
