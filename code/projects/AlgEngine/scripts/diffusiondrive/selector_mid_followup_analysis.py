"""Read-only MID evidence, fixed comparisons and optional completed formal join."""
from pathlib import Path
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_balance_data as old_data
import selector_balance_collections as old_dev
import selector_mid_followup_common as b
import selector_mid_followup_data as data
import selector_mid_followup_collections as dev
from selector_dual_domain_analysis import Evidence,require,select_cases,summary
from report_selector_mid_followup import analyze_block


def completed(ev, run, task):
    report=ev.read(run/'summary.json')
    contract=ev.read(run/'run_contract.json')
    require(report['status']=='PASS' and report['task']==task and not report['missing'], 'Complete task report required')
    require(contract['task']==task and ev.read(run/'ledger.json')['active'] is None,'Task still active or scope mismatch')
    release=ev.read(run/'archive/release_manifest.json')
    require(ev.use(release['summary'])==run/'summary.json','Foreign release summary')
    if 'contract' in release:require(ev.use(release['contract'])==run/'run_contract.json','Foreign release contract')
    for entry in release['outputs']:ev.use(entry)
    return report,contract


class Study:
    def __init__(self, source, sensitivity=None, progress=lambda *a,**k:None):
        self.run=Path(source).resolve();self.ev=ev=Evidence()
        original,contract=completed(ev,self.run,'development')
        require(contract['method']=='selector_p1_balance_v1' and original['complete_dev_conditions']==48,'Wrong MID source')
        self.inputs=ev.read(self.run/'balance_inputs.json')
        old_data.checked_inputs(self.run)
        self.arms=('CL_NLL','MID','P1','MID_SCALE_MATCH')
        self.coefficients=b.COEFFICIENTS
        self.scales={}
        self.seed_scales={s:old_data.scale(self.run,s) for s in b.SEEDS}
        ev.use(self.run/'scale_audit.json')
        self.models={f'{arm}_seed{s}':old_data.model_entry(self.run,f'{arm}_seed{s}') for arm in self.arms for s in b.SEEDS}
        owner={arm:(self.run,old_dev,self.inputs) for arm in (*self.arms,'scalar_v3')}
        if sensitivity:
            sensitivity=Path(sensitivity).resolve()
            report,co=completed(ev,sensitivity,'sensitivity')
            require(co['method']==b.METHOD and report['complete_dev_conditions']==72,'Incomplete sensitivity')
            extra=data.checked_inputs(sensitivity);ev.use(sensitivity/'followup_inputs.json')
            require(Path(extra['source_run']).resolve()==self.run and extra['parents']==self.inputs['parents']
                and extra['artifacts']['training_cache']==self.inputs['artifacts']['training_cache'], 'Sensitivity lineage mismatch')
            self.arms=b.ARMS
            for arm in b.NEW_ARMS:
                owner[arm]=(sensitivity,dev,extra)
                for seed in b.SEEDS:self.models[f'{arm}_seed{seed}']=data.model_entry(sensitivity,f'{arm}_seed{seed}')
        for policy,entry in self.models.items():
            ev.use(entry,model=True)
            require(entry['provenance']['policy']==policy and entry['score_mode']=='residual','Wrong selector identity')
        for entry in self.inputs['parents'].values():ev.use(entry,model=True)
        self.rows=c.load_pickle(ev.use(self.inputs['artifacts']['training_cache']))['rows']
        from selector_attribution_data import validate_rows
        validate_rows(self.rows,self.inputs['excluded_logs'])
        self.dense=ev.read(self.run/'dense/manifest.json')
        self.collections={};self.table={}
        for cohort in ('development','common'):
            ev.use(self.inputs['cohorts'][cohort]['scenario'])
            for mode in b.MODES:
                for arm in (*self.arms,'scalar_v3'):
                    root,api,inputs=owner[arm]
                    for policy in (['scalar_v3'] if arm=='scalar_v3' else [f'{arm}_seed{s}' for s in b.SEEDS]):
                        progress('verify_collection',cohort=cohort,mode=mode,policy=policy)
                        result=api.checked(root,cohort,mode,policy)
                        require(result is not None,'Missing audited collection')
                        audit=ev.read(result['audit']);path=ev.use(audit['collection'])
                        collection=ev.read(path);ev.use(collection['routing'])
                        merged=d.metrics_csv(ev.use(audit['merged_metrics']),{r['scene_id'] for r in self.inputs['cohorts'][cohort]['rows']})
                        require(merged==result['metrics'],'Changed collection metrics')
                        self.collections[(cohort,mode,policy)]=dict(path=path,collection=collection,audit=audit)
                        self.table[(cohort,mode,policy)]=merged

    def values(self, cohort, mode, arm):
        return [self.table[(cohort,mode,f'{arm}_seed{s}')] for s in b.SEEDS]

    def statistics(self):
        results={};cases=[];overlap={}
        dense_logs={r['origin_log'] for r in self.dense['records']}
        feedback_logs={r['origin_log'] for r in self.rows}
        for cohort in ('development','common'):
            rows=self.inputs['cohorts'][cohort]['rows'];logs={r['origin_log'] for r in rows}
            require(not logs & (dense_logs|feedback_logs),'Current train/eval log overlap')
            overlap[cohort]=dict(scenes=len(rows),logs=len(logs),dense_overlap=[],feedback_overlap=[],historically_exposed=True)
            for mode in b.MODES:
                scalar=self.table[(cohort,mode,'scalar_v3')]
                values={a:self.values(cohort,mode,a) for a in self.arms}
                results[cohort+'_'+mode]=dict(arms={a:summary(v,scalar,rows) for a,v in values.items()},
                    comparisons=analyze_block(values,scalar,rows))
                for other in self.arms:
                    if other=='MID':continue
                    for case in select_cases(values['MID'],values[other],rows):
                        case['reasons']=[r.replace('CL_NLL_better',other+'_better').replace('P1_better','MID_better') for r in case['reasons']]
                        cases.append(dict(case,cohort=cohort,mode=mode,left_arm='MID',right_arm=other))
        return dict(status='PASS',summaries=results,overlap=overlap,cases=cases,
            decision='REVIEW_REQUIRED',primary_coefficient=.5,automatic_selection=False,
            inference='Exploratory, seed average before 10000 log bootstraps; no independent test or noninferiority claim.')


def formal_comparison(ev, mid, reference):
    """Wait by explicit invocation: never read a partially completed reference table."""
    import selector_mid_followup_native as native
    import selector_balance_native as old_native
    import selector_snapshot_common as snapshot
    mid,reference=Path(mid).resolve(),Path(reference).resolve()
    mr,mc=completed(ev,mid,'mid-formal');rr,rc=completed(ev,reference,'formal')
    require(mc['method']==b.METHOD and rc['method']=='selector_p1_balance_v1','Wrong formal methods')
    require(mr['complete_formal_conditions']==24 and rr['complete_formal_conditions']==42,'Incomplete full comparison')
    mi,ri=snapshot.checked_inputs(mid),snapshot.checked_inputs(reference)
    require(mi['cohorts']==ri['cohorts'] and mi['open_tokens']==ri['open_tokens']
            and mc['formal_noise']==rc['formal_noise']==b.FORMAL_NOISE,'Unmatched formal protocol')
    ev.use(mid/'inputs.json');ev.use(reference/'inputs.json')
    for inputs in (mi,ri):
        for entry in inputs['models'].values():ev.use(entry,model=True)
    out={}
    for block in b.BLOCKS:
        values={}
        for arm,root,api in [('MID',mid,native),('P1',reference,old_native),('CL_NLL',reference,old_native)]:
            items=[api.checked_formal(root,f'{arm}_seed{s}',block) for s in b.SEEDS]
            require(all(x is not None for x in items),'Missing full condition')
            for x in items:ev.use(x['evidence'])
            values[arm]=[x['metrics'] for x in items]
        scalar=native.checked_formal(mid,'scalar_v3',block)
        other=old_native.checked_formal(reference,'scalar_v3',block)
        require(scalar is not None and other is not None and scalar['evidence']==other['evidence'],'Different scalar baseline')
        ev.use(scalar['evidence']);scalar=scalar['metrics']
        if block.startswith('open'):
            # Open-loop tokens have no log mapping in this protocol: report seed std, no invented clusters.
            from report_selector_balance import summary as seed_summary
            out[block]=dict(arms={a:seed_summary(v) for a,v in values.items()},interval_status='NO_LOG_MAPPING')
        else:
            cohort,mode=block.rsplit('_',1);rows=mi['cohorts'][cohort]
            out[block]=dict(arms={a:summary(v,scalar,rows) for a,v in values.items()},comparisons=analyze_block(values,scalar,rows))
            if cohort=='full':
                exposed={r['scene_id'] for r in mi['cohorts']['development']}
                for label,flag in [('development58',True),('remainder230',False)]:
                    subset=[r for r in rows if (r['scene_id'] in exposed)==flag]
                    out[label+'_'+mode]=dict(arms={a:summary(v,scalar,subset) for a,v in values.items()},comparisons=analyze_block(values,scalar,subset))
    return dict(status='PASS',decision='REVIEW_REQUIRED',blocks=out,historically_exposed=True)
