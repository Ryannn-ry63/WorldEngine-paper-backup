"""Verified experiment inputs and complete paired metrics; no historical writes."""
from pathlib import Path
from collections import defaultdict
import selector_dual_domain_analysis as h
import selector_attribution_common as a
import selector_cfpi_deployment_common as d
import cfpi_common as c
from selector_rare_common import cluster_interval


def sensitivity(delta, rows):
    h.require(set(delta)=={r['scene_id'] for r in rows}, 'Sensitivity coverage mismatch')
    n, total = len(delta), sum(delta.values())
    ranked = sorted(delta, key=lambda s: (-delta[s],s))
    groups = defaultdict(list)
    for r in rows:
        groups[r['origin_log']].append(r['scene_id'])
    return dict(improved=sum(v>.001 for v in delta.values()), worse=sum(v<-.001 for v in delta.values()),
                ranked=[dict(scene_id=s,delta=delta[s]) for s in ranked],
                leave_one_log_out=[dict(origin_log=log,mean=(total-sum(delta[s] for s in ids))/(n-len(ids)))
                                   for log,ids in sorted(groups.items()) if len(ids)<n],
                remove_top_positive=[dict(removed=k,mean=(total-sum(delta[s] for s in ranked[:k]))/(n-k))
                                     for k in (1,3) if k<n],
                note='Post-hoc sensitivity only; all scenes remain in the main result.')


class Study:
    def __init__(self, run, snapshot):
        self.run, self.snapshot = Path(run).resolve(), Path(snapshot).resolve()
        self.ev = h.Evidence()
        ev = self.ev
        self.report = ev.read(self.run/'attribution_report.json')
        self.inputs = ev.read(self.run/'attribution_inputs.json')
        self.contract = ev.read(self.run/'run_contract.json')
        h.require(self.report['status']=='PASS' and self.report['expected_conditions']==80, 'Incomplete attribution')
        h.require(self.contract['method']==a.METHOD and self.report['code_sha']==self.contract['code_sha'], 'Wrong experiment')
        h.require(ev.read(self.run/'decision_ledger.json')['active'] is None, 'Historical experiment still active')
        self.models = {f'{arm}_seed{x["seed"]}':x['model'] for arm in a.ARMS for x in self.report['feedback'][arm]}
        h.require(len(self.models)==18, 'Missing final models')
        for arm in a.ARMS:
            h.require({x['seed'] for x in self.report['feedback'][arm]}==set(a.SEEDS), 'Wrong seeds')
        for name,entry in self.models.items():
            ev.use(entry,model=True)
            h.require(entry['provenance']['policy']==name and entry['score_mode']=='residual', 'Wrong selector')
        for entry in self.inputs['parents'].values():
            ev.use(entry,model=True)
        self.rows = c.load_pickle(ev.use(self.report['labels']))['rows']
        from selector_attribution_data import validate_rows
        validate_rows(self.rows,self.inputs['excluded_logs'])
        self.dense = ev.read(self.run/'dense/manifest.json')
        self.snapshot_summary = ev.read(self.snapshot/'summary.json')
        ev.use(self.snapshot/'run_contract.json')
        ev.use(self.snapshot/'summary.md')
        self.collections, self.table = {}, {}
        for cohort in h.COHORTS:
            rows = self.inputs['cohorts'][cohort]['rows']
            ev.use(self.inputs['cohorts'][cohort]['scenario'])
            expected = {r['scene_id'] for r in rows}
            h.require(len(expected)==len(rows), 'Duplicate cohort scenes')
            for mode in a.MODES:
                for policy in ('scalar_v3','gate_v3',*self.models):
                    name = f'eval_{cohort}_{mode}_{policy}'
                    item = self.inputs['reused_collections'].get(name,self.run/'collections'/name/'deployment_collection.json')
                    path = ev.use(item)
                    collection = ev.read(path)
                    audit = ev.read(path.parent/'collection_audit.json')
                    h.require(ev.use(audit['collection'])==path and audit['status']=='PASS'
                            and audit['incomplete_or_dropped_scenes']==0, 'Wrong audit owner/coverage')
                    h.require(all(collection[k]==v for k,v in dict(cohort=cohort,react_type=mode,policy=policy,role='evaluation').items())
                            and collection['noise_namespace']==c.NOISE_NAMESPACE, 'Wrong cohort/model/noise')
                    h.require(collection['scenario']==self.inputs['cohorts'][cohort]['scenario'], 'Wrong scenario manifest')
                    routing = ev.read(collection['routing'])
                    h.require({r['scene_id'] for r in routing['routes']}==expected
                            and not any('feedback_target' in r for r in routing['routes']), 'Wrong routes')
                    h.require(set(routing['models'])=={policy}, 'Wrong routing model count')
                    model = routing['models'][policy]
                    h.require(model==(self.models.get(policy) or self.inputs['models'][policy]), 'Wrong model identity')
                    ev.use(model,model=True)
                    metrics = d.metrics_csv(ev.use(audit['merged_metrics']),expected)
                    h.require(metrics==audit['metrics'], 'Audited metrics changed')
                    self.collections[(cohort,mode,policy)] = dict(path=path,collection=collection,audit=audit)
                    self.table[(cohort,mode,policy)] = metrics
        h.require(len(self.table)==80, 'Incomplete study')

    def values(self, cohort, mode, arm):
        return [self.table[(cohort,mode,f'{arm}_seed{s}')] for s in a.SEEDS]

    def statistics(self):
        results, cases, overlap = {}, [], {}
        dense_logs = {r['origin_log'] for r in self.dense['records']}
        feedback_logs = {r['origin_log'] for r in self.rows}
        for cohort in h.COHORTS:
            rows = self.inputs['cohorts'][cohort]['rows']
            logs = {r['origin_log'] for r in rows}
            overlap[cohort] = dict(scenes=len(rows),logs=len(logs),dense_logs=sorted(logs&dense_logs),
                feedback_logs=sorted(logs&feedback_logs),
                historical_training_overlap='Not resolved by current manifests; legacy development exposure remains.')
            h.require(not logs&dense_logs and not logs&feedback_logs, 'Current training/evaluation log overlap')
            for mode in a.MODES:
                key = cohort+'_'+mode
                scalar = self.table[(cohort,mode,'scalar_v3')]
                strata = dict(failed=[r for r in rows if scalar[r['scene_id']]['success']==0],
                    safe_slow=[r for r in rows if scalar[r['scene_id']]['success']==1 and scalar[r['scene_id']]['ego_progress']<.5],
                    safe_other=[r for r in rows if scalar[r['scene_id']]['success']==1 and scalar[r['scene_id']]['ego_progress']>=.5])
                arms = {}
                for arm in a.ARMS:
                    values = self.values(cohort,mode,arm)
                    arms[arm] = h.summary(values,scalar,rows)
                    saved = self.report['summaries'][key][arm]['vs_scalar']['mean']
                    h.require(all(abs(saved[k]-arms[arm]['mean'][k])<1e-10 for k in d.METRICS), 'Report reproduction failed')
                    arms[arm]['strata'] = {k:h.summary(values,scalar,rs) if rs else None for k,rs in strata.items()}
                comparisons = {}
                for other in ('CL_NLL','Continue','L_Pair_NLL'):
                    left,right = h.average(self.values(cohort,mode,'P1')),h.average(self.values(cohort,mode,other))
                    deltas = {k:{s:right[s][k]-left[s][k] for s in left} for k in d.METRICS}
                    comparisons[other+'_minus_P1'] = dict(metrics={k:cluster_interval(v,rows) for k,v in deltas.items()},
                                                         sensitivity=sensitivity(deltas['score'],rows))
                results[key] = dict(arms=arms,comparisons=comparisons,
                    references={p:h.summary([self.table[(cohort,mode,p)]],scalar,rows) for p in ('scalar_v3','gate_v3')})
                cases.extend(dict(case,cohort=cohort,mode=mode) for case in
                             h.select_cases(self.values(cohort,mode,'P1'),self.values(cohort,mode,'CL_NLL'),rows))
        return dict(status='PASS',summaries=results,overlap=overlap,cases=cases,
                    full_snapshot_separate=self.snapshot_summary,
                    inference='Exploratory fixed-acquisition/fixed-noise comparisons; no multiplicity correction; not independent testing.')
