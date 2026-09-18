"""Paired all-seed attribution; exploratory intervals, never auto-promote a winner."""
import csv
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_attribution_common as a
import selector_attribution_data as data
from selector_attribution_collections import evaluation_conditions
from report_selector_cfpi_deployment import checked_collection, differences
from report_selector_decision import average, seed_summary
from selector_rare_common import cluster_interval
from selector_decision_data import scores


def contrast(means, coefficients, metric):
    if not coefficients or not set(coefficients) <= set(means):
        raise RuntimeError('Unregistered or incomplete contrast')
    scenes = set(means[next(iter(coefficients))])
    if any(set(means[k]) != scenes for k in coefficients):
        raise RuntimeError('Contrast must be paired on identical scenes')
    return {s: sum(weight*means[arm][s][metric] for arm,weight in coefficients.items()) for s in scenes}


def feedback_diagnostics(logits, rows):
    result = {}
    for mode in a.MODES:
        subset = [(i,r) for i,r in enumerate(rows) if r['mode'] == mode]
        item = dict(states=len(subset), unknown_argmax=0, sources={})
        for i,r in subset:
            item['unknown_argmax'] += int(int(logits[i].argmax()) not in r['candidate_indices'])
        for source in ('closed','local'):
            count = dict(informative=0, pair_correct=0, queried_winner_argmax=0, known_dominated_argmax=0)
            for i,r in subset:
                pref = a.preference(r, source)
                if pref is None:
                    continue
                winner, loser, _ = pref
                w,l = [r['candidate_indices'][j] for j in (winner,loser)]
                chosen = int(logits[i].argmax())
                count['informative'] += 1
                count['pair_correct'] += int(logits[i,w] > logits[i,l])
                count['queried_winner_argmax'] += int(chosen == w)
                count['known_dominated_argmax'] += int(chosen == l)
            item['sources'][source] = count
        result[mode] = item
    return result


def report(run):
    import torch
    torch.set_num_threads(4)
    inputs = data.checked_inputs(run)
    table, artifacts, missing = {}, [d.artifact(run/'attribution_inputs.json')], []
    for cohort, mode, policy in evaluation_conditions():
        name = f'eval_{cohort}_{mode}_{policy}'
        path = run/'collections'/name/'deployment_collection.json'
        reused = name in inputs['reused_collections']
        if not reused and not path.exists():
            missing.append(name); continue
        entry = inputs['reused_collections'][name] if reused else d.artifact(path)
        checked = checked_collection(entry)
        if checked is None:
            missing.append(name); continue
        condition = checked['collection']
        routing = d.verified_read(condition['routing'])
        if (any(condition.get(k) != v for k,v in dict(cohort=cohort, react_type=mode,
                policy=policy, role='evaluation').items())
                or condition['noise_namespace'] != c.NOISE_NAMESPACE
                or condition['scenario'] != inputs['cohorts'][cohort]['scenario']
                or routing['models'] != {policy: data.model_entry(run, policy)}
                or any('feedback_target' in r for r in routing['routes'])
                or set(checked['metrics']) != {r['scene_id'] for r in inputs['cohorts'][cohort]['rows']}):
            raise RuntimeError('Evaluation identity/model/coverage mismatch')
        if not reused and (condition['research_method'] != a.METHOD
                or condition['run_contract'] != d.artifact(run/'run_contract.json')
                or condition['inputs'] != d.artifact(run/'attribution_inputs.json')):
            raise RuntimeError('New evaluation has foreign provenance')
        table[(cohort,mode,policy)] = checked['metrics']
        artifacts.extend((entry, checked['audit']))
    if missing:
        result = dict(status='INCOMPLETE', decision='REVIEW_REQUIRED', missing=missing,
                      complete_conditions=len(table), expected_conditions=80,
                      automatic_new_method=False, full_rare288_authorized=False)
        c.atomic_json(run/'attribution_progress.json', result)
        return result
    summaries, contrasts, paired = {}, {}, []
    for cohort in ('development','common'):
        for mode in a.MODES:
            key = cohort+'_'+mode
            rows = inputs['cohorts'][cohort]['rows']
            scalar, gate = [table[(cohort,mode,p)] for p in ('scalar_v3','gate_v3')]
            means = {arm: average([table[(cohort,mode,f'{arm}_seed{s}')] for s in a.SEEDS]) for arm in a.ARMS}
            summaries[key] = {}
            strata = dict(failed=[r for r in rows if scalar[r['scene_id']]['success'] == 0],
                          safe_slow=[r for r in rows if scalar[r['scene_id']]['success'] == 1
                                     and scalar[r['scene_id']]['ego_progress'] < .5],
                          safe_other=[r for r in rows if scalar[r['scene_id']]['success'] == 1
                                      and scalar[r['scene_id']]['ego_progress'] >= .5])
            for arm in a.ARMS:
                values = [table[(cohort,mode,f'{arm}_seed{s}')] for s in a.SEEDS]
                summaries[key][arm] = dict(
                    vs_scalar=seed_summary(values,scalar,rows), vs_gate=seed_summary(values,gate,rows),
                    seed_mean_std={metric: float(np.std([np.mean([v[s][metric] for s in v])
                                                       for v in values], ddof=1)) for metric in d.METRICS},
                    seeds=[dict(seed=s, vs_scalar=differences(v,scalar,rows),
                                vs_gate=differences(v,gate,rows)) for s,v in zip(a.SEEDS,values)],
                    baseline_defined_strata={k: dict(scenes=len(rs), vs_scalar=seed_summary(values,scalar,rs))
                                             if rs else dict(scenes=0,vs_scalar=None) for k,rs in strata.items()})
                for seed, value in zip(a.SEEDS,values):
                    for row in rows:
                        scene = row['scene_id']
                        paired.append(dict(cohort=cohort,mode=mode,arm=arm,seed=seed,scene_id=scene,
                                           origin_log=row['origin_log'], **value[scene],
                                           **{'scalar_'+k:scalar[scene][k] for k in d.METRICS}))
            contrasts[key] = {}
            for name, coefficients in a.COMPARISONS.items():
                metrics = {}
                for metric in d.METRICS:
                    interval = cluster_interval(contrast(means,coefficients,metric),rows,samples=10000)
                    interval['exposure'] = a.EXPOSURE
                    metrics[metric] = dict(interval, verdict=a.interval_verdict(interval['scene_weighted_cluster_95']))
                contrasts[key][name] = dict(coefficients=coefficients, metrics=metrics)
    feedback_rows, cache = data.load_rows(run)
    from train_selector_attribution import initial_gradient_audit
    initial_gradient_audit(run)
    initial_gradients = d.artifact(run/'labels/initial_gradient_audit.json')
    feedback = {}
    for arm in a.ARMS:
        feedback[arm] = []
        for seed in a.SEEDS:
            entry = data.model_entry(run,f'{arm}_seed{seed}')
            model, _ = m.load_selector(d.verify(entry))
            z = scores(model,feedback_rows,'cpu')
            feedback[arm].append(dict(seed=seed, model=entry, diagnostics=feedback_diagnostics(z,feedback_rows)))
    path = run/'attribution_paired_scenes.csv'
    with path.with_suffix('.tmp').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)
    path.with_suffix('.tmp').replace(path)
    result = dict(status='PASS', method=a.METHOD, decision='REVIEW_REQUIRED',
                  code_sha=d.read(run/'run_contract.json')['code_sha'], summaries=summaries,
                  contrasts=contrasts, feedback=feedback, label_summary=a.label_summary(feedback_rows),
                  labels=cache, initial_gradient_audit=initial_gradients,
                  paired_scenes=d.artifact(path), artifacts=artifacts,
                  expected_conditions=80, historical_conditions=32, new_conditions=48,
                  exposure=a.EXPOSURE, local_definition=a.LOCAL_DEFINITION,
                  intervals='Seed-average first, scene-weighted log-cluster bootstrap 10000; equal-log sensitivity. Exploratory, no multiplicity correction.',
                  safety_guarantee=False, noninferiority_proven=False,
                  full_pipeline_feedback_necessity_proven=False, pure_horizon_attribution=False,
                  automatic_new_method=False, full_rare288_authorized=False,
                  follow_up='Review all seeds, NC/DAC, rescue/broken and hard strata; no automatic method or full benchmark expansion.')
    c.locked_json(run/'attribution_report.json',result)
    return result
