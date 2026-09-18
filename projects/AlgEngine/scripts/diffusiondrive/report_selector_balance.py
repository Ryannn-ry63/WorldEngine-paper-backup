"""Separate development/formal results, paired seeds/logs, no automatic promotion."""
import csv
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_balance_common as b
import selector_balance_data as data
import selector_balance_collections as dev
import selector_balance_native as native
import selector_snapshot_common as s
from selector_rare_common import cluster_interval


def average(values):
    b.require(len(values) == 3 and all(set(v) == set(values[0]) for v in values), 'Three matched seeds required')
    return {scene:{k:float(np.mean([v[scene][k] for v in values])) for k in d.METRICS} for scene in values[0]}


def summary(values):
    means = [s.mean(v) for v in values]
    return dict(seeds=[dict(seed=seed, mean=value) for seed,value in zip(b.SEEDS, means)],
                mean={k:float(np.mean([v[k] for v in means])) for k in d.METRICS},
                seed_std={k:float(np.std([v[k] for v in means], ddof=1)) for k in d.METRICS})


def compare(values, references, rows):
    actual, reference = average(values), average(references)
    ids = [r['scene_id'] for r in rows]
    b.require(set(ids) <= set(actual) and set(ids) <= set(reference) and len(set(ids)) == len(rows), 'Unpaired comparison')
    deltas = {k:{scene:actual[scene][k]-reference[scene][k] for scene in ids} for k in d.METRICS}
    intervals = {k:dict(cluster_interval(v, rows, samples=10000), exposure=b.EXPOSURE,
                        exploratory_unadjusted=True, conditional_on_acquisition_and_noise=True)
                 for k,v in deltas.items()}
    counts = [dict(seed=seed,
        rescued=sum(ref[i]['success'] == 0 and value[i]['success'] == 1 for i in ids),
        broken=sum(ref[i]['success'] == 1 and value[i]['success'] == 0 for i in ids))
        for seed,value,ref in zip(b.SEEDS, values, references)]
    logs = {log:[r['scene_id'] for r in rows if r['origin_log'] == log] for log in sorted({r['origin_log'] for r in rows})}
    contributions = sorted([[log, float(sum(deltas['score'][i] for i in members))/len(ids)]
                             for log,members in logs.items()], key=lambda v:(-abs(v[1]), v[0]))
    leave_one = {log:float(np.mean([deltas['score'][i] for i in ids if i not in set(members)]))
                 for log,members in logs.items() if len(members) < len(ids)}
    return dict(scenes=len(ids), delta={k:float(np.mean(list(v.values()))) for k,v in deltas.items()},
        intervals=intervals, rescued_broken=counts, largest_log_contributions=contributions[:3],
        leave_one_log_out_score=leave_one, main_result_drops_no_scenes=True)


def report(run):
    run = Path(run)
    task = b.task_from_run(run)
    spec = b.task_spec(task)
    inputs = data.checked_inputs(run)
    native_inputs = s.checked_inputs(run)
    missing, artifacts, dev_table, formal_table = [], [], {}, {}
    train_reports = {}
    for seed in b.SEEDS:
        for arm in (b.NEW_ARMS if task != 'formal' else ()):
            policy = f'{arm}_seed{seed}'
            path = run/'train'/policy/'step_500_report.json'
            if not path.exists():
                missing.append('train/'+policy)
            else:
                data.model_entry(run, policy)
                train_reports[policy] = d.artifact(path)
    for cohort, mode, policy in (b.dev_conditions() if task != 'formal' else ()):
        result = dev.checked(run, cohort, mode, policy)
        if result is None:
            missing.append(f'dev/{cohort}/{mode}/{policy}')
        else:
            dev_table[(cohort, mode, policy)] = result['metrics']
            artifacts.append(result['audit'])
    for policy, block in (b.formal_conditions() if task != 'development' else ()):
        result = native.checked_formal(run, policy, block)
        if result is None:
            missing.append('formal/'+policy+'/'+block)
        else:
            formal_table[(policy, block)] = result
            artifacts.append(result['evidence'])
    if task != 'development' and not (run/'bridge_report.json').exists():
        missing.append('native_bridge')
    elif task != 'development':
        native.require_bridge(run)
    result = dict(status='INCOMPLETE' if missing else 'PASS', decision='REVIEW_REQUIRED', missing=missing,
        complete_training_jobs=len(train_reports), complete_dev_conditions=len(dev_table),
        complete_formal_conditions=len(formal_table), **spec,
        development_noise=c.NOISE_NAMESPACE, formal_noise=b.FORMAL_NOISE, exposure=b.EXPOSURE,
        automatic_next_method=False, automatic_mid_full_evaluation=False, performance_gate_applied=False,
        training_reports=train_reports, artifacts=artifacts)
    # A finished line is useful even while the other line is incomplete.
    dev_summary, dev_comparisons, formal_summary, formal_comparisons, paired = {}, {}, {}, {}, []
    if len(dev_table) == 48:
        for cohort in ('development', 'common'):
            rows = inputs['cohorts'][cohort]['rows']
            for mode in b.MODES:
                key = cohort+'_'+mode
                scalar = dev.checked(run, cohort, mode, 'scalar_v3')['metrics']
                values = {arm:[dev_table[(cohort,mode,f'{arm}_seed{seed}')] for seed in b.SEEDS] for arm in b.ARMS}
                dev_summary[key] = {arm:summary(v) for arm,v in values.items()}
                dev_comparisons[key] = {}
                strata = dict(failed=[r for r in rows if scalar[r['scene_id']]['success'] == 0],
                    safe_slow=[r for r in rows if scalar[r['scene_id']]['success'] == 1 and scalar[r['scene_id']]['ego_progress'] < .5],
                    safe_other=[r for r in rows if scalar[r['scene_id']]['success'] == 1 and scalar[r['scene_id']]['ego_progress'] >= .5])
                dev_comparisons[key]['baseline_strata'] = {
                    name:{arm:compare(v, [scalar]*3, rs) for arm,v in values.items()} if rs else None
                    for name,rs in strata.items()}
                for arm in b.ARMS:
                    for reference in ('P1', 'CL_NLL', 'scalar_v3'):
                        if arm == reference:
                            continue
                        ref = [scalar]*3 if reference == 'scalar_v3' else values[reference]
                        dev_comparisons[key][arm+'__minus__'+reference] = compare(values[arm], ref, rows)
                    for seed, value in zip(b.SEEDS, values[arm]):
                        for row in rows:
                            scene = row['scene_id']
                            paired.append(dict(protocol='development', cohort=cohort, mode=mode, arm=arm,
                                seed=seed, **row, **{k:value[scene][k] for k in d.METRICS},
                                **{'scalar_'+k:scalar[scene][k] for k in d.METRICS}))
        result['joint_mean_candidates'] = {arm:b.joint_mean_candidate({
            key:dev_comparisons[key][arm+'__minus__P1']['delta']['score'] for key in dev_comparisons})
            for arm in b.NEW_ARMS}
        result['joint_mean_candidate_note'] = 'Descriptive four-cell mean condition only; not significance or noninferiority.'
    if len(formal_table) == 42:
        for block in b.BLOCKS:
            values = {arm:[formal_table[(f'{arm}_seed{seed}',block)]['metrics'] for seed in b.SEEDS]
                      for arm in b.FORMAL_ARMS}
            scalar = formal_table[('scalar_v3',block)]['metrics']
            formal_summary[block] = {arm:summary(v) for arm,v in values.items()}
            formal_summary[block]['scalar_v3'] = dict(mean=s.mean(scalar), fixed_reference=True)
            if block.startswith('open'):
                if block == 'open_navtest':
                    for arm in b.FORMAL_ARMS:
                        ade = [formal_table[(f'{arm}_seed{seed}',block)]['ade_fde'] for seed in b.SEEDS]
                        formal_summary[block][arm]['ade_fde_seeds'] = ade
                continue
            cohort, mode = block.rsplit('_', 1)
            rows = native_inputs['cohorts'][cohort]
            subsets = {cohort:rows}
            if cohort == 'full':
                exposed = {r['scene_id'] for r in native_inputs['cohorts']['development']}
                subsets.update(development58=[r for r in rows if r['scene_id'] in exposed],
                               remainder230=[r for r in rows if r['scene_id'] not in exposed])
            for label, subset in subsets.items():
                key = label+'_'+mode
                formal_comparisons[key] = {
                    'CL_NLL__minus__P1':compare(values['CL_NLL'], values['P1'], subset),
                    **{arm+'__minus__scalar_v3':compare(v, [scalar]*3, subset) for arm,v in values.items()}}
                # Fixed V3 strata, never selected using new methods.
                strata = dict(failed=[r for r in subset if scalar[r['scene_id']]['success'] == 0],
                    safe_slow=[r for r in subset if scalar[r['scene_id']]['success'] == 1
                               and scalar[r['scene_id']]['ego_progress'] < .5],
                    safe_other=[r for r in subset if scalar[r['scene_id']]['success'] == 1
                                and scalar[r['scene_id']]['ego_progress'] >= .5])
                formal_comparisons[key]['baseline_strata'] = {
                    name:{arm:compare(v, [scalar]*3, rs) for arm,v in values.items()} if rs else None
                    for name,rs in strata.items()}
            for arm in b.FORMAL_ARMS:
                for seed,value in zip(b.SEEDS,values[arm]):
                    for row in rows:
                        scene = row['scene_id']
                        paired.append(dict(protocol='formal', cohort=cohort, mode=mode, arm=arm, seed=seed,
                            **row, **{k:value[scene][k] for k in d.METRICS},
                            **{'scalar_'+k:scalar[scene][k] for k in d.METRICS}))
    if len(formal_table) == 42:
        keys = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
                'time_to_collision_within_bound', 'comfort', 'score')
        blocks = ('open_navtest', 'open_rare', 'full_NR', 'full_R')
        header = ['model', 'note', 'ADE', 'FDE', *[block+'_'+k for block in blocks for k in keys], 'CL_R_success']
        tsv = ['\t'.join(header)]
        for policy in b.FORMAL_POLICIES:
            means = {block:s.mean(formal_table[(policy,block)]['metrics']) for block in blocks}
            ade = formal_table[(policy,'open_navtest')]['ade_fde']
            row = [policy, 'paired optimization seeds; fixed native noise; not blind',
                   *[str(ade[k]) for k in ('ade_4s','fde_4s')],
                   *[str(means[block][k]) for block in blocks for k in keys], str(means['full_R']['success'])]
            tsv.append('\t'.join(row))
        temp = run/'summary.tsv.tmp'
        temp.write_text('\n'.join(tsv)+'\n'); temp.replace(run/'summary.tsv')
        result['formal_29_columns'] = d.artifact(run/'summary.tsv')
    result.update(development_summary=dev_summary, development_comparisons=dev_comparisons,
                  formal_summary=formal_summary, formal_comparisons=formal_comparisons)
    c.atomic_json(run/'progress.json', result)
    if paired:
        columns = sorted({k for row in paired for k in row})
        path = run/'paired_scenes.csv'
        temp = path.with_suffix('.tmp')
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader(); writer.writerows(paired)
        temp.replace(path)
        result['paired_scenes'] = d.artifact(path)
    lines = ['# P1 balance / '+task+': '+result['status'], '', 'Separate development/native protocols; all outcomes retained.',
             '', 'Decision: REVIEW_REQUIRED. No automatic promotion or next experiment.', '',
             '| Protocol | Block | Arm | PDM mean | Seed std |', '|---|---|---|---:|---:|']
    for protocol, table in (('development', dev_summary), ('formal', formal_summary)):
        for block, arms in table.items():
            for arm, value in arms.items():
                std = value.get('seed_std', {}).get('score')
                std_text = 'fixed' if std is None else f'{std:.6f}'
                lines.append(f"| {protocol} | {block} | {arm} | {value['mean']['score']:.6f} | {std_text} |")
    lines += ['', 'Common64 is a retention cohort, not full-common or an unseen test.',
              'Log intervals condition on one acquisition and evaluation noise; they do not replace seed uncertainty.',
              'Three optimization seeds; no observed-best selection. See JSON for all safety components and rescued/broken.',
              '', 'Missing: '+str(len(missing))]
    temp = run/'summary.md.tmp'; temp.write_text('\n'.join(lines)+'\n'); temp.replace(run/'summary.md')
    if not missing:
        c.locked_json(run/'summary.json', result)
        c.locked_json(run/'archive/release_manifest.json', dict(method=b.METHOD, status='PASS',
            contract=d.artifact(run/'run_contract.json'), audit=d.artifact(run/'audit_report.json'),
            inputs=d.artifact(run/'balance_inputs.json'), native_inputs=d.artifact(run/'inputs.json'),
            summary=d.artifact(run/'summary.json'), task=task,
            outputs=[d.artifact(run/'summary.md'), result['paired_scenes']]
                + ([result['formal_29_columns']] if 'formal_29_columns' in result else []),
            source_artifacts=inputs['artifacts'], exposure=b.EXPOSURE))
    return result
