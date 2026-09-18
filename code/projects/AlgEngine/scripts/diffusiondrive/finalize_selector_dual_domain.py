"""Strict physical-state completion of an offline analysis; no model updates.

The initial analysis contains selector-visible prefix comparisons. This consumes
that immutable result, adds raw physical state checks, and qualifies conclusions.
"""
import argparse
import copy
import json
from pathlib import Path
import cfpi_common as c
from selector_dual_domain_analysis import Evidence,output_location,require
from selector_dual_domain_state import physical_state,physical_equal
from selector_dual_domain_review import review,review_markdown
import numpy as np


def complete(args):
    source=Path(args.analysis_run).resolve()
    ev=Evidence()
    report=ev.read(source/'report.json')
    require(report['status']=='PASS' and report['optimizer_steps']==0 and report['new_rollouts']==0,'Incomplete source analysis')
    for item in report['outputs'].values():ev.use(item)
    manifest=ev.read(source/'input_manifest.json')
    artifacts={x['path']:x for x in manifest['artifacts']}
    inputs_path=next(x for x in artifacts if x.endswith('/attribution_inputs.json'))
    attribution=Path(inputs_path).parent
    snapshot=Path(report['command']['snapshot_run']).resolve()
    output=output_location(args.output,[source,attribution,snapshot])
    inputs=ev.read(artifacts[inputs_path])
    original_report=ev.read(artifacts[str(attribution/'attribution_report.json')])
    for arm,values in original_report['feedback'].items():
        for value in values:ev.use(value['model'],model=True)
    for entry in inputs['parents'].values():ev.use(entry,model=True)
    stats=ev.read(source/'statistics.json')
    probes=ev.read(source/'training_probes.json')
    cases=copy.deepcopy(ev.read(source/'decision_cases.json'))
    raw_records=0
    class Reader:
        pass
    reader=Reader();reader.ev=ev
    for index,case in enumerate(cases['cases']):
        print(json.dumps(dict(stage='physical_state',case=index+1,total=len(cases['cases']))),flush=True)
        physical={}
        for arm in ('P1','CL_NLL'):
            name=f'eval_{case["cohort"]}_{case["mode"]}_{arm}_seed{case["seed"]}'
            entry=inputs['reused_collections'].get(name,attribution/'collections'/name/'deployment_collection.json')
            root=ev.use(entry).parent
            audit=ev.read(artifacts[str(root/'collection_audit.json')])
            indexed={x['path']:x for x in audit['artifacts']}
            states={}
            for path in sorted(root.glob(f'split_*/WE_output/openscene_format/cfpi_deployment_records/{c.scene_token(case["scene_id"])}_*.json')):
                record=ev.read(indexed[str(path.resolve())]);step=record['validation']['decision_step']
                require(record['scene_id']==case['scene_id'] and record['state_step']==step-1,'Wrong action context')
                require(step in c.DECISION_STEPS and step not in states,'Wrong executed coverage')
                sidecar=c.load_pickle(ev.use(record['sidecar']))
                states[step]=physical_state(reader,sidecar,root,case['scene_id'],step)
                for key in ('waypoints','velocities','headings','angular_velocities'):
                    actual,expected=record['actual_action'][key],record['expected_action'][key]
                    require((actual is None and expected is None) or (actual is not None and expected is not None
                            and np.asarray(actual).shape==np.asarray(expected).shape
                            and np.allclose(actual,expected,rtol=0,atol=c.ACTION_TOLERANCE)), 'Consumed action mismatch')
                raw_records+=1
            require(set(states)==set(c.DECISION_STEPS),'Missing physical states')
            physical[arm]=states
        prefix=True
        for row in case['comparison']['steps']:
            step=row['step']
            same=physical_equal(physical['P1'][step],physical['CL_NLL'][step])
            row.update(physical_state_equal=same,shared_prefix=prefix,
                       comparable=bool(prefix and same and row['same_context']))
            prefix=bool(prefix and same and row['same_context'] and row['action_same'])
        case['comparison']['first_shared_context_action_divergence']=next(
            (v for v in case['comparison']['steps'] if v['comparable'] and not v['action_same']),None)
        case['comparison']['physical_state_verified']=True
    interpretation=review(stats,probes,cases)
    for path in Path(__file__).parent.glob('*dual_domain*.py'):ev.use(path)
    output.mkdir(parents=True,exist_ok=True)
    c.atomic_json(output/'decision_cases.json',cases)
    c.atomic_json(output/'interpretation.json',interpretation)
    with (output/'review.md').open('x') as stream:
        stream.write('# P1 双域分析：物理状态校验后的最终复核\n\n')
        stream.write('源统计与训练探针保持只读，见 source_report 引用；本报告将逐决策对照升级为实际 ego/周围交通状态与可见特征、执行前缀同时一致。\n')
        stream.write(review_markdown(interpretation))
    c.atomic_json(output/'input_manifest.json',ev.finish())
    c.atomic_json(output/'report.json',dict(status='PASS',decision='REVIEW_REQUIRED',
        optimizer_steps=0,new_rollouts=0,physical_records=raw_records,completed_conditions=80,
        seed_cases=len(cases['cases']),source_report=ev.entries[str(source/'report.json')],
        source_statistics=ev.entries[str(source/'statistics.json')],
        source_probes=ev.entries[str(source/'training_probes.json')],
        outputs={p.name:dict(path=str(p),sha256=c.sha256_file(p)) for p in sorted(output.iterdir())},
        automatic_training=False,automatic_full_evaluation=False))
    print(json.dumps(dict(stage='complete',output=str(output))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-run',required=True)
    parser.add_argument('--output',required=True)
    complete(parser.parse_args())


if __name__=='__main__':main()
