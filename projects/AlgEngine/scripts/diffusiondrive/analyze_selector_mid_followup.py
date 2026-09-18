"""Independent CPU-only evidence; no training, rollout, export or source writes."""
import argparse
import json
from pathlib import Path
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
from selector_dual_domain_analysis import output_location
from selector_mid_followup_analysis import Study,formal_comparison
from selector_mid_followup_probes import training_probes
from selector_mid_followup_cases import case_analysis
from run_selector_mid_followup import SOURCE,ROOT


def progress(stage,**details):
    print(json.dumps(dict(stage=stage,**details)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=SOURCE)
    parser.add_argument('--sensitivity-run',type=Path)
    parser.add_argument('--mid-formal-run',type=Path)
    parser.add_argument('--reference-formal-run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.threads<=16:parser.error('Use 1..16 CPU threads')
    if bool(args.mid_formal_run)!=bool(args.reference_formal_run):parser.error('Supply both completed formal runs together')
    roots=[x for x in (args.source_run,args.sensitivity_run,args.mid_formal_run,args.reference_formal_run) if x]
    output=output_location(args.output,roots);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    c.locked_json(output/'analysis_contract.json',dict(method='selector_mid_followup_analysis_v1',cpu_only=True,
        source_runs=[str(x.resolve()) for x in roots],primary_coefficient=.5,optimizer_steps=0,
        case_selection='All per-seed safety disagreements + top3 positive/negative independent logs per domain/mode/comparison',
        implementation={str(p.relative_to(ROOT)):c.sha256_file(p) for p in (ROOT/'projects/AlgEngine/scripts/diffusiondrive').glob('*.py')}))
    try:
        study=Study(args.source_run,args.sensitivity_run,progress)
        progress('statistics');stats=study.statistics();c.locked_json(output/'statistics.json',stats)
        progress('training_probes');probes=training_probes(study,progress);c.locked_json(output/'training_probes.json',probes)
        progress('decision_cases',count=len(stats['cases']))
        cases=case_analysis(study,stats['cases'],progress);c.locked_json(output/'decision_cases.json',cases)
        formal=dict(status='NOT_REQUESTED',note='Supply both completed formal runs to add the matched MID/P1/CL_NLL comparison.')
        if args.mid_formal_run:formal=formal_comparison(study.ev,args.mid_formal_run,args.reference_formal_run)
        c.locked_json(output/'formal_comparison.json',formal)
        from selector_mid_followup_hashes import uncached_hashes
        with uncached_hashes():
            evidence=study.ev.finish()
        c.locked_json(output/'input_manifest.json',evidence)
        review=['# MID evidence review','', 'Decision: REVIEW_REQUIRED. Primary lambda remains 0.5.', '',
            'All 256 training states and three seeds are retained. Training fit, gradient direction and scale are diagnostics; they do not establish a generalization mechanism.',
            'Only the first action divergence with matching physical state, visible context and shared executed prefix is comparable. Later states carry no counterfactual return.',
            'Safety disagreement cases and top positive/negative independent logs are post-hoc evidence, never additional training labels.',
            'Read statistics.json for paired PDM/success effects and log sensitivity; training_probes.json for pair/mass/outside-selection and gradients; decision_cases.json for executed evidence.',
            'A full native table is one part of a paper: stronger baselines, independent settings and a substantiated mechanism are still needed.',
            '', '| Block | Arm | PDM | Success | Seed PDM std |','|---|---|---:|---:|---:|']
        for block,table in stats['summaries'].items():
            for arm,v in table['arms'].items():review.append(f"| {block} | {arm} | {v['mean']['score']:.6f} | {v['mean']['success']:.6f} | {v['seed_std']['score']:.6f} |")
        (output/'review.md').write_text('\n'.join(review)+'\n')
        result=dict(status='PASS',decision='REVIEW_REQUIRED',cpu_only=True,optimizer_steps=0,new_rollouts=0,
            cases=len(cases['cases']),feedback_states=len(study.rows),formal_comparison=formal['status'],
            outputs=[d.artifact(p) for p in sorted(output.iterdir()) if p.is_file()])
        c.locked_json(output/'report.json',result);progress('complete',output=str(output))
    except BaseException as error:
        c.atomic_json(output/'failure.json',dict(status='FAILED',error=repr(error)));raise


if __name__=='__main__':
    from selector_mid_followup_hashes import verified_hash_cache
    with verified_hash_cache():
        main()
