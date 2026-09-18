"""Executed-step comparisons and cross-scoring; never simulate a changed action."""
import csv
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_feedback_common as f
import selector_cfpi_model as m
from selector_dual_domain_analysis import require
from selector_mid_followup_probes import score_rows,state_hash
from selector_dual_domain_state import physical_state,physical_equal
from audit_selector_cfpi_deployment import action_error


def same_context(left,right):
    return all(np.asarray(left[k]).shape==np.asarray(right[k]).shape and
               np.allclose(left[k],right[k],rtol=0,atol=c.ARRAY_TOLERANCE) for k in f.FINGERPRINT_FIELDS)


def first_divergence(left,right,left_physical,right_physical):
    require(set(left)==set(right)==set(c.DECISION_STEPS),'Need all executed steps 4..11')
    prefix=True
    result=[]
    for step in c.DECISION_STEPS:
        l,r=left[step],right[step]
        context=same_context(l,r)
        physical=physical_equal(left_physical[step],right_physical[step])
        action_same=bool(np.allclose(l['deployed_trajectory'],r['deployed_trajectory'],rtol=0,atol=c.ACTION_TOLERANCE))
        result.append(dict(step=step,shared_prefix=prefix,same_context=context,same_physical_state=physical,
            comparable=prefix and context and physical,left_index=int(l['selected_index']),right_index=int(r['selected_index']),
            action_same=action_same))
        prefix=prefix and context and physical and action_same
    first=next((v for v in result if v['comparable'] and not v['action_same']),None)
    return dict(steps=result,first_shared_context_action_divergence=first,
                note='Later diverged states are not matched counterfactual outcomes.')


def case_analysis(study,cases,progress):
    ev=study.ev
    outputs=[]
    for seed in (0,1,2):
        models={arm:m.load_selector(ev.use(study.models[f'{arm}_seed{seed}'],model=True))[0].eval()
                for arm in study.arms}
        before={arm:state_hash(model) for arm,model in models.items()}
        for ci,case in enumerate(cases):
            progress('case',seed=seed,index=ci+1,total=len(cases))
            scene,cohort,mode=case['scene_id'],case['cohort'],case['mode']
            token=c.scene_token(scene)
            trajectories={}
            physical={} 
            outcomes={}
            cross=[]
            for arm in (case['left_arm'],case['right_arm']):
                info=study.collections[(cohort,mode,f'{arm}_seed{seed}')]
                root=info['path'].parent
                audited={str(Path(x['path']).resolve()):x for x in info['audit']['artifacts']}
                decisions={}
                states={}
                worker=None
                for path in sorted(root.glob(f'split_*/WE_output/openscene_format/cfpi_deployment_records/{token}_*.json')):
                    record=ev.read(audited[str(path.resolve())])
                    require(record['scene_id']==scene and record['collection_sha256']==ev.entries[str(info['path'].resolve())]['sha256'], 'Foreign action record')
                    step=record['validation']['decision_step']
                    require(step in c.DECISION_STEPS and step not in decisions,'Duplicate or unexecuted step')
                    sidecar=c.load_pickle(ev.use(record['sidecar']))
                    require(sidecar['candidate_noise_namespace']==c.NOISE_NAMESPACE and int(sidecar['planner_step'])==step,'Wrong sidecar noise/step')
                    require(sidecar['cfpi_deployment']['selector_sha256']==study.models[f'{arm}_seed{seed}']['sha256'],'Foreign selector sidecar')
                    require(int(sidecar['selected_index'])==int(np.argmax(sidecar['current_logits'])),'Recorded action not argmax')
                    decisions[step]=sidecar
                    states[step]=physical_state(study,sidecar,root,scene,step)
                    require(action_error(record['actual_action'],record['expected_action'])<=c.ACTION_TOLERANCE,'Executed action differs from expected action')
                    worker=path.relative_to(root).parts[0]
                require(set(decisions)==set(c.DECISION_STEPS),'Missing executed decisions')
                trajectories[arm]=decisions
                physical[arm]=states
                metrics=dict(study.table[(cohort,mode,f'{arm}_seed{seed}')][scene])
                metric_path=root/worker/f'WE_output/openscene_format/all_scenes_pdm_averages_{mode}.csv'
                with ev.use(audited[str(metric_path.resolve())]).open() as stream:
                    original=[r for r in csv.DictReader(stream) if r['token']==scene]
                require(len(original)==1,'Missing raw outcome')
                raw=original[0].get('first_violation_step','')
                metrics.update(first_violation_step=int(float(raw)) if raw else None,
                    violation_time_status='RECORDED' if raw else ('NO_FAILURE' if metrics['success']==1 else 'NOT_RECORDED'))
                outcomes[arm]=metrics
                contexts=[decisions[s] for s in c.DECISION_STEPS]
                scores={name:score_rows(model,contexts) for name,model in models.items()}
                error=float(np.max(np.abs(scores[arm]-np.stack([x['current_logits'] for x in contexts]))))
                require(error<=c.MODEL_RECOMPUTE_TOLERANCE and
                    all(int(scores[arm][i].argmax())==int(row['selected_index']) for i,row in enumerate(contexts)),
                    f'CPU own-model parity failure {arm}/{seed}/{scene}: {error}')
                for i,step in enumerate(c.DECISION_STEPS):
                    cross.append(dict(visiting_arm=arm,step=step,own_model_max_abs=error,
                        argmax={k:int(z[i].argmax()) for k,z in scores.items()},
                        logits={k:z[i].tolist() for k,z in scores.items()},
                        fingerprint=f.fingerprint(contexts[i])))
            outputs.append(dict(case,seed=seed,outcomes=outcomes,cross_scores=cross,
                                comparison=first_divergence(trajectories[case['left_arm']],trajectories[case['right_arm']],physical[case['left_arm']],physical[case['right_arm']])))
        require(all(before[k]==state_hash(model) for k,model in models.items()),'Cross-scoring changed models')
    return dict(status='PASS',cases=outputs,optimizer_steps=0,
                note='Cross-scores have no new rollout return. Missing violation time stays unknown. Cases are post-hoc evidence, never training labels.')
