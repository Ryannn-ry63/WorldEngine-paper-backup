"""Evidence-qualified conclusions, not an automatic experiment launcher."""
import numpy as np


def review(stats,probes,cases):
    diagnostics={}
    for arm in ('shared','P1','CL_NLL','Continue'):
        for mode in ('NR','R'):
            selected=[v for v in probes['diagnostics'] if v['arm']==arm and v['mode']==mode
                      and v['source']=='closed' and 'pair_correct' in v]
            diagnostics[arm+'_'+mode]=dict(seed_state_observations=len(selected),
                **{k:float(np.mean([v[k] for v in selected])) for k in
                   ('pair_mass','pair_correct','winner_selected','known_loser_selected','outside_selected')})
    gradient_summary={}
    for arm in ('shared','P1','CL_NLL','Continue'):
        selected=[v for v in probes['gradients'] if v['arm']==arm]
        gradient_summary[arm]=dict(batches=len(selected),
            median_norms={k:float(np.median([v['norms'][k] for v in selected])) for k in
                          ('pair','set_mass','nll','dense','kl')},
            cosine={key:dict(available=len(values),median=float(np.median(values)) if values else None,
                             negative_fraction=float(np.mean(np.array(values)<0)) if values else None)
                    for key in ('pair__set_mass','pair__nll','pair__dense','set_mass__dense','nll__dense')
                    for values in [[v['cosines'][key] for v in selected if v['cosines'].get(key) is not None]]})
    comparable=[v for v in cases['cases'] if v['comparison']['first_shared_context_action_divergence'] is not None]
    unknown=sum(v['violation_time_status']=='NOT_RECORDED' for case in cases['cases'] for v in case['outcomes'].values())
    result=dict(decision='REVIEW_REQUIRED',recommended_next='WEIGHT_AND_SCALE_CONTROL_FIRST',
        diagnostics=diagnostics,gradient_summary=gradient_summary,
        case_summary=dict(seed_cases=len(cases['cases']),shared_state_action_divergences=len(comparable),
                          failure_outcomes_without_time=unknown),
        hypotheses={
            'nll_contains_pair':dict(verdict='SUPPORTED',reason='Exact same-gap/full-denominator identity; synthetic value and gradient tests.'),
            'pair_and_nll_intrinsically_opposite':dict(verdict='REJECTED_AS_GENERAL_CLAIM',reason='Same-state logit gradients are not opposite; parameter-space coupling is a different question.'),
            'dual_domain_tradeoff_is_unavoidable':dict(verdict='INSUFFICIENT_EVIDENCE',reason='Only fixed endpoint objectives observed; no matched-strength intermediate control.'),
            'partial_feedback_causes_common_failures':dict(verdict='INSUFFICIENT_EVIDENCE',reason='Unknown candidates have no true branch labels; cross-scores and gradients do not identify counterfactual outcomes.'),
            'cl_nll_is_globally_better':dict(verdict='NOT_SUPPORTED_BY_CURRENT_MEANS',reason='CL_NLL improves common means but lowers rare means; full protocol not yet evaluated.'),
        },
        next_protocol=dict(authorized_to_launch=False,primary='rare/common PDM jointly, NR/R separate; report safety components and new failures',
            controls=[dict(name='CL_NLL',pair_extra=0),dict(name='MID',pair_extra=.5),dict(name='P1',pair_extra=1),
                      dict(name='MID_SCALE_MATCH',pair_extra=.5)],
            scale_match='Per seed: sqrt(sum ||g_NLL+g_pair||^2 / sum ||g_NLL+0.5*g_pair||^2), over same initial shared-parent 16-state batches, training data only. Multiply the entire feedback objective; do not scale GRPO/KL. Freeze before evaluation; zero denominator stops.',
            training='Original shared parents, optimizer/RNG, 256 rows, dense cache, architecture, 500 steps; seeds0/1/2; final step only.',
            evaluation='rare58/common64 NR/R fixed noise; retain all seeds and scenes; endpoint reuse only after identity checks.',
            conclusion='Fixed-weight success is an engineering control, not paper novelty. New support-aware objectives require independent training-side evidence and matched controls; no P3 restart.',
            full_evaluation='Separate decision after review; freeze all3 CL_NLL seeds and evaluate P1 under identical native protocol, no observed-best picking.'),
        optimizer_steps=0,new_rollouts=0)
    return result


def review_markdown(result):
    lines=['','## 本次实际机制统计','',
           '| 模型/模式 | 对内排序正确率 | 已比较候选总概率 | 胜者实际top1 | 已知败者top1 |',
           '|---|---:|---:|---:|---:|']
    for key,v in result['diagnostics'].items():
        lines.append('| '+key+' | '+' | '.join(f'{v[k]:.4f}' for k in
                     ('pair_correct','pair_mass','winner_selected','known_loser_selected'))+' |')
    lines+=['','这是训练缓存上的三 seed 平均，不是闭环测试分数；不能将 state×seed 当作独立样本。','',
            '下一步建议：先排除固定权重与监督总强度解释，不立即设计复杂新模型，也不立即运行完整评测。','',
            '| 假说 | 判断 |','|---|---|']
    for key,v in result['hypotheses'].items():lines.append(f'| {key} | {v["verdict"]} |')
    lines+=['','未比较候选的真实反事实回报仍未知；此轮不宣称已经发现 common 失败的因果根因。',
            '具体四组最小验证协议见 interpretation.json；没有启动权限或自动执行入口。']
    return '\n'.join(lines)+'\n'
