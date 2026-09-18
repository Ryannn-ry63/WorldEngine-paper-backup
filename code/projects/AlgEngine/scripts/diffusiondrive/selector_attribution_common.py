"""P1 stage-two attribution, not a new policy or a novelty claim."""
import math
import numpy as np
import selector_feedback_common as f

METHOD = 'selector_p1_attribution_v1'
ARMS = ('L_Pair', 'L_Pair_NLL', 'P0', 'P1', 'Continue', 'CL_NLL')
NEW_ARMS = ('L_Pair', 'L_Pair_NLL', 'Continue', 'CL_NLL')
SEEDS = (0, 1, 2)
MODES = ('NR', 'R')
TRAIN = dict(steps=500, lr=1e-4, weight_decay=1e-4, grad_clip=10.,
             temperature=1., dense_batch=16, feedback_batch=16, teacher_kl=.01,
             pair_coefficient=1., winner_nll_coefficient=1.)
EXPOSURE = ('Second-stage attribution conditional on shared closed-loop-trained parents, '
            'seed0-acquired feedback and fixed noise; legacy-exposed rare58/common64; '
            'not independent acquisition, blind testing or full-pipeline attribution.')
LOCAL_DEFINITION = dict(
    name='preaction_logged_future_pdm_v1', ego_state='actual_visiting_preaction_state',
    candidates='original20_at_identical_visiting_state',
    surrounding_future='logged_scene_tracks_not_reactive_branch_outcomes',
    simulator='legacy_dynamic_reward_PDMSimulator_or_TorchSimulator',
    aggregation='legacy_pairwise_PDM_reference_normalization',
    horizon='40_poses_at_0.1_seconds', training_consumes='original_two_indices_only',
    pure_horizon_comparison=False, inference_uses_labels=False)
COMPARISONS = {
    'nll_under_closed': {'P1': 1., 'P0': -1.},
    'feedback_source_under_nll': {'P1': 1., 'L_Pair_NLL': -1.},
    'source_by_nll_interaction': {'P1': 1., 'P0': -1., 'L_Pair_NLL': -1., 'L_Pair': 1.},
    'feedback_vs_continue': {'P1': 1., 'Continue': -1.},
    'pair_increment': {'P1': 1., 'CL_NLL': -1.},
}


def components(arm):
    if arm not in ARMS:
        raise ValueError('Unknown attribution arm')
    return dict(source='local' if arm.startswith('L_') else 'closed',
                pair=arm in ('L_Pair', 'L_Pair_NLL', 'P0', 'P1'),
                nll=arm in ('L_Pair_NLL', 'P1', 'CL_NLL'))


def preference(row, source):
    if len(row['candidate_indices']) != 2 or len(set(row['candidate_indices'])) != 2:
        raise RuntimeError('Attribution must retain the original pair, no added queries')
    if source not in ('local', 'closed'):
        raise ValueError('Unknown label source')
    outcomes = row['branch_outcomes'] if source == 'closed' else row['local_pair_outcomes']
    if len(outcomes) != 2:
        raise RuntimeError('Missing pair outcomes')
    for v in outcomes:
        if v['success'] != int(f.safe(v)):
            raise RuntimeError('NC&DAC success label mismatch')
    return f.pareto_pair(*outcomes)


def feedback_loss(logits, rows, arm):
    import torch
    import torch.nn.functional as F
    spec = components(arm)
    if logits.shape != (len(rows), 20) or not rows:
        raise ValueError('Require full20 logits for all sampled feedback states')
    zero = logits.sum()*0.
    if not spec['pair'] and not spec['nll']:
        return zero, zero
    logp = F.log_softmax(logits, dim=-1) if spec['nll'] else None
    pairs, nlls = [], []
    for i, row in enumerate(rows):
        p = preference(row, spec['source'])
        if p is None:
            continue
        winner, loser, gap = p
        a, b = (row['candidate_indices'][k] for k in (winner, loser))
        if spec['pair']:
            pairs.append(float(gap)*F.softplus(logits[i, b]-logits[i, a]))
        if spec['nll']:
            nlls.append(-float(gap)*logp[i, a])
    # Same as frozen P0/P1 for two candidates: denominator is ALL sampled rows.
    return (torch.stack(pairs).sum()/len(rows) if pairs else zero,
            torch.stack(nlls).sum()/len(rows) if nlls else zero)


def local_outcomes(scores, values, indices):
    scores, values = np.asarray(scores), np.asarray(values)
    if (scores.shape != (20,) or values.shape != (20, 6) or
            not np.isfinite(scores).all() or not np.isfinite(values).all() or
            np.any((scores < 0) | (scores > 1)) or np.any((values < 0) | (values > 1))):
        raise RuntimeError('Invalid same-state local labels')
    if len(indices) != 2 or len(set(indices)) != 2 or any(type(i) is not int or not 0 <= i < 20 for i in indices):
        raise RuntimeError('Wrong original pair')
    names = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
             'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance')
    result = []
    for i in indices:
        v = dict(zip(names, map(float, values[i])))
        v.update(score=float(scores[i]), success=float(f.safe(v)))
        result.append(v)
    return result


def label_summary(rows):
    result = {}
    for mode in MODES:
        subset = [r for r in rows if r['mode'] == mode]
        item = dict(states=len(subset), reversal=0, both_informative=0)
        for source in ('closed', 'local'):
            ps = [preference(r, source) for r in subset]
            item[source] = dict(informative=sum(p is not None for p in ps),
                                gap_sum=sum(p[2] for p in ps if p is not None))
        for row in subset:
            a, b = preference(row, 'closed'), preference(row, 'local')
            if a is not None and b is not None:
                item['both_informative'] += 1
                item['reversal'] += a[0] != b[0]
        result[mode] = item
    return result


def interval_verdict(interval):
    low, high = interval
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError('Invalid interval')
    return 'POSITIVE_SIGNAL' if low > 0 else 'NEGATIVE_SIGNAL' if high < 0 else 'INCONCLUSIVE'
