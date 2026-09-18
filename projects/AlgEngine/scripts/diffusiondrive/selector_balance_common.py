"""Frozen P1 balance experiment. Weight controls are not a novelty claim."""
import math
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_attribution_common as a

METHOD = 'selector_p1_balance_v1'
SEEDS = (0, 1, 2)
MODES = ('NR', 'R')
ARMS = ('CL_NLL', 'MID', 'P1', 'MID_SCALE_MATCH')
NEW_ARMS = ('MID', 'MID_SCALE_MATCH')
FORMAL_ARMS = ('P1', 'CL_NLL')
FORMAL_POLICIES = ('scalar_v3',) + tuple(f'{a}_seed{s}' for a in FORMAL_ARMS for s in SEEDS)
NEW_NATIVE = ('P1_seed0', 'P1_seed1', 'CL_NLL_seed0', 'CL_NLL_seed1', 'CL_NLL_seed2')
BLOCKS = ('open_navtest', 'open_rare', 'full_NR', 'full_R', 'common_NR', 'common_R')
FORMAL_NOISE = 'formal_navtest_seed0'
TRAIN = dict(a.TRAIN, pair_coefficients={k: (0. if k == 'CL_NLL' else 1. if k == 'P1' else .5) for k in ARMS},
             scale_probe='eval-mode; original ordered256 rows in consecutive16 batches; initial shared parent',
             scale_applies_to='feedback_only', final_checkpoint_only=True)
EXPOSURE = dict(independent_blind_test=False, independent_acquisition_replicates=False,
                acquisition_seed=0, optimization_seeds=list(SEEDS), evaluation_noise_replicates=1,
                development_and_legacy_exposed=True, common64_is_full_common=False)
STAGES = ('audit', 'preflight', 'train', 'dev-eval', 'native-bridge', 'formal-eval', 'report', 'all')
TASKS = ('combined', 'development', 'formal')


def task_spec(task='combined'):
    require(task in TASKS, 'Unknown balance task')
    development, formal = task != 'formal', task != 'development'
    return dict(task=task, expected_training_jobs=6 if development else 0,
        expected_dev_conditions=48 if development else 0,
        reused_dev_conditions=24 if development else 0, new_dev_conditions=24 if development else 0,
        expected_bridge_conditions=10 if formal else 0,
        expected_formal_conditions=42 if formal else 0,
        reused_formal_conditions=12 if formal else 0, new_formal_conditions=30 if formal else 0)


def task_from_run(run):
    import json
    task = json.loads((Path(run)/'run_contract.json').read_text()).get('task', 'combined')
    task_spec(task)
    return task


def validate_stage(task, stage):
    task_spec(task)
    require(stage in STAGES, 'Unknown stage')
    require(not (task == 'formal' and stage in ('train', 'dev-eval'))
        and not (task == 'development' and stage in ('native-bridge', 'formal-eval')),
        'Stage is outside the frozen task scope')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def specification(arm, alpha=1.):
    require(arm in ARMS, 'Unregistered balance arm')
    require(math.isfinite(alpha) and alpha > 0, 'Invalid feedback scale')
    require(arm == 'MID_SCALE_MATCH' or alpha == 1., 'Only scale control may rescale feedback')
    return dict(pair_extra=TRAIN['pair_coefficients'][arm], feedback_scale=float(alpha))


def objective(logits, rows, arm, alpha=1.):
    spec = specification(arm, alpha)
    pair, nll = a.feedback_loss(logits, rows, 'P1')
    return alpha * (nll + spec['pair_extra'] * pair), pair, nll


def matched_scale(numerator, denominator):
    require(math.isfinite(numerator) and math.isfinite(denominator)
            and numerator > 0 and denominator > 0, 'Zero/nonfinite scale probe energy')
    result = math.sqrt(numerator / denominator)
    require(math.isfinite(result) and result > 0, 'Invalid frozen scale')
    return result


def dev_conditions():
    return [(cohort, mode, f'{arm}_seed{seed}') for cohort in ('development', 'common')
            for mode in MODES for arm in ARMS for seed in SEEDS]


def formal_conditions():
    return [(policy, block) for policy in FORMAL_POLICIES for block in BLOCKS]


def queue(task='combined'):
    """One allocation, deterministic interleaving; never two four-GPU runners."""
    tasks = []
    for seed in SEEDS:
        tasks += [('dev', cohort, mode, f'{arm}_seed{seed}') for arm in NEW_ARMS
                  for cohort in ('development', 'common') for mode in MODES]
        tasks += [('formal', f'{arm}_seed{seed}', block) for arm in FORMAL_ARMS
                  if f'{arm}_seed{seed}' in NEW_NATIVE for block in BLOCKS if not block.startswith('open')]
    tasks += [('formal', policy, block) for policy in NEW_NATIVE for block in BLOCKS if block.startswith('open')]
    task_spec(task)
    return [t for t in tasks if task == 'combined'
            or (t[0] == 'dev' if task == 'development' else t[0] == 'formal')]


def disjoint(output, sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        require(output != source and source not in output.parents and output not in source.parents,
                'Output overlaps an immutable source')


def validate_metrics(values, rows):
    require(set(values) == {r['scene_id'] for r in rows} and len(rows) == len(values),
            'Missing/duplicate/unexpected scene')
    require(all(math.isfinite(v[k]) for v in values.values() for k in d.METRICS), 'Nonfinite metric')


def joint_mean_candidate(deltas):
    require(set(deltas) == {'development_NR', 'development_R', 'common_NR', 'common_R'},
            'All four PDM comparisons are required')
    values = list(deltas.values())
    require(all(math.isfinite(v) for v in values), 'Nonfinite comparison')
    return all(v >= 0 for v in values) and any(v > 0 for v in values)
