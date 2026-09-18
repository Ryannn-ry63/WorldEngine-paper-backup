"""Fixed MID release and symmetric sensitivity; no coefficient search API."""
import math
from pathlib import Path
import selector_attribution_common as a
from selector_balance_common import require, disjoint, validate_metrics, EXPOSURE
import selector_cfpi_deployment_common as d

METHOD = 'selector_mid_followup_v1'
SEEDS = (0, 1, 2)
MODES = ('NR', 'R')
ARMS = ('CL_NLL', 'LOW04', 'MID', 'HIGH06', 'P1', 'MID_SCALE_MATCH')
NEW_ARMS = ('LOW04', 'HIGH06')
COEFFICIENTS = dict(CL_NLL=0., LOW04=.4, MID=.5, HIGH06=.6, P1=1., MID_SCALE_MATCH=.5)
FORMAL_ARMS = ('MID',)
NEW_NATIVE = tuple(f'MID_seed{s}' for s in SEEDS)
FORMAL_POLICIES = ('scalar_v3', *NEW_NATIVE)
BLOCKS = ('open_navtest', 'open_rare', 'full_NR', 'full_R', 'common_NR', 'common_R')
FORMAL_NOISE = 'formal_navtest_seed0'
TRAIN = dict(a.TRAIN, pair_coefficients={k:COEFFICIENTS[k] for k in NEW_ARMS},
             feedback_scale=1., final_checkpoint_only=True, primary_coefficient=.5)
TASKS = ('sensitivity', 'mid-formal')
STAGES = ('audit', 'preflight', 'train', 'dev-eval', 'native-bridge', 'formal-eval', 'report', 'all')
DEFAULT_BUDGET = {'sensitivity':48., 'mid-formal':72.}


def task_spec(task):
    require(task in TASKS, 'Unknown fixed task')
    dev = task == 'sensitivity'
    return dict(task=task, expected_training_jobs=6 if dev else 0,
        expected_dev_conditions=72 if dev else 0, reused_dev_conditions=48 if dev else 0,
        new_dev_conditions=24 if dev else 0, expected_bridge_conditions=0 if dev else 6,
        expected_formal_conditions=0 if dev else 24, reused_formal_conditions=0 if dev else 6,
        new_formal_conditions=0 if dev else 18)


def task_from_run(run):
    task = d.read(Path(run)/'run_contract.json')['task']
    task_spec(task)
    return task


def validate_stage(task, stage):
    task_spec(task)
    require(stage in STAGES, 'Unknown stage')
    require(not (task == 'mid-formal' and stage in ('train', 'dev-eval'))
        and not (task == 'sensitivity' and stage in ('native-bridge', 'formal-eval')),
        'Stage is outside the fixed task scope')


def specification(arm, alpha=1.):
    require(arm in NEW_ARMS and alpha == 1., 'Only fixed 0.4/0.6 training without scale changes')
    return dict(pair_extra=COEFFICIENTS[arm], feedback_scale=1.)


def objective(logits, rows, arm, alpha=1.):
    spec = specification(arm, alpha)
    pair,nll = a.feedback_loss(logits, rows, 'P1')
    return nll+spec['pair_extra']*pair, pair,nll


def dev_conditions():
    return [(cohort,mode,f'{arm}_seed{seed}') for cohort in ('development','common')
            for mode in MODES for arm in ARMS for seed in SEEDS]


def formal_conditions():
    return [(policy,block) for policy in FORMAL_POLICIES for block in BLOCKS]


def queue(task):
    task_spec(task)
    if task == 'sensitivity':
        return [('dev',cohort,mode,f'{arm}_seed{seed}') for seed in SEEDS for arm in NEW_ARMS
                for cohort in ('development','common') for mode in MODES]
    return [('formal',policy,block) for policy in NEW_NATIVE for block in BLOCKS if not block.startswith('open')] + [
            ('formal',policy,block) for policy in NEW_NATIVE for block in BLOCKS if block.startswith('open')]
