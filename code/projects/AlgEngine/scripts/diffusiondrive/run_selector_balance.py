"""One four-GPU queue: fixed weight controls and all-seed native evaluation."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import signal
import time
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_balance_common as b
import selector_balance_data as data
import selector_balance_collections as dev
import selector_balance_native as native
import selector_snapshot_common as s
import selector_snapshot_audit as native_audit
from run_selector_cfpi import ROOT, SCRIPT, CANONICAL, environment, implementation_inventory
from run_selector_cfpi_deployment import Runner as BaseRunner, check_budget
from run_selector_snapshot import Runner as NativeRunner
from run_selector_feedback import Runner as FeedbackRunner

WORKTREES = CANONICAL/'experiments/worktrees'
ATTRIBUTION = WORKTREES/'WorldEngine-selector-p1-attribution-v1/experiments/diffusiondrive/selector_p1_attribution_v1/runs/p1_attribution_20260913_v1'
SNAPSHOT = WORKTREES/'WorldEngine-selector-p1-seed2-snapshot-v1/experiments/diffusiondrive/selector_p1_snapshot_v1/runs/p1_seed2_full_20260913_v1'


def inventory():
    result = implementation_inventory()
    for folder in ('projects/AlgEngine/scripts', 'projects/AlgEngine/mmdet3d_plugin', 'research/p1_balance'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.sh', '.json', '.md', '.yaml'):
                result[str(path.relative_to(ROOT))] = c.sha256_file(path)
    name = 'run_diffusiondrive_selector_p1_balance_4h100.sh'
    result[name] = c.sha256_file(ROOT/name)
    name = 'run_diffusiondrive_selector_p1_task_4h100.sh'
    result[name] = c.sha256_file(ROOT/name)
    return result


class Runner(BaseRunner):
    charge_idle = FeedbackRunner.charge_idle

    def __init__(self, args):
        self.args = args
        b.validate_stage(args.task, args.stage)
        self.run = ROOT/'experiments/diffusiondrive/selector_p1_balance_v1/runs'/args.run_id
        b.disjoint(self.run, (args.attribution_run, args.snapshot_run))
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run/'runner.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Prevent different balance IDs on the same host from claiming the same four-card allocation.
        self.allocation_lock = None
        if args.stage not in ('audit', 'report'):
            import socket
            root = CANONICAL/'experiments/diffusiondrive/selector_p1_balance_v1/locks'
            root.mkdir(parents=True, exist_ok=True)
            self.allocation_lock = (root/(socket.gethostname()+'.lock')).open('a+')
            fcntl.flock(self.allocation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.phase = 'balance'
        self.phase_limits = dict(balance=args.gpu_hours)
        self.account_idle = args.stage not in ('audit', 'report')
        self.idle_since = time.monotonic()
        self.env = environment(args.source_worldengine_root)
        self.devices = f.gpu_devices(self.env, 4)
        self.alg, self.sim = Path(self.env['ALGENGINE_ROOT']), Path(self.env['SIMENGINE_ROOT'])
        self.dev_config = self.alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_balance.py'
        self.native_config = self.alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_balance_native.py'
        self.config = self.dev_config
        self.collection_timeout = 43200
        self.checkpoint_manifest = args.source_worldengine_root/'experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json'
        scalar = d.read(self.checkpoint_manifest)
        exp = args.source_worldengine_root.parent/'exp'
        self.env.update(CUDA_VISIBLE_DEVICES=','.join(self.devices),
            WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP=','.join(self.devices),
            DIFFUSIONDRIVE_FEEDBACK_RESEARCH_METHOD=b.METHOD,
            NAVSIM_EXP_ROOT=str(exp), NAVSIM_METRIC_CACHE_PATH=str(exp/'metric_cache_navtest'),
            NAVSIM_METRIC_CACHE_PATH_EVAL=str(exp/'metric_cache_navtest'),
            NAVSIM_METRIC_CACHE_PATH_TRAIN=str(exp/'metric_cache_trainval'),
            NAVSIM_DEVKIT_ROOT=str(args.source_worldengine_root.parent/'DiffusionDrive'),
            NAVSIM_RESCORE_SHARDS='4', NAVSIM_RESCORE_SPLIT='navtest',
            NAVSIM_OFFICIAL_RESCORE='always', PYTHON_BIN=self.env['ALGENGINE_PYTHON'],
            SELECTOR_SNAPSHOT_OPEN='0', DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE=b.FORMAL_NOISE,
            DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT=scalar['baseline'],
            DIFFUSIONDRIVE_SELECTOR_INIT_SHA256=scalar['baseline_sha256'])
        self.inventory = inventory()
        self.code_sha = s.digest(self.inventory)
        self.env['DIFFUSIONDRIVE_ROLLOUT_CODE_SHA'] = self.code_sha
        contract = dict(method=b.METHOD, code_sha=self.code_sha, implementation=self.inventory,
            attribution_run=str(args.attribution_run), snapshot_run=str(args.snapshot_run),
            source_worldengine_root=str(args.source_worldengine_root),
            source_reports=[d.artifact(args.attribution_run/'attribution_report.json'),
                            d.artifact(args.snapshot_run/'summary.json')],
            scorer=d.artifact(Path(self.env['NAVSIM_DEVKIT_ROOT'])/'navsim/planning/script/run_pdm_score_from_submission.py'),
            gpu_count=4, gpu_hour_limit=args.gpu_hours, phase_limits=self.phase_limits,
            training=b.TRAIN, seeds=list(b.SEEDS), arms=list(b.ARMS),
            queue=[list(t) for t in b.queue(args.task)], dev_noise=c.NOISE_NAMESPACE, formal_noise=b.FORMAL_NOISE,
            exposure=b.EXPOSURE, **b.task_spec(args.task),
            native_environment={k:self.env[k] for k in ('NAVSIM_METRIC_CACHE_PATH', 'NAVSIM_DEVKIT_ROOT',
                'DIFFUSIONDRIVE_SELECTOR_INIT_CHECKPOINT', 'DIFFUSIONDRIVE_SELECTOR_INIT_SHA256')},
            performance_gate_applied=False, automatic_next_method=False, automatic_mid_full_evaluation=False,
            accounting='All four reserved GPUs, including helpers; standalone audit/report uncharged')
        c.locked_json(self.run/'run_contract.json', contract)
        self.ledger_path = self.run/'ledger.json'
        self.ledger = d.read(self.ledger_path) if self.ledger_path.exists() else dict(run_id=args.run_id,
            gpu_hours_used=0., phase_gpu_hours=dict(balance=0.), phase_limits=self.phase_limits,
            active=None, events=[])
        b.require(self.ledger['phase_limits'] == self.phase_limits
            and set(self.ledger['phase_gpu_hours']) == {'balance'}
            and math.isfinite(self.ledger['gpu_hours_used']) and self.ledger['gpu_hours_used'] >= 0
            and abs(self.ledger['gpu_hours_used']-self.ledger['phase_gpu_hours']['balance']) < 1e-6,
            'Budget ledger drift; never reset it')
        self.children = []
        self.recover_accounting()
        signal.signal(signal.SIGINT, self.interrupt)
        signal.signal(signal.SIGTERM, self.interrupt)
        self.idle_since = time.monotonic()

    def execute(self, jobs, stage, gpus=0, timeout=21600):
        self.charge_idle()
        try:
            return BaseRunner.execute(self, jobs, stage, gpus, timeout)
        finally:
            self.idle_since = time.monotonic()

    def helper(self, task, *arguments, env=None):
        self.python(SCRIPT/'selector_balance_tasks.py', [task, '--run', self.run, *arguments],
                    stage=task+'_'+s.digest(list(map(str,arguments)))[:12], env=env, timeout=43200)

    def audit(self):
        self.record('audit', 'RUNNING', 'CPU only: no research optimization or rollouts')
        data.freeze(self.run, self.args.attribution_run, self.args.snapshot_run, ROOT)
        if self.args.task != 'formal':
            data.scale_audit(self.run)
        native.parity(self.run, 'cpu')
        b.require(Path(self.env['NAVSIM_METRIC_CACHE_PATH']).is_dir(), 'Missing official metric cache')
        code = {}
        for relative, expected in self.inventory.items():
            entry = s.preserved_copy(ROOT/relative, self.run/'archive/code'/relative)
            b.require(entry['sha256'] == expected, 'Code changed during archive')
            code[relative] = entry
        c.locked_json(self.run/'archive/code_manifest.json', code)
        for policy in (b.NEW_NATIVE if self.args.task != 'development' else ()):
            for cohort in ('bridge', 'full', 'common'):
                for mode in b.MODES:
                    native.make_collection(self.run, policy, cohort, mode)
        from selector_balance_forecast import forecast
        prediction = forecast(self.run)
        c.locked_json(self.run/'archive/reproduce.json', dict(worktree=str(ROOT),
            launcher='run_diffusiondrive_selector_p1_balance_4h100.sh', run_id=self.args.run_id,
            gpus=4, gpu_hours=self.args.gpu_hours, task=self.args.task, attribution_run=str(self.args.attribution_run),
            snapshot_run=str(self.args.snapshot_run), stages=list(b.STAGES),
            contract=d.artifact(self.run/'run_contract.json')))
        result = dict(status='PASS', code_sha=self.code_sha,
            inputs=d.artifact(self.run/'balance_inputs.json'), native_inputs=d.artifact(self.run/'inputs.json'),
            parity=d.artifact(self.run/'cached_parity_cpu.json'),
            code_archive=d.artifact(self.run/'archive/code_manifest.json'),
            forecast=d.artifact(self.run/'budget_forecast.json'), formal_gpu_experiments_started=False,
            optimizer_steps=0, new_rollouts=0, forecast_summary=prediction)
        if self.args.task != 'formal':
            result['scale'] = d.artifact(self.run/'scale_audit.json')
        c.locked_json(self.run/'audit_report.json', result)
        self.record('audit', 'COMPLETE', result)

    def capacity(self, kind, reference):
        import shutil
        forecast = d.read(self.run/'budget_forecast.json')
        tasks = [t for t in forecast['tasks'] if t['kind'] == kind and t['reference'] == reference]
        b.require(len(tasks) == 1, 'Missing task resource estimate')
        task = tasks[0]
        check_budget(self.ledger, self.phase, self.args.gpu_hours, task['seconds']*4/3600*1.35)
        b.require(shutil.disk_usage(self.run).free > task['bytes']*1.3+1024**3,
                  'Insufficient free storage for the next complete condition')

    def training(self):
        data.load_rows(self.run)
        tasks = [(arm, seed) for seed in b.SEEDS for arm in b.NEW_ARMS]
        for start in range(0, len(tasks), 4):
            jobs = []
            for (arm, seed), device in zip(tasks[start:start+4], self.devices):
                policy = f'{arm}_seed{seed}'
                if (self.run/'train'/policy/'step_500_report.json').exists():
                    data.model_entry(self.run, policy)
                    self.record(policy, 'REUSED_AUDITED')
                    continue
                command = [self.env['ALGENGINE_PYTHON'], SCRIPT/'train_selector_balance.py',
                           '--run-root', self.run, '--arm', arm, '--seed', str(seed)]
                jobs.append((command, ROOT, dict(self.env, CUDA_VISIBLE_DEVICES=device),
                             self.run/'logs'/f'train_{policy}.log'))
            if jobs:
                self.capacity('training', 'two batches of up to four jobs')
                self.execute(jobs, f'train_batch_{start//4}', gpus=4, timeout=14400)

    def dev_eval(self, cohort, mode, policy):
        if dev.checked(self.run, cohort, mode, policy) is not None:
            self.record(f'eval_{cohort}_{mode}_{policy}', 'REUSED_AUDITED')
            return
        self.capacity('development', f'eval_{cohort}_{mode}_CL_NLL_seed0')
        entry = dev.make(self.run, cohort, mode, policy)
        self.config = self.dev_config
        # The original routed protocol owns its development noise namespace.
        old = self.env['DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE']
        self.env['DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE'] = c.NOISE_NAMESPACE
        try:
            BaseRunner.collection(self, entry)
        finally:
            self.env['DIFFUSIONDRIVE_GRPO_EVAL_NOISE_NAMESPACE'] = old
        b.require(dev.checked(self.run, cohort, mode, policy) is not None, 'Development audit missing')

    def native_collection(self, policy, cohort, mode):
        self.config = self.native_config
        path = native.make_collection(self.run, policy, cohort, mode)
        if not (path.parent/'audit.json').exists():
            self.capacity('bridge' if cohort == 'bridge' else 'formal', f'{cohort}_{mode}_P1_seed2')
        NativeRunner.collection(self, path)

    def bridge(self):
        for policy in b.NEW_NATIVE:
            for mode in b.MODES:
                self.native_collection(policy, 'bridge', mode)
        self.helper('bridge')

    def formal_eval(self, policy, block):
        native.require_bridge(self.run)
        if native.checked_formal(self.run, policy, block) is not None:
            self.record(policy+'__'+block, 'REUSED_AUDITED')
            return
        if block.startswith('open'):
            self.capacity('formal', 'open_'+s.digest(['--policy', 'P1_seed2', '--block', block])[:12])
            self.helper('open', '--policy', policy, '--block', block)
        else:
            cohort, mode = block.rsplit('_', 1)
            self.native_collection(policy, cohort, mode)
        b.require(native.checked_formal(self.run, policy, block) is not None, 'Formal audit missing')

    def dispatch(self):
        stage = self.args.stage
        if stage == 'audit':
            return self.audit()
        if stage == 'report':
            from report_selector_balance import report
            result = report(self.run)
            print(json.dumps({k:result[k] for k in ('status', 'decision', 'missing') if k in result}), flush=True)
            return
        result = d.read(self.run/'audit_report.json')
        b.require(result['status'] == 'PASS' and result['code_sha'] == self.code_sha, 'Current CPU audit required')
        keys = ['inputs', 'native_inputs', 'parity', 'code_archive', 'forecast']
        if self.args.task != 'formal':
            keys.append('scale')
        for key in keys:
            d.verify(result[key])
        data.checked_inputs(self.run)
        b.require(d.read(self.run/'budget_forecast.json')['fits_budget'], 'Forecast exceeds frozen budget; no automatic expansion')
        d.verify(d.read(self.run/'run_contract.json')['scorer'])
        self.preflight()
        self.helper('parity', '--device', 'cuda')
        self.helper('config', env=dict(self.env, SELECTOR_SNAPSHOT_NOISE=b.FORMAL_NOISE,
            SELECTOR_SNAPSHOT_MODE='NR', SELECTOR_SNAPSHOT_CONDITION='preflight'))
        if stage == 'preflight':
            self.record(stage, 'COMPLETE', 'No formal rollout/training')
            return
        if stage in ('train', 'all') and self.args.task != 'formal':
            self.training()
        if stage in ('native-bridge', 'all') and self.args.task != 'development':
            self.bridge()
        if stage == 'all':
            for task in b.queue(self.args.task):
                if task[0] == 'dev':
                    self.dev_eval(*task[1:])
                else:
                    self.formal_eval(*task[1:])
        elif stage == 'dev-eval':
            for cohort, mode, policy in b.dev_conditions():
                self.dev_eval(cohort, mode, policy)
        elif stage == 'formal-eval':
            for policy, block in b.formal_conditions():
                self.formal_eval(policy, block)
        if stage in ('all', 'dev-eval', 'formal-eval'):
            self.helper('report')
        if stage == 'all':
            b.require(d.read(self.run/'progress.json')['status'] == 'PASS', 'Task report is incomplete')
        self.record(stage, 'COMPLETE', 'Stage boundary; REVIEW_REQUIRED, no automatic expansion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=b.STAGES)
    parser.add_argument('run_id')
    parser.add_argument('--task', choices=b.TASKS, default='combined')
    parser.add_argument('--attribution-run', type=Path, default=ATTRIBUTION)
    parser.add_argument('--snapshot-run', type=Path, default=SNAPSHOT)
    parser.add_argument('--source-worldengine-root', type=Path, default=CANONICAL)
    parser.add_argument('--gpus', type=int, choices=(4,), default=4)
    parser.add_argument('--gpu-hours', type=float, default=192.)
    args = parser.parse_args()
    b.validate_stage(args.task, args.stage)
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}', args.run_id):
        parser.error('Safe immutable run ID required')
    if not math.isfinite(args.gpu_hours) or args.gpu_hours <= 0:
        parser.error('Positive finite GPU-hour limit required')
    for key in ('attribution_run', 'snapshot_run', 'source_worldengine_root'):
        setattr(args, key, getattr(args, key).resolve())
    runner = Runner(args)
    try:
        runner.dispatch()
    except BaseException as error:
        runner.stop_children()
        runner.record(args.stage, 'STOPPED', str(error))
        raise
    finally:
        runner.charge_idle(enforce=False)


if __name__ == '__main__':
    main()
