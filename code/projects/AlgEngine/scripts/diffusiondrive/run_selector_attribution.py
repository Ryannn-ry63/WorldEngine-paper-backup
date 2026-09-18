"""Four-H100 attribution: audit -> preflight -> labels -> train -> eval -> report."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import time
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_feedback_common as f
import selector_attribution_common as a
import selector_attribution_data as data
import selector_attribution_collections as collections
from run_selector_cfpi import ROOT, SCRIPT, CANONICAL, environment, implementation_inventory
from run_selector_cfpi_deployment import Runner as BaseRunner, check_budget
from run_selector_feedback import Runner as FeedbackRunner
from run_selector_decision import history_samples, estimate_collection
from report_selector_cfpi_deployment import checked_collection

PRIOR = CANONICAL/'experiments/worktrees/WorldEngine-selector-decision-feedback-v1/experiments/diffusiondrive/selector_decision_feedback_v1/runs/decision_feedback_20260912_192h_v1'


class Runner(BaseRunner):
    charge_idle = FeedbackRunner.charge_idle

    def __init__(self, args):
        self.args = args
        self.run = ROOT/'experiments/diffusiondrive/selector_p1_attribution_v1/runs'/args.run_id
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run/'runner.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.phase = 'attribution'
        self.phase_limits = dict(attribution=args.gpu_hours)
        self.account_idle = args.stage not in ('audit','report')
        self.idle_since = time.monotonic()
        self.env = environment(args.source_worldengine_root)
        self.devices = f.gpu_devices(self.env,args.gpus)
        self.env.update(CUDA_VISIBLE_DEVICES=','.join(self.devices),
                        WORLDENGINE_FEEDBACK_CUDA_DEVICE_MAP=','.join(self.devices),
                        DIFFUSIONDRIVE_FEEDBACK_RESEARCH_METHOD=a.METHOD)
        self.alg, self.sim = Path(self.env['ALGENGINE_ROOT']), Path(self.env['SIMENGINE_ROOT'])
        self.config = self.alg/'configs/diffusiondrive/e2e_diffusiondrive_selector_attribution.py'
        self.collection_timeout = 43200
        self.checkpoint_manifest = args.source_worldengine_root/'experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/models/rare_tuned/seed0/checkpoint_manifest.json'
        self.inventory = implementation_inventory()
        for path in (ROOT/'research/p1_attribution').rglob('*'):
            if path.is_file() and path.suffix in ('.py','.md','.json','.sh'):
                self.inventory[str(path.relative_to(ROOT))] = c.sha256_file(path)
        launcher = ROOT/'run_diffusiondrive_selector_p1_attribution_4h100.sh'
        self.inventory[launcher.name] = c.sha256_file(launcher)
        self.code_sha = hashlib.sha256(json.dumps(self.inventory,sort_keys=True).encode()).hexdigest()
        snapshot = {p:d.artifact(args.prior_run/p) for p in ('run_contract.json','pilot_report.json','decision_ledger.json')}
        contract = dict(method=a.METHOD, base_snapshot_commit='077fbca', code_sha=self.code_sha,
                        implementation=self.inventory, source_snapshot=snapshot, prior_run=str(args.prior_run),
                        source_worldengine_root=str(args.source_worldengine_root), gpu_count=args.gpus,
                        gpu_hour_limit=args.gpu_hours, phase_limits=self.phase_limits, training=a.TRAIN,
                        arms=list(a.ARMS), new_arms=list(a.NEW_ARMS), seeds=list(a.SEEDS),
                        exposure=a.EXPOSURE, local_definition=a.LOCAL_DEFINITION, contrasts=a.COMPARISONS,
                        expected_conditions=80, reused_conditions=32, new_conditions=48,
                        inference_uses_reward_or_q=False, full_rare288_authorized=False,
                        endpoint='attribution_report_REVIEW_REQUIRED',
                        accounting='All four reserved GPUs during allocation stages, including CPU tasks; standalone audit/report uncharged.')
        c.locked_json(self.run/'run_contract.json', contract)
        self.ledger_path = self.run/'decision_ledger.json'
        self.ledger = d.read(self.ledger_path) if self.ledger_path.exists() else dict(
            run_id=args.run_id, gpu_hours_used=0., phase_gpu_hours=dict(attribution=0.),
            phase_limits=self.phase_limits, events=[], active=None)
        if (self.ledger['phase_limits'] != self.phase_limits or
                set(self.ledger['phase_gpu_hours']) != {'attribution'} or
                not math.isfinite(self.ledger['gpu_hours_used']) or self.ledger['gpu_hours_used'] < 0 or
                self.ledger['phase_gpu_hours']['attribution'] != self.ledger['gpu_hours_used']):
            raise RuntimeError('Invalid budget ledger; no reset permitted')
        self.children = []
        self.recover_accounting()
        signal.signal(signal.SIGINT,self.interrupt)
        signal.signal(signal.SIGTERM,self.interrupt)
        self.samples = history_samples(args.prior_run)
        self.idle_since = time.monotonic()

    def execute(self, jobs, stage, gpus=0, timeout=21600):
        self.charge_idle()
        try:
            return BaseRunner.execute(self,jobs,stage,gpus,timeout)
        finally:
            self.idle_since = time.monotonic()

    def task(self, action):
        self.python(SCRIPT/'selector_attribution_tasks.py',[action,'--run-root',self.run],
                    stage=action,timeout=14400)

    def collection(self, entry):
        condition = d.verified_read(entry)
        folder = Path(entry['path']).parent
        if folder.parent != self.run/'collections' or condition['research_method'] != a.METHOD:
            raise RuntimeError('Cannot execute/rewrite a historical or foreign condition')
        if (folder/'collection_audit.json').exists():
            if checked_collection(entry) is None:
                raise RuntimeError('Existing collection audit invalid')
            self.record(folder.name,'REUSED_AUDITED')
            return
        count = len(d.verified_read(condition['routing'])['routes'])
        estimate = estimate_collection(self.samples,count,condition['react_type'],self.args.gpus)
        check_budget(self.ledger,self.phase,self.args.gpu_hours,estimate)
        self.record(folder.name,'FORECAST',dict(gpu_hours=estimate,wall_hours=estimate/self.args.gpus,
                    scenes=count,workers=condition['worker_count'],note='Historical estimate, not a guarantee'))
        started = time.monotonic()
        BaseRunner.collection(self,entry)
        self.ledger.setdefault('runtime_samples',[]).append(dict(scenes=count,seconds=time.monotonic()-started,
            gpu_count=self.args.gpus,worker_count=condition['worker_count']))
        c.atomic_json(self.ledger_path,self.ledger)

    def audit(self):
        inputs = data.freeze(self.run,self.args.prior_run)
        data.cpu_preflight(self.run)
        bundle = collections.local_bundle(self.run)
        local = sum(estimate_collection(self.samples,len(d.verified_read(d.verified_read(e)['routing'])['routes']),
                    d.verified_read(e)['react_type'],4) for e in bundle['collections'])
        evaluation = sum(estimate_collection(self.samples,len(inputs['cohorts'][cohort]['rows']),mode,4)
                         for cohort,mode,policy in collections.evaluation_conditions()
                         if f'eval_{cohort}_{mode}_{policy}' not in inputs['reused_collections'])
        forecast = dict(status='PASS',gpu_hours=dict(local_observation=local,training_allowance=8.,evaluation=evaluation),
                        total_estimate=local+8.+evaluation,wall_hours_estimate=(local+8.+evaluation)/4,
                        gpu_hour_limit=self.args.gpu_hours,source_samples=len(self.samples),
                        fits_estimate=local+8.+evaluation < self.args.gpu_hours,
                        note='Local scoring overhead is not measured; budget cap is not ETA. No auto-expansion.')
        c.locked_json(self.run/'budget_forecast.json',forecast)
        result = dict(status='PASS',code_sha=self.code_sha,inputs=d.artifact(self.run/'attribution_inputs.json'),
                      cpu_preflight=d.artifact(self.run/'cached_preflight_cpu.json'),
                      forecast=d.artifact(self.run/'budget_forecast.json'),
                      local_bundle=d.artifact(self.run/'labels/collections.json'),
                      formal_gpu_experiments_started=False,actual_training_steps=0)
        c.locked_json(self.run/'audit_report.json',result)
        self.record('audit','COMPLETE',dict(**result,forecast_summary=forecast))

    def labels(self):
        bundle = d.read(self.run/'labels/collections.json')
        for entry in bundle['collections']:
            self.collection(entry)
        self.task('assemble_labels')
        self.record('labels','COMPLETE','All 256 states and all executed replay decisions passed')

    def train_controls(self):
        data.load_rows(self.run)
        # Four concurrent one-GPU trainers, same seed per batch; all three seeds required.
        for seed in a.SEEDS:
            jobs = []
            for arm, device in zip(a.NEW_ARMS,self.devices):
                policy = f'{arm}_seed{seed}'
                if (self.run/'train'/policy/'step_500_report.json').exists():
                    data.model_entry(self.run,policy)
                    self.record(policy,'REUSED_AUDITED')
                    continue
                command = [self.env['ALGENGINE_PYTHON'],SCRIPT/'train_selector_attribution.py',
                           '--run-root',self.run,'--arm',arm,'--seed',str(seed)]
                jobs.append((command,ROOT,dict(self.env,CUDA_VISIBLE_DEVICES=device),self.run/'logs'/f'train_{policy}.log'))
            if jobs:
                check_budget(self.ledger,self.phase,self.args.gpu_hours,.5)
                self.execute(jobs,f'train_seed{seed}',gpus=4,timeout=14400)

    def evaluate(self):
        self.task('evaluation_bundle')
        historical = data.checked_inputs(self.run)['reused_collections']
        for entry in d.read(self.run/'evaluation_collections.json')['collections']:
            if entry in historical.values():
                if checked_collection(entry) is None:
                    raise RuntimeError('Frozen historical result no longer valid')
                continue
            self.collection(entry)

    def dispatch(self):
        stage = self.args.stage
        if stage == 'audit':
            return self.audit()
        if stage == 'report':
            from report_selector_attribution import report
            result = report(self.run)
            print(json.dumps({k:result[k] for k in ('status','decision')},ensure_ascii=False),flush=True)
            return
        audit = d.read(self.run/'audit_report.json')
        if audit['status'] != 'PASS' or audit['code_sha'] != self.code_sha:
            raise RuntimeError('Run CPU audit with the frozen implementation before using GPUs')
        for key in ('inputs','cpu_preflight','forecast','local_bundle'):
            d.verify(audit[key])
        if stage in ('train','eval'):
            data.load_rows(self.run)  # fail before allocating simulation workers
        self.preflight()
        if stage == 'preflight':
            self.record(stage,'COMPLETE','No simulation or training launched')
            return
        if stage in ('labels','all'):
            self.labels()
        if stage in ('train','all'):
            self.train_controls()
        if stage in ('eval','all'):
            self.evaluate()
            self.task('report')
        self.record(stage,'COMPLETE','Stage boundary; REVIEW_REQUIRED, no new method/full benchmark auto-expansion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('audit','preflight','labels','train','eval','report','all'))
    parser.add_argument('run_id')
    parser.add_argument('--prior-run',type=Path,default=PRIOR)
    parser.add_argument('--source-worldengine-root',type=Path,default=CANONICAL)
    parser.add_argument('--gpus',type=int,choices=(4,),default=4)
    parser.add_argument('--gpu-hours',type=float,default=192.)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}',args.run_id):
        parser.error('Safe immutable run ID required')
    if not math.isfinite(args.gpu_hours) or args.gpu_hours <= 0:
        parser.error('Finite positive GPU-hour cap required')
    args.prior_run, args.source_worldengine_root = args.prior_run.resolve(),args.source_worldengine_root.resolve()
    runner = Runner(args)
    try:
        runner.dispatch()
    except BaseException as error:
        runner.stop_children()
        runner.record(args.stage,'STOPPED',str(error))
        raise
    finally:
        runner.charge_idle(enforce=False)


if __name__ == '__main__':
    main()

