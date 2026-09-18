"""Measured source timings and filesystem usage; no budget-as-ETA shortcut."""
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_snapshot_common as s
import selector_balance_common as b
import selector_balance_data as data


def durations(ledger):
    running, result = {}, {}
    for event in ledger['events']:
        if event['status'] == 'RUNNING':
            running[event['stage']] = datetime.fromisoformat(event['time'])
        elif event['status'] == 'COMPLETE' and event['stage'] in running:
            elapsed = (datetime.fromisoformat(event['time'])-running.pop(event['stage'])).total_seconds()
            if elapsed > 0:
                result.setdefault(event['stage'], []).append(elapsed)
    return {key:float(np.median(value)) for key,value in result.items()}


def bytes_used(path):
    result = subprocess.run(['du', '-sb', str(path)], check=True, capture_output=True, text=True)
    return int(result.stdout.split()[0])


def forecast(run):
    run = Path(run)
    if (run/'budget_forecast.json').exists():
        old = d.read(run/'budget_forecast.json')
        b.require(old['contract'] == d.artifact(run/'run_contract.json'), 'Forecast contract drift')
        return old
    inputs = data.checked_inputs(run)
    ar, sr = Path(inputs['attribution_run']), Path(inputs['snapshot_run'])
    ats = durations(d.read(ar/'decision_ledger.json'))
    sts = durations(d.read(sr/'ledger.json'))
    tasks = []
    for cohort in ('development', 'common'):
        for mode in b.MODES:
            name = f'eval_{cohort}_{mode}_CL_NLL_seed0'
            b.require(name in ats, 'Missing measured development runtime: '+name)
            tasks.append(dict(kind='development', reference=name, copies=6, seconds=ats[name],
                              bytes=bytes_used(ar/'collections'/name)))
    for block in b.BLOCKS:
        stage = ('open_'+s.digest(['--policy', 'P1_seed2', '--block', block])[:12]
                 if block.startswith('open') else block+'_P1_seed2')
        b.require(stage in sts, 'Missing measured native runtime: '+stage)
        path = sr/'openloop/P1_seed2'/block if block.startswith('open') else sr/'collections'/stage
        tasks.append(dict(kind='formal', reference=stage, copies=5, seconds=sts[stage], bytes=bytes_used(path)))
    for mode in b.MODES:
        stage = f'bridge_{mode}_P1_seed2'
        b.require(stage in sts, 'Missing measured bridge runtime')
        tasks.append(dict(kind='bridge', reference=stage, copies=5, seconds=sts[stage],
                          bytes=bytes_used(sr/'collections'/stage)))
    training = [v for k,v in ats.items() if k.startswith('train_seed')]
    b.require(bool(training), 'Missing source training batch runtimes')
    tasks.append(dict(kind='training', reference='two batches of up to four jobs', copies=2,
                      seconds=float(np.median(training)), bytes=0))
    task = b.task_from_run(run)
    kinds = {'development', 'training'} if task == 'development' else {'formal', 'bridge'}
    if task != 'combined':
        tasks = [t for t in tasks if t['kind'] in kinds]
    seconds = sum(t['copies']*t['seconds'] for t in tasks)
    byte_count = sum(t['copies']*t['bytes'] for t in tasks)
    free = shutil.disk_usage(run).free
    cap = d.read(run/'run_contract.json')['gpu_hour_limit']
    result = dict(status='PASS', task=task, contract=d.artifact(run/'run_contract.json'), tasks=tasks,
        measured_component_gpu_hours=seconds*4/3600, measured_component_wall_hours=seconds/3600,
        planning_gpu_hours=seconds*4/3600*1.35+4, planning_wall_hours=(seconds*4/3600*1.35+4)/4,
        gpu_hours_limit=cap, fits_budget=(seconds*4/3600*1.35+4) < cap,
        estimated_additional_bytes=byte_count, storage_allowance_bytes=int(byte_count*1.3),
        available_bytes_at_audit=free, fits_storage=free > byte_count*1.3,
        note='Historical condition timings plus 35% and 4 GPUh allowance. Not a guarantee; retries cost extra. Native exports already exist at audit time.')
    c.locked_json(run/'budget_forecast.json', result)
    return result
