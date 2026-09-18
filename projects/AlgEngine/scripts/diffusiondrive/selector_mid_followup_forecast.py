"""Measured fixed task estimates, with 35 percent and four GPUh overhead allowance."""
from pathlib import Path
import shutil
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_snapshot_common as s
import selector_mid_followup_common as b
import selector_mid_followup_data as data
from selector_balance_forecast import durations,bytes_used


def forecast(run):
    run=Path(run)
    if (run/'budget_forecast.json').exists():
        old=d.read(run/'budget_forecast.json')
        b.require(old['contract']==d.artifact(run/'run_contract.json'),'Forecast drift')
        return old
    inputs=data.checked_inputs(run)
    src=Path(inputs['source_run']);snapshot=Path(inputs['snapshot_run'])
    tasks=[]
    if b.task_from_run(run)=='sensitivity':
        measured=durations(d.read(src/'ledger.json'))
        for cohort in ('development','common'):
            for mode in b.MODES:
                source_stage=f'eval_{cohort}_{mode}_MID_seed0'
                b.require(source_stage in measured,'Missing measured source stage')
                tasks.append(dict(kind='development',reference=f'eval_{cohort}_{mode}_CL_NLL_seed0',
                    measured_stage=source_stage,copies=6,seconds=measured[source_stage],
                    bytes=bytes_used(src/'collections'/source_stage)))
        train=[v for k,v in measured.items() if k.startswith('train_batch_')]
        b.require(bool(train),'Missing measured training time')
        train_bytes=max(bytes_used(p) for p in (src/'train').iterdir() if p.is_dir())
        tasks.append(dict(kind='training',reference='two batches of up to four jobs',copies=2,
            seconds=max(train),bytes=4*train_bytes))
    else:
        measured=durations(d.read(snapshot/'ledger.json'))
        for block in b.BLOCKS:
            name='open_'+s.digest(['--policy','P1_seed2','--block',block])[:12] if block.startswith('open') else block+'_P1_seed2'
            folder=snapshot/'openloop/P1_seed2'/block if block.startswith('open') else snapshot/'collections'/name
            tasks.append(dict(kind='formal',reference=name,copies=3,seconds=measured[name],bytes=bytes_used(folder)))
        for mode in b.MODES:
            name=f'bridge_{mode}_P1_seed2'
            tasks.append(dict(kind='bridge',reference=name,copies=3,seconds=measured[name],bytes=bytes_used(snapshot/'collections'/name)))
    seconds=sum(t['seconds']*t['copies'] for t in tasks)
    size=sum(t['bytes']*t['copies'] for t in tasks)
    cap=d.read(run/'run_contract.json')['gpu_hour_limit'];free=shutil.disk_usage(run).free
    result=dict(status='PASS',contract=d.artifact(run/'run_contract.json'),task=b.task_from_run(run),tasks=tasks,
        measured_component_wall_hours=seconds/3600,planning_gpu_hours=seconds*4/3600*1.35+4,
        planning_wall_hours=seconds/3600*1.35+1,gpu_hours_limit=cap,
        fits_budget=seconds*4/3600*1.35+4<cap,estimated_additional_bytes=size,
        storage_allowance_bytes=int(size*1.3),available_bytes_at_audit=free,fits_storage=free>size*1.3,
        note='Source measured timings; 35% plus 4 GPUh allowance, not a guarantee. Audit/export overhead is separate; retries consume budget.')
    c.locked_json(run/'budget_forecast.json',result)
    return result
