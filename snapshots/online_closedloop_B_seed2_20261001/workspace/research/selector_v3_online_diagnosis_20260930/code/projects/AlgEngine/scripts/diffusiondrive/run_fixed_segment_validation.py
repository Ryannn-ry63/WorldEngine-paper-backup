#!/usr/bin/env python3
"""Freeze source and run independent, frozen-model GPU lanes. No training/DDP."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def write_json(path,data):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('settings','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','diagnostic','frozen'],required=True)
    p.add_argument('--devices',default='0,1,2,3,4,5,6,7')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--reaction',choices=['R','NR'],default='R')
    p.add_argument('--timeout',type=float,default=43200.)
    p.add_argument('--verify-serial',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    devices=args.devices.split(',')
    if len(set(devices))!=len(devices) or not all(x.isdigit() for x in devices):p.error('Use unique GPU numbers')
    if not 0<=args.workers<=20 or args.timeout<=0:p.error('Invalid worker count or timeout')
    count={'smoke':1,'diagnostic':32,'frozen':16}[args.mode]
    rows=[r for r in json.loads(args.manifest.read_text())['scenes'] if r['split']=='train']
    if len(rows)<count:raise ValueError('Insufficient distinct train scenes')
    rows=[dict(r,episode_index=i) for i,r in enumerate(rows[:count])]
    if len({r['origin_log'] for r in rows})!=count:raise ValueError('Expected independent source logs')
    devices=devices[:min(count,len(devices))]
    code=Path(__file__).resolve().parents[4];output=args.output.resolve()
    if output==code or code in output.parents:raise ValueError('Outputs must be outside active code')
    if output.exists():raise FileExistsError('Choose a fresh output directory')
    plan=dict(mode=args.mode,episodes=count,expected_reward_roots=count*2,devices=devices,
              workers_per_lane=args.workers,source=str(code),output=str(output),training=False)
    print(json.dumps(plan,indent=2),flush=True)
    if args.dry_run:return
    output.mkdir(parents=True)
    freeze=output/'frozen_code'
    shutil.copytree(code,freeze,symlinks=True,ignore=shutil.ignore_patterns('.git','__pycache__','core.*','tmp','.pytest_cache'))
    tools=freeze/'projects/AlgEngine/scripts/diffusiondrive'
    spec=importlib.util.spec_from_file_location('frozen_selector_runtime',tools/'selector_runtime.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    cfg=json.loads(args.settings.read_text());cfg['anchors']=str(freeze/'kmeans_navsim_traj_20.npy')
    settings=output/'settings.json';write_json(settings,cfg)
    write_json(output/'manifest.json',dict(scenes=rows,source_manifest=str(args.manifest.resolve())))
    children=[];logs=[];commands=[];started=time.monotonic();report=dict(status='RUNNING',plan=plan)
    def interrupted(signum,frame):raise InterruptedError('Signal '+str(signum))
    previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for lane,gpu in enumerate(devices):
            assignment=output/('lane_%d_assignments.json'%lane);write_json(assignment,rows[lane::len(devices)])
            _,env=runtime.environment(settings,gpu)
            env.update(OPENBLAS_CORETYPE='Prescott',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            command=[cfg['algengine_python'],'-u','-m','innovation3.fixed_visual_probe',
                '--settings',str(settings),'--assignments',str(assignment),'--output',str(output/('lane_%d'%lane)),
                '--mode',args.mode,'--workers',str(args.workers),'--reaction',args.reaction]
            if args.verify_serial:command.append('--verify-serial')
            log=(output/('lane_%d.log'%lane)).open('x');logs.append(log)
            children.append(subprocess.Popen(command,cwd=freeze/'projects/AlgEngine',env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            commands.append(dict(gpu=gpu,argv=command))
        write_json(output/'commands.json',commands)
        while True:
            codes=[p.poll() for p in children]
            if any(c is not None and c!=0 for c in codes):raise RuntimeError('Lane failure: '+repr(codes))
            if all(c==0 for c in codes):break
            if time.monotonic()-started>args.timeout:raise TimeoutError('Validation timed out')
            time.sleep(5)
        episodes=[];groups=[];lane_reports=[]
        for lane in range(len(devices)):
            lane_report=json.loads((output/('lane_%d'%lane)/'report.json').read_text())
            if lane_report['status']!='PASS_FROZEN_LANE' or lane_report['model_loads']!=1:raise RuntimeError('Invalid lane acceptance')
            lane_reports.append(lane_report)
            for episode in lane_report['episodes']:
                detail=json.loads(Path(episode['report']).read_text())
                if detail['status']!='PASS_FROZEN_VISUAL_SEGMENT':raise RuntimeError('Episode failed')
                episodes.append(episode)
                for event in detail['events']:
                    if event['feedback']['scored']:groups.append(event['feedback'])
        if sorted(e['episode_index'] for e in episodes)!=list(range(count)) or len(groups)!=count*2:
            raise RuntimeError('Exact episode/root coverage failed')
        report.update(status='PASS_FROZEN_VALIDATION',episodes=episodes,reward_roots=len(groups),
            elapsed_seconds=time.monotonic()-started,lane_seconds=[r['elapsed_seconds'] for r in lane_reports],
            all_zero_roots=sum(g['diagnostics']['all_zero'] for g in groups),
            reference_failed_roots=sum(g['diagnostics']['reference_failed'] for g in groups),
            mean_ep_saturation=sum(g['diagnostics']['ep_saturated_fraction'] for g in groups)/len(groups),
            delayed_safety_candidates=sum(g['diagnostics']['delayed_safety_candidates'] for g in groups),
            h1_comparisons=[g['diagnostics'].get('h1_comparison') for g in groups],
            official_pdms=False,driving_improvement_proven=False,training=False)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:
        for child in children:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        for log in logs:log.close()
        for sig,handler in previous.items():signal.signal(sig,handler)
        write_json(output/'report.json',report)

if __name__=='__main__':main()
