"""One GPU, resident frozen model, multiple independently reset episodes."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
import numpy as np
import torch
from .visual_model import configuration,load_frozen,reset_temporal_state
from .visual_online_probe import _sample_context,source_hashes as bridge_source_hashes
from .live_inputs import LiveInputs
from .live_audit import LiveWindow
from .image_transport import ImageBuffer,metadata_from_wire
from .transport import Channel,digest
from .paths import sha256_file
from .fixed_protocol import validate


def source_hashes():
    code=Path(__file__).resolve().parents[5]
    hashes=bridge_source_hashes()
    for relative in ('projects/SimEngine/worldengine/components',
                     'projects/SimEngine/worldengine/manager',
                     'projects/SimEngine/worldengine/engine',
                     'projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads',
                     'projects/AlgEngine/configs/diffusiondrive'):
        for path in (code/relative).rglob('*.py'):
            hashes[str(path.relative_to(code))]=sha256_file(path)
    return dict(sorted(hashes.items()))


def write_json(path,data):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def episode(cfg,settings,output,model,live,args,index):
    output.mkdir(exist_ok=False)
    left,right=socket.socketpair(socket.AF_UNIX,socket.SOCK_STREAM)
    channel=Channel(left,timeout=1800);images=ImageBuffer();child=None
    window=LiveWindow();expected_state=expected_history=None
    started=time.monotonic();report=dict(status='RUNNING',events=[],mode=args.mode)
    try:
        command=[cfg['simengine_python'],'-u','-m','innovation3.fixed_visual_worker',
            '--fd',str(right.fileno()),'--images-fd',str(images.fd),'--settings',str(settings),
            '--output',str(output),'--mode',args.mode,'--workers',str(args.workers),
            '--reaction',args.reaction,'--phase',str(index%4)]
        if args.verify_serial:command.append('--verify-serial')
        with (output/'worker.log').open('x') as log:
            child=subprocess.Popen(command,env=dict(os.environ),pass_fds=(right.fileno(),images.fd),stdout=log,stderr=subprocess.STDOUT)
            right.close();ready=channel.receive()
            if ready.get('kind')!='ready':raise RuntimeError('Worker startup: '+repr(ready))
            warmup=13 if args.mode=='diagnostic' else 3;steps=8 if args.mode=='frozen' else 2
            if ready['warmup']!=warmup or ready['steps']!=steps:raise ValueError('Window contract mismatch')
            report['worker']=ready;calls=0
            for step in range(warmup+steps):
                message=channel.receive()
                if message.get('kind')!='observation':raise RuntimeError('Expected observation: '+repr(message))
                identity=message['identity']
                if identity['step']!=step or identity['scene']!=ready['scene']:raise ValueError('Observation order mismatch')
                if expected_state is not None and identity['state_hash']!=expected_state:raise ValueError('Next render is not canonical')
                frame=metadata_from_wire(message['frame']);cameras=images.receive(message['images'],identity)
                window.append(frame,cameras,identity)
                if step<warmup:
                    channel.send(dict(kind='warmup',identity=identity));continue
                t=time.monotonic();data=live.prepare(list(window.frames),list(window.cameras))
                with torch.no_grad():results=model(return_loss=False,rescale=True,**data)
                torch.cuda.synchronize();calls+=1
                if not isinstance(results,list) or len(results)!=1:raise ValueError('Invalid model output')
                bank,selected,details=_sample_context(results[0],identity['token'],model.module.planning_head.scene_selector)
                channel.send(dict(kind='action',identity=identity,candidates=bank.tolist(),selected=selected,candidate_hash=digest(bank.tolist())))
                forward_seconds=time.monotonic()-t
                main=channel.receive();feedback=channel.receive()
                scored=args.mode!='frozen' or (step-warmup)%4==index%4
                validate(identity,selected,details['candidate_hash'],main,feedback,scored,expected_history)
                expected_state=main['next_hash'];expected_history=feedback['history_after_hash']
                report['events'].append(dict(step=step,identity=identity,selected=selected,**details,
                    forward_seconds=forward_seconds,feedback=feedback))
                write_json(output/'report.json',report)
                channel.send(dict(kind='ack',identity=identity,next_hash=expected_state))
            channel.send(dict(kind='close'));closed=channel.receive()
            if closed.get('kind')!='closed' or closed['idm_fallbacks']!=0:raise RuntimeError('Worker close failed: '+repr(closed))
            child.wait(timeout=60)
            if child.returncode:raise RuntimeError('Worker exit failure')
            if calls!=steps or closed['render_calls']!=warmup+steps or closed['reward_groups']!=2:
                raise RuntimeError('Unexpected forward/render/group count')
            report.update(status='PASS_FROZEN_VISUAL_SEGMENT',closed=closed,model_forward_calls=calls,
                          elapsed_seconds=time.monotonic()-started,online_updates=False)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        right.close();channel.sock.close();images.close();write_json(output/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('settings','assignments','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','diagnostic','frozen'],required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--reaction',choices=['R','NR'],default='R')
    p.add_argument('--verify-serial',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.settings.read_text());rows=json.loads(args.assignments.read_text())
    sources=source_hashes();started=time.monotonic();live=None
    report=dict(status='RUNNING',source_sha256=sources,episodes=[],model_loads=0,
                official_pdms=False,training=False,mode=args.mode)
    try:
        torch.manual_seed(0);np.random.seed(0)
        config=configuration(cfg,0);model=load_frozen(config,cfg);report['model_loads']=1
        report['gpu']=torch.cuda.get_device_name(0)
        live=LiveInputs(config.data.test)
        for row in rows:
            if row['split']!='train':raise ValueError('Only train-source diagnostics allowed')
            if source_hashes()!=sources:raise RuntimeError('Runtime source changed; use a frozen copy')
            reset_temporal_state(model)
            torch.manual_seed(row['candidate_seed']);np.random.seed(row['candidate_seed'])
            model.module.planning_head.set_candidate_noise_namespace('fixed_segment_seed'+str(row['candidate_seed']))
            local=dict(cfg,visual_scene_id=row['scene_id'],visual_scene_path=row['scene_path'],
                       visual_scene_sha256=row['scene_sha256'],visual_asset_path=row['asset_path'],
                       visual_asset_sha256=row['asset_sha256'],scene_seed=row['scene_seed'])
            settings=args.output/('episode_%03d_settings.json'%row['episode_index']);write_json(settings,local)
            result=episode(local,settings,args.output/('episode_%03d'%row['episode_index']),model,live,args,row['episode_index'])
            report['episodes'].append(dict(scene_id=row['scene_id'],episode_index=row['episode_index'],
                status=result['status'],elapsed_seconds=result['elapsed_seconds'],
                report=str(args.output/('episode_%03d'%row['episode_index'])/'report.json')))
            write_json(args.output/'report.json',report)
        if source_hashes()!=sources:raise RuntimeError('Runtime source changed during probe')
        report.update(status='PASS_FROZEN_LANE',elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if live:live.close()
        write_json(args.output/'report.json',report)

if __name__=='__main__':main()
