"""Frozen visual validation worker for the new, separate segment protocol."""
import argparse
import json
import logging
from pathlib import Path
import pickle
import socket
import traceback
import numpy as np
from .paths import sha256_file
from .scene_payload import single_scene_mapping
from .visual_assets import visual_asset
from .transport import Channel,digest
from .image_transport import ImageBuffer,metadata_to_wire
from worldengine.online.headless import HeadlessSimulator
from worldengine.online.observations import CanonicalObserver
from worldengine.online.actions import to_world_actions
from worldengine.online.fixed_rollout import FixedRewardSession,rollout
from worldengine.online.fixed_parallel import FixedBranchPool
from worldengine.online.fixed_diagnostics import compact_result,write_json
from worldengine.online.segment_reward import CONTRACT,SegmentReward
from worldengine.online.parallel_branches import _IDMFailures
from worldengine.online.state import structural_hash


def main():
    p=argparse.ArgumentParser()
    for name in ('fd','images-fd'):p.add_argument('--'+name,type=int,required=True)
    for name in ('settings','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','diagnostic','frozen'],required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--reaction',choices=['R','NR'],default='R')
    p.add_argument('--phase',type=int,default=0)
    p.add_argument('--verify-serial',action='store_true')
    args=p.parse_args()
    channel=Channel(socket.socket(fileno=args.fd),timeout=1800)
    images=ImageBuffer(args.images_fd);sim=pool=None
    failures=_IDMFailures();logging.getLogger().addHandler(failures)
    try:
        cfg=json.loads(args.settings.read_text())
        source=Path(cfg['visual_scene_path'])
        if sha256_file(source)!=cfg['visual_scene_sha256']:raise ValueError('Scene digest mismatch')
        with source.open('rb') as f:scenes=single_scene_mapping(pickle.load(f),cfg['visual_scene_id'])
        scene_id,asset_root,asset=visual_asset(cfg,scenes);scene=scenes[scene_id]
        warmup=13 if args.mode=='diagnostic' else 3
        steps=8 if args.mode=='frozen' else 2
        max_steps=warmup+steps-1+8
        sim=HeadlessSimulator(scene_id,scene,args.reaction,max_steps,cfg['scene_seed'])
        observer=CanonicalObserver(sim,asset_root)
        session=FixedRewardSession(sim,cfg['map_root'])
        if args.workers:
            pool=FixedBranchPool(scene_id,scene,args.reaction,max_steps,cfg['scene_seed'],
                sim.codec.bundle,session.offset,session.actor_types,workers=args.workers,timeout=900,
                global_config=sim.engine.global_config)
        channel.send(dict(kind='ready',scene=scene_id,warmup=warmup,steps=steps,
            contract=CONTRACT,reaction=args.reaction,workers=args.workers,
            pool_startup_seconds=pool.startup_seconds if pool else 0.,
            worker_receipts=pool.worker_receipts if pool else [],max_steps=max_steps,
            log_length=scene['log_length']))
        renders=groups=0
        for index in range(warmup+steps):
            frame,payload,state_hash=observer.observe();renders+=1
            identity=dict(scene=scene_id,step=index,state_hash=state_hash,token=frame['token'])
            descriptor=images.publish(payload,identity)
            channel.send(dict(kind='observation',identity=identity,frame=metadata_to_wire(frame),images=descriptor))
            request=channel.receive()
            if request.get('identity')!=identity:raise ValueError('Stale action identity')
            images.release(identity)
            if index<warmup:
                if request['kind']!='warmup':raise ValueError('Expected warmup')
                session.warmup(session.warmup_action());continue
            if request['kind']!='action' or digest(request['candidates'])!=request['candidate_hash']:
                raise ValueError('Invalid candidate message')
            selected=request['selected']
            if type(selected) is not int or not 0<=selected<20:raise ValueError('Invalid selection')
            if sim.snapshot().state_hash!=state_hash:raise ValueError('Root state changed')
            actions=to_world_actions(np.asarray(request['candidates'],dtype=np.float32),sim.engine.agent_manager.ego_agent)
            def canonical(snapshot,states):
                channel.send(dict(kind='main',identity=identity,next_hash=snapshot.state_hash,
                    candidate_hash=request['candidate_hash'],selected=selected))
            scored=args.mode!='frozen' or (index-warmup)%4==args.phase%4
            if scored:
                before=sim.snapshot()
                observer_hash=structural_hash((list(observer.history.raw),list(observer.history.frames)))
                result=session.group(actions,selected,pool=pool,on_main=canonical,compare_h1=args.mode=='diagnostic')
                if observer_hash!=structural_hash((list(observer.history.raw),list(observer.history.frames))):
                    raise RuntimeError('Branch mutated visual history')
                if args.verify_serial:
                    main=sim.snapshot()
                    try:
                        for i,action in enumerate(actions):
                            serial=rollout(sim,before,action,session.offset,session.actor_types)
                            expected=result['traces'][i]
                            if serial['state_hashes']!=expected['state_hashes'] or [f.state_hash for f in serial['frames']]!=[f.state_hash for f in expected['frames']]:
                                raise RuntimeError('Real candidate serial/parallel trace mismatch')
                    finally:sim.restore(main)
                record=compact_result(result)
                record.update(identity=identity,candidates=request['candidates'],candidate_hash=request['candidate_hash'],
                    candidate_source='live_frozen_DiffusionDrive',serial_verified=bool(args.verify_serial))
                trace_path=args.output/('root_%02d.json'%index)
                write_json(trace_path,record)
                feedback={k:v for k,v in record.items() if k not in ('traces','reference','candidates')}
                feedback['reference']={k:v for k,v in record['reference'].items() if k!='frames'}
                feedback['trace_path']=str(trace_path);feedback['trace_sha256']=sha256_file(trace_path)
                groups+=1
            else:
                previous=session.history.state_hash
                sim.step(actions[selected]);main=sim.snapshot();canonical(main,None)
                session.history.append(session.capture())
                feedback=dict(main_hash=main.state_hash,history_before_hash=previous,
                              history_after_hash=session.history.state_hash,contract=CONTRACT['name'])
            if failures.count:raise RuntimeError('IDM fallback invalidates probe')
            feedback.update(kind='feedback',identity=identity,candidate_hash=request['candidate_hash'],selected=selected,scored=scored)
            channel.send(feedback)
            if channel.receive()!=dict(kind='ack',identity=identity,next_hash=feedback['main_hash']):
                raise ValueError('Missing audited feedback acknowledgement')
        if channel.receive()!=dict(kind='close'):raise ValueError('Expected close')
        if pool:pool.close();pool=None
        channel.send(dict(kind='closed',idm_fallbacks=0,render_calls=renders,reward_groups=groups,
                          branch_render_calls=0,branch_model_calls=0))
    except BaseException as error:
        try:channel.send(dict(kind='error',error=repr(error),traceback=traceback.format_exc()))
        except (OSError,EOFError):pass
        raise
    finally:
        if pool:pool.close(abort=True)
        if sim:sim.close()
        images.close();channel.sock.close();logging.getLogger().removeHandler(failures)

if __name__=='__main__':main()
