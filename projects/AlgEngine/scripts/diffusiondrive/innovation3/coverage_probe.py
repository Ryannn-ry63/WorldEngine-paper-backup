"""Frozen resident visual probes: technical admission and matched-root diagnosis.

No optimizer exists here. Ground-truth/reference information remains inside the
simulator scorer. Continuations replay the same prefix from reset and verify the
root/candidate hashes before changing the first action.
"""
import argparse
import copy
import json
import os
import signal
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .paths import sha256_file
from .fixed_client import FixedWorkerClient
from .visual_model import configuration,load_frozen,reset_temporal_state
from .live_inputs import LiveInputs
from .live_audit import LiveWindow
from .candidate_inputs import from_export
from .image_transport import metadata_from_wire
from .transport import digest
from .fixed_protocol import validate,configured_contract
from .live_learning import state_digest
from .online_probe import source_hashes


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


class FrozenProbe:
    def __init__(self,cfg,out,workers=4,comparison=None):
        self.cfg=cfg;self.out=Path(out);self.workers=workers;self.client=None;self.live=None
        torch.manual_seed(0);np.random.seed(0)
        self.config=configuration(cfg,0)
        self.model=load_frozen(self.config,cfg)
        self.parent=self.model.module.planning_head.scene_selector
        self.initial_hash=state_digest(self.model.state_dict())
        self.policies={'parent':self.parent}
        if comparison:
            payload=torch.load(comparison,map_location='cpu')
            candidate=copy.deepcopy(self.parent).eval().requires_grad_(False)
            candidate.load_state_dict(payload['scene_selector_state'],strict=True)
            self.policies['online64']=candidate
        self.live=LiveInputs(self.config.data.test)
        self.sources=source_hashes(); self.serial=0

    def close(self):
        if self.client:self.client.close(abort=True);self.client=None
        if self.live:self.live.close();self.live=None

    def episode(self,row,out,score_steps=None,parity=False,forced=None,actor_policy="parent"):
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        steps=8 if forced is None else forced['root_decision']+8
        score_steps=list(range(steps)) if score_steps is None else score_steps
        cfg=dict(self.cfg,visual_scene_id=row['scene_id'],visual_scene_path=row['scene_path'],
            visual_scene_sha256=row['scene_sha256'],visual_asset_path=row['asset_path'],
            visual_asset_sha256=row['asset_sha256'],online_scene_seed=row['scene_seed'],
            probe_score_steps=score_steps,probe_scorer_parity=parity)
        if forced is not None:cfg['probe_continuation_root']=forced['root_decision']
        write(out/'settings.json',cfg)
        report=dict(status='RUNNING',scene=row,events=[],forced=forced,policy='parent_argmax',
            source_sha256=self.sources,online_updates=False,actor_policy=actor_policy)
        started=time.monotonic()
        try:
            reset_temporal_state(self.model)
            torch.manual_seed(row['candidate_seed']);np.random.seed(row['candidate_seed'])
            self.model.module.planning_head.set_candidate_noise_namespace('coverage_diagnostic_'+str(row['candidate_seed']))
            if self.client is None:
                self.serial+=1
                self.client=FixedWorkerClient(cfg,self.workers,cfg['fixed_backend'],
                    self.out/('worker_%04d.log'%self.serial),dict(os.environ))
            ready=self.client.begin(cfg,out/'traces',steps,0,0,32,mode='frozen')
            if ready.get('kind')!='ready' or ready['source']['sha256']!=row['scene_sha256'] or ready['asset']['sha256']!=row['asset_sha256']:
                raise RuntimeError('Wrong or failed worker: '+repr(ready))
            report['worker']=ready
            window=LiveWindow();history=None;next_hash=None
            for step in range(3+steps):
                message=self.client.channel.receive()
                if message.get('kind')!='observation':raise RuntimeError('Observation failed: '+repr(message))
                identity=message['identity']
                if identity['step']!=step or identity['scene']!=row['scene_id'] or (next_hash and identity['state_hash']!=next_hash):
                    raise RuntimeError('Canonical state/observation continuity failed')
                cameras=self.client.images.receive(message['images'],identity)
                frame=metadata_from_wire(message['frame']);window.append(frame,cameras,identity)
                self.client.images.release(identity)
                if step<3:
                    self.client.channel.send(dict(kind='warmup',identity=identity));continue
                decision=step-3
                data=self.live.prepare(list(window.frames),list(window.cameras))
                with torch.no_grad():
                    result=self.model(return_loss=False,rescale=True,**data)[0]
                    context,base,_=from_export(result,identity['token'],self.parent)
                    logits={name:(base+policy(**context))[0].cpu().tolist() for name,policy in self.policies.items()}
                bank=context['candidate_trajectories'][0].cpu().tolist();bank_hash=digest(bank)
                selected=int(np.argmax(logits[actor_policy]))
                if forced is not None and decision==forced['root_decision']:
                    if bank_hash!=forced['candidate_hash'] or identity['state_hash']!=forced['state_hash']:
                        raise RuntimeError('Counterfactual prefix failed to reproduce the exact root/candidates')
                    selected=forced['selected']
                action=dict(kind='action',identity=identity,candidates=bank,candidate_hash=bank_hash,selected=selected)
                self.client.channel.send(action)
                main=self.client.channel.receive();feedback=self.client.channel.receive()
                validate(identity,selected,bank_hash,main,feedback,decision in score_steps,history,
                         expected_contract=configured_contract(cfg))
                next_hash=main['next_hash'];history=feedback['history_after_hash']
                event=dict(decision=decision,identity=identity,selected=selected,candidate_hash=bank_hash,
                    logits=logits,main=main,feedback=feedback)
                if feedback['scored']:
                    rewards=np.array([r['reward'] for r in feedback['rewards']])
                    chosen={name:int(np.argmax(values)) for name,values in logits.items()}
                    chosen['oracle']=int(np.argmax(rewards))
                    event['choices']=chosen
                    event['choice_rewards']={name:float(rewards[idx]) for name,idx in chosen.items()}
                    event['oracle_gap']={name:float(rewards.max()-rewards[idx]) for name,idx in chosen.items()}
                report['events'].append(event)
                self.client.channel.send(dict(kind='ack',identity=identity,next_hash=next_hash))
            self.client.channel.send(dict(kind='close'));closed=self.client.channel.receive()
            if closed.get('kind')!='closed' or closed.get('idm_fallbacks')!=0:
                raise RuntimeError('Failed simulator close: '+repr(closed))
            if closed['reward_groups']!=len(score_steps) or closed['render_calls']!=3+steps:
                raise RuntimeError('Incomplete execution/scoring')
            if state_digest(self.model.state_dict())!=self.initial_hash or source_hashes()!=self.sources:
                raise RuntimeError('Frozen model or source changed')
            report.update(status='PASS_FROZEN_PROBE',closed=closed,groups=closed['reward_groups'],
                candidate_simulation_steps=closed['candidate_simulation_steps'])
        except BaseException as error:
            report.update(status='FAIL_FROZEN_PROBE',error=repr(error),traceback=traceback.format_exc())
            if self.client:self.client.close(abort=True);self.client=None
            raise
        finally:
            report['elapsed_seconds']=time.monotonic()-started;write(out/'report.json',report)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--assignments',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--purpose',choices=('admission','diagnostic','extra'),required=True)
    p.add_argument('--comparison-selector',type=Path)
    p.add_argument('--candidate-manifest',type=Path)
    p.add_argument('--receipt-dir',type=Path)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    def interrupted(signum, frame): raise InterruptedError('Signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.settings.read_text());rows=json.loads(a.assignments.read_text())
    if a.purpose in ('diagnostic','extra') and not a.comparison_selector:p.error('Comparison selector required')
    probe=FrozenProbe(cfg,a.output,a.workers,a.comparison_selector)
    summary=dict(status='RUNNING',purpose=a.purpose,episodes=[])
    try:
        for i,row in enumerate(rows):
            receipt_path=a.receipt_dir/(row['scene_id']+'.json') if a.receipt_dir else None
            if receipt_path and receipt_path.exists():
                existing=json.loads(receipt_path.read_text())
                if (existing['status']=='PASS_TECHNICAL_ADMISSION' and existing['scene']==row and
                        existing['candidate_manifest_sha256']==sha256_file(a.candidate_manifest) and
                        existing['report_sha256']==sha256_file(existing['report'])):
                    continue
                if existing.get('status')!='BLOCKED_RUNTIME_ERROR' or existing['scene']!=row:
                    raise RuntimeError('Mismatched admission receipt')
                archive=a.output/(row['scene_id']+'_previous_failure.json')
                write(archive,existing)  # preserve failed attempt before retrying unchanged inputs
            reaction=row.get('probe_reaction','R')
            if probe.cfg.get('probe_reaction','R')!=reaction and probe.client:
                probe.client.close();probe.client=None
            probe.cfg['probe_reaction']=reaction
            out=a.output/(row['scene_id']+('_'+reaction if a.purpose=='extra' else ''))
            try:
                result=probe.episode(row,out,score_steps=[0,4] if a.purpose=='diagnostic' else None,
                                     parity=a.purpose in ('diagnostic','extra'))
                summary['episodes'].append(str(out/'report.json'))
                if receipt_path:
                    manifest=json.loads(a.candidate_manifest.read_text())
                    write(receipt_path,dict(status='PASS_TECHNICAL_ADMISSION',scene=row,
                        candidate_manifest_sha256=sha256_file(a.candidate_manifest),source_sha256=manifest['source_sha256'],
                        report=str(out/'report.json'),report_sha256=sha256_file(out/'report.json')))
                if a.purpose=='extra':
                    other_out=out.with_name(out.name+'_online64')
                    other=probe.episode(row,other_out,parity=True,actor_policy='online64')
                    first_divergence=None;first_failure=None
                    for parent_event,online_event in zip(result['events'],other['events']):
                        if first_divergence is None and parent_event['selected']!=online_event['selected']:
                            if (parent_event['identity']['state_hash']!=online_event['identity']['state_hash'] or
                                    parent_event['candidate_hash']!=online_event['candidate_hash']):
                                raise RuntimeError('Paired policies diverged before first different action')
                            first_divergence=parent_event['decision']
                        component=online_event['feedback']['rewards'][online_event['selected']]
                        if first_failure is None and (component['NC']<1 or component['DAC']<1):
                            first_failure=online_event['decision']
                    write(out/'extra_roots.json',dict(first_divergence=first_divergence,first_predicted_failure=first_failure,
                        parent_report=str(out/'report.json'),online_report=str(other_out/'report.json'),
                        trigger=row['extra_trigger'],scope='headless matched-root follow-up; distinct from legacy evaluation trace',
                        included_in_representative_48=False))
                if a.purpose=='diagnostic':
                    # Three first actions, identical parent continuation policy.
                    # The root is replayed from reset to avoid restoring renderer caches.
                    continuation_reports=[]
                    for event in result['events']:
                        if 'choices' not in event:continue
                        for name,selected in event['choices'].items():
                            forced=dict(root_decision=event['decision'],candidate_hash=event['candidate_hash'],
                                state_hash=event['identity']['state_hash'],selected=selected,choice=name,
                                fixed_reward=event['choice_rewards'][name])
                            target=a.output/(row['scene_id']+'_root%d_%s'%(event['decision'],name))
                            probe.episode(row,target,score_steps=[event['decision']],forced=forced)
                            continuation_reports.append(str(target/'report.json'))
                    write(out/'continuations.json',continuation_reports)
            except BaseException as error:
                if receipt_path:
                    # A GPU, source/parity or runtime error is never converted
                    # into an excluded driving scene.
                    write(receipt_path,dict(status='BLOCKED_RUNTIME_ERROR',scene=row,error=repr(error),
                        candidate_manifest_sha256=sha256_file(a.candidate_manifest),
                        source_sha256=json.loads(a.candidate_manifest.read_text())['source_sha256']))
                raise
            print(json.dumps(dict(event='frozen_scene_complete',completed=i+1,total=len(rows),scene=row['scene_id'])),flush=True)
        summary['status']='PASS_FROZEN_LANE'
    finally:
        probe.close();write(a.output/'summary.json',summary)
if __name__=='__main__':main()
