"""Deterministic cached training; smoke runs and formal endpoints are separate."""
import copy
import time
import numpy as np
import torch
from .common import artifact, legacy_imports, lock, read, torch_save, verify, write
from .data import checked, feedback_dataset, preferences
from .objectives import feedback_terms, teacher_kl
from .specs import FEEDBACK, STRUCTURES


def setup(seed,device):
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.set_num_threads(4);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def feedback_training(run,arm,seed,device='cuda',smoke=False,stop_after=None):
    if arm not in FEEDBACK or seed not in (0,1,2): raise ValueError('Unknown arm/seed')
    manifest=checked(run)
    if manifest['queue']!='B': raise ValueError('Need feedback study manifest')
    setup(seed,device);legacy_imports()
    import selector_cfpi_model as m
    ds=feedback_dataset(manifest)
    teacher,config,_=m.load_incumbent(verify(manifest['source_contract']['incumbent_manifest']))
    teacher.to(device).eval().requires_grad_(False)
    model=copy.deepcopy(teacher).train().requires_grad_(True)
    training=manifest['training'];spec=FEEDBACK[arm]
    schedule=read(verify(manifest['source_contract']['schedules'][str(seed)]))
    def batch_loss(step):
        dense,feedback=ds.batch(schedule[step-1],'D4')
        rows=dense+feedback
        z=m.score(model,rows,list(range(32)),device,'residual')
        with torch.no_grad():ref=m.score(teacher,rows,list(range(32)),device,'residual')
        rewards=torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense]),device=device,dtype=torch.float32)
        valid=torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense]),device=device,dtype=torch.bool)
        dense_loss=m.v3.exact_group_loss(z[:16],ref[:16],rewards,valid,1.,0.)[0]
        pairs,outcomes=preferences(feedback)
        extra,terms=feedback_terms(z[16:],ref[16:],pairs,outcomes,spec,step,training['steps'])
        if spec.get('local20'):
            r=torch.as_tensor(np.stack([r['local20_rewards'] for r in feedback]),device=device,dtype=torch.float32)
            extra=m.v3.exact_group_loss(z[16:],ref[16:],r,torch.ones_like(r,dtype=torch.bool),1.,0.)[0]
            terms['local20']=extra
        # Removing feedback-state KL retains the original 1/32 dense contribution.
        kl=(.5*teacher_kl(z[:16],ref[:16]) if spec.get('feedback_kl') is False else teacher_kl(z,ref))
        loss=dense_loss+extra+training['teacher_kl']*kl
        return loss,dict(dense=dense_loss,feedback=extra,kl=kl,**terms)
    def save(path,provenance):m.save_selector(path,model,config,'residual',provenance)
    return optimize(run,arm,seed,model,config,training,batch_loss,save,device,smoke,stop_after)


def optimize(run,arm,seed,model,config,training,batch_loss,save,device,smoke,stop_after=None):
    legacy_imports()
    import grpo_selector_v3_cached_common as v3
    folder=run/('smoke' if smoke else 'checkpoints')/f'{arm}_seed{seed}'
    steps=4 if smoke else training['steps']
    provenance=dict(manifest=artifact(run/'manifest.json'),arm=arm,seed=seed,smoke=smoke,steps=steps,
        training=training,device_type=device.split(':')[0],torch_version=torch.__version__,
        device_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'cpu')
    with lock(folder/'train.lock'):
        if (folder/'report.json').exists():
            previous=read(folder/'report.json')
            if previous['provenance']!=provenance:raise RuntimeError('Training identity changed')
            verify(previous['selector']);verify(previous['resume']);return previous
        write(folder/'contract.json',provenance)
        optimizer=torch.optim.AdamW(model.parameters(),lr=training['lr'],weight_decay=training['weight_decay'],foreach=False)
        start=0;history=[];seconds=0.;resume=folder/'resume.pt'
        if resume.exists():
            state=torch.load(resume,map_location=device)
            if state['provenance']!=provenance:raise RuntimeError('Resume identity changed')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['rng'].cpu())
            if device.startswith('cuda'):torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
            start=state['step'];history=state['history'];seconds=state['seconds']
        end=min(steps,stop_after) if stop_after is not None else steps
        if end<start:raise ValueError('Stop precedes saved step')
        clock=time.monotonic()
        for step in range(start+1,end+1):
            optimizer.zero_grad(set_to_none=True)
            loss,terms=batch_loss(step)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();norm=v3.clip_grad_norm_cpu_(model.parameters(),training['grad_clip']);optimizer.step()
            if step%25==0 or step==end or smoke:
                row=dict(step=step,loss=float(loss.detach()),gradient_norm=norm,**{k:float(v.detach()) for k,v in terms.items()})
                history.append(row)
                torch_save(resume,dict(provenance=provenance,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                    history=history,seconds=seconds+time.monotonic()-clock))
                write(folder/'progress.json',dict(status='RUNNING',step=step,steps=steps,seconds=seconds+time.monotonic()-clock),False)
                print(dict(arm=arm,seed=seed,**row),flush=True)
            if not smoke and step in training.get('checkpoints',[]):save(folder/f'step_{step}_selector.pt',provenance)
        if end<steps:
            write(folder/'progress.json',dict(status='PAUSED',step=end,steps=steps),False)
            return dict(status='PAUSED',step=end)
        model.eval();selector=folder/'selector.pt';save(selector,provenance)
        result=dict(status='PASS',stage='SMOKE' if smoke else 'TRAINED_NOT_EVALUATED',provenance=provenance,
            selector=artifact(selector),resume=artifact(resume),steps=steps,seconds=seconds+time.monotonic()-clock,
            trainable_parameters=sum(p.numel() for p in model.parameters()),config=config,history=history,evaluation_complete=False)
        write(folder/'report.json',result)
        write(folder/'progress.json',dict(status=result['stage'],step=end,steps=steps),False)
        return result


def cached_training(run,arm,seed,device='cuda',smoke=False,stop_after=None):
    manifest=checked(run);queue=manifest['queue']
    if queue not in ('C','D','F'):raise ValueError('Need cached structural/source study')
    if queue=='C' and arm not in STRUCTURES:raise ValueError('Unknown structure')
    if queue=='D' and arm not in ('selector_sft','selector_grpo_matched'):raise ValueError('Use planner training for L2/BC')
    if queue=='F' and arm not in ('paired_common','rarelog','rarerollout'):raise ValueError('Unknown source')
    setup(seed,device);legacy_imports()
    if queue=='D':
        manifest=dict(manifest,_supervised_inputs=artifact(run/'supervised_inputs.json'))
    import grpo_selector_v3_cached_common as v3
    from . import models
    from .sources import CachedPool, SourcePool, expert_targets
    pool=SourcePool(manifest) if queue=='F' else CachedPool(manifest)
    example=pool.example
    overrides=STRUCTURES[arm] if queue=='C' else {}
    config=dict(example['scene_selector_config']);config.update(overrides)
    original=example['baseline_selector_state']
    model=models.build(config,original).to(device).train()
    teacher=copy.deepcopy(model).eval().requires_grad_(False) if config.get('paper_original_selector') else None
    training=manifest['training']
    def batch_loss(step):
        cache,indices=pool.batch(seed,step,64,arm)
        x=v3.batch_inputs(cache,indices,device)
        base=cache['reference_logits'][indices].to(device)
        z=models.forward(model,x,base,config)
        if teacher is None:ref=base
        else:
            with torch.no_grad():cached_teacher=teacher(**x)
            # Compressed candidate features do not reproduce stored FP32 logits exactly.
            # Anchor the learned scorer delta to the stored reference, as for every residual arm.
            z=base+(z-cached_teacher);ref=base
        if queue=='D' and arm=='selector_sft':
            from .objectives import supervised_selector
            target,mask=expert_targets(cache,indices,manifest,device)
            policy=supervised_selector(z,x['candidate_trajectories'],target,mask)
            kl=teacher_kl(z,ref);loss=policy+training['teacher_kl']*kl
        else:
            reward=cache['candidate_rewards'][indices].to(device);valid=cache['candidate_reward_valid_mask'][indices].to(device)
            loss,policy,kl=v3.exact_group_loss(z,ref,reward,valid,1.,training['teacher_kl'])
        return loss,dict(policy=policy,kl=kl)
    def save(path,provenance):
        torch_save(path,dict(schema=1,method='paper_cached_selector',config=config,original_selector_state=original,
            state={k:v.detach().cpu() for k,v in model.state_dict().items()},provenance=provenance,
            inference_uses_reward_or_q=False))
    return optimize(run,arm,seed,model,config,training,batch_loss,save,device,smoke,stop_after)
