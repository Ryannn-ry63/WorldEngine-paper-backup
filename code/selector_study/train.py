import copy
import fcntl
import os
import time
from pathlib import Path
import numpy as np
import torch
from .common import read,write,artifact,verify,legacy_imports
from .data import checked,Dataset,ARMS,TRAIN

def save_state(path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(state,tmp);os.replace(tmp,path)

def objective(model,teacher,dense,feedback,arm,device):
    legacy_imports()
    import selector_cfpi_model as m
    import selector_attribution_common as a
    from selector_rare_model import teacher_kl
    rows = dense+feedback
    logits = m.score(model,rows,list(range(len(rows))),device,'residual')
    with torch.no_grad():
        reference = m.score(teacher,rows,list(range(len(rows))),device,'residual')
    rewards = torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense]),dtype=torch.float32,device=device)
    valid = torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense]),dtype=torch.bool,device=device)
    grpo = m.v3.exact_group_loss(logits[:16],reference[:16],rewards,valid,1.,0.)[0]
    if ARMS[arm]['feedback'] == 'none':
        pair = nll = logits.sum()*0.
    else:
        pair,nll = a.feedback_loss(logits[16:],feedback,'P1')
    kl = teacher_kl(logits,reference)
    loss = grpo+pair+nll+TRAIN['teacher_kl']*kl
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training objective')
    return loss,dict(dense=grpo,pair=pair,winner_nll=nll,teacher_kl=kl)

def run_training(run,arm,seed,device='cuda',smoke=False,stop_after=None):
    legacy_imports()
    import selector_cfpi_model as m
    if arm not in ARMS or seed not in (0,1,2):
        raise ValueError('Unregistered arm or seed')
    manifest = checked(run)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no implicit CPU fallback')
    if not smoke:
        validation = read(run/'validation.json')
        if validation['status']!='PASS' or validation['manifest']!=artifact(run/'manifest.json'):
            raise RuntimeError('CPU and CUDA validation must pass before formal training')
        for item in (validation['cpu'],validation['gpu'],*validation['smoke_reports'].values()):
            verify(item)
    torch.set_num_threads(4);torch.manual_seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    dataset = Dataset(manifest)
    schedule = read(verify(manifest['schedules'][str(seed)]))
    teacher,config,_ = m.load_incumbent(verify(manifest['incumbent_manifest']))
    teacher.to(device).eval().requires_grad_(False)
    model = copy.deepcopy(teacher).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])
    steps = 4 if smoke else TRAIN['steps']
    folder=run/('smoke' if smoke else 'checkpoints')/f'{arm}_seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    provenance=dict(manifest=artifact(run/'manifest.json'),arm=arm,seed=seed,steps=steps,smoke=smoke,
        device_type='cuda' if device.startswith('cuda') else 'cpu',
        device_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else 'cpu',
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,initial_optimizer_state='empty',parent=manifest['parent'])
    write(folder/'contract.json',provenance)
    if (folder/'report.json').exists():
        result=read(folder/'report.json')
        if result['provenance']!=provenance:raise RuntimeError('Existing model contract differs')
        verify(result['selector']);verify(result['resume']);return result
    resume=folder/'resume.pt';start=0;history=[];seconds=0.
    if resume.exists():
        state=torch.load(resume,map_location=device)
        if state['provenance']!=provenance or not 0<=state['step']<=steps:raise RuntimeError('Resume contract changed')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state(state['cuda_rng'].cpu(),device)
        start=state['step'];history=state['history'];seconds=state['seconds']
    else:
        if any(not torch.equal(v,teacher.state_dict()[k]) for k,v in model.state_dict().items()):
            raise RuntimeError('Step-zero model differs from V3')
        if optimizer.state:raise RuntimeError('Optimizer unexpectedly inherited shared history')
    timer=time.monotonic()
    end=min(steps,stop_after) if stop_after is not None else steps
    if end<start:raise ValueError('Stop step precedes existing checkpoint')
    for step in range(start+1,end+1):
        dense,feedback=dataset.batch(schedule[step-1],arm)
        loss,terms=objective(model,teacher,dense,feedback,arm,device)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=float(m.v3.clip_grad_norm_cpu_(model.parameters(),TRAIN['grad_clip']))
        if not np.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        if step%25==0 or step==end or smoke:
            row=dict(step=step,loss=float(loss.detach()),gradient_norm=norm,
                     **{k:float(v.detach()) for k,v in terms.items()})
            history.append(row)
            elapsed=seconds+time.monotonic()-timer
            save_state(resume,dict(provenance=provenance,step=step,model=model.state_dict(),
                optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                history=history,seconds=elapsed))
            write(folder/'progress.json',dict(status='RUNNING',step=step,steps=steps,seconds=elapsed),immutable=False)
            print(dict(arm=arm,seed=seed,**row),flush=True)
    if end<steps:return dict(status='PAUSED',step=end,resume=artifact(resume))
    model.eval();out=folder/'selector.pt'
    m.save_selector(out,model,config,'residual',provenance)
    restored,_=m.load_selector(out);restored.to(device).eval()
    rows=dataset.feedback[:16]
    with torch.no_grad():
        a=m.score(model,rows,list(range(16)),device,'residual')
        b=m.score(restored,rows,list(range(16)),device,'residual')
    error=float((a-b).abs().max())
    if error>1e-5 or not torch.equal(a.argmax(-1),b.argmax(-1)):raise RuntimeError('Export parity failed')
    result=dict(status='PASS',stage='SMOKE' if smoke else 'TRAINED_NOT_EVALUATED',provenance=provenance,
        selector=artifact(out),resume=artifact(resume),steps=steps,seconds=seconds+time.monotonic()-timer,
        trainable_tensors=len(model.state_dict()),trainable_parameters=sum(p.numel() for p in model.parameters()),
        export_max_abs=error,history=history,evaluation_complete=False)
    write(folder/'report.json',result)
    write(folder/'progress.json',dict(status=result['stage'],step=steps,steps=steps,seconds=result['seconds']),immutable=False)
    return result
