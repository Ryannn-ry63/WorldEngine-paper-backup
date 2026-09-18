"""Real-data migration, forward/backward and exact restart checks."""
import copy
from pathlib import Path
import numpy as np
import torch
from .common import read,write,artifact,verify,sha,legacy_imports,CODE
from .data import checked,Dataset
from .train import objective,run_training

def validate(run,device):
    legacy_imports()
    import selector_cfpi_model as m
    manifest=checked(run);data=Dataset(manifest)
    torch.set_num_threads(4);torch.manual_seed(20260916)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=read(CODE.parent/'registry/migration.json')
    records={r['path']:r for r in source['files']}
    dependencies=['projects/AlgEngine/scripts/diffusiondrive/'+n for n in
        ('selector_cfpi_model.py','grpo_selector_v3_cached_common.py','selector_attribution_common.py','selector_rare_model.py')]
    dependencies.append('projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_scene_selector.py')
    for name in dependencies:
        expected=records[name]['sha256']
        if sha(CODE/name)!=expected or sha(Path(source['source'])/name)!=expected:
            raise RuntimeError('Inherited implementation differs: '+name)
    teacher,_,_=m.load_incumbent(verify(manifest['incumbent_manifest']))
    teacher.eval().requires_grad_(False)
    rows=data.feedback;maxerr=0.;mismatches=0
    deployed=copy.deepcopy(teacher).to(device).eval()
    with torch.no_grad():
        for start in range(0,len(rows),16):
            batch=rows[start:start+16];ids=list(range(len(batch)))
            cpu=m.score(teacher,batch,ids,'cpu','residual')
            other=m.score(deployed,batch,ids,device,'residual').cpu()
            maxerr=max(maxerr,float((cpu-other).abs().max()))
            mismatches+=int((cpu.argmax(-1)!=other.argmax(-1)).sum())
    if maxerr>1e-4 or mismatches:raise RuntimeError('CPU/device cached inference differs')
    step=read(verify(manifest['schedules']['0']))[0];teacher.to(device);probes={}
    for arm in ('D1','D2','D3','D4'):
        model=copy.deepcopy(teacher).train().requires_grad_(True)
        dense,fb=data.batch(step,arm)
        loss,terms=objective(model,teacher,dense,fb,arm,device);loss.backward()
        norm=float(m.v3.clip_grad_norm_cpu_(model.parameters(),10.))
        if not np.isfinite(norm) or norm<=0:raise RuntimeError('Invalid real-cache gradient')
        if arm in ('D1','D2') and any(float(terms[k])!=0 for k in ('pair','winner_nll')):
            raise RuntimeError('Closed labels reached no-preference controls')
        probes[arm]=dict(loss=float(loss.detach()),gradient_norm=norm,**{k:float(v.detach()) for k,v in terms.items()})
    result=dict(status='PASS',manifest=artifact(run/'manifest.json'),device=device,
        source_dependency_hash_parity=True,states=len(rows),cpu_device_max_abs=maxerr,
        argmax_mismatches=mismatches,probes=probes,optimizer_steps=0,
        physical_simulator_bridge_completed=False)
    write(run/f'validation_{device.replace(":","_")}.json',result)
    return result

def smoke_suite(run,device):
    results={a:run_training(run,a,0,device,smoke=True) for a in ('D1','D2','D3','D4')}
    folder=run/'smoke/D3_seed1'
    if not (folder/'report.json').exists():
        run_training(run,'D3',1,device,smoke=True,stop_after=2)
    resumed=run_training(run,'D3',1,device,smoke=True)
    legacy_imports()
    import selector_cfpi_model as m
    from .data import TRAIN
    manifest=checked(run);data=Dataset(manifest);torch.manual_seed(1)
    teacher,_,_=m.load_incumbent(verify(manifest['incumbent_manifest']))
    teacher.to(device).eval().requires_grad_(False)
    model=copy.deepcopy(teacher).train().requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])
    schedule=read(verify(manifest['schedules']['1']))
    for step in schedule[:4]:
        dense,fb=data.batch(step,'D3');loss,_=objective(model,teacher,dense,fb,'D3',device)
        optimizer.zero_grad(set_to_none=True);loss.backward();m.v3.clip_grad_norm_cpu_(model.parameters(),10.);optimizer.step()
    state=torch.load(verify(resumed['resume']),map_location=device)
    if any(not torch.equal(v,state['model'][k]) for k,v in model.state_dict().items()):
        raise RuntimeError('Interrupted vs uninterrupted model mismatch')
    for k,v in optimizer.state_dict()['state'].items():
        for name,value in v.items():
            other=state['optimizer']['state'][k][name]
            if torch.is_tensor(value):
                if not torch.equal(value.cpu(),other.cpu()):raise RuntimeError('Optimizer resume mismatch')
            elif value!=other:raise RuntimeError('Optimizer resume mismatch')
    cpu=read(run/'validation_cpu.json');gpu=read(run/f'validation_{device.replace(":","_")}.json')
    if not device.startswith('cuda') or any(x['status']!='PASS' or x['manifest']!=artifact(run/'manifest.json') for x in (cpu,gpu)):
        raise RuntimeError('Both CPU/CUDA validation required')
    result=dict(status='PASS',manifest=artifact(run/'manifest.json'),smoke_reports={a:artifact(run/'smoke'/f'{a}_seed0/report.json') for a in results},
        cpu=artifact(run/'validation_cpu.json'),gpu=artifact(run/f'validation_{device.replace(":","_")}.json'),
        resume_model_and_optimizer_bitwise_equal=True,trained_formal_models=0,
        physical_simulator_bridge_completed=False)
    write(run/'validation.json',result)
    return result
