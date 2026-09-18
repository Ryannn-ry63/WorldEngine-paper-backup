"""Assemble native checkpoints and prove frozen tensors and selector logits survive."""
import copy
import torch
from .common import artifact, legacy_imports, lock, read, torch_save, verify, write
from .data import checked
PREFIX='planning_head.scene_selector.'

def export_one(run,arm,seed):
    manifest=checked(run);legacy_imports()
    import selector_cfpi_model as m
    from materialize_grpo_selector_v3 import state_dict
    from selector_study.data import Dataset
    report_path=run/'checkpoints'/f'{arm}_seed{seed}'/'report.json';trained=read(report_path)
    if trained['stage']!='TRAINED_NOT_EVALUATED' or trained['provenance']['manifest']!=artifact(run/'manifest.json'):
        raise RuntimeError('Formal contract-matched endpoint required')
    selector=verify(trained['selector']);payload=torch.load(selector,map_location='cpu')
    incumbent=read(verify(manifest['source_contract']['incumbent_manifest']))
    base=dict(path=incumbent['baseline'],sha256=incumbent['baseline_sha256'])
    config=payload.get('config',payload.get('scene_selector_config'))
    weights=payload.get('state',payload.get('scene_selector_state'))
    folder=run/'native'/f'{arm}_seed{seed}'
    provenance=dict(training=artifact(report_path),selector=trained['selector'],base=base,code=manifest['code'])
    with lock(folder/'export.lock'):
        if (folder/'report.json').exists():
            old=read(folder/'report.json')
            if old['provenance']!=provenance:raise RuntimeError('Native export identity changed')
            verify(old['checkpoint']);return old
        checkpoint=torch.load(verify(base),map_location='cpu');frozen=state_dict(checkpoint)
        assembled=copy.deepcopy(checkpoint);target=state_dict(assembled)
        changed=set()
        if config.get('paper_original_selector'):
            branch='planning_head.diff_decoder.layers.1.task_decoder.plan_cls_branch.'
            for k,v in weights.items():
                key=branch+k.removeprefix('head.')
                if key not in target or target[key].shape!=v.shape:raise RuntimeError('Original selector tensor differs: '+key)
                target[key]=v.clone();changed.add(key)
        else:
            for k,v in weights.items():target[PREFIX+k]=v.clone()
        assembled.setdefault('meta',{})['paper_study']=dict(config=config,provenance=provenance,inference_uses_reward_or_q=False)
        output=folder/'checkpoint.pth';torch_save(output,assembled)
        saved=state_dict(torch.load(output,map_location='cpu'))
        if any(not torch.equal(v,saved[k]) for k,v in frozen.items() if k not in changed):
            raise RuntimeError('Frozen nonselector tensors changed')
        from .models import build, forward
        original=payload.get('original_selector_state')
        left=build(config,original);left.load_state_dict(weights);left.eval()
        right=build(config,original)
        restored=({k: saved['planning_head.diff_decoder.layers.1.task_decoder.plan_cls_branch.'+k.removeprefix('head.')] for k in weights}
                  if config.get('paper_original_selector') else {k:saved[PREFIX+k] for k in weights})
        right.load_state_dict(restored);right.eval()
        rows=Dataset(manifest['source_contract']).feedback
        max_error=0.;mismatch=0
        with torch.no_grad():
            for i in range(0,len(rows),16):
                batch=rows[i:i+16];x=m.visible_batch(batch,list(range(len(batch))),'cpu')
                base_z=torch.tensor(__import__('numpy').stack([r['reference_logits'] for r in batch]))
                a=forward(left,x,base_z,config);b=forward(right,x,base_z,config)
                max_error=max(max_error,float((a-b).abs().max()));mismatch+=int((a.argmax(-1)!=b.argmax(-1)).sum())
        if max_error>1e-5 or mismatch:raise RuntimeError('Native cached inference differs')
        result=dict(status='PASS',policy=f'{arm}_seed{seed}',provenance=provenance,checkpoint=artifact(output),
            config=config,unchanged_tensors=len(frozen)-len(changed),cached_states=len(rows),cached_max_abs=max_error,
            argmax_mismatches=mismatch,physical_simulator_bridge_completed=False)
        write(folder/'report.json',result);return result
