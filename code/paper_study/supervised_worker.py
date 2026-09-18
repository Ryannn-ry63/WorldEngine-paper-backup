"""Runs under the planner bootstrap environment; no images loaded to extract expert targets."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import CODE,artifact,lock,read,torch_save,verify,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import mmdet3d_plugin
    m=read(run/'manifest.json')
    with lock(run/'supervised_prepare.lock'):
        if (run/'supervised_inputs.json').exists():
            prior=read(run/'supervised_inputs.json')
            if prior['manifest']!=artifact(run/'manifest.json'):raise RuntimeError('Supervised contract differs')
            for e in [prior['expert_cache'],*prior['configs'].values()]:verify(e)
            return
        cm=read(verify(m['common_caches'][0]['manifest']))
        cache=torch.load(verify(m['common_caches'][0]['cache']),map_location='cpu');tokens=cache['tokens']
        verify(dict(path=cm['annotation_file'],sha256=cm['annotation_sha256']));verify(dict(path=cm['nav_filter'],sha256=cm['nav_filter_sha256']))
        base=dict(path=cm['checkpoint'],sha256=cm['checkpoint_sha256']);verify(base)
        config=Config.fromfile(str(CODE/'projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive.py'))
        data=dict(config.data.train);data.update(ann_file=cm['annotation_file'],nav_filter_path=cm['nav_filter'],pipeline=[])
        dataset=build_dataset(data)
        # Use the dataset's production coordinate/horizon transform, not a hand-written approximation.
        by_token={str(x['token']):i for i,x in enumerate(dataset.data_infos)}
        trajectory=[];masks=[];missing=[]
        for token in tokens:
            if token not in by_token:raise RuntimeError('Missing expert token '+token)
            info=dataset.get_data_info(by_token[token])
            if 'sdc_planning' not in info:
                ann=dataset.get_ann_info(by_token[token]);info=dict(info,**ann)
            t=np.asarray(info['sdc_planning'])[0];mask=np.asarray(info['sdc_planning_mask'])[0]
            if mask.ndim==2:mask=mask[:,0]
            if len(t)!=8:t=t[4::5];mask=mask[4::5]
            if t.shape!=(8,3) or mask.shape!=(8,) or not np.isfinite(t).all():
                raise RuntimeError('Invalid expert trajectory for '+token)
            if not mask.any():missing.append(str(token))
            trajectory.append(t);masks.append(mask)
        target=run/'data/expert.pt'
        torch_save(target,dict(tokens=tokens,trajectory=torch.tensor(np.stack(trajectory),dtype=torch.float32),
                              mask=torch.tensor(np.stack(masks),dtype=torch.float32),source_manifest=m['common_caches'][0]['manifest']))
        import yaml
        nav=yaml.safe_load(Path(cm['nav_filter']).read_text())
        nav['tokens']=[str(t) for t in tokens if str(t) not in set(missing)]
        if not nav['tokens']:raise RuntimeError('No valid expert futures')
        nav_path=run/'data/valid_expert_navtrain.yaml';nav_path.write_text(yaml.safe_dump(nav,sort_keys=False))
        configs={}
        for arm in ('planner_l2','planner_bc'):
            import copy
            cfg=copy.deepcopy(config);cfg.data.train.update(ann_file=cm['annotation_file'],nav_filter_path=str(nav_path))
            cfg.model.update(process_perception=False,freeze_img_backbone=True,freeze_img_neck=True,freeze_bev_encoder=True,freeze_bn=True)
            cfg.custom_imports=dict(imports=['mmdet3d_plugin','paper_study.supervised_plugin'])
            cfg.custom_hooks=[dict(type='PaperFreezePerceptionHook',priority='VERY_HIGH')]
            if arm=='planner_l2':cfg.model.planning_head.type='PaperL2PlanningHead'
            cfg.pop('total_epochs',None);cfg.pop('fp16',None)
            cfg.load_from=base['path'];cfg.optimizer=dict(type='AdamW',lr=1e-4,weight_decay=1e-4,foreach=False)
            cfg.optimizer_config=dict(type='PaperCumulativeOptimizerHook',cumulative_iters=8,grad_clip=dict(max_norm=10.,norm_type=2));cfg.lr_config=dict(policy='Fixed')
            cfg.runner=dict(type='IterBasedRunner',max_iters=19200*8);cfg.checkpoint_config=dict(by_epoch=False,interval=4800*8,max_keep_ckpts=4)
            cfg.log_config=dict(interval=25,hooks=[dict(type='TextLoggerHook')]);cfg.data.samples_per_gpu=2
            cfg.data.workers_per_gpu=2;cfg.workflow=[('train',1)];cfg.find_unused_parameters=True
            destination=run/'configs'/f'{arm}.py';destination.parent.mkdir(parents=True,exist_ok=True);cfg.dump(str(destination));configs[arm]=artifact(destination)
        write(run/'supervised_inputs.json',dict(status='PREPARED_NOT_TRAINED',manifest=artifact(run/'manifest.json'),
            expert_cache=artifact(target),valid_filter=artifact(nav_path),excluded_missing_expert=missing,valid_expert_tokens=len(nav['tokens']),configs=configs,base=base,planner_microbatch=8,accumulation=8,planner_effective_batch=64,selector_batch=64,
            exposure_note='Same source tokens, effective batch and optimizer update count; planner trains generator and scorer, not compute-matched'))
if __name__=='__main__':main()
