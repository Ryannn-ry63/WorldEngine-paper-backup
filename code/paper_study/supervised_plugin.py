"""Planner-only supervised baselines; registered outside the frozen projects tree."""
import torch
from mmcv.runner import HOOKS,Hook,GradientCumulativeOptimizerHook
from mmdet.models.builder import HEADS
from mmdet3d_plugin.navformer.dense_heads.diffusion_planning_head import DiffusionPlanningHead

@HEADS.register_module()
class PaperL2PlanningHead(DiffusionPlanningHead):
    def loss(self,result=None,sdc_planning=None,sdc_planning_mask=None,**kwargs):
        target=sdc_planning[:,0]
        mask=torch.ones(target.shape[:2],device=target.device) if sdc_planning_mask is None else sdc_planning_mask[:,0,:,0].float()
        if target.shape[1]!=self._num_poses:target=target[:,4::5];mask=mask[:,4::5]
        if (mask.sum(-1)==0).any():raise RuntimeError('No valid expert future')
        anchors=result['plan_anchor_expanded']
        distance=(((anchors-target[:,None,:,:2])**2).sum(-1)*mask[:,None]).sum(-1)/mask.sum(-1)[:,None]
        winner=distance.argmin(-1);loss=target.sum()*0
        for prediction in result['poses_reg_list']:
            chosen=prediction[torch.arange(len(target),device=target.device),winner]
            loss=loss+((chosen[...,:2]-target[...,:2]).square()*mask[...,None]).sum()/(2*mask.sum())
        return {'loss.paper_l2':loss/len(result['poses_reg_list'])}

@HOOKS.register_module()
class PaperFreezePerceptionHook(Hook):
    def __init__(self,out_dir=None):
        self.out_dir=out_dir
    def before_run(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        self.frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('planning_head.')}
        for name,p in model.named_parameters():
            p.requires_grad=name.startswith('planning_head.') and not name.endswith('plan_anchor')
        self._eval_frozen(model)
    def _eval_frozen(self,model):
        for name,module in model.named_children():
            if name!='planning_head':module.eval()
    def before_train_iter(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        self._eval_frozen(model)
    def after_run(self,runner):
        model=runner.model.module if hasattr(runner.model,'module') else runner.model
        state=model.state_dict()
        if any(not torch.equal(value,state[k].detach().cpu()) for k,value in self.frozen.items()):
            raise RuntimeError('Supervised baseline changed frozen perception tensors')


@HOOKS.register_module()
class PaperCumulativeOptimizerHook(GradientCumulativeOptimizerHook):
    def clip_grads(self,params):
        from paper_study.common import legacy_imports
        legacy_imports()
        from grpo_selector_v3_cached_common import clip_grad_norm_cpu_
        return clip_grad_norm_cpu_(params,self.grad_clip['max_norm'])
