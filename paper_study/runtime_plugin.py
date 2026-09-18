"""Runtime-only architecture adapter for a parameter-matched unary attention control."""
from mmdet.models.builder import HEADS
from mmdet3d_plugin.navformer.dense_heads.diffusion_grpo_online_planning_head import DiffusionGRPOOnlineSelectorPlanningHead
from paper_study.models import DiagonalSetEncoder

@HEADS.register_module()
class PaperUnaryPlanningHead(DiffusionGRPOOnlineSelectorPlanningHead):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.scene_selector.set_encoder=DiagonalSetEncoder(self.scene_selector.set_encoder)
