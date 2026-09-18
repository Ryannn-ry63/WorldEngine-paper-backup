"""Physical pre-action state checks; privileged fields never enter selector inputs."""
from pathlib import Path
import numpy as np
import cfpi_common as c
from selector_dual_domain_analysis import require

NUMERIC=('can_bus','ego2global_translation','ego2global_rotation','ego_dynamic_state',
         'gt_bboxes_global','gt_velocity')
CATEGORICAL=('scene_name','frame_idx','timestamp','roadblock_ids','traffic_lights','gt_inds','gt_names')


def physical_state(study, sidecar, root, scene, step):
    path=Path(sidecar['raw_observation_path']).resolve()
    require(Path(root).resolve() in path.parents,'Foreign raw observation')
    observation=c.load_pickle(study.ev.use(path))
    require(observation['scene_name']==scene and observation['frame_idx']==step-1,'Wrong pre-action frame')
    return {k:observation[k] for k in (*NUMERIC,*CATEGORICAL)}


def physical_equal(left,right):
    for key in NUMERIC:
        l,r=np.asarray(left[key]),np.asarray(right[key])
        if l.shape!=r.shape or not np.isfinite(l).all() or not np.isfinite(r).all():
            return False
        if not np.allclose(l,r,atol=c.ARRAY_TOLERANCE,rtol=0):
            return False
    return all(np.array_equal(left[k],right[k]) for k in CATEGORICAL)
