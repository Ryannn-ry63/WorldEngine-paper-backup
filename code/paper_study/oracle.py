"""Candidate diagnostics from a specified reward cache; never a deployable upper bound."""
import numpy as np
import torch
from .common import artifact, legacy_imports, read, verify, write

def analyze(cache_entry,output):
    legacy_imports()
    from grpo_selector_v3_cached_common import load_cache
    cache,manifest=load_cache(verify(cache_entry))
    r=cache['candidate_rewards'].numpy();valid=cache['candidate_reward_valid_mask'].numpy().astype(bool)
    if not np.isfinite(r[valid]).all():raise RuntimeError('Nonfinite observed rewards')
    selected=cache['reference_logits'].argmax(-1).numpy();n=np.arange(len(r))
    if not valid[n,selected].all() or not valid.any(-1).all():raise RuntimeError('Invalid selected candidate/cache row')
    best=np.where(valid,r,-np.inf).max(-1);chosen=r[n,selected]
    comp=cache['candidate_reward_components'].numpy()
    safe=valid & (comp[:,:,0]==1) & (comp[:,:,1]==1)
    result=dict(status='PASS_CACHE_DIAGNOSTIC',source=cache_entry,manifest=manifest,tokens=len(r),
        baseline_reward=float(chosen.mean()),oracle_reward=float(best.mean()),regret=float((best-chosen).mean()),
        safe_candidate_coverage=float(safe.any(-1).mean()),baseline_safe=float(safe[n,selected].mean()),
        oracle_is_deployable=False,closed_loop_upper_bound=False,official_formal_scores=False,
        interpretation='Cache-local reward; scoring version comes from source manifest. No cross-version score merging.')
    write(output,result);return result


def common_cache_entries():
    from pathlib import Path
    from .common import paths
    root=Path(paths()['legacy_root'])/'experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/cache'
    entries=[]
    for seed in range(3):
        manifest=root/f'train_seed{seed}/manifest.json';m=read(manifest)
        entries.append(dict(manifest=artifact(manifest),cache=dict(path=m['cache'],sha256=m['cache_sha256'])))
    return entries
