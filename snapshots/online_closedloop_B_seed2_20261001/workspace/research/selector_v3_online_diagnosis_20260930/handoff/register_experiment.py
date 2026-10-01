#!/usr/bin/env python3
"""Preregister independent jobs; no GPU process is launched by this command."""
from experiment_common import REGISTRY,SETTINGS,read,write,sha,source_hashes
from prepare_scene_pool import build
from evaluation_artifacts import frozen_policies


def main():
    candidates=REGISTRY/'scene_candidates.json'
    if not candidates.exists():build(candidates)
    candidate=read(candidates)
    if candidate['source_sha256']!=source_hashes() or candidate['settings_sha256']!=sha(SETTINGS):
        raise ValueError('Sources/settings changed since registration; do not mix experiment versions')
    frozen=frozen_policies();jobs=[]
    for shard in range(4):jobs.append(dict(id='admit_%d'%shard,kind='admission',shard=shard,shards=4))
    for method in ('A','B'):
        for seed in (0,1,2):
            jobs.append(dict(id='train_%s_s%d'%(method,seed),kind='train',method=method,seed=seed,
                update_timing='per_step' if method=='A' else 'after_rollout8',passes=6,checkpoints=[3,6]))
    for policy in ('parent','online64'):
        for reaction in ('R','NR'):
            jobs.append(dict(id='dev_%s_%s'%(policy,reaction),kind='evaluation',policy=policy,mode='cl_'+reaction,
                dataset='development',interval_seconds=.5,record_plans=True))
            jobs.append(dict(id='interval1_%s_%s'%(policy,reaction),kind='evaluation',policy=policy,mode='cl_'+reaction,
                dataset='development',interval_seconds=1.,record_plans=True,diagnostic_only=True))
    for method in ('A','B'):
        for seed in (0,1,2):
            for passes in (3,6):
                for reaction in ('R','NR'):
                    policy='train_%s_s%d_p%d'%(method,seed,passes)
                    jobs.append(dict(id='dev_'+policy+'_'+reaction,kind='evaluation',policy=policy,mode='cl_'+reaction,
                        dataset='development',interval_seconds=.5,record_plans=True,
                        depends_on_marker='jobs/train_%s_s%d/training.coverage%d.json'%(method,seed,passes)))
    jobs.append(dict(id='roots48',kind='diagnostic',purpose='matched_roots_and_continuations',
                     continuation_policy='frozen_parent_argmax',roots_per_scene=[0,4],scenes_per_log=2))
    for shard in range(4):
        jobs.append(dict(id='roots_extra_%d'%shard,kind='diagnostic_extra',shard=shard,shards=4,depends_on_old64_dev=True))
    for policy in ('parent','selected_s0','selected_s1','selected_s2'):
        for mode in ('ol_navtest','ol_rare','cl_NR','cl_R'):
            jobs.append(dict(id='formal_%s_%s'%(policy,mode),kind='evaluation',policy=policy,mode=mode,
                dataset='formal',interval_seconds=.5,record_plans=False,depends_on_selection=True))
    contract=dict(schema=1,experiment='coverage_timing_v3',status='REGISTERED_NOT_RUN',
        candidate_manifest_sha256=sha(candidates),source_sha256=source_hashes(),settings_sha256=sha(SETTINGS),
        parent=frozen['parent'],online64=frozen['online64'],world_size=8,branch_workers=4,
        deadline_seconds=64800,training_wall_limit_seconds=43200,
        budgets='E(p)=8*ceil(p*N_admitted_train/8); p in {3,6}',
        schedule='seeded_cyclic_global_episode_v2',seeds=[0,1,2],learning_rate=3e-5,kl_weight=.001,
        reward_contract=read(SETTINGS)['fixed_reward_contract'],main_replan_seconds=.5,
        selection=dict(budget_passes=6,R_score_min_delta=.005,NR_score_min_delta=-.005,
                       safety_component_min_delta=-.005,training_seeds=3,tie_break='A',
                       fallback_passes=3,fallback_is_final=False),jobs=jobs)
    path=REGISTRY/'experiment.json'
    if path.exists():
        if read(path)!=contract:raise ValueError('Existing experiment registration differs')
    else:write(path,contract,immutable=True)
    print('REGISTERED_NOT_RUN:',path,'jobs=',len(jobs))
if __name__=='__main__':main()
