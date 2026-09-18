"""Pre-registered comparisons. Changes require a new run, never a relabelled old arm."""
SEEDS = (0, 1, 2)
FEEDBACK = {
    'pair': dict(pair=1.),
    'nll': dict(pair=1., mass=1.),
    'pair_nll': dict(pair=2., mass=1.),
    'mid': dict(pair=1.5, mass=1.),
    'set_mass': dict(mass=1.),
    'group_preference': dict(group_preference=1.),
    'group_return': dict(group_return=1.),
    'dpo': dict(dpo=1., dpo_beta=1.),
    'local20': dict(local20=1.),
    'dense_no_feedback_kl': dict(feedback_kl=False),
    'pair_nll_mass_anchor': dict(pair=2., mass_anchor=1.),
    'pair_nll_interleaved': dict(pair=1., mass=1., curriculum='interleaved'),
    'pair_nll_concentrated': dict(pair=1., mass=1., curriculum='concentrated'),
}
STRUCTURES = {
    'full': {},
    'feature_only': dict(use_trajectory_geometry=False, use_route_bev=False,
                         use_scene_context=False, use_set_attention=False),
    'no_geometry': dict(use_trajectory_geometry=False),
    'no_route_bev': dict(use_route_bev=False),
    'no_scene': dict(use_scene_context=False),
    'no_set': dict(use_set_attention=False),
    'unary_matched': dict(use_set_attention=False, paper_unary_matched=True),
    'original_selector': dict(paper_original_selector=True),
}
QUEUES = {
 'A':dict(name='evaluation_oracle', actions=['prepare-eval','openloop','report','oracle']),
 'B':dict(name='feedback_objectives', arms=list(FEEDBACK), seeds=list(SEEDS), steps=1000),
 'C':dict(name='selector_structure', arms=list(STRUCTURES), seeds=list(SEEDS), steps=19200, checkpoints=[4800,19200]),
 'D':dict(name='supervised', arms=['selector_sft','selector_grpo_matched','planner_l2','planner_bc'], seeds=list(SEEDS), steps=19200),
 'E':dict(name='full_candidate_mechanism', queries=20, candidate_subsets=[2,4,8,20],
          requires=['audited_train_development_manifest','source_state_and_continuation_identity']),
 'F':dict(name='source_baselines', regimes=['paired_common','rarelog','rarerollout'], seeds=list(SEEDS), steps=19200,
          checkpoints=[4800,19200], feedback='separate source-consistent acquisition'),
}
TRAIN = dict(steps=1000, lr=1e-4, weight_decay=1e-4, grad_clip=10., temperature=1., teacher_kl=.01,
             dense_batch=16, feedback_batch=16, initialization='original_prefeedback_V3_rarelog_seed0')
EXPOSURE = dict(independent_pipeline_replicates=False, independent_blind_test=False,
                historical_benchmarks_exposed=True, optimization_seeds=[0,1,2], acquisition_seed=0)
