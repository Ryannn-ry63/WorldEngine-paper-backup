"""Native full evaluation: no routing, reward/Q inference, or new forward."""
import os
_base_ = ['./e2e_diffusiondrive_grpo_selector_v3.py']
def required(key):
    value = os.environ.get(key, '')
    if not value:
        raise RuntimeError('Balance native evaluation requires ' + key)
    return value
model = dict(planning_head=dict(online_reward=None, kl_weight=0.0,
    scene_selector_score_mode='residual', export_rollout_context=os.getenv('SELECTOR_SNAPSHOT_OPEN') != '1',
    candidate_noise_namespace=required('SELECTOR_SNAPSHOT_NOISE')))
selector_rollout_contract = dict(experiment='selector_mid_followup_v1',
    condition=required('SELECTOR_SNAPSHOT_CONDITION'), react_type=required('SELECTOR_SNAPSHOT_MODE'),
    inference_uses_reward_or_q=False, development_selected_single_checkpoint=False)
selector_reward_contract = dict(_delete_=True, training_performed=False,
    source_method='selector_mid_followup_v1/frozen_endpoint_evaluation',
    generator_frozen=True, inference_uses_reward_or_q=False)
