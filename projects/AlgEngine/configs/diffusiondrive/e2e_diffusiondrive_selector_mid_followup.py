"""Original routed development inference with explicit balance identity."""
import os
_base_ = ['./e2e_diffusiondrive_selector_feedback.py']
if os.environ.get('DIFFUSIONDRIVE_FEEDBACK_RESEARCH_METHOD') != 'selector_mid_followup_v1':
    raise RuntimeError('Explicit balance identity required')
selector_rollout_contract = dict(research_method='selector_mid_followup_v1')
