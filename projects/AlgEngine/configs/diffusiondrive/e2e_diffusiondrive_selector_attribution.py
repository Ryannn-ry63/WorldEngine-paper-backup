"""Same frozen forward and controller; labels never enter planner inference."""
import os
_base_ = ['./e2e_diffusiondrive_selector_feedback.py']
if os.environ.get('DIFFUSIONDRIVE_FEEDBACK_RESEARCH_METHOD') != 'selector_p1_attribution_v1':
    raise RuntimeError('Explicit attribution identity required')
selector_rollout_contract = dict(research_method='selector_p1_attribution_v1')
