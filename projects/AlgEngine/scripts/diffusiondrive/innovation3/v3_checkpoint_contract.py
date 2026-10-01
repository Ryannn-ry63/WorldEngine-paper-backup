"""Registered offline V3 selector payload contracts.

The V3 selector family uses a specific method name for the original common
checkpoint and namespaced method names for registered rare/rollout variants.
All accepted names still require schema 3 and are retained in provenance.
"""

BASE_V3_SELECTOR_METHOD = 'scene_conditioned_exact_group_grpo'
REGISTERED_V3_SELECTOR_METHODS = frozenset((
    BASE_V3_SELECTOR_METHOD,
    'scene_conditioned_exact_group_grpo_v3_rare_original_tuned_v1',
    'scene_conditioned_exact_group_grpo_v3_rare_rollout_v1',
))


def is_registered_v3_selector_payload(payload):
    return (isinstance(payload, dict) and
            payload.get('schema_version') == 3 and
            payload.get('method') in REGISTERED_V3_SELECTOR_METHODS)
