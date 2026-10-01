"""Versioned ego execution protocol; no simulator or ML imports."""
DEFAULT_EGO_CONTROLLER = 'log_play_controller'
EXECUTION_CONTRACTS = {
    'log_play_controller': 'upstream_log_play_h1_v1',
    'two_stage_controller': 'online_signed_accel_steering_rate_substeps_v3',
}


def execution_contract(controller=DEFAULT_EGO_CONTROLLER):
    if controller not in EXECUTION_CONTRACTS:
        raise ValueError('Unsupported online ego controller: ' + str(controller))
    return EXECUTION_CONTRACTS[controller]


def validate_resume_execution(payload, expected):
    if payload.get('dynamics_contract') != expected:
        raise ValueError('Resume execution contract mismatch or missing: use a fresh run '
                         'from the offline V3 selector, not a checkpoint from another controller')
