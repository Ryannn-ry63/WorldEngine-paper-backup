"""CPU-testable receipt validation. No one-step/whole-segment reward equality."""
import math
CONTRACT_NAME='fixed_candidate_4s_logplay_pdm_v1'
EXPERT_CONTRACT_NAME='fixed_candidate_4s_logplay_pdm_expert_v2'
SUPPORTED_CONTRACTS=frozenset((CONTRACT_NAME, EXPERT_CONTRACT_NAME))


def configured_contract(settings):
    name = settings.get('fixed_reward_contract')
    if name not in SUPPORTED_CONTRACTS:
        raise ValueError('Explicit supported fixed_reward_contract required')
    return name


def validate(identity,selected,candidate_hash,main,feedback,scored,history_hash=None,
             expected_contract=CONTRACT_NAME):
    if expected_contract not in SUPPORTED_CONTRACTS:
        raise ValueError('Unknown expected reward contract')
    if main.get('kind')!='main' or feedback.get('kind')!='feedback':
        raise ValueError('Expected canonical main then feedback')
    for message in (main,feedback):
        if message.get('identity')!=identity or message.get('selected')!=selected or message.get('candidate_hash')!=candidate_hash:
            raise ValueError('Stale candidate receipt')
    if feedback.get('contract')!=expected_contract or feedback.get('scored') is not scored:
        raise ValueError('Reward contract/schedule mismatch')
    if not main.get('next_hash') or feedback.get('main_hash')!=main['next_hash']:
        raise ValueError('Canonical hash mismatch')
    if history_hash is not None and history_hash!=feedback.get('history_before_hash'):
        raise ValueError('Canonical reward history discontinuity')
    if not feedback.get('history_after_hash'):raise ValueError('Missing canonical history hash')
    if not scored:return
    if expected_contract == EXPERT_CONTRACT_NAME:
        ref = feedback.get('reference', {})
        if (ref.get('kind') != 'logged_ego_scoring_reference' or ref.get('simulated') is not False
                or ref.get('root_step') != identity['step'] or ref.get('end_step') != identity['step']+8
                or len(ref.get('source_hash', '')) != 64 or len(ref.get('states_hash', '')) != 64
                or 'frames' in ref or 'poses' in ref):
            raise ValueError('Invalid scoring-only expert reference receipt')
    if (len(feedback.get('first_hashes',[]))!=20 or len(feedback.get('end_hashes',[]))!=20
        or feedback['first_hashes'][selected]!=main['next_hash']
        or not feedback.get('selected_first_state_parity') or not feedback.get('canonical_restored')):
        raise ValueError('First-step branch parity failed')
    rewards=feedback.get('rewards',[])
    if len(rewards)!=20:raise ValueError('Expected 20 complete rewards')
    for reward in rewards:
        for key in ('reward','NC','DAC','EP','gated_EP','TTC','comfort','direction','progress_m','reference_progress_m'):
            value=reward.get(key)
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('Invalid segment reward: '+key)
        if not 0<=reward['reward']<=1:raise ValueError('Reward outside unit interval')
