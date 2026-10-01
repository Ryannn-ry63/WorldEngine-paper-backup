import torch
from test_online_learner import model, context
from test_live_learning import identity
from innovation3.fixed_learning import FixedLiveLearning


def _request(online, x, base, ident):
    online.observe(ident)
    return online.choose(x, base, x['candidate_trajectories'][0].tolist())


def _feedback(request, ident, values, before='old', after='new'):
    selected = request['selected']
    reward = lambda x: dict(reward=float(x), NC=1., DAC=1., EP=1., gated_EP=1.,
                            TTC=1., comfort=1., direction=1., progress_m=1.,
                            reference_progress_m=1.)
    main = dict(kind='main', identity=ident, candidate_hash=request['candidate_hash'],
                selected=selected, next_hash='next')
    return main, dict(kind='feedback', identity=ident, candidate_hash=request['candidate_hash'],
        selected=selected, contract='fixed_candidate_4s_logplay_pdm_v1', scored=True,
        main_hash='next', history_before_hash=before, history_after_hash=after,
        first_hashes=['next']*20, end_hashes=['end']*20,
        selected_first_state_parity=True, canonical_restored=True,
        rewards=[reward(x) for x in values])


def test_fixed_feedback_updates_from_all20_segment_rewards():
    torch.set_num_threads(1); torch.manual_seed(8)
    learner = __import__('innovation3.learner', fromlist=['OnlineV3Learner']).OnlineV3Learner(model())
    online = FixedLiveLearning(learner); x=context(); base=torch.randn(1,20)
    ident=identity(); request=_request(online,x,base,ident)
    main, feedback=_feedback(request,ident,[x/19 for x in range(20)])
    online.main_executed(main)
    ack,audit=online.update(feedback)
    assert audit['optimized'] and ack['policy_version']==1


def test_fixed_feedback_rejects_h1_shaped_receipt():
    torch.set_num_threads(1); torch.manual_seed(8)
    learner = __import__('innovation3.learner', fromlist=['OnlineV3Learner']).OnlineV3Learner(model())
    online = FixedLiveLearning(learner); x=context(); base=torch.randn(1,20)
    ident=identity(); request=_request(online,x,base,ident)
    main, feedback=_feedback(request,ident,[1.]*20)
    feedback['contract']='causal_h1_pdm_components_v1'
    online.main_executed(main)
    try:
        online.update(feedback)
    except ValueError:
        return
    raise AssertionError('H1 feedback was accepted by the fixed segment gate')
