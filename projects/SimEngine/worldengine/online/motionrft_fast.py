"""Optional batched MotionRFT simulator bridge.

This is a fast screening backend, deliberately separate from the exact
WorldEngine closed-loop reward. MotionRFT's JAX batched kinematic/LQR simulator
can evaluate all 20 ego candidates in one call. It does not advance reactive R
NPCs, so its scores are eligible for NR/log-replay screening and top-k proposal
selection only. R acceptance must use fixed_rollout.py's actual IDM branches.
"""
from dataclasses import dataclass
import importlib
import numpy as np

CONTRACT = dict(name='motionrft_batched_ego_screen_v1', dt=.1, horizon_s=4.,
    candidates=20, controller='MotionRFT JAX LQR/kinematic bicycle',
    npc='shared log replay only', reaction='NR_screening_only',
    official_pdms=False, exact_R_reward=False)

@dataclass(frozen=True)
class FastScreenResult:
    states_01s: np.ndarray  # [N,41,11], rear axle
    states_05s: np.ndarray  # [N,9,11], rear axle
    compile_seconds: float
    run_seconds: float
    backend: str = 'MotionRFT'


def _interpolate_candidates(candidates):
    """Convert 8 x 0.5-second local [x,y,heading] to 41 x 0.1 points."""
    x = np.asarray(candidates, dtype=np.float64)
    if x.shape != (20,8,3) or not np.isfinite(x).all():
        raise ValueError('Expected finite 20x8x3 local candidates')
    # Candidate point zero is 0.5s; prepend the current rear-axle origin.
    t_src = np.arange(9, dtype=np.float64)*.5
    t_dst = np.arange(41, dtype=np.float64)*.1
    out = np.empty((20,41,3), dtype=np.float64)
    source = np.concatenate([np.zeros((20,1,3)),x],axis=1)
    for n in range(20):
        out[n,:,0] = np.interp(t_dst,t_src,source[n,:,0])
        out[n,:,1] = np.interp(t_dst,t_src,source[n,:,1])
        # unwrap before interpolation; preserve the candidate endpoint angle.
        out[n,:,2] = np.interp(t_dst,t_src,np.unwrap(source[n,:,2]))
    return out


def run(candidates, initial_ego_state):
    """Run the MotionRFT JAX batch simulator lazily.

    ``initial_ego_state`` is a nuPlan EgoState or a MotionRFT 11-vector. The
    result is only a candidate ego rollout; no logged future is read here.
    """
    try:
        ForwardSimulator = importlib.import_module('numax.simulator.tracker.batch_simulator').ForwardSimulator
    except ImportError as exc:
        raise RuntimeError('MotionRFT runtime is unavailable; use exact WorldEngine rollout') from exc
    import time
    trajectories = _interpolate_candidates(candidates)
    simulator = ForwardSimulator(dt=.1,num_frames=40)
    started=time.monotonic()
    states=simulator.forward(trajectories,initial_ego_state)
    elapsed=time.monotonic()-started
    states=np.asarray(states,dtype=np.float64)
    if states.shape!=(20,41,11) or not np.isfinite(states).all():
        raise ValueError('MotionRFT simulator returned invalid [20,41,11] states')
    return FastScreenResult(states_01s=states,states_05s=states[:,::5].copy(),
        compile_seconds=0.,run_seconds=elapsed)


def rank_with_exact_nr_reward(candidates, initial_ego_state, actor_frames, scorer, reference_progress_m):
    """Fast ego rollout followed by exact segment PDM scoring under shared NR actors.

    ``actor_frames[j]`` contains the actual log-replay actor tuple at each of the
    nine 0.5-second points. It must be detached from the live simulator. This
    function is intentionally rejected for R by the caller.
    """
    if len(actor_frames)!=9 or any(len(f.actors) != len(actor_frames[0].actors) for f in actor_frames):
        raise ValueError('Need nine aligned NR actor frames')
    result=run(candidates,initial_ego_state)
    from .reward import RewardFrame
    rewards=[]
    for i in range(20):
        frames=[]
        for t,state in enumerate(result.states_05s[i]):
            frames.append(RewardFrame(step=actor_frames[t].step,
                ego=np.asarray(state,dtype=np.float64).copy(),actors=actor_frames[t].actors))
        rewards.append(scorer.score(tuple(frames),reference_progress_m))
    return result,rewards
