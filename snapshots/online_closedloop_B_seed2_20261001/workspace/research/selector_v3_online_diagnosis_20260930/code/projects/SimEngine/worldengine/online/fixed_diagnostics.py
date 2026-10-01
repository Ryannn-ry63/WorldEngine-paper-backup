"""Serialization and diagnostics shared by CPU and frozen visual probes."""
import json
from pathlib import Path
import numpy as np


def frame_record(frame):
    return dict(step=frame.step,ego=frame.ego.tolist(),hash=frame.state_hash,
        actors=[dict(token=a.track_token,type=int(a.tracked_object_type.value),
                     pose=list(a.box.center),size=[a.box.length,a.box.width,a.box.height],
                     velocity=([a.velocity.x,a.velocity.y] if hasattr(a,'velocity') else [0.,0.]))
                for a in frame.actors])


def compact_result(result):
    out = {k:v for k,v in result.items() if k not in ('traces','reference')}
    ref = result['reference']
    out['reference'] = {k:v for k,v in ref.items() if k != 'frames'}
    out['reference']['frames'] = [frame_record(f) for f in ref['frames']]
    out['traces'] = [{**{k:v for k,v in trace.items() if k != 'frames'},
                      'frames':[frame_record(f) for f in trace['frames']]} for trace in result['traces']]
    values = np.array([r['reward'] for r in result['rewards']])
    out['diagnostics'] = dict(all_zero=bool(np.all(values==0)),reward_std=float(values.std()),
        ranking=np.argsort(-values,kind='stable').tolist(),
        ep_saturated_fraction=float(np.mean([r['EP']>=1 for r in result['rewards']])),
        reference_kind=ref.get('kind', 'map_idm_rollout'),
        reference_failed=ref.get('failure_step') is not None,
        delayed_safety_candidates=sum(any(t is not None and t>.5 for k,t in r['first_violation_seconds'].items()
                                         if k in ('NC','DAC')) for r in result['rewards']))
    if result.get('h1_rewards') is not None:
        old = np.array([r['reward'] for r in result['h1_rewards']])
        # Pairwise signs, with ties explicitly counted instead of broken by index.
        i,j = np.triu_indices(len(values),1)
        old_sign,new_sign = np.sign(old[i]-old[j]),np.sign(values[i]-values[j])
        both = (old_sign!=0)&(new_sign!=0)
        out['diagnostics']['h1_comparison'] = dict(
            ranking=np.argsort(-old,kind='stable').tolist(),
            untied_pairs=int(both.sum()),
            pairwise_agreement=float(np.mean(old_sign[both]==new_sign[both])) if both.any() else None,
            old_tied_pairs=int((old_sign==0).sum()),new_tied_pairs=int((new_sign==0).sum()),
            old_all_zero=bool(np.all(old==0)),
            delayed_failures_missed_by_h1=sum(o['NC']==1 and o['DAC']==1 and (n['NC']<1 or n['DAC']<1)
                                            for o,n in zip(result['h1_rewards'],result['rewards'])))
    return out


def write_json(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
