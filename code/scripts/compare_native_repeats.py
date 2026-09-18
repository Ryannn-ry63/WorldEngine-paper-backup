"""Compare observed native decisions and initial pixels without changing tolerances.

An incomplete observed prefix can diagnose nondeterminism; it cannot certify a
complete bridge or an evaluation table.
"""
import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from selector_study.common import CODE, artifact, read, verify, write, legacy_imports


def records(root):
    result={}
    for p in sorted(root.glob('split_*/WE_output/openscene_format/snapshot_records/*.json')):
        r=read(p);v=r['validation'];key=(v['scene_id'],v['decision_step'])
        if key in result:raise RuntimeError('Duplicate observed decision')
        result[key]=(p,r)
    return result


def metadata_equal(x,y):
    if isinstance(x,dict) and isinstance(y,dict):
        return x.keys()==y.keys() and all(metadata_equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)) and isinstance(y,(list,tuple)):
        return len(x)==len(y) and all(metadata_equal(a,b) for a,b in zip(x,y))
    if isinstance(x,np.ndarray) or isinstance(y,np.ndarray):return np.array_equal(x,y)
    return bool(x==y)


def compare(left,right,output):
    legacy_imports()
    import cfpi_common as c
    from selector_feedback_common import FINGERPRINT_FIELDS
    from audit_selector_cfpi_deployment import action_error
    left=left.resolve();right=right.resolve();a=records(left);b=records(right)
    lc=read(left/'collection.json');rc=read(right/'collection.json')
    for key in ('checkpoint','noise','mode','rows'):
        if lc[key]!=rc[key]:raise RuntimeError('Repeat input differs: '+key)
    shared=sorted(a.keys()&b.keys())
    if not shared:raise RuntimeError('No shared consumed decisions yet')
    rows=[];images=[]
    for scene,step in shared:
        lp,l=a[(scene,step)];rp,r=b[(scene,step)]
        x=pickle.load(verify(l['sidecar']).open('rb'));y=pickle.load(verify(r['sidecar']).open('rb'))
        values={k:float(np.max(np.abs(np.asarray(x[k])-np.asarray(y[k]))))
                for k in (*FINGERPRINT_FIELDS,'current_logits','deployed_trajectory')}
        if not all(np.isfinite(v) for v in values.values()):raise RuntimeError('Nonfinite comparison')
        action=action_error(l['actual_action'],r['actual_action'])
        pair_equal=(int(x['selected_index'])==int(y['selected_index']))
        passed=(pair_equal and action<=c.ACTION_TOLERANCE
            and values['current_logits']<=c.MODEL_RECOMPUTE_TOLERANCE
            and values['deployed_trajectory']<=c.ACTION_TOLERANCE
            and all(values[k]<=c.ARRAY_TOLERANCE for k in FINGERPRINT_FIELDS))
        rows.append(dict(scene=scene,step=step,left_record=artifact(lp),right_record=artifact(rp),
            selected_left=int(x['selected_index']),selected_right=int(y['selected_index']),
            max_abs=values,action_max_abs=action,within_historical_tolerances=passed))
        if step==min(c.DECISION_STEPS):
            raw=[Path(s['raw_observation_path']) for s in (x,y)]
            frames=[pickle.load(p.open('rb')) for p in raw]
            pixels=[]
            for camera,entry in frames[0]['cams'].items():
                paths=[p.parents[1]/'WE_output/openscene_format/sensor_blobs'/f['cams'][camera]['data_path'] for p,f in zip(raw,frames)]
                arrays=[np.asarray(Image.open(p)).astype(np.int16) for p in paths]
                if arrays[0].shape!=arrays[1].shape:raise RuntimeError('Camera dimensions changed')
                delta=np.abs(arrays[0]-arrays[1])
                pixels.append(dict(camera=camera,files=[artifact(p) for p in paths],
                    max_abs=int(delta.max()),mean_abs=float(delta.mean()),changed_fraction=float((delta!=0).mean())))
            images.append(dict(scene=scene,first_observation_metadata_equal=metadata_equal(*frames),
                observations=[artifact(p) for p in raw],cameras=pixels))
    complete_expected={(row['scene_id'],step) for row in lc['rows'] for step in c.DECISION_STEPS}
    complete=set(shared)==complete_expected
    result=dict(status='WITHIN_TOLERANCE_ON_OBSERVED_PREFIX' if all(r['within_historical_tolerances'] for r in rows) else 'DIFFERENCES_ON_OBSERVED_PREFIX',
        left=artifact(left/'collection.json'),right=artifact(right/'collection.json'),
        analyzer=artifact(__file__),observed_shared_states=len(shared),left_states=len(a),right_states=len(b),
        expected_states=len(complete_expected),complete_decision_coverage=complete,
        selected_mismatches=sum(r['selected_left']!=r['selected_right'] for r in rows),
        tolerance_failures=sum(not r['within_historical_tolerances'] for r in rows),
        tolerances=dict(array=c.ARRAY_TOLERANCE,logits=c.MODEL_RECOMPUTE_TOLERANCE,action=c.ACTION_TOLERANCE),
        first_observations=images,states=rows,formal_acceptance=False,
        conclusion='Observed-prefix diagnostic only. Does not compare complete metrics or certify bridge acceptance; hardware causality requires controlling other environment differences.')
    write(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('states','first_observations')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    compare(a.left,a.right,a.output)
