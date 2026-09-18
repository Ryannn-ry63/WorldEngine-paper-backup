"""Read-only historical inputs; new manifests explicitly retain exposure boundaries."""
import json
import pickle
from pathlib import Path
import numpy as np
from .common import CODE, WORKSPACE, artifact, code_identity, legacy_imports, lock, paths, read, verify, write
from .specs import FEEDBACK, STRUCTURES, TRAIN, EXPOSURE

FACTORIAL = '20260916_prefeedback_factorial_v1'

def prepare(run, queue):
    from .storage import ensure_run
    ensure_run(run)
    legacy_imports()
    from selector_study.data import checked
    source = WORKSPACE/'runs'/FACTORIAL
    m = checked(source)
    source_input = read(verify(m['source_inputs']))
    excluded = sorted(set(source_input['excluded_logs']))
    with lock(run/'prepare.lock'):
        result = dict(schema=1, queue=queue, code=code_identity(), source_manifest=artifact(source/'manifest.json'),
            training=dict(TRAIN), exposure=EXPOSURE, excluded_logs=excluded,
            source_contract=m, arms=list(FEEDBACK) if queue=='B' else list(STRUCTURES) if queue=='C' else [],
            evaluation='NOT_RUN', full_pipeline=False)
        if queue in ('C','D','F'):
            root = Path(paths()['legacy_root'])/'experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/cache'
            result['common_caches'] = []
            for seed in range(3):
                p = root/f'train_seed{seed}/manifest.json'; cm = read(p)
                if cm['status']!='PASS' or cm['split']!='train' or cm['noise_seed']!=seed:
                    raise RuntimeError('Unexpected corrected common cache')
                result['common_caches'].append(dict(manifest=artifact(p), cache=dict(path=cm['cache'],sha256=cm['cache_sha256'])))
            result['training'].update(steps=19200, batch_size=64, teacher_kl=.001, checkpoints=[4800,19200],
                                      initialization='immutable_epoch100_zero_residual')
            result['training']['source_note'] = 'C uses historical corrected common; F paired common pool differs'
        if queue=='D':result['arms']=['selector_sft','selector_grpo_matched','planner_l2','planner_bc']
        if queue=='F':
            result['arms']=['paired_common','rarelog','rarerollout']
            result['source_scope']='Paired pool source ablation; no new source-consistent feedback acquisition'
        if queue=='B':
            labels = read(verify(m['labels_audit']))
            local = []
            for item in labels['artifacts']:
                if '/attribution_local_records/' in item['path']:
                    local.append(item)
            if len(local)!=256: raise RuntimeError('Require all 256 same-state local20 labels')
            result['local_records'] = local
        write(run/'manifest.json',result)
    return result


def checked(run):
    m = read(run/'manifest.json')
    if m['code'] != code_identity(): raise RuntimeError('Paper study code changed; freeze a new run')
    legacy_imports()
    from selector_study.data import checked as source_checked
    if source_checked(Path(verify(m['source_manifest'])).parent) != m['source_contract']:
        raise RuntimeError('Frozen factorial source changed')
    return m


def feedback_dataset(manifest):
    legacy_imports()
    from selector_study.data import Dataset
    ds = Dataset(manifest['source_contract'])
    by_id = {}
    for item in manifest.get('local_records',[]):
        row = read(verify(item))
        if row['row_id'] in by_id: raise RuntimeError('Duplicate local20 state')
        by_id[row['row_id']] = row
    for row in ds.feedback:
        r = by_id[row['row_id']]
        if (r['source_fingerprint'] != row['source_fingerprint'] or r['candidate_bank_sha'] != row['candidate_bank_sha']
            or r['candidate_indices'] != row['candidate_indices']):
            raise RuntimeError('Local20 state/candidate identity mismatch')
        scores = np.asarray(r['scores'],dtype=np.float32)
        if scores.shape!=(20,) or not np.isfinite(scores).all(): raise RuntimeError('Invalid local20 rewards')
        row['local20_rewards'] = scores
    return ds


def preferences(rows):
    legacy_imports()
    import selector_attribution_common as a
    pairs=[]; returns=[]
    for row in rows:
        p=a.preference(row,'closed')
        pairs.append(None if p is None else (row['candidate_indices'][p[0]],row['candidate_indices'][p[1]],float(p[2])))
        returns.append([(int(j),float(v['score'])) for j,v in zip(row['candidate_indices'],row['branch_outcomes'])])
    return pairs,returns


def split_logs(rows, excluded, namespace='paper-study-v1', fractions=(.70,.15,.15)):
    import hashlib
    if abs(sum(fractions)-1)>1e-9: raise ValueError('Split fractions must sum to one')
    pools={k:[] for k in ('train','development','sealed')}; seen=set()
    for row in rows:
        if row['origin_log'] in excluded: continue
        key=row['scene_id']
        if key in seen: raise ValueError('Duplicate scene')
        seen.add(key)
        value=int.from_bytes(hashlib.sha256((namespace+':'+row['origin_log']).encode()).digest()[:8],'big')/2**64
        split='train' if value<fractions[0] else 'development' if value<sum(fractions[:2]) else 'sealed'
        pools[split].append(row)
    if any(not v for v in pools.values()): raise RuntimeError('Insufficient log-disjoint scene assets; do not relabel old benchmarks')
    return pools
