import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from .common import read, verify, artifact, write, code_identity, legacy_imports

ARMS = {
    'D1': dict(dense='real', feedback='none'),
    'D2': dict(dense='mixed', feedback='none'),
    'D3': dict(dense='real', feedback='pair_nll'),
    'D4': dict(dense='mixed', feedback='pair_nll'),
}
TRAIN = dict(steps=1000, lr=1e-4, weight_decay=1e-4, grad_clip=10.,
             temperature=1., dense_batch=16, feedback_batch=16, teacher_kl=.01,
             optimizer='AdamW', feedback_from_step=1, initialization='prefeedback_V3_rarelog_seed0',
             reset_optimizer=True, dense_fraction_common=.5)
EXPECTED_PARENT = 'aed5e36f1ccb0df96b703a901b54f48c41dcf1967cc86d63519eb4da37519eb2'

def make_schedule(records, rows, seed, steps):
    """Independent streams preserve exact common/feedback draws across all four arms."""
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    real = [i for i, r in enumerate(records) if r['kind'] == 'real_rare']
    mixed = list(range(len(records)))
    if not real or not any(r['kind'] == 'synthetic_rollout' for r in records):
        raise RuntimeError('Both real and synthetic hard sources are required')
    def draw(pool, group, rng):
        result = []
        for ri in rng.choice(pool, size=8):
            variants = records[int(ri)][group]
            vi = int(rng.integers(len(variants)))
            result.append([int(ri), vi])
        return result
    strata = [[i for i,r in enumerate(rows) if r['mode'] == mode] for mode in ('NR','R')]
    if any(not s for s in strata):
        raise RuntimeError('Both feedback modes required')
    result = []
    for _ in range(steps):
        result.append(dict(common=draw(mixed,'common',streams[0]),
            real=draw(real,'hard',streams[1]), mixed=draw(mixed,'hard',streams[2]),
            feedback=[int(i) for pool in strata for i in streams[3].choice(pool,size=8)]))
    return result

def prepare(run, paths):
    legacy_imports()
    import selector_attribution_data as old
    from selector_feedback_common import fingerprint
    if (run/'manifest.json').exists():
        return checked(run)
    source = Path(paths['attribution_run'])
    source_inputs = read(source/'attribution_inputs.json')
    labels = read(source/'labels/audit.json')
    if source_inputs['status'] != 'PASS' or labels['status'] != 'PASS':
        raise RuntimeError('Original source audit is incomplete')
    if labels['inputs'] != artifact(source/'attribution_inputs.json'):
        raise RuntimeError('Label/source input mismatch')
    feedback_inputs = read(verify(source_inputs['artifacts']['feedback_inputs']))
    incumbent_entry = feedback_inputs['artifacts']['scalar_manifest']
    incumbent = read(verify(incumbent_entry))
    parent = dict(path=incumbent['scene_selector_state'],sha256=incumbent['scene_selector_state_sha256'])
    if parent['sha256'] != EXPECTED_PARENT:
        raise RuntimeError('Must start from original prefeedback V3; shared/P1 parents forbidden')
    verify(parent)
    dense_entry = source_inputs['artifacts']['dense_manifest']
    dense = read(verify(dense_entry))
    rows = pickle.load(verify(labels['cache']).open('rb'))['rows']
    old.validate_rows(rows,source_inputs['excluded_logs'])
    if any(fingerprint(r) != r['source_fingerprint'] for r in rows):
        raise RuntimeError('Feedback state identity changed')
    if dense['status'] != 'PASS' or any(r['origin_log'] in set(source_inputs['excluded_logs']) for r in dense['records']):
        raise RuntimeError('Invalid dense manifest or excluded log overlap')
    snapshots = []
    for name, item in dense['sources'].items():
        for key, entry in item['arrays'].items():
            p = verify(entry)
            arr = np.load(p,mmap_mode='r',allow_pickle=False)
            snapshots.append(dict(entry,source=name,key=key,bytes=p.stat().st_size,
                                  mtime_ns=p.stat().st_mtime_ns,shape=list(arr.shape),dtype=str(arr.dtype)))
        print('verified dense source', name, flush=True)
    pairs = [json.loads(line) for line in verify(dense['artifacts']['pairs']).read_text().splitlines() if line]
    allowed_tokens = {r[k] for r in pairs for k in ('rare_token','common_token')}
    allowed_logs = {r['log_name'] for r in pairs}
    train_scenes = feedback_inputs['cohorts']['train']['rows']
    outside = [r['scene_id'] for r in train_scenes if r['origin_token'] not in allowed_tokens]
    outside_logs = sorted({r['origin_log'] for r in train_scenes} - allowed_logs)
    membership = dict(feedback_scenes=len(train_scenes),feedback_states=len(rows),
        outside_original_pair_tokens=outside,outside_original_pair_logs=outside_logs,
        strict_rarelog_main_table_eligible=False,
        interpretation='Fixed offline F collected by historical V3/shared policies; matched source membership alone does not establish a new on-policy pipeline or held-out evaluation.')
    schedules = {}
    exposure = {}
    for seed in (0,1,2):
        schedule = make_schedule(dense['records'],rows,seed,TRAIN['steps'])
        p = run/'schedules'/f'seed{seed}.json'
        write(p,schedule); schedules[str(seed)] = artifact(p)
        exposure[str(seed)] = {}
        for dense_source in ('real','mixed'):
            selected = [dense['records'][ri] for step in schedule for ri,vi in step[dense_source]]
            exposure[str(seed)][dense_source] = dict(draws=len(selected), kinds=dict(Counter(r['kind'] for r in selected)),
                unique_records=len({ri for step in schedule for ri,vi in step[dense_source]}),
                common_draws=8*TRAIN['steps'],feedback_draws=16*TRAIN['steps'])
    value = dict(schema=1,method='selector_prefeedback_factorial_v1',status='PREPARED',
        code=code_identity(),training=TRAIN,arms=ARMS,seeds=[0,1,2],parent=parent,
        incumbent_manifest=incumbent_entry,feedback_cache=labels['cache'],dense_manifest=dense_entry,
        source_inputs=artifact(source/'attribution_inputs.json'),labels_audit=artifact(source/'labels/audit.json'),
        arrays=snapshots,schedules=schedules,membership=membership,exposure=exposure,
        historical_P1_reused_as_D4=False,independent_acquisition_replicates=False,
        inference_uses_labels=False,old_288_is_blind=False,
        evaluation_status='NOT_RUN; use separately validated native evaluation, never training loss as evidence')
    write(run/'manifest.json',value)
    write(run/'data_audit.json',dict(status='PASS',membership=membership,
        dense_record_kinds=dict(Counter(r['kind'] for r in dense['records'])),
        feedback_visiting_policies=dict(Counter(r['visiting_policy'] for r in rows)),
        feedback_continuation_policies=dict(Counter(r['continuation_policy'] for r in rows)),
        checked_arrays=len(snapshots),bytes_hashed=sum(r['bytes'] for r in snapshots)))
    return value

def checked(run):
    value = read(run/'manifest.json')
    if value['code'] != code_identity() or value['training'] != TRAIN or value['arms'] != ARMS:
        raise RuntimeError('Frozen code/config changed; create a new run ID')
    for field in ('parent','incumbent_manifest','feedback_cache','dense_manifest','source_inputs','labels_audit'):
        verify(value[field])
    for item in value['arrays']:
        p = Path(item['path']); s = p.stat()
        if s.st_size != item['bytes'] or s.st_mtime_ns != item['mtime_ns']:
            raise RuntimeError('Audited array changed; create and audit a new run: '+str(p))
    for item in value['schedules'].values():
        verify(item)
    return value

class Dataset:
    def __init__(self,manifest):
        dense = read(verify(manifest['dense_manifest']))
        self.records = dense['records']
        self.arrays = {n:{k:np.load(e['path'],mmap_mode='r',allow_pickle=False) for k,e in s['arrays'].items()}
                       for n,s in dense['sources'].items()}
        self.feedback = pickle.load(verify(manifest['feedback_cache']).open('rb'))['rows']

    def batch(self,step,arm):
        rows = []
        for key, group in (('common','common'),(ARMS[arm]['dense'],'hard')):
            for ri, vi in step[key]:
                name, offset = self.records[ri][group][vi]
                rows.append({k:np.asarray(v[offset]) for k,v in self.arrays[name].items()})
        feedback = [self.feedback[i] for i in step['feedback']]
        return rows, feedback
