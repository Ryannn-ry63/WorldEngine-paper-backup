"""Describe historical source membership without relabelling it as an unseen split."""
from collections import Counter
from .common import WORKSPACE,artifact,legacy_imports,read,verify,write

def sources(run):
    legacy_imports()
    from selector_study.data import checked,Dataset
    m=checked(WORKSPACE/'runs/20260916_prefeedback_factorial_v1');ds=Dataset(m)
    dense=read(verify(m['dense_manifest']));excluded=set(read(verify(m['source_inputs']))['excluded_logs'])
    if any(r['origin_log'] in excluded for r in ds.feedback+ds.records):raise RuntimeError('Excluded evaluation log in replay')
    result=dict(status='PASS_HISTORICAL_MEMBERSHIP_AUDIT',source=artifact(WORKSPACE/'runs/20260916_prefeedback_factorial_v1/manifest.json'),
        dense_records=len(ds.records),dense_kinds=dict(Counter(r['kind'] for r in ds.records)),
        dense_logs=len({r['origin_log'] for r in ds.records}),feedback_states=len(ds.feedback),
        feedback_logs=len({r['origin_log'] for r in ds.feedback}),membership=m['membership'],
        visiting_policies=dict(Counter(r['visiting_policy'] for r in ds.feedback)),
        continuation_policies=dict(Counter(r['continuation_policy'] for r in ds.feedback)),
        excluded_logs=sorted(excluded),excluded_log_overlap=False,independent_blind_test=False,
        source_consistent_P1_main_table_ready=False,
        remaining='Audit unused scene assets, freeze log-disjoint development/sealed sets, acquire source-consistent feedback separately')
    write(run/'source_audit.json',result);return result
