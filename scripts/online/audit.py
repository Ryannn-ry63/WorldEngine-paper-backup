"""CPU audit of real all-rank checkpoints and post-migration continuity."""
import argparse
from collections import Counter
from pathlib import Path
import torch
from common import read, write, sha
from innovation3.live_learning import state_digest
from innovation3.ddp_session_boundary import receipt, assignment, validate_round
from innovation3.continuation import load_import


def check_states(records, paths, sources, attempts):
    parities=[]; rng=[]
    for rank,(row,cp) in enumerate(zip(records,paths)):
        if sha(cp['path']) != cp['sha256']:
            raise ValueError('Rank checkpoint changed')
        payload=torch.load(cp['path'],map_location='cpu')
        learner=payload['learner']
        if payload['source_sha256'] != sources or learner['attempts'] != attempts:
            raise ValueError('Checkpoint source or attempts changed')
        if learner['version'] != row['policy_version'] or state_digest(learner) != row['learner_hash']:
            raise ValueError('Complete learner/optimizer/RNG digest changed')
        parities.append({k:state_digest(learner[k]) for k in ('selector','reference','optimizer')})
        rng.append(state_digest(learner['rng']))
    if any(p != parities[0] for p in parities):
        raise ValueError('All-rank selector/reference/optimizer parity failed')
    return dict(shared_states=parities[0],per_rank_rng_sha256=rng,full_learner_digests_verified=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--first-round',action='store_true')
    p.add_argument('--p6-only',action='store_true');a=p.parse_args()
    c=read(a.campaign);path=c['imports'][str(a.seed)]
    imported,parent=load_import(dict(path=path,sha256=c['inputs_sha256'][path]))
    protection=read(c['backup_manifest'])
    for entry in protection['files'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Protected p6 release changed')
    before=check_states(parent['rounds'][-1],imported['checkpoints'],parent['contract']['source_sha256'],904)
    if a.p6_only:
        print('PASS_P6_FULL_STATE',a.seed,before);return
    base=Path(c['run_root'])/('B_s%d'%a.seed);state=read(base/'training.progress.json');spec=state['contract']
    if state['rounds'][:113] != parent['rounds'] or spec['source_sha256'] != imported['new_source_sha256']:
        raise ValueError('Prefix or migrated source changed')
    target=912 if a.first_round else 1200
    if state['completed'] < target or (not a.first_round and state['completed'] != 1200):
        raise ValueError('Target boundary is not complete')
    last=113 if a.first_round else 149
    for index in range(113,last+1):
        records=state['rounds'][index];validate_round(records,spec,index)
        for rank,row in enumerate(records):
            if sha(row['report'])!=row['report_sha256'] or receipt(row['report'],spec,index,rank,state['rounds'][index-1][rank])!=row:
                raise ValueError('Post-import report/continuity failed')
    final=state['rounds'][last]
    after=check_states(final,[r['checkpoint'] for r in final],spec['source_sha256'],target)
    visits=Counter(r['scene'] for rows in state['rounds'][:last+1] for r in rows)
    if not a.first_round and (len(visits)!=150 or set(visits.values())!={8}):
        raise ValueError('Eight-epoch coverage failed')
    out=dict(status='PASS_FIRST_ROUND_CONTINUITY' if a.first_round else 'PASS_P8_ALL_RANKS',
             parent=before,current=after,completed=target,scene_visits=dict(visits),
             preserved_best='B_s2_p6',gpu_round_receipts_verified=True)
    target_path=base/('first_round_gate.json' if a.first_round else 'p8_audit.json')
    if target_path.exists():
        if read(target_path)!=out:raise ValueError('Published audit changed')
    else:
        write(target_path,out,immutable=True)
    print(out['status'],a.seed,target)


if __name__=='__main__':main()
