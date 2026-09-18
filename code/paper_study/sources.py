"""Deterministic draw schedules with source labels retained in the run contract."""
import json
from pathlib import Path
import numpy as np
import torch
from .common import paths, read, verify

FIELDS=('candidate_features','candidate_trajectories_8','route_bev_features','status_tokens','ego_queries','agents_queries',
        'reference_logits','candidate_rewards','candidate_reward_valid_mask')

class CachedPool:
    def __init__(self,manifest):
        from grpo_selector_v3_cached_common import load_cache
        self.caches=[load_cache(verify(e['cache']),'train')[0] for e in manifest['common_caches']]
        self.example=self.caches[0]
        first=self.caches[0]
        for c in self.caches[1:]:
            if c['tokens']!=first['tokens'] or c['scenes']!=first['scenes']:raise RuntimeError('Training cache alignment differs')
        self.indices=np.arange(len(first['tokens']))
        if manifest.get('_supervised_inputs'):
            inp=read(verify(manifest['_supervised_inputs']));expert=torch.load(verify(inp['expert_cache']),map_location='cpu')
            if expert['tokens']!=first['tokens']:raise RuntimeError('Expert cache token alignment changed')
            self.indices=np.flatnonzero(expert['mask'].sum(-1).numpy()>0)
            if not len(self.indices):raise RuntimeError('No valid expert supervision')
        self.orders={}
    def batch(self,seed,step,batch_size,arm):
        n=len(self.indices);batches=(n+batch_size-1)//batch_size
        epoch=(step-1)//(batches*3);within=(step-1)%(batches*3);noise=within//batches;b=within%batches
        key=(seed,epoch,noise)
        if key not in self.orders:
            self.orders={key:np.random.default_rng(np.random.SeedSequence([seed,epoch,noise])).permutation(n)}
        return self.caches[noise],self.indices[self.orders[key][b*batch_size:(b+1)*batch_size]]

class SourcePool:
    def __init__(self,manifest):
        from selector_study.data import Dataset
        from selector_cfpi_model import load_incumbent
        source=manifest['source_contract'];self.ds=Dataset(source)
        _,config,_=load_incumbent(verify(source['incumbent_manifest']))
        self.example=dict(scene_selector_config=config,baseline_selector_state={})
        self.real=[i for i,r in enumerate(self.ds.records) if r['kind']=='real_rare']
        self.mixed=list(range(len(self.ds.records)))
        # Fixed common pool shared by all three arms; deduplicated source+index.
        self.common=sorted({tuple(v) for r in self.ds.records for v in r['common']})
    def batch(self,seed,step,batch_size,arm):
        rng=np.random.default_rng(np.random.SeedSequence([seed,step,903]))
        common=[self.common[i] for i in rng.integers(len(self.common),size=batch_size if arm=='paired_common' else batch_size//2)]
        refs=list(common)
        if arm!='paired_common':
            pool=self.real if arm=='rarelog' else self.mixed
            for ri in rng.choice(pool,size=batch_size//2):
                variants=self.ds.records[int(ri)]['hard'];refs.append(variants[int(rng.integers(len(variants)))])
        cache={k:torch.from_numpy(np.stack([self.ds.arrays[n][k][i] for n,i in refs])) for k in FIELDS}
        return cache,np.arange(batch_size)

_EXPERT={}
def expert_targets(cache,indices,manifest,device):
    entry=manifest.get('expert_cache')
    if entry is None and manifest.get('_supervised_inputs'):
        entry=read(verify(manifest['_supervised_inputs']))['expert_cache']
    if entry is None:raise RuntimeError('Audited expert_cache required for selector SFT; use prepare-supervised')
    key=entry['sha256']
    if key not in _EXPERT:
        data=torch.load(verify(entry),map_location='cpu')
        if data['tokens']!=cache['tokens']:raise RuntimeError('Expert token order differs')
        _EXPERT[key]=data
    data=_EXPERT[key]
    return data['trajectory'][indices].to(device),data['mask'][indices].to(device)
