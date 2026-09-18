"""Explicit support masks and independently controlled pair/mass losses."""
import torch
import torch.nn.functional as F


def masked_group(logits, rewards, valid, reduction='all'):
    if logits.shape != rewards.shape or valid.shape != logits.shape:
        raise ValueError('Logit/reward/mask shapes differ')
    valid = valid.bool()
    if not torch.isfinite(logits).all() or not torch.isfinite(rewards[valid]).all():
        raise ValueError('Nonfinite logits or observed rewards')
    n = valid.sum(-1)
    safe = torch.where(valid, rewards, torch.zeros_like(rewards))
    mean = safe.sum(-1)/n.clamp_min(1)
    centered = torch.where(valid, safe-mean[:,None], torch.zeros_like(safe))
    std = (centered.square().sum(-1)/n.clamp_min(1)).sqrt()
    active = (n >= 2) & (std > 1e-6)
    # Empty groups have finite zero gradients; unobserved entries never affect the target.
    masked = logits.masked_fill(~valid, -1e9)
    prob = masked.softmax(-1)*valid
    advantage = centered/std.clamp_min(1e-6)[:,None]
    row = -(prob*advantage).sum(-1)*active
    if reduction == 'active': return row.sum()/active.sum().clamp_min(1)
    if reduction != 'all': raise ValueError('Unknown reduction')
    return row.mean()


def feedback_terms(logits, reference, pairs, outcomes, spec, step=1, steps=1000):
    """pairs: (winner index, loser index, positive gap) or None, one per state."""
    if logits.ndim != 2 or logits.shape[1] != 20 or reference.shape != logits.shape or len(pairs) != len(logits):
        raise ValueError('Require aligned full20 feedback batches')
    zero = logits.sum()*0
    terms = {k:zero for k in ('pair','mass','group_preference','group_return','dpo','mass_anchor')}
    for i, pair in enumerate(pairs):
        if pair is None: continue
        w,l,gap = pair
        if w == l or not 0 <= w < 20 or not 0 <= l < 20 or gap <= 0:
            raise ValueError('Invalid observed preference')
        z = logits[i]; ref = reference[i]
        within = torch.logsumexp(z[[w,l]],0)
        mass_log = within-torch.logsumexp(z,0)
        ref_mass_log = torch.logsumexp(ref[[w,l]],0)-torch.logsumexp(ref,0)
        terms['pair'] = terms['pair']+gap*F.softplus(z[l]-z[w])
        terms['mass'] = terms['mass']-gap*mass_log
        p = z[[w,l]].softmax(0)
        terms['group_preference'] = terms['group_preference']-gap*(p[0]-p[1])
        beta = spec.get('dpo_beta',1.)
        terms['dpo'] = terms['dpo']+gap*F.softplus(-beta*((z[w]-z[l])-(ref[w]-ref[l])))
        # Bernoulli KL preserves queried-set mass without claiming unqueried actions are bad.
        q = ref_mass_log.exp().clamp(1e-7,1-1e-7)
        p_mass = mass_log.exp().clamp(1e-7,1-1e-7)
        terms['mass_anchor'] = terms['mass_anchor']+gap*(q*(q.log()-p_mass.log())+(1-q)*((1-q).log()-(1-p_mass).log()))
    for key in terms: terms[key] = terms[key]/len(logits)
    if spec.get('group_return'):
        reward = torch.zeros_like(logits); valid = torch.zeros_like(logits,dtype=torch.bool)
        if len(outcomes) != len(logits): raise ValueError('Observed return rows differ')
        for i, values in enumerate(outcomes):
            if len(values) != 2 or len({j for j,r in values}) != 2: raise ValueError('Need exactly two observed returns')
            for j,r in values:
                if not 0 <= j < 20: raise ValueError('Candidate index out of bounds')
                valid[i,j] = True; reward[i,j] = float(r)
        terms['group_return'] = masked_group(logits,reward,valid)
    coefficients = dict(spec)
    if 'curriculum' in spec:
        mass_step = (step%2 == 0) if spec['curriculum']=='interleaved' else step > steps//2
        coefficients.update(pair=0. if mass_step else 4., mass=2. if mass_step else 0.)
    loss = sum((float(coefficients.get(k,0))*v for k,v in terms.items()), zero)
    return loss, terms


def teacher_kl(logits, reference):
    q = reference.softmax(-1)
    return (q*(reference.log_softmax(-1)-logits.log_softmax(-1))).sum(-1).mean()


def supervised_selector(logits, trajectories, target, mask):
    if trajectories.shape[:2] != logits.shape or target.shape != trajectories[:,0].shape:
        raise ValueError('Supervised trajectory shapes differ')
    valid = mask.sum(-1)>0
    if not valid.all(): raise ValueError('Missing expert trajectory; never fabricate a target')
    distance = ((trajectories[...,:2]-target[:,None,:,:2]).square().sum(-1)*mask[:,None]).sum(-1)/mask.sum(-1)[:,None]
    return F.cross_entropy(logits, distance.argmin(-1))
