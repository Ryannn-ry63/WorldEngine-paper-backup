"""CPU-only forwards and autograd probes. Never construct an optimizer."""
from collections import defaultdict
import hashlib
import numpy as np
import torch
import selector_cfpi_model as m
import selector_attribution_common as a
import selector_feedback_common as f
from selector_rare_model import teacher_kl
from selector_dual_domain_analysis import require


def state_hash(model):
    digest = hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def group(row, source='closed'):
    pref = a.preference(row,source)
    if pref is None:
        return 'uninformative'
    wi,li,_ = pref
    outcomes = row['branch_outcomes'] if source=='closed' else row['local_pair_outcomes']
    w,l = outcomes[wi],outcomes[li]
    if any(w[k]>l[k] for k in ('success','no_at_fault_collisions','drivable_area_compliance')):
        return 'safety_improvement'
    return 'both_safe_score_improvement' if w['success']==l['success']==1 else 'both_unsafe'


def decomposition(logits, rows, source='closed'):
    require(logits.shape==(len(rows),20) and bool(torch.isfinite(logits).all()), 'Invalid logits')
    zero = logits.sum()*0
    values = {'pair':[], 'set_mass':[], 'nll':[]}
    strata = defaultdict(list)
    for i,row in enumerate(rows):
        pref = a.preference(row,source)
        if pref is None:
            continue
        wi,li,gap = pref
        w,l = row['candidate_indices'][wi],row['candidate_indices'][li]
        pair = float(gap)*torch.nn.functional.softplus(logits[i,l]-logits[i,w])
        nll = -float(gap)*logits[i].log_softmax(0)[w]
        mass = float(gap)*(logits[i].logsumexp(0)-logits[i,[w,l]].logsumexp(0))
        for key,v in zip(('pair','set_mass','nll'),(pair,mass,nll)):
            values[key].append(v)
        strata[group(row,source)].append(pair+nll)
    result = {k:torch.stack(v).sum()/len(rows) if v else zero for k,v in values.items()}
    result.update({'stratum_'+k:torch.stack(v).sum()/len(rows) for k,v in strata.items()})
    return result


def gradients(terms, model):
    parameters = tuple(model.parameters())
    vectors = {}
    for name,term in terms.items():
        grads = torch.autograd.grad(term,parameters,retain_graph=True,allow_unused=True)
        vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                                   for p,g in zip(parameters,grads)]).double()
    norms = {k:float(v.norm()) for k,v in vectors.items()}
    cosines = {}
    for i,k in enumerate(vectors):
        for j in list(vectors)[i+1:]:
            denom = norms[k]*norms[j]
            cosines[k+'__'+j] = float(torch.dot(vectors[k],vectors[j])/denom) if denom>1e-15 else None
    return dict(losses={k:float(v.detach()) for k,v in terms.items()},norms=norms,cosines=cosines)


def score_rows(model, rows, batch=16):
    with torch.no_grad():
        return torch.cat([m.score(model,rows,list(range(i,min(i+batch,len(rows)))),'cpu','residual')
                          for i in range(0,len(rows),batch)]).numpy()


def training_probes(study, progress):
    rows, ev = study.rows,study.ev
    label_summary = {}
    for mode in a.MODES:
        for source in ('closed','local'):
            groups = defaultdict(list)
            for row in rows:
                if row['mode']==mode:
                    groups[group(row,source)].append(row)
            label_summary[mode+'_'+source] = {key:dict(states=len(rs),logs=len({r['origin_log'] for r in rs}),
                gap_mass=sum(a.preference(r,source)[2] if a.preference(r,source) else 0 for r in rs)) for key,rs in groups.items()}
    feedback_inputs = ev.read(study.inputs['artifacts']['feedback_inputs'])
    manifest = ev.use(feedback_inputs['artifacts']['scalar_manifest'])
    teacher,_,manifest_data = m.load_incumbent(manifest)
    ev.use(dict(path=manifest_data['scene_selector_state'],sha256=manifest_data['scene_selector_state_sha256']),model=True)
    teacher.eval().requires_grad_(False)
    teacher_z = score_rows(teacher,rows)
    # Dense is a read-only mmap loader; only its sample method is used.
    from train_selector_feedback import Dense
    for source in study.dense['sources'].values():
        for entry in source['arrays'].values():
            ev.use(entry)
    dense = Dense(study.run)
    diagnostics, probes = [], []
    for seed in a.SEEDS:
        parent = torch.load(ev.use(study.inputs['parents'][str(seed)],model=True),map_location='cpu')
        require(parent['step']==500 and parent['provenance']['arm']=='shared', 'Wrong parent')
        rng = np.random.default_rng(seed)
        rng.bit_generator.state = parent['rng']
        dense_rows = dense.sample(rng)
        # Fixed exhaustive feedback batches, not a favorable sampled subset.
        selected = [rows[i:i+16] for i in range(0,len(rows),16)]
        models = [('shared',None)]+[(arm,study.models[f'{arm}_seed{seed}']) for arm in a.ARMS]
        for arm,entry in models:
            progress('probe',seed=seed,arm=arm)
            if entry is None:
                model = m.initialize(teacher,'residual')
                model.load_state_dict(parent['model'],strict=True)
            else:
                model,_ = m.load_selector(ev.use(entry,model=True))
            model.eval().requires_grad_(True)
            before = state_hash(model)
            z = score_rows(model,rows)
            for i,row in enumerate(rows):
                shifted = z[i]-z[i].max(); p = np.exp(shifted);p/=p.sum()
                pair_ids = row['candidate_indices']; outside = [k for k in range(20) if k not in pair_ids]
                for source in ('closed','local'):
                    pref = a.preference(row,source)
                    item = dict(arm=arm,seed=seed,row_id=row['row_id'],mode=row['mode'],source=source,
                        stratum=group(row,source),pair_mass=float(p[pair_ids].sum()),outside_max=float(p[outside].max()),
                        selected=int(z[i].argmax()),outside_selected=int(z[i].argmax()) in outside,
                        changed_from_v3=int(z[i].argmax())!=int(teacher_z[i].argmax()))
                    if pref:
                        wi,li,gap=pref;w,l=pair_ids[wi],pair_ids[li]
                        item.update(gap=gap,margin=float(z[i,w]-z[i,l]),pair_correct=bool(z[i,w]>z[i,l]),
                                    winner_selected=int(z[i].argmax())==w,known_loser_selected=int(z[i].argmax())==l)
                    diagnostics.append(item)
            # All shared parents and final P1/CL_NLL/Continue receive exhaustive probes.
            if arm in ('shared','P1','CL_NLL','Continue'):
                for batch_id,feedback in enumerate(selected):
                    batch = dense_rows+feedback
                    student=m.score(model,batch,list(range(len(batch))),'cpu','residual')
                    with torch.no_grad():
                        reference=m.score(teacher,batch,list(range(len(batch))),'cpu','residual')
                    rewards=torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense_rows]),dtype=torch.float32)
                    valid=torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense_rows]),dtype=torch.bool)
                    grpo=m.v3.exact_group_loss(student[:16],reference[:16],rewards,valid,1.,0.)[0]
                    terms=decomposition(student[16:],feedback)
                    require(float((terms['pair']+terms['set_mass']-terms['nll']).detach().abs())<1e-5,'Identity mismatch')
                    terms.update(dense=grpo,kl=.01*teacher_kl(student,reference))
                    probes.append(dict(arm=arm,seed=seed,batch=batch_id,row_ids=[r['row_id'] for r in feedback],
                                       **gradients(terms,model)))
            require(state_hash(model)==before,'Probe updated model')
            require(all(p.grad is None for p in model.parameters()),'Probe accumulated parameter gradients')
            del model
    return dict(status='PASS',labels=label_summary,diagnostics=diagnostics,gradients=probes,optimizer_steps=0,
                note='Eval-mode CPU parameter gradients; exhaustive feedback batches, one fixed dense batch per seed. Direction/scale are diagnostics, not causal proof or independent samples.')
