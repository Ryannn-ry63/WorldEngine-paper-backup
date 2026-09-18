"""Four new controls from frozen shared parents; no reset of optimizer or RNG."""
import argparse
from pathlib import Path
import numpy as np
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_attribution_common as a
import selector_attribution_data as data
from selector_rare_model import teacher_kl
from train_selector_feedback import Dense


def restore_state(state, model, optimizer, rng, device):
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['rng']
    torch.set_rng_state(state['torch_rng'].cpu())
    if device.startswith('cuda'):
        if state['cuda_rng'] is None:
            raise RuntimeError('CUDA RNG missing in parent/resume')
        torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)


def sample_feedback(rows, rng):
    strata = {mode: [r for r in rows if r['mode'] == mode] for mode in a.MODES}
    return [strata[mode][int(i)] for mode in a.MODES for i in rng.integers(len(strata[mode]), size=8)]


def gradient_norm(term, parameters):
    grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum((g.detach().double().square().sum() for g in grads if g is not None),
                                torch.zeros((), dtype=torch.float64, device=term.device))))


def initial_gradient_audit(run):
    """Compare ALL six objectives at matched parents/batches on CPU, without updates."""
    output = run/'labels/initial_gradient_audit.json'
    rows, cache = data.load_rows(run)
    inputs = data.checked_inputs(run)
    provenance = dict(code_sha=d.read(run/'run_contract.json')['code_sha'],
                      inputs=d.artifact(run/'attribution_inputs.json'), data=cache,
                      parents=inputs['parents'], training=a.TRAIN)
    if output.exists():
        result = d.read(output)
        if result['status'] != 'PASS' or result['provenance'] != provenance:
            raise RuntimeError('Initial gradient audit provenance changed')
        return result
    torch.set_num_threads(4)
    manifest = d.verified_read(inputs['artifacts']['feedback_inputs'])['artifacts']['scalar_manifest']
    dense = Dense(run)
    probes = []
    with torch.random.fork_rng(devices=[]):
        teacher, _, _ = m.load_incumbent(d.verify(manifest))
        teacher.eval().requires_grad_(False)
        for seed in a.SEEDS:
            state = torch.load(d.verify(inputs['parents'][str(seed)]), map_location='cpu')
            if (state['step'] != 500 or state['provenance']['arm'] != 'shared'
                    or state['provenance']['seed'] != seed):
                raise RuntimeError('Wrong initial gradient parent')
            model = m.initialize(teacher, 'residual').train().requires_grad_(True)
            model.load_state_dict(state['model'], strict=True)
            torch.set_rng_state(state['torch_rng'].cpu())
            rng = np.random.default_rng(seed)
            rng.bit_generator.state = state['rng']
            dense_rows = dense.sample(rng)
            feedback = sample_feedback(rows, rng)
            batch = dense_rows+feedback
            student = m.score(model,batch,list(range(32)),'cpu','residual')
            with torch.no_grad():
                reference = m.score(teacher,batch,list(range(32)),'cpu','residual')
            rewards = torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense_rows]),dtype=torch.float32)
            valid = torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense_rows]),dtype=torch.bool)
            grpo = m.v3.exact_group_loss(student[:16],reference[:16],rewards,valid,1.,0.)[0]
            kl = .01*teacher_kl(student,reference)
            parameters = tuple(model.parameters())
            for arm in a.ARMS:
                pair,nll = a.feedback_loss(student[16:],feedback,arm)
                terms = dict(dense=grpo,pair=pair,winner_nll=nll,teacher_kl=kl)
                probes.append(dict(seed=seed,arm=arm,
                    feedback_row_ids=[r['row_id'] for r in feedback],
                    weighted_losses={k:float(v.detach()) for k,v in terms.items()},
                    weighted_gradient_norms={k:gradient_norm(v,parameters) for k,v in terms.items()},
                    labels=a.label_summary(feedback)))
    if any(not np.isfinite(v) for p in probes for field in ('weighted_losses','weighted_gradient_norms')
           for v in p[field].values()):
        raise RuntimeError('Nonfinite initial gradient audit')
    result = dict(status='PASS',provenance=provenance,probes=probes,device='cpu',
                  optimizer_steps=0,model_parameters_updated=False,
                  note='Matched initial parent and batch; norms measure scale, not proof of usefulness.')
    c.locked_json(output,result)
    return result


def train(run, arm, seed, device='cuda'):
    if arm not in a.NEW_ARMS or seed not in a.SEEDS:
        raise RuntimeError('Only the four registered new controls may train; historical P0/P1 are immutable')
    audit = d.read(run/'audit_report.json')
    contract = d.read(run/'run_contract.json')
    if audit['status'] != 'PASS' or audit['code_sha'] != contract['code_sha']:
        raise RuntimeError('Frozen implementation audit required')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No silent CPU training fallback')
    inputs = data.checked_inputs(run)
    rows, cache = data.load_rows(run)
    parent = inputs['parents'][str(seed)]
    policy = f'{arm}_seed{seed}'
    folder = run/'train'/policy
    provenance = dict(method=a.METHOD, policy=policy, arm=arm, seed=seed,
                      code_sha=contract['code_sha'], inputs=d.artifact(run/'attribution_inputs.json'),
                      data=cache, parent=parent, start=0, end=500, training=a.TRAIN,
                      components=a.components(arm), exposure=a.EXPOSURE,
                      total_optimizer_steps=1000, inference_uses_reward_or_q=False)
    c.locked_json(folder/'contract.json', provenance)
    output = folder/'step_500_report.json'
    if output.exists():
        report = d.read(output)
        if report['status'] != 'PASS' or report['provenance'] != provenance:
            raise RuntimeError('Existing training report differs')
        d.verify(report['selector']); d.verify(report['resume'])
        return report
    manifest = d.verified_read(inputs['artifacts']['feedback_inputs'])['artifacts']['scalar_manifest']
    teacher, config, _ = m.load_incumbent(d.verify(manifest))
    teacher.to(device).eval().requires_grad_(False)
    model = m.initialize(teacher, 'residual').to(device).requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    state = torch.load(d.verify(parent), map_location=device)
    if state['step'] != 500 or state['provenance']['arm'] != 'shared' or state['provenance']['seed'] != seed:
        raise RuntimeError('Wrong shared parent; never initialize from trained P1')
    restore_state(state, model, optimizer, rng, device)
    resume = folder/'resume.pt'
    current, history, probes = 0, [], []
    if resume.exists():
        state = torch.load(resume, map_location=device)
        if state['provenance'] != provenance or not 0 <= state['step'] <= 500:
            raise RuntimeError('Resume identity/step drift')
        restore_state(state, model, optimizer, rng, device)
        current, history, probes = state['step'], state['history'], state['gradient_probes']
    dense = Dense(run)
    model.train()
    for step in range(current+1, 501):
        dense_rows = dense.sample(rng)
        feedback = sample_feedback(rows, rng)  # also sampled in Continue: same RNG and KL batch
        batch = dense_rows+feedback
        student = m.score(model, batch, list(range(32)), device, 'residual')
        with torch.no_grad():
            reference = m.score(teacher, batch, list(range(32)), device, 'residual')
        rewards = torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense_rows]), dtype=torch.float32, device=device)
        valid = torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense_rows]), dtype=torch.bool, device=device)
        grpo = m.v3.exact_group_loss(student[:16], reference[:16], rewards, valid, 1., 0.)[0]
        pair, nll = a.feedback_loss(student[16:], feedback, arm)
        kl = teacher_kl(student, reference)
        loss = grpo+pair+nll+.01*kl
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite objective')
        if step in (1,500):
            parameters = tuple(model.parameters())
            probes.append(dict(step=step, feedback_row_ids=[r['row_id'] for r in feedback],
                weighted_gradient_norms={k: gradient_norm(v, parameters)
                    for k,v in dict(dense=grpo, pair=pair, winner_nll=nll, teacher_kl=.01*kl).items()},
                local_and_closed_labels=a.label_summary(feedback), optimizer_step_taken=False))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(m.v3.clip_grad_norm_cpu_(model.parameters(), 10.))
        optimizer.step()
        if step % 25 == 0:
            history.append(dict(step=step, total=float(loss.detach()), dense=float(grpo.detach()),
                                pair=float(pair.detach()), winner_nll=float(nll.detach()),
                                teacher_kl=float(kl.detach()), gradient_norm_before_clip=norm))
            temp = resume.with_suffix('.tmp')
            torch.save(dict(provenance=provenance, step=step, model=model.state_dict(),
                            optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                            history=history, gradient_probes=probes), temp)
            temp.replace(resume)
            print(dict(policy=policy, **history[-1]), flush=True)
    model.eval()
    selector = folder/'step_500.pt'
    m.save_selector(selector, model, config, 'residual', provenance)
    restored, _ = m.load_selector(selector)
    restored.to(device).eval()
    with torch.no_grad():
        left = m.score(model, rows[:16], list(range(16)), device, 'residual')
        right = m.score(restored, rows[:16], list(range(16)), device, 'residual')
    error = float((left-right).abs().max())
    if error > c.ARRAY_TOLERANCE or not torch.equal(left.argmax(-1), right.argmax(-1)):
        raise RuntimeError('Export parity failed')
    result = dict(status='PASS', provenance=provenance, step=500, selector=d.artifact(selector),
                  resume=d.artifact(resume), history=history, gradient_probes=probes,
                  export_max_abs=error, trainable_names=sorted(k for k,_ in model.named_parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  inference_uses_reward_or_q=False)
    c.locked_json(output, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--arm', choices=a.NEW_ARMS, required=True)
    p.add_argument('--seed', type=int, choices=a.SEEDS, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    train(args.run_root, args.arm, args.seed, args.device)
