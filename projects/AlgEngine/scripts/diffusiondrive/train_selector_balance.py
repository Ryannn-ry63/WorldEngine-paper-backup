"""Only two new objectives, matched shared optimizer/RNG parents, final step500."""
import argparse
from pathlib import Path
import numpy as np
import torch
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_cfpi_model as m
import selector_attribution_common as a
import selector_balance_common as b
import selector_balance_data as data
from selector_rare_model import teacher_kl
from train_selector_feedback import Dense
from train_selector_attribution import restore_state, sample_feedback, gradient_norm


def train(run, arm, seed, device='cuda'):
    run = Path(run)
    b.require(arm in b.NEW_ARMS and seed in b.SEEDS, 'Only the six new training jobs are authorized')
    contract = d.read(run/'run_contract.json')
    audit = d.read(run/'audit_report.json')
    b.require(audit['status'] == 'PASS' and audit['code_sha'] == contract['code_sha'], 'Frozen audit required')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    b.require(not device.startswith('cuda') or torch.cuda.is_available(), 'No CPU fallback for CUDA training')
    inputs = data.checked_inputs(run)
    rows, cache = data.load_rows(run)
    alpha = data.scale(run, seed) if arm == 'MID_SCALE_MATCH' else 1.
    spec = b.specification(arm, alpha)
    parent = inputs['parents'][str(seed)]
    policy = f'{arm}_seed{seed}'
    folder = run/'train'/policy
    provenance = dict(method=b.METHOD, policy=policy, arm=arm, seed=seed,
        code_sha=contract['code_sha'], inputs=d.artifact(run/'balance_inputs.json'),
        data=cache, parent=parent, start=0, end=500, training=b.TRAIN,
        components=spec, scale_audit=d.artifact(run/'scale_audit.json'),
        exposure=b.EXPOSURE, total_optimizer_steps=1000, inference_uses_reward_or_q=False)
    c.locked_json(folder/'contract.json', provenance)
    output = folder/'step_500_report.json'
    if output.exists():
        data.model_entry(run, policy)
        result = d.read(output)
        b.require(result['provenance'] == provenance, 'Existing training identity changed')
        return result
    manifest = d.verified_read(inputs['artifacts']['feedback_inputs'])['artifacts']['scalar_manifest']
    teacher, config, _ = m.load_incumbent(d.verify(manifest))
    teacher.to(device).eval().requires_grad_(False)
    model = m.initialize(teacher, 'residual').to(device).requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    state = torch.load(d.verify(parent), map_location=device)
    b.require(state['step'] == 500 and state['provenance']['arm'] == 'shared'
              and state['provenance']['seed'] == seed, 'Never start from final P1')
    restore_state(state, model, optimizer, rng, device)
    resume = folder/'resume.pt'
    current, history, probes, clip_count = 0, [], [], 0
    if resume.exists():
        state = torch.load(resume, map_location=device)
        b.require(state['provenance'] == provenance and 0 <= state['step'] <= 500
                  and state['step'] % 25 == 0, 'Resume identity/step drift')
        restore_state(state, model, optimizer, rng, device)
        current, history, probes = state['step'], state['history'], state['gradient_probes']
        clip_count = state['clip_count']
    dense = Dense(run)
    model.train()
    for step in range(current+1, 501):
        dense_rows = dense.sample(rng)
        feedback_rows = sample_feedback(rows, rng)
        batch = dense_rows + feedback_rows
        student = m.score(model, batch, list(range(32)), device, 'residual')
        with torch.no_grad():
            reference = m.score(teacher, batch, list(range(32)), device, 'residual')
        rewards = torch.as_tensor(np.stack([r['candidate_rewards'] for r in dense_rows]), dtype=torch.float32, device=device)
        valid = torch.as_tensor(np.stack([r['candidate_reward_valid_mask'] for r in dense_rows]), dtype=torch.bool, device=device)
        grpo = m.v3.exact_group_loss(student[:16], reference[:16], rewards, valid, 1., 0.)[0]
        feedback, pair, nll = b.objective(student[16:], feedback_rows, arm, alpha)
        kl = teacher_kl(student, reference)
        loss = grpo + feedback + .01*kl
        b.require(bool(torch.isfinite(loss)), 'Nonfinite objective')
        if step in (1, 250, 500):
            parameters = tuple(model.parameters())
            norms = {key: gradient_norm(value, parameters) for key,value in dict(
                dense=grpo, pair=alpha*spec['pair_extra']*pair, winner_nll=alpha*nll,
                feedback=feedback, teacher_kl=.01*kl, total=loss).items()}
            b.require(all(np.isfinite(v) for v in norms.values()), 'Nonfinite gradient probe')
            probes.append(dict(step=step, feedback_row_ids=[r['row_id'] for r in feedback_rows],
                weighted_gradient_norms=norms, optimizer_step_taken=False))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(m.v3.clip_grad_norm_cpu_(model.parameters(), 10.))
        b.require(np.isfinite(norm), 'Nonfinite total gradient')
        clip_count += int(norm > 10.)
        optimizer.step()
        if step % 25 == 0:
            history.append(dict(step=step, total=float(loss.detach()), dense=float(grpo.detach()),
                pair=float(pair.detach()), winner_nll=float(nll.detach()), feedback=float(feedback.detach()),
                teacher_kl=float(kl.detach()), gradient_norm_before_clip=norm,
                clip_count=clip_count, clip_fraction=clip_count/step))
            temp = resume.with_suffix('.tmp')
            torch.save(dict(provenance=provenance, step=step, model=model.state_dict(),
                optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.startswith('cuda') else None,
                history=history, gradient_probes=probes, clip_count=clip_count), temp)
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
    b.require(np.isfinite(error) and error <= c.ARRAY_TOLERANCE
              and torch.equal(left.argmax(-1), right.argmax(-1)), 'Export parity failed')
    result = dict(status='PASS', provenance=provenance, step=500, selector=d.artifact(selector),
        resume=d.artifact(resume), history=history, gradient_probes=probes,
        export_max_abs=error, clip_count=clip_count, clip_fraction=clip_count/500,
        trainable_names=sorted(k for k,_ in model.named_parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters()), inference_uses_reward_or_q=False)
    c.locked_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--arm', choices=b.NEW_ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=b.SEEDS, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    train(args.run_root, args.arm, args.seed, args.device)
