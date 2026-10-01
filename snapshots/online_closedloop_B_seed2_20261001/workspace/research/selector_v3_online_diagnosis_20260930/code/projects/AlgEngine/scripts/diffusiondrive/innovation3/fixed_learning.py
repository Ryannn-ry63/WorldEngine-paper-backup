"""Four-second feedback gate; no invented one-step reward equality."""
from .live_audit import LiveStepGate
from .fixed_protocol import validate, CONTRACT_NAME, SUPPORTED_CONTRACTS
from .live_learning import LiveLearning, state_digest
import time

UPDATE_TIMINGS = ('per_step', 'after_rollout8')


class FixedStepGate(LiveStepGate):
    def __init__(self, policy_version=0, reward_contract=CONTRACT_NAME, update_timing="per_step"):
        super().__init__(policy_version)
        if reward_contract not in SUPPORTED_CONTRACTS:
            raise ValueError('Unknown fixed reward contract')
        self.reward_contract = reward_contract
        if update_timing not in UPDATE_TIMINGS:
            raise ValueError('Unknown update timing')
        self.update_timing = update_timing
        self.decisions = 0

    def feedback(self, branches):
        if self.main is None or self.gate.phase != 'executed':
            raise RuntimeError('Segment feedback requires exactly one canonical transition')
        action = self.gate.action
        validate(self.raw_identity, action.selected, action.candidate_hash,
                 self.main, branches, True, self.history_hash, self.reward_contract)
        self.history_hash = branches['history_after_hash']
        self.gate.phase = 'feedback'

    def updated(self, ack):
        if self.update_timing == 'per_step':
            return super().updated(ack)
        expected = dict(kind='feedback_ack', identity=self.raw_identity,
                        candidate_hash=self.gate.action.candidate_hash,
                        next_hash=self.gate.next_hash, update_timing='after_rollout8')
        fields = {'policy_version', 'optimized', 'optimizer_attempts', 'optimizer_steps'}
        if self.gate.phase != 'feedback' or set(ack) != set(expected) | fields or any(
                ack[k] != v for k, v in expected.items()):
            raise ValueError('Invalid delayed-update acknowledgement')
        attempts = 8 if self.decisions == 7 else 0
        steps = ack['optimizer_steps']
        if (self.decisions >= 8 or type(steps) is not int or not 0 <= steps <= attempts or
                type(ack['optimizer_attempts']) is not int or ack['optimizer_attempts'] != attempts or
                type(ack['optimized']) is not bool or ack['optimized'] != bool(steps) or
                type(ack['policy_version']) is not int or
                ack['policy_version'] != self.gate.policy_version + steps):
            raise ValueError('Delayed update must consume eight groups at the eighth feedback only')
        self.decisions += 1
        self.gate.policy_version = ack['policy_version']
        self.gate.phase = 'ready'
        return dict(kind='updated', identity=self.raw_identity,
                    candidate_hash=self.gate.action.candidate_hash,
                    next_hash=self.gate.next_hash, policy_version=self.gate.policy_version)


class FixedLiveLearning(LiveLearning):
    def __init__(self, learner, reward_contract=CONTRACT_NAME, update_timing='per_step'):
        super().__init__(learner)
        self.update_timing = update_timing
        self.wire = FixedStepGate(learner.version, reward_contract, update_timing)
        self.collected = []
        self.collection_version = learner.version
        self.collection_selector_hash = self.last_selector_hash

    def update(self, branches):
        if self.update_timing == 'per_step':
            return super().update(branches)
        # Reward generation is unchanged. A collection acknowledgement never
        # changes the policy; the eighth acknowledgement commits eight updates.
        import torch
        self.wire.feedback(branches)
        self.times['feedback'] = time.monotonic_ns()
        learner = self.learner
        if learner.version != self.collection_version or self.before_hash != self.collection_selector_hash:
            raise RuntimeError('Collection policy changed inside an eight-step rollout')
        identity = self.wire.gate.identity
        rewards = [[r['reward'] for r in branches['rewards']]]
        reward_tensor = torch.as_tensor(rewards, dtype=self.base.dtype, device=self.base.device)
        if reward_tensor.shape != (1, 20) or not torch.isfinite(reward_tensor).all():
            raise ValueError('Complete finite full20 rewards required')
        if learner.pending is None or learner.pending[0] != identity:
            raise RuntimeError('Missing collection decision')
        # Discard the collection graph. No old logits or RNG draws are reused
        # for learning. Each cached context is evaluated with current weights.
        learner.pending = None
        self.collected.append(dict(identity=identity, context={k:v.detach().cpu().clone()
            for k,v in self.context.items()}, base=self.base.detach().cpu().clone(), rewards=rewards,
            context_hash=self.context_hash))
        optimizer_before = state_digest(learner.optimizer.state_dict())
        updates = []
        if len(self.collected) == 8:
            parameter = next(learner.selector.parameters())
            for order, group in enumerate(self.collected):
                context = {k:v.to(parameter.device) for k,v in group['context'].items()}
                base = group['base'].to(parameter.device)
                if state_digest(context) != group['context_hash']:
                    raise RuntimeError('Cached context changed')
                with torch.no_grad():
                    frozen = learner.reference_logits(base, context)
                    before_logits = learner.current_logits(base, context, frozen)
                logits = learner.current_logits(base, context, frozen)
                learner.logit_drift = float((before_logits-logits.detach()).abs().max())
                if not torch.allclose(before_logits, logits.detach(), atol=1e-5, rtol=1e-4):
                    raise RuntimeError('Replay training/inference logits disagree')
                version_before = learner.version
                learner.pending = (group['identity'], logits, frozen)
                update = learner.update(group['rewards'], group['identity'])
                with torch.no_grad():
                    after_logits = learner.current_logits(base, context, frozen)
                updates.append(dict(update, version_before=version_before, version_after=learner.version,
                    collection_policy_version=self.collection_version, collection_index=order,
                    context_hash=group['context_hash'],
                    same_context_logit_change_after_update=float((after_logits-before_logits).abs().max())))
            self.collected.clear()
        after_hash = state_digest(learner.selector.state_dict())
        optimizer_after = state_digest(learner.optimizer.state_dict())
        steps = sum(u['optimized'] for u in updates)
        if not steps and (after_hash != self.before_hash or optimizer_before != optimizer_after):
            raise RuntimeError('Collection/no-signal boundary changed model or optimizer')
        if state_digest(learner.reference.state_dict()) != self.reference_hash or any(
                p.grad is not None or p.requires_grad for p in learner.reference.parameters()):
            raise RuntimeError('Frozen reference was changed')
        if state_digest(self.context) != self.context_hash:
            raise RuntimeError('Live context mutated')
        with torch.no_grad():
            after_logits = learner.current_logits(self.base, self.context, self.frozen)
        if not torch.isfinite(after_logits).all():
            raise RuntimeError('Non-finite post-update logits')
        self.last_selector_hash = self.last_residual_hash = after_hash
        self.times['update_complete'] = time.monotonic_ns()
        ack = dict(kind='feedback_ack', identity=self.wire.raw_identity,
            candidate_hash=self.wire.gate.action.candidate_hash, next_hash=self.wire.gate.next_hash,
            policy_version=learner.version, optimized=bool(steps), update_timing=self.update_timing,
            optimizer_attempts=len(updates), optimizer_steps=steps)
        self.awaiting_ack = self.wire.updated(ack)
        audit = dict(self.action_audit, update_timing=self.update_timing, optimizer_updates=updates,
            update=dict(policy_version=learner.version, optimized=bool(steps), attempts=learner.attempts,
                reward_std=float(reward_tensor.std(unbiased=False)),
                local_group_signal=bool(float(reward_tensor.std(unbiased=False)) > 1e-6)),
            optimizer_attempt_count=len(updates), optimizer_step_count=steps,
            online_selector_hash_after=after_hash, residual_hash_after=after_hash,
            parameterization=learner.parameterization, optimizer_hash_before=optimizer_before,
            optimizer_hash_after=optimizer_after, post_update_logits=after_logits[0].tolist(),
            same_context_logit_change_after_update=float((after_logits-self.before_logits).abs().max()),
            version_after=learner.version, optimized=bool(steps), times_ns=dict(self.times))
        self.context = None
        return ack, audit
