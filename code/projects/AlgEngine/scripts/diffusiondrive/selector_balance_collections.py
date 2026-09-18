"""Development routing only; no new acquisition or interventions."""
from pathlib import Path
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_balance_common as b
import selector_balance_data as data
from report_selector_cfpi_deployment import checked_collection


def make(run, cohort, mode, policy):
    run = Path(run)
    b.require((cohort, mode, policy) in b.dev_conditions(), 'Unregistered development condition')
    inputs = data.checked_inputs(run)
    name = f'eval_{cohort}_{mode}_{policy}'
    if name in inputs['reused_dev']:
        return inputs['reused_dev'][name]
    b.require(policy.startswith(b.NEW_ARMS), 'Historical endpoint must never be overwritten')
    folder = run/'collections'/name
    rows = inputs['cohorts'][cohort]['rows']
    routing = dict(status='PASS', method=d.METHOD, research_method=b.METHOD, policy=policy,
        models={policy: data.model_entry(run, policy)}, terminal_publication_decision=12,
        incumbent_selector_sha256=inputs['selector_models']['scalar_v3']['sha256'],
        inference_uses_reward_or_q=False,
        routes=[dict(scene_id=r['scene_id'], origin_token=c.scene_token(r['scene_id']),
                     fold=None, start_decision=4, model_key=policy) for r in sorted(rows, key=lambda r:r['scene_id'])])
    c.locked_json(folder/'routing.json', routing)
    contract = d.read(run/'run_contract.json')
    value = dict(status='PASS', method=d.METHOD, research_method=b.METHOD, collection_id=name,
        cohort=cohort, react_type=mode, policy=policy, continuation_policy=None, role='evaluation',
        sentinel=False, audit_targets={}, initial_contexts={},
        routing=d.artifact(folder/'routing.json'), scenario=inputs['cohorts'][cohort]['scenario'],
        asset_folder=f"{contract['source_worldengine_root']}/data/sim_engine/assets/{inputs['cohorts'][cohort]['asset_family']}/assets",
        inputs=d.artifact(run/'balance_inputs.json'), run_contract=d.artifact(run/'run_contract.json'),
        checkpoint_sha256=c.CHECKPOINT_SHA256, noise_namespace=c.NOISE_NAMESPACE,
        expected_decisions=list(c.DECISION_STEPS), worker_count=4)
    c.locked_json(folder/'deployment_collection.json', value)
    return d.artifact(folder/'deployment_collection.json')


def checked(run, cohort, mode, policy):
    run = Path(run)
    inputs = data.checked_inputs(run)
    name = f'eval_{cohort}_{mode}_{policy}'
    entry = inputs['reused_dev'].get(name)
    if entry is None:
        path = run/'collections'/name/'deployment_collection.json'
        if not path.exists():
            return None
        entry = d.artifact(path)
    result = checked_collection(entry)
    if result is None:
        return None
    co = result['collection']
    routing = d.verified_read(co['routing'])
    b.require(co['cohort'] == cohort and co['react_type'] == mode and co['policy'] == policy
              and co['role'] == 'evaluation' and co['noise_namespace'] == c.NOISE_NAMESPACE
              and co['scenario'] == inputs['cohorts'][cohort]['scenario']
              and routing['models'] == {policy: data.model_entry(run, policy)}
              and not any('feedback_target' in r for r in routing['routes']), 'Development identity drift')
    if name not in inputs['reused_dev']:
        b.require(co['research_method'] == b.METHOD
                  and co['inputs'] == d.artifact(run/'balance_inputs.json')
                  and co['run_contract'] == d.artifact(run/'run_contract.json'), 'Foreign development condition')
    b.validate_metrics(result['metrics'], inputs['cohorts'][cohort]['rows'])
    return result
