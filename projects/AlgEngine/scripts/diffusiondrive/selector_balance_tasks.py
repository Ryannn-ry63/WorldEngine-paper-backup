"""CPU reports/audits and explicit native workers; never implicit training."""
import argparse
import os
from pathlib import Path
import selector_balance_common as b
import selector_balance_native as native


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=('parity', 'audit', 'bridge', 'open', 'report', 'config'))
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--collection', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--policy', choices=b.NEW_NATIVE)
    p.add_argument('--block', choices=('open_navtest', 'open_rare'))
    args = p.parse_args()
    if args.task == 'parity':
        result = native.parity(args.run, args.device)
    elif args.task == 'audit':
        from selector_snapshot_audit import audit
        result = audit(args.collection)
    elif args.task == 'bridge':
        result = native.bridge_report(args.run)
    elif args.task == 'open':
        result = native.openloop(args.run, args.policy, args.block)
    elif args.task == 'report':
        from report_selector_balance import report
        result = report(args.run)
    else:
        from mmcv import Config
        root = Path(os.environ['ALGENGINE_ROOT'])
        cfg = Config.fromfile(str(root/'configs/diffusiondrive/e2e_diffusiondrive_selector_balance_native.py'))
        b.require(cfg.model.planning_head.online_reward is None and not cfg.get('cfpi_deployment_routing')
            and cfg.model.planning_head.candidate_noise_namespace == b.FORMAL_NOISE
            and cfg.selector_rollout_contract.experiment == b.METHOD, 'Native configuration drift')
        _ = cfg.pretty_text
        result = dict(status='PASS')
    print({k:result[k] for k in ('status', 'decision', 'count', 'missing') if k in result}, flush=True)


if __name__ == '__main__':
    main()
