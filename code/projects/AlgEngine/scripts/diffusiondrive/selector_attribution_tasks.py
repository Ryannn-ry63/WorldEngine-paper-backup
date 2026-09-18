"""Internal CPU tasks; no implicit rollout or training."""
import argparse
from pathlib import Path
import selector_attribution_collections as collections
from report_selector_attribution import report

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('assemble_labels','evaluation_bundle','report'))
    p.add_argument('--run-root',type=Path,required=True)
    args = p.parse_args()
    actions = dict(assemble_labels=collections.assemble_labels,
                   evaluation_bundle=collections.evaluation_bundle,report=report)
    result = actions[args.action](args.run_root)
    print(dict(action=args.action,status=result['status']),flush=True)

