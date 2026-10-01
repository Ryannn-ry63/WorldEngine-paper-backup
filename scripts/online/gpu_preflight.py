"""Named GPU/import checks with a persistent report, before any training writes."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback
sys.dont_write_bytecode = True
from common import write


def validate(info):
    """Keep the original 8-H100 requirement; explain each mismatch explicitly."""
    errors = []
    if info['device_count'] != 8:
        errors.append('GPU_COUNT: expected 8 visible CUDA devices, got %d (CUDA_VISIBLE_DEVICES=%r)' %
                      (info['device_count'], info.get('cuda_visible_devices')))
    if not info['cuda_available']:
        errors.append('CUDA_UNAVAILABLE: torch.cuda.is_available() is false')
    if len(info['device_names']) != info['device_count']:
        errors.append('GPU_QUERY: could not enumerate every visible device')
    if any('H100' not in name for name in info['device_names']):
        errors.append('GPU_MODEL: expected H100 on every rank, got ' + repr(info['device_names']))
    for key in ('learner_atol', 'fixed_learning_atol'):
        if info[key] != 5e-5:
            errors.append('LOGIT_TOLERANCE: %s must be 5e-5, got %r' % (key, info[key]))
    root = Path(info['code_root']).resolve()
    for name, path in info['module_paths'].items():
        if root not in Path(path).resolve().parents:
            errors.append('IMPORT_SOURCE: %s loaded outside %s: %s' % (name, root, path))
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    info = dict(host=socket.gethostname(), pid=os.getpid(), python=sys.executable,
                code_root=os.environ['WORLDENGINE_ROOT'],
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                nvidia_visible_devices=os.environ.get('NVIDIA_VISIBLE_DEVICES'),
                training_started=False)
    errors = []
    try:
        smi = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader'],
                             text=True, capture_output=True, timeout=20)
        info['nvidia_smi'] = dict(returncode=smi.returncode, stdout=smi.stdout, stderr=smi.stderr)
    except Exception as e:
        info['nvidia_smi'] = dict(error=repr(e))
    try:
        import torch
        info.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                    cuda_available=torch.cuda.is_available(), device_count=torch.cuda.device_count())
        info['device_names'] = [torch.cuda.get_device_name(i) for i in range(info['device_count'])]
        # Record hardware even if a subsequent import fails.
        print(json.dumps(dict(event='GPU_OBSERVED', **info), ensure_ascii=False), flush=True)
        from innovation3 import ddp_online_session, online_probe, learner, fixed_learning
        info.update(learner_atol=learner.LOGIT_DRIFT_ATOL,
                    fixed_learning_atol=fixed_learning.LOGIT_DRIFT_ATOL,
                    module_paths={m.__name__:m.__file__ for m in
                                  (ddp_online_session, online_probe, learner, fixed_learning)})
        errors = validate(info)
    except Exception as e:
        errors.append('PREFLIGHT_EXCEPTION: ' + repr(e))
        info['traceback'] = traceback.format_exc()
    result = dict(status='FAIL_GPU_PREFLIGHT' if errors else 'PASS_8_H100_AND_CLEAN_ALG_IMPORTS',
                  errors=errors, observed=info)
    write(a.report, result, immutable=True)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if errors:
        print('Training was NOT started. Preflight report: ' + str(a.report), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
