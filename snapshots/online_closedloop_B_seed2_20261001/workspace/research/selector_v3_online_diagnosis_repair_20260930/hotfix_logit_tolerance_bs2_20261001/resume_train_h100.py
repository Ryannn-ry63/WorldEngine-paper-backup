#!/usr/bin/env python3
"""Use the validated repair runner, locks, deadline and registry with a numeric overlay."""
import argparse
import json
import os
from pathlib import Path
import sys

HOTFIX = Path(__file__).resolve().parent
REPAIR = HOTFIX.parent
OVERLAY = HOTFIX / 'overlay'
sys.path.insert(0, str(REPAIR))
import repair_run as runner


def verify_hotfix():
    manifest = runner.read(HOTFIX / 'manifest.json')
    for name, expected in manifest['files'].items():
        if runner.sha(HOTFIX / name) != expected:
            raise RuntimeError('Hotfix source changed: ' + name)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', choices=('train_B_s2',))
    p.add_argument('--devices', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7'))
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    verify_hotfix()
    registration = runner.registration
    environment = runner.configured_environment
    training = runner.training_job
    write = runner.write
    initial_status_path = runner.REGISTRY / 'jobs' / a.job / 'status.json'
    previous_status = runner.read(initial_status_path) if initial_status_path.exists() else None
    provenance = dict(manifest=str(HOTFIX / 'manifest.json'),
                      sha256=runner.sha(HOTFIX / 'manifest.json'),
                      atol=5e-5, rtol=1e-4,
                      original_source_contract_preserved=True)

    def checked_registration():
        verify_hotfix()
        return registration()

    def patched_environment(*args, **kwargs):
        cfg, env = environment(*args, **kwargs)
        env['PYTHONPATH'] = str(OVERLAY) + os.pathsep + env['PYTHONPATH']
        return cfg, env

    def annotated_write(path, value, **kwargs):
        if Path(path) == initial_status_path:
            value = dict(value, numeric_hotfix=provenance)
        return write(path, value, **kwargs)

    def patched_training(job, folder, devices, deadline, check_only=False):
        if not check_only:
            # Called under the ORIGINAL job.lock by runner.run. Preserve the
            # committed boundary and previous failure evidence before resume.
            write(folder / 'numeric_hotfix.json', provenance, immutable=True)
            write(folder / 'previous_status.json', previous_status, immutable=True)
            progress = runner.REGISTRY / 'jobs' / job['id'] / 'training.progress.json'
            if not progress.exists():
                raise RuntimeError('Hotfix is resume-only; committed progress missing')
            write(folder / 'progress_before_hotfix.json', runner.read(progress), immutable=True)
        return training(job, folder, devices, deadline, check_only)

    runner.registration = checked_registration
    runner.configured_environment = patched_environment
    runner.training_job = patched_training
    runner.write = annotated_write
    # This variable is used only for the runtime entry in training_job.
    # All original experiment, settings, resume and queue contracts remain active.
    runner.TOOLS = OVERLAY
    spec = runner.registration()
    job = next(j for j in spec['jobs'] if j['id'] == a.job)
    result = runner.run(job, a.devices, a.check_only, True)
    return runner.result_exit_code(result, a.check_only)


if __name__ == '__main__':
    raise SystemExit(main())
