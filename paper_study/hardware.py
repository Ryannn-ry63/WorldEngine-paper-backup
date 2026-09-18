"""GPU admission for new paper queues; CUDA compatibility is checked by preflight."""
import subprocess


def hardware(ids):
    result = []
    for device in ids:
        name = subprocess.check_output(
            ['nvidia-smi', '--id='+device, '--query-gpu=name', '--format=csv,noheader'],
            text=True).strip()
        if not any(model in name for model in ('H100', 'H200')):
            raise RuntimeError('Selected GPU '+device+' must be H100 or H200: '+name)
        result.append(dict(device=device, name=name))
    return result
