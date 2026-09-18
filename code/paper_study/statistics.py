"""Paired log-cluster bootstrap conditional on the frozen optimization seeds."""
import numpy as np

def paired_log_interval(candidate,reference,scene_logs,metric,seed=0,repeats=2000):
    if set(candidate)!=set(reference) or set(candidate)!=set(scene_logs) or not candidate:
        raise ValueError('Paired metrics must have identical nonempty scene membership')
    groups={}
    for scene in sorted(candidate):groups.setdefault(scene_logs[scene],[]).append(candidate[scene][metric]-reference[scene][metric])
    sums=np.array([sum(v) for v in groups.values()]);sizes=np.array([len(v) for v in groups.values()])
    if not np.isfinite(sums).all():raise ValueError('Nonfinite paired differences')
    point=float(sums.sum()/sizes.sum())
    if len(groups)<2:return dict(delta=point,ci95=None,logs=len(groups),reason='Insufficient independent logs')
    rng=np.random.default_rng(seed);draw=rng.integers(len(sums),size=(repeats,len(sums)))
    samples=sums[draw].sum(1)/sizes[draw].sum(1)
    return dict(delta=point,ci95=np.quantile(samples,[.025,.975]).tolist(),logs=len(groups),scenes=int(sizes.sum()),
                method='paired log-cluster percentile bootstrap; fixed optimization seeds',repeats=repeats,seed=seed)
