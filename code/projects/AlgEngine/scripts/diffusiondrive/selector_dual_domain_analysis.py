"""Offline P1 review. No training, rollout, export or historical report writes."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_attribution_common as a
from selector_rare_common import cluster_interval

METHOD = 'selector_p1_dual_domain_audit_v1'
COHORTS = ('development', 'common')
SAFETY = ('success', 'no_at_fault_collisions', 'drivable_area_compliance')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def output_location(output, roots):
    output = Path(output).resolve()
    for root in (Path(p).resolve() for p in roots):
        require(output != root and root not in output.parents and output not in root.parents,
                'Output must not overlap a historical run')
    require(not output.exists() or not any(output.iterdir()), 'Use a new empty output directory')
    return output


class Evidence:
    """Hash consumed files; recheck file stamps and rehash models on completion."""
    def __init__(self):
        self.entries, self.stamps, self.models = {}, {}, set()

    @staticmethod
    def stamp(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def use(self, item, model=False):
        path = Path(item['path'] if isinstance(item, dict) else item).resolve()
        name = str(path)
        if name not in self.entries:
            self.entries[name] = d.artifact(path)
            self.stamps[name] = self.stamp(path)
        if isinstance(item, dict):
            require(self.entries[name]['sha256'] == item['sha256'], 'Hash mismatch: '+name)
        require(self.stamp(path) == self.stamps[name], 'Input changed: '+name)
        if model:
            self.models.add(name)
        return path

    def read(self, item):
        return d.read(self.use(item))

    def finish(self):
        for name, stamp in self.stamps.items():
            require(self.stamp(Path(name)) == stamp, 'Input modified: '+name)
        for name in sorted(self.models):
            require(c.sha256_file(name) == self.entries[name]['sha256'], 'Model modified: '+name)
        return dict(status='PASS', model_files_rehashed=len(self.models), optimizer_steps=0,
                    model_parameters_updated=False, artifacts=list(self.entries.values()),
                    scope='Consumed artifacts hashed; existing full audits referenced, not rewritten; input stamps rechecked and models rehashed.')


def average(values):
    require(bool(values) and all(set(v) == set(values[0]) for v in values), 'Unpaired seed coverage')
    return {s: {k: float(np.mean([v[s][k] for v in values])) for k in d.METRICS}
            for s in sorted(values[0])}


def summary(values, reference, rows):
    ids = [r['scene_id'] for r in rows]
    require(bool(ids) and set(ids) <= set(values[0]), 'Invalid stratum')
    seeds = [{k: float(np.mean([v[s][k] for s in ids])) for k in d.METRICS} for v in values]
    return dict(scenes=len(ids), logs=len({r['origin_log'] for r in rows}), seed_means=seeds,
                mean={k: float(np.mean([v[k] for v in seeds])) for k in d.METRICS},
                seed_std={k: float(np.std([v[k] for v in seeds], ddof=1)) if len(seeds)>1 else None for k in d.METRICS},
                rescued=[sum(reference[s]['success']==0 and v[s]['success']==1 for s in ids) for v in values],
                broken=[sum(reference[s]['success']==1 and v[s]['success']==0 for s in ids) for v in values])


def select_cases(left, right, rows):
    require(len(left)==len(right)==3, 'Three matched seeds required')
    lm, rm = average(left), average(right)
    require(set(lm)==set(rm)=={r['scene_id'] for r in rows}, 'Case coverage mismatch')
    logs = {r['scene_id']:r['origin_log'] for r in rows}
    reasons = defaultdict(list)
    delta = {s: rm[s]['score']-lm[s]['score'] for s in lm}
    for s in lm:
        if any(left[i][s][k] != right[i][s][k] for i in range(3) for k in SAFETY):
            reasons[s].append('per_seed_safety_disagreement')
    for name, sign in (('CL_NLL_better',1),('P1_better',-1)):
        seen = set()
        for s in sorted(delta, key=lambda s: (-sign*delta[s],s)):
            if sign*delta[s]<=.001 or logs[s] in seen:
                continue
            reasons[s].append('top3_independent_logs_'+name)
            seen.add(logs[s])
            if len(seen)==3:
                break
    return [dict(scene_id=s,origin_log=logs[s],reasons=reasons[s],delta=delta[s]) for s in sorted(reasons)]
