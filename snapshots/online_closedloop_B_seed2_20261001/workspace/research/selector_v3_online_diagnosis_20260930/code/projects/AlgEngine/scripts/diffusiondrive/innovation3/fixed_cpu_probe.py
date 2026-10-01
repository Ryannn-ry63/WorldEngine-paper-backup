"""Real-map CPU acceptance. Analytic candidates are NOT model diagnostics."""
import argparse
import json
import logging
from pathlib import Path
import pickle
import time
import traceback
import numpy as np
from .paths import sha256_file
from .scene_payload import single_scene_mapping
from worldengine.online.headless import HeadlessSimulator,diagnostic_candidates
from worldengine.online.fixed_rollout import FixedRewardSession,rollout,suffix
from worldengine.online.fixed_parallel import FixedBranchPool
from worldengine.online.fixed_diagnostics import compact_result,write_json
from worldengine.online.parallel_branches import _IDMFailures
from worldengine.online.state import structural_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene-index',type=int,default=0)
    p.add_argument('--root',type=int,default=3)
    p.add_argument('--reaction',choices=['R','NR'],default='R')
    p.add_argument('--workers',default='0,2,4,8')
    p.add_argument('--compare-backends',action='store_true')
    p.add_argument('--scorer-parity',action='store_true')
    p.add_argument('--cached-only',action='store_true', help='Bounded v2 integration check on the cached backend')
    args=p.parse_args()
    if args.cached_only and args.compare_backends:
        p.error('Choose cached-only or compare-backends')
    workers=[int(x) for x in args.workers.split(',')]
    if not workers or workers[0]!=0 or any(x<0 or x>20 for x in workers):
        p.error('Start with serial 0, followed by worker counts 1..20')
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.settings.read_text())
    from .fixed_protocol import configured_contract, EXPERT_CONTRACT_NAME
    session_class = FixedRewardSession
    if configured_contract(cfg) == EXPERT_CONTRACT_NAME:
        from worldengine.online.expert_fixed_rollout import ExpertFixedRewardSession
        session_class = ExpertFixedRewardSession
    rows=[r for r in json.loads(args.manifest.read_text())['scenes'] if r['split']=='train']
    row=rows[args.scene_index]
    if sha256_file(Path(row['scene_path']))!=row['scene_sha256']:
        raise ValueError('Scene digest mismatch')
    with Path(row['scene_path']).open('rb') as stream:
        scene=single_scene_mapping(pickle.load(stream),row['scene_id'])[row['scene_id']]
    failures=_IDMFailures();logging.getLogger().addHandler(failures)
    report=dict(status='RUNNING',candidate_source='analytic_engineering_probes_not_DiffusionDrive',
                scene=row,reaction=args.reaction,root=args.root,profiles=[],gpu_used=False)
    started=time.monotonic();sim=None
    try:
        sim=HeadlessSimulator(row['scene_id'],scene,args.reaction,args.root+8,row['scene_seed'])
        session=session_class(sim,cfg['map_root'])
        for _ in range(args.root):session.warmup(session.warmup_action())
        before=sim.snapshot();history=session.history.snapshot();actions=diagnostic_candidates(sim)
        bank_hash=structural_hash(actions);gold=None
        configurations = ([('reference', 0, 'full'), ('cached', 0, 'full')] +
                          [('cached', n, 'boundary') for n in workers]) if args.compare_backends else [
                              ('reference', n, 'full') for n in workers]
        if args.cached_only:
            configurations = [('cached', n, 'boundary') for n in workers]
        gold_states = None
        for backend,n,audit in configurations:
            sim.restore(before);session.history.restore(history)
            sim.set_backend(backend)
            pool=None;t=time.monotonic()
            try:
                if n:
                    pool=FixedBranchPool(row['scene_id'],scene,args.reaction,args.root+8,row['scene_seed'],
                        sim.codec.bundle,session.offset,session.actor_types,workers=n,timeout=900,
                        global_config=sim.engine.global_config,backend=backend)
                result=session.group(actions,12,pool=pool,compare_h1=args.root>=13,audit=audit)
                signature=(result['first_hashes'],result['end_hashes'],result['rewards'],
                    [[f.state_hash for f in b['frames']] for b in result['traces']])
                if gold is None:
                    gold=signature
                    gold_states=[b['state_hashes'] for b in result['traces']]
                elif signature!=gold:raise RuntimeError('Serial/parallel full-segment parity failed')
                if audit=='full' and [b['state_hashes'] for b in result['traces']]!=gold_states:
                    raise RuntimeError('Accelerated intermediate full-state parity failed')
                record=compact_result(result)
                if args.scorer_parity:
                    from worldengine.online.scorer_parity import compare
                    record['same_trace_legacy_parity']=[compare(session,b['frames'],b['route_roadblock_ids']) for b in result['traces']]
                name=('%s_%s_workers_%d.json'%(backend,audit,n)) if args.compare_backends else 'workers_%d.json'%n
                write_json(args.output/name,record)
                report['profiles'].append(dict(workers=n,backend=backend,audit=audit,session_seconds=time.monotonic()-t,
                    startup_seconds=pool.startup_seconds if pool else 0.,**result['profile']))
            finally:
                if pool:pool.close()
        # Independent manual suffix execution, then reverse-order replay.
        manual=[]
        for action in actions:
            sim.restore(before);frames=[session.capture().state_hash];hashes=[]
            for step in range(8):
                sim.step(suffix(action,step),allow_two_point_logplay=True)
                hashes.append(sim.snapshot().state_hash);frames.append(session.capture().state_hash)
            manual.append((hashes,frames))
        if [x[0][0] for x in manual]!=gold[0] or [x[0][-1] for x in manual]!=gold[1] or [x[1] for x in manual]!=gold[3]:
            raise RuntimeError('Independent serial execution differs')
        for i in reversed(range(20)):
            trace=rollout(sim,before,actions[i],session.offset,session.actor_types)
            if trace['end_hash']!=gold[1][i] or [f.state_hash for f in trace['frames']]!=gold[3][i]:
                raise RuntimeError('Candidate ordering changed execution')
        if structural_hash(actions)!=bank_hash or failures.count:
            raise RuntimeError('Candidate mutation or IDM fallback')
        report.update(status='PASS_CPU_FIXED_SEGMENT_PARITY',serial_parallel_all20=True,
            reference_cached_full_states_all20=bool(args.compare_backends),
            independent_serial_all20=True,reverse_order_all20=True,idm_fallbacks=0,
            elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        report.update(status='FAIL',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        if sim:sim.close()
        logging.getLogger().removeHandler(failures)
        write_json(args.output/'report.json',report)
        print(json.dumps(report),flush=True)

if __name__=='__main__':main()
