#!/usr/bin/env python3
"""B p6 -> p8 continuation and exact-coverage six-stage evaluation per seed."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid
sys.dont_write_bytecode = True
from common import read, write, sha, directory_lock, archive_tree
from eval_runtime import environment, CODE
from acceptance import accept_stage

TOOLS = CODE / 'projects/AlgEngine/scripts/diffusiondrive'
MODES = ('cl_R', 'cl_NR', 'ol_rare', 'ol_navtest')


def verify_files(files):
    for path, digest in files.items():
        if sha(path) != digest:
            raise ValueError('Frozen input changed: ' + str(path))


def checked_campaign(path, devices):
    campaign = read(path)
    if len(devices.split(',')) != 8 or len(set(devices.split(','))) != 8:
        raise ValueError('Exactly eight distinct devices are required')
    verify_files({str(CODE / p): h for p, h in campaign['code_sha256'].items()})
    verify_files(campaign['inputs_sha256'])
    cfg, env = environment(campaign['settings'], devices)
    stop = cfg.get('campaign_stop_file')
    if stop and Path(stop).exists():
        raise RuntimeError('Correctness stop is active: ' + stop)
    for key, value in cfg.items():
        if key in ('campaign_stop_file', 'campaign_deadline_file'):
            continue
        if isinstance(value, str) and value.startswith('/') and not Path(value).exists():
            raise FileNotFoundError(key + ': ' + value)
    for value in cfg.get('extra_python_paths', []):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    verify_files({cfg[key]: digest for key, digest in cfg.get('expected_sha256', {}).items()})
    return campaign, cfg, env


def command(argv, env):
    print(json.dumps(dict(argv=list(map(str, argv))), ensure_ascii=False), flush=True)
    subprocess.run(list(map(str, argv)), env=env, cwd=CODE, check=True)


def gpu_preflight(campaign, cfg, env):
    command([cfg['algengine_python'], '-c',
        'import torch; from pathlib import Path; import os; '
        'from innovation3 import ddp_online_session, online_probe, learner, fixed_learning; '
        'assert torch.cuda.device_count()==8; '
        'assert all("H100" in torch.cuda.get_device_name(i) for i in range(8)); '
        'assert learner.LOGIT_DRIFT_ATOL==5e-5 and fixed_learning.LOGIT_DRIFT_ATOL==5e-5; '
        'assert Path(os.environ["WORLDENGINE_ROOT"]) in Path(learner.__file__).parents; '
        'print("PASS_8_H100_AND_CLEAN_ALG_IMPORTS")'], env)
    command([cfg['simengine_python'], '-c',
        'from innovation3 import fixed_worker; from worldengine.online import scorer_parity; '
        'print("PASS_SIM_IMPORTS", fixed_worker.__file__)'], env)
    scratch_base(campaign)


def scratch_base(campaign):
    p = Path(os.environ.get('EVAL_SCRATCH_ROOT', '/tmp/worldengine_online_p8')).absolute()
    p.mkdir(parents=True, exist_ok=True)
    if p.is_symlink() or p.stat().st_dev == Path(campaign['run_root']).stat().st_dev:
        raise RuntimeError('EVAL_SCRATCH_ROOT must be on node-local disk')
    if shutil.disk_usage(p).free < 80 * 2**30:
        raise RuntimeError('Need 80 GiB free node-local scratch')
    if shutil.disk_usage(campaign['run_root']).free < 64 * 2**30:
        raise RuntimeError('Need 64 GiB free shared storage')
    return p


def training_argv(campaign, cfg, seed, progress_exists, gate=False):
    base = Path(campaign['run_root']) / ('B_s%d' % seed)
    argv = [cfg['algengine_python'], '-u', '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node', '8', '-m', 'innovation3.ddp_online_session',
            '--settings', campaign['settings'], '--scene-manifest', campaign['scene_manifest'],
            '--output', base / 'training', '--episodes', '1200', '--steps', '8',
            '--seed', str(seed), '--split', 'train', '--branch-workers', '4',
            '--fixed-segment', '--wall-seconds', '43200', '--update-timing', 'after_rollout8']
    argv += ['--resume'] if progress_exists else ['--continuation-manifest', campaign['imports'][str(seed)]]
    if gate:
        argv += ['--stop-after-rounds', '1']
    return argv


def audit(campaign, cfg, env, seed, gate=False):
    argv = [cfg['algengine_python'], Path(__file__).with_name('audit.py'),
            '--campaign', campaign['_path'], '--seed', seed]
    if gate:
        argv += ['--first-round']
    command(argv, env)


def train(campaign, cfg, env, seed):
    base = Path(campaign['run_root']) / ('B_s%d' % seed)
    base.mkdir(parents=True, exist_ok=True)
    progress = base / 'training.progress.json'
    if not (base / 'first_round_gate.json').exists():
        if not progress.exists() or read(progress)['completed'] == 904:
            command(training_argv(campaign, cfg, seed, progress.exists(), gate=True), env)
        audit(campaign, cfg, env, seed, gate=True)
    if read(progress)['completed'] < 1200:
        command(training_argv(campaign, cfg, seed, True), env)
    if read(progress)['completed'] != 1200:
        raise RuntimeError('Training stopped at a committed boundary; inspect status before resuming')
    audit(campaign, cfg, env, seed)


def materialize(campaign, cfg, env, seed):
    base = Path(campaign['run_root']) / ('B_s%d' % seed)
    prepared = base / 'p8/prepared.json'
    if prepared.exists():
        p = read(prepared)
        verify_files({p[k]: p[h] for k,h in [('path','sha256'), ('selector','selector_sha256'), ('marker','marker_sha256')]})
        return p
    marker = base / 'training.coverage8.json'
    m = read(marker)
    if m['completed'] != 1200 or set(m['coverage']['scene_visits'].values()) != {8}:
        raise ValueError('Export requires eight complete visits')
    checkpoint = m['final_records'][0]['checkpoint']
    verify_files({checkpoint['path']: checkpoint['sha256']})
    attempt = base / 'p8' / ('attempt_' + uuid.uuid4().hex)
    attempt.mkdir(parents=True)
    selector, target = attempt / 'selector.pt', attempt / 'checkpoint.pth'
    cpu = dict(env, CUDA_VISIBLE_DEVICES='')
    command([cfg['algengine_python'], '-m', 'innovation3.export_online_selector_v3',
        '--checkpoint', checkpoint['path'], '--offline-selector', cfg['selector_state'],
        '--output', selector, '--manifest', attempt / 'export.json'], cpu)
    command([cfg['algengine_python'], TOOLS / 'materialize_grpo_selector_v3.py',
        '--baseline', cfg['baseline'], '--scene-selector-state', selector,
        '--expected-selector-sha256', sha(selector), '--expected-method', 'scene_conditioned_exact_group_grpo',
        '--release-name', 'B_s%d_p8' % seed, '--output', target, '--manifest', attempt / 'materialize.json'], cpu)
    command([cfg['algengine_python'], TOOLS / 'audit_grpo_selector_v3_checkpoint.py',
        '--baseline', cfg['baseline'], '--checkpoint', target, '--output', attempt / 'frozen_baseline_audit.json'], cpu)
    p = dict(path=str(target), sha256=sha(target), selector=str(selector), selector_sha256=sha(selector),
             marker=str(marker), marker_sha256=sha(marker))
    write(prepared, p, immutable=True)
    return p


def evaluation_command(campaign, cfg, devices, artifact, dataset, mode, output):
    data = campaign[dataset]
    argv = [cfg['algengine_python'], '-u', Path(__file__).with_name('eval_runtime.py'),
            '--settings', campaign['settings'], '--devices', devices,
            'openloop' if mode.startswith('ol_') else 'closedloop',
            '--config', campaign['eval_config'], '--checkpoint', artifact['path'], '--output', output,
            '--eval-seed', '0', '--timeout', '86400', '--diagnostic-interval', '0.5', '--record-plans']
    if mode.startswith('ol_'):
        argv += ['--split', 'navtest' if mode == 'ol_navtest' else 'rare']
    else:
        argv += ['--reaction', mode[3:], '--scenarios', data['scenarios'], '--assets', data['assets']]
    return argv


def accepted_result(path):
    result = read(path)
    verify_files(result['files_sha256_by_path'])
    accepted = accept_stage(result['output'], result['mode'], result['expected_tokens'])
    if accepted['rows'] != result['rows'] or accepted['means'] != result['means']:
        raise ValueError('Accepted evaluation changed')
    return result


def evaluate(campaign, cfg, env, devices, seed):
    audit(campaign, cfg, env, seed)
    artifact = materialize(campaign, cfg, env, seed)
    for dataset, modes in [('development', MODES[:2]), ('formal', MODES)]:
        for mode in modes:
            folder = Path(campaign['run_root']) / ('B_s%d' % seed) / 'evaluation' / (dataset + '_' + mode)
            folder.mkdir(parents=True, exist_ok=True)
            done = folder / 'accepted.json'
            if done.exists():
                accepted_result(done)
                continue
            attempt = folder / ('attempt_' + uuid.uuid4().hex)
            attempt.mkdir()
            local = scratch_base(campaign) / ('B_s%d_' % seed + uuid.uuid4().hex)
            local.mkdir()
            output, shared = local / 'evaluation', attempt / 'evaluation'
            argv = evaluation_command(campaign, cfg, devices, artifact, dataset, mode, output)
            write(attempt / 'inputs.json', dict(argv=list(map(str,argv)), artifact=artifact, dataset=dataset,
                  mode=mode, host=socket.gethostname(), scratch=str(local)), immutable=True)
            childenv = dict(env)
            with socket.socket() as sock:
                sock.bind(('', 0)); childenv['MASTER_PORT'] = str(sock.getsockname()[1])
            try:
                command(argv, childenv)
            except BaseException:
                archive_tree(output, shared, failures_only=True)
                raise
            checked_campaign(campaign['_path'], devices)
            files = archive_tree(output, shared)
            files += archive_tree(local / 'evaluation_inputs', attempt / 'evaluation_inputs')
            write(attempt / 'archive.json', files, immutable=True)
            expected = campaign[dataset]['expected_tokens'][mode]
            result = accept_stage(shared, mode, expected)
            result.update(output=str(shared), dataset=dataset, mode=mode, seed=seed,
                          expected_tokens=expected, artifact=artifact,
                          files_sha256_by_path={p:result['files_sha256'][k] for k,p in result['files'].items()})
            write(done, result, immutable=True)
            shutil.rmtree(local)  # only this attempt, after shared exact-coverage acceptance


def report(campaign, cfg, env, seed):
    command([cfg['algengine_python'], Path(__file__).with_name('curves.py'),
             '--progress', Path(campaign['run_root']) / ('B_s%d/training.progress.json' % seed),
             '--output', Path(campaign['run_root']) / ('B_s%d/report' % seed)], dict(env, CUDA_VISIBLE_DEVICES=''))
    rows = []
    for path in sorted((Path(campaign['run_root']) / ('B_s%d/evaluation' % seed)).glob('*/accepted.json')):
        r = accepted_result(path)
        row = dict(dataset=r['dataset'], mode=r['mode'], count=r['counts'],
                   p8={k:100*v for k,v in r['means'].items()})
        if r['dataset']=='formal':
            for label in ('parent', 'B_s%d_p6' % seed):
                old = campaign['p6_formal_scores'][label][r['mode']]
                row[label] = old
                row['delta_vs_' + label] = {k:row['p8'][k]-old[k] for k in old}
        rows.append(row)
    write(Path(campaign['run_root']) / ('B_s%d/report/evaluation.json' % seed),
          dict(seed=seed, rows=rows, best_verified_p6_unchanged=True, p8_not_automatically_selected=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check', 'dry-run', 'train', 'eval', 'report', 'pipeline', 'verify'))
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=(0,1,2), required=True)
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    a = p.parse_args()
    campaign, cfg, env = checked_campaign(a.campaign, a.devices)
    campaign['_path'] = str(a.campaign.resolve())
    if a.action == 'check':
        print('PASS_CPU_PATH_SOURCE_AND_INPUT_CHECK; GPU validation pending')
        return
    if a.action == 'dry-run':
        print(json.dumps(dict(first_round=list(map(str, training_argv(campaign,cfg,a.seed,False,True))),
            remaining=list(map(str,training_argv(campaign,cfg,a.seed,True))),
            evaluations=[dict(dataset=d,mode=m,count=len(campaign[d]['expected_tokens'][m]))
                for d,modes in [('development',MODES[:2]),('formal',MODES)] for m in modes]), indent=2))
        return
    if a.action == 'verify':
        audit(campaign,cfg,env,a.seed)
        return
    runroot=Path(campaign['run_root'])
    with directory_lock(runroot / 'nodes' / (socket.gethostname()+'.lock')):
        with directory_lock(runroot / ('B_s%d' % a.seed) / 'pipeline.lock'):
            if a.action in ('train','eval','pipeline'):
                gpu_preflight(campaign,cfg,env)
            if a.action in ('train','pipeline'):
                train(campaign,cfg,env,a.seed)
            if a.action in ('eval','pipeline'):
                evaluate(campaign,cfg,env,a.devices,a.seed)
            if a.action in ('report','pipeline'):
                report(campaign,cfg,env,a.seed)
    checked_campaign(a.campaign, a.devices)
    print('COMPLETE', a.action, 'seed', a.seed)


if __name__ == '__main__':
    main()
