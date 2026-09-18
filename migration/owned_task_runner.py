"""Migration-only entrypoints. Scientific drivers and frozen contracts stay unchanged."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

M = Path(__file__).resolve().parent
C = M.parent
sys.path.insert(0, str(C))
import owned_runtime as owned

PERSONAL = owned.N.parent
DATA = owned.N / 'WorldEngine/data'
OLD_DATA = '/inspire/hdd2/project/roboticsystem2/public/worldengine/data/WE_release_data'
PROJECTS = ['/inspire/hdd/project', '/inspire/hdd2/project', '/inspire/ssd/project', '/inspire/ssd2/project']
FULL_SCENARIO = '/inspire/hdd2/project/roboticsystem2/public/worldengine/data/original_scenarios_WE/navtest_failures/combined/all_scenarios.pkl'
TASKS = {q: dict(run=f'20260917_paper_{q}_v3', stage='eval' if q in 'BCF' else 'all') for q in 'ABCDEF'}
DATA_ALIASES = [
    ('', OLD_DATA),
    ('raw/openscene-v1.1', '/inspire/hdd/project/roboticsystem2/public/OpenScene/openscene-v1.1'),
    ('raw/nuplan', '/inspire/hdd2/project/roboticsystem2/public/nuplan'),
    ('sim_engine/assets', '/inspire/hdd2/project/roboticsystem2/public/worldengine/data/nuplan/WE_processed'),
    ('alg_engine/openscene-synthetic', '/inspire/hdd2/project/roboticsystem2/public/worldengine/data/openscene-synthetic'),
    ('sim_engine/scenarios/original/navtest_failures/all_scenarios.pkl', FULL_SCENARIO),
]
# These are separate alias endpoints because the old data tree contains absolute
# second-level links. Re-resolve from the user's data link on every invocation.
DATA_ENDPOINTS = ['raw/openscene-v1.1/sensor_blobs/trainval',
                  'raw/openscene-v1.1/sensor_blobs/test',
                  'raw/openscene-v1.1/sensor_blobs/mini',
                  'raw/nuplan/dataset/maps',
                  'alg_engine/merged_infos_navformer',
                  'sim_engine/assets/navtrain/assets',
                  'sim_engine/assets/navtest_failures/assets']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2) + '\n'); os.replace(tmp, p)


def within(path, root):
    return path == root or root in path.parents


def make_view():
    empty = owned.O / 'runtime/hidden_legacy_private'
    empty.mkdir(parents=True, exist_ok=True)
    entries = [dict(backing=str(empty), logical=p, kind='hidden') for p in PROJECTS]
    for host, guest in owned.mappings():
        physical = host.resolve(strict=True)
        if not within(physical, PERSONAL):
            raise RuntimeError('Non-data dependency outside personal global_user: ' + str(host))
        entries.append(dict(backing=str(physical), logical=guest, kind='owned'))
    # Remain usable for audits during copy: reports may not have been copied yet.
    if not DATA.is_dir():
        raise RuntimeError('WorldEngine/data link is not accessible')
    for relative, legacy in DATA_ALIASES:
        p = DATA / relative
        if not p.exists():
            if relative=='alg_engine/openscene-synthetic': continue
            raise RuntimeError('Required data endpoint missing (including nested links): ' + str(p))
        physical = p.resolve(strict=True)
        for logical in {str(p), legacy, OLD_DATA + ('/' + relative if relative else '')}:
            entries.append(dict(backing=str(physical), logical=logical, kind='data'))
    for relative in DATA_ENDPOINTS:
        p = DATA / relative
        if relative.endswith('/mini') and not p.exists():
            continue
        if not p.exists():
            raise RuntimeError('Required data endpoint missing: ' + str(p))
        for logical in {str(p), OLD_DATA + '/' + relative}:
            entries.append(dict(backing=str(p.resolve(strict=True)), logical=logical, kind='data'))
    # Checkpoints and metric caches are not dataset exceptions: the migrated
    # model and existing personal caches must be used, not these old shortcuts.
    for relative in ('alg_engine/ckpts', 'alg_engine/pdms_cache'):
        for base in (str(DATA), OLD_DATA):
            entries.append(dict(backing=str(empty), logical=base + '/' + relative, kind='hidden'))
    # Specific aliases take precedence. Reject inconsistent duplicate endpoints.
    unique = {}
    for item in entries:
        old = unique.get(item['logical'])
        if old and old != item:
            raise RuntimeError('Conflicting path mapping: ' + item['logical'])
        unique[item['logical']] = item
    return sorted(unique.values(), key=lambda x: (len(Path(x['logical']).parts), x['logical']))


def command(task, devices):
    if task == 'prior':
        return ['bash', str(C / 'run_h100_full_eval_resume.sh'), '--devices', devices]
    spec = TASKS[task]
    return ['bash', str(C / 'run_paper_study.sh'), 'queue', '--queue', task,
            '--run', spec['run'], '--devices', devices, '--stage', spec['stage']]


def map_back(path, entries):
    path = str(path)
    for item in sorted(entries, key=lambda x: len(x['logical']), reverse=True):
        key = item['logical']
        if path == key or path.startswith(key + '/'):
            return Path(item['backing'] + path[len(key):]), item['kind']
    return Path(path), 'unmapped'


def roots_for(task):
    if task == 'prior':
        return [owned.W / 'runs/20260916_factorial_bridges_4h100_v1',
                owned.W / 'runs/20260917_factorial_full_resume_v1']
    root = owned.W / 'runs' / TASKS[task]['run']
    return [root, root.with_name(root.name + '_eval')] if task in 'BCDF' else [root]


def audit_paths(task, entries):
    """Bounded metadata dependency audit; no data-tree or result-image traversal."""
    files = []
    for root in roots_for(task):
        if not root.exists():
            if root.name.endswith('_eval'): continue
            raise RuntimeError('Required migrated run missing: ' + str(root))
        files += [p for p in root.glob('*.json') if p.name not in ('attempt_status.json', 'resume_status.json')]
        files += list(root.glob('native/*/report.json'))
    files += [owned.W / 'registry/local_paths.json']
    found = {}; missing = []; external = []; inspected = set()
    def inspect(value, source):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and 'sha256' in value:
                path = Path(value['path'])
                classify(path, source, required=True)
                # Follow input manifests, not millions of per-frame artifacts.
                if path.name in ('manifest.json', 'inputs.json', 'run_contract.json', 'checkpoint_manifest.json',
                                 'attribution_inputs.json', 'mechanism.json') and path.exists():
                    files.append(path)
            for v in value.values(): inspect(v, source)
        elif isinstance(value, list):
            for v in value: inspect(v, source)
        elif isinstance(value, str) and value.startswith(('/inspire/', '/root/', '/home/')):
            classify(Path(value), source, required=False)
    def classify(path, source, required):
        physical, kind = map_back(path, entries)
        if path.exists():
            resolved = path.resolve()
            backing, category = map_back(resolved, entries)
            if kind != 'hidden': physical, kind = backing, category
        elif required:
            missing.append(dict(path=str(path), source=str(source)))
        if kind == 'hidden' or (kind != 'data' and not within(physical, PERSONAL)):
            external.append(dict(path=str(path), backing=str(physical), kind=kind, source=str(source)))
        found[str(path)] = dict(backing=str(physical), kind=kind, exists=path.exists())
    while files:
        p = files.pop()
        if str(p) in inspected: continue
        if len(inspected) >= 1000: raise RuntimeError('Metadata audit exceeded bound; inspect dependency closure')
        inspected.add(str(p)); inspect(read(p), p)
    # System libs/drivers remain supplied by the instance image; project-owned
    # Python executables, bootstrap modules and code trees must be personal.
    for p in [owned.ALG / 'bin/python', owned.SIM / 'bin/python', C,
              owned.N / 'DiffusionDrive', owned.N / 'mmcv',
              owned.N / 'exp/metric_cache_navtest', owned.N / 'exp/metric_cache_trainval']:
        classify(p, 'runtime', True)
    result = dict(status='PASS_SCOPED_PATH_AUDIT' if not missing and not external else 'BLOCKED_PATHS',
                  task=task, metadata_files=len(inspected), referenced_paths=len(found),
                  missing=missing, external_nondata=external,
                  paths=found, all_payload_hashes_verified=False,
                  scope='Bounded task metadata dependencies; full frozen checks execute in production drivers')
    return result


def archive_partial_evaluations(task):
    """Preserve only unaudited interrupted conditions, under the production locks."""
    if task not in 'BCDF': return
    from paper_study.common import lock, artifact
    from scripts.resume_h100_full_eval import archive_partial
    root = owned.W / 'runs' / (TASKS[task]['run'] + '_eval')
    if not root.exists(): return
    contract=artifact(root/'run_contract.json'); inputs=read(root/'inputs.json')
    with lock(root / 'active.lock'):
        for p in sorted((root / 'collections').glob('*')):
            if not p.is_dir() or p.is_symlink(): continue
            if (p / 'audit.json').exists() or (p / 'accepted.json').exists(): continue
            if not ((p / 'logs').exists() or any(p.glob('split_*'))): continue
            co=read(p/'collection.json')
            policy=co.get('policy'); cohort=co.get('cohort'); mode=co.get('mode')
            if (policy not in inputs['models'] or cohort not in ('bridge','full','common') or mode not in ('NR','R')
                    or co.get('id')!=p.name or p.name!=f'{cohort}_{mode}_{policy}'
                    or co.get('contract')!=contract or co.get('checkpoint')!=inputs['models'][policy]
                    or co.get('rows')!=inputs['cohorts'][cohort]):
                raise RuntimeError('Unrecognized partial collection; preserved for review: '+str(p))
            archive_partial(p, root / 'archive/migration_interrupted',
                            'Interrupted before audited endpoint; preserve all files and retry complete condition')


def validate_completed_training(task):
    if task not in 'BCF': return
    from paper_study.common import artifact, verify
    from paper_study.data import checked
    root = owned.W / 'runs' / TASKS[task]['run']; manifest = checked(root)
    from paper_study.specs import FEEDBACK, STRUCTURES
    arms = list(FEEDBACK) if task == 'B' else list(STRUCTURES) if task == 'C' else ['paired_common', 'rarelog', 'rarerollout']
    for arm in arms:
        for seed in (0, 1, 2):
            report = root / 'checkpoints' / f'{arm}_seed{seed}' / 'report.json'; r = read(report)
            if r['status'] != 'PASS' or r['stage'] != 'TRAINED_NOT_EVALUATED' or r['provenance']['manifest'] != artifact(root / 'manifest.json'):
                raise RuntimeError('Training endpoint invalid: ' + str(report))
            verify(r['selector']); verify(r['resume'])
            native = read(root / 'native' / f'{arm}_seed{seed}' / 'report.json')
            if native['status'] != 'PASS' or native['provenance']['training'] != artifact(report):
                raise RuntimeError('Native provenance mismatch')
            verify(native['checkpoint'])


def child(args, report_dir):
    entries = read(report_dir / 'view.json')
    # Imports performed in this stricter view cannot use hidden project trees.
    for prefix in PROJECTS:
        if Path(prefix, 'roboticsystem2/wangcaojun-240208020180/setup.sh').exists():
            raise RuntimeError('Old private project still visible')
    tasks = list(TASKS) + ['prior'] if args.task == 'all' else [args.task]
    reports = []
    for task in tasks:
        r = audit_paths(task, entries); write(report_dir / (task + '.paths.json'), r)
        reports.append({k:v for k,v in r.items() if k != 'paths'})
    write(report_dir / 'summary.json', dict(status='PASS_SCOPED_PATHS' if all(r['status']=='PASS_SCOPED_PATH_AUDIT' for r in reports) else 'BLOCKED_PATHS', tasks=reports))
    if any(r['status'] != 'PASS_SCOPED_PATH_AUDIT' for r in reports):
        raise RuntimeError('Missing/external dependencies; see ' + str(report_dir))
    if args.mode == 'audit':
        print(json.dumps(reports, indent=2)); return
    if args.mode == 'check':
        # Same tested non-data checker, with all old project roots hidden except
        # explicitly resolved data and personal non-data compatibility mappings.
        cmd = [str(owned.ALG / 'bin/python'), str(M / 'check_owned_runtime.py'), '--gpu', args.device]
        subprocess.run(cmd, env=os.environ.copy(), check=True)
        write(report_dir / ('cpu_check.json' if args.device=='none' else 'gpu_check.json'), read(owned.O / ('runtime/latest_cpu_check.json' if args.device=='none' else 'runtime/latest_gpu_check.json')))
        return
    if args.mode == 'plan':
        print(json.dumps(dict(task=args.task, command=command(args.task,args.devices), formal_launch=False),indent=2))
        if args.task == 'prior':
            subprocess.run(command('prior',args.devices)+['--plan'],check=True)
        return
    # Production commands enforce scientific identities. The wrapper holds a
    # shared migration lock for their whole lifetime; a copier needs exclusive.
    sys.path.insert(0, str(C / 'scripts'))
    from paper_study.common import legacy_imports, lock
    legacy_imports()
    from paper_study.hardware import hardware
    import run_study_bridges_4h100 as four
    ids=four.devices(args.devices); hardware(ids)
    if four.cpu_capacity() < 36:
        raise RuntimeError('Existing contracts require at least 36 allocated CPU cores')
    subprocess.run([str(owned.ALG/'bin/python'),str(M/'check_owned_runtime.py'),'--gpu',ids[0]], env=dict(os.environ,CUDA_VISIBLE_DEVICES=ids[0]),check=True)
    validate_completed_training(args.task)
    # Exclude the original queue while preparing its interrupted evaluation
    # directories, then pass that queue lock through to a wrapper subprocess.
    # The production launcher reacquires queue.lock, so release immediately
    # before dispatch; active collection locks protect each archive operation.
    if args.task != 'prior':
        root = owned.W / 'runs' / TASKS[args.task]['run']
        with lock(root / 'queue.lock'): archive_partial_evaluations(args.task)
    subprocess.run(command(args.task,args.devices),check=True,cwd=C)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['audit','check','plan','start','_child'])
    p.add_argument('task',choices=list(TASKS)+['prior','all'])
    p.add_argument('--devices',default='0,1,2,3');p.add_argument('--device',default='0')
    p.add_argument('--report-dir',type=Path);p.add_argument('--child-mode')
    p.add_argument('--allow-original-data',action='store_true',help='Explicitly permit original-project data for formal execution after result migration')
    a=p.parse_args()
    if a.mode=='_child':
        if os.environ.get('WORLDENGINE_OWNED_RUNTIME')!='1' or a.child_mode not in ('audit','check','plan','start'):
            p.error('Internal invocation requires owned runtime')
        a.mode=a.child_mode;return child(a,a.report_dir)
    if a.mode in ('plan','start') and a.task=='all':p.error('Choose one task per instance')
    if not a.device.isdigit() and not (a.mode=='check' and a.device=='none'):p.error('One numeric --device required (or none for CPU check)')
    if read(M/'owned_environment_ready.json')['status']!='PASS_ENVIRONMENT_FILES':raise RuntimeError('Environment not verified')
    migration_lock=None
    if a.mode=='start':
        migration_lock=(owned.OUT/'copy.lock').open('r')
        try:fcntl.flock(migration_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Migration is still running; refusing to modify experiment outputs')
        if not (owned.OUT/'copy_status.txt').read_text().startswith('PASS_BYTE_COPY_ONLY'):
            raise RuntimeError('Full migration checksum not finished; refusing formal resume')
    entries=make_view()
    legacy_data=[e for e in entries if e['kind']=='data' and any(e['backing'].startswith(prefix+'/roboticsystem2/') for prefix in PROJECTS)]
    if a.mode=='start' and legacy_data and not a.allow_original_data:
        raise RuntimeError('Data still resolves to original roboticsystem2 project; relink new data first, or explicitly use --allow-original-data for original-project execution')
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(os.getpid())
    report_dir=owned.O/'runtime/task_checks'/stamp;report_dir.mkdir(parents=True)
    write(report_dir/'view.json',entries)
    write(report_dir/'data_location.json',dict(original_project_data=bool(legacy_data),data_link=str(DATA),physical_root=str(DATA.resolve()),allowed_original_data=a.allow_original_data))
    write(owned.O/'runtime/path_map.json',[dict(backing=e['backing'],logical=e['logical']) for e in entries])
    print('TASK_CHECK_DIRECTORY='+str(report_dir),flush=True)
    exe=owned.O/'tools/proot/usr/bin/proot';cmd=[str(exe),'-v','-1','-w',str(C)]
    for item in entries:cmd+=['-b',item['backing']+':'+item['logical']+'!']
    cmd += [str(owned.ALG/'bin/python'),str(Path(__file__).resolve()),'_child',a.task,
            '--child-mode',a.mode,'--report-dir',str(report_dir),'--device',a.device,'--devices',a.devices]
    env=owned.environment()
    if a.mode=='check':env['CUDA_VISIBLE_DEVICES']='' if a.device=='none' else a.device
    elif a.mode in ('audit','plan'):env['CUDA_VISIBLE_DEVICES']=''
    proc=subprocess.Popen(cmd,env=env,cwd=C,start_new_session=True)
    def stop(*_):raise KeyboardInterrupt('Owned task interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:rc=proc.wait()
    except BaseException:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=45)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise
    finally:
        if migration_lock:migration_lock.close()
    write(report_dir/'exit.json',dict(exit_code=rc,finished=time.time(),mode=a.mode,task=a.task))
    raise SystemExit(rc)


if __name__=='__main__':
    try:main()
    except (RuntimeError,FileNotFoundError) as exc:
        print('OWNED_TASK_NOT_READY: '+str(exc),file=sys.stderr);sys.exit(2)
