"""Place new experiment outputs on the configured shared volume, keeping public run paths."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
from .common import WORKSPACE,artifact,read,sha,write


def settings():
    path=WORKSPACE/'registry/paper_storage.json'
    return read(path) if path.exists() else None


def writable_probe(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.paper_write_probe_',dir=directory)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(b'\0'*(1024*1024));stream.flush();os.fsync(stream.fileno())
    finally:
        if os.path.exists(name):os.unlink(name)


def ensure_run(run):
    run=Path(run);spec=settings()
    if spec is None:
        run.mkdir(parents=True,exist_ok=True);writable_probe(run);return run
    destination=Path(spec['runs_root'])/run.name
    if not destination.is_absolute():raise ValueError('Shared runs_root must be absolute')
    if run.is_symlink():
        if run.resolve()!=destination.resolve():raise RuntimeError('Run points to a different storage volume: '+str(run))
    elif run.exists():
        raise RuntimeError('Existing local run must be preserved or explicitly migrated before using configured paper storage: '+str(run))
    else:
        destination.mkdir(parents=True,exist_ok=True);run.parent.mkdir(parents=True,exist_ok=True)
        try:run.symlink_to(destination,target_is_directory=True)
        except FileExistsError:
            if not run.is_symlink() or run.resolve()!=destination.resolve():raise
    writable_probe(destination)
    write(destination/'storage.json',dict(schema=1,logical_run=str(run),physical_run=str(destination),configuration=artifact(WORKSPACE/'registry/paper_storage.json')))
    return run


def snapshot(source,target,inventory):
    """Copy only the audited input inventory, never accumulated runtime output/core dumps."""
    source,target=Path(source),Path(target)
    own_contract=target.parent/'run_contract.json'
    if own_contract.exists():
        for relative,expected in read(own_contract)['implementation'].items():
            if sha(target/relative)!=expected:raise RuntimeError('Frozen runtime changed: '+relative)
        return
    if target.exists():
        # Retain failed preparation evidence on the output volume.
        archive=target.parent/'failed_preparations';archive.mkdir(exist_ok=True)
        holder=Path(tempfile.mkdtemp(prefix='runtime_',dir=archive));target.rename(holder/'runtime')
    stage=Path(tempfile.mkdtemp(prefix='.runtime_build_',dir=target.parent))
    try:
        for relative,expected in sorted(inventory.items()):
            rel=Path(relative)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid inventory path')
            original=source/rel;destination=stage/rel
            if sha(original)!=expected:raise RuntimeError('Source runtime input changed: '+relative)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,destination)
            if sha(destination)!=expected:raise RuntimeError('Runtime copy verification failed: '+relative)
        # The frozen evaluator resolves its data root relative to the runtime.
        data=source/'data'
        if data.is_symlink(): (stage/'data').symlink_to(data.resolve(),target_is_directory=True)
        stage.rename(target)
    finally:
        if stage.exists():shutil.rmtree(stage)


def stopped(run,exception):
    """A full disk must never replace the original exception with a logging exception."""
    import time
    value=dict(status='STOPPED',reason=str(exception),finished=time.time())
    print('PAPER_QUEUE_STOPPED: '+repr(exception),file=sys.stderr,flush=True)
    try:write(Path(run)/'attempt_status.json',value,False)
    except OSError as reporting_error:
        print('Unable to persist attempt_status.json: '+repr(reporting_error),file=sys.stderr,flush=True)


def reports_directory(name):
    logical=WORKSPACE/'reports'/name;spec=settings()
    if spec is None:
        logical.mkdir(parents=True,exist_ok=True);return logical
    target=Path(spec['reports_root'])/name;target.mkdir(parents=True,exist_ok=True)
    if logical.is_symlink():
        if logical.resolve()!=target.resolve():raise RuntimeError('Report storage mapping changed')
    elif not logical.exists():
        logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(target,target_is_directory=True)
    else:
        raise RuntimeError('Existing local report directory preserved; use a new run ID')
    return logical
