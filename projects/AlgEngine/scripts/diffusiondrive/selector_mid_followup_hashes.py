"""Process-local hash reuse; changed file identity always invalidates the cache."""
from contextlib import contextmanager
from pathlib import Path
import time
import cfpi_common as c

# Capture the byte-reading implementation before entering any cache context.
BYTE_HASH = c.sha256_file


def stamp(path):
    st=path.stat()
    return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns


@contextmanager
def verified_hash_cache():
    previous=c.sha256_file
    entries={}
    def checked_hash(path):
        path=Path(path).expanduser().resolve()
        before=stamp(path)
        cached=entries.get(path)
        # Some filesystems round timestamps. Never reuse a hash for a file
        # created/changed within the last two seconds, even if stat is unchanged.
        stable_age = time.time_ns()-max(before[-2:]) > 2_000_000_000
        if cached is not None and cached[0]==before and stable_age:
            return cached[1]
        result=BYTE_HASH(path)
        if stamp(path)!=before:
            raise RuntimeError('File changed while hashing: '+str(path))
        entries[path]=(before,result)
        return result
    c.sha256_file=checked_hash
    try:
        yield
    finally:
        c.sha256_file=previous


@contextmanager
def uncached_hashes():
    previous=c.sha256_file
    c.sha256_file=BYTE_HASH
    try:
        yield
    finally:
        c.sha256_file=previous
