import os
import time
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'projects/AlgEngine/scripts/diffusiondrive'))
import cfpi_common as c
import selector_cfpi_deployment_common as d
import selector_mid_followup_hashes as h


class HashTests(unittest.TestCase):
    def test_reuse_change_invalidation_and_force_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data';path.write_bytes(b'original')
            time.sleep(2.1)
            original=h.BYTE_HASH
            with patch.object(h,'BYTE_HASH',wraps=original) as reader:
                with h.verified_hash_cache():
                    first=d.artifact(path)
                    for _ in range(4):d.verify(first)
                    self.assertEqual(reader.call_count,1)
                    with h.uncached_hashes():d.verify(first)
                    self.assertEqual(reader.call_count,2)
                    oldtime=path.stat().st_mtime_ns
                    path.write_bytes(b'changed!')
                    os.utime(path,ns=(oldtime,oldtime))
                    with self.assertRaises(RuntimeError):d.verify(first)
                    self.assertEqual(reader.call_count,3)
                self.assertIs(c.sha256_file,original)

    def test_replacement_and_changed_during_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data';path.write_bytes(b'original')
            with h.verified_hash_cache():
                first=d.artifact(path)
                replacement=path.with_suffix('.tmp');replacement.write_bytes(b'replaced');replacement.replace(path)
                with self.assertRaises(RuntimeError):d.verify(first)
            original=h.BYTE_HASH
            def mutation(p):
                result=original(p);p.write_bytes(b'modified during read');return result
            with patch.object(h,'BYTE_HASH',side_effect=mutation):
                with h.verified_hash_cache():
                    with self.assertRaisesRegex(RuntimeError,'while hashing'):c.sha256_file(path)


if __name__=='__main__':unittest.main()
