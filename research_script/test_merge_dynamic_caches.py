import json
from pathlib import Path
import tempfile
import unittest

import merge_dynamic_caches as merge


class MergeTests(unittest.TestCase):
    def test_expected_grid(self):
        self.assertEqual(len(merge.expected_keys()), 72)
        for leave in range(1, 5):
            self.assertEqual(sum(k[2] == leave for k in merge.expected_keys()), 18)

    def test_existing_destination_is_not_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root/'keep'
            marker.write_text('original')
            with self.assertRaises(FileExistsError):
                merge.merge(root)
            self.assertEqual(marker.read_text(), 'original')

    def test_copy_is_independent_and_only_binding_is_relocated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, staging, final = root/'source', root/'staging', root/'unified'
            src.mkdir()
            staging.mkdir()
            merge.write(src/'manifest.json', {'status': 'COMPLETED'})
            merge.write(src/'checkpoint_binding.json', {'cache': str(src), 'status': 'COMPLETED'})
            (src/'numeric.csv').write_text('action,budget\n0,0\n1,30\n')
            source_binding = (src/'checkpoint_binding.json').read_bytes()
            entry = dict(source_cache=str(src), relative_path='checkpoint_leave01/sub001_game2',
                cache_manifest_sha256=merge.sha256(src/'manifest.json'),
                source_binding_sha256=merge.sha256(src/'checkpoint_binding.json'), file_count=3)
            result = merge.copy_leaf(entry, staging, final)
            dest = staging/entry['relative_path']
            self.assertEqual((src/'checkpoint_binding.json').read_bytes(), source_binding)
            self.assertEqual((src/'numeric.csv').read_bytes(), (dest/'numeric.csv').read_bytes())
            self.assertEqual((src/'manifest.json').read_bytes(), (dest/'manifest.json').read_bytes())
            self.assertNotEqual((src/'numeric.csv').stat().st_ino, (dest/'numeric.csv').stat().st_ino)
            self.assertFalse((dest/'numeric.csv').is_symlink())
            self.assertEqual(merge.read(dest/'checkpoint_binding.json')['cache'], str(final/entry['relative_path']))
            self.assertEqual(result['file_sha256']['numeric.csv'], merge.sha256(src/'numeric.csv'))
            (dest/'numeric.csv').write_text('changed')
            self.assertEqual((src/'numeric.csv').read_text(), 'action,budget\n0,0\n1,30\n')


if __name__ == '__main__':
    unittest.main()
