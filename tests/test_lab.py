import ast
import csv
import json
import tempfile
import unittest
import uuid
from pathlib import Path

import torch

from biohub_lab.fusion import division_aware_reverse_weight
from biohub_lab.patch import patch_source
from biohub_lab.submission import COLUMNS, read_and_validate


class DivisionGuardTests(unittest.TestCase):
    def setUp(self):
        self.sources = torch.tensor([[[10., 10., 10.], [20., 20., 20.]]])
        self.targets = torch.tensor([[[10., 9., 10.], [10., 11., 10.]]])

    def test_strong_division_reduces_only_parent_reverse_weight(self):
        p = torch.tensor([[[.95, .95], [.05, .05]]])
        w = division_aware_reverse_weight(p, self.sources, self.targets)
        torch.testing.assert_close(w, torch.tensor([[[0.], [.15]]]))

    def test_continuations_are_unchanged(self):
        p = torch.tensor([[[.95, .05], [.05, .95]]])
        w = division_aware_reverse_weight(p, self.sources, self.targets)
        torch.testing.assert_close(w, torch.full((1, 2, 1), .15))

    def test_far_daughters_do_not_protect_false_division(self):
        p = torch.tensor([[[.95, .95], [.05, .05]]])
        far = self.targets.clone()
        far[:, 1, 0] += 10  # 16.25 um along z, not 4.06 um in xy.
        w = division_aware_reverse_weight(p, self.sources, far)
        torch.testing.assert_close(w, torch.full((1, 2, 1), .15))

    def test_probability_ramp_and_batch_shape(self):
        p = torch.tensor([[[.7, .7], [.3, .3]]]).repeat(2, 1, 1)
        w = division_aware_reverse_weight(p, self.sources.repeat(2, 1, 1), self.targets.repeat(2, 1, 1))
        self.assertEqual(w.shape, (2, 2, 1))
        self.assertAlmostEqual(w[0, 0, 0].item(), .075, places=6)

    def test_empty_and_single_target(self):
        for n in (0, 1):
            w = division_aware_reverse_weight(torch.ones(1, 2, n), self.sources, self.targets[:, :n])
            torch.testing.assert_close(w, torch.full((1, 2, 1), .15))

    def test_disabled_is_exact_control(self):
        p = torch.tensor([[[.95, .95], [.05, .05]]])
        w = division_aware_reverse_weight(p, self.sources, self.targets, strength=0)
        torch.testing.assert_close(w, torch.full((1, 2, 1), .15))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            division_aware_reverse_weight(torch.full((1, 2, 2), float('nan')), self.sources, self.targets)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path('artifacts/test_tmp') / uuid.uuid4().hex
        self.tmp.mkdir(parents=True)
        self.path = self.tmp / 'sample.csv'
        self.rows = [
            [0, 'a', 'node', 4, 0, 2, 2, 2, -1, -1],
            [1, 'a', 'node', 5, 1, 2, 2, 3, -1, -1],
            [2, 'a', 'edge', -1, -1, -1, -1, -1, 4, 5],
        ]

    def tearDown(self):
        self.path.unlink(missing_ok=True)
        self.tmp.rmdir()

    def validate(self, shapes=None):
        with self.path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerows(self.rows)
        return read_and_validate(self.path, shapes or {'a': (3, 8, 8, 8)})

    def test_valid_non_contiguous_node_ids(self):
        self.assertEqual(self.validate()['a'][1], [(4, 5)])

    def test_out_of_volume_hub_rejected(self):
        self.rows[0][4] = -1000
        with self.assertRaisesRegex(ValueError, 'out of bounds'):
            self.validate()

    def test_gap_edge_rejected(self):
        self.rows[1][4] = 2
        with self.assertRaisesRegex(ValueError, 'non-consecutive'):
            self.validate()

    def test_missing_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing datasets'):
            self.validate({'a': (3, 8, 8, 8), 'b': (3, 8, 8, 8)})

    def test_duplicate_edge_rejected(self):
        self.rows.append([3, *self.rows[2][1:]])
        with self.assertRaisesRegex(ValueError, 'duplicate edge'):
            self.validate()

    def test_dangling_edge_rejected(self):
        self.rows[2][-1] = 999
        with self.assertRaisesRegex(ValueError, 'dangling'):
            self.validate()


class PackagingTests(unittest.TestCase):
    def test_patch_has_one_call_and_compiles_both_arms(self):
        source = Path('baseline/harmonic_inference.py').read_text(encoding='utf8')
        for candidate in (False, True):
            patched = patch_source(source, candidate=candidate)
            compile(patched, 'runtime.py', 'exec')
            tree = ast.parse(patched)
            value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_bi_new' for t in n.targets))
            # Parse the injected block in a valid surrounding function/if scope.
            compile('def f():\n    if True:\n        if True:\n' + value + '                pass\n', 'patch.py', 'exec')
            self.assertEqual(value.count('from biohub_lab.fusion import'), int(candidate))
            self.assertNotIn('"PYTHONPATH": "src"}', patched)

    def test_upstream_drift_fails_closed(self):
        with self.assertRaises(ValueError):
            patch_source('print("changed upstream")', candidate=True)


if __name__ == '__main__':
    unittest.main()
