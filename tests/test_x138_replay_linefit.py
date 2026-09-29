"""Compare CPU replay against the actual upstream notebook implementation."""
import ast
import collections
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('replay', ROOT / 'kaggle/x138_xr/replay_linefit.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
nb = json.loads((ROOT / 'kaggle/x138_xr/upstream/biohub-x138.ipynb').read_text(encoding='utf-8'))
source = ''.join(nb['cells'][5]['source'])
func = next(ast.get_source_segment(source, n) for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == 'linefit_smooth_output_graph')


class ReplayTest(unittest.TestCase):
    def check(self, nodes, edges, window=2, weight=0.8):
        scope = dict(np=np, OUTPUT_LINEFIT_SMOOTH=True, OUTPUT_LINEFIT_WINDOW=window, OUTPUT_LINEFIT_WEIGHT=weight)
        exec(func, scope)
        ref = scope['linefit_smooth_output_graph']({k: dict(v) for k, v in nodes.items()}, edges, collections.Counter())
        expected = {k: replay.rounded_node(v) for k, v in ref.items()}
        self.assertEqual(replay.full_linefit_round(nodes, edges, window, weight), expected)

    def test_edits_divisions_gaps_and_capture_precision(self):
        rng = np.random.default_rng(43)
        for _ in range(80):
            nodes = {i: dict(node_id=i, t=i//3, **dict(zip(('z','y','x'), rng.uniform(-2, 100, 3).astype(np.float32)))) for i in range(60)}
            edges = [dict(source_id=i, target_id=i+3) for i in range(57) if rng.random() < 0.8]
            edges += [dict(source_id=6, target_id=10), dict(source_id=12, target_id=18)]
            self.check(nodes, edges)
            self.check(nodes, edges, window=1, weight=1)

    def test_half_integer_rounding_and_empty(self):
        nodes = {i: dict(node_id=i,t=i,z=0.5,y=2.5,x=i+0.5) for i in range(12)}
        edges = [dict(source_id=i,target_id=i+1) for i in range(11)]
        for weight in (0, 0.8, 1):
            self.check(nodes, edges, weight=weight)
        self.check(nodes, [])
        self.check({}, [])


if __name__ == '__main__':
    unittest.main()
