import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch

from biohub_lab.detection_dag import build_dag
from biohub_lab.event_data import neighborhoods, node_inputs, sparse_labels, select_split, save_json
from biohub_lab.event_model import EventGraphNet
from biohub_lab.event_train import TRAIN_CONFIG, train_and_calibrate, load_video, tensor_video, threshold
from biohub_lab.event_inference import top_per_mother, score_graph, write_csv
from biohub_lab.event_solver import SOLVER_CONFIG, select_graph, solve_window, validate_edges
from biohub_lab.submission import read_and_validate


def synthetic():
    coords = np.array([[0, 8, 20, 20], [1, 8, 20, 16], [1, 8, 20, 24],
                       [0, 8, 40, 20], [1, 8, 40, 20], [1, 8, 60, 20]], np.int32)
    graph = build_dag(coords, np.ones(len(coords)), np.zeros(len(coords), np.int8), (2, 16, 80, 80))
    truth = coords[:5].astype(np.float32); gt_edges = np.array([[0, 1], [0, 2], [3, 4]])
    return graph, truth, gt_edges


class EventGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sparse_unknowns_and_known_incompatible_parent(self):
        graph, truth, edges = synthetic(); labels, counts = sparse_labels(graph, truth, edges)
        pair_labels = {(int(m), *sorted((int(a), int(b)))): y for (m, a, b), y in zip(labels['triples'], labels['triple_y'])}
        self.assertEqual(pair_labels[(0, 1, 2)], 1)
        self.assertEqual(pair_labels[(0, 1, 4)], 0)
        self.assertNotIn((0, 1, 5), pair_labels)
        self.assertFalse(labels['quality_mask'][5])
        self.assertEqual(counts['represented_unique_divisions'], 1)
        self.assertFalse(any(t == 5 for _, t in labels['edges']))

    def test_ambiguous_mapping_is_not_supervised(self):
        graph, truth, edges = synthetic()
        truth = np.r_[truth, np.array([[0, 8, 20, 21]], np.float32)]
        labels, _ = sparse_labels(graph, truth, edges)
        self.assertFalse(labels['quality_mask'][0])
        self.assertFalse(any(m == 0 for m, _ in labels['edges']))

    def test_filename_split_disjoint_reproducible(self):
        names = [f'{group}_{i:04}' for group in ('a', 'b') for i in range(60)]
        evaluation = names[:4]; excluded = names[4:8]
        one = select_split(names, evaluation, excluded); two = select_split(names[::-1], evaluation, excluded)
        self.assertEqual(one, two)
        self.assertEqual((len(one['fit']), len(one['calibration'])), (48, 16))
        self.assertEqual(sum(len(one[k]) for k in ('fit', 'calibration', 'evaluation', 'excluded', 'reserved')), len(names))

    def test_division_head_symmetric_and_context_active(self):
        graph, _, _ = synthetic(); model = EventGraphNet(width=16, layers=2).eval()
        torch.manual_seed(4); x = torch.randn(6, 72); p = torch.tensor(graph['coords'], dtype=torch.float32)/20
        neighbors = torch.tensor(neighborhoods(graph['coords']).astype(np.int64))
        h = model.encode(x, p, neighbors)
        a = model.divisions(h, p, torch.tensor([[0, 1, 2]]))
        b = model.divisions(h, p, torch.tensor([[0, 2, 1]]))
        torch.testing.assert_close(a, b)
        changed = x.clone(); changed[1] *= -3
        self.assertFalse(torch.allclose(h[0], model.encode(changed, p, neighbors)[0]))
        a.sum().backward(); self.assertIsNotNone(model.layers[0].qkv.weight.grad)

    def test_future_context_changes_local_link_choice(self):
        coords = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 20], [2, 0, 0, 20], [3, 0, 0, 20]])
        graph = dict(coords=coords, shape=np.array([4, 5, 30, 30]), origin=np.zeros(5, np.int8))
        edges = np.array([[0, 1], [0, 2], [2, 3], [3, 4]])
        # Node 2 can also be born as a separate track. Its continuation gain of
        # 3 beats the competing 4 only after birth/death continuity costs matter.
        gains = np.array([4., 3., 3., 3.])
        config = dict(SOLVER_CONFIG, birth_cost=2., death_cost=2., window=3, stride=2, time_limit=10.)
        chosen, report = select_graph(graph, edges, gains, np.empty((0, 3), int), np.empty(0), np.ones(5), config)
        self.assertIn((0, 2), map(tuple, chosen)); self.assertNotIn((0, 1), map(tuple, chosen))
        self.assertIn((3, 4), map(tuple, chosen)); self.assertFalse(report['fallback_windows'])
        validate_edges(coords, chosen)

    def test_division_and_incoming_constraints_across_windows(self):
        coords = np.array([[0, 0, 0, 10], [1, 0, 0, 0], [1, 0, 0, 20], [2, 0, 0, 0], [2, 0, 0, 20]])
        graph = dict(coords=coords, shape=np.array([3, 3, 3, 30]), origin=np.zeros(5, np.int8))
        edges = np.array([[0, 1], [1, 3], [2, 4], [1, 4]])
        selected, report = select_graph(graph, edges, np.array([2., 4., 4., 3.]), np.array([[0, 1, 2]]),
                                        np.array([6.]), np.ones(5), dict(SOLVER_CONFIG, window=2, stride=1))
        self.assertEqual(set(map(tuple, selected)), {(0, 1), (0, 2), (1, 3), (2, 4)})
        self.assertEqual(report['selected_divisions'], 1)

    def test_fallback_is_explicit_and_feasible(self):
        coords = np.array([[0, 0, 0, 0], [0, 0, 0, 20], [1, 0, 0, 10]])
        events = np.array([[0, 2, -1], [1, 2, -1]])
        with patch('biohub_lab.event_solver.milp', return_value=SimpleNamespace(x=None, status=1, message='time limit')):
            chosen, report = solve_window(coords, events, np.array([5., 4.]), np.ones(3)*.1,
                                            np.zeros(3, bool), SOLVER_CONFIG)
        self.assertTrue(report['fallback']); self.assertEqual(len(chosen), 1)

    def test_pruning_and_all_pairs_scored(self):
        graph, _, _ = synthetic()
        indices = np.array([[0, 3], [0, 1], [1, 5], [1, 2]])
        chosen, gains = top_per_mother(indices, np.array([1., 4., 7., 3.]), 1)
        self.assertEqual(chosen.tolist(), [[0, 1], [1, 5]])
        video = dict(graph=graph, inputs=node_inputs(graph, np.zeros((6, 64))), neighbors=neighborhoods(graph['coords']))
        model = EventGraphNet(width=16, layers=2).eval()
        scored, stats = score_graph(model, video, {'edge': {'logit': 0}, 'division': {'logit': -100}}, device='cpu')
        self.assertEqual(stats['scored_pairs'], graph['pair_counts'].sum())
        self.assertLessEqual(len(scored['triples']), 2*len(graph['coords']))

    def test_training_checkpoint_calibration_and_final_csv(self):
        graph, truth, edges = synthetic(); labels, counts = sparse_labels(graph, truth, edges)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); split = dict(fit=['fit'], calibration=['cal'], evaluation=['eval'])
            rng = np.random.default_rng(5)
            for name in ('fit', 'cal'):
                folder = root/'videos'/name; folder.mkdir(parents=True)
                np.savez_compressed(folder/'graph.npz', **graph)
                np.savez_compressed(folder/'features.npz', visual=rng.normal(size=(6, 64)).astype(np.float16),
                                    neighbors=neighborhoods(graph['coords']))
                np.savez_compressed(folder/'labels.npz', **labels)
            calibration = train_and_calibrate(root, split, device='cpu', config=dict(TRAIN_CONFIG, steps=2, validate_every=1))
            self.assertTrue((root/'best.pt').is_file()); self.assertTrue(np.isfinite(calibration['edge']['logit']))
            self.assertFalse(json.loads((root/'training.json').read_text())['evaluation_labels_read'])
            folder = root/'videos/eval'; folder.mkdir()
            np.savez_compressed(folder/'prediction.npz', coords=graph['coords'], edges=edges, shape=graph['shape'])
            write_csv(root, ['eval'], root/'candidate.csv')
            self.assertIn('eval', read_and_validate(root/'candidate.csv', {'eval': graph['shape']}))


if __name__ == '__main__':
    unittest.main()
