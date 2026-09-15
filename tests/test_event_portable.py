import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from biohub_lab.event_data import neighborhoods, sparse_labels
from biohub_lab.event_inference import score_graph
from biohub_lab.event_local_train import train, calibrate
from biohub_lab.event_model import EventGraphNet
from biohub_lab.event_portable import VideoCache, input_identity, load_portable, prepare_mmap
from biohub_lab.event_train import TRAIN_CONFIG
from test_event_graph import synthetic


class PortableEventTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        graph, truth, edges = synthetic(); labels, _ = sparse_labels(graph, truth, edges)
        split = dict(fit=['fit_a', 'fit_b'], calibration=['cal'], evaluation=['eval'])
        rng = np.random.default_rng(1)
        for name in split['fit']+split['calibration']:
            folder = root/'videos'/name; folder.mkdir(parents=True)
            np.savez_compressed(folder/'graph.npz', **graph)
            np.savez_compressed(folder/'features.npz', visual=rng.normal(size=(6, 64)).astype(np.float16),
                                neighbors=neighborhoods(graph['coords']))
            np.savez_compressed(folder/'labels.npz', **labels)
        return split

    def test_streaming_preserves_all_neighbors_across_query_blocks(self):
        torch.manual_seed(11); n = 53; model = EventGraphNet(width=16, layers=4).eval()
        x = torch.randn(n, 72); p = torch.randn(n, 4)
        neighbors = torch.randint(0, n, (n, 25)); neighbors[:, 0] = torch.arange(n); neighbors[:, -2:] = -1
        with torch.inference_mode():
            expected = model.encode(x, p, neighbors)
            streamed = model.encode_streamed(x, p, neighbors, 'cpu', chunk=7)
        torch.testing.assert_close(streamed, expected, atol=2e-6, rtol=2e-5)

    def test_activation_checkpoint_preserves_gradients(self):
        torch.manual_seed(3); one = EventGraphNet(width=16, layers=4); two = EventGraphNet(width=16, layers=4)
        two.load_state_dict(one.state_dict()); two.checkpoint_activations = True; two.attention_chunk = 3
        x = torch.randn(15, 72); p = torch.randn(15, 4); neighbors = torch.randint(0, 15, (15, 8))
        one.encode(x, p, neighbors).square().mean().backward()
        two.encode(x, p, neighbors).square().mean().backward()
        for a, b in zip(one.parameters(), two.parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_mmap_cache_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); split = self.fixture(root)
            before = load_portable(root/'videos/fit_a')
            for name in split['fit']+split['calibration']:
                prepare_mmap(root/'videos'/name)
            after = load_portable(root/'videos/fit_a')
            np.testing.assert_array_equal(before['inputs'], after['inputs'])
            self.assertIsInstance(after['neighbors'], np.memmap)
            cache = VideoCache(root, 1)
            cache['fit_a']; cache['fit_b']; self.assertEqual(list(cache.items), ['fit_b'])
            input_identity(root, split)
            path = root/'videos/fit_a/arrays/graph_scores.npy'
            with path.open('ab') as handle:
                handle.write(b'tamper')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                input_identity(root, split)
            cache.clear(); del before, after

    def test_stop_resume_matches_continuous_optimizer_and_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            full = Path(temp)/'full'; resumed = Path(temp)/'resumed'; split = self.fixture(full)
            shutil.copytree(full, resumed)
            config = dict(TRAIN_CONFIG, steps=4, validate_every=2)
            execution = dict(checkpoint_every=1, attention_chunk=3)
            train(full, split, 'cpu', config=config, execution=execution)
            paused = train(resumed, split, 'cpu', config=config, execution=execution, stop_after=2)
            self.assertEqual(paused['status'], 'paused')
            train(resumed, split, 'cpu', config=config, execution=execution, resume=True)
            a = torch.load(full/'resume.pt', weights_only=False)
            b = torch.load(resumed/'resume.pt', weights_only=False)
            self.assertEqual(a['trace'], b['trace']); self.assertEqual(a['history'], b['history'])
            self.assertEqual(a['scheduler'], b['scheduler'])
            for key in a['state_dict']:
                torch.testing.assert_close(a['state_dict'][key], b['state_dict'][key], atol=0, rtol=0)
            for key, state in a['optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b['optimizer']['state'][key][name], atol=0, rtol=0)
            self.assertTrue((resumed/'resume.pt.previous').is_file())
            result = calibrate(resumed, split, device='cpu', chunk=3)
            self.assertTrue(np.isfinite(result['edge']['logit']))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                train(resumed, split, 'cpu', config=dict(config, learning_rate=.01), execution=execution, resume=True)

    def test_streamed_scoring_keeps_same_pairs_and_gains(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); self.fixture(root); video = load_portable(root/'videos/fit_a')
            model = EventGraphNet(width=16, layers=4).eval()
            cal = dict(edge={'logit': 0.}, division={'logit': -100.})
            a, ca = score_graph(model, video, cal, 'cpu')
            b, cb = score_graph(model, video, cal, 'cpu', stream_chunk=2)
            self.assertEqual(ca, cb)
            for key in a:
                np.testing.assert_allclose(a[key], b[key], atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
