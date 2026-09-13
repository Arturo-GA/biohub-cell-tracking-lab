import unittest
import itertools
import numpy as np
from biohub_lab.hoct_linker import HOCTConfig, candidate_edges, frame_features, solve_lineage, _context_tiles


class HOCTTests(unittest.TestCase):
    def test_physical_candidates_and_missing_frame(self):
        coords = np.array([[0,0,0,0], [1,10,0,0], [1,0,10,0], [3,0,0,0]])
        np.testing.assert_array_equal(candidate_edges(coords), [[0,2]])

    def test_morphology_reads_image_and_keeps_centers(self):
        z,y,x = np.indices((12,40,40))
        image = np.exp(-((z-5)**2/2 + (y-12)**2/16 + (x-12)**2/16))
        image += .7*np.exp(-((z-6)**2/2 + (y-27)**2/9 + (x-27)**2/9))
        points = np.array([[5,12,12],[6,27,27]])
        features, sizes = frame_features(image, points, 3)
        self.assertEqual(features.shape, (2,19))
        np.testing.assert_equal(features[:,:4], np.c_[[3,3], points])
        self.assertTrue((sizes > 8).all())
        self.assertGreater(features[0,4], features[1,4])
        self.assertGreater(features[0,7], features[1,7])

    def test_division_and_competing_parent(self):
        coords = np.array([[0,0,0,0],[0,0,10,0],[1,0,1,0],[1,0,2,0],[1,0,11,0]])
        edges = np.array([[0,2],[0,3],[0,4],[1,2],[1,3],[1,4]])
        selected = solve_lineage(coords,edges,np.array([.95,.90,.01,.01,.02,.98]),np.zeros(5))
        np.testing.assert_array_equal(selected,[[0,2],[0,3],[1,4]])

    def test_orphan_is_not_forced_to_link(self):
        coords = np.array([[0,0,0,0],[1,0,1,0]])
        selected = solve_lineage(coords,np.array([[0,1]]),np.array([.01]),np.ones(2))
        self.assertEqual(selected.shape,(0,2))

    def test_matching_global_conflict_resolution(self):
        coords = np.array([[0,0,0,0],[0,0,5,0],[1,0,1,0],[1,0,4,0]])
        edges = np.array([[0,2],[0,3],[1,2],[1,3]])
        config = HOCTConfig(division_cost=100.)
        selected = solve_lineage(coords, edges, np.array([.8,.7,.79,.01]), np.zeros(4), config)
        np.testing.assert_array_equal(selected,[[0,3],[1,2]])

    def test_tiling_keeps_every_parent_and_complete_ownership(self):
        rng = np.random.default_rng(32)
        coords = np.c_[np.repeat([0,1],50),rng.uniform(0,20,(100,3))]
        edges = candidate_edges(coords)
        seen = []
        config = HOCTConfig(max_context_edges=45)
        for context, own, targets in _context_tiles(coords,edges,np.arange(len(edges)),config):
            self.assertLessEqual(len(context),45)
            seen.extend(context[own])
            for t in targets:
                self.assertEqual(sum(edges[context,1]==t),sum(edges[:,1]==t))
        self.assertEqual(sorted(seen),list(range(len(edges))))

    def test_solver_rejects_skipped_time(self):
        with self.assertRaises(ValueError):
            solve_lineage(np.array([[0,0,0,0],[2,0,0,0]]),[[0,1]],[.9],[0,0])

    def test_sparse_matching_matches_exhaustive_lineage_objective(self):
        coords = np.array([[0,0,0,0],[0,0,5,0],[1,0,1,0],[1,0,4,0],[1,0,6,0]])
        edges = np.array(list(itertools.product([0,1],[2,3,4])))
        rng = np.random.default_rng(76)
        config = HOCTConfig()
        for _ in range(12):
            probabilities, orphan = rng.random(6), rng.random(5)
            def cost(selection):
                outgoing = np.bincount([s for s,t in selection],minlength=5)
                incoming = np.bincount([t for s,t in selection],minlength=5)
                if max(outgoing)>2 or max(incoming)>1:
                    return float('inf')
                total = sum(config.edge_bias-probabilities[i] for i,e in enumerate(edges) if tuple(e) in selection)
                total += sum(config.appearance_cost*(1-orphan[t]) for t in (2,3,4) if incoming[t]==0)
                total += sum(config.disappearance_cost for s in (0,1) if outgoing[s]==0)
                total += sum(config.division_cost for s in (0,1) if outgoing[s]==2)
                return total
            optimum = min(cost({tuple(edges[i]) for i in range(6) if mask&(1<<i)}) for mask in range(64))
            selected = {tuple(e) for e in solve_lineage(coords,edges,probabilities,orphan,config)}
            self.assertAlmostEqual(cost(selected),optimum,places=10)


if __name__ == '__main__':
    unittest.main()
