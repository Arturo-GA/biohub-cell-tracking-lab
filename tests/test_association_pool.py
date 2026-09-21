import unittest
from biohub_lab.association_pool import pool

class PoolTests(unittest.TestCase):
    def test_confident_original_is_exactly_preserved(self):
        p={(1,3):.95,(2,3):.05};q={(1,3):.01,(2,3):.99}
        self.assertEqual(pool(p,q,'uncertainty')[0],p)
    def test_uncertain_target_uses_more_decisive_specialist(self):
        p={(1,3):.49,(2,3):.51};q={(1,3):.9,(2,3):.1}
        mixed,report=pool(p,q,'uncertainty')
        self.assertGreater(mixed[1,3],mixed[2,3]);self.assertGreater(report['mean_weight'],0)
    def test_absent_sparse_probabilities_are_not_zeros(self):
        p={(1,3):.5,(2,3):.5};q={(1,3):.6,(4,3):.9}
        mixed,_=pool(p,q,'balanced')
        self.assertEqual(mixed[2,3],p[2,3]);self.assertEqual(set(mixed),set(p))
