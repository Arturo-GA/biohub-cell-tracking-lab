import unittest
from biohub_lab.edge_complementarity import overlap,feature_bins
class ComplementarityTests(unittest.TestCase):
    def test_overlap_and_missing_endpoints(self):
        truth={(0,1),(1,2),(3,4),(4,5)}
        r=overlap({(0,1),(1,2)},{(0,1),(3,4)},truth,{0,1,2,3})
        self.assertEqual([r[k] for k in ['both_correct','harmonic_only','tissue_only','neither_correct']],[1,1,1,1])
        self.assertEqual(r['rescued_with_missing_harmonic_endpoint'],1)
        self.assertEqual(r['diagnostic_union_tp'],3)
    def test_association_rescue(self):
        r=overlap(set(),{(1,2)},{(1,2)},{1,2})
        self.assertEqual(r['rescued_with_both_endpoints_matched_in_harmonic'],1)
    def test_reject_non_gt(self):
        with self.assertRaises(ValueError):overlap({(1,2)},set(),set(),set())
    def test_geometry_no_annotations(self):
        nodes={i:dict(t=i,z=0,y=0,x=i) for i in range(4)}
        bins=feature_bins(nodes,[(0,1),(1,2),(2,3)])
        self.assertEqual(bins[(1,2)],('distance_le3','two_sided_context','acceleration_le2'))
if __name__=='__main__':unittest.main()
