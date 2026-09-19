import unittest
import numpy as np
from biohub_lab.detection_identity import mappings,identity_audit,project_detections,maximum_vote_edges


class IdentityTests(unittest.TestCase):
    def test_duplicate_positive_edges_can_disappear_under_unique_matching(self):
        truth=np.array([[0,0,0,0],[1,0,0,0]],float)
        coords=np.array([[0,0,0,1],[0,0,0,2],[1,0,0,2],[1,0,0,1]],float)
        report=identity_audit(coords,np.array([[0,2],[1,3]]),truth,np.array([[0,1]]))
        self.assertEqual(report['many_to_one_edges']['positive_edge_instances'],2)
        self.assertEqual(report['one_to_one_edges']['positive_edge_instances'],0)
        self.assertEqual(report['gt_with_multiple_predictions'],2)

    def test_unmatched_is_unknown_and_single_annotation_supported(self):
        many,one=mappings(np.array([[0,0,0,1],[0,0,0,100]],float),np.array([[0,0,0,0]],float))
        self.assertEqual(many.tolist(),[0,-1]);self.assertEqual(one.tolist(),[0,-1])

    def test_projection_prefers_harmonic_and_collapses_parallel_tracks(self):
        coords=np.array([[0,0,0,1],[0,0,0,2],[1,0,0,1],[1,0,0,2]])
        edges,report=project_detections(coords,np.array([[0,2],[1,3]]),np.array([1,0,1,0]))
        self.assertEqual(edges.tolist(),[[1,3]])
        self.assertEqual(report['output_nodes'],2)

    def test_assignment_uses_global_votes_not_greedy(self):
        coords=np.array([[0,0,0,0],[0,0,0,10],[1,0,0,0],[1,0,0,10]])
        edges=maximum_vote_edges(coords,np.array([[0,2],[0,3],[1,2]]),np.array([9,8,8]))
        self.assertEqual(edges.tolist(),[[0,3],[1,2]])


if __name__=='__main__':unittest.main()
