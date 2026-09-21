import unittest,numpy as np
from biohub_lab.detector_complements import refine,complement,standardized
class ComplementTests(unittest.TestCase):
    def test_one_to_one_refinement(self):
        anchors=np.array([[0,0,0],[0,0,2]])
        out=refine(anchors,np.array([[0,0,1]]))
        self.assertEqual(np.sum(np.all(out==[0,0,1],axis=1)),1)
        self.assertEqual(len(np.unique(out,axis=0)),2)
    def test_outside_radius_unchanged(self):
        anchors=np.array([[0,0,0]])
        np.testing.assert_array_equal(refine(anchors,np.array([[2,0,0]])),anchors)
    def test_budget_and_novelty(self):
        dense=np.array([[i*3,0,0] for i in range(8)])
        out=complement(dense,np.array([[0,0,0],[0,9,0]]),4)
        self.assertEqual(len(out),4)
        self.assertTrue(np.any(np.all(out==[0,9,0],axis=1)))
    def test_preserves_existing_coincident_nodes_without_new_collision(self):
        anchors=np.array([[0.,0,0],[0,0,0],[0,0,2]])
        out=refine(anchors,np.array([[0,0,0],[0,0,1]]))
        np.testing.assert_array_equal(out[:2],anchors[:2])
        np.testing.assert_array_equal(out[2],[0,0,1])
        self.assertEqual(len(out),3)
    def test_constant_map_finite(self):
        self.assertTrue(np.all(np.isfinite(standardized(np.ones((5,5,5))))))
if __name__=='__main__':unittest.main()
