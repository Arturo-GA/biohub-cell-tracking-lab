import unittest
import numpy as np
from dense_detector_evaluate_runner import link

class DenseTrackingTests(unittest.TestCase):
    def test_competing_sources_cannot_share_child(self):
        coords=np.array([[0,0,0,0],[0,0,0,1],[1,0,0,0],[2,0,0,0]],float)
        keep,edges=link(coords,minimum_length=3)
        self.assertEqual(len(keep),3)
        self.assertEqual(len(edges),2)
        self.assertEqual(len({b for a,b in edges}),len(edges))

    def test_physical_gate_and_temporal_gap(self):
        # Nine axial voxels = 14.625um, outside radius14, despite short voxel distance.
        coords=np.array([[0,0,0,0],[1,9,0,0],[3,9,0,0]],float)
        keep,edges=link(coords,minimum_length=2)
        self.assertEqual(len(keep),0)
        self.assertEqual(edges,[])

    def test_full_six_frame_track_survives(self):
        coords=np.array([[t,0,0,t] for t in range(6)],float)
        keep,edges=link(coords)
        self.assertEqual(len(keep),6);self.assertEqual(len(edges),5)

if __name__=='__main__':unittest.main()
