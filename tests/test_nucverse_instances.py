import unittest
import numpy as np
from biohub_lab.nucverse_instances import centers_from_fields,matched_truth
class InstancesTest(unittest.TestCase):
    def test_two_attractors_remain_separate(self):
        grid=np.indices((16,32,32)).transpose(1,2,3,0)
        centers=np.array([[8,8,8],[8,24,24]])
        distance=np.linalg.norm(grid[...,None,:]-centers,axis=-1);nearest=distance.argmin(-1)
        fields=np.zeros((16,32,32,4));fields[...,0]=(distance.min(-1)<=3)
        fields[...,1:]=.3*(centers[nearest]-grid)
        actual,report=centers_from_fields(fields)
        self.assertEqual(report['instances'],2)
        self.assertEqual(len(matched_truth(actual,centers,2)),2)
    def test_matching_prefers_cardinality_and_respects_z_scale(self):
        self.assertEqual(len(matched_truth([[0,0,0],[0,0,15]],[[0,0,0],[0,0,-15]])),2)
        self.assertEqual(len(matched_truth([[5,0,0]],[[0,0,0]])),0)
