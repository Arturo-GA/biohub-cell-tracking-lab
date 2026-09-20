import unittest
import numpy as np
from biohub_lab.point_supervision import supervision
class PointSupervisionTest(unittest.TestCase):
    def test_unlabeled_bright_cell_is_not_negative(self):
        image=np.ones((8,16,16));image[0]=0
        labels=supervision(image,[[4,4,4]])
        self.assertTrue(labels['positive'][4,4,4]);self.assertFalse(labels['negative'][4,12,12])
        self.assertTrue(labels['negative'][0,12,12]);self.assertFalse(np.any(labels['positive']&labels['negative']))
        self.assertTrue(np.all(labels['flow'][~labels['flow_mask']]==0))
        bright=supervision(np.ones((8,16,16)),[[4,4,4]])
        self.assertFalse(bright['negative'].any())
