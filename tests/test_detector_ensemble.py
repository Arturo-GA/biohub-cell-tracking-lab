"""Candidate budget and physical matching invariants for E044."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from ensemble_evaluate_runner import fuse,matched

class EnsembleTests(unittest.TestCase):
    def test_overlapping_sources_do_not_waste_budget(self):
        dense=np.array([[0,0,0],[5,0,0],[9,0,0]])
        static=np.array([[0,0,0],[3,0,0],[7,0,0]])
        points,sources=fuse(dense,static,4)
        self.assertEqual(len(points),4)
        self.assertEqual(len(np.unique(points,axis=0)),4)
        self.assertIn('static',sources)
    def test_one_prediction_cannot_cover_two_annotations(self):
        self.assertEqual(len(matched(np.array([[0,0,0]]),np.array([[0,0,0],[.5,0,0]]),3)),1)
    def test_matching_uses_physical_units(self):
        self.assertEqual(len(matched(np.array([[2,0,0]]),np.array([[0,0,0]]),3)),0)
        self.assertEqual(len(matched(np.array([[2,0,0]]),np.array([[0,0,0]]),7)),1)
    def test_empty_sources(self):
        self.assertEqual(fuse(np.empty((0,3)),np.empty((0,3)))[0].shape,(0,3))
if __name__=='__main__':unittest.main()
