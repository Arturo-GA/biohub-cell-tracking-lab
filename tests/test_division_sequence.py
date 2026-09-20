import unittest
import numpy as np
from biohub_lab.division_sequence import candidates,features,labels,decode,descriptors

class DivisionSequenceTests(unittest.TestCase):
    def setUp(self):
        self.coords=np.array([[0,10,20,20],[1,10,20,20],[2,10,20,20],[3,10,20,16],[4,10,20,14],[5,10,20,12],[3,10,20,24],[4,10,20,26],[5,10,20,28]],float)
        self.edges=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[6,7],[7,8]])
    def test_orphan_daughter_context_and_symmetry(self):
        triples,c=candidates(self.coords,self.edges,True);self.assertEqual(triples.tolist(),[[2,3,6]])
        h=np.random.default_rng(1).normal(size=(9,6))
        a=features(self.coords,h,c);b=features(self.coords,h,c[:,[0,1,2,6,7,8,3,4,5]])
        for x,y in zip(a,b):np.testing.assert_allclose(x,y)
        self.assertEqual([x.shape[1] for x in a],[22,18,54])
    def test_sparse_labels_and_no_stealing(self):
        triples=np.array([[2,3,6]]);truth=np.vstack([self.edges,[2,6]])
        self.assertEqual(labels(triples,np.arange(9),truth).tolist(),[1])
        ids=np.arange(9);ids[6]=-1;self.assertEqual(labels(triples,ids,truth).tolist(),[-1])
        e,changes=decode(self.edges,triples,np.array([.9]),.8,self.coords);self.assertEqual(len(e),8);self.assertEqual(len(changes),1)
        protected=np.vstack([self.edges,[1,6]])
        self.assertEqual(len(decode(protected,triples,np.array([.9]),.8,self.coords)[1]),0)
    def test_image_signal(self):
        v=np.zeros((16,16,16),np.float32);v[6:10,6:10,6:10]=1
        f=descriptors(v,np.array([[8,32,32],[2,8,8]]));self.assertEqual(f.shape,(2,6));self.assertTrue(np.isfinite(f).all());self.assertGreater(f[0,1],f[1,1])
if __name__=='__main__':unittest.main()
