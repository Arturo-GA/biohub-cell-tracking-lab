import unittest
from biohub_lab.anchored_tracklets import recover
def node(t,x=0):return dict(t=t,z=0,y=0,x=x)
class TrackletTests(unittest.TestCase):
    def test_bridge_missing_cells(self):
        ref={0:node(0),4:node(4)};donor={i:node(i) for i in range(5)}
        n,e,r=recover(ref,[],donor,[(i,i+1) for i in range(4)])
        self.assertEqual(len(e),4);self.assertEqual(r['added_nodes'],3);self.assertEqual(r['counts']['accepted_bridges'],1)
    def test_preserve_divisions(self):
        ref={0:node(0),1:node(1,10),2:node(1,-10)};old=[(0,1),(0,2)]
        donor={i:node(i) for i in range(5)}
        n,e,r=recover(ref,old,donor,[(i,i+1) for i in range(4)])
        self.assertEqual(set(e),set(old));self.assertEqual(n,ref)
    def test_reject_short_extension(self):
        n,e,r=recover({0:node(0)},[],{i:node(i) for i in range(3)},[(0,1),(1,2)])
        self.assertEqual(e,[])
    def test_block_duplicate_near_reference(self):
        ref={0:node(0),10:node(1,4)};donor={i:node(i) for i in range(5)}
        n,e,r=recover(ref,[],donor,[(i,i+1) for i in range(4)])
        self.assertEqual(e,[]);self.assertEqual(r['blocked_near_reference'],1)
if __name__=='__main__':unittest.main()
