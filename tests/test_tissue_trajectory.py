import unittest
import numpy as np
from biohub_lab.tissue_trajectory import tissue_flow,track,objective,edge_utilities

class TissueTrajectoryTests(unittest.TestCase):
    def moving_cells(self):
        coords=np.array([[t,0,j*30,t*2] for t in range(5) for j in range(4)],float)
        edges=np.array([[t*4+j,(t+1)*4+j] for t in range(4) for j in range(4)])
        return coords,edges
    def test_collective_translation(self):
        coords,edges=self.moving_cells();flow,_=tissue_flow(coords)
        np.testing.assert_allclose(flow[coords[:,0]<4],np.tile([0,0,.8125],(16,1)),atol=1e-6)
    def test_preserves_coherent_tracks_and_capacity(self):
        coords,edges=self.moving_cells();chosen,report=track(coords,edges)
        self.assertEqual(set(map(tuple,chosen)),set(map(tuple,edges)))
        self.assertTrue(all(v['objective_nondecreasing'] for v in report['starts'].values()))
        self.assertEqual(len(np.unique(chosen[:,0])),len(chosen));self.assertEqual(len(np.unique(chosen[:,1])),len(chosen))
    def test_repairs_swapped_trajectories(self):
        coords,edges=self.moving_cells();wrong=edges.copy();wrong[4:6,1]=wrong[4:6,1][::-1]
        chosen,report=track(coords,wrong)
        self.assertEqual(set(map(tuple,chosen)),set(map(tuple,edges)))
        self.assertGreater(report['starts']['geometric']['final_objective'],report['starts']['geometric']['initial_objective'])
    def test_local_utility_matches_full_objective_change(self):
        coords,edges=self.moving_cells();coords[10,3]+=1.;flow,_=tissue_flow(coords)
        index=6;pair=edges[index:index+1];remaining=np.delete(edges,index,axis=0)
        utility=edge_utilities(coords,flow,pair,edges)[0]
        distance=np.linalg.norm((coords[pair[0,1],1:]-coords[pair[0,0],1:])*[1.625,.40625,.40625])
        self.assertAlmostEqual(objective(coords,flow,edges)-objective(coords,flow,remaining),utility+.01/(1+distance),places=6)
if __name__=='__main__':unittest.main()
