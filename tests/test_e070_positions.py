import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'kaggle/x138_xr'))
import context_rules as context
import e070_positions as module
from e070_variants import variant
module.ctx_graph=context.ctx_graph
module.topology_smoothing=context.topology_smoothing


def graph(xs):
    nodes={i:dict(t=i,z=0.,y=0.,x=float(x)/.40625) for i,x in enumerate(xs)}
    edges=[dict(source_id=i,target_id=i+1) for i in range(len(xs)-1)]
    return nodes,edges


class PositionTests(unittest.TestCase):
    def test_curvature_protects_coherent_bend(self):
        ns,es=graph([0,1,4,9,16]);smooth={k:dict(v) for k,v in ns.items()};smooth[2]['x']=6/.40625
        out,n=module.e070_positions(ns,es,smooth,dict(position_mode='curvature'))
        self.assertEqual(n,1);self.assertAlmostEqual(out[2]['x']*.40625,4.875)

    def test_straight_trajectory_unchanged(self):
        ns,es=graph([0,1,2,3,4])
        for mode in ('curvature','outlier'):
            out,n=module.e070_positions(ns,es,ns,dict(position_mode=mode));self.assertEqual(n,0);self.assertEqual(out,ns)

    def test_outlier_requires_bilateral_agreement_and_bounded_shift(self):
        ns,es=graph([0,1,7,3,4]);out,n=module.e070_positions(ns,es,ns,dict(position_mode='outlier'))
        self.assertEqual(n,1);self.assertAlmostEqual(out[2]['x']*.40625,5.)
        self.assertAlmostEqual(ns[2]['x']*.40625,7.)

    def test_fork_protected(self):
        ns,es=graph([0,1,7,3,4]);ns[8]=dict(t=3,z=0.,y=0.,x=0.);es.append(dict(source_id=2,target_id=8))
        out,n=module.e070_positions(ns,es,ns,dict(position_mode='outlier'));self.assertEqual(n,0);self.assertEqual(out,ns)

    def test_configuration_isolation(self):
        a=variant('F8_joint');a['xr']['prune']='oops'
        self.assertNotEqual(variant('F8_joint')['xr']['prune'],'oops')


if __name__=='__main__':unittest.main()
