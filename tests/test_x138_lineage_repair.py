import importlib.util
from pathlib import Path
import unittest

p=Path(__file__).resolve().parents[1]/'kaggle/x138_xr/lineage_repair.py'
spec=importlib.util.spec_from_file_location('lineage',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class LineageTest(unittest.TestCase):
    def graph(self):
        ns={0:dict(t=0,z=0,y=0,x=0),1:dict(t=1,z=0,y=0,x=1),
            2:dict(t=1,z=0,y=0,x=20),3:dict(t=2,z=0,y=0,x=2)}
        es=[dict(source_id=0,target_id=1),dict(source_id=0,target_id=2)]
        return ns,es

    def test_extend_without_new_nodes_or_mutation(self):
        ns,es=self.graph();out,edges=m.extend_daughters(ns,es)
        self.assertEqual(out,ns);self.assertEqual(len(es),2)
        self.assertEqual([(e['source_id'],e['target_id']) for e in edges],[(0,1),(0,2),(1,3)])

    def test_no_stealing_or_nonmutual_or_outside_radius(self):
        ns,es=self.graph();es.append(dict(source_id=2,target_id=3))
        self.assertEqual(m.extend_daughters(ns,es)[1],es)
        ns,es=self.graph();ns[4]=dict(t=1,z=0,y=0,x=2)
        self.assertEqual(m.extend_daughters(ns,es)[1],es)
        ns,es=self.graph();ns[3]['x']=100
        self.assertEqual(m.extend_daughters(ns,es)[1],es)


if __name__=='__main__':unittest.main()
