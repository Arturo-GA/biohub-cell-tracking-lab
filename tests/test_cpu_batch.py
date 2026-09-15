import sys
from pathlib import Path
import unittest
import json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kaggle_cpu_prepare import select_batch
from run_local_prepared_batch import packages

class CPUBatchTests(unittest.TestCase):
    def test_budget_keeps_contiguous_prefix_and_does_not_skip_large_video(self):
        names=['a','b','c','d'];sizes=dict(a=3,b=8,c=1,d=1)
        self.assertEqual(select_batch(names,0,4,sizes,budget=10),(['a'],3))
        with self.assertRaisesRegex(ValueError,'First video'):
            select_batch(names,1,3,sizes,budget=5)

    def test_bounds_and_empty_store_rejected(self):
        for offset,count in ((-1,1),(0,0),(0,5),(3,2)):
            with self.assertRaises(ValueError):select_batch(['a','b','c','d'],offset,count,{})
        with self.assertRaisesRegex(ValueError,'empty'):
            select_batch(['a'],0,1,{'a':0})

    def test_local_batch_requires_complete_ordered_children(self):
        record=dict(status='complete',videos=['a','b'],completed=['a','b'],offset=0,next_offset=2,
                    source_bytes=20,budget_bytes=30,annotations_read=False,accelerator='none')
        children={'a':dict(status='complete',video='a',offset=0),
                  'b':dict(status='complete',video='b',offset=1)}
        def read(path,*args,**kwargs):
            if path.name=='event_graph_split.json':return json.dumps({'split':{'fit':['a','b'],'calibration':[]}})
            if path.name=='batch_result.json':return json.dumps(record)
            return json.dumps(children[path.parent.name])
        with patch.object(Path,'read_text',read):
            self.assertEqual([p.name for p in packages('outputs/test_batch')],['a','b'])
            children['b']['status']='failed'
            with self.assertRaisesRegex(ValueError,'Child'):packages('outputs/test_batch')
            children['b']['status']='complete';record['completed']=['a']
            with self.assertRaisesRegex(ValueError,'Incomplete'):packages('outputs/test_batch')

if __name__=='__main__':unittest.main()
