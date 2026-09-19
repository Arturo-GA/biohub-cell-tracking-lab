import csv,tempfile,unittest
from pathlib import Path
from biohub_lab.calibration_export import export_control
from biohub_lab.submission import COLUMNS

class ExportTests(unittest.TestCase):
    def run_export(self,x):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'original.csv';target=p/'validated.csv'
            with source.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(COLUMNS)
                w.writerow([0,'video','node',0,0,0,0,x,-1,-1])
                w.writerow([1,'video','node',1,1,0,0,0,-1,-1])
                w.writerow([2,'video','edge',-1,-1,-1,-1,-1,0,1])
            before=source.read_bytes();r=export_control(source,target,{'video':(2,2,2,2)})
            self.assertEqual(source.read_bytes(),before);self.assertEqual(r['edges'],1)
            return r
    def test_valid_untouched(self):self.assertEqual(self.run_export(1)['changed_spatial_coordinates'],{})
    def test_upper_border_only(self):self.assertEqual(self.run_export(2)['changed_spatial_coordinates'],{'x':1})
    def test_reject_other_excursions(self):
        for x in [-1,3,.5]:
            with self.subTest(x=x),self.assertRaises(ValueError):self.run_export(x)
if __name__=='__main__':unittest.main()
