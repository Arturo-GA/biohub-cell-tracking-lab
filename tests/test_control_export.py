import csv
from pathlib import Path
import tempfile
import unittest

from biohub_lab.control_export import repair_export
from biohub_lab.submission import COLUMNS, read_and_validate


class ControlExportTests(unittest.TestCase):
    def source(self,path,z):
        with path.open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(COLUMNS)
            writer.writerows([[0,'v','node',5,0,z,1,1,-1,-1],[1,'v','node',9,1,1,1,1,-1,-1],
                              [2,'v','edge',-1,-1,-1,-1,-1,5,9]])

    def test_upper_rounding_repair_preserves_ids_edges_and_original(self):
        with tempfile.TemporaryDirectory() as temp:
            original=Path(temp)/'original.csv';output=Path(temp)/'corrected.csv';self.source(original,4)
            before=original.read_bytes();report=repair_export(original,output,{'v':(2,4,4,4)})
            nodes,edges=read_and_validate(output,{'v':(2,4,4,4)})['v']
            self.assertEqual(nodes[5]['z'],3);self.assertEqual(edges,[(5,9)])
            self.assertEqual(original.read_bytes(),before);self.assertFalse(report['topology_changed'])

    def test_larger_excursion_and_unexpected_count_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            original=Path(temp)/'original.csv';output=Path(temp)/'corrected.csv';self.source(original,5)
            with self.assertRaisesRegex(ValueError,'excursion'):
                repair_export(original,output,{'v':(2,4,4,4)})
            self.source(original,3)
            with self.assertRaisesRegex(ValueError,'number'):
                repair_export(original,output,{'v':(2,4,4,4)})
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
