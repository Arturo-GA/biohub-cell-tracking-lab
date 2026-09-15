import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_event_video import local_detector_tree


class LocalDetectorPathTests(unittest.TestCase):
    def test_only_log_path_changes_and_inference_statements_are_preserved(self):
        source="from pathlib import Path\nlog=Path('/kaggle/working')/'guard.jsonl'\nweights=Path('model.pth')\nvalue=3*7\n"
        directory=Path('outputs/local-test-logs').resolve()
        tree=local_detector_tree(source,'detector.py',directory)
        namespace={};exec(compile(tree,'detector.py','exec'),namespace)
        self.assertEqual(namespace['log'],directory/'guard.jsonl')
        self.assertEqual(namespace['weights'],Path('model.pth'))
        self.assertEqual(namespace['value'],21)
        # Reverting the one constant must reproduce the entire original AST.
        constants=[n for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value==str(directory)]
        self.assertEqual(len(constants),1)
        constants[0].value='/kaggle/working'
        self.assertEqual(ast.dump(tree),ast.dump(ast.parse(source)))

    def test_changed_upstream_path_contract_is_rejected(self):
        for source in ("x=1", "a=Path('/kaggle/working');b=Path('/kaggle/working')"):
            with self.assertRaisesRegex(ValueError,'exactly one'):
                local_detector_tree(source,'detector.py','.')


if __name__=='__main__':unittest.main()
