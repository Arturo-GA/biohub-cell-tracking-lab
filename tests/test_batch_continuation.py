import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from continue_downloaded_batch import validate_prepared_next


class BatchContinuationTests(unittest.TestCase):
    def setUp(self):
        self.plan=dict(offset=6,notebook_sha256=hashlib.sha256(b'expected notebook').hexdigest())

    def test_accepts_reviewed_code_and_expected_next_offset(self):
        with patch.object(Path,'read_text',return_value=json.dumps({'code_file':'notebook.ipynb'})),\
             patch.object(Path,'read_bytes',return_value=b'expected notebook'):
            validate_prepared_next({'next_offset':6},'outputs/fake_upload',self.plan)

    def test_rejects_changed_code_or_wrong_next_offset(self):
        with patch.object(Path,'read_text',return_value=json.dumps({'code_file':'notebook.ipynb'})):
            for code,offset in ((b'changed',6),(b'expected notebook',10)):
                with patch.object(Path,'read_bytes',return_value=code),self.assertRaises(ValueError):
                    validate_prepared_next({'next_offset':offset},'outputs/fake_upload',self.plan)

    def test_rejects_code_file_outside_prepared_directory(self):
        with patch.object(Path,'read_text',return_value=json.dumps({'code_file':'../outside.ipynb'})):
            with self.assertRaises(ValueError):
                validate_prepared_next({'next_offset':6},'outputs/fake_upload',self.plan)


if __name__=='__main__':unittest.main()
