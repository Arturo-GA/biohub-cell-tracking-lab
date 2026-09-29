import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

GATE = Path(__file__).resolve().parents[1] / 'kaggle/x138_xr/gate.py'
TARGETS = ','.join(f'video{i}=100/90' for i in range(4))
COUNTS = '\n'.join(f'  video{i}: nodes=100 edges=90 divisions=2' for i in range(4))
VALID = COUNTS + '\nFinal submission.csv rows=760  config=base\n'


class GateTest(unittest.TestCase):
    def run_gate(self, text, tolerance='0.004', targets=TARGETS):
        out = GATE.parents[2] / 'outputs'
        out.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=out, suffix='.log', delete=False) as f:
            p = Path(f.name)
        try:
            p.write_text(text, encoding='utf-8')
            return subprocess.run([sys.executable, str(GATE), str(p), targets, tolerance], capture_output=True).returncode
        finally:
            p.unlink()

    def test_complete(self):
        self.assertEqual(self.run_gate(VALID), 0)

    def test_reject_incomplete_degraded_or_disabled_checks(self):
        for bad in (COUNTS, VALID.replace('760','761'), VALID.replace('base','sweep'),
                    VALID+"repair_fallback: 1\n", VALID+"{'deadline_degraded': 1}\n",
                    VALID+'xr_post_filter skipped (non-fatal): ValueError\n', VALID+VALID):
            with self.subTest(log=bad):
                self.assertNotEqual(self.run_gate(bad), 0)
        self.assertNotEqual(self.run_gate(VALID, tolerance='1e9'), 0)
        self.assertNotEqual(self.run_gate(VALID, targets=TARGETS.replace('100/90','1/1')), 0)


if __name__ == '__main__':
    unittest.main()
