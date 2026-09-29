"""Conditional leaderboard sends must not consume quota before valid results."""
import sys
import unittest
from decimal import InvalidOperation
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'kaggle/x138_xr'))
from e070_submit import require_unfavorable_scores


def result(score, status='COMPLETE'):
    return SimpleNamespace(public_score=score, status=SimpleNamespace(name=status))


class ConditionalSubmissionGate(unittest.TestCase):
    def test_equal_scores_allow_requested_fallback(self):
        self.assertEqual(require_unfavorable_scores([result('0.955'), result('0.954')], .955), ['0.955', '0.954'])

    def test_any_positive_improvement_preserves_quota(self):
        for rows in ([result('0.956'), result('0.954')], [result('0.955'), result('0.9551')]):
            with self.subTest(rows=rows), self.assertRaises(AssertionError):
                require_unfavorable_scores(rows, .955)

    def test_pending_or_error_is_not_a_bad_score(self):
        for state in ('PENDING', 'RUNNING', 'ERROR'):
            with self.subTest(state=state), self.assertRaises(AssertionError):
                require_unfavorable_scores([result('0.954'), result('0', state)], .955)

    def test_missing_or_nonfinite_results_preserve_quota(self):
        for rows in ([None, result('0.954')], [result(''), result('0.954')],
                     [result('NaN'), result('0.954')], [result('Infinity'), result('0.954')]):
            with self.subTest(rows=rows), self.assertRaises((AssertionError, InvalidOperation)):
                require_unfavorable_scores(rows, .955)


if __name__ == '__main__':
    unittest.main()
