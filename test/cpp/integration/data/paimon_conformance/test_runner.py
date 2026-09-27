"""Failure-oriented tests for the reference harness; no Paimon/GPU dependency."""

import copy
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest

from run_conformance import (
    compare_rows,
    decode_results,
    execute,
    scan_sql,
    select_cases,
)


class ReferenceHarnessTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "id": "duplicates",
            "columns": [["id", "BIGINT"], ["amount", "DECIMAL(12,2)"]],
            "rows": [[1, "10.00"], [1, "10.00"], [2, None]],
        }

    def test_missing_duplicate_fails(self):
        with self.assertRaises(AssertionError):
            compare_rows(
                self.case,
                [{"id": 1, "amount": Decimal("10.00")}, {"id": 2, "amount": None}],
            )

    def test_null_is_not_empty_string(self):
        case = {"columns": [["label", "VARCHAR"]], "rows": [[None], [""]]}
        with self.assertRaises(AssertionError):
            compare_rows(case, [{"label": ""}, {"label": ""}])

    def test_decimal_precision_is_preserved(self):
        rows = decode_results('[{"v":12345678901234567890.01}]')[0]
        compare_rows(
            {
                "columns": [["v", "DECIMAL(38,2)"]],
                "rows": [["12345678901234567890.01"]],
            },
            rows,
        )
        with self.assertRaises(AssertionError):
            compare_rows(
                {
                    "columns": [["v", "DECIMAL(38,2)"]],
                    "rows": [["12345678901234567890.02"]],
                },
                rows,
            )

    def test_unordered_results_keep_multiplicity(self):
        compare_rows(
            self.case,
            [
                {"id": 2, "amount": None},
                {"id": 1, "amount": 10},
                {"id": 1, "amount": 10},
            ],
        )

    def test_ordered_results_detect_wrong_order(self):
        case = copy.deepcopy(self.case)
        case["ordered"] = True
        with self.assertRaises(AssertionError):
            compare_rows(
                case,
                [
                    {"id": 2, "amount": None},
                    {"id": 1, "amount": 10},
                    {"id": 1, "amount": 10},
                ],
            )

    def test_extra_column_is_rejected(self):
        with self.assertRaises(AssertionError):
            compare_rows(self.case, [{"id": 1, "amount": 10, "_VALUE_KIND": 0}])

    def test_unknown_or_zero_cases_fail(self):
        for cases, requested in [
            ([], []),
            ([self.case], ["typo"]),
            ([self.case, self.case], []),
        ]:
            with self.subTest(cases=cases, requested=requested), self.assertRaises(
                ValueError
            ):
                select_cases(cases, requested)

    def test_invalid_snapshot_fails(self):
        for bad in (0, -1, "1; SELECT 1", True):
            with self.subTest(snapshot=bad), self.assertRaises(ValueError):
                scan_sql("/tmp/table", bad)

    def test_sql_paths_escape_quotes(self):
        self.assertEqual(
            scan_sql("/tmp/aaron's/table", 4),
            "paimon_scan('/tmp/aaron''s/table', snapshot_from_id=4)",
        )

    def test_nonzero_process_is_not_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            RuntimeError, "exited 7"
        ):
            execute(
                [sys.executable, "-c", "import sys; sys.exit(7)"],
                timeout=5,
                cwd=tmp,
                log=Path(tmp) / "failure",
            )

    def test_process_timeout_fails(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            RuntimeError, "timed out"
        ):
            execute(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout=0.05,
                cwd=tmp,
            )

    def test_missing_executable_fails(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            execute([Path(tmp) / "missing"], timeout=5, cwd=tmp)

    def test_malformed_output_is_not_ignored(self):
        with self.assertRaises(ValueError):
            decode_results('warning or corrupted output\n[{"v":1}]')


if __name__ == "__main__":
    unittest.main()
