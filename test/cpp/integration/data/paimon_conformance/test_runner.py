"""Failure-oriented tests for the reference harness; no Paimon/GPU dependency."""

import copy
from decimal import Decimal
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from run_conformance import (
    ProcessError,
    compare_rows,
    decode_results,
    execute,
    expect_planning_rejection,
    query_result,
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

    def query_output(self, documents):
        extension = {"version": "5e89198", "duckdb_version": "v1.5.5"}
        results = [
            [{"extension_version": extension["version"]}],
            [{"version": extension["duckdb_version"]}],
            [{"gpu": False}],
            [{"column_name": "id", "column_type": "BIGINT"}],
            *documents,
            [{"probe": "LIVENESS_OK"}],
        ]
        return "\n".join(json.dumps(r) for r in results), extension, [["id", "BIGINT"]]

    def test_explicit_empty_query_result_passes(self):
        self.assertEqual(query_result(*self.query_output([[]])), [])

    def test_missing_query_result_fails(self):
        with self.assertRaisesRegex(AssertionError, "exactly six SQL results"):
            query_result(*self.query_output([]))

    def test_extra_query_result_fails(self):
        with self.assertRaisesRegex(AssertionError, "exactly six SQL results"):
            query_result(*self.query_output([[], []]))

    def test_nonempty_query_result_passes(self):
        rows = [{"id": 42}]
        self.assertEqual(query_result(*self.query_output([rows])), rows)

    def test_query_identity_and_liveness_are_checked(self):
        output, extension, columns = self.query_output([[]])
        for index in (0, 1, 2, 3, 5):
            with self.subTest(index=index), self.assertRaises(AssertionError):
                results = json.loads("[" + output.replace("\n", ",") + "]")
                results[index] = []
                query_result(
                    "\n".join(json.dumps(r) for r in results), extension, columns
                )

    def rejection_process(self, ending, diagnostic=None):
        if diagnostic is None:
            diagnostic = (
                "GPU plan generation failed: Table function 'paimon_scan' "
                "is not supported in Sirius"
            )
        script = (
            f"import sys\nprint({diagnostic!r}, file=sys.stderr, flush=True)\n{ending}"
        )
        with tempfile.TemporaryDirectory() as tmp:
            expect_planning_rejection(
                lambda: execute([sys.executable, "-c", script], timeout=5, cwd=tmp)
            )

    def test_normal_planning_rejection_passes(self):
        self.rejection_process("sys.exit(1)")

    def test_unexpected_exit_with_matching_diagnostic_fails(self):
        for code in (2, 7, 139):
            with self.subTest(code=code), self.assertRaises(ProcessError) as error:
                self.rejection_process(f"sys.exit({code})")
            self.assertEqual(error.exception.returncode, code)

    @unittest.skipUnless(os.name == "posix", "POSIX signal exit status")
    def test_crash_with_matching_diagnostic_fails(self):
        with self.assertRaises(ProcessError) as error:
            self.rejection_process(
                "import os, resource, signal\n"
                "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
                "os.kill(os.getpid(), signal.SIGSEGV)"
            )
        self.assertEqual(error.exception.returncode, -signal.SIGSEGV)

    def test_unrelated_sql_error_is_not_expected_rejection(self):
        with self.assertRaises(ProcessError):
            self.rejection_process("sys.exit(1)", "Binder Error: missing table")

    def test_success_with_matching_diagnostic_is_not_rejection(self):
        with self.assertRaisesRegex(AssertionError, "Expected planning-time rejection"):
            self.rejection_process("sys.exit(0)")

    def test_timeout_preserves_partial_logs(self):
        for stdout, stderr in [
            (b"out-marker", b"err-marker"),
            ("out-marker", "err-marker"),
            (None, None),
        ]:
            with self.subTest(stdout=stdout), tempfile.TemporaryDirectory() as tmp:
                log = Path(tmp) / "timeout"
                failure = subprocess.TimeoutExpired(
                    ["fake"], 1, output=stdout, stderr=stderr
                )
                with patch(
                    "run_conformance.subprocess.run", side_effect=failure
                ), self.assertRaisesRegex(RuntimeError, "timed out"):
                    execute(["fake"], timeout=1, cwd=tmp, log=log)
                self.assertEqual(
                    log.with_suffix(".stdout").read_text(),
                    "out-marker" if stdout else "",
                )
                error = log.with_suffix(".stderr").read_text()
                self.assertIn("timed out", error)
                if stderr:
                    self.assertIn("err-marker", error)

    def test_real_timeout_preserves_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "timeout"
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                execute(
                    [
                        sys.executable,
                        "-c",
                        "import sys,time; print('out-marker',flush=True); print('err-marker',file=sys.stderr,flush=True); time.sleep(10)",
                    ],
                    timeout=1,
                    cwd=tmp,
                    log=log,
                )
            self.assertIn("out-marker", log.with_suffix(".stdout").read_text())
            self.assertIn("err-marker", log.with_suffix(".stderr").read_text())

    def test_invalid_decimal_categories_fail(self):
        case = {"columns": [["amount", "DECIMAL(12,2)"]], "rows": [["1.00"]]}
        for value in (True, False, 1.0, "NaN", "Infinity"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_rows(case, [{"amount": value}])

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
