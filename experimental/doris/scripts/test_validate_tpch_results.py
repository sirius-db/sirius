"""Regression checks for result validation, independent of a dataset or GPU."""

import argparse
import decimal
import tempfile
import unittest
from pathlib import Path

from validate_tpch_results import Result, Summary, Tolerance, Verdict, cmd_validate, compare, scalars_match


class ValidatorTest(unittest.TestCase):
    def test_missing_results_fail_the_run(self):
        summary = Summary()
        summary.add(Verdict("q06", "SKIPPED", "no plan"))
        self.assertEqual(summary.finish(None), 1)

    def test_integer_truncation_is_not_hidden_by_actual_scale(self):
        self.assertFalse(scalars_match("26", "25.52", Tolerance(ulps=decimal.Decimal(1))))
        self.assertEqual(
            compare(
                "q06",
                "select revenue from lineitem",
                Result(["revenue"], [["26"]]),
                Result(["revenue"], [["25.52"]]),
                Tolerance(ulps=decimal.Decimal(1)),
            ).status,
            "MISMATCH",
        )

    def test_q01_avg_allows_declared_decimal_cast_only(self):
        actual = Result(["sum_qty", "avg_qty"], [["25", "25.5220"]])
        expected = Result(["sum_qty", "avg_qty"], [["25", "25.522005853257337"]])
        self.assertEqual(compare("q01", "select sum_qty, avg_qty", actual, expected, Tolerance()).status, "OK")
        actual.rows[0][0] = "26"
        self.assertEqual(compare("q01", "select sum_qty, avg_qty", actual, expected, Tolerance()).status, "MISMATCH")

    def test_q08_share_allows_one_unit_at_declared_scale_only(self):
        actual = Result(["o_year", "mkt_share"], [["1995", "0.02864874"]])
        expected = Result(["o_year", "mkt_share"], [["1995", "0.028648741305617557"]])
        self.assertEqual(compare("q08", "select o_year, mkt_share", actual, expected, Tolerance()).status, "OK")
        actual.rows[0][1] = "0.01825027"
        expected.rows[0][1] = "0.018250279107962147"
        self.assertEqual(compare("q08", "select o_year, mkt_share", actual, expected, Tolerance()).status, "OK")
        actual.rows[0][1] = "0.02864873"
        expected.rows[0][1] = "0.028648741305617557"
        self.assertEqual(compare("q08", "select o_year, mkt_share", actual, expected, Tolerance()).status, "MISMATCH")

    def test_mysql_empty_result_uses_expected_header_and_reports_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sql = root / "sql"
            expected = root / "expected"
            actual = root / "actual"
            sql.mkdir()
            expected.mkdir()
            (actual / "empty").mkdir(parents=True)
            (sql / "empty.sql").write_text("select id from t where false;")
            (expected / "empty.tsv").write_text("id\n")
            (actual / "empty" / "result.tsv").write_text("")
            args = argparse.Namespace(
                actual=actual, expected=expected, sql_dir=sql, queries="empty",
                tolerance=decimal.Decimal("1e-9"), ulps=decimal.Decimal("0.5"), csv=None,
            )
            self.assertEqual(cmd_validate(args), 0)
            (actual / "empty" / "error.txt").write_text("query failed\n")
            self.assertEqual(cmd_validate(args), 1)


if __name__ == "__main__":
    unittest.main()
