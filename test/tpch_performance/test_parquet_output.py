# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path

import duckdb
from parquet_output import compare_exports, copy_statements, export_metadata


class ParquetOutputTests(unittest.TestCase):
    def test_setup_cleanup_and_quoted_paths(self):
        with tempfile.TemporaryDirectory() as root, duckdb.connect() as con:
            directory = Path(root) / "output's directory"
            setup, sql, cleanup = copy_statements(
                con,
                "CREATE VIEW v AS SELECT 1 a, 'x;y' b; SELECT * FROM v; DROP VIEW v;",
                directory,
                "snappy",
            )
            for statement in setup:
                con.execute(statement)
            completion = con.execute(sql).fetchall()
            metadata = export_metadata(con, directory, completion, "snappy")
            self.assertEqual(metadata["rows"], 1)
            with self.assertRaisesRegex(RuntimeError, "non-cuDF"):
                export_metadata(con, directory, completion, "snappy", require_gpu=True)
            self.assertEqual(
                con.execute(
                    "SELECT b FROM read_parquet(?)", [str(directory / "*.parquet")]
                ).fetchall(),
                [("x;y",)],
            )
            for statement in cleanup:
                con.execute(statement)
            self.assertEqual(len(cleanup), 1)

    def test_comparison_detects_wrong_values_with_the_same_copy_count(self):
        with tempfile.TemporaryDirectory() as root, duckdb.connect() as con:
            left, right = Path(root) / "left", Path(root) / "right"
            for parent, query in ((left, "SELECT 1 AS a"), (right, "SELECT 2 AS a")):
                parent.mkdir()
                _, sql, _ = copy_statements(con, query, parent / "parquet_iter0")
                con.execute(sql)
            with self.assertRaisesRegex(ValueError, "value/multiplicity"):
                compare_exports(left, right)

    def test_empty_export_and_schema_validation(self):
        with tempfile.TemporaryDirectory() as root, duckdb.connect() as con:
            for name in ("left", "right"):
                parent = Path(root) / name
                parent.mkdir()
                directory = parent / "parquet_iter0"
                _, sql, _ = copy_statements(con, "SELECT 1 AS a WHERE false", directory)
                metadata = export_metadata(
                    con, directory, con.execute(sql).fetchall(), "snappy"
                )
                self.assertEqual(metadata["rows"], 0)
                self.assertGreater(metadata["files"], 0)
            compare_exports(Path(root) / "left", Path(root) / "right")

    def test_reject_multiple_result_statements(self):
        with duckdb.connect() as con:
            with self.assertRaises(ValueError):
                copy_statements(con, "SELECT 1; SELECT 2", "/tmp/unused")


if __name__ == "__main__":
    unittest.main()
