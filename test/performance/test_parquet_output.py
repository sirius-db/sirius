# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import performance_test as benchmark

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

    def test_external_scripts_execute_copy_per_iteration_and_time_only_export(self):
        query = "CREATE VIEW v AS SELECT 42 a; SELECT * FROM v; DROP VIEW v;"
        for precmd in ("nsys", "gdb"):
            with self.subTest(precmd=precmd), tempfile.TemporaryDirectory() as qdir:
                sql_path = benchmark._build_precmd_temp_sql(
                    15,
                    query,
                    "unused.duckdb",
                    2,
                    "none",
                    qdir,
                    precmd,
                    "duckdb",
                    "parquet",
                    "zstd",
                )
                script = Path(sql_path).read_text()
                self.assertEqual(script.count("SET enable_duckdb_fallback = false;"), 2)
                for iteration in range(2):
                    saved = Path(qdir) / f"parquet_iter{iteration}.sql"
                    self.assertIn("COPY (SELECT * FROM v)", saved.read_text())
                    self.assertIn(f"parquet_iter{iteration}", saved.read_text())
                # Run the generated SQL using CPU DuckDB to verify its syntax,
                # Q15 setup/cleanup, iteration paths and timing marker selection.
                cpu_script = "\n".join(
                    line
                    for line in script.splitlines()
                    if not line.startswith(
                        (
                            "LOAD ",
                            "SET gpu_execution",
                            "SET enable_duckdb_fallback",
                            "CALL profiler_",
                        )
                    )
                )
                with duckdb.connect() as con:
                    con.execute(cpu_script)
                    for iteration in range(2):
                        self.assertEqual(
                            con.execute(
                                "SELECT * FROM read_parquet(?)",
                                [
                                    str(
                                        Path(qdir)
                                        / f"parquet_iter{iteration}/*.parquet"
                                    )
                                ],
                            ).fetchall(),
                            [(42,)],
                        )
                    timings = con.execute(
                        "SELECT step FROM read_csv_auto(?)",
                        [str(Path(qdir) / "timings.csv")],
                    ).fetchall()
                    self.assertEqual(timings, [("views",), ("iter_1",), ("iter_2",)])
                with self.assertRaisesRegex(RuntimeError, "non-cuDF"):
                    benchmark._inspect_precmd_exports(qdir, 2, "zstd")

    def test_external_export_refuses_reusing_iteration_paths(self):
        with tempfile.TemporaryDirectory() as qdir:
            (Path(qdir) / "parquet_iter0").mkdir()
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                benchmark._build_precmd_temp_sql(
                    1,
                    "SELECT 1",
                    "unused.duckdb",
                    1,
                    "none",
                    qdir,
                    "gdb",
                    "duckdb",
                    "parquet",
                    "snappy",
                )

    def test_s3_profile_child_exports_each_iteration_with_fallback_disabled(self):
        events = []

        class CpuConnection:
            def __init__(self):
                self.inner = duckdb.connect()

            def extract_statements(self, sql):
                return self.inner.extract_statements(sql)

            def execute(self, sql):
                events.append(sql)
                if sql.startswith(
                    (
                        "SET gpu_execution",
                        "SET enable_duckdb_fallback",
                        "CALL profiler_",
                    )
                ):
                    return self
                self.inner.execute(sql)
                return self

            def fetchall(self):
                return self.inner.fetchall()

            def close(self):
                self.inner.close()

        with tempfile.TemporaryDirectory() as qdir, patch.object(
            benchmark, "S3Input", return_value=Mock()
        ), patch.object(benchmark, "open_connection", return_value=CpuConnection()):
            benchmark._run_nsys_s3_child(
                "s3://bucket/tpch",
                1,
                2,
                "none",
                qdir,
                "SELECT 42 a",
                "parquet",
                "snappy",
            )
            self.assertEqual(sum(sql.startswith("COPY (") for sql in events), 2)
            self.assertEqual(events.count("SET enable_duckdb_fallback = false"), 2)
            for iteration in range(2):
                self.assertTrue(
                    list((Path(qdir) / f"parquet_iter{iteration}").glob("*.parquet"))
                )
            self.assertNotIn(
                "s3://", "".join(path.read_text() for path in Path(qdir).glob("*.sql"))
            )

    def test_wrapper_normal_parser_accepts_export_flags(self):
        wrapper = Path(__file__).with_name("benchmark_and_validate.sh")
        cases = [
            ("parquet", "zstd", "--pinning-mode must"),
            ("bad", "snappy", "--result-format must"),
            ("parquet", "bad", "unsupported --parquet-compression"),
        ]
        for result_format, compression, expected in cases:
            result = subprocess.run(
                [
                    "bash",
                    str(wrapper),
                    "--result-format",
                    result_format,
                    "--parquet-compression",
                    compression,
                    "--pinning-mode",
                    "invalid",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
