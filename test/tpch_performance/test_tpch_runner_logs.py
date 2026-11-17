# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Runner log checks without a GPU or DuckDB build."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class RunnerLogChecks:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tpch-runner-logs-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "test/tpch_performance"
        queries = scripts / "tpch_queries/orig"
        queries.mkdir(parents=True)
        self.runner = scripts / self.runner_name
        shutil.copyfile(Path(__file__).with_name(self.runner_name), self.runner)
        for query in (1, 2):
            (queries / f"q{query}.sql").write_text(f"SELECT {query};\n")
        self.setup_sql = self.root / "setup.sql"
        self.setup_sql.write_text("-- Test views are supplied by the fake shell.\n")
        self.database = self.root / "tpch.duckdb"
        self.database.touch()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.output = self.root / "output"
        self.temp = self.root / "tmp"
        self.temp.mkdir()
        duckdb = self.root / "sirius-duckdb/build/release/duckdb"
        duckdb.parent.mkdir(parents=True)
        duckdb.write_text(
            f"#!{sys.executable}\n"
            "import os, shutil, sys\n"
            "from pathlib import Path\n"
            "for log in Path(os.environ['TPCH_TEST_LOG_DIR']).glob('*.log'):\n"
            "    shutil.copyfile(log, Path(os.environ['SIRIUS_LOG_DIR']) / log.name)\n"
            "sql = Path(sys.argv[sys.argv.index('-f') + 1]).read_text()\n"
            "for line in sql.splitlines():\n"
            "    if line.startswith('.print '):\n"
            "        print(line[7:])\n"
            "    elif line.startswith('SELECT '):\n"
            "        print('result')\n"
            "        print('Run Time (s): real 0.01 user 0.00 sys 0.00')\n"
            "sys.exit(int(os.environ.get('TPCH_TEST_EXIT', '0')))\n"
        )
        duckdb.chmod(0o755)

    @staticmethod
    def begin(query, iteration):
        return (
            f"[info] QueryBegin: instance=0x1 connection=1 query={query * 2 + iteration} "
            f"SQL: SELECT {query};\n"
        )

    @staticmethod
    def complete():
        return "[info] Transparent GPU execution: query completed\n"

    def query_log(self, query):
        return "".join(self.begin(query, i) + self.complete() for i in (1, 2))

    def run_runner(self, first_day, second_day=None, exit_code=0):
        # Create the newer file first so creation order cannot stand in for dates.
        if second_day is not None:
            (self.logs / "sirius_2026-11-09.log").write_text(second_day)
        (self.logs / "sirius_2026-11-08.log").write_text(first_day)
        args = ["bash", str(self.runner)]
        if self.runner_name == "run_tpch_duckdb.sh":
            args += ["--duckdb-file", str(self.database)]
        else:
            args += ["--setup-sql", str(self.setup_sql)]
        args += ["--iterations", "2", "--timeout", "0", "sirius", "1", "1", "2"]
        env = dict(
            os.environ,
            OUTPUT_DIR=str(self.output),
            TMPDIR=str(self.temp),
            TPCH_TEST_LOG_DIR=str(self.logs),
            TPCH_TEST_EXIT=str(exit_code),
        )
        result = subprocess.run(
            args, env=env, capture_output=True, text=True, timeout=15
        )
        self.assertEqual(list(self.temp.iterdir()), [], "merged log was not cleaned up")
        sql = (self.output / "all_queries.sql").read_text()
        self.assertNotIn("SET gpu_execution", sql)
        self.assertNotIn("SET enable_duckdb_fallback", sql)
        return result

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for q in (1, 2):
            self.assertEqual(
                (self.output / f"q{q}/sirius.log").read_text(), self.query_log(q)
            )

    def test_single_day(self):
        self.assert_success(self.run_runner(self.query_log(1) + self.query_log(2)))

    def test_rotation_between_queries(self):
        self.assert_success(self.run_runner(self.query_log(1), self.query_log(2)))

    def test_rotation_during_query(self):
        self.assert_success(
            self.run_runner(
                self.begin(1, 1),
                self.complete()
                + self.begin(1, 2)
                + self.complete()
                + self.query_log(2),
            )
        )

    def test_missing_gpu_completion_does_not_fail_run(self):
        result = self.run_runner(
            self.begin(1, 1), self.begin(1, 2) + self.complete() + self.query_log(2)
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.output / "q1/sirius.log").read_text(),
            self.begin(1, 1) + self.begin(1, 2) + self.complete(),
        )

    def test_cpu_fallback_does_not_fail_run(self):
        fallback = "[info] falling back to DuckDB CPU\n"
        first = "".join(self.begin(1, i) + fallback for i in (1, 2))
        result = self.run_runner(first, self.query_log(2))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "q1/sirius.log").read_text(), first)


class ParquetRunnerLogTests(RunnerLogChecks, unittest.TestCase):
    runner_name = "run_tpch_parquet.sh"

    def test_process_failure_is_preserved(self):
        result = self.run_runner(self.query_log(1), self.query_log(2), exit_code=42)
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)


class DuckdbRunnerLogTests(RunnerLogChecks, unittest.TestCase):
    runner_name = "run_tpch_duckdb.sh"


if __name__ == "__main__":
    unittest.main()
