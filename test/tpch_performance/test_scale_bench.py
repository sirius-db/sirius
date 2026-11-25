# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""CPU fixtures verify workload semantics and lifecycle, not GPU qualification."""

import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import duckdb

import performance_test as benchmark
from scale_bench import (
    ScaleBench,
    SCHEMAS,
    SQL_QUERIES,
    TABLE_DIRECTORIES,
    parse_queries,
)


HERE = Path(__file__).resolve().parent


def write_fixture(root):
    """Duplicate join keys, nested payloads, a directed cycle and a disconnected edge."""
    with duckdb.connect() as con:
        for table, schema in SCHEMAS.items():
            columns = ",".join(f'"{name}" {typ}' for name, typ in schema.items())
            con.execute(f"CREATE TABLE {table} ({columns})")

        con.execute(
            """INSERT INTO table_a VALUES
            (1,1000,10,1001,'1'),(1,1000,10,1001,'1'),
            (2,1001,20,1002,'2'),(3,1002,30,1000,'none'),(4,2000,40,2001,'4')"""
        )
        for table in ("table_b", "table_c", "table_d", "table_e"):
            defaults = {
                name: (
                    [1, 2]
                    if typ == "INTEGER[]"
                    else (
                        ["x", "y"]
                        if typ == "VARCHAR[]"
                        else "payload" if typ == "VARCHAR" else 0
                    )
                )
                for name, typ in SCHEMAS[table].items()
            }
            overrides = {
                "table_b": {
                    "col_b_8": 1,
                    "col_b_3": 1000,
                    "col_b_9": 10,
                    "col_b_1": 1001,
                },
                "table_c": {"col_c_10": 1, "col_c_9": 1000, "col_c_11": 1},
                "table_d": {"col_d_0": 1, "col_d_1": 10},
                "table_e": {"col_e_0": 1},
            }[table]
            defaults.update(overrides)
            values = list(defaults.values())
            # Two dimension matches multiply the first two input rows twice.
            for _ in range(2 if table in ("table_b", "table_e") else 1):
                con.execute(
                    f"INSERT INTO {table} VALUES ({','.join('?' for _ in values)})",
                    values,
                )
        for table, directory in TABLE_DIRECTORIES.items():
            folder = root / directory
            folder.mkdir(parents=True)
            path = str(folder / "part.parquet").replace("'", "''")
            con.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET)")


def run_cli(source, output, *extra, check=True):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("SIRIUS_CONFIG_FILE", "TPCH_RESULT_FORMAT", "SIRIUS_PRE_SQL")
    }
    return subprocess.run(
        [
            sys.executable,
            str(HERE / "performance_test.py"),
            "--benchmark",
            "scale_bench",
            "--input",
            str(source),
            "--engine",
            "cpu",
            "--output",
            str(output),
            *extra,
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
        check=check,
    )


class ScaleBenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="scale-bench-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "input's data"
        write_fixture(self.source)

    def prepared(self, names):
        run = self.root / "run"
        (run / "queries").mkdir(parents=True, exist_ok=True)
        (run / "metadata.json").write_text("{}\n")
        workload = ScaleBench(self.source, names, self.root / "scratch")
        workload.iterations = 2
        return run, workload

    def test_cli_all_queries_twice_preserves_counts_graphs_and_cleanup(self):
        output, scratch = self.root / "output", self.root / "scratch"
        scratch.mkdir()
        sentinel = scratch / "keep-me.txt"
        sentinel.write_text("user data")
        result = run_cli(
            self.source, output, "--iterations", "2", "--temp-dir", str(scratch)
        )
        self.assertIn("Benchmark run complete", result.stdout)
        run = next(output.iterdir())
        metadata = json.loads((run / "metadata.json").read_text())
        self.assertEqual(metadata["benchmark"], "scale_bench")
        self.assertIsNone(metadata["scale_factor"])
        self.assertEqual(metadata["scale_bench"]["temporary_cleanup"], "removed")
        self.assertFalse(Path(metadata["scale_bench"]["temporary_directory"]).exists())
        self.assertEqual(list(scratch.iterdir()), [sentinel])
        expected = {
            "sort1": 5,
            "sort2": 5,
            "gb1": 4,
            "gb2": 4,
            "join1": 7,
            "workflow": 11,
            "bfs": 3,
        }
        for name, count in expected.items():
            rows = benchmark._load_result_file(run / "duckdb" / name / "result.txt")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], count, name)
        self.assertEqual(
            benchmark._load_result_file(run / "duckdb/bfs/result.txt"), [(3, 3003, 3)]
        )
        for name in ("gb1", "gb2"):
            self.assertEqual(
                benchmark._load_result_file(run / "duckdb" / name / "result.txt")[0][
                    -1
                ],
                5,
            )
        skipped = metadata["scale_bench"]["queries"]["workflow"]["skipped_columns"]
        self.assertEqual(len(skipped), 5)
        with (run / "csv/runtimes.csv").open() as stream:
            timings = list(csv.DictReader(stream))
        self.assertEqual(sum(r["query"] != "TOTAL" for r in timings), 16)
        self.assertEqual(
            {r["query"] for r in timings}, set(parse_queries(None)) | {"TOTAL"}
        )
        self.assertEqual(
            benchmark._resolve_duckdb_results_dir(str(run), ["gb2"]),
            str(run / "duckdb"),
        )
        shutil.copytree(run / "duckdb", run / "sirius")
        scope = type("Scope", (), {"iterations": 2})()
        checks = benchmark.validate(
            str(run / "sirius"), str(run / "duckdb"), parse_queries(None), scope
        )
        self.assertTrue(all(r["status"] == "success" for r in checks.values()))
        (run / "sirius/gb2/result.txt").write_text("(999,)\n")
        self.assertEqual(
            benchmark.validate(str(run / "sirius"), str(run / "duckdb"), ["gb2"])[
                "gb2"
            ]["status"],
            "validation",
        )

    def test_prepared_cast_matches_numeric_string_and_retains_nested_payload(self):
        run, workload = self.prepared(["workflow"])
        with workload:
            workload.prepare(run)
            with benchmark.open_connection(str(self.source), workload=workload) as con:
                self.assertEqual(
                    dict(
                        (r[0], r[1])
                        for r in con.execute("DESCRIBE table_c_prep").fetchall()
                    )["col_c_11"],
                    "VARCHAR",
                )
                self.assertEqual(
                    con.execute("SELECT col_c_11 FROM table_c_prep").fetchall(),
                    [("1",)],
                )
                self.assertEqual(
                    con.execute("SELECT col_c_1 FROM table_c_prep").fetchall(),
                    [([1, 2],)],
                )
                rows = con.execute(SQL_QUERIES["workflow"]).fetchall()
                self.assertEqual(len(rows), 11)
                self.assertEqual(sum(row[0] == 1 for row in rows), 8)

    def test_parquet_and_rows_use_existing_output_paths(self):
        for fmt in ("rows", "parquet"):
            output = self.root / fmt
            run_cli(
                self.source, output, "--queries", "workflow", "--result-format", fmt
            )
            run = next(output.iterdir())
            if fmt == "parquet":
                with duckdb.connect() as con:
                    rows = con.execute(
                        "SELECT count(*) FROM read_parquet(?)",
                        [str(run / "duckdb/workflow/parquet_iter0/*.parquet")],
                    ).fetchone()[0]
                self.assertEqual(rows, 11)
            else:
                self.assertEqual(
                    len(
                        benchmark._load_result_file(run / "duckdb/workflow/result.txt")
                    ),
                    11,
                )
            self.assertEqual(
                json.loads((run / "metadata.json").read_text())["scale_bench"][
                    "temporary_cleanup"
                ],
                "removed",
            )

    def test_failure_after_copy_cleans_only_owned_directory(self):
        run, workload = self.prepared(["workflow"])
        with self.assertRaisesRegex(duckdb.Error, "missing_column"):
            with workload, patch.dict(
                SQL_QUERIES, workflow="SELECT missing_column FROM table_c_prep"
            ):
                workload.prepare(run)
        self.assertFalse(workload.root.exists())
        self.assertTrue(self.source.exists())
        self.assertEqual(list((self.root / "scratch").iterdir()), [])
        self.assertEqual(
            json.loads((run / "metadata.json").read_text())["scale_bench"][
                "temporary_cleanup"
            ],
            "removed",
        )

    def test_sigterm_removes_prepared_data(self):
        run, _ = self.prepared(["workflow"])
        script = """import json,sys,time
from scale_bench import ScaleBench
with ScaleBench(sys.argv[1], ['workflow'], sys.argv[3]) as workload:
    workload.prepare(sys.argv[2])
    print(workload.root, flush=True)
    while True: time.sleep(1)
"""
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(self.source),
                str(run),
                str(self.root / "scratch"),
            ],
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # The child prints only after preparation succeeds; communicate has
            # a bounded wait after termination, with a kill fallback on failure.
            import selectors

            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                self.assertTrue(
                    selector.select(timeout=20), "Preparation did not finish"
                )
                line = proc.stdout.readline().strip()
            self.assertTrue(line)
            prepared = Path(line)
            self.assertTrue(prepared.exists())
            proc.send_signal(signal.SIGTERM)
            _, error = proc.communicate(timeout=20)
            self.assertEqual(proc.returncode, 143, error)
            self.assertFalse(prepared.exists())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()

    def test_cleanup_failure_records_residual_path_and_fails(self):
        run, workload = self.prepared(["workflow"])
        with self.assertWarnsRegex(UserWarning, "Temporary data remains"):
            with self.assertRaisesRegex(RuntimeError, "could not be removed"):
                with patch("scale_bench.shutil.rmtree", side_effect=OSError("busy")):
                    with workload:
                        workload.prepare(run)
        self.assertTrue(workload.root.exists())
        doc = json.loads((run / "metadata.json").read_text())["scale_bench"]
        self.assertEqual(doc["temporary_cleanup"], "failed")
        self.assertEqual(doc["temporary_directory"], str(workload.root))

    def test_signals_interrupt_active_statements_and_clean_owned_files(self):
        script = """import sys
import performance_test as benchmark
import scale_bench
benchmark.log = lambda message: None

def execute_long_query(con):
    con.execute('SET threads=1')
    def ready():
        print(workload.root, flush=True)
        return 0.0
    con.create_function('signal_ready', ready, [], 'DOUBLE', side_effects=True)
    con.execute('SELECT sum(sin(i) + (SELECT signal_ready())) FROM range(10000000000) t(i)')

with scale_bench.ScaleBench(sys.argv[1], ['workflow'], sys.argv[3]) as workload:
    if sys.argv[4] == 'prepare':
        original = scale_bench.fingerprint_sql
        def fingerprint(con, sql):
            execute_long_query(con)
            return original(con, sql)
        scale_bench.fingerprint_sql = fingerprint
        workload.prepare(sys.argv[2])
    else:
        workload.prepare(sys.argv[2])
        con = benchmark.open_connection(sys.argv[1], workload=workload)
        try:
            execute_long_query(con)
        finally:
            workload.close_connection(con)
"""
        import selectors

        for phase in ("prepare", "run"):
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                with self.subTest(phase=phase, signal=sig):
                    run, _ = self.prepared(["workflow"])
                    scratch = self.root / "scratch"
                    scratch.mkdir(exist_ok=True)
                    sentinel = scratch / "keep-me"
                    sentinel.write_text("unrelated data")
                    proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            script,
                            str(self.source),
                            str(run),
                            str(scratch),
                            phase,
                        ],
                        cwd=HERE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    try:
                        # The UDF announces readiness from inside the active
                        # statement, after the prepared C copy exists.
                        with selectors.DefaultSelector() as selector:
                            selector.register(proc.stdout, selectors.EVENT_READ)
                            self.assertTrue(selector.select(timeout=20))
                            prepared = Path(proc.stdout.readline().strip())
                        self.assertTrue((prepared / "table_c_prep").exists())
                        time.sleep(0.2)  # Let the UDF return into native SQL work.
                        started = time.monotonic()
                        proc.send_signal(sig)
                        _, error = proc.communicate(timeout=5)
                        self.assertLess(time.monotonic() - started, 5)
                        self.assertEqual(proc.returncode, 128 + sig, error)
                        self.assertFalse(prepared.exists())
                        self.assertEqual(list(scratch.iterdir()), [sentinel])
                        metadata = json.loads((run / "metadata.json").read_text())
                        self.assertEqual(
                            metadata["scale_bench"]["temporary_cleanup"], "removed"
                        )
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                        proc.communicate()

    def test_signal_handlers_and_active_connection_are_restored(self):
        signals = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
        handlers = {sig: signal.getsignal(sig) for sig in signals}
        run, workload = self.prepared(["workflow"])
        with workload:
            workload.prepare(run)
            self.assertIsNone(workload.connection)
            with workload.connection_scope(duckdb.connect()) as con:
                self.assertIs(workload.connection, con)
            self.assertIsNone(workload.connection)
            with patch.object(
                benchmark,
                "_configure_connection",
                side_effect=ValueError("setup failed"),
            ):
                with self.assertRaisesRegex(ValueError, "setup failed"):
                    benchmark.open_connection(str(self.source), workload=workload)
            self.assertIsNone(workload.connection)
        self.assertEqual({sig: signal.getsignal(sig) for sig in signals}, handlers)

    def test_bfs_reports_traversal_depth_and_warns_for_root_only(self):
        for edges, depth, reached in (
            ([(5, 6)], 0, 0),
            ([(5, 1000)], 0, 0),
            ([(1000, 1000)], 0, 0),
            ([(1000, 1), (1, 2)], 2, 2),
        ):
            with self.subTest(edges=edges):
                run, workload = self.prepared(["bfs"])
                with workload:
                    workload.prepare(run)
                    with workload.connection_scope(
                        benchmark.open_connection(str(self.source), workload=workload)
                    ) as con:
                        con.execute("DROP VIEW table_a")
                        con.execute("CREATE TABLE table_a (col_b BIGINT, col_d BIGINT)")
                        con.executemany("INSERT INTO table_a VALUES (?, ?)", edges)
                        workload.run_graph(con, "bfs", False)
                        if reached == 0:
                            with self.assertWarnsRegex(
                                UserWarning, "does not measure traversal"
                            ):
                                result = workload.check_graph(con, "bfs", False)
                            self.assertIn("warning", result)
                        else:
                            result = workload.check_graph(con, "bfs", False)
                            self.assertNotIn("warning", result)
                        self.assertTrue(result["passed"])
                        self.assertEqual(result["max_distance"], depth)
                        self.assertEqual(result["reached_beyond_root"], reached)
                        self.assertEqual(result["expanded"], reached > 0)

    def test_null_graph_endpoints_fail_preparation_but_sql_queries_run(self):
        path = self.source / TABLE_DIRECTORIES["table_a"] / "part.parquet"
        for column in ("col_b", "col_d"):
            write_source = str(path).replace("'", "''")
            with duckdb.connect() as con:
                con.execute(f"CREATE TABLE a AS FROM read_parquet('{write_source}')")
                con.execute("UPDATE a SET col_b=1000, col_d=1001")
                con.execute(f"UPDATE a SET {column}=NULL")
                con.execute(f"COPY a TO '{write_source}' (FORMAT PARQUET)")
            for query in ("bfs", "pagerank"):
                with self.subTest(column=column, query=query):
                    run, workload = self.prepared([query, "workflow"])
                    with workload, self.assertRaisesRegex(
                        ValueError, "non-NULL.*col_b and col_d"
                    ):
                        workload.prepare(run)
                    self.assertIsNone(workload.root)
                    self.assertIsNone(workload.connection)
            output = self.root / column
            run_cli(self.source, output, "--queries", "gb2")

    def test_root_only_bfs_warning_is_retained_in_iteration_metadata(self):
        path = str(self.source / TABLE_DIRECTORIES["table_a"] / "part.parquet").replace(
            "'", "''"
        )
        with duckdb.connect() as con:
            con.execute(f"CREATE TABLE a AS FROM read_parquet('{path}')")
            con.execute("UPDATE a SET col_b=5, col_d=6")
            con.execute(f"COPY a TO '{path}' (FORMAT PARQUET)")
        output = self.root / "root-only"
        result = run_cli(self.source, output, "--queries", "bfs")
        self.assertIn("does not measure traversal performance", result.stderr)
        run = next(output.iterdir())
        metadata = json.loads((run / "duckdb/bfs/metadata_iter0.json").read_text())
        checks = metadata["invariants"]
        self.assertTrue(checks["passed"])
        self.assertFalse(checks["expanded"])
        self.assertEqual(checks["max_distance"], 0)
        self.assertEqual(checks["reached_beyond_root"], 0)
        self.assertIn("only root 1000", checks["warning"])

    def test_named_queries_split_logs_and_preserve_tpch_names(self):
        log_dir = self.root / "logs"
        log_dir.mkdir()
        sql = {"gb1": SQL_QUERIES["gb1"], "gb2": SQL_QUERIES["gb2"]}
        (log_dir / "sirius.log").write_text(
            "QueryBegin: query=1 SQL: " + sql["gb1"] + "\nfirst\n"
            "QueryBegin: query=2 SQL: " + sql["gb2"] + "\nsecond\n"
        )
        run = self.root / "logs-output"
        benchmark.split_sirius_log(str(log_dir), str(run), ["gb1", "gb2"], 1, sql)
        self.assertIn("first", (run / "sirius/gb1/sirius.log").read_text())
        self.assertNotIn("second", (run / "sirius/gb1/sirius.log").read_text())
        self.assertEqual(benchmark.query_name(6), "q6")

    def test_sequential_mode_runs_named_queries(self):
        output = self.root / "output"
        run_cli(
            self.source,
            output,
            "--queries",
            "gb1,workflow",
            "--mode",
            "sequential",
            "--iterations",
            "2",
        )
        run = next(output.iterdir())
        with (run / "csv/runtimes.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(
            [(r["query"], r["iteration"]) for r in rows if r["query"] != "TOTAL"],
            [("gb1", "0"), ("workflow", "0"), ("gb1", "1"), ("workflow", "1")],
        )

    def test_only_required_tables_and_no_temporary_copy_for_gb2(self):
        for name in ("table_b", "table_c", "table_d", "table_e"):
            shutil.rmtree(self.source / TABLE_DIRECTORIES[name])
        output = self.root / "output"
        run_cli(self.source, output, "--queries", "gb2")
        doc = json.loads((next(output.iterdir()) / "metadata.json").read_text())[
            "scale_bench"
        ]
        self.assertEqual(doc["temporary_cleanup"], "not_needed")
        self.assertNotIn("prepared_table_c", doc)

    def test_unsupported_options_and_missing_schema_fail_before_scratch(self):
        for options in (
            ("--queries", "shuffle"),
            ("--queries", "gb2,gb2"),
            ("--pin", "gpu"),
            ("--queries", "bfs", "--result-format", "parquet"),
            ("--concurrency", "2"),
            ("--precmd", "nsys"),
            ("--scale-factor", "1"),
            ("--iterations", "0"),
        ):
            with self.subTest(options=options):
                proc = run_cli(self.source, self.root / "output", *options, check=False)
                self.assertNotEqual(proc.returncode, 0)
                self.assertFalse((self.root / "output").exists())
        shutil.rmtree(self.source / TABLE_DIRECTORIES["table_c"])
        run, workload = self.prepared(["workflow"])
        with workload, self.assertRaisesRegex(FileNotFoundError, "table_c"):
            workload.prepare(run)
        self.assertIsNone(workload.root)

    def test_pagerank_rejects_nonfinite_result(self):
        run, workload = self.prepared(["pagerank"])
        with workload:
            workload.prepare(run)
            with benchmark.open_connection(str(self.source), workload=workload) as con:
                workload.run_graph(con, "pagerank", False)
                self.assertTrue(workload.check_graph(con, "pagerank", False)["passed"])
                ranks = dict(con.execute("SELECT id, rank FROM pr_rank").fetchall())
                # Closed-form ten-step personalized rank on the three-node
                # cycle. The duplicate edge must not change its transition
                # probability, and disconnected vertices retain zero mass.
                expected = {
                    1000: 0.15 * sum(0.85**i for i in (0, 3, 6, 9)),
                    1001: 0.15 * sum(0.85**i for i in (1, 4, 7)) + 0.85**10,
                    1002: 0.15 * sum(0.85**i for i in (2, 5, 8)),
                    2000: 0,
                    2001: 0,
                }
                self.assertEqual(ranks.keys(), expected.keys())
                for vertex, rank in expected.items():
                    self.assertAlmostEqual(ranks[vertex], rank, places=12)
                con.execute("UPDATE pr_rank SET rank='NaN'::DOUBLE")
                with self.assertRaisesRegex(ValueError, "invariants failed"):
                    workload.check_graph(con, "pagerank", False)

    def test_graph_cpu_profile_is_not_overwritten_by_untimed_checks(self):
        output = self.root / "profile"
        run_cli(self.source, output, "--queries", "bfs", "--duckdb-profiling")
        run = next(output.iterdir())
        profile = json.loads((run / "duckdb/bfs/profile_iter0.json").read_text())
        # DuckDB v2 exports an operator tree without a query_name field. The
        # final fingerprint sums the columns; the untimed invariant check
        # instead uses count(DISTINCT vertex) and min(distance).
        text = json.dumps(profile).lower()
        self.assertIn("bfs_visited", text)
        self.assertIn("sum", text)
        self.assertNotIn("distinct", text)
        self.assertNotIn("min(", text)


if __name__ == "__main__":
    unittest.main()
