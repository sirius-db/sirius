# Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
"""Run with pixi run python -m unittest discover -s test/tpch_performance -p test_concurrent_benchmark.py."""
import argparse
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import duckdb
import concurrent_benchmark as cb
import performance_test as perf
from tpch_pin_columns import emit_pin_all, union_columns_by_table


def arguments(**overrides):
    defaults = dict(
        concurrency=2,
        concurrency_mode="identical",
        stream_order=None,
        mode=None,
        profile=None,
        precmd="none",
        nsys_profile=False,
        iterations=2,
        pin_after_iteration=0,
        pin="none",
        pin_compression=False,
        data_source="duckdb",
        run_timeout=30,
        query_timeout=3,
        validation_memory_limit_mb=1,
        queries="1,6",
        scale_factor="1",
        duckdb_profiling=False,
    )
    return argparse.Namespace(**(defaults | overrides))


class ConcurrentBenchmarkTests(unittest.TestCase):
    def test_invalid_combinations(self):
        for values in (
            dict(concurrency=0),
            dict(mode="grouped"),
            dict(profile="cold"),
            dict(precmd="nsys"),
            dict(nsys_profile=True),
            dict(iterations=0),
            dict(pin_after_iteration=1),
            dict(queries="1,1"),
            dict(queries="23"),
            dict(run_timeout=float("nan")),
            dict(query_timeout=-1),
            dict(concurrency=1, stream_order="same"),
            dict(concurrency_mode="cooperative", stream_order="permuted"),
            dict(concurrency_mode="cooperative", concurrency=1, profile="cold"),
            dict(pin="parquet"),
            dict(pin_compression=True),
            dict(validation_memory_limit_mb=0),
        ):
            with self.subTest(values=values), self.assertRaises(SystemExit):
                cb.check_arguments(arguments(**values))
        cb.check_arguments(arguments(concurrency=1, mode="isolated", profile="cold"))
        cb.check_arguments(arguments(pin="host", profile="hot"))
        cb.check_arguments(arguments(concurrency_mode="cooperative", concurrency=1))
        cb.check_arguments(
            arguments(concurrency_mode="cooperative", stream_order="same")
        )

    def test_cooperative_next_request_goes_to_first_available_stream(self):
        initial_queries = threading.Barrier(3, timeout=3)
        fourth_claimed = threading.Event()

        class Connection:
            def execute(self, sql):
                self.query = int(sql)
                self.fetched = False
                if self.query <= 3:
                    initial_queries.wait()
                    if self.query != 2 and not fourth_claimed.wait(3):
                        raise RuntimeError("stream 2 did not claim query 4")
                elif self.query == 4:
                    fourth_claimed.set()

            def fetchmany(self, _):
                if self.fetched:
                    return []
                self.fetched = True
                return [(self.query,)]

            def close(self):
                pass

        output = io.StringIO()
        with patch.object(
            perf,
            "queries_for_scale_factor",
            return_value={f"q{q}": str(q) for q in range(1, 11)},
        ):
            records, captures, errors = cb.run_phase(
                SimpleNamespace(cursor=Connection),
                arguments(
                    concurrency=3,
                    concurrency_mode="cooperative",
                    iterations=1,
                    query_timeout=0,
                ),
                "duckdb",
                Path("."),
                list(range(1, 11)),
                False,
                cb.CaptureBudget(0),
                csv.DictWriter(output, fieldnames=cb.FIELDS),
                output,
            )
        self.assertFalse(errors)
        self.assertFalse(captures)
        self.assertTrue(all(r["status"] == "success" for r in records))
        self.assertEqual(len(records), 10)
        by_query = {r["query"]: r for r in records}
        self.assertEqual(set(by_query), {f"q{q}" for q in range(1, 11)})
        for q in range(1, 4):
            self.assertEqual(by_query[f"q{q}"]["stream_id"], q)
        self.assertEqual(by_query["q4"]["stream_id"], 2)
        self.assertLessEqual(by_query["q2"]["end_s"], by_query["q4"]["start_s"])

    def test_cooperative_iterations_capture_each_query_once(self):
        for concurrency in (1, 3, 8):
            with self.subTest(concurrency=concurrency):
                root = duckdb.connect()
                output = io.StringIO()
                try:
                    with patch.object(
                        perf,
                        "queries_for_scale_factor",
                        return_value={"q1": "select 42", "q6": "select 99"},
                    ):
                        records, captures, errors = cb.run_phase(
                            root,
                            arguments(
                                concurrency=concurrency, concurrency_mode="cooperative"
                            ),
                            "duckdb",
                            Path("."),
                            [6, 1],
                            True,
                            cb.CaptureBudget(1 << 20),
                            csv.DictWriter(output, fieldnames=cb.FIELDS),
                            output,
                        )
                finally:
                    root.close()
                self.assertFalse(errors)
                self.assertTrue(cb.summarize(records, 4, errors)["complete"])
                self.assertEqual(len(captures), 4)
                self.assertCountEqual(
                    [(r["iteration"], r["query"]) for r in records],
                    [(0, "q6"), (0, "q1"), (1, "q6"), (1, "q1")],
                )
                verdicts = cb.validate_captures(
                    captures, {1: [(42,)], 6: [(99,)]}, [6, 1]
                )
                self.assertTrue(all(v["status"] == "success" for v in verdicts))
                if concurrency == 1:
                    self.assertEqual(
                        [r["query"] for r in records], ["q6", "q1", "q6", "q1"]
                    )

    def test_cooperative_supervisor_metadata_and_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            database = directory / "input.duckdb"
            duckdb.connect(str(database)).close()
            (directory / "csv").mkdir()
            (directory / "metadata.json").write_text("{}")
            args = arguments(
                concurrency=3,
                concurrency_mode="cooperative",
                input=str(database),
                engine="cpu",
            )

            def child(command, **kwargs):
                return SimpleNamespace(
                    returncode=cb.run(json.loads(Path(command[-1]).read_text()))
                )

            with (
                patch.object(cb, "configured_limit", return_value=(1, None)),
                patch.object(cb.subprocess, "run", side_effect=child),
                patch.object(perf, "drop_os_cache"),
                patch.object(
                    perf,
                    "queries_for_scale_factor",
                    return_value={"q1": "select 42", "q6": "select 99"},
                ),
            ):
                self.assertEqual(
                    cb.supervise(args, directory, None, True, None, None), 0
                )
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["concurrency_mode"], "cooperative")
            self.assertEqual(metadata["executions_per_engine"], 4)
            self.assertEqual(metadata["shared_query_order"], [1, 6])
            self.assertIsNone(metadata["stream_queries"])
            summary = json.loads((directory / "concurrent_summary.json").read_text())
            self.assertEqual(summary["status"], "success")
            self.assertEqual(summary["engines"]["duckdb"]["expected"], 4)
            self.assertEqual(summary["engines"]["duckdb"]["completed"], 4)
            self.assertEqual(summary["validation"], {"checked": 4, "mismatches": 0})

    def test_stream_orders_and_selected_union(self):
        self.assertEqual(cb.orders_for([6, 1], 2, "same"), [[6, 1], [6, 1]])
        orders = cb.orders_for(list(range(1, 23)), 4, "permuted")
        self.assertEqual(len({tuple(x) for x in orders}), 4)
        for order in orders:
            self.assertEqual(sorted(order), list(range(1, 23)))
        self.assertEqual(set(union_columns_by_table([6])), {"lineitem"})
        self.assertEqual(
            set(union_columns_by_table([6])["lineitem"]),
            {"l_extendedprice", "l_discount", "l_shipdate", "l_quantity"},
        )

    def test_selected_union_with_remote_pin_globs(self):
        sql = emit_pin_all(
            "s3://bucket/tpch",
            "parquet",
            {"lineitem": "s3://bucket/tpch/lineitem/*.parquet"},
            queries=[6],
        )
        self.assertEqual(sql.count("CALL pin_table("), 1)
        self.assertIn("s3://bucket/tpch/lineitem/*.parquet", sql)
        self.assertNotIn("l_orderkey", sql)

    def test_shared_memory_database_and_every_execution_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = duckdb.connect(":memory:")
            root.execute("create table sample as select range as i from range(100)")
            out = io.StringIO()
            budget = cb.CaptureBudget(1 << 20)
            s3_input = Mock()
            with patch.object(
                perf,
                "queries_for_scale_factor",
                return_value={
                    "q1": "select sum(i) from sample",
                    "q6": "select count(*) from sample",
                },
            ):
                records, captures, errors = cb.run_phase(
                    root,
                    arguments(),
                    "duckdb",
                    Path(tmp),
                    [1, 6],
                    True,
                    budget,
                    csv.DictWriter(out, fieldnames=cb.FIELDS),
                    out,
                    s3_input=s3_input,
                )
            self.assertEqual(s3_input.refresh_secret.call_count, 8)
            self.assertTrue(
                all(not call.args[1] for call in s3_input.refresh_secret.call_args_list)
            )
            root.close()
            self.assertFalse(errors)
            self.assertEqual(len(records), 8)
            self.assertEqual(len(captures), 8)
            self.assertTrue(all(r["status"] == "success" for r in records))
            verdicts = cb.validate_captures(
                captures, {1: [(4950,)], 6: [(100,)]}, [1, 6]
            )
            self.assertTrue(all(r["status"] == "success" for r in verdicts))
            captures[("duckdb", 1, 1, 0)] = [(123,)]
            verdicts = cb.validate_captures(
                captures, {1: [(4950,)], 6: [(100,)]}, [1, 6]
            )
            self.assertEqual(sum(r["status"] == "mismatch" for r in verdicts), 1)
            self.assertGreater(budget.used, 0)
            self.assertTrue(cb.summarize(records, 8, errors)["complete"])

    def test_overlapping_workers_without_retaining_results(self):
        in_query = threading.Barrier(2, timeout=3)
        connections = []

        class Connection:
            fetched = False

            def execute(self, sql):
                self.fetched = False
                in_query.wait()
                return self

            def fetchmany(self, _):
                if self.fetched:
                    return []
                self.fetched = True
                return [(42,)]

            def close(self):
                pass

        class Root:
            def cursor(self):
                con = Connection()
                connections.append(con)
                return con

        budget = cb.CaptureBudget(0)  # Any accidental capture would fail.
        output = io.StringIO()
        records, captures, errors = cb.run_phase(
            Root(),
            arguments(query_timeout=0),
            "duckdb",
            Path("."),
            [1],
            False,
            budget,
            csv.DictWriter(output, fieldnames=cb.FIELDS),
            output,
        )
        self.assertFalse(errors)
        self.assertEqual(len(connections), 2)
        self.assertEqual(len(records), 4)
        self.assertFalse(captures)
        self.assertEqual(budget.used, 0)
        self.assertTrue(all(r["status"] == "success" for r in records))

    def test_concurrent_q11_uses_requested_scale_factor(self):
        root = duckdb.connect(":memory:")
        try:
            root.execute(
                "create table nation as select 1 n_nationkey, 'GERMANY' n_name"
            )
            root.execute("create table supplier as select 1 s_suppkey, 1 s_nationkey")
            root.execute(
                "create table partsupp as select * from (values "
                "(1, 100000, 1, 1), (2, 5, 1, 1)) "
                "t(ps_partkey, ps_supplycost, ps_availqty, ps_suppkey)"
            )
            for scale_factor, expected in (
                ("1", [(1, 100000)]),
                ("10", [(1, 100000), (2, 5)]),
            ):
                with self.subTest(scale_factor=scale_factor):
                    output = io.StringIO()
                    records, captures, errors = cb.run_phase(
                        root,
                        arguments(scale_factor=scale_factor, iterations=1),
                        "duckdb",
                        Path("."),
                        [11],
                        True,
                        cb.CaptureBudget(1 << 20),
                        csv.DictWriter(output, fieldnames=cb.FIELDS),
                        output,
                    )
                    self.assertFalse(errors)
                    self.assertEqual(len(records), 2)
                    self.assertEqual(len(captures), 2)
                    self.assertTrue(all(r["status"] == "success" for r in records))
                    for rows in captures.values():
                        self.assertEqual(rows, expected)
        finally:
            root.close()

    def test_setup_failure_breaks_barrier(self):
        class Root:
            def cursor(self):
                raise RuntimeError("injected setup failure")

        output = io.StringIO()
        records, _, errors = cb.run_phase(
            Root(),
            arguments(),
            "duckdb",
            Path("."),
            [1],
            False,
            cb.CaptureBudget(0),
            csv.DictWriter(output, fieldnames=cb.FIELDS),
            output,
        )
        self.assertFalse(records)
        self.assertTrue(errors)

    def test_query_failure_does_not_end_peer_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = duckdb.connect()
            output = io.StringIO()
            with patch.object(
                perf,
                "queries_for_scale_factor",
                return_value={"q1": "select * from missing_table", "q6": "select 42"},
            ):
                records, _, errors = cb.run_phase(
                    root,
                    arguments(),
                    "duckdb",
                    Path(tmp),
                    [1, 6],
                    False,
                    cb.CaptureBudget(0),
                    csv.DictWriter(output, fieldnames=cb.FIELDS),
                    output,
                )
            root.close()
            self.assertFalse(errors)
            summary = cb.summarize(records, 8, errors)
            self.assertEqual(summary["completed"], 4)
            self.assertEqual(summary["failed"], 4)
            self.assertFalse(summary["complete"])

    def test_timeout_does_not_interrupt_next_request(self):
        class Connection:
            interrupted = threading.Event()

            def execute(self, sql):
                self.interrupted.wait(2)
                raise RuntimeError("interrupted")

            def interrupt(self):
                self.interrupted.set()

        con = Connection()
        result = cb.execute_request(con, "select", 0.01, False, cb.CaptureBudget(0))
        self.assertEqual(result[4], "timeout")
        con.interrupted.clear()
        self.assertFalse(con.interrupted.wait(0.03))

    def test_capture_limit_discards_failed_partial_result(self):
        root = duckdb.connect()
        budget = cb.CaptureBudget(1)
        result = cb.execute_request(root, "select * from range(10)", 0, True, budget)
        root.close()
        self.assertEqual(result[4], "error")
        self.assertIn("capture budget", result[5])
        self.assertIsNone(result[3])
        self.assertEqual(budget.used, 0)

    def test_tolerant_multiset_validation(self):
        self.assertTrue(
            cb.matching_rows(
                [(1, None, Decimal("3"), 1.0)], [(1, None, Decimal("3"), 1.0 + 1e-11)]
            )
        )
        self.assertFalse(cb.matching_rows([(1,), (1,)], [(1,), (2,)]))
        self.assertTrue(cb.matching_rows([(1,), (2,)], [(2,), (1,)]))
        # Greedy matching would consume the only candidate for the second row.
        self.assertTrue(
            cb.matching_rows([(0.0,), (1.5e-10,)], [(0.75e-10,), (-0.75e-10,)])
        )
        self.assertFalse(cb.matching_rows([(None,)], [(0,)]))

    def test_elapsed_is_not_sum_of_latencies(self):
        records = [
            dict(stream_id=i, query="q1", runtime_s=2, end_s=2.1, status="success")
            for i in (1, 2)
        ]
        summary = cb.summarize(records, 2, [])
        self.assertEqual(summary["elapsed_s"], 2.1)
        self.assertAlmostEqual(summary["successful_queries_per_s"], 2 / 2.1)

    def test_keyed_logs_do_not_mix_interleaved_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "log_dir").mkdir()
            (directory / "log_dir" / "sirius.log").write_text(
                "QueryBegin: instance=0xa connection=1 query=1 SQL: CALL sirius_set_query_label('readonly_s1_q1_i0')\n"
                "QueryBegin: instance=0xa connection=2 query=1 SQL: CALL sirius_set_query_label('readonly_s2_q1_i0')\n"
                "QueryBegin: instance=0xa connection=1 query=2 SQL: SELECT 1\n"
                "QueryBegin: instance=0xa connection=2 query=2 SQL: SELECT 1\n"
                "done instance=0xa connection=1 window=10 query=2\n"
                "done instance=0xa connection=2 window=11 query=2\n"
                "unattributed message\n"
            )
            cb.split_keyed_logs(directory)
            log = (
                directory / "sirius/execution_logs/readonly_s1_q1_i0.log"
            ).read_text()
            self.assertIn("window=10", log)
            self.assertNotIn("connection=2", log)
            self.assertNotIn("unattributed", log)

    def test_keyed_logs_bound_open_files_and_preserve_evicted_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "log_dir").mkdir()
            folder = directory / "sirius" / "execution_logs"
            folder.mkdir(parents=True)
            expected = {}
            lines = []
            for stream in range(1, 41):
                label = f"readonly_s{stream}_q1_i0"
                query = (
                    f"QueryBegin: instance=0xa connection={stream} query=2 "
                    "SQL: SELECT 1\n"
                )
                lines.extend(
                    [
                        f"QueryBegin: instance=0xa connection={stream} query=1 "
                        f"SQL: CALL sirius_set_query_label('{label}')\n",
                        query,
                    ]
                )
                expected[label] = query
                # A fresh split must replace stale files, then append on reopen.
                (folder / f"{label}.log").write_text("stale output\n")
            for stream in range(1, 41):
                label = f"readonly_s{stream}_q1_i0"
                line = f"done instance=0xa connection={stream} query=2\n"
                lines.append(line)
                expected[label] += line
            (directory / "log_dir" / "sirius.log").write_text("".join(lines))

            opened = []
            original_open = Path.open

            def bounded_open(path, *args, **kwargs):
                if path.parent == folder:
                    if sum(not output.closed for output in opened) >= 32:
                        raise OSError("too many open log destinations")
                    output = original_open(path, *args, **kwargs)
                    opened.append(output)
                    return output
                return original_open(path, *args, **kwargs)

            with patch.object(Path, "open", bounded_open):
                cb.split_keyed_logs(directory)

            self.assertTrue(all(output.closed for output in opened))
            self.assertGreater(len(opened), len(expected))
            for label, content in expected.items():
                self.assertEqual((folder / f"{label}.log").read_text(), content)
            index = json.loads((directory / "log_index.json").read_text())
            self.assertEqual(len(index["queries"]), len(expected))

    def test_reference_provenance_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "duckdb/q1").mkdir(parents=True)
            (directory / "duckdb/q1/result.txt").write_text("(42,)\n")
            (directory / "metadata.json").write_text(
                json.dumps({"input_identity": "old", "query_sha256": {"1": "a"}})
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                cb.check_references(
                    directory / "duckdb", {"input_identity": "new"}, [1]
                )


if __name__ == "__main__":
    unittest.main()
