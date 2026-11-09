# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Host-only tests for scripts/dynamic_filter_key_parity.py over fixture logs.

Run with:
    pixi run python -I -B -m unittest discover -s test/scripts -p 'test_dynamic_filter_key_parity.py'
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import stat
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "dynamic_filter_key_parity.py"
)
_spec = importlib.util.spec_from_file_location("dynamic_filter_key_parity", SCRIPT)
parity = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = parity
_spec.loader.exec_module(parity)

SIG = "6b85214e8d4b623a"


def key_line(
    kind="hash_in_list",
    size=90,
    zone_map=0,
    reason="-",
    sig=SIG,
    occ=0,
    condition=0,
    join=17,
):
    return (
        f"[2026-11-06 10:00:00.000] [debug] [dynamic_filter_key_outcome.cpp:1] "
        f"[dynamic_filter_key] v=1 sig={sig} occ={occ} join={join} condition={condition} "
        f"build_ordinal=0 rows=100 valid=90 kind={kind} size={size} zone_map={zone_map} "
        f"pushed=1 reason={reason}"
    )


class Fixture:
    """Writes run output directories: one subdirectory per query."""

    def __init__(self, root: pathlib.Path):
        self.root = root

    def query(
        self,
        run: str,
        query: str,
        lines: list[str],
        result_match=True,
        gpu_fallback=False,
    ):
        directory = self.root / run / query
        directory.mkdir(parents=True, exist_ok=True)
        unrelated = "[2026-11-06 10:00:00.000] [info] [x.cpp:1] unrelated message"
        (directory / "sirius_2026-11-06.log").write_text(
            "\n".join([unrelated, *lines]) + "\n"
        )
        if result_match is not None:
            recorded = {"result_match": result_match}
            if gpu_fallback is not None:
                recorded["gpu_fallback"] = gpu_fallback
            (directory / "result.json").write_text(json.dumps(recorded))
        return directory

    def diff(self) -> tuple[int, str]:
        out = io.StringIO()
        status = parity.diff(self.root / "base", self.root / "candidate", stream=out)
        return status, out.getvalue()


class ParseLine(unittest.TestCase):
    def test_parses_the_logged_form(self):
        line = parity.parse_line(key_line(kind="bloom", size=4096, zone_map=1))
        self.assertEqual(line.sig, SIG)
        self.assertEqual((line.occ, line.condition, line.join), (0, 0, "17"))
        self.assertEqual((line.kind, line.size, line.zone_map), ("bloom", 4096, True))
        self.assertEqual(line.strength, 1)

    def test_ignores_other_lines(self):
        self.assertIsNone(
            parity.parse_line("[info] [dynamic_filter_domain] path=ordinary")
        )

    def test_rejects_unknown_versions_and_shapes(self):
        with self.assertRaises(parity.ParityError):
            parity.parse_line(key_line().replace("v=1", "v=2"))
        with self.assertRaises(parity.ParityError):
            parity.parse_line(key_line().replace(" pushed=1", ""))
        with self.assertRaises(parity.ParityError):
            parity.parse_line(key_line(kind="zone"))

    def test_strength_ranks_exact_lists_equal(self):
        ranks = {
            kind: parity.parse_line(key_line(kind=kind)).strength
            for kind in parity.STRENGTH
        }
        self.assertEqual(
            ranks, {"none": 0, "bloom": 1, "small_in_list": 2, "hash_in_list": 2}
        )


class Diff(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fixture = Fixture(pathlib.Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def pair(self, base: list[str], candidate: list[str], query="q01", **kwargs):
        self.fixture.query("base", query, base)
        self.fixture.query("candidate", query, candidate, **kwargs)
        return self.fixture.diff()

    def test_identical_runs_pass(self):
        status, out = self.pair([key_line()], [key_line(join=23)])
        self.assertEqual(status, 0, out)
        self.assertIn("summary: equal=1 failures=0", out)

    def test_small_and_hash_in_lists_are_equally_strong(self):
        status, out = self.pair(
            [key_line(kind="hash_in_list")], [key_line(kind="small_in_list", size=5)]
        )
        self.assertEqual(status, 0, out)
        self.assertIn("info equal", out)  # The size change is reported for information.

    def test_stronger_keys_pass(self):
        status, out = self.pair(
            [key_line(kind="bloom")], [key_line(kind="hash_in_list")]
        )
        self.assertEqual(status, 0, out)
        self.assertIn("info stronger", out)

    def test_weaker_keys_fail_unless_a_declared_delta_explains_them(self):
        status, out = self.pair(
            [key_line(kind="bloom")], [key_line(kind="none", reason="domain_gate")]
        )
        self.assertEqual(status, 1, out)
        self.assertIn("FAIL weaker", out)

    def test_declared_deltas_explain_weaker_keys(self):
        for reason in sorted(parity.DECLARED_DELTAS):
            with self.subTest(reason=reason):
                status, out = self.pair(
                    [key_line(kind="bloom")],
                    [key_line(kind="none", reason=reason)],
                    query=f"q_{reason}",
                )
                self.assertEqual(status, 0, out)

    def test_missing_keys_are_plan_drift(self):
        status, out = self.pair([key_line(), key_line(condition=1)], [key_line()])
        self.assertEqual(status, 1, out)
        self.assertIn("FAIL missing", out)

    def test_extra_keys_are_upgrades(self):
        status, out = self.pair(
            [key_line()], [key_line(), key_line(sig="00000000000000ff")]
        )
        self.assertEqual(status, 0, out)
        self.assertIn("info extra", out)

    def test_occurrence_separates_joins_with_one_signature(self):
        base = [key_line(occ=0, kind="bloom"), key_line(occ=1, kind="hash_in_list")]
        swapped = [key_line(occ=0, kind="hash_in_list"), key_line(occ=1, kind="bloom")]
        status, out = self.pair(base, swapped)
        self.assertEqual(status, 1, out)
        self.assertIn("info stronger", out)
        self.assertIn("FAIL weaker", out)

    def test_lost_zone_maps_fail_unless_declared(self):
        status, out = self.pair([key_line(zone_map=1)], [key_line(zone_map=0)])
        self.assertEqual(status, 1, out)
        self.assertIn("zone_map=lost", out)
        _, out = self.pair(
            [key_line(zone_map=1)],
            [key_line(zone_map=0, reason="declined_pipeline_barrier")],
            query="q02",
        )
        self.assertIn("info equal zone_map=lost query=q02", out)
        self.assertIn("failures=1", out)  # Only q01 still fails.

    def test_gained_zone_maps_pass(self):
        status, out = self.pair([key_line(zone_map=0)], [key_line(zone_map=1)])
        self.assertEqual(status, 0, out)
        self.assertIn("zone_map=gained", out)

    def test_result_mismatches_fail(self):
        status, out = self.pair([key_line()], [key_line()], result_match=False)
        self.assertEqual(status, 1, out)
        self.assertIn("FAIL result mismatch: candidate query=q01", out)
        status, out = self.pair(
            [key_line()], [key_line()], query="q02", result_match=None
        )
        self.assertEqual(status, 1, out)
        self.assertIn("FAIL result unknown: candidate query=q02", out)

    def test_a_gpu_fallback_on_either_side_fails(self):
        self.fixture.query("base", "q01", [key_line()], gpu_fallback=True)
        self.fixture.query("candidate", "q01", [key_line()])
        status, out = self.fixture.diff()
        self.assertEqual(status, 1, out)
        self.assertIn("FAIL gpu fell back: baseline query=q01", out)

    def test_an_unrecorded_fallback_check_fails(self):
        status, out = self.pair([key_line()], [key_line()], gpu_fallback=None)
        self.assertEqual(status, 1, out)
        self.assertIn("FAIL gpu unknown: candidate query=q01", out)

    def test_a_key_logged_twice_fails(self):
        status, out = self.pair([key_line()], [key_line(), key_line()])
        self.assertEqual(status, 1, out)
        self.assertIn("FAIL duplicate", out)

    def test_main_reports_malformed_input(self):
        self.fixture.query("base", "q01", [key_line().replace("v=1", "v=9")])
        self.fixture.query("candidate", "q01", [key_line()])
        root = self.fixture.root
        self.assertEqual(
            parity.main(["diff", str(root / "base"), str(root / "candidate")]), 2
        )


class ResultComparison(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def compare(self, gpu: str, cpu: str) -> bool:
        (self.root / "gpu.csv").write_text(gpu)
        (self.root / "cpu.csv").write_text(cpu)
        return parity.results_match(self.root / "gpu.csv", self.root / "cpu.csv")

    def test_rows_may_arrive_in_any_order(self):
        self.assertTrue(self.compare("k,v\n1,a\n2,b\n", "k,v\n2,b\n1,a\n"))

    def test_floats_compare_with_a_relative_tolerance(self):
        self.assertTrue(self.compare("s\n1234.5678901234\n", "s\n1234.5678901235\n"))
        self.assertFalse(self.compare("s\n1234.5\n", "s\n1234.6\n"))

    def test_different_rows_or_headers_mismatch(self):
        self.assertFalse(self.compare("k\n1\n", "k\n1\n1\n"))
        self.assertFalse(self.compare("k\n1\n", "j\n1\n"))


FAKE_DUCKDB = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import os, pathlib, sys
    script = sys.stdin.read()
    outputs = [line.split('"')[1] for line in script.splitlines() if line.startswith(".once")]
    for path in outputs:
        pathlib.Path(path).write_text("k\\n1\\n")
    if os.environ.get("FAKE_FALLBACK"):
        print("=============================================")
        print("Error in Sirius GPU execution, fallback to DuckDB")
        print("=============================================")
    fallback = ("[warn] Transparent execution: GPU execution failed at runtime; falling back to "
                "DuckDB CPU within the same transaction. GPU error: validity row_offset (3) "
                "not byte-aligned\\n") if os.environ.get("FAKE_FALLBACK") else ""
    log = pathlib.Path(os.environ["SIRIUS_LOG_DIR"]) / "sirius_2026-11-06.log"
    log.write_text(fallback + "[dynamic_filter_key] v=1 sig=00000000000000aa occ=0 join=3 condition=0 "
                   "build_ordinal=0 rows=4 valid=4 kind=small_in_list size=4 zone_map=0 "
                   "pushed=1 reason=-\\n"
                   + "LEVEL=" + os.environ["SIRIUS_LOG_LEVEL"] + "\\n"
                   + "ARGS=" + " ".join(sys.argv[1:]) + "\\n"
                   + script)
    """
)


class ParquetTables(unittest.TestCase):
    def test_a_name_with_both_layouts_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "orders").mkdir()
            (root / "orders" / "a.parquet").write_bytes(b"")
            (root / "orders.parquet").write_bytes(b"")
            with self.assertRaises(parity.ParityError):
                parity.parquet_tables(root)

    def test_a_directory_without_tables_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "schema.sql").write_text("")
            with self.assertRaises(parity.ParityError):
                parity.parquet_tables(pathlib.Path(tmp))

    def test_quotes_are_escaped(self):
        statements = parity.view_statements({"o'k": [pathlib.Path("/d/it's.parquet")]})
        self.assertEqual(
            statements,
            [
                "CREATE VIEW \"o'k\" AS SELECT * FROM read_parquet(['/d/it''s.parquet']);"
            ],
        )


class Run(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        self.cli = self.root / "duckdb"
        self.cli.write_text(FAKE_DUCKDB)
        self.cli.chmod(self.cli.stat().st_mode | stat.S_IXUSR)
        queries = self.root / "queries"
        queries.mkdir()
        (queries / "q01.sql").write_text("SELECT 1 AS k;\n")

    def tearDown(self):
        self._tmp.cleanup()

    def run_queries(self, out: pathlib.Path, *extra: str) -> int:
        return parity.main(
            [
                "run",
                "--extension", "ext/sirius.duckdb_extension",
                "--db", "tpch.duckdb",
                "--queries", str(self.root / "queries"),
                "--out", str(out),
                "--duckdb", str(self.cli),
                *extra,
            ]
        )  # fmt: skip

    def test_runs_gpu_then_cpu_with_debug_logs_per_query(self):
        out = self.root / "out"
        self.assertEqual(
            self.run_queries(
                out, "--set", "max_dynamic_filter_bloom_bytes_per_gpu=1024"
            ),
            0,
        )
        log = (out / "q01" / "sirius_2026-11-06.log").read_text()
        self.assertIn("LEVEL=debug", log)
        self.assertIn("ARGS=-no-agent -unsigned -bail -readonly tpch.duckdb", log)
        self.assertIn("SET enable_dynamic_zone_map_filter = true;", log)
        self.assertIn("SET max_dynamic_filter_bloom_bytes_per_gpu = 1024;", log)
        gpu_at = log.index("gpu.csv")
        self.assertLess(gpu_at, log.index("SET gpu_execution = false;"))
        self.assertLess(log.index("SET gpu_execution = false;"), log.index("cpu.csv"))
        self.assertEqual(log.count("SELECT 1 AS k;"), 2)
        result = json.loads((out / "q01" / "result.json").read_text())
        self.assertTrue(result["result_match"])
        self.assertFalse(result["gpu_fallback"])
        self.assertIsNone(result["gpu_error"])
        # The run's own output diffs clean against itself.
        self.assertEqual(parity.diff(out, out, stream=io.StringIO()), 0)

    def test_a_gpu_fallback_fails_the_run_and_names_the_error(self):
        out = self.root / "out"
        with mock.patch.dict(os.environ, {"FAKE_FALLBACK": "1"}):
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                self.assertEqual(self.run_queries(out), 1)
        self.assertIn("FALLBACK(ok) q01", printed.getvalue())
        self.assertIn(
            "GPU error: validity row_offset (3) not byte-aligned", printed.getvalue()
        )
        result = json.loads((out / "q01" / "result.json").read_text())
        self.assertTrue(result["result_match"])
        self.assertTrue(result["gpu_fallback"])
        self.assertEqual(
            result["gpu_error"], "validity row_offset (3) not byte-aligned"
        )
        self.assertEqual(parity.diff(out, out, stream=io.StringIO()), 1)

    def test_parquet_input_creates_views_without_gpu_execution(self):
        data = self.root / "data"
        (data / "lineitem").mkdir(parents=True)
        (data / "lineitem" / "part.1.parquet").write_bytes(b"")
        (data / "lineitem" / "part.0.parquet").write_bytes(b"")
        (data / "date_dim.parquet").write_bytes(b"")
        (data / "metadata.json").write_text("{}")
        (data / "schema.sql").write_text("")
        out = self.root / "out"
        status = parity.main(
            [
                "run",
                "--extension", "ext/sirius.duckdb_extension",
                "--parquet", str(data),
                "--queries", str(self.root / "queries"),
                "--out", str(out),
                "--duckdb", str(self.cli),
            ]
        )  # fmt: skip
        self.assertEqual(status, 0)
        log = (out / "q01" / "sirius_2026-11-06.log").read_text()
        self.assertIn("ARGS=-no-agent -unsigned -bail\n", log)
        lineitem = data.resolve() / "lineitem"
        self.assertIn(
            f"CREATE VIEW \"lineitem\" AS SELECT * FROM read_parquet(['{lineitem / 'part.0.parquet'}', "
            f"'{lineitem / 'part.1.parquet'}']);",
            log,
        )
        self.assertIn(
            f"CREATE VIEW \"date_dim\" AS SELECT * FROM read_parquet(['{data.resolve() / 'date_dim.parquet'}']);",
            log,
        )
        self.assertNotIn("metadata", log)
        disabled = log.index("SET gpu_execution = false;")
        self.assertLess(disabled, log.index("CREATE VIEW"))
        self.assertLess(log.rindex("CREATE VIEW"), log.index("RESET gpu_execution;"))
        self.assertLess(log.index("RESET gpu_execution;"), log.index("gpu.csv"))

    def test_exactly_one_data_source_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.run_queries(self.root / "out", "--parquet", str(self.root))
            with self.assertRaises(SystemExit):
                parity.main(
                    [
                        "run",
                        "--extension", "x",
                        "--queries", str(self.root / "queries"),
                        "--out", str(self.root / "out"),
                    ]
                )  # fmt: skip

    def test_refuses_to_reuse_an_output_directory(self):
        out = self.root / "out"
        self.assertEqual(self.run_queries(out), 0)
        self.assertEqual(self.run_queries(out), 2)


if __name__ == "__main__":
    unittest.main()
