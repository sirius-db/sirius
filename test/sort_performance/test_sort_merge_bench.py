# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Tests of sort_merge_bench.py. No GPU and no Sirius extension needed.

Run: pixi run python -B test/sort_performance/test_sort_merge_bench.py   (also collectable by pytest)
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import bench_metrics  # noqa: E402
import sort_merge_bench as smb  # noqa: E402

FIXTURES = SCRIPT_DIR / "fixtures"
GIB = 1 << 30


def fixture_log():
    return "".join(
        (FIXTURES / n).read_text()
        for n in (
            "merge_sort_stats.log",
            "downgrade.log",
            "oom_retries.log",
            "discard_sink.log",
        )
    )


class Scratch(unittest.TestCase):
    """A temp dir per test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def make_log_dir(self, text, name="sirius.log"):
        log_dir = self.tmp / "log"
        log_dir.mkdir(exist_ok=True)
        (log_dir / name).write_text(text)
        return str(log_dir)


# -----------------------------------------------------------------------------
# Datasets, sizing, SQL
# -----------------------------------------------------------------------------


class DatasetTest(unittest.TestCase):
    def test_five_datasets_in_order(self):
        self.assertEqual(
            list(smb.DATASETS),
            [
                "uniform-1key",
                "zipf-1key",
                "normal-multikey",
                "few_distinct-strings",
                "sequential",
            ],
        )

    def test_every_key_is_a_column(self):
        for name, ds in smb.DATASETS.items():
            cols = [c["name"] for c in ds["columns"]]
            self.assertTrue(ds["keys"], name)
            for key in ds["keys"]:
                self.assertIn(key, cols, name)
            self.assertTrue(ds["description"], name)

    def test_column_distributions(self):
        cols = {n: ds["columns"] for n, ds in smb.DATASETS.items()}
        self.assertEqual(
            [(c["type"], c["dist"]) for c in cols["uniform-1key"]],
            [("BIGINT", "uniform"), ("DOUBLE", "uniform")],
        )
        zipf = cols["zipf-1key"][0]
        self.assertEqual((zipf["type"], zipf["dist"]), ("BIGINT", "zipf"))
        self.assertAlmostEqual(zipf["p1"], 1.1)
        self.assertEqual(zipf["cardinality"], 10**9)
        self.assertEqual(
            [(c["type"], c["dist"]) for c in cols["normal-multikey"]],
            [("INTEGER", "normal"), ("BIGINT", "uniform"), ("VARCHAR", "uniform")],
        )
        s = cols["normal-multikey"][2]
        self.assertEqual((s["p1"], s["p2"]), (8, 16))
        fd = cols["few_distinct-strings"]
        self.assertEqual(
            [(c["type"], c["dist"]) for c in fd], [("VARCHAR", "few_distinct")]
        )
        self.assertEqual(cols["sequential"][0]["dist"], "sequential")
        self.assertEqual(smb.DATASETS["normal-multikey"]["keys"], ["k1", "k2", "s"])

    def test_row_width_from_column_types(self):
        self.assertEqual(smb.row_width("uniform-1key"), 16)  # BIGINT + DOUBLE
        self.assertEqual(smb.row_width("zipf-1key"), 16)
        self.assertEqual(smb.row_width("sequential"), 16)
        # INTEGER 4 + BIGINT 8 + VARCHAR (4 + mean of 8..16 = 12)
        self.assertEqual(smb.row_width("normal-multikey"), 28)
        self.assertEqual(smb.row_width("few_distinct-strings"), 4 + 18)

    def test_size_dataset_hits_the_target(self):
        target = 10 * 4 * GIB
        for name in smb.DATASETS:
            sizing = smb.size_dataset(name, target, 8_000_000)
            width = smb.row_width(name)
            self.assertEqual(sizing["row_width"], width)
            self.assertEqual(sizing["rows_per_batch"], 8_000_000)
            self.assertEqual(sizing["rows"], sizing["num_batches"] * 8_000_000)
            self.assertEqual(sizing["bytes"], sizing["rows"] * width)
            # Within half a batch of the target.
            self.assertLessEqual(abs(sizing["bytes"] - target), width * 8_000_000 / 2)

    def test_size_dataset_exact_and_minimum(self):
        sizing = smb.size_dataset("uniform-1key", 16 * 1000 * 7, 1000)
        self.assertEqual(sizing["num_batches"], 7)
        self.assertEqual(sizing["rows"], 7000)
        self.assertEqual(sizing["bytes"], 112000)
        tiny = smb.size_dataset("uniform-1key", 1, 8_000_000)
        self.assertEqual(tiny["num_batches"], 1)

    def test_generate_call(self):
        sizing = smb.size_dataset("normal-multikey", 28 * 100 * 3, 100)
        call = smb.generate_call("normal-multikey", sizing)
        self.assertTrue(call.startswith("sirius_generate("))
        self.assertIn("num_batches := 3", call)
        self.assertIn("rows_per_batch := 100", call)
        self.assertIn(f"seed := {smb.SEED}", call)
        self.assertIn(
            "{name: 'k1', type: 'INTEGER', dist: 'normal', p1: 0, p2: 1000000}", call
        )
        self.assertIn(
            "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 8, p2: 16}", call
        )
        zipf = smb.generate_call("zipf-1key", sizing)
        self.assertIn("dist: 'zipf', p1: 1.1, cardinality: 1000000000}", zipf)

    def test_column_sql_quotes_names(self):
        sql = smb._column_sql({"name": "o'k", "type": "BIGINT", "dist": "uniform"})
        self.assertEqual(sql, "{name: 'o''k', type: 'BIGINT', dist: 'uniform'}")

    def test_build_query(self):
        sizing = smb.size_dataset("normal-multikey", 28 * 100 * 3, 100)
        sql = smb.build_query("normal-multikey", sizing, validate=False)
        self.assertTrue(sql.startswith("COPY (SELECT * FROM sirius_generate("))
        self.assertIn(") ORDER BY k1, k2, s)", sql)
        self.assertTrue(sql.endswith("TO '/dev/null' (FORMAT sirius_discard)"))
        self.assertNotIn("check_order", sql)
        checked = smb.build_query("normal-multikey", sizing, validate=True)
        self.assertTrue(
            checked.endswith("TO '/dev/null' (FORMAT sirius_discard, check_order true)")
        )

    def test_session_settings(self):
        self.assertEqual(
            smb.session_settings("windowed", True),
            [
                "SET gpu_execution = true",
                "SET sort_merge_mode = 'windowed'",
                "SET spill_compression = true",
            ],
        )
        self.assertEqual(
            smb.session_settings("one_shot", False)[1:],
            ["SET sort_merge_mode = 'one_shot'", "SET spill_compression = false"],
        )


# -----------------------------------------------------------------------------
# Helpers and config
# -----------------------------------------------------------------------------


class CheckResultTest(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(
            smb.check_result({"rows": 10, "rows_counted": 10}, False), ("ok", "")
        )
        status, err = smb.check_result({"rows": 10, "rows_counted": 9}, False)
        self.assertEqual(status, "row_mismatch")
        self.assertIn("9", err)

    def test_validate(self):
        base = {"rows": 10, "rows_counted": 10}
        checked = {"order_violations": 0, "order_checked": 1, "ranges_checked": 2}
        self.assertEqual(smb.check_result(dict(base), True)[0], "error")
        self.assertEqual(
            smb.check_result({**base, **checked, "discard_rows": 8}, True)[0],
            "row_mismatch",
        )
        self.assertEqual(
            smb.check_result(
                {**base, **checked, "discard_rows": 10, "order_violations": 3}, True
            )[0],
            "order_violation",
        )
        self.assertEqual(
            smb.check_result({**base, **checked, "discard_rows": 10}, True),
            ("ok", ""),
        )
        # Without --validate the sink line is not required.
        self.assertEqual(
            smb.check_result({**base, "discard_rows": 3}, False), ("ok", "")
        )

    def test_validate_requires_that_the_order_was_checked(self):
        base = {"rows": 10, "rows_counted": 10, "discard_rows": 10}
        unchecked = [
            # A sink line without the keys: zero violations alone prove nothing.
            {"order_violations": 0},
            {"order_violations": 0, "order_checked": None, "ranges_checked": None},
            # The planner turned check_order off.
            {"order_violations": 0, "order_checked": 0, "ranges_checked": 0},
            # Checking was on but nothing was compared.
            {"order_violations": 0, "order_checked": 1, "ranges_checked": 0},
            {"order_violations": 0, "order_checked": 1},
            # Two sinks, one of them unchecked, whatever way they are combined.
            {"order_violations": 0, "order_checked": 2, "ranges_checked": 4},
        ]
        for sink in unchecked:
            with self.subTest(sink=sink):
                status, err = smb.check_result({**base, **sink}, True)
                self.assertEqual(status, "order_unchecked")
                self.assertIn("--validate", err)
        # An empty result has no ranges to compare.
        empty = {"rows": 0, "rows_counted": 0, "discard_rows": 0}
        self.assertEqual(
            smb.check_result(
                {
                    **empty,
                    "order_violations": 0,
                    "order_checked": 1,
                    "ranges_checked": 0,
                },
                True,
            ),
            ("ok", ""),
        )
        # Not required without --validate.
        self.assertEqual(
            smb.check_result({**base, "order_checked": 0}, False), ("ok", "")
        )
        self.assertNotIn("order_unchecked", smb.ONE_SHOT_EXPECTED)


class ScanRunLogTest(Scratch):
    def test_fixture_log(self):
        out, tail, error = smb.scan_run_log(
            self.make_log_dir(fixture_log()), bench_metrics, 5
        )
        self.assertIsNone(error)
        # The discard-sink fixture carries one more merge_sort stats line (op 7 again).
        stats = bench_metrics.parse_merge_sort_stats(fixture_log())
        self.assertEqual(out["merge_sort_operators"], len(stats))
        for key, value in bench_metrics.sum_merge_sort_stats(stats).items():
            self.assertEqual(out[key], value, key)
        totals = bench_metrics.parse_downgrade_totals(fixture_log())
        self.assertGreater(totals.requests, 0)
        self.assertEqual(out["spill_bytes_host"], totals.to_host_bytes)
        self.assertEqual(out["spill_bytes_disk"], totals.to_disk_bytes)
        self.assertEqual(out["spill_batches_host"], totals.to_host_batches)
        self.assertEqual(out["spill_batches_disk"], totals.to_disk_batches)
        self.assertEqual(out["downgrade_requests"], totals.requests)
        self.assertEqual(
            out["oom_retries"], bench_metrics.count_oom_retries(fixture_log())
        )
        self.assertEqual(out["discard_rows"], 1500300)
        self.assertEqual(out["discard_bytes"], 12002400)
        self.assertEqual(out["order_violations"], 0)
        self.assertEqual(tail, fixture_log().splitlines()[-5:])

    def test_log_files_are_read_oldest_first_and_half_lines_dropped(self):
        log_dir = self.tmp / "log"
        log_dir.mkdir()
        (log_dir / "sirius_b.log").write_text("second\nhalf a li")
        (log_dir / "sirius_a.log").write_text("first\n")
        (log_dir / "other.txt").write_text("ignored\n")
        os.utime(log_dir / "sirius_a.log", (1000, 1000))
        os.utime(log_dir / "sirius_b.log", (2000, 2000))
        out, tail, error = smb.scan_run_log(str(log_dir), bench_metrics, 5)
        self.assertIsNone(error)
        self.assertEqual(tail, ["first", "second"])
        self.assertEqual(out["merge_sort_operators"], 0)
        self.assertNotIn("discard_rows", out)


class DryRunCliTest(Scratch):
    def test_dry_run_prints_the_5x2x2_matrix(self):
        out_dir = self.tmp / "out"
        spill = self.tmp / "spill"
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "sort_merge_bench.py"),
                "--gpu-limit-bytes",
                "4294967296",
                "--dry-run",
                f"--out-dir={out_dir}",
                f"--spill-dir={spill}",
                "--extension=/nonexistent/sirius.duckdb_extension",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        self.assertIn("5 datasets x 2 modes x 2 compression x 1 repeat = 20 runs", out)
        self.assertEqual(out.count("===== ["), 20)
        self.assertEqual(out.count("FORMAT sirius_discard"), 20)
        self.assertEqual(out.count("usage_limit_bytes: 4294967296"), 20)
        self.assertEqual(out.count("spill_min_batch_bytes: 1048576"), 10)
        self.assertEqual(out.count("SET sort_merge_mode = 'one_shot'"), 10)
        self.assertEqual(out.count("SET spill_compression = true"), 10)
        for name in smb.DATASETS:
            self.assertIn(f"{name}__one_shot__uncompressed__r1 =====", out)
            self.assertIn(f"{name}__windowed__compressed__r1 =====", out)
        self.assertIn("Dry run: nothing was executed", out)
        # Nothing is created and no extension is needed.
        self.assertFalse(out_dir.exists())
        self.assertFalse(spill.exists())


if __name__ == "__main__":
    unittest.main()
