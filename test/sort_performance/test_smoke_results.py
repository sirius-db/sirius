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

"""Checks of the committed smoke run (results/smoke/) against the bench-smoke done criteria.

No GPU needed: these read the CSV, the per-run files and the log tails the workstation run left
behind, and cross-check them with bench_metrics.py's parser. They fail if the smoke results are
regenerated from a run that no longer meets the criteria.

Run: pixi run python test/sort_performance/test_smoke_results.py   (also collectable by pytest)
"""

import csv
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_metrics  # noqa: E402

SMOKE_DIR = Path(__file__).resolve().parent / "results" / "smoke"
DISK_TIER_DIR = SMOKE_DIR / "disk-tier"

DATASETS = {
    "uniform-1key",
    "zipf-1key",
    "normal-multikey",
    "few_distinct-strings",
    "sequential",
}
MODES = {"one_shot", "windowed"}
GPU_LIMIT_BYTES = 4294967296

# Whole-device memory.used of a process that has only loaded the extension (NOTES.md: the 20
# runs of the first attempt, which failed in 0.0 s). peak_gpu_mib is whole-device, so the part a
# query adds is peak - this baseline.
IDLE_DEVICE_MIB = 4682
PEAK_MARGIN_MIB = 512

PER_RUN_FILES = (
    "config.yaml",
    "query.sql",
    "job.json",
    "result.json",
    "row.json",
    "child_stdout.txt",
    "sirius_log_tail.txt",
)

# Columns the [merge_sort][stats] log line feeds; the CSV must agree with the log tail.
STATS_COLUMNS = (
    "windowed_partitions",
    "one_shot_partitions",
    "mergers_built",
    "rounds",
    "refill_bytes_compressed",
    "runs_read_compressed",
)


def _read_rows(directory: Path) -> list[dict[str, str]]:
    with open(directory / "results.csv", newline="") as f:
        return list(csv.DictReader(f))


def _run_name(row: dict[str, str]) -> str:
    arm = "compressed" if row["compression"] == "True" else "uncompressed"
    return f"{row['dataset']}__{row['mode']}__{arm}__r{row['repeat']}"


def _int(row: dict[str, str], column: str) -> int:
    return int(row[column]) if row[column] != "" else 0


class SmokeChecks:
    """Checks shared by the main matrix and the disk-tier run."""

    directory: Path

    @classmethod
    def setUpClass(cls):
        if not (cls.directory / "results.csv").exists():
            raise unittest.SkipTest(f"no smoke results under {cls.directory}")
        cls.rows = _read_rows(cls.directory)

    def windowed(self):
        return [r for r in self.rows if r["mode"] == "windowed"]

    def test_every_run_has_its_per_run_files(self):
        for row in self.rows:
            run_dir = self.directory / "per_run" / _run_name(row)
            for name in PER_RUN_FILES:
                with self.subTest(run=run_dir.name, file=name):
                    self.assertTrue((run_dir / name).is_file())

    def test_row_json_matches_csv(self):
        for row in self.rows:
            run_dir = self.directory / "per_run" / _run_name(row)
            with self.subTest(run=run_dir.name):
                row_json = json.loads((run_dir / "row.json").read_text())
                self.assertEqual(row_json["dataset"], row["dataset"])
                self.assertEqual(row_json["mode"], row["mode"])
                self.assertEqual(str(row_json["compression"]), row["compression"])
                self.assertEqual(str(row_json["rows"]), row["rows"])

    def test_query_and_config_match_the_arm(self):
        for row in self.rows:
            run_dir = self.directory / "per_run" / _run_name(row)
            with self.subTest(run=run_dir.name):
                sql = (run_dir / "query.sql").read_text()
                self.assertIn(f"SET sort_merge_mode = '{row['mode']}'", sql)
                self.assertIn(
                    f"SET spill_compression = {row['compression'].lower()}", sql
                )
                self.assertIn("FORMAT sirius_discard", sql)
                self.assertIn("check_order true", sql)
                config = (run_dir / "config.yaml").read_text()
                self.assertIn(f"usage_limit_bytes: {row['gpu_limit_bytes']}", config)

    def test_windowed_runs_are_complete_and_ordered(self):
        windowed = self.windowed()
        self.assertTrue(windowed)
        for row in windowed:
            with self.subTest(run=_run_name(row)):
                self.assertEqual(row["status"], "ok", row["error"])
                self.assertEqual(row["error"], "")
                self.assertEqual(row["rows_counted"], row["rows"])
                self.assertEqual(row["discard_rows"], row["rows"])
                self.assertEqual(_int(row, "order_violations"), 0)

    def test_windowed_route_was_taken(self):
        for row in self.windowed():
            with self.subTest(run=_run_name(row)):
                self.assertGreater(_int(row, "windowed_partitions"), 0)
                self.assertEqual(_int(row, "one_shot_partitions"), 0)
                self.assertGreater(_int(row, "mergers_built"), 0)
                self.assertGreater(_int(row, "rounds"), 0)
                self.assertGreater(_int(row, "rows_out"), 0)

    def test_compressed_arm_reads_compressed_runs(self):
        for row in self.windowed():
            with self.subTest(run=_run_name(row)):
                if row["compression"] == "True":
                    self.assertGreater(_int(row, "refill_bytes_compressed"), 0)
                    self.assertGreater(_int(row, "runs_read_compressed"), 0)
                else:
                    self.assertEqual(_int(row, "refill_bytes_compressed"), 0)
                    self.assertEqual(_int(row, "runs_read_compressed"), 0)

    def test_csv_stats_agree_with_the_log_tail(self):
        for row in self.windowed():
            run_dir = self.directory / "per_run" / _run_name(row)
            with self.subTest(run=run_dir.name):
                log = (run_dir / "sirius_log_tail.txt").read_text(errors="replace")
                stats = bench_metrics.parse_merge_sort_stats(log)
                self.assertTrue(stats, "no [merge_sort][stats] line in the log tail")
                summed = bench_metrics.sum_merge_sort_stats(stats)
                for column in STATS_COLUMNS:
                    self.assertEqual(summed[column], _int(row, column), column)

    def test_peak_gpu_above_idle_is_within_the_limit(self):
        limit_mib = GPU_LIMIT_BYTES // (1024 * 1024)
        for row in self.rows:
            with self.subTest(run=_run_name(row)):
                self.assertEqual(int(row["gpu_limit_bytes"]), GPU_LIMIT_BYTES)
                peak = _int(row, "peak_gpu_mib")
                self.assertGreater(peak, 0)
                self.assertLessEqual(
                    peak - IDLE_DEVICE_MIB, limit_mib + PEAK_MARGIN_MIB
                )


class TestSmokeMatrix(SmokeChecks, unittest.TestCase):
    directory = SMOKE_DIR

    def test_matrix_is_complete(self):
        cells = {(r["dataset"], r["mode"], r["compression"]) for r in self.rows}
        expected = {
            (d, m, c) for d in DATASETS for m in MODES for c in ("True", "False")
        }
        self.assertEqual(cells, expected)
        self.assertEqual(len(self.rows), len(expected))

    def test_one_shot_runs_complete_or_fail_cleanly(self):
        for row in self.rows:
            if row["mode"] != "one_shot":
                continue
            with self.subTest(run=_run_name(row)):
                result = json.loads(
                    (SMOKE_DIR / "per_run" / _run_name(row) / "result.json").read_text()
                )
                if row["status"] == "ok":
                    self.assertEqual(row["error"], "")
                    self.assertEqual(row["rows_counted"], row["rows"])
                    self.assertEqual(_int(row, "order_violations"), 0)
                    self.assertEqual(_int(row, "windowed_partitions"), 0)
                else:
                    # A failure is a recorded status with its error text, never a harness crash.
                    self.assertIn(row["status"], {"oom", "error", "timeout"})
                    self.assertNotEqual(row["error"], "")
                    self.assertNotEqual(result["error"], "")

    def test_no_spill_reached_disk(self):
        # The 12 GiB smoke data fits the host tier; the disk path is covered by disk-tier/.
        for row in self.rows:
            with self.subTest(run=_run_name(row)):
                self.assertEqual(_int(row, "spill_bytes_disk"), 0)


class TestSmokeDiskTier(SmokeChecks, unittest.TestCase):
    directory = DISK_TIER_DIR

    def test_runs_spilled_to_disk(self):
        for row in self.rows:
            with self.subTest(run=_run_name(row)):
                self.assertEqual(row["mode"], "windowed")
                self.assertGreater(_int(row, "spill_bytes_disk"), 0)

    def test_base_config_caps_the_host_tier(self):
        self.assertIn(
            "capacity_bytes: 4294967296",
            (DISK_TIER_DIR / "base_config.yml").read_text(),
        )


if __name__ == "__main__":
    unittest.main()
