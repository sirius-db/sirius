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

"""Checks of the committed full run (results/full/) and RESULTS.md against the bench-full-run
done criteria.

No GPU needed: these read the CSVs, the per-run files and the log tails the workstation runs left
behind, check that the matrix was run with the owner's sizing (whole GPU, 5x data, 96 GiB host
cap, windowed/one_shot pairs, one_shot timeout 4x windowed and at least 30 min), and that every
number RESULTS.md quotes in its tables matches the CSV it came from.

Run: pixi run python test/sort_performance/test_full_results.py   (also collectable by pytest)
"""

import csv
import json
import re
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
FULL_DIR = HERE / "results" / "full"
AB_DIR = FULL_DIR / "whole-gpu-5x"
EARLIER_DIR = FULL_DIR / "limit-24g-10x"
PROBE_DIR = AB_DIR / "probe"
RESULTS_MD = HERE / "RESULTS.md"

# Run order of the driver: datasets in this order, uncompressed then compressed, windowed then
# one_shot within a cell.
DATASET_ORDER = (
    "uniform-1key",
    "zipf-1key",
    "sequential",
    "few_distinct-strings",
    "normal-multikey",
)
EARLIER_DATASETS = {"uniform-1key", "zipf-1key", "normal-multikey"}

# 96 GiB is rejected at startup; 94 GiB is the largest value that started (RESULTS.md).
REQUESTED_GPU_LIMIT = 96 * 2**30
AB_GPU_LIMIT = 94 * 2**30
EARLIER_GPU_LIMIT = 24 * 2**30
HOST_CAP = 96 * 2**30
SCALE = 5
EARLIER_SCALE = 10
WINDOWED_TIMEOUT_S = 10800
MIN_ONE_SHOT_TIMEOUT_S = 1800

ONE_SHOT_FAILURES = {"oom", "error", "timeout"}

PER_RUN_FILES = (
    "config.yaml",
    "query.sql",
    "job.json",
    "row.json",
    "child_stdout.txt",
    "sirius_log_tail.txt",
)

STATS_COLUMNS = (
    "windowed_partitions",
    "one_shot_partitions",
    "mergers_built",
    "rounds",
    "rows_out",
    "refill_bytes_compressed",
    "runs_read_compressed",
    "shrinks",
)


def _read_rows(directory: Path) -> list[dict[str, str]]:
    with open(directory / "results.csv", newline="") as f:
        return list(csv.DictReader(f))


def _arm(row: dict[str, str]) -> str:
    return "compressed" if row["compression"] == "True" else "uncompressed"


def _run_name(row: dict[str, str]) -> str:
    return f"{row['dataset']}__{row['mode']}__{_arm(row)}__r{row['repeat']}"


def _int(row: dict[str, str], column: str) -> int:
    return int(row[column]) if row[column] != "" else 0


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _num(cell: str) -> int:
    return int(cell.replace(",", ""))


def _gb(value: str) -> int:
    return round(int(value or 0) / 1e9)


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin)]


def _table_rows(text: str) -> list[list[str]]:
    """Body rows of every markdown table in `text`, cells stripped."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def _per_dataset_tables(section: str, heading: str) -> dict[str, list[list[str]]]:
    """Run tables (Mode | Compression | Status | ...) keyed by the dataset of their heading."""
    tables = {}
    parts = re.split(rf"^{heading} `([^`]+)`.*$", section, flags=re.MULTILINE)
    for dataset, body in zip(parts[1::2], parts[2::2]):
        tables[dataset] = [
            r for r in _table_rows(body) if r[0] in ("windowed", "one_shot")
        ]
    return tables


class RunChecks:
    """Checks shared by the A/B matrix and the earlier 24 GiB runs."""

    directory: Path
    gpu_limit: int
    scale: int

    @classmethod
    def setUpClass(cls):
        if not (cls.directory / "results.csv").exists():
            raise unittest.SkipTest(f"no full results under {cls.directory}")
        cls.rows = _read_rows(cls.directory)

    def windowed(self):
        return [r for r in self.rows if r["mode"] == "windowed"]

    def run_dir(self, row) -> Path:
        return self.directory / "per_run" / _run_name(row)

    def run_info(self, row) -> dict:
        """The run_info.json of the harness invocation that ran `row`."""
        return json.loads((self.run_dir(row) / "run_info.json").read_text())

    def test_every_run_has_its_per_run_files(self):
        for row in self.rows:
            files = PER_RUN_FILES + (
                ("result.json",) if row["status"] != "timeout" else ()
            )
            for name in files:
                with self.subTest(run=_run_name(row), file=name):
                    self.assertTrue((self.run_dir(row) / name).is_file())

    def test_row_json_matches_csv(self):
        for row in self.rows:
            with self.subTest(run=_run_name(row)):
                row_json = json.loads((self.run_dir(row) / "row.json").read_text())
                for column in (
                    "dataset",
                    "mode",
                    "status",
                    "rows",
                    "bytes",
                    "gpu_limit_bytes",
                ):
                    self.assertEqual(str(row_json[column]), row[column], column)
                self.assertEqual(str(row_json["compression"]), row["compression"])
                self.assertEqual(row_json["error"], row["error"])

    def test_query_and_config_match_the_arm(self):
        for row in self.rows:
            run_dir = self.run_dir(row)
            with self.subTest(run=run_dir.name):
                sql = (run_dir / "query.sql").read_text()
                self.assertIn(f"SET sort_merge_mode = '{row['mode']}'", sql)
                self.assertIn(
                    f"SET spill_compression = {row['compression'].lower()}", sql
                )
                self.assertIn("FORMAT sirius_discard", sql)
                config = (run_dir / "config.yaml").read_text()
                self.assertIn(f"usage_limit_bytes: {self.gpu_limit}", config)
                self.assertIn(f"capacity_bytes: {HOST_CAP}", config)

    def test_gpu_limit_and_data_size(self):
        target = self.scale * self.gpu_limit
        for row in self.rows:
            with self.subTest(run=_run_name(row)):
                self.assertEqual(int(row["gpu_limit_bytes"]), self.gpu_limit)
                rows, size = int(row["rows"]), int(row["bytes"])
                self.assertEqual(
                    rows, int(row["num_batches"]) * int(row["rows_per_batch"])
                )
                batch_bytes = size // int(row["num_batches"])
                # The harness rounds the batch count so the data is the scale times the limit,
                # give or take one batch.
                self.assertLessEqual(abs(size - target), batch_bytes)

    def test_run_info_records_the_sizing(self):
        for row in self.rows:
            with self.subTest(run=_run_name(row)):
                argv = self.run_info(row)["argv"]
                self.assertEqual(
                    int(_argv_value(argv, "--gpu-limit-bytes")), self.gpu_limit
                )
                self.assertEqual(int(_argv_value(argv, "--scale")), self.scale)
                self.assertEqual(_argv_value(argv, "--device"), "0")
                self.assertIn("--base-config", argv)

    def test_windowed_runs_are_complete(self):
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
                self.assertEqual(_int(row, "fallback_conversions"), 0)
                self.assertGreater(_int(row, "rounds"), 0)
                # Intermediate passes re-emit rows, so at least every input row comes out once.
                self.assertGreaterEqual(_int(row, "rows_out"), int(row["rows"]))

    def test_compressed_arm_reads_compressed_runs(self):
        for row in self.windowed():
            with self.subTest(run=_run_name(row)):
                if row["compression"] == "True":
                    self.assertGreater(_int(row, "refill_bytes_compressed"), 0)
                    self.assertGreater(_int(row, "runs_read_compressed"), 0)
                else:
                    self.assertEqual(_int(row, "refill_bytes_compressed"), 0)
                    self.assertEqual(_int(row, "runs_read_compressed"), 0)

    def test_data_exceeds_the_gpu_so_runs_spill(self):
        for row in self.windowed():
            with self.subTest(run=_run_name(row)):
                self.assertGreater(_int(row, "spill_bytes_host"), 0)

    def test_csv_stats_agree_with_the_log_tail(self):
        for row in self.windowed():
            run_dir = self.run_dir(row)
            with self.subTest(run=run_dir.name):
                log = (run_dir / "sirius_log_tail.txt").read_text(errors="replace")
                stats = bench_metrics.parse_merge_sort_stats(log)
                self.assertTrue(stats, "no [merge_sort][stats] line in the log tail")
                summed = bench_metrics.sum_merge_sort_stats(stats)
                for column in STATS_COLUMNS:
                    self.assertEqual(summed[column], _int(row, column), column)


class TestWholeGpuMatrix(RunChecks, unittest.TestCase):
    directory = AB_DIR
    gpu_limit = AB_GPU_LIMIT
    scale = SCALE

    def test_every_run_has_its_own_run_info(self):
        for row in self.rows:
            with self.subTest(run=_run_name(row)):
                self.assertTrue((self.run_dir(row) / "run_info.json").is_file())

    def test_matrix_has_a_row_for_every_run(self):
        cells = [
            (r["dataset"], r["mode"], r["compression"], r["repeat"]) for r in self.rows
        ]
        expected = {
            (d, m, c, "1")
            for d in DATASET_ORDER
            for m in ("windowed", "one_shot")
            for c in ("True", "False")
        }
        self.assertEqual(set(cells), expected)
        self.assertEqual(len(cells), len(expected))

    def test_rows_are_in_run_pair_order(self):
        expected = [
            (d, m, c)
            for d in DATASET_ORDER
            for c in ("False", "True")
            for m in ("windowed", "one_shot")
        ]
        self.assertEqual(
            [(r["dataset"], r["mode"], r["compression"]) for r in self.rows], expected
        )

    def test_runs_started_in_pair_order(self):
        started = [
            datetime.fromisoformat(
                json.loads((self.run_dir(r) / "run_info.json").read_text())["started"]
            )
            for r in self.rows
        ]
        self.assertEqual(started, sorted(started))
        self.assertEqual(len(set(started)), len(started))

    def test_each_run_was_its_own_invocation(self):
        for row in self.rows:
            with self.subTest(run=_run_name(row)):
                argv = json.loads((self.run_dir(row) / "run_info.json").read_text())[
                    "argv"
                ]
                self.assertEqual(_argv_value(argv, "--datasets"), row["dataset"])
                self.assertEqual(_argv_value(argv, "--modes"), row["mode"])
                self.assertEqual(
                    _argv_value(argv, "--compression"), row["compression"].lower()
                )

    def test_one_shot_timeout_is_four_times_windowed_and_at_least_30_min(self):
        windowed = {(r["dataset"], r["compression"]): r for r in self.windowed()}
        for row in self.rows:
            argv = json.loads((self.run_dir(row) / "run_info.json").read_text())["argv"]
            timeout = float(_argv_value(argv, "--timeout-s"))
            with self.subTest(run=_run_name(row)):
                if row["mode"] == "windowed":
                    self.assertEqual(timeout, WINDOWED_TIMEOUT_S)
                    self.assertLess(float(row["wall_s"]), timeout)
                    continue
                w = int(float(windowed[(row["dataset"], row["compression"])]["wall_s"]))
                self.assertEqual(timeout, max(4 * w, MIN_ONE_SHOT_TIMEOUT_S))
                if row["status"] == "timeout":
                    self.assertEqual(
                        row["error"], f"killed after --timeout-s {timeout}"
                    )
                    self.assertGreaterEqual(float(row["wall_s"]), timeout)
                else:
                    self.assertLess(float(row["wall_s"]), timeout)

    def test_one_shot_runs_are_recorded_results(self):
        for row in self.rows:
            if row["mode"] != "one_shot":
                continue
            with self.subTest(run=_run_name(row)):
                if row["status"] == "ok":
                    self.assertEqual(row["error"], "")
                    self.assertEqual(row["rows_counted"], row["rows"])
                    self.assertEqual(_int(row, "order_violations"), 0)
                    self.assertEqual(_int(row, "windowed_partitions"), 0)
                else:
                    self.assertIn(row["status"], ONE_SHOT_FAILURES)
                    self.assertNotEqual(row["error"], "")

    def test_base_config_caps_the_host_tier(self):
        self.assertIn(
            f"capacity_bytes: {HOST_CAP}", (AB_DIR / "base_config.yml").read_text()
        )

    def test_probe_rejects_whole_gpu_and_accepts_the_limit_used(self):
        statuses = {}
        for path in PROBE_DIR.glob("*/row.json"):
            row_json = json.loads(path.read_text())
            self.assertEqual(int(path.parent.name), row_json["gpu_limit_bytes"])
            statuses[row_json["gpu_limit_bytes"]] = row_json["status"]
        self.assertEqual(statuses.get(REQUESTED_GPU_LIMIT), "error")
        self.assertEqual(statuses.get(AB_GPU_LIMIT), "ok")
        # The limit used is the largest value that started.
        self.assertEqual(max(v for v, s in statuses.items() if s == "ok"), AB_GPU_LIMIT)


class TestEarlierRuns(RunChecks, unittest.TestCase):
    directory = EARLIER_DIR
    gpu_limit = EARLIER_GPU_LIMIT
    scale = EARLIER_SCALE

    def run_info(self, row) -> dict:
        # The six runs were one harness invocation, with one run_info.json for all of them.
        return json.loads((EARLIER_DIR / "run_info.json").read_text())

    def test_six_windowed_runs(self):
        cells = {(r["dataset"], r["mode"], r["compression"]) for r in self.rows}
        expected = {
            (d, "windowed", c) for d in EARLIER_DATASETS for c in ("True", "False")
        }
        self.assertEqual(cells, expected)
        self.assertEqual(len(self.rows), len(expected))


class TestResultsMarkdown(unittest.TestCase):
    """RESULTS.md quotes the CSVs; every number in its run tables must match them."""

    @classmethod
    def setUpClass(cls):
        if not RESULTS_MD.exists() or not (AB_DIR / "results.csv").exists():
            raise unittest.SkipTest("no RESULTS.md or full results")
        cls.text = RESULTS_MD.read_text()
        cls.ab = {
            (r["dataset"], r["mode"], r["compression"]): r for r in _read_rows(AB_DIR)
        }
        cls.earlier = {
            (r["dataset"], r["mode"], r["compression"]): r
            for r in _read_rows(EARLIER_DIR)
        }

    def test_required_sections(self):
        for heading in (
            "## Summary",
            "## Setup",
            "## Results per dataset",
            "## Observations",
            "## Recommendation",
            "## Earlier runs",
            '## What "Peak GPU" measures',
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, self.text)

    def test_setup_states_gpu_limit_and_host_cap(self):
        setup = _section(self.text, "## Setup", "## Results per dataset")
        self.assertIn(f"usage_limit_bytes = {AB_GPU_LIMIT}", setup)
        self.assertIn(f"capacity_bytes = {HOST_CAP}", setup)
        self.assertIn(f"--scale {SCALE}", setup)

    def test_setup_data_sizes_match_the_csv(self):
        setup = _section(self.text, "## Setup", "## Results per dataset")
        sizes = {
            r[0].strip("`"): r
            for r in _table_rows(setup)
            if r[0].strip("`") in DATASET_ORDER
        }
        self.assertEqual(set(sizes), set(DATASET_ORDER))
        for dataset, cells in sizes.items():
            row = self.ab[(dataset, "windowed", "False")]
            with self.subTest(dataset=dataset):
                self.assertEqual(_num(cells[1]), int(row["num_batches"]))
                self.assertEqual(_num(cells[2]), int(row["rows"]))
                self.assertEqual(int(cells[3]), int(row["bytes"]) // int(row["rows"]))
                self.assertEqual(cells[4], f"{int(row['bytes']) / 2**30:.1f} GiB")

    def test_summary_table_matches_the_csv(self):
        summary = _section(self.text, "## Summary", "## Setup")
        rows = [r for r in _table_rows(summary) if r[0] in DATASET_ORDER]
        self.assertEqual(len(rows), len(self.ab) // 2)
        for cells in rows:
            dataset, comp = cells[0], "True" if cells[1] == "on" else "False"
            windowed = self.ab[(dataset, "windowed", comp)]
            one_shot = self.ab[(dataset, "one_shot", comp)]
            with self.subTest(dataset=dataset, compression=comp):
                self.assertEqual(cells[2], windowed["status"])
                self.assertEqual(_num(cells[3]), round(float(windowed["wall_s"])))
                gib_s = int(windowed["bytes"]) / 2**30 / float(windowed["wall_s"])
                self.assertAlmostEqual(float(cells[4]), gib_s, delta=0.006)
                self.assertTrue(cells[5].startswith(one_shot["status"]), cells[5])
                self.assertEqual(_num(cells[6]), round(float(one_shot["wall_s"])))
                timeout = max(
                    4 * int(float(windowed["wall_s"])), MIN_ONE_SHOT_TIMEOUT_S
                )
                self.assertEqual(_num(cells[7]), timeout)

    def _check_run_table(self, tables, by_cell, expected_datasets):
        self.assertEqual(set(tables), expected_datasets)
        seen = 0
        for dataset, rows in tables.items():
            for cells in rows:
                mode, comp = cells[0], "True" if cells[1] == "on" else "False"
                row = by_cell[(dataset, mode, comp)]
                seen += 1
                with self.subTest(dataset=dataset, mode=mode, compression=comp):
                    self.assertTrue(cells[2].startswith(row["status"]), cells[2])
                    self.assertEqual(_num(cells[3]), round(float(row["wall_s"])))
                    self.assertEqual(_num(cells[4]), int(row["peak_gpu_mib"]))
                    self.assertEqual(_num(cells[5]), _gb(row["spill_bytes_host"]))
                    self.assertEqual(_num(cells[6]), _gb(row["spill_bytes_disk"]))
                    self.assertEqual(_num(cells[7]), int(row["downgrade_requests"]))
                    self.assertEqual(_num(cells[8]), int(row["oom_retries"]))
        self.assertEqual(seen, len(by_cell))

    def test_per_dataset_tables_match_the_csv(self):
        section = _section(self.text, "## Results per dataset", "## Observations")
        self._check_run_table(
            _per_dataset_tables(section, "###"), self.ab, set(DATASET_ORDER)
        )

    def test_earlier_run_tables_match_the_csv(self):
        section = _section(self.text, "## Earlier runs", "## Recommendation")
        self.assertIn(f"{EARLIER_GPU_LIMIT:,} bytes", section)
        self._check_run_table(
            _per_dataset_tables(section, "####"), self.earlier, EARLIER_DATASETS
        )

    def test_one_shot_tally_matches_the_csv(self):
        one_shot = [r for r in self.ab.values() if r["mode"] == "one_shot"]
        windowed_ok = sum(
            r["status"] == "ok" for r in self.ab.values() if r["mode"] == "windowed"
        )
        one_shot_ok = sum(r["status"] == "ok" for r in one_shot)
        summary = _section(self.text, "## Summary", "## Setup")
        self.assertIn(f"finished all {windowed_ok} runs", summary)
        self.assertIn(f"finished none of its {len(one_shot)}", summary)
        self.assertEqual(one_shot_ok, 0)
        timeouts = sum(r["status"] == "timeout" for r in one_shot)
        ooms = sum(r["status"] == "oom" for r in one_shot)
        row_limit = sum("exceeds row limit" in r["error"] for r in one_shot)
        self.assertEqual(timeouts + ooms + row_limit, len(one_shot))
        self.assertIn(f"{timeouts} runs were killed at the timeout", summary)
        self.assertIn(f"{row_limit} stopped on a cuDF row-count limit", summary)
        self.assertIn(f"{ooms} ran out of GPU memory", summary)

    def test_recommendation_gives_a_timeline_and_conditions(self):
        rec = _section(self.text, "## Recommendation", "## Problems found")
        self.assertIn("Deprecate `one_shot` now", rec)
        self.assertRegex(rec, r"(?m)^1\. \*\*`windowed` is within \d+ % of `one_shot`")


if __name__ == "__main__":
    unittest.main()
