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

"""Tests of bench_metrics.py. No GPU needed: the sampler runs against a fake nvidia-smi.

Run: pixi run python test/sort_performance/test_bench_metrics.py   (also collectable by pytest)
"""

import os
import stat
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_metrics  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


class MergeSortStatsTest(unittest.TestCase):
    def setUp(self):
        self.stats = bench_metrics.parse_merge_sort_stats(
            read_fixture("merge_sort_stats.log")
        )

    def test_one_entry_per_operator_and_malformed_lines_ignored(self):
        # The fixture also holds a non-integer value, a line without op, a token without '=',
        # a bare tag and a "[merge_sort]" line without "[stats]".
        self.assertEqual([entry.op for entry in self.stats], [7, 9, 11])

    def test_values_exclude_op(self):
        first = self.stats[0]
        self.assertNotIn("op", first.values)
        self.assertEqual(len(first.values), 19)
        self.assertEqual(first.values["windowed_partitions"], 4)
        self.assertEqual(first.values["rows_out"], 1000000)
        self.assertEqual(first.values["refill_bytes_disk"], 3200000)
        self.assertEqual(first.values["shrinks"], 1)

    def test_appended_unknown_key_is_kept(self):
        self.assertEqual(self.stats[1].values["future_counter"], 11)
        self.assertNotIn("future_counter", self.stats[0].values)

    def test_sum(self):
        self.assertEqual(
            bench_metrics.sum_merge_sort_stats(self.stats),
            {
                "windowed_partitions": 6,
                "one_shot_partitions": 4,
                "windowed_tasks": 18,
                "intermediate_tasks": 3,
                "mergers_built": 7,
                "rounds": 50,
                "rows_out": 1500300,
                "round_rows_min": 250,  # the 0 of the operator with no rounds is skipped
                "round_rows_max": 80000,
                "refill_rows": 1300000,
                "refill_bytes": 10400000,
                "refill_bytes_host": 7200000,
                "refill_bytes_disk": 3200000,
                "refill_bytes_compressed": 1000000,
                "runs_read_compressed": 6,
                "window_growths": 2,
                "compacted_rows": 1234,
                "fallback_conversions": 2,
                "shrinks": 1,
                "future_counter": 11,
            },
        )

    def test_sum_min_stays_zero_when_every_min_is_zero(self):
        stats = [
            bench_metrics.MergeSortStats(op=1, values={"round_rows_min": 0}),
            bench_metrics.MergeSortStats(op=2, values={"round_rows_min": 0}),
        ]
        self.assertEqual(
            bench_metrics.sum_merge_sort_stats(stats), {"round_rows_min": 0}
        )

    def test_empty(self):
        self.assertEqual(bench_metrics.parse_merge_sort_stats(""), [])
        self.assertEqual(bench_metrics.sum_merge_sort_stats([]), {})

    def test_tag_constant(self):
        self.assertEqual(bench_metrics.MERGE_SORT_STATS_TAG, "[merge_sort][stats]")


class DowngradeTotalsTest(unittest.TestCase):
    def test_sums_monitor_request_done_lines(self):
        # The fixture also holds the DEBUG "[downgrade] ... request done" line (no monitor tag),
        # a monitor line that is not "request done" and one with a non-integer batch count.
        totals = bench_metrics.parse_downgrade_totals(read_fixture("downgrade.log"))
        self.assertEqual(
            totals,
            bench_metrics.DowngradeTotals(
                requests=3,
                batches=8,
                bytes=11000000,
                to_host_batches=2,
                to_host_bytes=2000000,
                to_disk_batches=6,
                to_disk_bytes=9000000,
            ),
        )

    def test_no_lines(self):
        self.assertEqual(
            bench_metrics.parse_downgrade_totals("nothing here\n"),
            bench_metrics.DowngradeTotals(0, 0, 0, 0, 0, 0, 0),
        )


class OomRetriesTest(unittest.TestCase):
    def test_counts_only_oom_reschedules(self):
        # Three OOM reschedules; the kernel-launch reschedule, the "shrank for retry" line, the
        # ERROR line mentioning OOM and the garbled line do not count.
        self.assertEqual(
            bench_metrics.count_oom_retries(read_fixture("oom_retries.log")), 3
        )

    def test_no_lines(self):
        self.assertEqual(bench_metrics.count_oom_retries(""), 0)


class DiscardSinkTest(unittest.TestCase):
    def test_parses_line(self):
        self.assertEqual(
            bench_metrics.parse_discard_sink(read_fixture("discard_sink.log")),
            {"rows": 1500300, "bytes": 12002400, "order_violations": 0},
        )

    def test_missing_line(self):
        self.assertIsNone(
            bench_metrics.parse_discard_sink(read_fixture("downgrade.log"))
        )

    def test_malformed_line_ignored(self):
        self.assertIsNone(
            bench_metrics.parse_discard_sink("[discard_sink] rows=1 bytes=many\n")
        )
        self.assertIsNone(bench_metrics.parse_discard_sink("[discard_sink] rows=1\n"))

    def test_several_sinks_are_summed(self):
        text = (
            "[info] [discard_sink] rows=10 bytes=80 order_violations=1\n"
            "[info] [discard_sink] rows=5 bytes=40 order_violations=2\n"
        )
        self.assertEqual(
            bench_metrics.parse_discard_sink(text),
            {"rows": 15, "bytes": 120, "order_violations": 3},
        )


class LogTailTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = os.path.join(self._dir.name, "sirius.log")

    def _append(self, text: str) -> None:
        with open(self.path, "a") as log_file:
            log_file.write(text)

    def test_missing_file_then_appends(self):
        tail = bench_metrics.LogTail(self.path)
        self.assertEqual(tail.read_new(), "")
        self._append("line 1\nline 2\n")
        self.assertEqual(tail.read_new(), "line 1\nline 2\n")
        self.assertEqual(tail.read_new(), "")
        self._append("line 3\n")
        self.assertEqual(tail.read_new(), "line 3\n")

    def test_partial_line_is_held_until_complete(self):
        tail = bench_metrics.LogTail(self.path)
        self._append("line 1\n[merge_sort][stats] op=1 rou")
        self.assertEqual(tail.read_new(), "line 1\n")
        self.assertEqual(tail.read_new(), "")
        self._append("nds=3\n")
        text = tail.read_new()
        self.assertEqual(text, "[merge_sort][stats] op=1 rounds=3\n")
        self.assertEqual(
            bench_metrics.parse_merge_sort_stats(text),
            [bench_metrics.MergeSortStats(op=1, values={"rounds": 3})],
        )

    def test_truncation_restarts_from_the_beginning(self):
        tail = bench_metrics.LogTail(self.path)
        self._append("a long first run line\nand another\n")
        self.assertEqual(tail.read_new(), "a long first run line\nand another\n")
        with open(self.path, "w") as log_file:
            log_file.write("new run\n")
        self.assertEqual(tail.read_new(), "new run\n")
        self._append("more\n")
        self.assertEqual(tail.read_new(), "more\n")

    def test_file_removed_after_reading(self):
        tail = bench_metrics.LogTail(self.path)
        self._append("line 1\n")
        self.assertEqual(tail.read_new(), "line 1\n")
        os.remove(self.path)
        self.assertEqual(tail.read_new(), "")


class GpuMemorySamplerTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.bin_dir = os.path.join(self._dir.name, "bin")
        os.mkdir(self.bin_dir)
        self.args_file = os.path.join(self._dir.name, "args")
        self.count_file = os.path.join(self._dir.name, "count")

    def _install_fake_nvidia_smi(self, body: str) -> None:
        path = os.path.join(self.bin_dir, "nvidia-smi")
        with open(path, "w") as script:
            script.write("#!/bin/sh\n" + body)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

    def _with_path(self, path: str):
        patcher = mock.patch.dict(os.environ, {"PATH": path})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _wait_for_samples(self, count: int) -> None:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with open(self.count_file) as count_file:
                    if len(count_file.read()) >= count:
                        return
            except FileNotFoundError:
                pass
            time.sleep(0.01)
        self.fail(f"the fake nvidia-smi was not called {count} times")

    def test_peak_of_the_samples(self):
        # Prints 100, 5000, 300, then 200 for every later call: the peak is in the middle.
        self._install_fake_nvidia_smi(
            f'echo "$*" > "{self.args_file}"\n'
            f'printf x >> "{self.count_file}"\n'
            f'n=$(wc -c < "{self.count_file}")\n'
            'case "$n" in\n'
            "  1) echo 100 ;;\n"
            "  2) echo 5000 ;;\n"
            "  3) echo 300 ;;\n"
            "  *) echo 200 ;;\n"
            "esac\n"
        )
        self._with_path(self.bin_dir + os.pathsep + os.environ["PATH"])
        sampler = bench_metrics.GpuMemorySampler(device_index=1, interval_s=0.01)
        sampler.start()
        self._wait_for_samples(4)
        self.assertEqual(sampler.stop(), 5000)
        with open(self.args_file) as args_file:
            self.assertEqual(
                args_file.read().strip(),
                "--query-gpu=memory.used --format=csv,noheader,nounits -i 1",
            )

    def test_short_run_still_gets_a_sample(self):
        self._install_fake_nvidia_smi(
            f'printf x >> "{self.count_file}"\necho " 4321"\n'
        )
        self._with_path(self.bin_dir + os.pathsep + os.environ["PATH"])
        sampler = bench_metrics.GpuMemorySampler(device_index=0, interval_s=60)
        sampler.start()
        self.assertEqual(sampler.stop(), 4321)

    def test_nvidia_smi_unavailable(self):
        self._with_path(self.bin_dir)  # empty directory: no nvidia-smi anywhere on PATH
        sampler = bench_metrics.GpuMemorySampler(device_index=0, interval_s=0.01)
        sampler.start()
        time.sleep(0.05)
        self.assertIsNone(sampler.stop())

    def test_nvidia_smi_failing_or_printing_garbage(self):
        self._install_fake_nvidia_smi(
            f'printf x >> "{self.count_file}"\n'
            f'n=$(wc -c < "{self.count_file}")\n'
            'if [ "$n" -eq 1 ]; then echo "No devices were found"; exit 6; fi\n'
            'echo "[N/A]"\n'
        )
        self._with_path(self.bin_dir + os.pathsep + os.environ["PATH"])
        sampler = bench_metrics.GpuMemorySampler(device_index=0, interval_s=0.01)
        sampler.start()
        self._wait_for_samples(2)
        self.assertIsNone(sampler.stop())

    def test_stop_without_start(self):
        self.assertIsNone(bench_metrics.GpuMemorySampler(device_index=0).stop())


if __name__ == "__main__":
    unittest.main()
