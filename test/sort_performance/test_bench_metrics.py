# =============================================================================
# Copyright 2026, Sirius Contributors.
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

Run: pixi run python -B test/sort_performance/test_bench_metrics.py   (also collectable by pytest)
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
            {
                "rows": 1500300,
                "bytes": 12002400,
                "order_violations": 0,
                "order_checked": 1,
                "ranges_checked": 12,
            },
        )

    def test_line_without_the_trailing_keys_still_parses(self):
        text = "[info] [discard_sink] rows=10 bytes=80 order_violations=0\n"
        self.assertEqual(
            bench_metrics.parse_discard_sink(text),
            {"rows": 10, "bytes": 80, "order_violations": 0},
        )

    def test_unknown_trailing_key_is_kept(self):
        text = "[discard_sink] rows=1 bytes=8 order_violations=0 future_key=5\n"
        self.assertEqual(bench_metrics.parse_discard_sink(text)["future_key"], 5)

    def test_missing_line(self):
        self.assertIsNone(
            bench_metrics.parse_discard_sink(read_fixture("downgrade.log"))
        )

    def test_malformed_line_ignored(self):
        self.assertIsNone(
            bench_metrics.parse_discard_sink("[discard_sink] rows=1 bytes=many\n")
        )
        self.assertIsNone(bench_metrics.parse_discard_sink("[discard_sink] rows=1\n"))
        self.assertIsNone(
            bench_metrics.parse_discard_sink(
                "[discard_sink] rows=1 bytes=8 order_checked=1 ranges_checked=2\n"
            )
        )

    def test_several_sinks_are_summed(self):
        text = (
            "[info] [discard_sink] rows=10 bytes=80 order_violations=1\n"
            "[info] [discard_sink] rows=5 bytes=40 order_violations=2\n"
        )
        self.assertEqual(
            bench_metrics.parse_discard_sink(text),
            {"rows": 15, "bytes": 120, "order_violations": 3},
        )

    def test_several_sinks_order_checked_only_if_all_checked(self):
        checked = "[discard_sink] rows=10 bytes=80 order_violations=0 order_checked=1 ranges_checked=4\n"
        unchecked = "[discard_sink] rows=5 bytes=40 order_violations=0 order_checked=0 ranges_checked=0\n"
        older = "[discard_sink] rows=5 bytes=40 order_violations=0\n"
        self.assertEqual(
            bench_metrics.parse_discard_sink(checked + checked),
            {
                "rows": 20,
                "bytes": 160,
                "order_violations": 0,
                "order_checked": 1,
                "ranges_checked": 8,
            },
        )
        for text in (checked + unchecked, unchecked + checked, older + checked):
            totals = bench_metrics.parse_discard_sink(text)
            self.assertEqual(totals["order_checked"], 0, text)
            self.assertEqual(totals["ranges_checked"], 4, text)
            self.assertEqual(totals["rows"], 15, text)


class FirstProblemLinesTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)

    def write(self, name: str, text: str) -> str:
        path = os.path.join(self._dir.name, name)
        with open(path, "w") as log_file:
            log_file.write(text)
        return path

    def test_keeps_warning_and_above_in_order(self):
        lines = bench_metrics.first_problem_lines([str(FIXTURES / "oom_retries.log")])
        fixture = read_fixture("oom_retries.log").splitlines()
        self.assertEqual(
            lines, [l for l in fixture if "] [warning] " in l or "] [error] " in l]
        )
        self.assertTrue(any("] [error] " in l for l in lines))
        # INFO and DEBUG lines are not problems.
        self.assertEqual(
            bench_metrics.first_problem_lines([str(FIXTURES / "downgrade.log")]), []
        )

    def test_level_is_read_from_the_prefix_only(self):
        path = self.write(
            "a.log",
            "[2026-11-13 03:46:00.079] [info] [x.cpp:1] the text says [error] here\n"
            "[2026-11-13 03:46:00.080] [critical] [x.cpp:2] stop\n"
            "=== Pipeline Overview === [warning]\n"
            "[2026-11-13 03:46:00.081] [error] [x.cpp:3] CUDF failure\n",
        )
        self.assertEqual(
            bench_metrics.first_problem_lines([path]),
            [
                "[2026-11-13 03:46:00.080] [critical] [x.cpp:2] stop",
                "[2026-11-13 03:46:00.081] [error] [x.cpp:3] CUDF failure",
            ],
        )

    def test_bounded_and_files_in_the_given_order(self):
        first = self.write(
            "a.log",
            "".join(f"[t] [error] [x.cpp:1] failure {'a' * i}\n" for i in range(150)),
        )
        second = self.write(
            "b.log",
            "".join(f"[t] [warning] [x.cpp:1] retry {'b' * i}\n" for i in range(150)),
        )
        lines = bench_metrics.first_problem_lines([first, second], max_lines=200)
        self.assertEqual(len(lines), 200)
        self.assertEqual(lines[0], "[t] [error] [x.cpp:1] failure ")
        self.assertEqual(lines[150], "[t] [warning] [x.cpp:1] retry ")
        self.assertEqual(len(bench_metrics.first_problem_lines([first], 7)), 7)

    def test_a_repeated_message_cannot_fill_the_result(self):
        # The same warning for every batch, as the discard sink logs it, then the real error.
        noise = "".join(
            f"[2026-11-13 03:38:00.{i:03d}] [warning] [gpu_pipeline_task.cpp:83] operator "
            f"'DISCARD_SINK' (id=5) output batch {i} column count mismatch: got 1, expected 2\n"
            for i in range(900)
        )
        path = self.write("a.log", noise + "[t] [error] [x.cpp:3] CUDF failure\n")
        lines = bench_metrics.first_problem_lines([path], max_lines=200)
        self.assertEqual(len(lines), 6)
        self.assertIn("output batch 4 column", lines[4])
        self.assertTrue(lines[5].endswith("CUDF failure"))
        self.assertEqual(
            len(bench_metrics.first_problem_lines([path], max_per_message=2)), 3
        )

    def test_long_lines_are_cut_and_missing_files_skipped(self):
        path = self.write("a.log", "[t] [error] [x.cpp:1] " + "x" * 5000 + "\n")
        missing = os.path.join(self._dir.name, "missing.log")
        lines = bench_metrics.first_problem_lines([missing, path], max_line_chars=100)
        self.assertEqual([len(l) for l in lines], [100])
        self.assertEqual(bench_metrics.first_problem_lines([]), [])


class GpuMemorySamplerTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.bin_dir = os.path.join(self._dir.name, "bin")
        os.mkdir(self.bin_dir)
        self.args_file = os.path.join(self._dir.name, "args")
        self.count_file = os.path.join(self._dir.name, "count")
        self.launch_file = os.path.join(self._dir.name, "launches")
        self.pid_file = os.path.join(self._dir.name, "pid")

    def _install_fake_nvidia_smi(self, body: str) -> None:
        # Records every launch and its pid, then runs `body`.
        path = os.path.join(self.bin_dir, "nvidia-smi")
        with open(path, "w") as script:
            script.write(
                "#!/bin/sh\n"
                f'printf x >> "{self.launch_file}"\n'
                f'echo $$ > "{self.pid_file}"\n' + body
            )
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
        self.fail(f"the fake nvidia-smi did not print {count} samples")

    def _launches(self) -> int:
        with open(self.launch_file) as launch_file:
            return len(launch_file.read())

    def test_peak_of_the_samples_from_one_process(self):
        # One process printing 100, 5000, 300, then 200 until it is stopped: the peak is in the
        # middle.
        self._install_fake_nvidia_smi(
            f'echo "$*" > "{self.args_file}"\n'
            "for v in 100 5000 300; do\n"
            f'  echo "$v"; printf x >> "{self.count_file}"\n'
            "done\n"
            "while :; do\n"
            f'  echo 200; printf x >> "{self.count_file}"; sleep 0.01\n'
            "done\n"
        )
        self._with_path(self.bin_dir + os.pathsep + os.environ["PATH"])
        sampler = bench_metrics.GpuMemorySampler(device_index=1, interval_s=0.01)
        sampler.start()
        sampler.start()  # a second start does not launch a second process
        self._wait_for_samples(6)
        self.assertEqual(sampler.stop(), 5000)
        self.assertEqual(self._launches(), 1)
        with open(self.args_file) as args_file:
            self.assertEqual(
                args_file.read().strip(),
                "--query-gpu=memory.used --format=csv,noheader,nounits -i 1 --loop-ms=10",
            )

    def test_default_interval_is_one_second(self):
        self._install_fake_nvidia_smi(
            f'echo "$*" > "{self.args_file}"\necho 7\nexec sleep 60\n'
        )
        self._with_path(self.bin_dir + os.pathsep + os.environ["PATH"])
        sampler = bench_metrics.GpuMemorySampler(device_index=0)
        sampler.start()
        self.assertEqual(sampler.stop(), 7)
        with open(self.args_file) as args_file:
            self.assertTrue(args_file.read().strip().endswith(" --loop-ms=1000"))

    def test_short_run_still_gets_a_sample_and_stop_ends_the_process(self):
        self._install_fake_nvidia_smi('sleep 0.2\necho " 4321"\nexec sleep 60\n')
        self._with_path(self.bin_dir + os.pathsep + os.environ["PATH"])
        sampler = bench_metrics.GpuMemorySampler(device_index=0, interval_s=60)
        sampler.start()
        started = time.monotonic()
        self.assertEqual(sampler.stop(), 4321)
        self.assertLess(time.monotonic() - started, 30)
        with open(self.pid_file) as pid_file:
            pid = int(pid_file.read())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertEqual(sampler.stop(), 4321)  # a second stop changes nothing

    def test_restart_forgets_the_previous_peak(self):
        self._install_fake_nvidia_smi(
            f'n=$(wc -c < "{self.launch_file}")\n'
            'if [ "$n" -eq 1 ]; then echo 900; else echo 50; fi\n'
            "exec sleep 60\n"
        )
        self._with_path(self.bin_dir + os.pathsep + os.environ["PATH"])
        sampler = bench_metrics.GpuMemorySampler(device_index=0)
        sampler.start()
        self.assertEqual(sampler.stop(), 900)
        sampler.start()
        self.assertEqual(sampler.stop(), 50)
        self.assertEqual(self._launches(), 2)

    def test_nvidia_smi_unavailable(self):
        self._with_path(self.bin_dir)  # empty directory: no nvidia-smi anywhere on PATH
        sampler = bench_metrics.GpuMemorySampler(device_index=0, interval_s=0.01)
        sampler.start()
        self.assertIsNone(sampler.stop())

    def test_nvidia_smi_failing(self):
        self._install_fake_nvidia_smi('echo "No devices were found"\nexit 6\n')
        self._with_path(self.bin_dir + os.pathsep + os.environ["PATH"])
        sampler = bench_metrics.GpuMemorySampler(device_index=0, interval_s=0.01)
        sampler.start()
        self.assertIsNone(sampler.stop())
        self.assertEqual(self._launches(), 1)

    def test_nvidia_smi_printing_garbage(self):
        self._install_fake_nvidia_smi(
            "while :; do\n"
            f'  echo "[N/A]"; printf x >> "{self.count_file}"; sleep 0.01\n'
            "done\n"
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
