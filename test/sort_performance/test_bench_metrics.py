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

"""Tests of the log parsers of bench_metrics.py. No GPU needed.

Run: pixi run python -B test/sort_performance/test_bench_metrics.py   (also collectable by pytest)
"""

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
