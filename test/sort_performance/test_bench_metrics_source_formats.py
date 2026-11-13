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

"""Drift guard for bench_metrics.py: renders the format strings found in the C++ sources and
checks that the parsers read the result. The fixtures in fixtures/ are hand-written; these tests
fail when a C++ log line changes and the parser no longer matches it.

Run: pixi run python -B test/sort_performance/test_bench_metrics_source_formats.py
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_metrics  # noqa: E402
import sort_merge_bench  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SPDLOG_PREFIX = "[2026-11-12 10:15:02.310] [info] [140213] "


def _concatenated_literal(source: str, start: int) -> str:
    """Joins the adjacent C++ string literals that begin at or after `start`."""
    literals = re.compile(r'\s*"((?:[^"\\]|\\.)*)"')
    pos = source.index('"', start)
    parts = []
    while True:
        match = literals.match(source, pos)
        if match is None:
            break
        parts.append(match.group(1))
        pos = match.end()
    return "".join(parts)


def _read(relative: str) -> str:
    return (REPO / relative).read_text()


class MergeSortStatsFormatTest(unittest.TestCase):
    def setUp(self):
        source = _read("src/op/merge/merge_sort_stats_log.cpp")
        self.fmt = _concatenated_literal(source, source.index("std::format("))
        header = _read("src/op/merge/merge_sort_stats_log.hpp")
        self.tag = re.search(r'kMergeSortStatsTag\s*=\s*"([^"]*)"', header).group(1)

    def test_tag_matches_cpp(self):
        self.assertEqual(self.tag, bench_metrics.MERGE_SORT_STATS_TAG)

    def test_rendered_line_parses_every_counter(self):
        keys = re.findall(r"([A-Za-z_]+)=\{\}", self.fmt)
        self.assertEqual(keys[0], "op")
        self.assertGreaterEqual(len(keys), 20)
        values = list(range(100, 100 + len(keys)))
        line = self.fmt.format(self.tag, *values)
        stats = bench_metrics.parse_merge_sort_stats(SPDLOG_PREFIX + line + "\n")
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].op, 100)
        self.assertEqual(stats[0].values, dict(zip(keys[1:], values[1:])))

    def test_harness_columns_are_the_cpp_counters(self):
        # MERGE_STATS_KEYS fixes the CSV column order and the markdown columns: a counter renamed
        # in C++ would otherwise leave an empty column.
        keys = re.findall(r"([A-Za-z_]+)=\{\}", self.fmt)
        self.assertEqual(keys[1:], sort_merge_bench.MERGE_STATS_KEYS)

    def test_sum_of_two_rendered_lines(self):
        keys = re.findall(r"([A-Za-z_]+)=\{\}", self.fmt)
        first = {key: 2 for key in keys}
        second = {key: 3 for key in keys}
        first["round_rows_min"], second["round_rows_min"] = 0, 7
        first["round_rows_max"], second["round_rows_max"] = 90, 40
        text = "".join(
            SPDLOG_PREFIX + self.fmt.format(self.tag, *row.values()) + "\n"
            for row in (first, second)
        )
        totals = bench_metrics.sum_merge_sort_stats(
            bench_metrics.parse_merge_sort_stats(text)
        )
        self.assertEqual(totals["round_rows_min"], 7)
        self.assertEqual(totals["round_rows_max"], 90)
        self.assertEqual(totals["rows_out"], 5)
        self.assertNotIn("op", totals)


class DowngradeFormatTest(unittest.TestCase):
    def test_rendered_line_parses(self):
        source = _read("src/downgrade/downgrade_executor.cpp")
        start = source.index('"[downgrade][monitor] [{}] request done')
        fmt = _concatenated_literal(source, start)
        # spdlog/std::format prints a bool as true/false; Python would print True/False.
        fmt = fmt.replace("satisfied={}", "satisfied=true")
        rendered = [
            SPDLOG_PREFIX + fmt.format("gpu0", *args) + "\n"
            for args in (
                (4, 4096, 1.25, 3.3, 1, 1024, 3, 3072),
                (2, 50, 0.5, 0.1, 0, 0, 2, 50),
            )
        ]
        totals = bench_metrics.parse_downgrade_totals("".join(rendered))
        self.assertEqual(
            totals,
            bench_metrics.DowngradeTotals(
                requests=2,
                batches=6,
                bytes=4146,
                to_host_batches=1,
                to_host_bytes=1024,
                to_disk_batches=5,
                to_disk_bytes=3122,
            ),
        )


class OomRetryFormatTest(unittest.TestCase):
    def setUp(self):
        source = _read("src/pipeline/gpu_pipeline_executor.cpp")
        self.fmt = _concatenated_literal(
            source, source.index('"GPU Pipeline Executor: reschedule (retry')
        )

    def _line(self, reason: str) -> str:
        return SPDLOG_PREFIX + self.fmt.format(1, 8, 42, 41, 2, reason) + "\n"

    def test_every_oom_reschedule_reason_counts(self):
        task_source = _read("src/pipeline/gpu_pipeline_task.cpp")
        reasons = []
        for match in re.finditer(r"throw oom_reschedule_exception\(", task_source):
            window = task_source[match.end() : match.end() + 400]
            literal = re.search(r'"((?:[^"\\]|\\.)*)"', window)
            self.assertIsNotNone(literal, "oom_reschedule_exception without a message")
            # The literal alone: one of these messages is logged with nothing appended.
            reasons.append(literal.group(1))
        self.assertGreater(len(reasons), 0)
        text = "".join(self._line(reason) for reason in reasons)
        self.assertEqual(bench_metrics.count_oom_retries(text), len(reasons))

    def test_non_oom_reason_does_not_count(self):
        text = self._line("cudaErrorLaunchFailure at operator ORDER_BY (index 1)")
        self.assertEqual(bench_metrics.count_oom_retries(text), 0)


class DiscardSinkFormatTest(unittest.TestCase):
    def setUp(self):
        source = _read("src/op/sirius_physical_discard_sink.cpp")
        self.fmt = _concatenated_literal(source, source.index('"[discard_sink]'))
        self.keys = re.findall(r"([A-Za-z_]+)=\{\}", self.fmt)

    def test_tag_matches_cpp(self):
        self.assertTrue(self.fmt.startswith(bench_metrics.DISCARD_SINK_TAG + " "))

    def test_rendered_line_parses_every_key(self):
        self.assertEqual(self.keys[:3], ["rows", "bytes", "order_violations"])
        values = list(range(10, 10 + len(self.keys)))
        line = self.fmt.format(*values)
        self.assertEqual(
            bench_metrics.parse_discard_sink(SPDLOG_PREFIX + line + "\n"),
            dict(zip(self.keys, values)),
        )

    def test_fixture_has_the_keys_the_cpp_logs(self):
        fixture = (
            Path(__file__).resolve().parent / "fixtures" / "discard_sink.log"
        ).read_text()
        parsed = bench_metrics.parse_discard_sink(fixture)
        self.assertEqual(list(parsed), self.keys)


if __name__ == "__main__":
    unittest.main()
