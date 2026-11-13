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

"""Drift guard for the harness: reads the `[merge_sort][stats]` format string from the C++
sources and fails when its tag or its counters no longer match what the harness expects.

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

    def test_harness_columns_are_the_cpp_counters(self):
        # MERGE_STATS_KEYS fixes the CSV column order and the markdown columns: a counter renamed
        # in C++ would otherwise leave an empty column.
        keys = re.findall(r"([A-Za-z_]+)=\{\}", self.fmt)
        self.assertEqual(keys[1:], sort_merge_bench.MERGE_STATS_KEYS)


if __name__ == "__main__":
    unittest.main()
