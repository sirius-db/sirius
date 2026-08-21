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

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import duckdb  # noqa: F401
except ModuleNotFoundError:
    # These focused tests exercise pure parsing/serialization helpers without requiring DuckDB.
    sys.modules["duckdb"] = types.ModuleType("duckdb")
import performance_test


class _FakeResult:
    def __init__(self, columns, rows):
        self.description = [(column,) for column in columns]
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, columns, rows):
        self._result = _FakeResult(columns, rows)
        self.commands = []

    def execute(self, sql):
        self.commands.append(sql)
        return self._result


class DynamicFilterObservabilityTest(unittest.TestCase):
    def _write_iteration(self, before, after):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "observability.jsonl")
        performance_test._write_dynamic_filter_iteration_observability(
            path, 8, 5, before, after
        )
        return path

    def test_sql_snapshot_discovers_stats_and_global_event(self):
        columns = [
            "record_type",
            "event_id",
            "event_high_watermark",
            "join_operator_id",
            "exact_contribution_count",
            "device_id",
            "build_rows",
            "filters_built",
            "active_targets",
            "filters_pushed",
            "stats_publications_finished",
        ]
        rows = [
            ("stats", None, 1, None, None, None, None, None, None, None, 4),
            ("global_accumulator_completion", 1, None, 9, 3, 1, 6, 1, 1, 1, None),
        ]
        connection = _FakeConnection(columns, rows)

        snapshot = performance_test._read_dynamic_filter_observability(connection)

        self.assertEqual(connection.commands[0], "SET gpu_execution = false;")
        self.assertIn("sirius_dynamic_filter_observability", connection.commands[1])
        self.assertEqual(snapshot["event_high_watermark"], 1)
        self.assertEqual(snapshot["stats"], {"publications_finished": 4})
        self.assertEqual(
            snapshot["events"],
            [
                {
                    "event_id": 1,
                    "record_type": "global_accumulator_completion",
                    "join_operator_id": 9,
                    "exact_contribution_count": 3,
                    "device_id": 1,
                    "build_rows": 6,
                    "filters_built": 1,
                    "active_targets": 1,
                    "filters_pushed": 1,
                }
            ],
        )

    def test_sql_snapshot_rejects_noncontiguous_journal(self):
        columns = [
            "record_type",
            "event_id",
            "event_high_watermark",
            "join_operator_id",
            "exact_contribution_count",
            "device_id",
            "build_rows",
            "filters_built",
            "active_targets",
            "filters_pushed",
            "stats_publications_finished",
        ]
        rows = [
            ("stats", None, 2, None, None, None, None, None, None, None, 1),
            ("global_accumulator_completion", 2, None, 9, 2, 0, 6, 1, 1, 1, None),
        ]
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            performance_test._read_dynamic_filter_observability(
                _FakeConnection(columns, rows)
            )

    def test_iteration_record_subtracts_all_fields_and_selects_new_events(self):
        before = {
            "event_high_watermark": 2,
            "stats": {"a": 10, "b": 20},
            "events": [{"event_id": 1}, {"event_id": 2}],
        }
        after = {
            "event_high_watermark": 4,
            "stats": {"a": 13, "b": 20},
            "events": [
                {"event_id": 1},
                {"event_id": 2},
                {"event_id": 3, "record_type": "global_accumulator_completion"},
                {"event_id": 4, "record_type": "global_accumulator_completion"},
            ],
        }
        path = self._write_iteration(before, after)
        with open(path) as source:
            record = json.loads(source.readline())

        self.assertEqual(record["query"], "q8")
        self.assertEqual(record["iteration"], 5)
        self.assertEqual(record["stats_delta"], {"a": 3, "b": 0})
        self.assertEqual([event["event_id"] for event in record["events"]], [3, 4])

    def test_iteration_record_rejects_counter_decrease(self):
        before = {"event_high_watermark": 0, "stats": {"a": 2}, "events": []}
        after = {"event_high_watermark": 0, "stats": {"a": 1}, "events": []}
        with self.assertRaisesRegex(RuntimeError, "counter 'a' decreased"):
            self._write_iteration(before, after)

    def test_iteration_record_rejects_missing_new_event(self):
        before = {"event_high_watermark": 2, "stats": {"a": 1}, "events": []}
        after = {
            "event_high_watermark": 4,
            "stats": {"a": 1},
            "events": [{"event_id": 1}, {"event_id": 2}, {"event_id": 3}],
        }
        with self.assertRaisesRegex(RuntimeError, "expected IDs.*3, 4.*got.*3"):
            self._write_iteration(before, after)

    def test_skip_cache_drop_is_recorded_in_metadata(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        benchmark_dir, _, _ = performance_test.setup_benchmark_dir(
            directory.name,
            "grouped",
            1,
            "gpu",
            [7],
            "",
            "none",
            name="cache-policy",
            dynamic_filter_observability=True,
            skip_os_cache_drop=True,
        )
        with open(os.path.join(benchmark_dir, "metadata.json")) as source:
            metadata = json.load(source)
        self.assertEqual(metadata["os_cache_policy"], "skip")
        self.assertTrue(metadata["dynamic_filter_observability"])
        self.assertEqual(
            metadata["dynamic_filter_observability_file"],
            os.path.join("csv", "dynamic_filter_observability.jsonl"),
        )

    def test_command_line_flags_are_parsed(self):
        argv = [
            "performance_test.py",
            "--input",
            "/tmp/tpch",
            "--dynamic-filter-observability",
            "--skip-os-cache-drop",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = performance_test.parse_args()

        self.assertTrue(args.dynamic_filter_observability)
        self.assertTrue(args.skip_os_cache_drop)

    def test_gpu_timing_restores_execution_after_cpu_snapshot(self):
        connection = _FakeConnection(["value"], [(1,)])
        original_query = performance_test.QUERIES.get("q99")
        performance_test.QUERIES["q99"] = "SELECT 99"

        def restore_query():
            if original_query is None:
                performance_test.QUERIES.pop("q99", None)
            else:
                performance_test.QUERIES["q99"] = original_query

        self.addCleanup(restore_query)

        _, rows = performance_test.time_query(connection, 99, use_gpu=True)

        self.assertEqual(
            connection.commands, ["SET gpu_execution = true;", "SELECT 99"]
        )
        self.assertEqual(rows, [(1,)])


if __name__ == "__main__":
    unittest.main()
