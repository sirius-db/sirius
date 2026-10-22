"""CPU-only tests for operator attribution, provenance, and missing evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import operator_profiles as profiles

PREFIX = "[2026-10-22 12:00:00.000]"


def info(message):
    return f"{PREFIX} [info] [sirius_context.cpp:100] {message}"


def trace(message):
    return f"{PREFIX} [trace] [gpu_pipeline_task.cpp:100] {message}"


def produced(operator, duration, operator_id=1):
    return trace(
        f"[GPU:0] Pipeline 1: operator {operator} (id={operator_id}) task=2 "
        "produced 1 batches, num rows: 10  , size: 80 bytes (0.00 MB). "
        f"execution time: {duration:.2f} ms, peak allocated: 1024 bytes (0.00 MB)"
    )


def gpu_log(sample_id="gpu_projection_0", duration=2.25):
    return [
        info(
            f"QueryBegin: instance=0x1 connection=1 query=3 SQL: SELECT f(value) FROM bench_input /*sirius_stage:{sample_id}*/"
        ),
        info("[window] begin instance=0x1 connection=1 window=1 query=3 outcome=-"),
        info(
            "[gpu_pool] GPU:0 QueryBegin instance=0x1 connection=1 window=1 query=3 allocated=1024 bytes peak=4096 bytes reserved=8192 bytes"
        ),
        trace("Pipeline 1: operator TABLE_SCAN (id=1) prepare execution time: 1.50 ms"),
        produced("TABLE_SCAN", 0.50),
        produced("PROJECTION", duration, 2),
        # An observer SQL line is not a GPU-window boundary.
        info(
            "QueryBegin: instance=0x1 connection=2 query=1 SQL: SELECT * FROM sirius_execution_stats()"
        ),
        trace(
            "Pipeline 1: operator RESULT_COLLECTOR (id=3) sink execution time: 0.25 ms"
        ),
        info(
            "[gpu_pool] GPU:0 QueryEnd instance=0x1 connection=1 window=1 query=3 allocated=1024 bytes peak=8192 bytes reserved=0 bytes"
        ),
        info("[window] end instance=0x1 connection=1 window=1 query=3 outcome=ok"),
    ]


class OperatorProfilesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="operator-profiles-test-")
        self.addCleanup(self.temporary.cleanup)
        self.job = Path(self.temporary.name)
        (self.job / "engine-log").mkdir()

    def entry(self, engine="gpu", sample_id=None):
        sample_id = sample_id or f"{engine}_projection_0"
        return {
            "sample_id": sample_id,
            "engine": engine,
            "cpu_mode": engine,
            "cpu_baseline": engine,
            "query_kind": "projection",
            "warmup": False,
            "iteration": 0,
            "query_wall_ms": 1.0,
            "profile_file": f"cpu-profile-{sample_id}.json" if engine == "cpu" else "",
            "sql": f"SELECT f(value) FROM bench_input /*sirius_stage:{sample_id}*/",
            **(
                {
                    "gpu_counters": {
                        "executions": 1,
                        "plan_fallbacks": 0,
                        "runtime_fallbacks": 0,
                    }
                }
                if engine == "gpu"
                else {}
            ),
        }

    def write_log(self, lines):
        (self.job / "engine-log" / "sirius.log").write_text("\n".join(lines) + "\n")

    def collect_gpu(self, lines=None, entry=None):
        self.write_log(gpu_log() if lines is None else lines)
        return profiles.collect(self.job, {"profiles": [entry or self.entry()]}, 8)[0]

    def collect_cpu(self, document):
        entry = self.entry("cpu")
        (self.job / entry["profile_file"]).write_text(json.dumps(document))
        return profiles.collect(self.job, {"profiles": [entry]}, 8)[0]

    def test_cpu_modern_uses_exclusive_timing_not_cumulative_or_latency(self):
        result = self.collect_cpu(
            {
                "query": {"total_time": 80, "cpu_time": 999},
                "operator": [
                    {
                        "type": "PROJECTION",
                        "timing": 0.002,
                        "children": [{"type": "TABLE_SCAN", "timing": 0.001}],
                    }
                ],
            }
        )
        self.assertEqual(result["stages"]["projection_expressions"], 2)
        self.assertEqual(result["stages"]["input_prepare_scan"], 1)
        self.assertEqual(result["total_operator_work_ms"], 3)
        self.assertFalse(result["stage_available"]["output_transfer_materialization"])
        self.assertTrue(result["timings_not_additive_to_query_wall"])
        self.assertGreater(result["total_operator_work_ms"], result["query_wall_ms"])

    def test_cpu_legacy_tree_does_not_double_count_cpu_time(self):
        result = self.collect_cpu(
            {
                "latency": 2,
                "cpu_time": 90,
                "children": [
                    {
                        "operator_type": "UNGROUPED_AGGREGATE",
                        "operator_timing": 0.0005,
                        "cpu_time": 30,
                        "children": [
                            {
                                "operator_type": "PROJECTION",
                                "operator_timing": 0.0002,
                                "cpu_time": 20,
                                "children": [
                                    {
                                        "operator_type": "TABLE_SCAN",
                                        "operator_timing": 0.0001,
                                        "cpu_time": 10,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )
        self.assertAlmostEqual(result["total_operator_work_ms"], 0.8)
        self.assertEqual(result["stages"]["aggregation"], 0.5)

    def test_cpu_metrics_wrapped_operator_tree(self):
        result = self.collect_cpu(
            {"metrics": {"operator": [{"type": "PROJECTION", "timing": 0.003}]}}
        )
        self.assertEqual(result["stages"]["projection_expressions"], 3)

    def test_cpu_operator_metrics_wrapped_per_node(self):
        result = self.collect_cpu(
            {
                "operator": [
                    {
                        "metrics": {"type": "PROJECTION", "timing": 0.003},
                        "children": [
                            {"metrics": {"type": "TABLE_SCAN", "timing": 0.001}}
                        ],
                    }
                ]
            }
        )
        self.assertEqual(result["total_operator_work_ms"], 4)

    def test_cpu_missing_unknown_or_invalid_profile_fails(self):
        for document in (
            {},
            {"latency": 1, "cpu_time": 99},
            {"operator": [{"type": "TABLE_SCAN", "timing": 1}]},
            {"operator": [{"type": "PROJECTION"}]},
            {"operator": [{"type": "PROJECTION", "timing": -1}]},
            {"operator": [{"type": "PROJECTION", "timing": float("nan")}]},
            {"operator": [{"type": "PROJECTION", "timing": True}]},
        ):
            with self.subTest(document=document), self.assertRaises(ValueError):
                self.collect_cpu(document)

    def test_cpu_missing_file_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing CPU profile"):
            profiles.collect(self.job, {"profiles": [self.entry("cpu")]}, 8)

    def test_cpu_profile_sql_cannot_belong_to_another_sample(self):
        for key in ("query", "query_name"):
            document = {
                "operator": [{"type": "PROJECTION", "timing": 0.003}],
                key: (
                    {"sql": "SELECT 1 /*sirius_stage:another_sample*/"}
                    if key == "query"
                    else "SELECT 1 /*sirius_stage:another_sample*/"
                ),
            }
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "does not match"
            ):
                self.collect_cpu(document)

    def test_cpu_profile_path_must_stay_inside_job(self):
        entry = self.entry("cpu")
        entry["profile_file"] = "../outside.json"
        with self.assertRaisesRegex(ValueError, "escapes"):
            profiles.collect(self.job, {"profiles": [entry]}, 8)

    def test_gpu_measured_stage_durations_not_inferred_timestamps(self):
        result = self.collect_gpu()
        self.assertEqual(result["stages"]["input_prepare_scan"], 2)
        self.assertEqual(result["stages"]["projection_expressions"], 2.25)
        self.assertEqual(result["stages"]["output_transfer_materialization"], 0.25)
        self.assertEqual(result["total_operator_work_ms"], 4.5)
        self.assertEqual(result["precision_resolution_ms"], 0.01)
        self.assertEqual(
            result["memory_observations"]["pool_boundaries"][1]["peak_bytes"], 8192
        )
        self.assertIn("not per-query", result["memory_observations"]["pool_peak_scope"])
        self.assertFalse(result["memory_observations"]["pressure_observed"])

    def test_gpu_zero_rounded_duration_is_available(self):
        result = self.collect_gpu(gpu_log(duration=0))
        self.assertEqual(result["stages"]["projection_expressions"], 0)
        self.assertTrue(result["stage_available"]["projection_expressions"])

    def test_gpu_multiple_worker_tasks_are_accumulated(self):
        lines = gpu_log()
        lines.insert(6, produced("PROJECTION", 3.25, 2))
        result = self.collect_gpu(lines)
        self.assertEqual(result["stages"]["projection_expressions"], 5.5)
        self.assertGreater(result["total_operator_work_ms"], result["query_wall_ms"])

    def test_gpu_aggregation_execute_and_sink_both_preserved(self):
        lines = gpu_log()
        lines.insert(6, produced("UNGROUPED_AGGREGATE", 0.50, 4))
        lines.insert(
            7,
            trace(
                "Pipeline 1: operator UNGROUPED_AGGREGATE (id=4) sink execution time: 0.05 ms"
            ),
        )
        result = self.collect_gpu(lines)
        self.assertEqual(result["stages"]["aggregation"], 0.55)

    def test_gpu_missing_tag_window_log_or_projection_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing Sirius TRACE"):
            profiles.collect(self.job, {"profiles": [self.entry()]}, 8)
        cases = [
            [],
            [
                line.replace("sirius_stage:gpu_projection_0", "sirius_stage:other")
                for line in gpu_log()
            ],
            gpu_log()[:-1],
            [line.replace("outcome=ok", "outcome=unwind") for line in gpu_log()],
            [line for line in gpu_log() if "operator PROJECTION" not in line],
            [line for line in gpu_log() if "sink execution time:" not in line],
        ]
        for lines in cases:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                self.collect_gpu(lines)

    def test_gpu_malformed_counters_fail(self):
        for counters in (
            None,
            {},
            {"executions": True, "plan_fallbacks": 0, "runtime_fallbacks": 0},
            {"executions": 0, "plan_fallbacks": 0, "runtime_fallbacks": 0},
            {"executions": 1, "plan_fallbacks": 1, "runtime_fallbacks": 0},
            {"executions": 1, "plan_fallbacks": 0, "runtime_fallbacks": 1},
        ):
            entry = self.entry()
            entry["gpu_counters"] = counters
            with self.subTest(counters=counters), self.assertRaisesRegex(
                ValueError, "prove one execution"
            ):
                self.collect_gpu(entry=entry)

    def test_gpu_observer_sql_does_not_close_gpu_window(self):
        result = self.collect_gpu()
        self.assertEqual(len(result["raw_operators"]), 4)
        self.assertEqual(result["stages"]["output_transfer_materialization"], 0.25)

    def test_gpu_duplicate_or_overlapping_windows_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate GPU execution window"):
            self.collect_gpu(gpu_log() + gpu_log())
        lines = gpu_log()
        lines.insert(
            6,
            info("[window] begin instance=0x1 connection=3 window=2 query=4 outcome=-"),
        )
        with self.assertRaisesRegex(ValueError, "Overlapping GPU windows"):
            self.collect_gpu(lines)

    def test_gpu_trace_format_drift_fails(self):
        lines = [line.replace("num rows: 10", "changed rows: 10") for line in gpu_log()]
        with self.assertRaisesRegex(ValueError, "format drift"):
            self.collect_gpu(lines)

    def test_gpu_memory_pressure_is_reported_without_claiming_query_peak(self):
        lines = gpu_log()
        lines[5:5] = [
            f"{PREFIX} [error] [gpu_pipeline_task.cpp:528] Pipeline 1: OOM at operator PROJECTION (id=2, index 1/2), rescheduling task 2",
            f"{PREFIX} [warn] [gpu_pipeline_executor.cpp:486] GPU Pipeline Executor: reschedule (retry 1/32) for task 2 (original task 2), resuming from operator index 1: OOM",
            f"{PREFIX} [debug] [downgrade_executor.cpp:348] [downgrade] [GPU:0] request monitor done: 3 batches, 2112 bytes in 6.04 ms (0.3 MB/s) | repos: 3/2112 batches/bytes, pipeline_queue: 0/0 | to_host: 3/2112 batches/bytes, to_disk: 0/0 batches/bytes",
        ]
        observations = self.collect_gpu(lines)["memory_observations"]
        self.assertEqual(observations["oom_event_count"], 1)
        self.assertEqual(observations["reschedule_count"], 1)
        self.assertEqual(observations["downgrade_count"], 1)
        self.assertEqual(observations["downgrade_bytes"], 2112)
        self.assertTrue(observations["pressure_observed"])

    def test_manifest_rejects_duplicate_ids_or_mismatched_tag(self):
        entry = self.entry("cpu")
        for manifest in (
            {"profiles": [entry, copy.deepcopy(entry)]},
            {"profiles": [{**entry, "sql": "SELECT 1"}]},
            {"profiles": []},
        ):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                profiles.collect(self.job, manifest, 8)


if __name__ == "__main__":
    unittest.main()
