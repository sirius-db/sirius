"""CPU-only checks for stage profiling configuration and evidence gates."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_stages


class StageRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sirius-stage-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / "build with spaces"
        self.build.mkdir()
        host = self.build / "bench_host"
        host.write_text("#!/bin/sh\nexit 7\n")
        host.chmod(0o755)
        for module in ("cudf", "cuda", "q1"):
            (self.build / f"benchmark_{module}.so").touch()
        self.output = self.root / "result with spaces"
        self.argv = [
            "--build-dir",
            str(self.build),
            "--output-dir",
            str(self.output),
            "--workloads",
            "sql",
            "--sizes",
            "64",
            "--warmups",
            "1",
            "--repetitions",
            "5",
        ]
        self.args = run_stages.parse_args(self.argv)

    def invalid_args(self, extra):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                run_stages.parse_args(self.argv + extra)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def manifest(self):
        profiles = []
        for mode, label, engine in (
            ("native", "Vectorized CPU SQL", "cpu"),
            ("gpu", "GPU", "gpu"),
        ):
            for kind in ("aggregate", "projection"):
                for iteration in range(6):
                    sample = {
                        "sample_id": f"{mode}_{kind}_{iteration}",
                        "engine": engine,
                        "cpu_mode": mode,
                        "cpu_baseline": label,
                        "query_kind": kind,
                        "iteration": iteration,
                        "warmup": iteration == 0,
                        "query_wall_ms": 1.25,
                        "sql": "SELECT value FROM bench_input",
                    }
                    if engine == "gpu":
                        sample["gpu_counters"] = {
                            "executions": 1,
                            "plan_fallbacks": 0,
                            "runtime_fallbacks": 0,
                        }
                    profiles.append(sample)
        return {
            "correctness_verified": True,
            "input_rows": 64,
            "gpu_validation_queries": 2,
            "validation_records": [
                {
                    "sample": sample,
                    "cache_mode": "warm_storage",
                    "rows_compared": 64,
                    "gpu_executions": 1,
                    "plan_fallbacks": 0,
                    "runtime_fallbacks": 0,
                }
                for sample in ("prefix", "dispersed")
            ],
            "gpu_counters": {
                "executions": 12,
                "plan_fallbacks": 0,
                "runtime_fallbacks": 0,
            },
            "profiles": profiles,
        }

    def test_defaults_bound_inputs_and_use_one_worker(self):
        args = run_stages.parse_args(self.argv[:4])
        self.assertEqual(args.sizes, [100_000, 1_000_000])
        self.assertEqual(args.workloads, ["sql", "cudf", "cuda"])
        self.assertEqual((args.threads, args.batch_mib), (1, 8))
        self.assertEqual((args.gpu_budget_mib, args.host_budget_mib), (512, 1024))
        self.assertTrue(args.projection)
        self.assertFalse(args.pin)

    def test_invalid_runs_fail_before_creating_outputs(self):
        for extra in (
            ["--batch-mib", "0"],
            ["--gpu-budget-mib", "0"],
            ["--host-budget-mib", "1"],
            ["--threads", "0"],
            ["--repetitions", "4"],
            ["--sizes", "64", "64"],
            ["--workloads", "cuda", "cuda"],
            ["--workloads", "udfbench_q1"],
            [
                "--output-dir",
                str(run_stages.run_bench.REPO / "forbidden-stage-results"),
            ],
        ):
            with self.subTest(extra=extra):
                self.invalid_args(extra)

    def test_parallel_cpu_profile_is_explicit(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            args = run_stages.parse_args(self.argv + ["--threads", "8"])
        self.assertEqual(args.threads, 8)
        self.assertIn("not query latency speedups", stderr.getvalue())

    def test_gpu_pool_target_is_absolute_and_does_not_depend_on_card_size(self):
        small = {
            "total_bytes": 8 * 1024 * run_stages.MIB,
            "free_bytes": 6 * 1024 * run_stages.MIB,
        }
        large = {
            "total_bytes": 48 * 1024 * run_stages.MIB,
            "free_bytes": 32 * 1024 * run_stages.MIB,
        }
        first = run_stages.configuration(self.args, small)
        self.assertEqual(first, run_stages.configuration(self.args, large))
        self.assertIn("usage_limit_bytes: 536870912", first)
        self.assertNotIn("usage_limit_fraction", first)
        self.assertIn("scan_task_batch_size: 8388608", first)
        self.assertIn("pipeline: { num_threads: 1 }", first)

    def test_pool_budget_larger_than_free_memory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "currently free"):
            run_stages.configuration(
                self.args,
                {
                    "total_bytes": 1024 * run_stages.MIB,
                    "free_bytes": 256 * run_stages.MIB,
                },
            )

    def test_complete_manifest_is_accepted(self):
        run_stages.validate_manifest(self.manifest(), self.args, "sql", 64)

    def test_missing_duplicate_and_mislabeled_samples_are_rejected(self):
        for mutation in (
            "missing",
            "duplicate",
            "warmup",
            "rows",
            "gpu_count",
            "gpu_fallback",
            "correctness",
        ):
            manifest = self.manifest()
            if mutation == "missing":
                manifest["profiles"].pop()
            elif mutation == "duplicate":
                manifest["profiles"].append(copy.deepcopy(manifest["profiles"][-1]))
            elif mutation == "warmup":
                manifest["profiles"][-1]["warmup"] = True
            elif mutation == "rows":
                manifest["input_rows"] = 63
            elif mutation == "gpu_count":
                manifest["gpu_counters"]["executions"] -= 1
            elif mutation == "gpu_fallback":
                manifest["profiles"][-1]["gpu_counters"]["runtime_fallbacks"] = 1
            else:
                manifest["validation_records"][0]["rows_compared"] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                run_stages.validate_manifest(manifest, self.args, "sql", 64)

    def test_summary_excludes_warmups_and_unavailable_stages(self):
        records = []
        for engine in ("cpu", "gpu"):
            for warmup, duration in ((True, 500), (False, 2), (False, 4)):
                records.append(
                    {
                        "workload": "sql",
                        "rows": 64,
                        "engine": engine,
                        "cpu_mode": "native" if engine == "cpu" else "gpu",
                        "cpu_baseline": (
                            "Vectorized CPU SQL" if engine == "cpu" else "GPU"
                        ),
                        "query_kind": "projection",
                        "warmup": warmup,
                        "query_wall_ms": 9,
                        "memory_observations": {"pressure_observed": False},
                        "stages": {"projection_expressions": duration, "other": 0},
                        "stage_available": {
                            "projection_expressions": True,
                            "other": engine == "gpu",
                        },
                    }
                )
        summary, comparisons = run_stages.summarize(records)
        projections = [s for s in summary if s["stage"] == "projection_expressions"]
        self.assertEqual(len(projections), 2)
        self.assertTrue(
            all(
                s["samples"] == 2 and s["median_operator_work_ms"] == 3
                for s in projections
            )
        )
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0]["operator_work_ratio"], 1)
        for record in records:
            if record["engine"] == "gpu":
                record["stages"]["projection_expressions"] = 0
        self.assertIsNone(run_stages.summarize(records)[1][0]["operator_work_ratio"])

    def test_pressure_or_missing_evidence_suppresses_only_the_ratio(self):
        records = [
            {
                "workload": "sql",
                "rows": 64,
                "engine": engine,
                "cpu_mode": engine,
                "cpu_baseline": engine,
                "query_kind": "projection",
                "warmup": False,
                "query_wall_ms": 7,
                "stages": {"projection_expressions": 4 if engine == "cpu" else 2},
                "stage_available": {"projection_expressions": True},
                "memory_observations": {
                    "pressure_observed": engine == "gpu",
                    "task_allocator_peaks_bytes": [1024, 2048],
                },
            }
            for engine in ("cpu", "gpu")
        ]
        summary, comparisons = run_stages.summarize(records)
        gpu = next(row for row in summary if row["engine"] == "gpu")
        self.assertEqual(gpu["median_operator_work_ms"], 2)
        self.assertEqual(gpu["gpu_pressure_samples"], 1)
        self.assertEqual(gpu["max_task_allocator_peak_bytes"], 2048)
        self.assertIsNone(comparisons[0]["operator_work_ratio"])
        self.assertEqual(comparisons[0]["comparison_status"], "gpu_memory_pressure")
        records[1].pop("memory_observations")
        comparison = run_stages.summarize(records)[1][0]
        self.assertIsNone(comparison["operator_work_ratio"])
        self.assertIsNone(comparison["gpu_pressure_samples"])
        self.assertEqual(comparison["comparison_status"], "gpu_pressure_unknown")

    def test_failed_host_preserves_generated_inputs_and_nonzero_report(self):
        device = {
            "device_count": 1,
            "total_bytes": 8 * 1024 * run_stages.MIB,
            "free_bytes": 6 * 1024 * run_stages.MIB,
            "name": "test device",
        }
        with patch.object(run_stages, "device_info", return_value=device), patch.object(
            run_stages.run_bench,
            "metadata",
            return_value={},
        ), contextlib.redirect_stderr(io.StringIO()):
            result = run_stages.main(self.argv)
        self.assertEqual(result, 1)
        report = json.loads((self.output / "results.json").read_text())
        self.assertEqual(len(report["failures"]), 1)
        self.assertFalse(report["records"])
        job = self.output / "jobs/sql/64"
        self.assertTrue((job / "setup.sql").is_file())
        self.assertTrue((job / "sirius.yaml").is_file())
        self.assertTrue((job / "run.log").is_file())


if __name__ == "__main__":
    unittest.main()
