"""CPU-only checks for benchmark setup and correctness-evidence enforcement."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_bench


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sirius-benchmark-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / "build with spaces"
        self.build.mkdir()
        host = self.build / "bench_host"
        host.write_text("#!/bin/sh\nexit 0\n")
        host.chmod(0o755)
        for name in ("cudf", "cuda", "q1"):
            (self.build / f"benchmark_{name}.so").touch()
        self.output = self.root / "run with spaces"
        self.argv = [
            "--build-dir",
            str(self.build),
            "--output-dir",
            str(self.output),
            "--workloads",
            "sql",
            "--sizes",
            "64",
            "--pin",
            "--projection",
        ]
        self.args = run_bench.parse_args(self.argv)

    def invalid_args(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                run_bench.parse_args(argv)
        self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.output.exists())

    def measurements(self):
        records = []
        for cache in ("warm_storage", "pinned_gpu"):
            for kind in ("aggregate", "projection"):
                records.append(
                    {
                        "workload": "sql",
                        "rows": 64,
                        "cpu_baseline": "Vectorized CPU SQL",
                        "cache_mode": cache,
                        "query_kind": f"{kind} / {cache}",
                        "verification": "passed",
                        "cpu_samples_ms": [4, 5, 6, 7, 8, 9, 10],
                        "gpu_samples_ms": [1, 2, 3, 4, 5, 6, 7],
                        "gpu_counters": {
                            "executions": 10,
                            "plan_fallbacks": 0,
                            "runtime_fallbacks": 0,
                        },
                    }
                )
        return {
            "correctness_verified": True,
            "gpu_validation_queries": 4,
            "input_rows": 64,
            "correctness_sample_rows": 64,
            "dispersed_sample_rows": 64,
            "validation_records": [
                {
                    "cache_mode": cache,
                    "sample": sample,
                    "rows_compared": 64,
                    "gpu_executions": 1,
                    "plan_fallbacks": 0,
                    "runtime_fallbacks": 0,
                }
                for cache in ("warm_storage", "pinned_gpu")
                for sample in ("prefix", "dispersed")
            ],
            "records": records,
        }

    def test_default_workloads_need_no_external_data(self):
        args = run_bench.parse_args(self.argv[:4])
        self.assertEqual(args.workloads, ["sql", "cudf", "cuda"])
        self.assertEqual(args.sizes, [100_000, 1_000_000, 10_000_000])
        self.assertEqual((args.warmups, args.repetitions, args.threads), (3, 7, 8))

    def test_invalid_measurement_parameters_fail_before_output_creation(self):
        for option, value in (
            ("--repetitions", "4"),
            ("--warmups", "0"),
            ("--threads", "0"),
            ("--sizes", "-1"),
        ):
            with self.subTest(option=option):
                self.invalid_args(self.argv + [option, value])

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "existing.txt"
        sentinel.write_text("keep")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_bench.parse_args(self.argv)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_symlink_cannot_put_results_inside_checkout(self):
        link = self.root / "checkout-link"
        link.symlink_to(run_bench.REPO, target_is_directory=True)
        argv = list(self.argv)
        argv[3] = str(link / "uncreated-benchmark-output")
        self.invalid_args(argv)

    def test_missing_native_module_is_reported_before_running(self):
        (self.build / "benchmark_cuda.so").unlink()
        self.invalid_args(self.argv[:4] + ["--workloads", "cuda"])

    def test_q1_requires_prepared_data_explicitly(self):
        self.invalid_args(self.argv[:4] + ["--workloads", "udfbench_q1"])

    def test_q1_data_path_is_sql_quoted_and_oracle_is_separate(self):
        source = self.root / "public 'data.csv"
        source.touch()
        args = run_bench.parse_args(
            self.argv[:4] + ["--workloads", "udfbench_q1", "--q1-data", str(source)]
        )
        setup = run_bench.definitions("udfbench_q1", 64, args)[0]
        self.assertIn("public ''data.csv", setup)
        self.assertIn("CREATE TABLE q1_oracle", setup)
        self.assertIn("JOIN q1_oracle", setup)

    def test_metadata_uses_actual_build_manifest_without_disclosing_status_paths(self):
        manifest = {"compiler_version": "14.4.0", "cudf_version": "26.08.01"}
        (self.build / "benchmark-build.json").write_text(json.dumps(manifest))
        with patch.object(
            run_bench,
            "command_output",
            side_effect=lambda command, *unused: (
                " M private.txt" if "status" in command else "available"
            ),
        ):
            metadata = run_bench.metadata(self.args)
        self.assertEqual(metadata["build"], manifest)
        self.assertTrue(metadata["source_dirty"])
        self.assertNotIn("private.txt", json.dumps(metadata))
        self.assertIn("logical_cpus", metadata)

    def test_unavailable_build_manifest_is_explicit(self):
        (self.build / "benchmark-build.json").write_text("{broken")
        with patch.object(run_bench, "command_output", return_value="unavailable"):
            metadata = run_bench.metadata(self.args)
        self.assertEqual(metadata["build"], {"status": "unavailable"})
        self.assertIsNone(metadata["source_dirty"])

    def test_complete_evidence_produces_numeric_medians(self):
        measurements = self.measurements()
        run_bench.validate_measurements(measurements, self.args, "sql", 64)
        record = measurements["records"][0]
        self.assertEqual(record["cpu_median_ms"], 7)
        self.assertEqual(record["gpu_median_ms"], 4)
        self.assertEqual(record["speedup"], 1.75)

    def test_cpu_fallback_or_missing_gpu_query_cannot_be_a_result(self):
        for key, value in (
            ("executions", 9),
            ("plan_fallbacks", 1),
            ("runtime_fallbacks", 1),
        ):
            with self.subTest(counter=key):
                measurements = self.measurements()
                measurements["records"][0]["gpu_counters"][key] = value
                with self.assertRaisesRegex(ValueError, "execution/fallback"):
                    run_bench.validate_measurements(measurements, self.args, "sql", 64)

    def test_incomplete_variants_or_nonfinite_samples_are_rejected(self):
        measurements = self.measurements()
        measurements["records"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete or duplicated"):
            run_bench.validate_measurements(measurements, self.args, "sql", 64)
        measurements = self.measurements()
        measurements["records"][0]["gpu_samples_ms"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid gpu_samples"):
            run_bench.validate_measurements(measurements, self.args, "sql", 64)

    def test_success_exit_without_correctness_proof_is_rejected(self):
        for field in (
            "correctness_verified",
            "gpu_validation_queries",
            "correctness_sample_rows",
            "dispersed_sample_rows",
        ):
            with self.subTest(field=field):
                measurements = self.measurements()
                measurements.pop(field)
                with self.assertRaises(ValueError):
                    run_bench.validate_measurements(measurements, self.args, "sql", 64)

    def test_total_gpu_count_does_not_replace_per_cache_proof(self):
        measurements = self.measurements()
        measurements["validation_records"][2]["cache_mode"] = "warm_storage"
        with self.assertRaisesRegex(ValueError, "correctness gates"):
            run_bench.validate_measurements(measurements, self.args, "sql", 64)

    def test_q1_requires_all_original_oracle_outputs_in_addition_to_samples(self):
        measurements = self.measurements()
        with self.assertRaisesRegex(ValueError, "correctness gates"):
            run_bench.validate_measurements(measurements, self.args, "udfbench_q1", 64)
        measurements["validation_records"].append(
            {
                "cache_mode": "warm_storage",
                "sample": "q1_original_python_oracle",
                "rows_compared": 49999,
                "gpu_executions": 1,
                "plan_fallbacks": 0,
                "runtime_fallbacks": 0,
            }
        )
        measurements["gpu_validation_queries"] += 1
        with self.assertRaisesRegex(ValueError, "all public oracle rows"):
            run_bench.validate_measurements(measurements, self.args, "udfbench_q1", 64)

    def test_correctness_fallback_is_rejected_even_when_timings_are_valid(self):
        measurements = self.measurements()
        measurements["validation_records"][0]["plan_fallbacks"] = 1
        with self.assertRaisesRegex(ValueError, "without fallback"):
            run_bench.validate_measurements(measurements, self.args, "sql", 64)

    def test_query_count_must_agree_with_individual_correctness_records(self):
        measurements = self.measurements()
        measurements["gpu_validation_queries"] += 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            run_bench.validate_measurements(measurements, self.args, "sql", 64)

    def test_validation_only_also_requires_actual_enabled_query_shapes(self):
        measurements = self.measurements()
        measurements["records"] = []
        args = copy.copy(self.args)
        args.validate_only = True
        with self.assertRaisesRegex(ValueError, "correctness gates"):
            run_bench.validate_measurements(measurements, args, "sql", 64)

    def test_failed_host_evidence_keeps_queries_logs_and_failure_report(self):
        measurements = self.measurements()
        measurements["records"][0]["gpu_counters"]["runtime_fallbacks"] = 1

        def fake_host(command, **kwargs):
            job = Path(command[1])
            self.assertEqual(Path(command[0]).parent, self.build)
            self.assertEqual(kwargs["cwd"], job)
            (job / "measurements.json").write_text(json.dumps(measurements))
            kwargs["stdout"].write("host diagnostics retained\n")
            return type("Process", (), {"returncode": 0})()

        with patch.object(run_bench, "metadata", return_value={}), patch.object(
            run_bench.subprocess, "run", side_effect=fake_host
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run_bench.main(self.argv), 1)
        report = json.loads((self.output / "results.json").read_text())
        self.assertEqual(report["results"], [])
        self.assertEqual(len(report["failures"]), 1)
        job = self.output / "jobs/sql/64"
        self.assertIn("host diagnostics", (job / "run.log").read_text())
        self.assertTrue((job / "measurements.json").is_file())
        self.assertTrue((job / "aggregate_gpu.sql").is_file())

    def test_validation_only_writes_proof_without_timing_claims(self):
        measurements = copy.deepcopy(self.measurements())
        measurements["records"] = []
        for cache in ("warm_storage", "pinned_gpu"):
            for kind in ("aggregate", "projection"):
                measurements["validation_records"].append(
                    {
                        "cache_mode": cache,
                        "sample": kind,
                        "rows_checked": 1 if kind == "aggregate" else 64,
                        "rows_compared": 1 if kind == "aggregate" else 0,
                        "verification": (
                            "exact_values"
                            if kind == "aggregate"
                            else "row_count_and_logical_types"
                        ),
                        "gpu_executions": 1,
                        "plan_fallbacks": 0,
                        "runtime_fallbacks": 0,
                    }
                )
                measurements["gpu_validation_queries"] += 1

        def fake_host(command, **kwargs):
            (Path(command[1]) / "measurements.json").write_text(
                json.dumps(measurements)
            )
            return type("Process", (), {"returncode": 0})()

        with patch.object(run_bench, "metadata", return_value={}), patch.object(
            run_bench.subprocess, "run", side_effect=fake_host
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_bench.main(self.argv + ["--validate-only"]), 0)
        report = json.loads((self.output / "results.json").read_text())
        self.assertEqual(report["results"], [])
        self.assertEqual(report["failures"], [])
        self.assertTrue(report["jobs"][0]["correctness_verified"])


if __name__ == "__main__":
    unittest.main()
