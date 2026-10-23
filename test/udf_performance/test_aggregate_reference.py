"""Exercise the original executable source gate, including fail-before-profile."""

import contextlib
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import aggregate_reference
import run_bench
import run_stages


def floating(value):
    return {"ieee_bits": struct.pack("!d", value).hex()}


class AggregateReferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sirius-aggregate-reference-")
        self.addCleanup(temporary.cleanup)
        self.job = Path(temporary.name)

    def document(self, workload="sum_squares"):
        results = (
            {
                "global": [[floating(6.0)]],
                "grouped_uniform": [["0", floating(5.0)], ["1", floating(1.0)]],
                "grouped_skewed": [["0", floating(5.0)], ["1", floating(1.0)]],
            }
            if workload == "sum_squares"
            else {
                "global": [[floating(0.25)]],
                "grouped_uniform": [["0", floating(1.5)], ["1", floating(-1.0)]],
                "grouped_skewed": [["0", floating(1.5)], ["1", floating(-1.0)]],
            }
        )
        cases = []
        for kind in run_bench.AGGREGATE_KINDS:
            result = {
                "types": (["INTEGER"] if kind != "global" else []) + ["DOUBLE"],
                "rows": results[kind],
            }
            cases.append(
                {
                    "query_kind": kind,
                    "candidate_cpu": copy.deepcopy(result),
                    "candidate_gpu": copy.deepcopy(result),
                    "gpu_counters": {
                        "executions": 1,
                        "plan_fallbacks": 0,
                        "runtime_fallbacks": 0,
                    },
                }
            )
        return {
            "version": 1,
            "input": {
                "types": ["BIGINT", "DOUBLE", "DOUBLE", "INTEGER", "INTEGER"],
                "rows": [
                    ["0", floating(1.0), floating(1.0), "0", "0"],
                    ["1", None, floating(3.0), "1", "0"],
                    ["2", floating(2.0), floating(1.0), "0", "0"],
                    ["3", floating(-1.0), floating(2.0), "1", "1"],
                ],
            },
            "cases": cases,
        }

    def write(self, document):
        (self.job / "aggregate-preflight.json").write_text(json.dumps(document))

    def test_both_originals_validate_complete_global_and_grouped_results(self):
        for workload in run_bench.AGGREGATE_WORKLOADS:
            with self.subTest(workload=workload):
                self.write(self.document(workload))
                result = aggregate_reference.verify(self.job, workload, 4)
                self.assertTrue(result["correctness_verified"])
                self.assertEqual(len(result["cases"]), 3)
                self.assertTrue(
                    all(case["input_rows_compared"] == 4 for case in result["cases"])
                )
                request = json.loads(
                    (self.job / "source-reference-global.request.json").read_text()
                )
                self.assertEqual(request["parameters"]["merge_partitions"], 7)
                self.assertEqual(len(request["input"]["rows"]), 4)

    def test_wrong_finalizer_null_type_or_bits_cannot_pass(self):
        for fault in ("finalizer", "null", "type", "bits"):
            document = self.document()
            actual = document["cases"][0]["candidate_gpu"]
            if fault == "finalizer":
                actual["rows"][0][0] = floating(3.0)
            elif fault == "null":
                actual["rows"][0][0] = None
            elif fault == "type":
                actual["types"] = ["FLOAT"]
            else:
                actual["rows"][0][0] = {"ieee_bits": "4018000000000001"}
            self.write(document)
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, "gpu"):
                aggregate_reference.verify(self.job, "sum_squares", 4)

    def test_full_input_query_kinds_and_gpu_evidence_are_mandatory(self):
        for fault in ("input", "order", "case", "fallback"):
            document = self.document()
            if fault == "input":
                document["input"]["rows"].pop()
            elif fault == "order":
                document["input"]["rows"].reverse()
            elif fault == "case":
                document["cases"].pop()
            else:
                document["cases"][0]["gpu_counters"]["plan_fallbacks"] = 1
            self.write(document)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                aggregate_reference.verify(self.job, "sum_squares", 4)

    def test_boolean_float_versions_and_counters_are_not_integer_evidence(self):
        for fault in (
            "version_bool",
            "version_float",
            "execution_bool",
            "execution_float",
            "plan_bool",
            "runtime_float",
        ):
            document = self.document()
            if fault.startswith("version"):
                document["version"] = True if fault.endswith("bool") else 1.0
            else:
                name = {
                    "execution": "executions",
                    "plan": "plan_fallbacks",
                    "runtime": "runtime_fallbacks",
                }[fault.split("_")[0]]
                value = 1 if name == "executions" else 0
                document["cases"][0]["gpu_counters"][name] = (
                    bool(value) if fault.endswith("bool") else float(value)
                )
            self.write(document)
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                aggregate_reference.verify(self.job, "sum_squares", 4)

    def test_malformed_export_schema_raises_value_error_before_adapter(self):
        for fault in (
            "top_level",
            "input",
            "case_kind",
            "row",
            "double_bits",
            "integer",
        ):
            document = self.document()
            if fault == "top_level":
                document = []
            elif fault == "input":
                document["input"] = []
            elif fault == "case_kind":
                document["cases"][0]["query_kind"] = []
            elif fault == "row":
                document["input"]["rows"][0] = None
            elif fault == "double_bits":
                document["cases"][0]["candidate_gpu"]["rows"][0][0] = {
                    "ieee_bits": True
                }
            else:
                document["input"]["rows"][0][3] = True
            self.write(document)
            with self.subTest(fault=fault), patch.object(
                aggregate_reference.subprocess, "run"
            ) as execute:
                with self.assertRaises(ValueError):
                    aggregate_reference.verify(self.job, "sum_squares", 4)
                execute.assert_not_called()

    def test_malformed_original_response_or_boolean_version_is_rejected(self):
        self.write(self.document())
        adapter = self.job / "malformed original.py"
        for response in (
            [],
            {"version": True, "result": {}},
            {"version": 1, "result": {"types": ["DOUBLE"], "rows": [[True]]}},
        ):
            adapter.write_text(f"print({json.dumps(json.dumps(response))})\n")
            with self.subTest(response=response), self.assertRaises(ValueError):
                aggregate_reference.verify(self.job, "sum_squares", 4, adapter=adapter)

    def test_source_adapter_error_is_a_failure_without_verdict(self):
        self.write(self.document())
        adapter = self.job / "faulty original.py"
        adapter.write_text(
            'print(\'{"version":1,"error":{"message":"source failed"}}\')\n'
        )
        with self.assertRaisesRegex(ValueError, "invalid original source"):
            aggregate_reference.verify(self.job, "sum_squares", 4, adapter=adapter)
        self.assertFalse((self.job / "source-reference-verdict.json").exists())

    def test_aggregate_setup_needs_no_native_module_and_labels_real_query_kinds(self):
        build = self.job / "build"
        build.mkdir()
        host = build / "bench_host"
        host.write_text("#!/bin/sh\nexit 0\n")
        host.chmod(0o755)
        args = run_stages.parse_args(
            [
                "--build-dir",
                str(build),
                "--output-dir",
                str(self.job / "results"),
                "--workloads",
                "sum_squares",
                "weighted_mean",
                "--sizes",
                "100000",
            ]
        )
        for workload in args.workloads:
            job = run_bench.make_job(workload, 100000, args)
            self.assertIn("plugin=-", (job / "config.txt").read_text())
            self.assertFalse((job / "projection_gpu.sql").exists())
            self.assertEqual(
                set((job / "query_kinds.tsv").read_text().splitlines()),
                set(run_bench.AGGREGATE_KINDS),
            )
            self.assertIn("bench_", (job / "global_gpu.sql").read_text())
            self.assertNotIn(
                "bench_" + workload, (job / "global_native.sql").read_text()
            )
            self.assertIn(
                "GROUP BY g_skewed", (job / "grouped_skewed_gpu.sql").read_text()
            )

    def test_manifest_preserves_aggregate_shapes_and_requires_all_queries(self):
        profiles = []
        validation = []
        for kind in run_bench.AGGREGATE_KINDS:
            validation.append(
                {
                    "cache_mode": "warm_storage",
                    "sample": kind,
                    "rows_compared": 1,
                    "gpu_executions": 1,
                    "plan_fallbacks": 0,
                    "runtime_fallbacks": 0,
                }
            )
            for mode in ("native", "gpu"):
                for iteration in range(6):
                    profile = {
                        "sample_id": f"{mode}_{kind}_{iteration}",
                        "engine": "gpu" if mode == "gpu" else "cpu",
                        "cpu_mode": mode,
                        "query_kind": kind,
                        "cpu_baseline": (
                            "GPU" if mode == "gpu" else "Equivalent DuckDB CPU SQL"
                        ),
                        "iteration": iteration,
                        "warmup": iteration == 0,
                        "query_wall_ms": 1,
                        "sql": "SELECT sum(value) FROM bench_input",
                    }
                    if mode == "gpu":
                        profile["gpu_counters"] = {
                            "executions": 1,
                            "plan_fallbacks": 0,
                            "runtime_fallbacks": 0,
                        }
                    profiles.append(profile)
        manifest = {
            "correctness_verified": True,
            "input_rows": 64,
            "validation_records": validation,
            "gpu_validation_queries": 3,
            "gpu_counters": {
                "executions": 18,
                "plan_fallbacks": 0,
                "runtime_fallbacks": 0,
            },
            "profiles": profiles,
        }
        args = SimpleNamespace(warmups=1, repetitions=5)
        run_stages.validate_manifest(manifest, args, "sum_squares", 64)
        for fault in (
            "total_float",
            "validation_float",
            "proof_bool",
            "proof_float",
            "input_float",
            "profile_key",
        ):
            malformed = copy.deepcopy(manifest)
            if fault == "total_float":
                malformed["gpu_counters"]["executions"] = 18.0
            elif fault == "validation_float":
                malformed["gpu_validation_queries"] = 3.0
            elif fault == "proof_bool":
                malformed["validation_records"][0]["gpu_executions"] = True
            elif fault == "proof_float":
                malformed["validation_records"][0]["plan_fallbacks"] = 0.0
            elif fault == "input_float":
                malformed["input_rows"] = 64.0
            else:
                malformed["profiles"][0]["cpu_mode"] = []
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                run_stages.validate_manifest(malformed, args, "sum_squares", 64)
        manifest["profiles"][-1]["query_kind"] = "projection"
        with self.assertRaisesRegex(ValueError, "unexpected profile"):
            run_stages.validate_manifest(manifest, args, "sum_squares", 64)

    def test_failed_original_gate_never_starts_profile_host(self):
        build = self.job / "build"
        build.mkdir()
        host = build / "bench_host"
        host.write_text("#!/bin/sh\nexit 0\n")
        host.chmod(0o755)
        output = self.job / "results"
        device = {
            "device_count": 1,
            "total_bytes": 8 * 1024 * run_stages.MIB,
            "free_bytes": 6 * 1024 * run_stages.MIB,
        }
        with patch.object(run_stages, "device_info", return_value=device), patch.object(
            run_bench, "metadata", return_value={}
        ), patch.object(
            aggregate_reference, "verify", side_effect=ValueError("wrong finalizer")
        ), patch.object(
            run_stages.subprocess,
            "run",
            return_value=type("Process", (), {"returncode": 0})(),
        ) as execute, contextlib.redirect_stderr(
            io.StringIO()
        ):
            result = run_stages.main(
                [
                    "--build-dir",
                    str(build),
                    "--output-dir",
                    str(output),
                    "--workloads",
                    "sum_squares",
                    "--sizes",
                    "64",
                ]
            )
        self.assertEqual(result, 1)
        self.assertEqual(execute.call_count, 1)
        self.assertIn("--aggregate-preflight", execute.call_args.args[0])
        report = json.loads((output / "results.json").read_text())
        self.assertFalse(report["records"])
        self.assertIn("wrong finalizer", report["failures"][0]["error"])
        self.assertFalse((output / "jobs/sum_squares/64/run.log").exists())


if __name__ == "__main__":
    unittest.main()
