# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Aggregate protocol, equivalence and failure-before-benchmark checks."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = load(ROOT / "assets/common/aggregate.py", "aggregate_test_runner")
runner = load(ROOT / "assets/common/run.py", "aggregate_test_legacy")
scaffold = load(ROOT / "scripts/scaffold.py", "aggregate_test_scaffold")


def value(number):
    return {"ieee_bits": struct.pack("!d", number).hex()}


def result(rows=None, types=None):
    return {
        "types": types or ["DOUBLE"],
        "rows": [[value(13.0)]] if rows is None else rows,
    }


def counts(executions=1, plan=0, runtime=0):
    return {
        "executions": executions,
        "plan_fallbacks": plan,
        "runtime_fallbacks": runtime,
    }


class ResultTests(unittest.TestCase):
    def test_exact_types_nulls_order_and_ieee_bits(self):
        expected = result([[None], [value(-0.0)], [value(float("nan"))]])
        aggregate.compare(expected, expected, "candidate")
        for bad in (
            result([[None], [value(0.0)], [value(float("nan"))]]),
            result([[value(-0.0)], [None], [value(float("nan"))]]),
            result([[None]]),
            result([], ["BIGINT"]),
        ):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                aggregate.compare(expected, bad, "candidate")

    def test_tolerance_is_explicit_finite_and_column_specific(self):
        expected, actual = result([[value(1.0)]]), result([[value(1.00001)]])
        with self.assertRaises(AssertionError):
            aggregate.compare(expected, actual, "candidate")
        policy = {"float_tolerances": {"0": {"atol": 0.0001, "rtol": 0}}}
        aggregate.compare(expected, actual, "candidate", policy)
        with self.assertRaises(AssertionError):
            aggregate.compare(
                result([[value(float("inf"))]]),
                result([[value(1e308)]]),
                "candidate",
                policy,
            )
        with self.assertRaises(AssertionError):
            aggregate.compare(
                result([[value(float("nan"))]]),
                result([[value(1.0)]]),
                "candidate",
                policy,
            )
        for limits in (
            {"atol": -1, "rtol": 0},
            {"atol": True, "rtol": 0},
            {"atol": float("nan"), "rtol": 0},
            {"atol": 1},
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                aggregate.compare(
                    expected, actual, "candidate", {"float_tolerances": {"0": limits}}
                )
        with self.assertRaises(ValueError):
            aggregate.compare(
                result([["1"]], ["BIGINT"]),
                result([["1"]], ["BIGINT"]),
                "candidate",
                policy,
            )

    def test_typed_protocol_rejects_lossy_or_malformed_cells(self):
        for malformed in (
            {"types": ["DOUBLE"], "rows": [[1.5]]},
            result([[{"ieee_bits": "00"}]]),
            result([[{"ieee_bits": "z" * 16}]]),
            result([[value(1.0), None]]),
            {"types": [], "rows": []},
            {"types": ["BIGINT"], "rows": [[17]]},
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                aggregate.table(malformed, "source")
        aggregate.compare(result([], ["VARCHAR"]), result([], ["VARCHAR"]), "empty")
        text = result([['quote"\\\n\x00é']], ["VARCHAR"])
        aggregate.compare(text, json.loads(json.dumps(text)), "string")

    def test_gpu_proof_cannot_be_missing_fallback_or_forged(self):
        aggregate.gpu_evidence(counts(), True)
        aggregate.gpu_evidence(counts(0), False)
        for evidence in (
            counts(0),
            counts(2),
            counts(plan=1),
            counts(runtime=1),
            counts(True),
            {},
            counts(-1),
        ):
            with self.subTest(evidence=evidence), self.assertRaises(
                (AssertionError, ValueError)
            ):
                aggregate.gpu_evidence(evidence, True)


class SourceAndScaffoldTests(unittest.TestCase):
    def test_original_state_contracts_cover_null_zero_and_partial_merge(self):
        source = load(
            ROOT / "assets/aggregate/original.py", "independent_aggregate_source"
        )
        squares = source.SumSquares()
        self.assertIsNone(squares.finalize())
        squares.update(None)
        self.assertIsNone(squares.finalize())
        squares.update(-3.0)
        partial = source.SumSquares()
        partial.update(2.0)
        squares.merge(partial)
        self.assertEqual(squares.finalize(), 13.0)
        weighted = source.WeightedMean()
        for pair in ((None, 3.0), (6.0, None), (10.0, 0.0)):
            weighted.update(*pair)
        self.assertIsNone(weighted.finalize())
        weighted.update(2.0, 1.0)
        weighted.update(10.0, -1.0)
        self.assertIsNone(weighted.finalize())
        partial = source.WeightedMean()
        partial.update(4.0, 2.0)
        weighted.merge(partial)
        self.assertEqual(weighted.finalize(), 0.0)

    def test_aggregate_scaffold_and_scalar_default_are_compatible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scalar = scaffold.scaffold("sql", root / "scalar")
            self.assertEqual(
                json.loads((scalar / "project.json").read_text())["kind"], "scalar"
            )
            project = scaffold.scaffold("sql", root / "aggregate", kind="aggregate")
            self.assertEqual(
                json.loads((project / "project.json").read_text())["kind"], "aggregate"
            )
            suite = aggregate.read_manifest(project)
            self.assertEqual(len(suite["cases"]), 13)
            for target in ("cudf", "cuda"):
                output = root / target
                with self.assertRaisesRegex(ValueError, "scalar-only"):
                    scaffold.scaffold(target, output, kind="aggregate")
                self.assertFalse(output.exists())


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aggregate-runner-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        self.root.mkdir()
        for name, text in (
            ("setup.sql", "SETUP @ROWS@"),
            ("input.sql", "INPUT"),
            ("candidate.sql", "CANDIDATE"),
            ("registration.sql", "REGISTER"),
        ):
            (self.root / name).write_text(text)
        self.suite = {
            "version": 1,
            "oracle": {"argv": ["{python}", "oracle.py"], "timeout_seconds": 1},
            "benchmark_case": "grouped",
            "cases": [
                {
                    "id": "grouped",
                    "setup": "setup.sql",
                    "input": "input.sql",
                    "candidate": "candidate.sql",
                    "parameters": {},
                    "gpu_required": True,
                }
            ],
        }
        self.write_suite()
        self.args = argparse.Namespace(
            host=self.root / "host",
            extension=None,
            rows=123,
            report=Path(self.temporary.name) / "report.json",
            validate_only=False,
        )
        self.exported = {
            "version": 1,
            "input": result([[value(2.0)], [value(3.0)]]),
            "candidate_cpu": result(),
            "candidate_gpu": result(),
            "gpu_counters": counts(),
        }
        self.project = {"reference": "independent synthetic original"}

    def write_suite(self):
        (self.root / "cases.json").write_text(json.dumps(self.suite))

    def test_manifest_requires_unique_cases_valid_paths_and_gpu_gate(self):
        aggregate.read_manifest(self.root)
        variants = []
        no_gpu = json.loads(json.dumps(self.suite))
        no_gpu["cases"][0]["gpu_required"] = False
        variants.append(no_gpu)
        duplicate = json.loads(json.dumps(self.suite))
        duplicate["cases"].append(duplicate["cases"][0])
        variants.append(duplicate)
        escaped = json.loads(json.dumps(self.suite))
        escaped["cases"][0]["setup"] = "../setup.sql"
        variants.append(escaped)
        invalid_version = json.loads(json.dumps(self.suite))
        invalid_version["version"] = True
        variants.append(invalid_version)
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                (self.root / "cases.json").write_text(json.dumps(variant))
                aggregate.read_manifest(self.root)

    def test_success_gates_benchmark_and_validate_only_omits_timing(self):
        measured = {
            "timing": "query wall time",
            "cpu": {"samples": [1]},
            "gpu": {"samples": [2]},
        }
        with mock.patch.object(
            aggregate, "export_native", return_value=self.exported
        ) as export, mock.patch.object(
            aggregate, "original_reference", return_value=result()
        ) as oracle:
            benchmark = mock.Mock(return_value=measured)
            report = aggregate.run_project(
                self.root, self.args, self.project, None, benchmark
            )
            self.assertEqual(report["validation"], "passed")
            self.assertEqual(report["cases"][0]["input_rows"], 2)
            self.assertIn("correctness-only", report["cpu_baseline"])
            self.assertEqual(benchmark.call_args.args[2], "SETUP 123")
            self.assertEqual(benchmark.call_args.args[-2:], ("CANDIDATE", "CANDIDATE"))
            export.assert_called_once()
            oracle.assert_called_once()
            self.args.validate_only = True
            benchmark.reset_mock()
            report = aggregate.run_project(
                self.root, self.args, self.project, None, benchmark
            )
            self.assertEqual(report["validation"], "passed")
            self.assertNotIn("cpu", report)
            benchmark.assert_not_called()

    def test_late_wrong_finalizer_or_null_prevents_every_benchmark(self):
        second = dict(self.suite["cases"][0], id="second")
        self.suite["cases"].append(second)
        self.write_suite()
        for bad in (result([[value(14.0)]]), result([[None]])):
            broken = dict(self.exported, candidate_gpu=bad)
            benchmark = mock.Mock()
            with self.subTest(bad=bad), mock.patch.object(
                aggregate, "export_native", side_effect=[self.exported, broken]
            ), mock.patch.object(
                aggregate, "original_reference", return_value=result()
            ), self.assertRaises(
                AssertionError
            ):
                aggregate.run_project(
                    self.root, self.args, self.project, None, benchmark
                )
            benchmark.assert_not_called()
            saved = json.loads(self.args.report.read_text())
            self.assertEqual(saved["validation"], "failed")
            self.assertEqual(saved["active_case"], "second")
            self.assertEqual(len(saved["cases"]), 1)
            self.assertTrue(Path(saved["evidence_directory"]).is_dir())

    def test_missing_benchmark_case_cannot_produce_timing(self):
        self.suite.pop("benchmark_case")
        self.write_suite()
        with mock.patch.object(
            aggregate, "export_native", return_value=self.exported
        ), mock.patch.object(
            aggregate, "original_reference", return_value=result()
        ), self.assertRaisesRegex(
            ValueError, "benchmark_case"
        ):
            aggregate.run_project(self.root, self.args, self.project, None, mock.Mock())

    def test_reference_protocol_argv_roundtrip_and_failures(self):
        work = Path(self.temporary.name) / "oracle-evidence"
        work.mkdir()
        response = subprocess.CompletedProcess(
            [], 0, json.dumps({"version": 1, "result": result()}), ""
        )
        with mock.patch.object(
            aggregate.subprocess, "run", return_value=response
        ) as execute:
            actual = aggregate.original_reference(
                self.root, self.suite["oracle"], self.suite["cases"][0], result(), work
            )
            self.assertEqual(actual, result())
            self.assertNotIn("shell", execute.call_args.kwargs)
            self.assertEqual(execute.call_args.kwargs["cwd"], self.root)
            self.assertEqual(
                json.loads(execute.call_args.kwargs["input"])["case_id"], "grouped"
            )
        for output, code in (
            ("not-json", 0),
            (json.dumps({"version": 2, "result": result()}), 0),
            (json.dumps({"version": 1, "error": {"message": "original error"}}), 0),
            ("", 1),
        ):
            with self.subTest(output=output), mock.patch.object(
                aggregate.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], code, output, ""),
            ), self.assertRaises((ValueError, RuntimeError)):
                aggregate.original_reference(
                    self.root,
                    self.suite["oracle"],
                    self.suite["cases"][0],
                    result(),
                    work,
                )
        with mock.patch.object(
            aggregate.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("oracle", 1),
        ), self.assertRaises(subprocess.TimeoutExpired):
            aggregate.original_reference(
                self.root, self.suite["oracle"], self.suite["cases"][0], result(), work
            )

    def test_python_adapter_rejects_multiple_or_mutating_statements(self):
        adapter = runner.PythonDuckDB.__new__(runner.PythonDuckDB)
        connection = mock.Mock()
        adapter.connection = connection
        select = types.SimpleNamespace(type=types.SimpleNamespace(name="SELECT"))
        for parsed in (
            [select, select],
            [],
            [types.SimpleNamespace(type=types.SimpleNamespace(name="DELETE"))],
        ):
            connection.extract_statements.return_value = parsed
            with self.subTest(parsed=parsed), self.assertRaisesRegex(
                ValueError, "exactly one SELECT"
            ):
                adapter.query("candidate")
            connection.execute.assert_not_called()
        self.assertEqual(runner.encode_cell(True, "BOOLEAN"), "true")
        self.assertEqual(runner.encode_cell(False, "BOOLEAN"), "false")

    def test_native_framing_preserves_unicode_line_separators(self):
        payload = dict(
            self.exported, input=result([["line\u2028next\x85last\u2029"]], ["VARCHAR"])
        )
        # ensure_ascii=False exercises legal Unicode inside the native JSON record.
        process = subprocess.CompletedProcess(
            [],
            0,
            "log\nSIRIUS_UDF_CASE=" + json.dumps(payload, ensure_ascii=False) + "\n",
            "",
        )
        work = Path(self.temporary.name) / "framing"
        work.mkdir()
        with mock.patch.object(aggregate.subprocess, "run", return_value=process):
            actual = aggregate.export_native(
                self.args, work, "setup", "registration", "input", "candidate"
            )
        self.assertEqual(actual["input"], payload["input"])

    def test_measurement_reuses_validated_database_and_requires_gpu_samples(self):
        self.args.warmups, self.args.iterations = 1, 2
        measured = {
            "measurement": "complete",
            "timing": "query wall time",
            "cpu": {"samples_seconds": [0.1, 0.2], "median_seconds": 0.15},
            "gpu": {
                "samples_seconds": [0.1, 0.2],
                "median_seconds": 0.15,
                "gpu_counters": counts(3),
            },
        }
        database = self.root / "already-validated.db"
        database.write_text("sentinel")
        response = subprocess.CompletedProcess(
            [], 0, "SIRIUS_UDF_MEASUREMENT=" + json.dumps(measured), ""
        )
        with mock.patch.object(
            aggregate.subprocess, "run", return_value=response
        ) as execute:
            actual = aggregate.measure_native(
                self.args,
                database,
                "never-run-setup",
                "never-run-register",
                "ref",
                "candidate",
            )
        self.assertEqual(actual, measured)
        self.assertEqual(
            execute.call_args.args[0][1:3], ["--measure-case", str(database)]
        )
        self.assertEqual(database.read_text(), "sentinel")
        measured["gpu"]["gpu_counters"] = counts(2)
        response.stdout = "SIRIUS_UDF_MEASUREMENT=" + json.dumps(measured)
        with mock.patch.object(
            aggregate.subprocess, "run", return_value=response
        ), self.assertRaisesRegex(ValueError, "every query"):
            aggregate.measure_native(
                self.args, database, "setup", "register", "ref", "candidate"
            )

    def test_python_export_adapter_closes_and_restores_environment(self):
        class Adapter:
            def __init__(self):
                self.commands = []
                self.queries = 0
                self.closed = False

            def execute(self, sql):
                self.commands.append(sql)

            def query(self, sql):
                self.queries += 1
                return result()

            def snapshot(self):
                return counts(1 if self.queries >= 3 else 0)

            def close(self):
                self.closed = True

        adapter = Adapter()
        work = Path(self.temporary.name) / "python-export"
        work.mkdir()
        with mock.patch.dict(os.environ, {"SIRIUS_LOG_DIR": "prior"}):
            actual = aggregate.export_python(
                self.args,
                work,
                "setup",
                "register",
                "input",
                "candidate",
                lambda *args: adapter,
            )
            self.assertEqual(os.environ["SIRIUS_LOG_DIR"], "prior")
        self.assertTrue(adapter.closed)
        self.assertEqual(actual["gpu_counters"], counts())
        self.assertEqual(adapter.commands[:3], ["setup", "register", "CHECKPOINT"])
        failing_work = Path(self.temporary.name) / "python-export-failing"
        failing_work.mkdir()
        with mock.patch.dict(os.environ, {"SIRIUS_LOG_DIR": "prior"}):
            with self.assertRaisesRegex(RuntimeError, "loading failed"):
                aggregate.export_python(
                    self.args,
                    failing_work,
                    "setup",
                    "register",
                    "input",
                    "candidate",
                    mock.Mock(side_effect=RuntimeError("loading failed")),
                )
            self.assertEqual(os.environ["SIRIUS_LOG_DIR"], "prior")

    def test_legacy_native_validate_only_has_no_timing(self):
        native = {"validation": "passed", "gpu_counters": counts()}
        runner.validate_native_report(native, 0)
        with self.assertRaisesRegex(RuntimeError, "timing"):
            runner.validate_native_report(dict(native, gpu={}), 0)
        args = argparse.Namespace(
            host=self.root / "host",
            plugin=None,
            warmups=1,
            iterations=2,
            validate_only=True,
        )
        response = subprocess.CompletedProcess(
            [], 0, "SIRIUS_UDF_RESULT=" + json.dumps(native), ""
        )
        with mock.patch.object(
            runner.subprocess, "run", return_value=response
        ) as execute:
            runner.run_native(
                args,
                self.root / "test.db",
                "setup",
                "register",
                "reference",
                "candidate",
            )
        self.assertEqual(execute.call_args.args[0][1], "--validate-only")


if __name__ == "__main__":
    unittest.main()
