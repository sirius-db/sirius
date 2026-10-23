# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Pure-Python checks; these do not assert native or GPU readiness."""

import argparse
import importlib.util
import json
from pathlib import Path
import py_compile
import shutil
import struct
import subprocess
import sys
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


scaffold = load(ROOT / "scripts" / "scaffold.py", "sirius_udf_scaffold")
runner = load(ROOT / "assets" / "common" / "run.py", "sirius_udf_runner")


def native_report():
    return {
        "validation": "passed",
        "gpu_counters": {"executions": 1, "plan_fallbacks": 0, "runtime_fallbacks": 0},
        "gpu": {
            "gpu_counters": {
                "executions": 3,
                "plan_fallbacks": 0,
                "runtime_fallbacks": 0,
            }
        },
    }


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sirius-udf-skill-test-")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_all_targets_are_self_contained(self):
        for target in scaffold.TARGETS:
            with self.subTest(target=target):
                output = scaffold.scaffold(target, self.directory / target)
                project = json.loads((output / "project.json").read_text())
                self.assertEqual(project["target"], target)
                self.assertTrue(project["synthetic"])
                for filename in (
                    "run.py",
                    "setup.sql",
                    "reference.sql",
                    "candidate.sql",
                    "registration.sql",
                    "USAGE.md",
                ):
                    self.assertTrue((output / filename).is_file(), filename)
                py_compile.compile(str(output / "run.py"), doraise=True)
                if target == "sql":
                    self.assertFalse((output / "CMakeLists.txt").exists())
                else:
                    self.assertTrue((output / "host.cpp").exists())
                    self.assertTrue(
                        (
                            output / ("udf.cpp" if target == "cudf" else "udf.cu")
                        ).exists()
                    )
                    if target == "cuda":
                        self.assertTrue((output / "udf.cpp").is_file())
                        self.assertTrue((output / "gpu.hpp").is_file())

    def test_existing_output_is_not_overwritten(self):
        output = self.directory / "existing"
        output.mkdir()
        sentinel = output / "customer.txt"
        sentinel.write_text("preserve this")
        with self.assertRaisesRegex(ValueError, "already exists"):
            scaffold.scaffold("sql", output)
        self.assertEqual(sentinel.read_text(), "preserve this")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_generated_project_ignores_python_caches(self):
        assets = self.directory / "assets"
        for name in ("common", "sql"):
            (assets / name).mkdir(parents=True)
        (assets / "common" / "run.py").write_text("pass\n")
        cache = assets / "common" / "__pycache__"
        cache.mkdir()
        (cache / "run.pyc").write_bytes(bytes([255, 0, 254]))
        output = scaffold.scaffold("sql", self.directory / "output", assets=assets)
        self.assertFalse((output / "__pycache__").exists())
        self.assertEqual((output / "run.py").read_text(), "pass\n")

    def test_conflicting_template_files_fail_before_creation(self):
        assets = self.directory / "assets"
        for name in ("common", "sql"):
            (assets / name).mkdir(parents=True)
            (assets / name / "run.py").write_text("pass\n")
        output = self.directory / "output"
        with self.assertRaisesRegex(ValueError, "duplicate template"):
            scaffold.scaffold("sql", output, assets=assets)
        self.assertFalse(output.exists())

    def test_repository_destination_rejected_before_creation(self):
        repository = self.directory / "repo"
        repository.mkdir()
        # Worktrees use a .git file; support that as well as a directory.
        (repository / ".git").write_text("gitdir: unused")
        output = repository / "private-artifacts"
        with self.assertRaisesRegex(ValueError, "outside the Sirius"):
            scaffold.scaffold("cudf", output, repository_root=repository)
        self.assertFalse(output.exists())

    def test_symlink_into_repository_rejected(self):
        repository = self.directory / "repo"
        repository.mkdir()
        (repository / ".git").mkdir()
        (repository / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        alias = self.directory / "external-looking"
        alias.symlink_to(repository, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "outside the Sirius"):
            scaffold.scaffold("cuda", alias / "output", repository_root=repository)
        self.assertFalse((repository / "output").exists())

    def test_external_versioned_package_is_allowed(self):
        sirius = self.directory / "sirius"
        sirius.mkdir()
        external = self.directory / "external"
        external.mkdir()
        (external / ".git").mkdir()
        (external / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        output = scaffold.scaffold("cudf", external / "package", repository_root=sirius)
        self.assertTrue((output / "udf.cpp").is_file())

    def test_relative_path_and_missing_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            scaffold.scaffold("sql", Path("relative-output"))
        with self.assertRaisesRegex(ValueError, "parent"):
            scaffold.scaffold("sql", self.directory / "missing" / "output")

    def test_invalid_target_or_templates_do_not_create_output(self):
        output = self.directory / "output"
        with self.assertRaisesRegex(ValueError, "unknown target"):
            scaffold.scaffold("python", output)
        with self.assertRaisesRegex(ValueError, "missing template"):
            scaffold.scaffold("sql", output, assets=self.directory)
        self.assertFalse(output.exists())

    def test_cli_reports_existing_destination_failure(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "scaffold.py"),
                "--target",
                "sql",
                "--output",
                str(self.directory),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("already exists", completed.stderr)

    def test_cli_uses_canonical_skill_location_to_protect_repository(self):
        repository = self.directory / "sirius"
        script = repository / ".agents/skills/sirius-udf/scripts/scaffold.py"
        script.parent.mkdir(parents=True)
        script.write_text((ROOT / "scripts/scaffold.py").read_text())
        output = repository / "private-package"
        result = subprocess.run(
            [sys.executable, str(script), "--target", "sql", "--output", str(output)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the Sirius repository", result.stderr)
        self.assertFalse(output.exists())


class VerificationTests(unittest.TestCase):
    def test_python_log_override_precedes_loading_and_restores_on_failure(self):
        with tempfile.TemporaryDirectory(
            prefix="sirius-udf-private-logs-"
        ) as temporary:
            database = Path(temporary) / "validation.db"
            original_cwd = Path.cwd()

            def fail_loading(*args):
                self.assertEqual(
                    runner.os.environ["SIRIUS_LOG_DIR"], str(database.parent / "log")
                )
                self.assertEqual(Path.cwd(), original_cwd)
                raise RuntimeError("loading failed")

            with mock.patch.dict(
                runner.os.environ, {"SIRIUS_LOG_DIR": "caller-log"}
            ), mock.patch.object(
                runner, "_run_python_queries", side_effect=fail_loading
            ):
                with self.assertRaisesRegex(RuntimeError, "loading failed"):
                    runner.run_python(
                        None, database, "setup", "register", "ref", "candidate"
                    )
                self.assertEqual(runner.os.environ["SIRIUS_LOG_DIR"], "caller-log")

    def test_python_log_override_restores_absent_environment_variable(self):
        with tempfile.TemporaryDirectory(
            prefix="sirius-udf-private-logs-"
        ) as temporary, mock.patch.dict(runner.os.environ):
            runner.os.environ.pop("SIRIUS_LOG_DIR", None)
            with runner.private_validation_logs(Path(temporary) / "validation.db"):
                self.assertTrue(Path(runner.os.environ["SIRIUS_LOG_DIR"]).is_dir())
            self.assertNotIn("SIRIUS_LOG_DIR", runner.os.environ)

    def test_native_report_requires_validation_and_complete_gpu_proof(self):
        valid = native_report()
        self.assertEqual(runner.validate_native_report(valid, 3), valid)
        invalid = [None, {}, {"validation": "passed"}]
        for field, value in (("validation", "failed"), ("gpu", {})):
            report = native_report()
            report[field] = value
            invalid.append(report)
        for field, value in (
            ("executions", 0),
            ("plan_fallbacks", 1),
            ("runtime_fallbacks", 1),
            ("executions", True),
        ):
            report = native_report()
            report["gpu_counters"][field] = value
            invalid.append(report)
        report = native_report()
        report["gpu"]["gpu_counters"]["executions"] = 2
        invalid.append(report)
        for report in invalid:
            with self.subTest(report=report), self.assertRaises(
                (RuntimeError, AssertionError)
            ):
                runner.validate_native_report(report, 3)

    def test_sdk_revision_guard_with_standard_cpp_compiler(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("standard C++ compiler unavailable")
        source = r"""
#include "compatibility.hpp"
int main() {
  require_compatible_duckdb("88ecd0a49d7d3d838e002b2fbc0ce91ccf74110e", "88ecd0a49d");
  require_compatible_duckdb("88ecd0a49d", "88ecd0a49d7d3d838e002b2fbc0ce91ccf74110e");
  require_compatible_duckdb("88ecd0a49d", "88ecd0a49d");
  for (auto pair : {std::pair{"", "88ecd0a49d"},
                    std::pair{"88ecd0a49d", ""},
                    std::pair{"88ecd0a49d", "88ecd0a49e"},
                    std::pair{"unknown", "unknown"},
                    std::pair{"8", "88ecd0a49d"}}) {
    bool rejected = false;
    try { require_compatible_duckdb(pair.first, pair.second); }
    catch (std::runtime_error const&) { rejected = true; }
    if (!rejected) { return 1; }
  }
}
"""
        with tempfile.TemporaryDirectory(
            prefix="sirius-udf-sdk-identity-"
        ) as temporary:
            directory = Path(temporary)
            path = directory / "check.cpp"
            path.write_text(source)
            binary = directory / "check"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-I",
                    str(ROOT / "assets" / "native"),
                    str(path),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
            )
            subprocess.run([str(binary)], check=True, capture_output=True)

    def test_ieee_comparison_preserves_signed_zero_and_nan_payload(self):
        self.assertNotEqual(
            runner.encode_cell(0.0, "DOUBLE"), runner.encode_cell(-0.0, "DOUBLE")
        )
        first = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
        second = struct.unpack("!d", bytes.fromhex("fff8000000000002"))[0]
        self.assertNotEqual(
            runner.encode_cell(first, "DOUBLE"), runner.encode_cell(second, "DOUBLE")
        )
        self.assertEqual(
            runner.encode_cell(first, "DOUBLE")["ieee_bits"], "7ff8000000000001"
        )

    def test_missing_duckdb_runtime_is_actionable(self):
        with mock.patch.dict(sys.modules, {"duckdb": types.SimpleNamespace()}):
            with self.assertRaisesRegex(RuntimeError, "not a runnable DuckDB"):
                runner.PythonDuckDB(Path("unused"), Path("unused"))

    def test_stats_query_temporarily_disables_gpu(self):
        adapter = runner.PythonDuckDB.__new__(runner.PythonDuckDB)
        adapter.connection = mock.Mock()
        adapter.connection.execute.return_value.fetchone.return_value = (12, 3, 4)
        self.assertEqual(
            adapter.snapshot(),
            {"executions": 12, "plan_fallbacks": 3, "runtime_fallbacks": 4},
        )
        self.assertEqual(
            adapter.connection.execute.call_args_list,
            [
                mock.call("SET gpu_execution=false"),
                mock.call(runner.STATS_SQL),
                mock.call("SET gpu_execution=true"),
            ],
        )

    def test_gpu_proof_rejects_absence_and_both_fallback_kinds(self):
        before = {"executions": 4, "plan_fallbacks": 1, "runtime_fallbacks": 2}
        for after in (
            before,
            {**before, "plan_fallbacks": 2, "executions": 5},
            {**before, "runtime_fallbacks": 3, "executions": 5},
        ):
            with self.subTest(after=after), self.assertRaises(AssertionError):
                runner.require_gpu(before, after)
        self.assertEqual(
            runner.require_gpu(before, {**before, "executions": 6}),
            {"executions": 2, "plan_fallbacks": 0, "runtime_fallbacks": 0},
        )

    def test_result_comparison_distinguishes_null_string_and_types(self):
        expected = {"types": ["VARCHAR"], "rows": [[None], ["NULL"]]}
        runner.compare(expected, expected, "baseline")
        for actual in (
            {"types": ["VARCHAR"], "rows": [["NULL"], ["NULL"]]},
            {"types": ["VARCHAR"], "rows": [[None]]},
            {"types": ["DOUBLE"], "rows": expected["rows"]},
        ):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                runner.compare(expected, actual, "candidate")

    def test_gpu_failure_prevents_benchmark(self):
        class Adapter:
            closed = False

            def execute(self, query):
                self.commands.append(query)

            def query(self, query):
                if "CHECKPOINT" not in self.commands:
                    raise AssertionError("setup was not persisted before comparison")
                return {"types": ["DOUBLE"], "rows": [["1.0"], [None]]}

            def snapshot(self):
                return {"executions": 0, "plan_fallbacks": 0, "runtime_fallbacks": 0}

            def close(self):
                self.closed = True

        adapter = Adapter()
        adapter.commands = []
        args = argparse.Namespace(extension=Path("unused"), warmups=0, iterations=1)
        with tempfile.TemporaryDirectory(
            prefix="sirius-udf-checkpoint-"
        ) as temporary, mock.patch.object(
            runner, "PythonDuckDB", return_value=adapter
        ), mock.patch.object(
            runner, "benchmark"
        ) as benchmark:
            with self.assertRaisesRegex(AssertionError, "did not execute"):
                runner.run_python(
                    args,
                    Path(temporary) / "validation.db",
                    "setup",
                    "registration",
                    "ref",
                    "candidate",
                )
            benchmark.assert_not_called()
        self.assertTrue(adapter.closed)
        self.assertEqual(adapter.commands[:3], ["setup", "registration", "CHECKPOINT"])

    def test_benchmark_excludes_warmups_and_checks_gpu(self):
        class Adapter:
            count = 0

            def execute(self, query):
                pass

            def query(self, query):
                self.count += 1

            def snapshot(self):
                return {
                    "executions": self.count,
                    "plan_fallbacks": 0,
                    "runtime_fallbacks": 0,
                }

        adapter = Adapter()
        with mock.patch.object(
            runner.time, "perf_counter", side_effect=[0, 9, 10, 12, 20, 24]
        ):
            result = runner.benchmark(adapter, "query", 1, 2, True)
        self.assertEqual(result["samples_seconds"], [2, 4])
        self.assertEqual(result["median_seconds"], 3)
        self.assertEqual(result["gpu_counters"]["executions"], 3)

    def test_benchmark_requires_gpu_execution_for_every_query(self):
        class Adapter:
            count = 0

            def execute(self, query):
                pass

            def query(self, query):
                self.count += 1

            def snapshot(self):
                return {
                    "executions": min(self.count, 1),
                    "plan_fallbacks": 0,
                    "runtime_fallbacks": 0,
                }

        with self.assertRaisesRegex(AssertionError, "every measured query"):
            runner.benchmark(Adapter(), "query", 1, 2, True)

    def test_native_invocation_preserves_paths_and_does_not_use_shell(self):
        with tempfile.TemporaryDirectory(prefix="sirius udf ; ") as temporary:
            directory = Path(temporary)
            args = argparse.Namespace(
                host=directory / "host",
                plugin=directory / "plugin.so",
                warmups=1,
                iterations=2,
            )
            response = subprocess.CompletedProcess(
                [],
                0,
                stdout="engine logging\nSIRIUS_UDF_RESULT="
                + json.dumps(native_report())
                + "\n",
                stderr="",
            )
            with mock.patch.object(
                runner.subprocess, "run", return_value=response
            ) as execute:
                result = runner.run_native(
                    args, directory / "test.db", "setup", "register", "ref", "candidate"
                )
            self.assertEqual(result["validation"], "passed")
            argv = execute.call_args.args[0]
            self.assertEqual(argv[0], str(args.host))
            self.assertEqual(argv[2], str(args.plugin))
            self.assertNotIn("shell", execute.call_args.kwargs)
            self.assertNotIn("cwd", execute.call_args.kwargs)
            self.assertEqual(
                execute.call_args.kwargs["env"]["SIRIUS_LOG_DIR"],
                str(directory / "log"),
            )
            self.assertEqual((directory / "setup.sql").read_text(), "setup")

    def test_sql_host_invocation_omits_native_plugin(self):
        with tempfile.TemporaryDirectory(prefix="sirius-udf-sql-host-") as temporary:
            directory = Path(temporary)
            args = argparse.Namespace(
                host=directory / "host", plugin=None, warmups=1, iterations=2
            )
            response = subprocess.CompletedProcess(
                [],
                0,
                stdout="SIRIUS_UDF_RESULT=" + json.dumps(native_report()) + "\n",
                stderr="",
            )
            with mock.patch.object(
                runner.subprocess, "run", return_value=response
            ) as execute:
                result = runner.run_native(
                    args, directory / "test.db", "setup", "macro", "ref", "candidate"
                )
            self.assertEqual(result["validation"], "passed")
            self.assertEqual(execute.call_args.args[0][2], "-")


if __name__ == "__main__":
    unittest.main()
