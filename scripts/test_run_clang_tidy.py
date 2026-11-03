"""Behavior checks for standalone analysis selection and failure reporting."""

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_clang_tidy as tidy


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "src/example.cpp"
        self.source.parent.mkdir()
        self.source.write_text("int main() {}\n")
        self.build = self.root / "build"
        self.build.mkdir()
        self.command = tidy.CompileCommand.parse(
            {
                "directory": str(self.build),
                "file": str(self.source),
                "arguments": [
                    "clang++",
                    "-std=c++20",
                    "-o",
                    "CMakeFiles/sirius_objects.dir/example.o",
                    "-c",
                    str(self.source),
                ],
            }
        )

    def test_header_change_requires_explicit_full_run(self):
        with self.assertRaisesRegex(ValueError, "clang-tidy-all"):
            tidy.select_commands(
                [self.command], [self.source.with_suffix(".hpp")], False
            )

    def test_unconfigured_source_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no supported compile command"):
            tidy.select_commands([self.command], [self.root / "src/new.cpp"], False)

    def test_full_run_preserves_distinct_compile_contexts(self):
        variant = tidy.CompileCommand(
            self.source, self.build, ("clang++", "-DOTHER_VARIANT"), self.command.output
        )
        self.assertEqual(
            tidy.select_commands([self.command, variant], [], True),
            [self.command, variant],
        )

    def test_cuda_selection_reports_excluded_scope(self):
        with self.assertRaisesRegex(ValueError, "CUDA files are outside"):
            tidy.select_commands(
                [self.command], [self.source.with_suffix(".cu")], False
            )

    def test_shell_command_database_preserves_quoted_arguments(self):
        command = tidy.CompileCommand.parse(
            {
                "directory": str(self.build),
                "file": str(self.source),
                "command": "clang++ '-DNAME=two words' -o CMakeFiles/sirius_objects.dir/example.o -c example.cpp",
            }
        )
        self.assertIn("-DNAME=two words", command.arguments)
        self.assertEqual(command.output, self.command.output)

    def test_relative_database_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            tidy.CompileCommand.parse({"directory": "build"})

    def test_cli_usage_and_missing_database_fail(self):
        for arguments in [
            [],
            ["--all", "file.cpp"],
            ["-j", "0", "--all"],
            ["-p", str(self.build), "--all"],
        ]:
            with self.subTest(arguments=arguments), patch.object(
                sys, "argv", ["run_clang_tidy.py", *arguments]
            ):
                if arguments == ["-p", str(self.build), "--all"]:
                    self.assertEqual(tidy.main(), 2)
                else:
                    with self.assertRaises(SystemExit) as error:
                        tidy.main()
                    self.assertEqual(error.exception.code, 2)

    def test_scope_excludes_cuda_generated_and_optional_targets(self):
        tracked = {self.source}
        self.assertTrue(self.command.owned(tracked))
        self.assertFalse(self.command.owned(set()))
        generated = tidy.CompileCommand(
            self.build / "generated.c", self.build, (), self.command.output
        )
        self.assertFalse(generated.owned(tracked))
        optional = tidy.CompileCommand(
            self.source,
            self.build,
            (),
            Path("CMakeFiles/parquet_benchmark.dir/example.o"),
        )
        self.assertFalse(optional.owned(tracked))
        cuda = tidy.CompileCommand(
            self.source.with_suffix(".cu"), self.build, (), self.command.output
        )
        self.assertFalse(cuda.owned({cuda.source}))

    def prepare(self):
        (self.build / "compile_commands.json").write_text(
            json.dumps([self.command.as_json()])
        )
        for name in [
            "rust/crates/telemetry/bridge/include/rust/cxx.h",
            "rust/crates/telemetry/bridge/include/telemetry-bridge/gen/quent.hpp",
            "simpatico_codegen/generated/codegen/jit/embedded_headers.h",
        ]:
            path = self.build / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        tool = self.root / "clang-tidy"
        tool.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "database = pathlib.Path(sys.argv[sys.argv.index('-p') + 1]) / 'compile_commands.json'\n"
            "print(database.read_text())\n"
            "print('advisory finding')\n"
            "assert '--warnings-as-errors=-*' in sys.argv\n"
            "assert any(arg.startswith('--header-filter=^') for arg in sys.argv)\n"
            "sys.exit(int(os.environ.get('TIDY_TEST_EXIT', '0')))\n"
        )
        tool.chmod(0o755)
        return tool

    def test_warning_success_and_compiler_failure_reach_caller(self):
        tool = self.prepare()
        args = argparse.Namespace(
            build_dir=self.build, files=[self.source], all=False, jobs=2
        )
        for status in [0, 1]:
            with self.subTest(status=status), patch.object(
                tidy, "ROOT", self.root
            ), patch.object(
                tidy.subprocess, "check_output", return_value=b"src/example.cpp\0"
            ), patch.object(
                tidy.shutil, "which", return_value=str(tool)
            ), patch.dict(
                os.environ, {"TIDY_TEST_EXIT": str(status)}
            ), redirect_stdout(
                io.StringIO()
            ) as output:
                self.assertEqual(tidy.run(args), status)
                self.assertIn("-std=c++20", output.getvalue())
                self.assertIn("advisory finding", output.getvalue())

    def test_missing_generated_headers_fail_before_analysis(self):
        self.prepare()
        (
            self.build
            / "rust/crates/telemetry/bridge/include/telemetry-bridge/gen/quent.hpp"
        ).unlink()
        args = argparse.Namespace(
            build_dir=self.build, files=[self.source], all=False, jobs=1
        )
        with patch.object(tidy, "ROOT", self.root), patch.object(
            tidy.subprocess, "check_output", return_value=b"src/example.cpp\0"
        ), self.assertRaisesRegex(ValueError, "missing generated headers"):
            tidy.run(args)


if __name__ == "__main__":
    unittest.main()
