#!/usr/bin/env python3
"""Exercise the real activation script without a Pixi environment or GPU."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/pixi_activate.sh"
BASH = shutil.which("bash")


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sirius activation ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "prefix with spaces"
        (self.prefix / "bin").mkdir(parents=True)
        self.compiler = self.prefix / "bin/clang-cpp"
        self.compiler.write_text("#!/bin/sh\nexit 0\n")
        self.compiler.chmod(0o755)
        self.alias = self.prefix / "bin/clang++"
        self.cwd = self.root / "work"
        self.cwd.mkdir()

    def activate(self, **overrides):
        env = os.environ.copy()
        for name in ("CONDA_PREFIX", "LIBCLANG_PATH", "BASH_ENV", "ENV"):
            env.pop(name, None)
        env.update(CONDA_PREFIX=str(self.prefix))
        for name, value in overrides.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        return subprocess.run(
            [
                BASH,
                "--noprofile",
                "--norc",
                "-c",
                'source "$1"; printf "%s\\n" "${LIBCLANG_PATH-}"',
                "bash",
                str(SCRIPT),
            ],
            cwd=self.cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def require_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_correct_alias_is_not_replaced_on_repeated_activation(self):
        self.alias.symlink_to(self.compiler)
        (self.cwd / "build").mkdir()
        before = self.alias.lstat()
        for _ in range(2):
            result = self.activate()
            self.require_success(result)
            self.assertEqual(result.stdout.strip(), str(self.prefix / "lib"))
            after = self.alias.lstat()
            self.assertEqual(
                (after.st_ino, after.st_mtime_ns, after.st_ctime_ns),
                (before.st_ino, before.st_mtime_ns, before.st_ctime_ns),
            )
            self.assertEqual(os.readlink(self.alias), str(self.compiler))

    def test_missing_alias_is_created_and_build_directory_initialized(self):
        self.require_success(self.activate())
        self.assertEqual(os.readlink(self.alias), str(self.compiler))
        self.assertTrue((self.cwd / "build").is_dir())

    def test_equivalent_relative_aliases_are_not_replaced(self):
        for target in ("clang-cpp", str(self.prefix / "bin/../bin/clang-cpp")):
            with self.subTest(target=target):
                if self.alias.is_symlink():
                    self.alias.unlink()
                self.alias.symlink_to(target)
                before = self.alias.lstat()
                self.require_success(self.activate())
                self.assertEqual(self.alias.lstat().st_ino, before.st_ino)
                self.assertEqual(os.readlink(self.alias), target)
                self.alias.unlink()

    def test_prepared_alias_and_build_need_no_external_initializer(self):
        self.alias.symlink_to(self.compiler)
        (self.cwd / "build").mkdir()
        # Bash is selected by its real absolute path; initializer tools are unavailable.
        self.require_success(self.activate(PATH=str(self.root / "empty-search-path")))

    def test_alias_repair_failure_stops_before_build_initialization(self):
        # Real ln refuses to overwrite this nested directory, including when run as root.
        occupied = self.alias / "clang-cpp"
        occupied.mkdir(parents=True)
        marker = occupied / "preserve"
        marker.write_text("directory content\n")
        result = self.activate()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertFalse((self.cwd / "build").exists())
        self.assertEqual(marker.read_text(), "directory content\n")

    def test_wrong_and_dangling_alias_are_repaired_without_changing_target(self):
        foreign = self.root / "other compiler"
        foreign.write_text("foreign bytes\n")
        for target in (foreign, self.root / "missing compiler"):
            with self.subTest(target=str(target)):
                if self.alias.is_symlink():
                    self.alias.unlink()
                self.alias.symlink_to(target)
                self.require_success(self.activate())
                self.assertEqual(os.readlink(self.alias), str(self.compiler))
                self.assertEqual(foreign.read_text(), "foreign bytes\n")

    def test_regular_alias_file_is_replaced_with_expected_symlink(self):
        self.alias.write_text("old compiler\n")
        self.require_success(self.activate())
        self.assertEqual(os.readlink(self.alias), str(self.compiler))

    def test_unavailable_compiler_does_not_replace_existing_alias(self):
        for unavailable in ("nonexecutable", "absent"):
            with self.subTest(compiler=unavailable):
                if unavailable == "nonexecutable":
                    self.compiler.chmod(0o644)
                else:
                    self.compiler.unlink()
                self.alias.write_text("preserve\n")
                self.require_success(self.activate())
                self.assertEqual(self.alias.read_text(), "preserve\n")

    def test_existing_libclang_path_is_preserved(self):
        result = self.activate(LIBCLANG_PATH="already selected/lib")
        self.require_success(result)
        self.assertEqual(result.stdout.strip(), "already selected/lib")

    def test_unset_prefix_exits_without_initialization(self):
        self.require_success(self.activate(CONDA_PREFIX=None))
        self.assertFalse(self.alias.exists())
        self.assertFalse((self.cwd / "build").exists())

    def test_existing_build_file_fails_instead_of_hiding_error(self):
        (self.cwd / "build").write_text("occupied\n")
        result = self.activate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build", result.stderr)
        self.assertEqual((self.cwd / "build").read_text(), "occupied\n")


if __name__ == "__main__":
    unittest.main()
