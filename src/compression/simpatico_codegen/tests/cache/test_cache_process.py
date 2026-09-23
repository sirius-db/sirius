"""Production cache regressions. Every worker receives its environment before exec."""

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

WORKERS = {}


class CacheProcesses(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="simpatico-cache-process-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.environment = dict(os.environ)
        self.environment.pop("SIMPATICO_JIT_CCCL_INCLUDE", None)
        self.environment.update(
            SIMPATICO_JIT_CACHE_DIR=str(self.cache),
            SIMPATICO_JIT_STATS="1",
            CUDA_CACHE_DISABLE="1",
            CUDA_CACHE_PATH=str(self.root / "cuda-cache"),
        )

    def worker(self, name="normal", mode="embedded", expected=1, **settings):
        result = subprocess.run(
            [WORKERS[name], mode],
            env=self.environment | settings,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        if mode != "cycle":
            self.assertIn(f"RESULT value={expected}", result.stdout)
        stats = re.search(
            r"compiles=(\d+) mem_hits=(\d+) disk_hits=(\d+)", result.stderr
        )
        self.assertIsNotNone(stats, result.stderr)
        return tuple(map(int, stats.groups()))

    def test_actual_disk_write_and_second_process_hit(self):
        self.assertEqual(self.worker(), (1, 1, 0))
        paths = list(self.cache.glob("v2/*/*.cubin"))
        self.assertEqual(len(paths), 1)
        original = paths[0].read_bytes()
        self.assertGreater(len(original), 0)
        self.assertEqual(self.worker(), (0, 1, 1))
        self.assertEqual(original, paths[0].read_bytes())

    def test_project_and_cccl_header_invalidation_through_production(self):
        self.assertEqual(self.worker(), (1, 1, 0))
        for variant, expected in (("project", 2), ("cccl", 3)):
            self.assertEqual(self.worker(variant, expected=expected), (1, 1, 0))
            self.assertEqual(self.worker(variant, expected=expected), (0, 1, 1))
        self.assertEqual(len(list(self.cache.glob("v2/*/*.cubin"))), 3)
        self.assertEqual(self.worker(), (0, 1, 1))

    def test_external_fallback_and_unsetting(self):
        self.assertEqual(self.worker(), (1, 1, 0))
        headers = self.root / "headers"
        headers.mkdir()
        fallback = headers / "cache_fallback.cuh"
        fallback.write_text("#define CACHE_FALLBACK_VALUE 41\n")
        settings = {"SIMPATICO_JIT_CCCL_INCLUDE": str(headers)}
        self.assertEqual(
            self.worker(mode="external", expected=41, **settings), (2, 0, 0)
        )
        fallback.write_text("#define CACHE_FALLBACK_VALUE 42\n")
        self.assertEqual(
            self.worker(mode="external", expected=42, **settings), (2, 0, 0)
        )
        # An external copy cannot override a named in-memory header.
        (headers / "codegen").mkdir()
        (headers / "codegen/stdint_shim.hpp").write_text(
            "#error should not be selected\n"
        )
        self.assertEqual(self.worker(**settings), (2, 0, 0))
        self.assertEqual(len(list(self.cache.glob("v2/*/*.cubin"))), 1)
        self.assertEqual(self.worker(), (0, 1, 1))

    def test_external_changes_within_one_process_preserve_old_handles(self):
        headers = self.root / "headers"
        headers.mkdir()
        (headers / "cache_fallback.cuh").write_text("#define CACHE_FALLBACK_VALUE 41\n")
        self.assertEqual(
            self.worker(mode="cycle", SIMPATICO_JIT_CCCL_INCLUDE=str(headers)),
            (3, 1, 0),
        )
        self.assertEqual(self.worker(), (0, 1, 1))

    def test_actual_shared_compiler_artifacts_and_relocated_prefix(self):
        result = subprocess.run(
            [WORKERS["normal"], "compiler-path"],
            env=self.environment,
            text=True,
            capture_output=True,
            check=True,
        )
        path = Path(
            re.search(r"^NVRTC (.+)$", result.stdout, re.MULTILINE).group(1)
        ).resolve()
        if not path.name.startswith("libnvrtc.so"):
            self.skipTest(
                "shared-library replacement test does not apply to static NVRTC"
            )
        relocated = self.root / "compiler"
        relocated.mkdir()
        for candidate in path.parent.glob("libnvrtc*.so*"):
            original = candidate.resolve()
            destination = relocated / original.name
            if not destination.exists():
                shutil.copyfile(original, destination)
            if candidate.name != original.name:
                (relocated / candidate.name).symlink_to(original.name)
        settings = {
            "LD_LIBRARY_PATH": str(relocated)
            + ":"
            + os.environ.get("LD_LIBRARY_PATH", "")
        }
        self.assertEqual(self.worker(), (1, 1, 0))
        self.assertEqual(self.worker(**settings), (0, 1, 1))
        # An ELF trailer does not change execution, but changes the real compiler
        # artifact. Keep its name/version unchanged to expose version-only keys.
        with (relocated / path.name).open("ab") as output:
            output.write(b"\0")
        self.assertEqual(self.worker(**settings), (1, 1, 0))
        self.assertEqual(self.worker(**settings), (0, 1, 1))
        builtin = next(
            candidate
            for candidate in relocated.glob("libnvrtc-builtins.so*")
            if not candidate.is_symlink()
        )
        with builtin.open("ab") as output:
            output.write(b"\0")
        self.assertEqual(self.worker(**settings), (1, 1, 0))
        self.assertEqual(len(list(self.cache.glob("v2/*/*.cubin"))), 3)
        self.assertEqual(self.worker(), (0, 1, 1))

    def test_untracked_cwd_headers_are_not_inputs(self):
        (self.root / "cache_fallback.cuh").write_text(
            "#define CACHE_FALLBACK_VALUE 99\n"
        )
        result = subprocess.run(
            [WORKERS["normal"], "external"],
            env=self.environment,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cache_fallback.cuh", result.stderr)
        self.assertEqual(list(self.cache.glob("v2/*/*.cubin")), [])

    def test_disabled_and_unwritable_cache(self):
        for value in ("off", "0", "", str(self.root / "file" / "child")):
            (self.root / "file").write_text("not a directory")
            self.assertEqual(self.worker(SIMPATICO_JIT_CACHE_DIR=value), (1, 1, 0))
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--project-worker", required=True)
    parser.add_argument("--cccl-worker", required=True)
    arguments, remaining = parser.parse_known_args()
    WORKERS = {
        "normal": str(Path(arguments.worker).resolve()),
        "project": str(Path(arguments.project_worker).resolve()),
        "cccl": str(Path(arguments.cccl_worker).resolve()),
    }
    unittest.main(argv=[__file__, *remaining])
