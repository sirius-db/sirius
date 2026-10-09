"""Production cache regressions. Every worker receives its environment before exec."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

WORKERS = {}
XXHSUM = "xxhsum"


def checksum(content):
    result = subprocess.run(
        [XXHSUM, "-H128"], input=content, capture_output=True, check=True
    )
    return bytes.fromhex(result.stdout.decode().split()[0])


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
        self.assertRegex(paths[0].parent.name, r"^[0-9a-f]{32}$")
        self.assertRegex(paths[0].name, r"^[0-9a-f]{32}\.cubin$")
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

    def test_removed_override_does_not_change_reuse(self):
        headers = self.root / "headers"
        headers.mkdir()
        settings = {"SIMPATICO_JIT_CCCL_INCLUDE": str(headers)}
        (headers / "codegen").mkdir()
        (headers / "codegen/stdint_shim.hpp").write_text(
            "#error should not be selected\n"
        )
        self.assertEqual(self.worker(**settings), (1, 1, 0))
        self.assertEqual(self.worker(**settings), (0, 1, 1))
        self.assertEqual(len(list(self.cache.glob("v2/*/*.cubin"))), 1)
        self.assertEqual(self.worker(), (0, 1, 1))

    def test_removed_override_cannot_add_headers(self):
        headers = self.root / "headers"
        headers.mkdir()
        (headers / "cache_fallback.cuh").write_text("#define CACHE_FALLBACK_VALUE 41\n")
        self.assert_external_rejected(SIMPATICO_JIT_CCCL_INCLUDE=str(headers))

    def compiler_path(self, **settings):
        result = subprocess.run(
            [WORKERS["normal"], "compiler-path"],
            env=self.environment | settings,
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        return Path(
            re.search(r"^NVRTC (.+)$", result.stdout, re.MULTILINE).group(1)
        ).resolve()

    def test_actual_shared_compiler_artifacts_and_relocated_prefix(self):
        path = self.compiler_path()
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
        self.assertEqual(
            self.compiler_path(**settings),
            relocated / path.name,
            "worker must load the relocated NVRTC library before testing its identity",
        )
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
        self.assert_external_rejected()

    def assert_external_rejected(self, **settings):
        result = subprocess.run(
            [WORKERS["normal"], "external"],
            env=self.environment | settings,
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

    def test_corrupt_truncated_and_unloadable_records_are_repaired(self):
        self.assertEqual(self.worker(), (1, 1, 0))
        path = next(self.cache.glob("v2/*/*.cubin"))
        original = path.read_bytes()
        self.assertTrue(original.startswith(b"SIMPJIT3"))
        self.assertEqual(len(original), 32 + int.from_bytes(original[8:16], "big"))
        self.assertEqual(original[16:32], checksum(original[32:]))
        changed = bytearray(original)
        changed[-1] ^= 1
        # Valid storage record containing an invalid CUDA image: tests the
        # driver's rejection path independently of the storage checksum.
        invalid_image = bytes(4096)
        unloadable = (
            b"SIMPJIT3"
            + len(invalid_image).to_bytes(8, "big")
            + checksum(invalid_image)
            + invalid_image
        )
        for damaged in (b"", original[:100], changed, unloadable):
            path.write_bytes(damaged)
            self.assertEqual(self.worker(), (1, 1, 0))
            self.assertEqual(self.worker(), (0, 1, 1))

    def test_simultaneous_processes_and_threads(self):
        with ThreadPoolExecutor(max_workers=4) as workers:
            results = list(workers.map(lambda _: self.worker(mode="threads"), range(4)))
        self.assertGreaterEqual(sum(compiles for compiles, _, _ in results), 1)
        self.assertEqual(self.worker(), (0, 1, 1))
        self.assertEqual(len(list(self.cache.glob("v2/*/*.cubin"))), 1)
        self.assertEqual(list(self.cache.rglob("*.tmp.*")), [])

    def test_legacy_ignored_and_explicit_cleanup(self):
        self.cache.mkdir()
        legacy = self.cache / "0123456789abcdef_a120_c13030_d13030.cubin"
        legacy.write_bytes(b"not a valid old cubin")
        unrelated = self.cache / "unrelated.cubin"
        unrelated.write_text("preserve me")
        self.assertEqual(self.worker(), (1, 1, 0))
        self.assertTrue(legacy.exists())
        self.assertEqual(self.worker("project", expected=2), (1, 1, 0))
        paths = list(self.cache.glob("v2/*/*.cubin"))
        self.assertEqual(len(paths), 2)
        abandoned = Path(str(paths[0]) + ".tmp.Abc123")
        abandoned.write_text("interrupted publication")
        result = subprocess.run(
            [WORKERS["normal"], "clear"],
            env=self.environment,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(legacy.exists())
        self.assertFalse(abandoned.exists())
        self.assertEqual(list(self.cache.glob("v2/*/*.cubin")), [])
        self.assertEqual(unrelated.read_text(), "preserve me")
        self.assertEqual(self.worker(), (1, 1, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--project-worker", required=True)
    parser.add_argument("--cccl-worker", required=True)
    parser.add_argument("--xxhsum", default="xxhsum")
    arguments, remaining = parser.parse_known_args()
    XXHSUM = arguments.xxhsum
    WORKERS = {
        "normal": str(Path(arguments.worker).resolve()),
        "project": str(Path(arguments.project_worker).resolve()),
        "cccl": str(Path(arguments.cccl_worker).resolve()),
    }
    unittest.main(argv=[__file__, *remaining])
