#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
"""Exercise the real loader with CUDA-free backend and driver fixtures."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duckdb", type=Path, required=True)
parser.add_argument("--build-dir", type=Path, required=True)
args, remaining = parser.parse_known_args()
args.duckdb = args.duckdb.resolve()
args.build_dir = args.build_dir.resolve()


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="sirius-loader-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.loader = args.build_dir / "extension/sirius/sirius.duckdb_extension"
        self.fixtures = args.build_dir / "extension/sirius/test"

    def backend(self, version):
        name = f"sirius_cuda{version}"
        return self.fixtures / f"{name}.duckdb_extension"

    def query(self, sql, version=13000, driver="driver", env=None, autoinstall=False):
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = (
            str(self.fixtures / driver)
            + os.pathsep
            + environment.get("LD_LIBRARY_PATH", "")
        )
        environment["SIRIUS_TEST_DRIVER_VERSION"] = str(version)
        environment.pop("SIRIUS_TEST_DRIVER_ERROR", None)
        environment.pop("SIRIUS_TEST_BACKEND_FAILURE", None)
        environment.update(env or {})
        prelude = (
            f"SET extension_directories=[{quote(self.root / 'installed')}];"
            f"SET autoinstall_known_extensions={'true' if autoinstall else 'false'};"
        )
        return subprocess.run(
            [str(args.duckdb), "-unsigned", "-json", "-c", prelude + sql],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def assert_failure(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message, result.stderr)
        self.assertNotIn('"exception_type"', result.stderr)

    def test_driver_boundaries_and_repeated_load(self):
        for version, backend in [
            (12000, 12),
            (12090, 12),
            (12999, 12),
            (13000, 13),
            (13020, 13),
            (14000, 13),
        ]:
            with self.subTest(version=version):
                result = self.query(
                    f"INSTALL {quote(self.backend(backend))};"
                    f"INSTALL {quote(self.loader)};"
                    "LOAD sirius; LOAD sirius;"
                    "SELECT sirius_test_backend() AS backend;",
                    version=version,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), [{"backend": backend}])

    def test_unsupported_driver(self):
        for version in [0, 11080, 11999]:
            with self.subTest(version=version):
                self.assert_failure(
                    self.query(f"LOAD {quote(self.loader)};", version=version),
                    "requires an NVIDIA driver supporting CUDA 12 or newer",
                )

    def test_driver_query_error(self):
        self.assert_failure(
            self.query(
                f"LOAD {quote(self.loader)};", env={"SIRIUS_TEST_DRIVER_ERROR": "999"}
            ),
            "cuDriverGetVersion failed with error 999",
        )

    def test_missing_version_symbol(self):
        self.assert_failure(
            self.query(f"LOAD {quote(self.loader)};", driver="missing-symbol"),
            "has no cuDriverGetVersion",
        )

    def test_missing_driver(self):
        self.assert_failure(
            self.query(
                f"LOAD {quote(self.loader)};",
                env={
                    "LD_PRELOAD": str(self.fixtures / "libsirius_test_hide_driver.so")
                },
            ),
            "cannot open libcuda.so.1",
        )

    def test_does_not_fall_back_to_cuda12(self):
        result = self.query(
            f"INSTALL {quote(self.backend(12))}; LOAD {quote(self.loader)};"
        )
        self.assert_failure(
            result, "Sirius selected sirius_cuda13 for CUDA driver version 13.0"
        )
        self.assertIn("not found", result.stderr)
        self.assertNotIn("automatically install", result.stderr)

    def test_selected_backend_already_loaded(self):
        result = self.query(
            f"LOAD {quote(self.backend(13))}; LOAD {quote(self.loader)};"
            "SELECT sirius_test_backend() AS backend;"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [{"backend": 13}])

    def test_backend_initialization_error(self):
        for autoinstall in [False, True]:
            with self.subTest(autoinstall=autoinstall):
                result = self.query(
                    f"INSTALL {quote(self.backend(13))}; LOAD {quote(self.loader)};",
                    env={"SIRIUS_TEST_BACKEND_FAILURE": "1"},
                    autoinstall=autoinstall,
                )
                self.assert_failure(result, "Backend initialization failed for test")
                self.assertIn("Sirius selected sirius_cuda13", result.stderr)
                self.assertNotIn("run INSTALL", result.stderr)
                self.assertNotIn("automatically install", result.stderr)

    def test_backends_reject_aliases_before_registration(self):
        for backend, other in [(12, 13), (13, 12)]:
            for alias in ["previous_backend", f"sirius_cuda{other}"]:
                with self.subTest(backend=backend, alias=alias):
                    result = self.query(
                        f"LOAD {quote(self.backend(backend))} AS {alias};",
                        env={"SIRIUS_TEST_BACKEND_FAILURE": "1"},
                    )
                    self.assert_failure(
                        result, f"Cannot load sirius_cuda{backend} under an alias"
                    )
                    self.assertNotIn(
                        "Backend initialization failed for test", result.stderr
                    )

    def test_loader_rejects_backend_names(self):
        for backend in [12, 13]:
            with self.subTest(backend=backend):
                self.assert_failure(
                    self.query(
                        f"LOAD {quote(self.loader)} AS sirius_cuda{backend};",
                        version=backend * 1000,
                    ),
                    "Cannot load sirius under a CUDA backend name",
                )

    def test_conflicting_loaded_backend(self):
        self.assert_failure(
            self.query(f"LOAD {quote(self.backend(12))}; LOAD {quote(self.loader)};"),
            "Cannot load sirius_cuda13 while sirius_cuda12 is already loaded",
        )

    def test_auto_install_from_repository(self):
        result = self.query(
            "SELECT library_version, source_id, platform FROM pragma_version(), pragma_platform();"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        host = json.loads(result.stdout)[0]
        version = (
            host["source_id"]
            if "dev" in host["library_version"]
            else host["library_version"]
        )
        repository = self.root / "repository"
        destination = repository / version / host["platform"] / self.backend(13).name
        destination.parent.mkdir(parents=True)
        shutil.copyfile(self.backend(13), destination)
        result = self.query(
            f"SET autoinstall_extension_repository={quote(repository)};"
            f"LOAD {quote(self.loader)}; SELECT sirius_test_backend() AS backend;",
            autoinstall=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [{"backend": 13}])
        self.assertTrue(
            list((self.root / "installed").rglob("sirius_cuda13.duckdb_extension"))
        )

    def test_loader_has_no_cuda_or_engine_dependency(self):
        result = subprocess.run(
            ["readelf", "-d", str(self.loader)],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if "(NEEDED)" in line:
                self.assertNotRegex(line, r"lib(cuda|cudart|cudf|cuvs|rmm|sirius)")


unittest.main(argv=[__file__, *remaining])
