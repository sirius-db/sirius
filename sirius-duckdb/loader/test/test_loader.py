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
        environment.pop("SIRIUS_DISABLE", None)
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

    def rows(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def status(self, sql, **kwargs):
        return self.rows(self.query(sql + "SELECT * FROM sirius_status();", **kwargs))[
            0
        ]

    def assert_fallback(self, sql, reason, **kwargs):
        """LOAD succeeds, queries run on the CPU, and explicit GPU requests fail with the reason."""
        status = self.status(sql, **kwargs)
        self.assertFalse(status["backend_loaded"])
        self.assertFalse(status["gpu_execution"])
        self.assertIn(reason, status["fallback_reason"])
        self.assertEqual(
            self.rows(
                self.query(
                    sql
                    + "SET gpu_execution = false;"
                    + "SELECT sum(i)::BIGINT AS total, current_setting('gpu_execution') AS enabled"
                    + " FROM range(1000) t(i);",
                    **kwargs,
                )
            ),
            [{"total": 499500, "enabled": False}],
        )
        for request in [
            "SET gpu_execution = true;",
            "SET GLOBAL gpu_execution = true;",
            "SELECT * FROM gpu_execution('SELECT 42');",
        ]:
            with self.subTest(request=request):
                self.assert_failure(self.query(sql + request, **kwargs), reason)
        return status

    def test_driver_boundaries_and_repeated_load(self):
        for version, backend in [
            (12080, 12),
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
                    "SELECT sirius_test_backend() AS test_backend, s.*"
                    " FROM sirius_status() s;",
                    version=version,
                )
                self.assertEqual(
                    self.rows(result),
                    [
                        {
                            "test_backend": backend,
                            "driver_found": True,
                            "cuda_driver_version": f"{version // 1000}.{version % 1000 // 10}",
                            "backend": f"sirius_cuda{backend}",
                            "backend_loaded": True,
                            "gpu_execution": True,
                            "fallback_reason": None,
                        }
                    ],
                )

    def test_status_reports_disabled_gpu_execution(self):
        status = self.status(
            f"INSTALL {quote(self.backend(13))}; LOAD {quote(self.loader)};"
            "SET gpu_execution = false;"
        )
        self.assertTrue(status["backend_loaded"])
        self.assertFalse(status["gpu_execution"])

    def test_status_reports_disabled_runtime(self):
        status = self.status(
            f"INSTALL {quote(self.backend(13))}; LOAD {quote(self.loader)};",
            env={"SIRIUS_DISABLE": "1"},
        )
        self.assertTrue(status["backend_loaded"])
        self.assertFalse(status["gpu_execution"])
        self.assertIsNone(status["fallback_reason"])

    def test_prepared_status_reads_current_setting(self):
        result = self.query(
            f"INSTALL {quote(self.backend(13))}; LOAD {quote(self.loader)};"
            "SET gpu_execution = false;"
            "PREPARE status AS SELECT gpu_execution FROM sirius_status();"
            "SET gpu_execution = true;"
            "EXECUTE status;"
        )
        self.assertEqual(self.rows(result), [{"gpu_execution": True}])

    def test_loaded_backend_owns_gpu_execution(self):
        result = self.query(
            f"INSTALL {quote(self.backend(13))}; LOAD {quote(self.loader)};"
            "SELECT count(*) AS stubs FROM duckdb_functions()"
            " WHERE function_name = 'gpu_execution';"
        )
        self.assertEqual(self.rows(result), [{"stubs": 0}])

    def test_unsupported_driver(self):
        for version in [0, 11080, 11999, 12000, 12070]:
            with self.subTest(version=version):
                status = self.assert_fallback(
                    f"LOAD {quote(self.loader)};",
                    "Sirius requires an NVIDIA driver supporting CUDA 12.8 or newer: "
                    f"reported CUDA version {version // 1000}.{version % 1000 // 10}",
                    version=version,
                )
                self.assertTrue(status["driver_found"])
                self.assertEqual(
                    status["cuda_driver_version"],
                    f"{version // 1000}.{version % 1000 // 10}",
                )
                self.assertIsNone(status["backend"])

    def test_driver_query_error(self):
        status = self.assert_fallback(
            f"LOAD {quote(self.loader)};",
            "cuDriverGetVersion failed with error 999",
            env={"SIRIUS_TEST_DRIVER_ERROR": "999"},
        )
        self.assertTrue(status["driver_found"])
        self.assertIsNone(status["cuda_driver_version"])

    def test_missing_version_symbol(self):
        self.assert_fallback(
            f"LOAD {quote(self.loader)};",
            "has no cuDriverGetVersion",
            driver="missing-symbol",
        )

    def test_missing_driver(self):
        status = self.assert_fallback(
            f"LOAD {quote(self.loader)};",
            "Sirius requires an NVIDIA driver supporting CUDA 12.8 or newer: "
            "cannot open libcuda.so.1",
            env={"LD_PRELOAD": str(self.fixtures / "libsirius_test_hide_driver.so")},
        )
        self.assertEqual(
            {k: status[k] for k in ["driver_found", "cuda_driver_version", "backend"]},
            {"driver_found": False, "cuda_driver_version": None, "backend": None},
        )

    def test_fallback_is_silent_and_logged(self):
        hide = {"LD_PRELOAD": str(self.fixtures / "libsirius_test_hide_driver.so")}
        result = self.query(f"LOAD {quote(self.loader)};", env=hide)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""))
        result = self.query(
            "CALL enable_logging(level = 'info', storage = 'memory');"
            f"LOAD {quote(self.loader)};"
            "SELECT message FROM duckdb_logs WHERE message LIKE '%Sirius%';",
            env=hide,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        # The CLI first prints an empty result for CALL enable_logging.
        [entry] = json.loads(result.stdout.splitlines()[-1])
        self.assertRegex(
            entry["message"],
            r"^Sirius GPU execution is unavailable; queries run on DuckDB's CPU engine\. "
            r"Sirius requires an NVIDIA driver supporting CUDA 12.8 or newer: cannot open",
        )

    def test_does_not_fall_back_to_cuda12(self):
        status = self.assert_fallback(
            f"INSTALL {quote(self.backend(12))}; LOAD {quote(self.loader)};",
            "Sirius selected sirius_cuda13 for CUDA driver version 13.0",
        )
        self.assertEqual(status["backend"], "sirius_cuda13")
        self.assertIn("not found", status["fallback_reason"])
        self.assertNotIn("automatically install", status["fallback_reason"])
        self.assert_failure(
            self.query(
                f"INSTALL {quote(self.backend(12))}; LOAD {quote(self.loader)};"
                "SELECT sirius_test_backend();"
            ),
            "sirius_test_backend does not exist",
        )

    def test_unavailable_repository(self):
        load = (
            f"SET autoinstall_extension_repository={quote(self.root / 'missing')};"
            f"LOAD {quote(self.loader)};"
        )
        status = self.assert_fallback(
            load,
            "Sirius selected sirius_cuda13 for CUDA driver version 13.0",
            autoinstall=True,
        )
        self.assertEqual(status["backend"], "sirius_cuda13")
        result = self.query(load, autoinstall=True)
        self.assertEqual((result.returncode, result.stdout), (0, ""))
        self.assertRegex(
            result.stderr,
            r"^Sirius: installing sirius_cuda13\. .*\n"
            r"Sirius: installing sirius_cuda13 failed after \d+\.\d s\.\n"
            r"Sirius: queries run on DuckDB's CPU engine; .*sirius_status\(\).*\n$",
        )

    def test_selected_backend_already_loaded(self):
        result = self.query(
            f"LOAD {quote(self.backend(13))}; LOAD {quote(self.loader)};"
            "SELECT sirius_test_backend() AS backend, backend_loaded FROM sirius_status();"
        )
        self.assertEqual(self.rows(result), [{"backend": 13, "backend_loaded": True}])

    def test_backend_initialization_error(self):
        # Once the backend starts initializing, a failure is reported rather than hidden.
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
        self.assertRegex(
            result.stderr,
            r"^Sirius: installing sirius_cuda13\. .*\n"
            r"Sirius: installed sirius_cuda13 in \d+\.\d s\.\n$",
        )
        self.assertTrue(
            list((self.root / "installed").rglob("sirius_cuda13.duckdb_extension"))
        )
        # Installed backends load without a message.
        result = self.query(
            f"SET autoinstall_extension_repository={quote(repository)};"
            f"LOAD {quote(self.loader)}; SELECT sirius_test_backend() AS backend;",
            autoinstall=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

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
