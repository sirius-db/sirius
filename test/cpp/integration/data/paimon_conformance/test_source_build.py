"""Source build acceptance tests; these doubles do not qualify native behavior."""

import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import build_extension as builder
from corpus_checks import digest


class SourceBuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.cli = self.root / "duckdb"
        self.cli.write_bytes(b"test CLI")
        self.extension = self.root / "paimon.duckdb_extension"
        self.extension.write_bytes(b"test extension")
        self.recipe = builder.read_recipe()
        self.identity = {
            "version": self.recipe["duckdb_version"],
            "platform": self.recipe["platform"],
        }
        self.receipt = {
            "format_version": 1,
            "state": "built",
            "recipe_sha256": digest(builder.RECIPE),
            "builder_sha256": digest(Path(builder.__file__)),
            "pixi_lock_sha256": digest(builder.ROOT / "pixi.lock"),
            "sources": self.recipe["sources"],
            "cmake_options": self.recipe["cmake_options"],
            "duckdb_identity": self.identity,
            "duckdb_sha256": digest(self.cli),
            "duckdb_source_id": self.recipe["sources"]["duckdb"]["commit"][:8],
            "toolchain": {key: "test tool" for key in ("cc", "cxx", "cmake", "ninja")},
            "build_environment": {key: "" for key in builder.BUILD_ENVIRONMENT_KEYS},
            "sha256": digest(self.extension),
            "size_bytes": self.extension.stat().st_size,
        }
        self.receipt_path = self.root / "build-receipt.json"

    def verify(self, receipt):
        self.receipt_path.write_text(json.dumps(receipt))
        return builder.validate_receipt(
            self.receipt_path, self.extension, self.cli, self.identity
        )

    def test_receipt_identifies_built_output_without_claiming_qualification(self):
        result = self.verify(self.receipt)
        self.assertEqual(result["kind"], "source_build")
        self.assertEqual(result["sha256"], digest(self.extension))
        self.assertEqual(result["version"], self.recipe["extension_version"])

    def test_missing_stale_and_incomplete_receipts_fail(self):
        with self.assertRaises(FileNotFoundError):
            builder.validate_receipt(
                self.receipt_path, self.extension, self.cli, self.identity
            )
        for key in self.receipt:
            with self.subTest(key=key):
                receipt = copy.deepcopy(self.receipt)
                receipt.pop(key)
                with self.assertRaises(ValueError):
                    self.verify(receipt)
        for key, bad in (
            ("recipe_sha256", "0" * 64),
            ("builder_sha256", "0" * 64),
            ("pixi_lock_sha256", "0" * 64),
            ("state", "failed"),
            ("duckdb_identity", {"version": "v1.5.5", "platform": "linux_amd64"}),
            ("duckdb_source_id", "deadbeef"),
            ("sources", {}),
            ("cmake_options", {}),
            ("toolchain", {}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify({**self.receipt, key: bad})

    def test_modified_binary_and_runtime_fail(self):
        self.extension.write_bytes(b"tampered extension")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            self.verify(self.receipt)
        self.extension.write_bytes(b"test extension")
        self.cli.write_bytes(b"different CLI")
        with self.assertRaisesRegex(ValueError, "duckdb_sha256"):
            self.verify(self.receipt)

    def test_source_id_accepts_real_git_abbreviations_only(self):
        commit = self.recipe["sources"]["duckdb"]["commit"]
        for count in (7, 8, 10, 40):
            self.assertTrue(builder.matches_source_id(commit[:count], commit))
        for wrong in (None, "", commit[:6], "deadbeef", commit + "0", "../" + commit):
            self.assertFalse(builder.matches_source_id(wrong, commit))

    def test_source_fetch_checks_hash_and_retries_only_transport_errors(self):
        target = self.root / "source.tar.gz"
        record = {
            "url": "https://example.com/source",
            "sha256": digest(self.extension),
            "size_bytes": self.extension.stat().st_size,
        }
        fetch = Mock(
            side_effect=[
                URLError("interrupted"),
                io.BytesIO(self.extension.read_bytes()),
            ]
        )
        builder.fetch_source(record, target, opener=fetch, delay=Mock())
        self.assertEqual(fetch.call_count, 2)
        for response, error in (
            (io.BytesIO(b"x" * record["size_bytes"]), ValueError),
            (
                HTTPError(record["url"], 404, "missing", {}, None),
                builder.DownloadFailure,
            ),
        ):
            fetch = (
                Mock(side_effect=response)
                if isinstance(response, Exception)
                else Mock(return_value=response)
            )
            with self.assertRaises(error):
                builder.fetch_source(record, target, opener=fetch, delay=Mock())
            self.assertEqual(fetch.call_count, 1)
        fetch = Mock(side_effect=URLError("offline"))
        with self.assertRaises(builder.DownloadFailure):
            builder.fetch_source(record, target, opener=fetch, delay=Mock())
        self.assertEqual(fetch.call_count, 3)

    def test_archive_paths_cannot_escape_staging(self):
        archive = self.root / "unsafe.tar"
        with tarfile.open(archive, "w") as output:
            info = tarfile.TarInfo("../../escaped")
            info.size = 1
            output.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(tarfile.FilterError):
            builder.extract_source(archive, self.root / "source")
        self.assertFalse((self.root / "source").exists())

    def exercise_build(
        self, output, fail=False, mutate=None, source_cache=None, real_child=False
    ):
        actual_run = subprocess.run
        actual_logged_run = builder.run

        def probe_or_run(args, **kwargs):
            if args[0] == str(self.cli):
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout=json.dumps(
                        [{"source_id": self.receipt["duckdb_source_id"]}]
                    ),
                )
            return actual_run(args, **kwargs)

        def extract(_archive, destination):
            destination.mkdir()
            if destination.name in ("extension", "native"):
                (destination / "third_party").mkdir()
            if destination.name == "native":
                (destination / "third_party/versions.txt").write_text("")

        def command(args, _log, _env):
            if "--build" in args:
                path = output / "build/extension/paimon/paimon.duckdb_extension"
                path.parent.mkdir(parents=True)
                if real_child:
                    actual_logged_run(
                        [
                            sys.executable,
                            "-c",
                            f"from pathlib import Path; import sys; Path({str(path)!r}).write_bytes(b'partial'); print('compiler failed after partial output'); sys.exit(1)",
                        ],
                        _log,
                        _env,
                    )
                    return
                path.write_bytes(b"partially or fully built")
                if fail:
                    raise subprocess.CalledProcessError(1, args)
                if mutate:
                    mutate()

        with (
            patch.object(builder.platform, "system", return_value="Linux"),
            patch.object(builder.platform, "machine", return_value="x86_64"),
            patch.dict(
                os.environ,
                CONDA_PREFIX=str(self.root),
                CC=sys.executable,
                CXX=sys.executable,
            ),
            patch("run_conformance.probe_identity", return_value=self.identity),
            patch.object(
                builder.subprocess,
                "run",
                side_effect=probe_or_run,
            ),
            patch.object(
                builder.subprocess, "check_output", return_value="test toolchain"
            ),
            patch.object(builder, "fetch_source"),
            patch.object(builder, "extract_source", side_effect=extract),
            patch.object(builder, "run", side_effect=command),
        ):
            builder.build(self.cli, output, 2, source_cache)

    def test_build_failure_never_publishes_receipt_and_reuse_is_refused(self):
        output = self.root / "failed-build"
        with self.assertRaises(subprocess.CalledProcessError):
            self.exercise_build(output, fail=True)
        self.assertFalse((output / "build-receipt.json").exists())
        self.assertFalse((output / "paimon.duckdb_extension").exists())
        with self.assertRaises(FileExistsError):
            self.exercise_build(output)

    def test_success_receipt_only_after_completed_build(self):
        output = self.root / "success-build"
        self.exercise_build(output)
        result = builder.validate_receipt(
            output / "build-receipt.json",
            output / "paimon.duckdb_extension",
            self.cli,
            self.identity,
        )
        self.assertEqual(result["sha256"], digest(output / "paimon.duckdb_extension"))

    def test_corrupt_source_cache_fails_before_configure(self):
        cached = self.root / "cache"
        cached.mkdir()
        source = self.recipe["sources"]["duckdb"]
        (cached / (source["sha256"] + ".tar.gz")).write_bytes(b"corrupt source")
        output = self.root / "corrupt-cache-build"
        with self.assertRaisesRegex(ValueError, "Cached source archive"):
            self.exercise_build(output, source_cache=cached)
        self.assertFalse((output / "build-receipt.json").exists())

    def test_inputs_changed_during_build_cannot_receive_success_receipt(self):
        for key in ("cli", "recipe", "builder", "lock"):
            with self.subTest(key=key):
                recipe = self.root / (key + "-recipe.json")
                recipe.write_text(json.dumps(self.recipe))
                lock_root = self.root / (key + "-root")
                lock_root.mkdir()
                (lock_root / "pixi.lock").write_text("test lock")
                script = self.root / (key + "-builder.py")
                script.write_text("test builder")
                path = {
                    "cli": self.cli,
                    "recipe": recipe,
                    "builder": script,
                    "lock": lock_root / "pixi.lock",
                }[key]
                output = self.root / (key + "-mutated-build")
                with patch.object(builder, "RECIPE", recipe), patch.object(
                    builder, "ROOT", lock_root
                ), patch.object(builder, "__file__", str(script)):
                    with self.assertRaisesRegex(
                        ValueError, "changed during source build"
                    ):
                        self.exercise_build(
                            output, mutate=lambda: path.write_bytes(b"changed")
                        )
                self.assertFalse((output / "build-receipt.json").exists())

    def test_actual_nonzero_build_child_cannot_publish_partial_output(self):
        output = self.root / "real-failing-build"
        with self.assertRaises(subprocess.CalledProcessError):
            self.exercise_build(output, real_child=True)
        self.assertFalse((output / "build-receipt.json").exists())
        self.assertFalse((output / "paimon.duckdb_extension").exists())
        self.assertIn(
            "compiler failed after partial output", (output / "build.log").read_text()
        )

    def test_native_source_cache_uses_only_pinned_hashes(self):
        native = self.root / "native"
        (native / "third_party").mkdir(parents=True)
        cache = self.root / "native-cache"
        cache.mkdir()
        checksum = digest(self.extension)
        versions = native / "third_party/versions.txt"
        versions.write_text(
            "PAIMON_TEST_BUILD_VERSION=1.0\n"
            f"PAIMON_TEST_BUILD_SHA256_CHECKSUM={checksum}\n"
            "PAIMON_TEST_PKG_NAME=test-${PAIMON_TEST_BUILD_VERSION}.tar.gz\n"
        )
        (cache / "unlisted.tar.gz").write_bytes(b"unlisted")
        builder.populate_dependency_cache(native, cache)
        self.assertFalse((native / "third_party/test-1.0.tar.gz").exists())
        cached = cache / (checksum + ".tar.gz")
        cached.write_bytes(self.extension.read_bytes())
        builder.populate_dependency_cache(native, cache)
        self.assertEqual(
            (native / "third_party/test-1.0.tar.gz").read_bytes(),
            self.extension.read_bytes(),
        )
        cached.write_bytes(b"corrupt cached source")
        with self.assertRaisesRegex(ValueError, "Cached native dependency"):
            builder.populate_dependency_cache(native, cache)
