"""Source build acceptance tests; these doubles do not qualify native behavior."""

import copy
import io
import json
import os
from pathlib import Path
import shlex
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
            "downloader_sha256": digest(builder.DOWNLOADER),
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
        for key in ("SCCACHE_RECACHE", "CCACHE_DISABLE", "GIT_CEILING_DIRECTORIES"):
            receipt = copy.deepcopy(self.receipt)
            del receipt["build_environment"][key]
            with self.subTest(missing_cache_control=key), self.assertRaisesRegex(
                ValueError, "lacks build environment provenance"
            ):
                self.verify(receipt)
        for key, bad in (
            ("recipe_sha256", "0" * 64),
            ("builder_sha256", "0" * 64),
            ("downloader_sha256", "0" * 64),
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
        self,
        output,
        fail=False,
        mutate=None,
        source_cache=None,
        real_child=False,
        build_check=None,
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
                if build_check:
                    build_check(_env)
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
        with patch.dict(os.environ, SCCACHE_RECACHE="1", CCACHE_DISABLE="1"):
            self.exercise_build(output)
        receipt = json.loads((output / "build-receipt.json").read_text())
        for key in ("SCCACHE_RECACHE", "CCACHE_DISABLE"):
            self.assertEqual(receipt["build_environment"][key], "1")
        result = builder.validate_receipt(
            output / "build-receipt.json",
            output / "paimon.duckdb_extension",
            self.cli,
            self.identity,
        )
        self.assertEqual(result["sha256"], digest(output / "paimon.duckdb_extension"))

    def test_archive_patches_apply_inside_an_unrelated_git_checkout(self):
        subprocess.run(
            ["git", "init", "--quiet", str(self.root)],
            env={
                key: value
                for key, value in os.environ.items()
                if not key.startswith("GIT_")
            },
            check=True,
        )
        parent_file = self.root / "CMakeLists.txt"
        parent_file.write_text("parent must stay unchanged\n")
        patch_file = self.root / "install.patch"
        patch_file.write_text(
            "diff --git a/CMakeLists.txt b/CMakeLists.txt\n"
            "--- a/CMakeLists.txt\n+++ b/CMakeLists.txt\n"
            "@@ -1 +1 @@\n-before\n+after\n"
        )
        for overrides in (False, True):
            with self.subTest(inherited_git_overrides=overrides):
                output = self.root / ("build-overrides" if overrides else "build")

                def check(env):
                    native = output / "extension/third_party/paimon-cpp"
                    cmake = native / "CMakeLists.txt"
                    cmake.write_text("before\n")
                    command = f"git -C {shlex.quote(str(native))} apply "
                    quoted_patch = shlex.quote(str(patch_file))
                    # Same idempotent patch decision as the pinned extension.
                    subprocess.run(
                        [
                            "sh",
                            "-c",
                            f"{command}--reverse --check {quoted_patch} "
                            f"2>/dev/null || {command}{quoted_patch}",
                        ],
                        env=env,
                        check=True,
                    )
                    self.assertEqual(cmake.read_text(), "after\n")
                    self.assertEqual(
                        parent_file.read_text(), "parent must stay unchanged\n"
                    )

                inherited = (
                    {
                        "GIT_DIR": str(self.root / ".git"),
                        "GIT_WORK_TREE": str(self.root),
                        "GIT_CEILING_DIRECTORIES": "/",
                    }
                    if overrides
                    else {}
                )
                with patch.dict(os.environ, inherited):
                    self.exercise_build(output, build_check=check)
                receipt = json.loads((output / "build-receipt.json").read_text())
                self.assertEqual(
                    receipt["build_environment"]["GIT_CEILING_DIRECTORIES"], str(output)
                )

    def test_ambiguous_git_ceiling_path_fails_before_build(self):
        output = self.root / "ambiguous:build"
        with self.assertRaisesRegex(ValueError, "Git ceiling separator"):
            self.exercise_build(output)
        self.assertFalse(output.exists())

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
        for key in ("cli", "recipe", "builder", "downloader", "lock"):
            with self.subTest(key=key):
                recipe = self.root / (key + "-recipe.json")
                recipe.write_text(json.dumps(self.recipe))
                lock_root = self.root / (key + "-root")
                lock_root.mkdir()
                (lock_root / "pixi.lock").write_text("test lock")
                script = self.root / (key + "-builder.py")
                script.write_text("test builder")
                downloader = self.root / (key + "-downloader.py")
                downloader.write_text("test downloader")
                path = {
                    "cli": self.cli,
                    "recipe": recipe,
                    "builder": script,
                    "downloader": downloader,
                    "lock": lock_root / "pixi.lock",
                }[key]
                output = self.root / (key + "-mutated-build")
                with patch.object(builder, "RECIPE", recipe), patch.object(
                    builder, "ROOT", lock_root
                ), patch.object(builder, "__file__", str(script)), patch.object(
                    builder, "DOWNLOADER", downloader
                ), patch.object(
                    builder, "validate_checkout_recipe"
                ):

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


class HostCompatibilityTests(unittest.TestCase):
    """Exercise the real entry point with a real Git tree and an identity-only CLI."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.recipe = builder.read_recipe()
        self.here = self.root / "test/cpp/integration/data/paimon_conformance"
        self.here.mkdir(parents=True)
        for source in builder.HERE.glob("*.py"):
            (self.here / source.name).write_bytes(source.read_bytes())
        (self.here / "source-build.json").write_bytes(builder.RECIPE.read_bytes())
        (self.root / "pixi.lock").write_text("test lock")
        self.git("init", "--quiet")
        self.commit_pins(self.recipe["sources"]["duckdb"]["commit"])
        self.trace = self.root / "cli-invocations"
        self.cli = self.root / "duckdb"
        self.write_cli()
        self.output = self.root / "build"

    def git(self, *args):
        return subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                *args,
            ],
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        )

    def commit_pins(self, commit, wrapper=None):
        for path, pin in (
            ("duckdb", commit),
            ("sirius-duckdb/duckdb", wrapper or commit),
        ):
            self.git("update-index", "--add", "--cacheinfo", f"160000,{pin},{path}")
        self.git("commit", "--quiet", "--allow-empty", "-m", "test host pins")

    def write_cli(self, version=None, source_id=None):
        version = version or self.recipe["duckdb_version"]
        source_id = source_id or self.recipe["sources"]["duckdb"]["commit"][:10]
        self.cli.write_text(
            f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
            f"with Path({str(self.trace)!r}).open('a') as f: f.write(repr(sys.argv)+'\\n')\n"
            "sql = sys.argv[sys.argv.index('-c') + 1]\n"
            "if sql == 'SELECT version() AS version; PRAGMA platform;':\n"
            f" print(json.dumps([{{'version':{version!r}}}]))\n"
            f" print(json.dumps([{{'platform':{self.recipe['platform']!r}}}]))\n"
            "elif sql == 'SELECT source_id FROM pragma_version();':\n"
            f" print(json.dumps([{{'source_id':{source_id!r}}}]))\n"
            "else: raise AssertionError('Preflight must not load extensions or run data queries')\n"
        )
        self.cli.chmod(0o755)

    def invoke(self, check_only=False, env=None):
        return subprocess.run(
            [
                sys.executable,
                str(self.here / "build_extension.py"),
                "--duckdb",
                str(self.cli),
                *(["--check-only"] if check_only else ["--output", str(self.output)]),
            ],
            cwd=self.root,
            env=env or self.env,
            text=True,
            capture_output=True,
            timeout=20,
        )

    def test_matching_pins_and_cli_check_without_build_output(self):
        # An inherited Git override must not redirect inspection to another tree.
        result = self.invoke(
            True, {**self.env, "GIT_DIR": str(self.root / "wrong.git")}
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("native conformance is still required", result.stdout)
        self.assertEqual(len(self.trace.read_text().splitlines()), 2)
        self.assertFalse(self.output.exists())

    def test_source_drift_refused_before_cli_download_or_output(self):
        pin = self.recipe["sources"]["duckdb"]["commit"]
        for engine, wrapper in (("a" * 40, "a" * 40), (pin, "b" * 40)):
            self.commit_pins(engine, wrapper)
            for check_only in (False, True):
                with self.subTest(
                    engine=engine, wrapper=wrapper, check_only=check_only
                ):
                    result = self.invoke(check_only)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("recipe needs migration", result.stderr)
                    self.assertNotIn("Fetching", result.stdout)
                    self.assertFalse(self.output.exists())
                    self.assertFalse(self.trace.exists())

    def test_wrong_cli_version_or_source_refused_before_download_or_output(self):
        for overrides, diagnostic in (
            ({"version": "v1.5.6"}, "CLI does not match"),
            ({"source_id": "deadbeef00"}, "CLI source ID mismatch"),
        ):
            self.write_cli(**overrides)
            for check_only in (False, True):
                with self.subTest(overrides=overrides, check_only=check_only):
                    result = self.invoke(check_only)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(diagnostic, result.stderr)
                    self.assertNotIn("Fetching", result.stdout)
                    self.assertFalse(self.output.exists())
