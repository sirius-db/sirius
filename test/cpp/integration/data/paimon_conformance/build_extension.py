#!/usr/bin/env python3
"""Build the pinned local-filesystem Paimon reader; never install community bytes."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.error import HTTPError

from corpus_checks import digest
from qualified_extension import download, DownloadFailure, open_https

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RECIPE = HERE / "source-build.json"
DOWNLOADER = HERE / "qualified_extension.py"
BUILD_ENVIRONMENT_KEYS = (
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "CMAKE_ARGS",
    "CONDA_PREFIX",
    "SCCACHE_RECACHE",
    "CCACHE_DISABLE",
    "GIT_CEILING_DIRECTORIES",
)


def read_recipe():
    recipe = json.loads(RECIPE.read_text())
    if recipe["format_version"] != 1:
        raise ValueError("Unsupported source build recipe")
    return recipe


def validate_checkout_recipe(recipe):
    """The recipe must follow both committed Sirius host pins, never a branch tip."""
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    result = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-tree",
            "HEAD",
            "--",
            "duckdb",
            "sirius-duckdb/duckdb",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    expected = recipe["sources"]["duckdb"]["commit"]
    pins = {}
    for line in result.stdout.splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, commit = metadata.split()
        if mode == "160000" and kind == "commit":
            pins[path] = commit
    if pins != {"duckdb": expected, "sirius-duckdb/duckdb": expected}:
        raise ValueError(
            f"Paimon recipe needs migration: recipe DuckDB={expected}; "
            f"Sirius HEAD pins={pins}. Update the reader recipe and requalify it; "
            "do not bypass this check."
        )


def check_compatibility(cli, recipe):
    """Reject stale source pins and host binaries before downloads or build output."""
    from run_conformance import probe_identity

    validate_checkout_recipe(recipe)
    with tempfile.TemporaryDirectory(prefix="paimon-host-check-") as directory:
        identity = probe_identity(cli, Path(directory), 90)
        expected = {"version": recipe["duckdb_version"], "platform": recipe["platform"]}
        if identity != expected:
            raise ValueError(
                f"DuckDB CLI does not match the Paimon recipe: expected {expected}, "
                f"got {identity}. Rebuild the matching Sirius host or migrate the recipe."
            )
        env = os.environ.copy()
        env["SIRIUS_DISABLE"] = "1"
        result = subprocess.run(
            [
                str(cli),
                "-init",
                "/dev/null",
                "-json",
                "-batch",
                "-bail",
                ":memory:",
                "-c",
                "SELECT source_id FROM pragma_version();",
            ],
            env=env,
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        source_id = json.loads(result.stdout)[0]["source_id"]
        expected_commit = recipe["sources"]["duckdb"]["commit"]
        if not matches_source_id(source_id, expected_commit):
            raise ValueError(
                f"DuckDB CLI source ID mismatch: expected {expected_commit}, got {source_id}. "
                "Rebuild the matching Sirius host; a matching version string is insufficient."
            )
    return identity, source_id


def fetch_source(record, destination, *, opener=open_https, delay=time.sleep):
    """Retry transport errors, but never bypass a source checksum failure."""
    for attempt in range(1, 4):
        try:
            download(record["url"], destination, record["size_bytes"], opener)
        except DownloadFailure as error:
            cause = error.__cause__
            if (
                isinstance(cause, HTTPError)
                and cause.code not in (408, 429)
                and cause.code < 500
            ):
                raise
            if attempt == 3:
                raise
            delay(attempt * 5)
            continue
        if digest(destination) != record["sha256"]:
            raise ValueError("Source archive SHA256 mismatch")
        return


def extract_source(archive, destination):
    # Python's data filter blocks escaping paths, devices and unsafe link targets.
    # Extract into a new staging directory, then require the expected single root.
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        with tarfile.open(archive) as source:
            source.extractall(temporary, filter="data")
        roots = list(Path(temporary).iterdir())
        if len(roots) != 1 or not roots[0].is_dir() or roots[0].is_symlink():
            raise ValueError("Expected one source archive root directory")
        roots[0].rename(destination)


def populate_dependency_cache(native, cache):
    """Reuse source archives only when the pinned native manifest supplies a hash."""
    if cache is None:
        return
    variables = {}
    for line in (native / "third_party/versions.txt").read_text().splitlines():
        match = re.fullmatch(r"([A-Z0-9_]+)=(.*)", line)
        if match:
            name, value = match.groups()
            variables[name] = re.sub(
                r"\$\{([A-Z0-9_]+)\}", lambda item: variables[item[1]], value
            )
    for name, package in variables.items():
        if not name.endswith("_PKG_NAME"):
            continue
        checksum = variables.get(
            name.removesuffix("_PKG_NAME") + "_BUILD_SHA256_CHECKSUM"
        )
        if checksum is None:
            continue
        cached = cache / (checksum + ".tar.gz")
        if not cached.exists():
            continue
        if Path(package).name != package or digest(cached) != checksum:
            raise ValueError(
                "Cached native dependency archive differs from pinned source"
            )
        shutil.copyfile(cached, native / "third_party" / package)


def run(command, log, env):
    print("Running: " + " ".join(map(str, command)), flush=True)
    with log.open("ab") as output:
        output.write(("\n$ " + " ".join(map(str, command)) + "\n").encode())
        output.flush()
        subprocess.run(
            command, stdout=output, stderr=subprocess.STDOUT, env=env, check=True
        )


def matches_source_id(source_id, commit):
    # DuckDB uses the checkout's abbreviated Git hash, whose length can vary.
    return (
        isinstance(source_id, str)
        and re.fullmatch(r"[0-9a-f]{7,40}", source_id) is not None
        and commit.startswith(source_id)
    )


def validate_receipt(receipt_path, extension, cli, identity):
    """Check trusted-builder provenance, not a signature or a correctness oracle."""
    recipe = read_recipe()
    receipt = json.loads(Path(receipt_path).read_text())
    expected = {
        "format_version": 1,
        "state": "built",
        "recipe_sha256": digest(RECIPE),
        "builder_sha256": digest(Path(__file__)),
        "downloader_sha256": digest(DOWNLOADER),
        "pixi_lock_sha256": digest(ROOT / "pixi.lock"),
        "sources": recipe["sources"],
        "cmake_options": recipe["cmake_options"],
        "duckdb_identity": identity,
        "duckdb_sha256": digest(cli),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"Source build receipt mismatch: {key}")
    if identity != {
        "version": recipe["duckdb_version"],
        "platform": recipe["platform"],
    }:
        raise ValueError("Runtime does not match approved source build")
    if not matches_source_id(
        receipt.get("duckdb_source_id"), recipe["sources"]["duckdb"]["commit"]
    ):
        raise ValueError("Source build receipt has the wrong DuckDB source ID")
    if (
        receipt.get("sha256") != digest(extension)
        or receipt.get("size_bytes") != extension.stat().st_size
    ):
        raise ValueError("Source-built extension bytes differ from build receipt")
    toolchain = receipt.get("toolchain")
    if not isinstance(toolchain, dict) or any(
        not isinstance(toolchain.get(name), str) or not toolchain[name]
        for name in ("cc", "cxx", "cmake", "ninja")
    ):
        raise ValueError("Source build receipt lacks toolchain provenance")
    environment = receipt.get("build_environment")
    if not isinstance(environment, dict) or any(
        not isinstance(environment.get(name), str) for name in BUILD_ENVIRONMENT_KEYS
    ):
        raise ValueError("Source build receipt lacks build environment provenance")
    return {
        "kind": "source_build",
        "version": recipe["extension_version"],
        "duckdb_version": recipe["duckdb_version"],
        "platform": recipe["platform"],
        "source_commit": recipe["sources"]["extension"]["commit"],
        "native_commit": recipe["sources"]["native"]["commit"],
        "sha256": receipt["sha256"],
        "size_bytes": receipt["size_bytes"],
        "receipt_sha256": digest(receipt_path),
        "recipe_sha256": receipt["recipe_sha256"],
    }


def build(cli, output, jobs, source_cache=None):
    input_hashes = {
        "recipe_sha256": digest(RECIPE),
        "builder_sha256": digest(Path(__file__)),
        "downloader_sha256": digest(DOWNLOADER),
        "pixi_lock_sha256": digest(ROOT / "pixi.lock"),
    }
    recipe = read_recipe()
    cli_sha256 = digest(cli)
    identity, source_id = check_compatibility(cli, recipe)
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("The source recipe currently supports Linux x86_64 only")
    if ":" in str(output):
        raise ValueError("Build output cannot contain the Git ceiling separator ':'")
    if (
        not os.environ.get("CONDA_PREFIX")
        or not os.environ.get("CC")
        or not os.environ.get("CXX")
    ):
        raise ValueError(
            "Run through the repository's locked Pixi environment (CC/CXX required)"
        )
    # No reuse of old build directories or receipts, even after a failed attempt.
    output.mkdir(parents=True, exist_ok=False)
    log = output / "build.log"
    env = os.environ.copy()
    # Inherited dependency overrides must not substitute unrelated local libraries.
    for key in list(env):
        if key.startswith(("PAIMON_", "ARROW_", "GIT_")) or key in (
            "CMAKE_PREFIX_PATH",
            "CMAKE_TOOLCHAIN_FILE",
            "CMAKE_PROJECT_INCLUDE",
            "CMAKE_PROJECT_TOP_LEVEL_INCLUDES",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "C_INCLUDE_PATH",
        ):
            env.pop(key)
    # These are source archives, even when output is inside the Sirius checkout.
    # An ancestor Git worktree makes upstream `git apply` silently skip patches.
    env["GIT_CEILING_DIRECTORIES"] = str(output)
    # Compiler selection is shared by DuckDB's outer build and nested native build.
    env["CC"] = str(Path(shutil.which(env["CC"]) or env["CC"]).resolve(strict=True))
    env["CXX"] = str(Path(shutil.which(env["CXX"]) or env["CXX"]).resolve(strict=True))
    toolchain = {}
    for name, executable in (
        ("cc", env["CC"]),
        ("cxx", env["CXX"]),
        ("cmake", "cmake"),
        ("ninja", "ninja"),
    ):
        toolchain[name] = subprocess.check_output(
            [executable, "--version"], env=env, text=True
        ).strip()
    for name, record in recipe["sources"].items():
        print(f"Fetching pinned {name} source", flush=True)
        archive = output / (name + ".tar.gz")
        cached = source_cache / (record["sha256"] + ".tar.gz") if source_cache else None
        if cached is not None and cached.exists():
            if (
                cached.stat().st_size != record["size_bytes"]
                or digest(cached) != record["sha256"]
            ):
                raise ValueError("Cached source archive differs from approved recipe")
            shutil.copyfile(cached, archive)
        else:
            fetch_source(record, archive)
        extract_source(archive, output / name)
    native = output / "extension/third_party/paimon-cpp"
    # The gitlink is an empty directory in GitHub's source archive.
    if native.exists():
        native.rmdir()
    (output / "native").rename(native)
    populate_dependency_cache(native, source_cache)
    config = output / "extensions.cmake"
    # Bracket quoting permits paths with spaces without interpreting escapes.
    if "]=]" in str(output):
        raise ValueError("Unsupported delimiter in build path")
    config.write_text(
        f'duckdb_extension_load(paimon SOURCE_DIR [=[{output / "extension"}]=] '
        f'EXTENSION_VERSION {recipe["extension_version"]})\n'
    )
    command = [
        "cmake",
        "-S",
        str(output / "duckdb"),
        "-B",
        str(output / "build"),
        "-G",
        "Ninja",
    ]
    command += [f"-D{key}={value}" for key, value in recipe["cmake_options"].items()]
    command += [
        f"-DDUCKDB_EXTENSION_CONFIGS={config}",
        f"-DGIT_COMMIT_HASH={recipe['sources']['duckdb']['commit']}",
        f"-DCMAKE_C_COMPILER={env['CC']}",
        f"-DCMAKE_CXX_COMPILER={env['CXX']}",
    ]
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    run(command, log, env)
    run(
        [
            "cmake",
            "--build",
            str(output / "build"),
            "--target",
            "paimon_loadable_extension",
            "--parallel",
            str(jobs),
        ],
        log,
        env,
    )
    artifact = output / "build/extension/paimon/paimon.duckdb_extension"
    if artifact.stat().st_size == 0:
        raise ValueError("Build produced an empty extension")
    shutil.copyfile(artifact, output / "paimon.duckdb_extension")
    if digest(cli) != cli_sha256:
        raise ValueError("DuckDB executable changed during source build")
    for name, path in (
        ("recipe_sha256", RECIPE),
        ("builder_sha256", Path(__file__)),
        ("downloader_sha256", DOWNLOADER),
        ("pixi_lock_sha256", ROOT / "pixi.lock"),
    ):
        if digest(path) != input_hashes[name]:
            raise ValueError(f"Build input changed during source build: {name}")
    receipt = {
        "format_version": 1,
        "state": "built",
        "recipe_sha256": digest(RECIPE),
        "builder_sha256": digest(Path(__file__)),
        "downloader_sha256": digest(DOWNLOADER),
        "pixi_lock_sha256": digest(ROOT / "pixi.lock"),
        "sources": recipe["sources"],
        "cmake_options": recipe["cmake_options"],
        "duckdb_identity": identity,
        "duckdb_sha256": digest(cli),
        "duckdb_source_id": source_id,
        "toolchain": toolchain,
        "build_environment": {
            name: env.get(name, "") for name in BUILD_ENVIRONMENT_KEYS
        },
        "sha256": digest(artifact),
        "size_bytes": artifact.stat().st_size,
    }
    temporary = output / "build-receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(output / "build-receipt.json")
    print(f"Built: {output / 'paimon.duckdb_extension'}", flush=True)
    print(
        "Build success is not conformance success; run the reader tests next.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="New build directory; must not exist"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check committed Sirius pins and CLI identity without downloading or building",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="Optional read-only archive cache: <sha256>.tar.gz; every entry is verified",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.check_only:
        check_compatibility(args.duckdb.resolve(strict=True), read_recipe())
        print(
            "Compatible Sirius pins and DuckDB CLI; native conformance is still required."
        )
        return
    if args.output is None:
        parser.error("--output is required unless --check-only is used")
    build(
        args.duckdb.resolve(strict=True),
        args.output.resolve(),
        args.jobs,
        args.source_cache.resolve() if args.source_cache else None,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Paimon source build failed: {error}", file=sys.stderr)
        sys.exit(1)
