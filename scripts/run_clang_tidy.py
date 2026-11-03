#!/usr/bin/env python3
"""Run advisory clang-tidy checks against a prepared Sirius Clang build."""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
TARGETS = {
    "sirius_objects",
    "sirius_shared",
    "sirius_spdlog_sink",
    "sirius_host_allocation_fault",
    "sirius_unittest",
    "sirius_context_config_test",
    "sirius_extension_host",
    "sirius_cuda_fault_test",
    "sirius_nvtx_startup",
    "simpatico",
    "simpatico_cache_identity",
}


@dataclass(frozen=True)
class CompileCommand:
    source: Path
    directory: Path
    arguments: tuple[str, ...]
    output: Path

    @classmethod
    def parse(cls, entry: dict) -> CompileCommand:
        directory = Path(entry["directory"])
        if not directory.is_absolute():
            raise ValueError("compilation database directories must be absolute")
        arguments = tuple(entry.get("arguments") or shlex.split(entry["command"]))
        output = entry.get("output")
        if output is None:
            output = arguments[arguments.index("-o") + 1]
        return cls(
            (directory / entry["file"]).resolve(),
            directory,
            arguments,
            Path(output),
        )

    def owned(self, tracked: set[Path]) -> bool:
        return (
            self.source in tracked
            and self.source.suffix in SOURCE_SUFFIXES
            and any(f"{target}.dir" in self.output.parts for target in TARGETS)
        )

    def as_json(self) -> dict:
        return {
            "directory": str(self.directory),
            "file": str(self.source),
            "arguments": list(self.arguments),
            "output": str(self.output),
        }


def select_commands(
    commands: list[CompileCommand], paths: list[Path], all_files: bool
) -> list[CompileCommand]:
    if all_files:
        return commands
    known = {command.source for command in commands}
    for path in paths:
        if path.suffix in {".cu", ".cuh"}:
            raise ValueError(f"{path}: CUDA files are outside this hook's scope")
        if path.suffix not in SOURCE_SUFFIXES:
            raise ValueError(
                f"{path}: header or configuration changes require the full run: "
                "pixi run pre-commit run --hook-stage manual clang-tidy-all"
            )
        if path not in known:
            raise ValueError(
                f"{path}: no supported compile command; track the source, configure and build the "
                "Clang targets first (optional targets are outside this hook's scope)"
            )
    selected = set(paths)
    return [command for command in commands if command.source in selected]


def run(args: argparse.Namespace) -> int:
    build = args.build_dir.resolve()
    database = build / "compile_commands.json"
    if not database.is_file():
        raise ValueError(
            f"{database} is missing; configure and build clang-debug first"
        )
    tracked = {
        ROOT / os.fsdecode(path)
        for path in subprocess.check_output(
            ["git", "ls-files", "-z", "--", "src", "include", "test"], cwd=ROOT
        ).split(b"\0")
        if path
    }
    commands = [
        CompileCommand.parse(entry) for entry in json.loads(database.read_text())
    ]
    commands = [command for command in commands if command.owned(tracked)]
    commands = select_commands(
        commands, [path.resolve() for path in args.files], args.all
    )
    if not commands:
        raise ValueError("no supported source files selected")
    required = [
        "rust/crates/telemetry/bridge/include/rust/cxx.h",
        "rust/crates/telemetry/bridge/include/telemetry-bridge/gen/quent.hpp",
        "simpatico_codegen/generated/codegen/jit/embedded_headers.h",
    ]
    missing = [str(build / name) for name in required if not (build / name).is_file()]
    if missing:
        raise ValueError(
            "missing generated headers:\n  "
            + "\n  ".join(missing)
            + "\nBuild clang-debug first. If Cargo reports a cached build while headers "
            "are missing, use a fresh build directory."
        )
    tool = shutil.which("clang-tidy")
    if tool is None:
        raise ValueError(
            "clang-tidy is missing; run through the project's Pixi environment"
        )
    sources = sorted({command.source for command in commands})
    print(
        f"Checking {len(sources)} sources ({len(commands)} compile contexts) "
        f"with {args.jobs} workers; findings are advisory.",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="sirius-clang-tidy-") as directory:
        Path(directory, "compile_commands.json").write_text(
            json.dumps([command.as_json() for command in commands])
        )

        def check(source: Path) -> subprocess.CompletedProcess:
            return subprocess.run(
                [
                    tool,
                    "-p",
                    directory,
                    "--warnings-as-errors=-*",
                    f"--header-filter=^{re.escape(str(ROOT))}/(src|include|test)/",
                    str(source),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

        failed = False
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(check, source): source for source in sources}
            try:
                for future in as_completed(futures):
                    source = futures.pop(future)
                    result = future.result()
                    print(
                        f"{source.relative_to(ROOT)}: exit {result.returncode}",
                        flush=True,
                    )
                    if result.stdout:
                        print(
                            result.stdout,
                            end="" if result.stdout.endswith("\n") else "\n",
                        )
                    failed |= result.returncode != 0
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                raise
        return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--build-dir",
        type=Path,
        default=Path(
            os.environ.get("SIRIUS_CLANG_TIDY_BUILD_DIR", ROOT / "build/clang-debug")
        ),
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=os.environ.get("SIRIUS_CLANG_TIDY_JOBS", "4")
    )
    parser.add_argument(
        "--all", action="store_true", help="check all supported configured sources"
    )
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    if args.jobs < 1 or (not args.all and not args.files) or (args.all and args.files):
        parser.error("use --all or source filenames, and a positive worker count")
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (
        ValueError,
        KeyError,
        IndexError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"clang-tidy: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
