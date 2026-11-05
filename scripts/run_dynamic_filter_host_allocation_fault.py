#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Run the opt-in dynamic-filter host OOM tests against an unstripped Linux binary."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=Path,
        default=root / "build/release/extension/sirius/test/cpp/sirius_unittest",
    )
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--compute-sanitizer", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    helper = (
        args.helper or binary.parents[2] / "libsirius_host_allocation_fault.so"
    ).resolve()
    if not helper.is_file():
        parser.error(
            "build the sirius_host_allocation_fault target before running these tests"
        )
    symbols = subprocess.run(
        ["nm", "-S", "--defined-only", str(binary)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    environment = os.environ.copy()
    for variable, token in (
        ("SIRIUS_HOST_FAULT_SHELL_RANGE", "accumulated_bloom_builder12make_filters"),
    ):
        matches = [
            row.split()
            for row in symbols
            if token in row and ".cold" not in row and "$got" not in row
        ]
        if len(matches) != 1 or len(matches[0]) != 4:
            parser.error(f"expected one unstripped function symbol for {token}")
        address, size, _, _ = matches[0]
        environment[variable] = f"{address}:{size}"
    prior = environment.get("LD_PRELOAD", "")
    environment["LD_PRELOAD"] = str(helper) + (":" + prior if prior else "")
    command = [str(binary), "[host_allocation_fault]"]
    if args.compute_sanitizer:
        command = [
            "compute-sanitizer",
            "--tool",
            "memcheck",
            "--track-stream-ordered-races=all",
            "--error-exitcode=99",
            *command,
        ]
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
