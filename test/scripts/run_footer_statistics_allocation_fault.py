#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Run hidden footer-cache allocation tests with the existing opt-in new helper.

The sites are selected in x86-64 GCC code, where the build inlines the insertions:
the return PC of the allocation before the LRU hook, and of the merged-map and
row-count nodes before their rehash checks. On any other machine the script exits
with an error before it reads the binary. It requires the named containers at those
sites and fails on missing or ambiguous matches. No node sizes, private layouts, or
fixed binary addresses are assumed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess


def instructions(binary, symbols, name):
    matches = [
        row.split(maxsplit=3)
        for row in symbols
        if f"transparent::(anonymous namespace)::{name}(" in row and ".cold" not in row
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one unstripped {name} symbol")
    address, size, _, _ = matches[0]
    begin = int(address, 16)
    assembly = subprocess.check_output(
        [
            "objdump",
            "-dC",
            "--no-show-raw-insn",
            f"--start-address={begin:#x}",
            f"--stop-address={begin + int(size, 16):#x}",
            str(binary),
        ],
        text=True,
    )
    decoded = []
    for line in assembly.splitlines():
        if match := re.match(r"\s*([0-9a-f]+):\s+(.+)", line):
            decoded.append((int(match[1], 16), match[2]))
    return assembly, decoded


def allocation_before(code, callee, container):
    anchors = [
        i
        for i, (_, text) in enumerate(code)
        if "call" in text
        and callee in text
        and any(container in prior for _, prior in code[max(0, i - 14) : i])
    ]
    if len(anchors) != 1:
        raise RuntimeError(f"expected one {container} call to {callee}")
    anchor = anchors[0]
    candidates = [
        i
        for i, (_, text) in enumerate(code[:anchor])
        if "call" in text and "operator new(unsigned long)" in text
    ]
    if not candidates or anchor - candidates[-1] > 100:
        raise RuntimeError(f"cannot identify allocation before {callee}")
    return code[candidates[-1] + 1][0]


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary", type=Path, default=root / "build/release/test/cpp/sirius_unittest"
    )
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.machine() != "x86_64":
        parser.exit(
            2,
            f"{parser.prog}: the allocation sites are selected in x86-64 GCC code; "
            f"this machine is {platform.machine()}\n",
        )
    binary = args.binary.resolve()
    helper = (
        args.helper or binary.parents[2] / "libsirius_host_allocation_fault.so"
    ).resolve()
    if not helper.is_file():
        parser.error("build the sirius_host_allocation_fault target first")
    args.output.mkdir(parents=True, exist_ok=True)
    symbols = subprocess.check_output(
        ["nm", "-S", "-C", "--defined-only", str(binary)], text=True
    ).splitlines()
    result = {
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "cases": {},
    }
    failed = False
    for label, function, callee, container in (
        ("lru", "statistics_of", "_List_node_base::_M_hook", "::cache_order"),
        ("merge", "visit", "_M_need_rehash", "::merges"),
        ("row_count", "keep_row_count_locked", "_M_need_rehash", "::row_counts"),
    ):
        assembly, code = instructions(binary, symbols, function)
        (args.output / f"{label}.asm").write_text(assembly)
        address = allocation_before(code, callee, container)
        environment = os.environ.copy()
        environment["SIRIUS_HOST_FAULT_SHELL_RANGE"] = f"{address:x}:1"
        prior = environment.get("LD_PRELOAD", "")
        environment["LD_PRELOAD"] = str(helper) + (":" + prior if prior else "")
        spec = f"[footer_fix_{label}]"
        command = [str(binary), spec, "--order", "decl"]
        with (args.output / f"{label}.log").open("w") as log:
            run = subprocess.run(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        text = (args.output / f"{label}.log").read_text()
        passed = run.returncode == 0 and re.search(
            r"All tests passed .* in 1 test case", text
        )
        result["cases"][label] = {
            "command": command,
            "return_pc": hex(address),
            "exit": run.returncode,
            "one_case_passed": bool(passed),
        }
        failed |= not passed
        print(
            f"{label}: return PC {address:#x}, exit {run.returncode}, passed {bool(passed)}",
            flush=True,
        )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
