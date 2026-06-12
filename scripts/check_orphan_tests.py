#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Pre-commit guard against orphaned test files.

A ``test_*.cpp`` that exists under ``test/cpp`` but is not listed in
CMakeLists.txt's TEST_SOURCES never compiles, so its coverage silently
vanishes. This check fails when such a file exists, unless it carries an
explicit, reasoned entry in ``test/cpp/orphan_test_allowlist.txt``.

Runs from the repo root (pre-commit's default cwd). No third-party deps.
"""

from __future__ import annotations

import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "test" / "cpp"
CMAKELISTS = REPO_ROOT / "CMakeLists.txt"
ALLOWLIST = TEST_DIR / "orphan_test_allowlist.txt"

# Matches any test/cpp/.../test_*.cpp reference in CMakeLists.txt, including
# entries inside option-gated blocks (a gated test is not an orphan when its
# option is off).
CMAKE_TEST_REF = re.compile(r"test/cpp/[A-Za-z0-9_/]*test_[A-Za-z0-9_]+\.cpp")


def read_allowlist() -> set[str]:
    entries: set[str] = set()
    if not ALLOWLIST.exists():
        return entries
    for raw in ALLOWLIST.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries


def main() -> int:
    tree_tests = {
        p.relative_to(REPO_ROOT).as_posix() for p in TEST_DIR.rglob("test_*.cpp")
    }
    cmake_tests = set(CMAKE_TEST_REF.findall(CMAKELISTS.read_text()))
    allowlist = read_allowlist()

    orphans = sorted(tree_tests - cmake_tests - allowlist)
    stale_allowlist = sorted(allowlist - tree_tests)

    status = 0
    if orphans:
        status = 1
        print("Orphaned test file(s) — present in the tree but not in TEST_SOURCES,")
        print("so they never compile and their coverage silently vanishes:")
        for path in orphans:
            print(f"  {path}")
        print(
            "Add them to TEST_SOURCES in CMakeLists.txt (or, with a reason, to "
            "test/cpp/orphan_test_allowlist.txt)."
        )
    if stale_allowlist:
        status = 1
        print("Stale allowlist entr(y/ies) — listed but no longer in the tree:")
        for path in stale_allowlist:
            print(f"  {path}")
        print("Remove them from test/cpp/orphan_test_allowlist.txt.")
    return status


if __name__ == "__main__":
    sys.exit(main())
