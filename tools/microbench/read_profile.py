#!/usr/bin/env python3
# Copyright 2025, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Load microbench_sweep.json and print eval-safe shell exports for bash."""

from __future__ import annotations

import argparse
import json
import pathlib
import shlex
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=pathlib.Path, help="Path to microbench_sweep.json")
    p.add_argument("profile", help="Profile name (e.g. daily, weekly, full)")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    try:
        prof = cfg["profiles"][args.profile]
    except KeyError:
        print(f"Unknown profile {args.profile!r}", file=sys.stderr)
        return 1

    filt = prof.get("benchmark_filter", ".*")
    extra = prof.get("extra_args") or []
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        print("extra_args must be a list of strings", file=sys.stderr)
        return 1

    print(f"SIRIUS_MICROBENCH_FILTER={shlex.quote(filt)}")
    # Bash array assignment
    print("SIRIUS_MICROBENCH_EXTRA_ARGS=(")
    for a in extra:
        print(f"  {shlex.quote(a)}")
    print(")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
