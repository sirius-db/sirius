#!/usr/bin/env python3
"""Check query-cost tools before running a comparison or I/O trace."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=("compare", "trace"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    print("Checking query-cost tools...", file=sys.stderr, flush=True)
    checks = subprocess.run(
        [sys.executable, "-B", str(directory / "test_query_cost_tools.py")]
    )
    if checks.returncode:
        return checks.returncode
    script = {
        "compare": "compare_query_cost.py",
        "trace": "trace_query_io.py",
    }[args.tool]
    return subprocess.run(
        [sys.executable, "-B", str(directory / script), *args.arguments]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
