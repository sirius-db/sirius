#!/usr/bin/env python3
"""Check Sirius against the pyiceberg-recorded conformance expectations.

Usage:
    python3 run_conformance.py <corpus-dir> [--duckdb <path>] [--timeout N]

Gates on two things, both of which the 13 hand-built in-tree fixtures missed:

  1. CORRECTNESS - rows must match what pyiceberg's own scan returned. This is the
     gate that catches silent wrong results (drop-and-re-add returning stale data).

  2. LIVENESS - each case issues a SECOND query on the SAME connection. A runtime
     GPU fallback poisons the connection and the next GPU query deadlocks forever,
     so "did query 2 answer?" is the observable test for "did we fall back at
     runtime instead of declining at plan time".

Route (GPU vs CPU) is NOT asserted here - the transparent-execution counters are not
SQL-exposed. Route assertions belong in the C++ suite, which already has them.

Each case runs in its own duckdb process: the deadlock is per-connection, so one
hung case must not be able to hide or block the others.

`SET GLOBAL`, not `SET` - a plain SET does not reach the delete gate's fresh probe
connection, so the gate declines the table and hides everything downstream of it.
"""
import argparse
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("corpus")
ap.add_argument("--duckdb", default="build/release/duckdb")
ap.add_argument("--timeout", type=int, default=90)
ap.add_argument("--case", default=None, help="run only this case")
args = ap.parse_args()

with open(os.path.join(args.corpus, "expectations.json")) as f:
    cases = json.load(f)["cases"]
if args.case:
    cases = [c for c in cases if c["name"] == args.case]

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = f"{os.getcwd()}/.pixi/envs/default/lib:" + env.get(
    "LD_LIBRARY_PATH", ""
)


def norm(rows):
    """Order-insensitive, type-loose comparison. Iceberg does not promise scan order."""
    out = []
    for r in rows:
        out.append(tuple("NULL" if v is None else str(v) for v in r))
    return sorted(out)


def run_case(case):
    cols = ", ".join(f'"{c}"' for c in case["columns"])
    sql = (
        "LOAD iceberg;\n"
        "SET GLOBAL unsafe_enable_version_guessing=true;\n"
        f"SELECT {cols} FROM iceberg_scan('{case['table_dir']}');\n"
        # Second query, SAME connection: if query 1 fell back at runtime, this hangs.
        "SELECT 'LIVENESS_OK' AS probe;\n"
    )
    try:
        p = subprocess.run(
            [args.duckdb, "-unsigned", "-json", "-c", sql],
            capture_output=True,
            text=True,
            timeout=args.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return (
            "DEADLOCK",
            f"no answer in {args.timeout}s (runtime fallback poisoned the connection)",
        )

    if "LIVENESS_OK" not in p.stdout:
        err = (p.stderr or p.stdout).strip().splitlines()
        return "NO_LIVENESS", "second query never answered: " + (
            err[-1] if err else "no output"
        )

    # -json emits one JSON array per statement; the scan is the first non-empty one.
    blocks = []
    depth, start = 0, None
    for i, ch in enumerate(p.stdout):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(p.stdout[start : i + 1])
                start = None
    if not blocks:
        return "NO_ROWS", "scan produced no parseable result"

    try:
        got = json.loads(blocks[0])
    except json.JSONDecodeError as e:
        return "BAD_OUTPUT", f"could not parse scan result: {e}"

    got_rows = [[row.get(c) for c in case["columns"]] for row in got]
    if norm(got_rows) != norm(case["expected_rows"]):
        return (
            "WRONG_ROWS",
            f"pyiceberg={norm(case['expected_rows'])} sirius={norm(got_rows)}",
        )
    return "OK", f"{len(got_rows)} rows match the reference"


fails = 0
print(f"corpus: {args.corpus}\nduckdb: {args.duckdb}\n")
for case in cases:
    status, detail = run_case(case)
    ok = status == "OK"
    fails += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {case['name']:<14} {status:<12} {detail}")
    if not ok:
        print(f"           why this case exists: {case['why']}")

print(f"\n{len(cases) - fails}/{len(cases)} cases match the Apache reference")
sys.exit(1 if fails else 0)
