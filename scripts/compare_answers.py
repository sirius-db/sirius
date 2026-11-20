#!/usr/bin/env python3
"""Compares the answers of a benchmark run with reference answers.

The run is the output of a DuckDB shell in `.mode csv` with `.timer on`, where each query follows
a marker statement `SELECT 'Q<n> run <k>' AS marker;` (see
docs/super-sirius/distributed-experiments-no-slurm.md). Rows compare in any order; numbers match
within 1e-6 relative, or within the rounding of the one printed with fewer decimals (at most
0.01). A query the reference has and the run does not counts as a failure. A text diff does not
work: float digits and the order of equal rows differ between engines.

  compare_answers.py client.txt --ref-client ref.txt
      the reference has the same markers; run 1 of each query is the reference
  compare_answers.py client.txt --ref-dir DIR --ref-pattern "q{n:02d}.csv"
      the reference is one CSV file with a header per query

  --ties Q:col,col   Q ends in ORDER BY col, col LIMIT. A difference counts as a tie, not an
                     error, when both answers have as many rows, the same order-column values in
                     their last row, and every differing row carries exactly those values.

Prints one table row per run, with the query's seconds, and exits 1 if any run differs.
"""

import argparse
import csv
import io
import os
import re
import sys
from collections import Counter


def parse_client(path):
    """Returns {(query, run): {"header", "rows", "time", "error"}}."""
    runs, current, state = {}, None, None
    with open(path, encoding="utf-8", errors="replace") as file:
        lines = file.read().splitlines()
    for line in lines:
        marker = re.fullmatch(r'"?Q(\d+) run (\d+)"?', line.strip())
        if marker:
            current = (int(marker.group(1)), int(marker.group(2)))
            runs[current] = {"lines": [], "time": None, "error": None}
            state = "marker_time"
            continue
        if current is None:
            continue
        timer = re.match(r"Run Time \(s\): real ([0-9.]+)", line)
        if timer:
            # The first Run Time after a marker is the marker's, the second the query's.
            if state == "marker_time":
                state = "body"
            elif state == "body":
                runs[current]["time"] = float(timer.group(1))
                state = "done"
        elif state == "body":
            runs[current]["lines"].append(line)
    for run in runs.values():
        text = "\n".join(run["lines"])
        if re.search(r"(?m)^(\w+ )*Error:|^Error|Exception", text):
            run["error"] = text[:1500]
        rows = [row for row in csv.reader(io.StringIO(text)) if row]
        run["header"], run["rows"] = (rows[0], rows[1:]) if rows else ([], [])
    return runs


def parse_csv_file(path):
    with open(path, encoding="utf-8", errors="replace") as file:
        lines = [
            line
            for line in file.read().splitlines()
            if line.strip() and not line.startswith("Run Time")
        ]
    rows = [row for row in csv.reader(io.StringIO("\n".join(lines))) if row]
    return rows[1:]


def number(cell):
    try:
        return float(cell)
    except ValueError:
        return None


def decimals(cell):
    """Digits after the decimal point, as printed; None for an exponent form."""
    match = re.fullmatch(r"-?\d*(?:\.(\d*))?", cell.strip())
    return len(match.group(1) or "") if match else None


def cells_match(a, b):
    if a == b:
        return True
    x, y = number(a), number(b)
    if x is None or y is None:
        return False
    if abs(x - y) <= 1e-6 * max(abs(x), abs(y)):
        return True
    # One engine rounds where the other prints more digits (a DECIMAL against a DOUBLE): they
    # may differ by half a unit of the last place of the shorter one. A fixed 0.01 would also
    # pass 0.05 against 0.06.
    places = [d for d in (decimals(a), decimals(b)) if d is not None]
    if len(places) < 2:
        return False
    return abs(x - y) <= min(0.01, 0.5 * 10 ** -min(places)) * (1 + 1e-9)


def rows_match(a, b):
    return len(a) == len(b) and all(cells_match(x, y) for x, y in zip(a, b))


def exact_cells(row):
    """The cells that must match exactly: text and integers."""
    return tuple(c for c in row if number(c) is None or re.fullmatch(r"-?\d+", c))


def differing_rows(got, ref):
    """Multiset difference with the numeric tolerance: (only in got, only in ref)."""
    left = Counter(tuple(row) for row in ref)
    unmatched = []
    for row in got:
        key = tuple(row)
        if left.get(key, 0) > 0:
            left[key] -= 1
        else:
            unmatched.append(row)
    buckets = {}
    for key, count in left.items():
        for _ in range(count):
            buckets.setdefault(exact_cells(key), []).append(list(key))
    only_got = []
    for row in unmatched:
        candidates = buckets.get(exact_cells(row), [])
        for i, candidate in enumerate(candidates):
            if rows_match(row, candidate):
                del candidates[i]
                break
        else:
            only_got.append(row)
    return only_got, [row for bucket in buckets.values() for row in bucket]


def is_tie(run, ref_rows, only_got, only_ref, columns):
    if not run["rows"] or len(run["rows"]) != len(ref_rows):
        return False, None
    if any(column not in run["header"] for column in columns):
        return False, None
    indexes = [run["header"].index(column) for column in columns]
    cut = [run["rows"][-1][i] for i in indexes]
    ref_cut = [ref_rows[-1][i] for i in indexes]
    same_cut = all(cells_match(x, y) for x, y in zip(cut, ref_cut))
    at_cut = all(
        all(cells_match(row[i], value) for i, value in zip(indexes, cut))
        for row in only_got + only_ref
    )
    return same_cut and at_cut, cut


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("client", help="the run's output")
    parser.add_argument("--ref-client", help="a reference output with the same markers")
    parser.add_argument("--ref-dir", help="a folder with one reference CSV per query")
    parser.add_argument("--ref-pattern", default="q{n:02d}.csv")
    parser.add_argument("--ties", action="append", default=[], metavar="Q:col,col")
    args = parser.parse_args()
    if bool(args.ref_client) == bool(args.ref_dir):
        parser.error("give one of --ref-client and --ref-dir")
    ties = {int(t.split(":")[0]): t.split(":")[1].split(",") for t in args.ties}
    runs = parse_client(args.client)
    reference = parse_client(args.ref_client) if args.ref_client else None
    if not runs:
        print(f"no 'Q<n> run <k>' marker in {args.client}", file=sys.stderr)
        return 1

    bad = 0
    notes = []
    print("| query | run | seconds | rows | reference rows | result |")
    print("|---|---|---|---|---|---|")
    for query, k in sorted(runs):
        run = runs[(query, k)]
        if reference is not None:
            ref = reference.get((query, 1))
            ref_rows = None if ref is None or ref["error"] else ref["rows"]
        else:
            path = os.path.join(args.ref_dir, args.ref_pattern.format(n=query))
            ref_rows = parse_csv_file(path) if os.path.isfile(path) else None
        if ref_rows is None:
            print(
                f"| Q{query} | {k} | {run['time']} | {len(run['rows'])} | - | NO REFERENCE |"
            )
            bad += 1
            continue
        if run["error"] or run["time"] is None:
            why = (run["error"] or "no Run Time line").replace("\n", " ")[:300]
            print(
                f"| Q{query} | {k} | {run['time']} | - | {len(ref_rows)} | FAILED: {why} |"
            )
            bad += 1
            continue
        only_got, only_ref = differing_rows(run["rows"], ref_rows)
        if not only_got and not only_ref:
            verdict = "match"
        else:
            tie, cut = (
                is_tie(run, ref_rows, only_got, only_ref, ties[query])
                if query in ties
                else (False, None)
            )
            if tie:
                verdict = (
                    f"match up to a LIMIT tie: {len(only_got)} row(s) differ, all with order "
                    f"columns {ties[query]} = {cut}, the values of the last row in both answers"
                )
            else:
                verdict = (
                    f"DIFF: {len(only_got)} row(s) only in the run, "
                    f"{len(only_ref)} only in the reference"
                )
                bad += 1
            notes += [
                f"Q{query} run {k} only in the run: {row}" for row in only_got[:5]
            ]
            notes += [
                f"Q{query} run {k} only in the reference: {row}" for row in only_ref[:5]
            ]
        print(
            f"| Q{query} | {k} | {run['time']} | {len(run['rows'])} | {len(ref_rows)} | {verdict} |"
        )
    ran = {query for query, _ in runs}
    if reference is not None:
        expected = {query for query, _ in reference}
    else:
        name = re.compile(
            re.sub(r"\\\{n[^}]*\\\}", r"(\\d+)", re.escape(args.ref_pattern))
        )
        expected = {
            int(match.group(1))
            for match in map(name.fullmatch, os.listdir(args.ref_dir))
            if match
        }
    # A client that stopped early leaves no marker for the queries it never reached.
    for query in sorted(expected - ran):
        print(f"| Q{query} | - | - | - | - | NOT RUN |")
        bad += 1
    if notes:
        print()
        for note in notes:
            print("- " + note)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
