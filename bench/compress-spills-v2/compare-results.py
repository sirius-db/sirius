#!/usr/bin/env python3
"""Diff the per-query GPU result files of two run-suite.sh arms.

Used to validate a change that should not alter answers -- e.g. enabling more
Simpatico pin-compression plans. The plans under
src/compression/simpatico_codegen/plans/tpch_sf1000 were renamed *_disabled.txt
in cbd95163 "pending performance/correctness validation", so turning them back on
needs an answer check, not just a timing.

Comparing two GPU arms rather than GPU-vs-DuckDB on purpose: a CPU reference run
of TPC-H SF1000 does not fit in this box's RAM (it is what OOM-killed the machine),
whereas the 2-plan arm is already known-good.

Usage: compare-results.py <reference>.tsv <candidate>.tsv
"""

import csv
import os
import re
import sys


def load(tsv):
    """-> {qnum: (status, dir)}"""
    out = {}
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            q = int(re.sub(r"\D", "", row["query"]))
            out[q] = (row["status"], row["dir"])
    return out


def result_path(d, q):
    return os.path.join(d, "sirius", f"q{q}", "result.txt")


def main(ref_tsv, cand_tsv):
    ref, cand = load(ref_tsv), load(cand_tsv)
    same = diff = missing = 0
    print(f"{'q':>4} {'ref':>10} {'cand':>10}  verdict")
    for q in sorted(set(ref) | set(cand)):
        rs, rd = ref.get(q, ("absent", ""))
        cs, cd = cand.get(q, ("absent", ""))
        rp, cp = result_path(rd, q) if rd else "", result_path(cd, q) if cd else ""
        if not (os.path.isfile(rp) and os.path.isfile(cp)):
            # A query that failed in either arm has no result to compare; the
            # status columns already say so.
            print(f"{q:>4} {rs:>10} {cs:>10}  - (no result file in one arm)")
            missing += 1
            continue
        a, b = open(rp, errors="replace").read(), open(cp, errors="replace").read()
        if a == b:
            print(f"{q:>4} {rs:>10} {cs:>10}  IDENTICAL")
            same += 1
        else:
            print(f"{q:>4} {rs:>10} {cs:>10}  *** DIFFERS ***")
            diff += 1
            al, bl = a.splitlines(), b.splitlines()
            if len(al) != len(bl):
                print(f"       row count {len(al)} -> {len(bl)}")
            for i, (x, y) in enumerate(zip(al, bl)):
                if x != y:
                    print(f"       first diff at row {i}:")
                    print(f"         ref : {x[:150]}")
                    print(f"         cand: {y[:150]}")
                    break
    print(f"\nidentical={same}  differing={diff}  uncomparable={missing}")
    return 1 if diff else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
