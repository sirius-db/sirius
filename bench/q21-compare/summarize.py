#!/usr/bin/env python3
"""Summarize performance_test.py runs written by run.sh.

For each benchmark directory under the results root, prints the median over iterations of q21 and
the sum over queries of the per-query medians (TOTAL rows are ignored), the number of OOM
reschedules in its Sirius log, and PR #2468's merge decisions.

usage: summarize.py <results_root>
"""

import csv
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

TAG = re.compile(r"_iter\d+_(?P<tag>.+)$")


def per_query_medians(runtimes_csv):
    samples = defaultdict(list)
    with open(runtimes_csv) as f:
        for row in csv.reader(f):
            if len(row) != 4 or row[1] == "TOTAL" or not row[1].startswith("q"):
                continue
            try:
                value = float(row[3])
            except ValueError:
                continue
            if value == value:  # drop NaN (failed or timed-out iterations)
                samples[row[1]].append(value)
    return {q: statistics.median(v) for q, v in samples.items() if v}


def count_in_logs(bench_dir, pattern):
    hits = []
    for path in glob.glob(
        os.path.join(bench_dir, "log_dir", "**", "*.log"), recursive=True
    ):
        with open(path, errors="replace") as f:
            hits.extend(line.strip() for line in f if pattern in line)
    return hits


def main(root):
    rows = []
    for bench_dir in sorted(glob.glob(os.path.join(root, "tpch_*"))):
        m = TAG.search(os.path.basename(bench_dir))
        runtimes = os.path.join(bench_dir, "csv", "runtimes.csv")
        if not m or not os.path.isfile(runtimes):
            continue
        medians = per_query_medians(runtimes)
        retries = len(count_in_logs(bench_dir, "reschedule (retry"))
        decisions = count_in_logs(bench_dir, "[sirius_plan_exists_rewrite]")
        # One "Rewrote" line per query execution that merged; one "Not merging" per declined chain.
        merged = sum("Rewrote" in d for d in decisions)
        not_merged = sum("Not merging" in d for d in decisions)
        rows.append((m.group("tag"), medians, retries, merged, not_merged))

    print(
        "| run | q21 median (s) | queries | sum of medians (s) | OOM reschedules "
        "| #2468 merged / kept plans |"
    )
    print("|---|---|---|---|---|---|")
    for tag, medians, retries, merged, not_merged in rows:
        q21 = f"{medians['q21']:.3f}" if "q21" in medians else "-"
        total = f"{sum(medians.values()):.3f}" if medians else "-"
        decisions = f"{merged} / {not_merged}" if merged or not_merged else "-"
        print(f"| {tag} | {q21} | {len(medians)} | {total} | {retries} | {decisions} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
