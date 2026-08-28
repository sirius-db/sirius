#!/usr/bin/env python3
"""Compare two bench.sh timing CSVs (unpinned arm A vs pinned arm B).

Usage: compare.py A/timings.csv B/timings.csv [B/fix.csv ...]

Extra CSVs overlay earlier ones per query (e.g. re-measurements after a re-pin
fix). Prints the per-query table (warm-run means) and totals over the queries
that pass in BOTH arms.
"""
import collections
import csv
import sys


def load(paths):
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in paths:
        per_query = collections.defaultdict(lambda: collections.defaultdict(list))
        for row in csv.DictReader(open(path)):
            per_query[row["query"]][row["phase"]].append((row["status"], int(row["ms"])))
        for query, phases in per_query.items():
            data[query] = phases  # later files replace earlier per query
    return data


def warm(entry):
    runs = entry.get("warm", [])
    status = runs[0][0] if runs else entry.get("cold", [("missing", 0)])[0][0]
    ok = [ms for s, ms in runs if s == "pass"]
    return status, (sum(ok) / len(ok) if ok else None)


def main():
    a = load([sys.argv[1]])
    b = load(sys.argv[2:])
    tot_a = tot_b = 0.0
    both = 0
    print(f"{'query':6} {'A status':9} {'A ms':>8} {'B status':9} {'B ms':>8} {'speedup':>8}")
    for query in sorted(set(a) | set(b)):
        sa, ma = warm(a.get(query, {}))
        sb, mb = warm(b.get(query, {}))
        speedup = f"{ma / mb:.2f}x" if ma and mb else "-"
        print(f"{query:6} {sa:9} {ma or 0:8.0f} {sb:9} {mb or 0:8.0f} {speedup:>8}")
        if ma and mb:
            tot_a += ma
            tot_b += mb
            both += 1
    if both:
        print(f"\nTOTAL over {both} queries passing in both arms: "
              f"A={tot_a / 1000:.1f}s  B={tot_b / 1000:.1f}s  speedup={tot_a / tot_b:.2f}x")


if __name__ == "__main__":
    main()
