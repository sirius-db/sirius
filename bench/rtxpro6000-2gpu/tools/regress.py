#!/usr/bin/env python3
"""Regression check: new sweep vs a baseline CSV.

Flags three distinct regression classes, because they have different causes:
  STATUS  - a query that passed in the baseline no longer passes  (hard regression)
  TIMING  - warm median moved beyond the noise band               (perf regression)
  MISSING - a baseline query absent from the new run              (coverage gap)

The timing band defaults to +-15%: run-to-run spread on this harness is a few percent,
so 15% is loose enough not to cry wolf and tight enough to catch a real change.

Usage: regress.py <new.csv> <baseline.csv> [band_pct]
"""
import collections
import csv
import statistics
import sys


def load(path):
    warm, status = collections.defaultdict(list), collections.defaultdict(set)
    for r in csv.DictReader(open(path)):
        status[r["query"]].add(r["status"])
        if r.get("phase", "warm") == "warm" and r["status"] == "pass":
            warm[r["query"]].append(int(r["ms"]))
    return warm, status


new_p, base_p = sys.argv[1], sys.argv[2]
band = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

nw, ns = load(new_p)
bw, bs = load(base_p)

rows, regressions = [], []
for q in sorted(bw):                      # iterate the BASELINE's passing set
    b = statistics.median(bw[q])
    if q not in nw or not nw[q]:
        st = ",".join(sorted(ns.get(q, {"absent"})))
        rows.append((q, f"{b:.0f}", "-", "", f"STATUS: now {st}"))
        regressions.append((q, "STATUS", f"passed in baseline, now {st}"))
        continue
    n = statistics.median(nw[q])
    d = (n - b) / b * 100
    flag = ""
    if d > band:
        flag = f"TIMING: {d:+.1f}% slower"
        regressions.append((q, "TIMING", f"{b:.0f} -> {n:.0f} ms ({d:+.1f}%)"))
    elif d < -band:
        flag = f"faster {d:+.1f}%"
    rows.append((q, f"{b:.0f}", f"{n:.0f}", f"{d:+.1f}%", flag))

print(f"{'query':6} {'baseline':>9} {'new':>9} {'delta':>8}  note")
for r in rows:
    print(f"{r[0]:6} {r[1]:>9} {r[2]:>9} {r[3]:>8}  {r[4]}")

nb = [statistics.median(bw[q]) for q in sorted(bw) if nw.get(q)]
nn = [statistics.median(nw[q]) for q in sorted(bw) if nw.get(q)]
if nb:
    print(f"\ncommon-query total: baseline {sum(nb):.0f} ms -> new {sum(nn):.0f} ms "
          f"({(sum(nn)/sum(nb)-1)*100:+.1f}%)")
print(f"baseline passing: {len(bw)}   new passing (of those): {sum(1 for q in bw if nw.get(q))}")

if regressions:
    print(f"\n!! {len(regressions)} REGRESSION(S) (band +-{band:g}%)")
    for q, kind, detail in regressions:
        print(f"   {q}  {kind}: {detail}")
else:
    print(f"\nNo regressions (band +-{band:g}%).")
