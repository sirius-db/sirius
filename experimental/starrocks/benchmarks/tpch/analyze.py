#!/usr/bin/env python3
"""A-vs-B TPC-H comparison: markdown table + grouped bar plot from two bench.sh CSVs.

Usage: analyze.py <a_timings.csv> <b_timings.csv> [out.md] [out.png]
"""
import csv
import statistics
import sys
from pathlib import Path

A_CSV = Path(sys.argv[1])
B_CSV = Path(sys.argv[2])
OUT_MD = Path(sys.argv[3]) if len(sys.argv) > 3 else A_CSV.parent / "results.md"
OUT_PNG = Path(sys.argv[4]) if len(sys.argv) > 4 else A_CSV.parent / "tpch_a_vs_b.png"


def load(path):
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            entry = rows.setdefault(row["query"], {"status": row["status"], "ms": []})
            if row["status"] == "pass":
                entry["status"] = "pass"
                entry["ms"].append(int(row["ms"]))
            elif entry["status"] != "pass":
                entry["status"] = row["status"]
    return rows


a, b = load(A_CSV), load(B_CSV)
queries = [f"q{i:02d}" for i in range(1, 23)]

lines = [
    "| Query | A (Sirius GPU) median ms | B (StarRocks) median ms | A/B speedup |",
    "|---|---|---|---|",
]
plot_q, plot_a, plot_b = [], [], []
a_pass = b_pass = both = 0
geo = []
for q in queries:
    ea, eb = a.get(q), b.get(q)
    ma = statistics.median(ea["ms"]) if ea and ea["ms"] else None
    mb = statistics.median(eb["ms"]) if eb and eb["ms"] else None
    if ma:
        a_pass += 1
    if mb:
        b_pass += 1
    sa = f"{ma:.0f}" if ma else (ea["status"] if ea else "n/a")
    sb = f"{mb:.0f}" if mb else (eb["status"] if eb else "n/a")
    if ma and mb:
        both += 1
        ratio = mb / ma
        geo.append(ratio)
        speed = f"{ratio:.2f}x" + (" (A faster)" if ratio > 1 else " (B faster)")
        plot_q.append(q.upper())
        plot_a.append(ma)
        plot_b.append(mb)
    else:
        speed = "—"
    lines.append(f"| {q.upper()} | {sa} | {sb} | {speed} |")

if geo:
    gm = statistics.geometric_mean(geo)
    lines.append(
        f"\n**Summary**: A passes {a_pass}/22, B passes {b_pass}/22, {both} comparable. "
        f"Geometric-mean speedup on comparable queries: **{gm:.2f}x** "
        f"({'A' if gm > 1 else 'B'} faster)."
    )
OUT_MD.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

if plot_q:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(plot_q))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w / 2, plot_a, w, label="A: Sirius GPU CN", color="#76b900")
    ax.bar(x + w / 2, plot_b, w, label="B: StarRocks BE", color="#4477aa")
    ax.set_yscale("log")
    ax.set_ylabel("median wall-clock ms (log scale)")
    ax.set_title("TPC-H over external parquet (FILES) — same FE topology, same host, sequential")
    ax.set_xticks(x, plot_q, rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nplot: {OUT_PNG}")
