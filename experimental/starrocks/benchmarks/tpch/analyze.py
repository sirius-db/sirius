#!/usr/bin/env python3
"""A-vs-B TPC-H comparison: markdown table + grouped bar plot from two bench.sh CSVs.

Usage: analyze.py [--allow-mismatch] <a_timings.csv> <b_timings.csv> [out.md] [out.png]

Compares the row counts bench.sh records, not just the times, so a query answering
with the wrong number of rows cannot score as a speedup:

  * the two engines disagreeing on a query's row count => MISMATCH;
  * one engine disagreeing with itself across its own runs => UNSTABLE (no single
    row count exists to compare).

Both are listed at the top of the markdown, excluded from the geometric mean, hatched
in the plot, and exit this script 1 unless --allow-mismatch.

Cold rows (bench.sh --cold: phase=cold, run 0) are kept out of the warm medians and
reported separately. CSVs without a `phase` column are read as warm.
"""
import csv
import statistics
import sys
from pathlib import Path

argv = [a for a in sys.argv[1:] if a != "--allow-mismatch"]
ALLOW_MISMATCH = len(argv) != len(sys.argv) - 1
if len(argv) < 2:
    sys.exit(__doc__.splitlines()[2].strip())

A_CSV = Path(argv[0])
B_CSV = Path(argv[1])
OUT_MD = Path(argv[2]) if len(argv) > 2 else A_CSV.parent / "results.md"
OUT_PNG = Path(argv[3]) if len(argv) > 3 else A_CSV.parent / "tpch_a_vs_b.png"


def load(path):
    """query -> {status, ms: [warm ms], rows: {warm row counts}, cold: [(status, ms, rows)]}."""
    rows = {}
    if not path.exists():
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            # Older CSVs have no phase column. bench.sh only ever wrote a run-0 row
            # when run 0 failed, so run==0 there means the same thing as phase=cold.
            phase = row.get("phase") or ("cold" if row["run"] == "0" else "warm")
            entry = rows.setdefault(
                row["query"], {"status": "n/a", "ms": [], "rows": set(), "cold": []}
            )
            ms, nrows, status = int(row["ms"]), int(row.get("rows") or 0), row["status"]
            if phase == "cold":
                entry["cold"].append((status, ms, nrows))
            elif status == "pass":
                entry["status"] = "pass"
                entry["ms"].append(ms)
                entry["rows"].add(nrows)
            elif entry["status"] != "pass":
                entry["status"] = status
    for entry in rows.values():
        # A query that never got past its first execution still needs a status.
        if entry["status"] == "n/a" and entry["cold"]:
            entry["status"] = entry["cold"][0][0]
    return rows


def median_ms(entry):
    return statistics.median(entry["ms"]) if entry and entry["ms"] else None


def rowcount(entry):
    """The engine's row count for this query, or None if it never passed."""
    return next(iter(entry["rows"])) if entry and len(entry["rows"]) == 1 else None


def unstable(entry):
    return bool(entry and len(entry["rows"]) > 1)


a, b = load(A_CSV), load(B_CSV)
queries = [f"q{i:02d}" for i in range(1, 23)]

table = [
    "| Query | A (Sirius GPU) median ms | A rows | B (StarRocks) median ms | B rows | A/B speedup |",
    "|---|---|---|---|---|---|",
]
plot_q, plot_a, plot_b = [], [], []
bad = []  # (query, kind, a_rows, b_rows, ma, mb) -- excluded from the geomean
a_pass = b_pass = both = 0
geo = []
for q in queries:
    ea, eb = a.get(q), b.get(q)
    ma, mb = median_ms(ea), median_ms(eb)
    ra, rb = rowcount(ea), rowcount(eb)
    if ma:
        a_pass += 1
    if mb:
        b_pass += 1
    sa = f"{ma:.0f}" if ma else (ea["status"] if ea else "n/a")
    sb = f"{mb:.0f}" if mb else (eb["status"] if eb else "n/a")
    ta = "unstable " + "/".join(str(r) for r in sorted(ea["rows"])) if unstable(ea) else (
        str(ra) if ra is not None else "—")
    tb = "unstable " + "/".join(str(r) for r in sorted(eb["rows"])) if unstable(eb) else (
        str(rb) if rb is not None else "—")
    if ma and mb:
        kind = None
        if unstable(ea) or unstable(eb):
            kind = "UNSTABLE"
        elif ra != rb:
            kind = "MISMATCH"
        if kind:
            bad.append((q, kind, ta, tb, ma, mb))
            speed = f"**{kind}**"
        else:
            both += 1
            ratio = mb / ma
            geo.append(ratio)
            speed = f"{ratio:.2f}x" + (" (A faster)" if ratio > 1 else " (B faster)")
        plot_q.append(q.upper())
        plot_a.append(ma)
        plot_b.append(mb)
    else:
        speed = "—"
    table.append(f"| {q.upper()} | {sa} | {ta} | {sb} | {tb} | {speed} |")

lines = []
if bad:
    lines += [
        f"## :rotating_light: ROW-COUNT DISAGREEMENT on {len(bad)} quer"
        f"{'y' if len(bad) == 1 else 'ies'} — excluded from the geometric mean",
        "",
        "The engines did not return the same number of rows, so their timings are not",
        "comparable: a faster wrong answer is not a win. Fix or explain these before",
        "quoting any speedup below.",
        "",
        "| Query | kind | A rows | B rows | A median ms | B median ms |",
        "|---|---|---|---|---|---|",
    ]
    for q, kind, ta, tb, ma, mb in bad:
        lines.append(f"| {q.upper()} | {kind} | {ta} | {tb} | {ma:.0f} | {mb:.0f} |")
    lines += [
        "",
        "(`MISMATCH` = the two engines disagree; `UNSTABLE` = one engine disagreed with",
        "itself across its own runs, so it has no single row count to compare.)",
        "",
    ]
lines += table

if geo:
    gm = statistics.geometric_mean(geo)
    lines.append(
        f"\n**Summary**: A passes {a_pass}/22, B passes {b_pass}/22, {both} comparable. "
        f"Geometric-mean speedup on comparable queries: **{gm:.2f}x** "
        f"({'A' if gm > 1 else 'B'} faster)."
        + (f" {len(bad)} quer{'y' if len(bad) == 1 else 'ies'} excluded for row-count "
           "disagreement." if bad else "")
    )

cold_rows = []
for q in queries:
    for name, eng in (("A", a.get(q)), ("B", b.get(q))):
        if not eng or not eng["cold"]:
            continue
        warm = median_ms(eng)
        for status, ms, nrows in eng["cold"]:
            ratio = f"{ms / warm:.1f}x" if warm and status == "pass" else "—"
            cold_rows.append(
                f"| {q.upper()} | {name} | {status} | {ms} | "
                f"{nrows if status == 'pass' else '—'} | "
                f"{f'{warm:.0f}' if warm else '—'} | {ratio} |"
            )
if cold_rows:
    lines += [
        "",
        "## Cold start (run 0, recorded by `bench.sh --cold`)",
        "",
        "First execution on the cluster — the run the sweep normally throws away.",
        "",
        "| Query | Engine | cold status | cold ms | cold rows | warm median ms | cold/warm |",
        "|---|---|---|---|---|---|---|",
    ] + cold_rows

OUT_MD.write_text("\n".join(lines) + "\n")
print("\n".join(lines))

if plot_q:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    flagged = {q.upper() for q, *_ in bad}
    x = np.arange(len(plot_q))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    bars_a = ax.bar(x - w / 2, plot_a, w, label="A: Sirius GPU CN", color="#76b900")
    bars_b = ax.bar(x + w / 2, plot_b, w, label="B: StarRocks BE", color="#4477aa")
    # Draw the disagreeing queries rather than dropping them -- omitting them would
    # hide the very thing the gate exists to surface.
    for i, q in enumerate(plot_q):
        if q in flagged:
            for bar in (bars_a[i], bars_b[i]):
                bar.set_hatch("//")
                bar.set_edgecolor("#cc3311")
                bar.set_linewidth(1.5)
    ax.set_yscale("log")
    ax.set_ylabel("median wall-clock ms (log scale)")
    ax.set_title("TPC-H over external parquet (FILES) — same FE topology, same host, sequential")
    ax.set_xticks(x, plot_q, rotation=45)
    handles, _ = ax.get_legend_handles_labels()
    if flagged:
        handles.append(Patch(facecolor="white", edgecolor="#cc3311", hatch="//",
                             label="row counts disagree — not comparable"))
    ax.legend(handles=handles)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nplot: {OUT_PNG}")

if bad and not ALLOW_MISMATCH:
    print(f"\nFAIL: {len(bad)} quer{'y' if len(bad) == 1 else 'ies'} with disagreeing row "
          "counts (pass --allow-mismatch to report them without failing)", file=sys.stderr)
    sys.exit(1)
