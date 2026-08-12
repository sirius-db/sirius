#!/usr/bin/env python3
"""Study 1 (scale-out) figure: TPC-H SF500 across 2 / 4 / 8 A100 GPUs.

Reads the archived timing CSVs and regenerates, so re-running after a new arm lands picks
it up. No numbers are hard-coded.

A query's 2-GPU point is DROPPED WHERE IT DOES NOT EXIST rather than imputed or zeroed:
q03, q07 and q17 cannot run at 2 CNs on an 80 GiB card at all (the staging arena and the
RMM pool both scale as 1/N and their sum exceeds the card), so an absent marker is the
honest encoding. The connector then starts at 4 GPUs for those rows.

Form (per the dataviz method):
  Panel A -- magnitude per item across an ordered scale -> connected dot plot, ORDINAL ramp.
  Panel B -- compare magnitude -> grouped bars, the same ordinal ramp, one bar per doubling.
Colors are validated blue-ramp steps, not eyeballed (validate_palette.js --ordinal):
lightness monotone, adjacent dL >= 0.06, single hue (3 deg spread), light end 2.91:1.
That light end is below 3:1, so the *relief rule* applies: this figure ships visible direct
labels, and the table view lives in SCALE-OUT-SUMMARY.md.

Usage:  python3 plot_study1.py [out.png]
"""
import csv
import math
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "study1-scaleout.png")

# ---- palette (validated ordinal ramp; see module docstring) ------------------------
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"
CRITICAL = "#d03b3b"          # status:critical, reserved -- reference rule only
RAMP = {2: "#5598e7", 4: "#256abf", 8: "#0d366b"}   # more GPUs -> darker

WRONG_VALUES = {"q01", "q03", "q07", "q14", "q19"}  # FP64 decimal defect; marked with a dagger
ARMS = (2, 4, 8)


def load(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["query"], []).append(r)
    return rows


def warm_median(runs):
    v = [int(r["ms"]) for r in runs if r["phase"] == "warm" and r["status"] == "pass"]
    return st.median(v) if v else None


def passed(runs):
    return bool(runs) and all(r["status"] == "pass" for r in runs)


# SOURCE: the UNIFORM-config campaign (16 GiB arena / 62 GiB pool at every arm).
#
# Why not the per-arm validated runs, which are better configured? Because a scaling curve
# must be internally consistent, and the validated runs do not form one:
#   * There is no usable validated 2-CN arm. At 32 GiB/46 GiB the pool starves and the engine
#     livelocks in its OOM-retry loop -- q04 cold took 211 s there against 6.5 s here. That
#     measures starvation, not scaling.
#   * The two campaigns are ~2x apart on identical cells (8 CN geomean 2.03x, 4 CN 2.22x),
#     so splicing a 2-CN point from one into a 4/8 pair from the other would import a
#     confound LARGER than the effect being plotted.
# The uniform campaign ran all three arms back to back with one config, so its arms are
# comparable to each other. It is mis-sized for two of the three arms -- which is exactly why
# q03/q07/q17 have no 2-CN point -- but that is a coverage cost, not a comparability one.
data = {n: load(os.path.join(RESULTS, f"sf500-{n}cn-timings.csv")) for n in ARMS}

# The curve's backbone is 4 vs 8: every query must have both. The 2-GPU point is optional.
core = [q for q in sorted(set(data[4]) & set(data[8]))
        if passed(data[4][q]) and passed(data[8][q])]
excluded = [q for q in sorted(set(data[4]) | set(data[8])) if q not in core]

rows = []
for q in core:
    t = {n: (warm_median(data[n][q]) / 1000.0
             if q in data[n] and passed(data[n][q]) else None) for n in ARMS}
    rows.append({"q": q, "t": t, "s48": t[4] / t[8],
                 "s24": (t[2] / t[4]) if t[2] else None})
rows.sort(key=lambda r: r["s48"])          # best speedup ends up on top

y = list(range(len(rows)))
labels = [r["q"] + ("†" if r["q"] in WRONG_VALUES else "") for r in rows]
have2 = [r for r in rows if r["t"][2]]

geo48 = math.exp(sum(math.log(r["s48"]) for r in rows) / len(rows))
geo24 = (math.exp(sum(math.log(r["s24"]) for r in have2) / len(have2))
         if have2 else None)
tot = {n: sum(r["t"][n] for r in rows if r["t"][n]) for n in ARMS}

fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(14.0, 0.50 * len(rows) + 3.6),
    gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.28},
)
fig.patch.set_facecolor(SURFACE)

# ---- Panel A: connected dot plot ---------------------------------------------------
axA.set_facecolor(SURFACE)
for i, r in enumerate(rows):
    pts = [(r["t"][n], n) for n in ARMS if r["t"][n]]
    axA.plot([p[0] for p in pts], [i] * len(pts), color=BASELINE, lw=1.6,
             zorder=1, solid_capstyle="round")
    for val, n in pts:
        axA.plot(val, i, "o", ms=9.5, color=RAMP[n], zorder=2 + n,
                 mec=SURFACE, mew=2)          # 2px surface ring on overlapping marks

axA.set_yticks(y)
axA.set_yticklabels(labels, fontsize=10.5, color=INK)
axA.set_xlabel("warm median runtime  (s, log scale)", fontsize=10.5, color=INK_2)
axA.set_xscale("log")
axA.set_title("Runtime per query", fontsize=12.5, color=INK, pad=12, loc="left")
axA.grid(axis="x", color=GRID, lw=0.8, ls="-", zorder=0)   # solid hairline, never dashed
axA.set_axisbelow(True)
for s in ("top", "right", "left"):
    axA.spines[s].set_visible(False)
axA.spines["bottom"].set_color(BASELINE)
axA.tick_params(colors=MUTED, length=0)

# Relief rule: label the two ends of each row (the interior point is carried by the axis).
# Anchor by POSITION, not by arm order: more GPUs is faster, so the 8-GPU point sits at the
# LEFT end of a time axis. Labelling pts[0]/pts[-1] with fixed left/right offsets put both
# labels on the inside of the row and they collided whenever the row was short.
for i, r in enumerate(rows):
    vals = [r["t"][n] for n in ARMS if r["t"][n]]
    lo, hi = min(vals), max(vals)
    axA.annotate(f"{lo:.1f}", (lo, i), textcoords="offset points", xytext=(-10, 0),
                 va="center", ha="right", fontsize=8.8, color=INK_2)
    axA.annotate(f"{hi:.1f}", (hi, i), textcoords="offset points", xytext=(10, 0),
                 va="center", ha="left", fontsize=8.8, color=INK_2)

axA.legend(handles=[Line2D([], [], marker="o", ls="", ms=9.5, mfc=RAMP[n], mec=SURFACE,
                           mew=2, label=f"{n} GPUs") for n in ARMS],
           loc="lower right", frameon=False, fontsize=10, labelcolor=INK_2)

# ---- Panel B: per-doubling speedup -------------------------------------------------
axB.set_facecolor(SURFACE)
h = 0.36
for i, r in enumerate(rows):
    if r["s24"]:
        axB.barh(i + h / 2 + 0.02, r["s24"], height=h, color=RAMP[4], zorder=2)
        axB.annotate(f"{r['s24']:.2f}×", (r["s24"], i + h / 2 + 0.02),
                     textcoords="offset points", xytext=(5, 0), va="center",
                     fontsize=8.2, color=INK_2)
    axB.barh(i - h / 2 - 0.02, r["s48"], height=h, color=RAMP[8], zorder=2)
    axB.annotate(f"{r['s48']:.2f}×", (r["s48"], i - h / 2 - 0.02),
                 textcoords="offset points", xytext=(5, 0), va="center",
                 fontsize=8.2, color=INK_2)

axB.set_yticks(y)
axB.set_yticklabels([])
axB.set_xlabel("speedup per GPU doubling  (×)", fontsize=10.5, color=INK_2)
axB.set_title("Speedup per doubling", fontsize=12.5, color=INK, pad=12, loc="left")
axB.grid(axis="x", color=GRID, lw=0.8, ls="-", zorder=0)
axB.set_axisbelow(True)
for s in ("top", "right", "left"):
    axB.spines[s].set_visible(False)
axB.spines["bottom"].set_color(BASELINE)
axB.tick_params(colors=MUTED, length=0)

best = max([r["s24"] for r in have2] + [r["s48"] for r in rows])
axB.set_xlim(0, max(2.6, best * 1.22))
axB.axvline(2.0, color=INK_2, lw=1.4, zorder=4)
# Reference labels live in the headroom above the top bar, never over a bar.
axB.annotate("linear 2.0×", (2.0, len(rows) - 1 + 0.62), textcoords="offset points",
             xytext=(6, 0), fontsize=9, color=INK_2, va="center")

axB.legend(handles=[Patch(facecolor=RAMP[4], label="2 → 4 GPUs"),
                    Patch(facecolor=RAMP[8], label="4 → 8 GPUs")],
           loc="lower right", frameon=False, fontsize=10, labelcolor=INK_2)

for ax in (axA, axB):                       # identical limits keep the rows aligned
    ax.set_ylim(-0.78, len(rows) - 1 + 0.98)

# ---- titles & provenance -----------------------------------------------------------
fig.suptitle("Sirius GPU scale-out · TPC-H SF500 · 2 → 4 → 8 × A100 80 GB",
             fontsize=15.5, color=INK, x=0.052, ha="left", y=0.988)

sub1 = (f"{len(rows)} queries · 4→8 GPUs {geo48:.2f}× geomean "
        f"({geo48 / 2 * 100:.0f}% of linear)")
if geo24:
    sub1 += f" · 2→4 GPUs {geo24:.2f}× over the {len(have2)} that run at 2 GPUs"
sub2 = ("one memory config across all arms (16 GiB arena / 62 GiB pool) so the arms are "
        "comparable · n=3 warm runs per cell")
fig.text(0.052, 0.958, sub1, fontsize=10.5, color=INK_2, ha="left")
fig.text(0.052, 0.936, sub2, fontsize=9.6, color=MUTED, ha="left")

missing2 = [r["q"] for r in rows if not r["t"][2]]
foots = [
    "† values numerically wrong (FP64 decimal lowering) — timing-valid only; no result here "
    "has been diffed against an oracle.",
    "Each cell is n=3 warm runs; within-cell spreads reach 2.5×, so per-query rankings are "
    "not reliable. The 8-GPU arm ran first in every campaign — run order is confounded with "
    "GPU count.",
]
if missing2:
    foots.append("No 2-GPU point for " + ", ".join(missing2) +
                 ": the 16 GiB arena is below their 2-CN minimum. No 2-CN split tried "
                 "(16/62, 32/46, 48/30, 64/14 GiB arena/pool) runs all three.")
if excluded:
    foots.append("Excluded: " + ", ".join(excluded) +
                 " — q11 returns empty; q02 fails at 8 GPUs; q15 non-deterministic; "
                 "q17 needs a >16 GiB arena at 4 CN; q21 lease-lifecycle bug.")
for k, line in enumerate(foots):
    fig.text(0.052, 0.060 - 0.0165 * k, line, fontsize=8.5, color=MUTED, ha="left")

fig.subplots_adjust(top=0.893, bottom=0.150, left=0.052, right=0.977)
fig.savefig(OUT, dpi=200, facecolor=SURFACE)

print(f"wrote {OUT}")
print(f"  {len(rows)} queries; {len(have2)} have a 2-GPU point")
print(f"  4→8 geomean {geo48:.3f}x" + (f" · 2→4 geomean {geo24:.3f}x" if geo24 else ""))
print(f"  totals: " + "  ".join(f"{n}GPU={tot[n]:.1f}s" for n in ARMS))
if missing2:
    print(f"  no 2-GPU point: {' '.join(missing2)}")
if excluded:
    print(f"  excluded: {' '.join(excluded)}")
