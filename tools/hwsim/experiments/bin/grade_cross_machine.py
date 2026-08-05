#!/usr/bin/env python3
"""Target-box half of the definitive cross-machine experiment (WS19).

Grade a predictions CSV (from ``predict_cross_machine.py``, produced on the
SOURCE box) against this box's measured baseline telemetry session.
See docs/cross-machine-experiment.md for the success bands per tier.

Usage (on the target box, e.g. the RTX PRO 6000 workstation):
  python3 grade_cross_machine.py predictions_sf100.csv \
      hwsim_run/telemetry_data/<baseline-session-uuid> [--iters 2,3]

Real wall per query = median of the given iterations' traced exec walls.
Prints per-query E% for the nominal and optimistic vectors, medians,
time-weighted suite aggregate, Spearman rank correlation, and band coverage.
"""

import argparse
import csv
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sim"))

from hwsim.trace import load_session_model  # noqa: E402

QNUM_RE = re.compile(r"q(\d+)")


def qnum(label):
    m = QNUM_RE.search(label)
    return int(m.group(1)) if m else None


def iter_of(label):
    m = re.search(r"iter(\d+)", label)
    return int(m.group(1)) if m else None


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_csv")
    ap.add_argument("baseline_session_dir")
    ap.add_argument(
        "--iters",
        default="2,3",
        help="iterations to median for the real wall (default 2,3)",
    )
    args = ap.parse_args()
    iters = {int(x) for x in args.iters.split(",")}

    with open(args.predictions_csv) as f:
        preds = list(csv.DictReader(f))

    model = load_session_model(args.baseline_session_dir)
    real = {}
    for q in model.queries:
        n, it = qnum(q.label), iter_of(q.label)
        if n is None or it not in iters or not q.exec_wall_ns:
            continue
        real.setdefault(n, []).append(q.exec_wall_ns / 1e6)
    real = {n: statistics.median(v) for n, v in real.items()}

    rows, e_nom, e_opt, in_band = [], [], [], 0
    sum_real = sum_nom = sum_opt = 0.0
    for p in preds:
        n = int(p["q"])
        if n not in real:
            print(f"q{n}: no measured wall in baseline session", file=sys.stderr)
            continue
        r = real[n]
        nom, opt = float(p["pred_nominal_ms"]), float(p["pred_optimistic_ms"])
        en, eo = 100.0 * (nom / r - 1.0), 100.0 * (opt / r - 1.0)
        lo, hi = sorted((nom, opt))
        band = lo <= r <= hi
        in_band += band
        e_nom.append(en)
        e_opt.append(eo)
        sum_real += r
        sum_nom += nom
        sum_opt += opt
        rows.append((n, r, nom, en, opt, eo, p["path"], band))

    rows.sort()
    print(
        f"{'q':>3} {'real ms':>10} {'nominal':>10} {'E_nom%':>8} "
        f"{'optim.':>10} {'E_opt%':>8} {'path':>7} {'in-band':>7}"
    )
    for n, r, nom, en, opt, eo, path, band in rows:
        print(
            f"{n:>3} {r:>10.1f} {nom:>10.1f} {en:>+8.1f} "
            f"{opt:>10.1f} {eo:>+8.1f} {path:>7} {str(band):>7}"
        )
    if not rows:
        print("nothing graded")
        return 1
    med = statistics.median
    print(
        f"\nnominal   : median E {med(e_nom):+.1f}%, median |E| "
        f"{med([abs(x) for x in e_nom]):.1f}%, suite time-weighted "
        f"{100.0 * (sum_nom / sum_real - 1.0):+.1f}%"
    )
    print(
        f"optimistic: median E {med(e_opt):+.1f}%, median |E| "
        f"{med([abs(x) for x in e_opt]):.1f}%, suite time-weighted "
        f"{100.0 * (sum_opt / sum_real - 1.0):+.1f}%"
    )
    reals = [r[1] for r in rows]
    noms = [r[2] for r in rows]
    print(f"rank rho(real, nominal): {spearman(reals, noms):.2f}")
    print(f"band coverage: {in_band}/{len(rows)} reals inside [nom, opt]")
    print(
        "success bands (docs/cross-machine-experiment.md): grade per tier, "
        "not just the medians."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
