#!/usr/bin/env python3
"""A/B two run-each.sh arms: per-query time, spill, compression, and answers.

    python3 bench/s3-sf1000/ab-report.py <base>.tsv <cmp>.tsv

For every query: status and best time in each arm, the change, the bytes each
arm moved to the host and disk tiers (summed over iterations, logical
uncompressed sizes), what spill compression did in the compare arm, and whether
the two arms' result files are identical. A change that should not alter
answers -- enabling compression, a new plan set -- needs that check as well as a
timing, and comparing two GPU arms avoids needing a CPU reference, which does
not fit in host memory at these scale factors.

Spill evidence is the downgrade executor's per-request summary, logged at
debug level (run.sh's default); a run without it reports no spill.

Also reads the older 3-column TSV (query, status, seconds) by locating each
query's run directory from the TSV's name.
"""

import csv
import glob
import os
import re
import sys

DONE = re.compile(r"request done: .*?to_host: \d+/(\d+) batches/bytes, to_disk: \d+/(\d+) batches/bytes")
SPILLED = re.compile(r"spilled (\d+)B → (\d+)B compressed")
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "test", "tpch_performance", "output")


def load(tsv):
    """-> {q: {"status", "best", "dir"}}"""
    m = re.match(r"each_sf(\d+)_(.+)\.tsv$", os.path.basename(tsv))
    name = m.group(2) if m else None
    out = {}
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            q = int(re.sub(r"\D", "", row["query"]))
            best = row.get("best_s", row.get("seconds", "")) or "-"
            d = row.get("dir", "-")
            if (not d or d == "-") and name:
                hits = sorted(glob.glob(os.path.join(OUTDIR, f"tpch_*_{name}_q{q}")), key=os.path.getmtime)
                d = hits[-1] if hits else "-"
            out[q] = {
                "status": row["status"].split(":")[0],
                "best": float(best) if best not in ("-", "") else None,
                "dir": d,
            }
    return out


def spill(d):
    host = disk = cin = cout = 0
    for log in glob.glob(os.path.join(d, "log_dir", "*.log")) if d and d != "-" else []:
        with open(log, errors="replace") as f:
            for line in f:
                m = DONE.search(line)
                if m:
                    host += int(m.group(1))
                    disk += int(m.group(2))
                    continue
                m = SPILLED.search(line)
                if m:
                    cin += int(m.group(1))
                    cout += int(m.group(2))
    return host / 1e9, disk / 1e9, cin / 1e9, cout / 1e9


def answer(d, q):
    """Last-iteration result file of query q under run dir d, or None."""
    if not d or d == "-":
        return None
    base = os.path.join(d, "sirius", f"q{q}")
    files = sorted(glob.glob(os.path.join(base, "result_iter*.txt"))) or glob.glob(os.path.join(base, "result.txt"))
    return open(files[-1], errors="replace").read() if files else None


def fmt(x, w, p=2):
    return f"{x:{w}.{p}f}" if x is not None else f"{'-':>{w}}"


def main(base_tsv, cmp_tsv):
    a, b = load(base_tsv), load(cmp_tsv)
    print(f"{'q':>4} {'base':>9} {'cmp':>9} {'change':>7}  {'host GB':>13} {'disk GB':>13}  {'compressed GB':>14}  answers")
    tot_a = tot_b = 0.0
    both = 0
    for q in sorted(set(a) | set(b)):
        ra, rb = a.get(q, {"status": "absent", "best": None, "dir": "-"}), b.get(q, {"status": "absent", "best": None, "dir": "-"})
        ha, da, _, _ = spill(ra["dir"])
        hb, db, cin, cout = spill(rb["dir"])
        ta = ra["best"] if ra["status"] == "ok" else None
        tb = rb["best"] if rb["status"] == "ok" else None
        if ta is not None and tb is not None:
            change = f"{(tb / ta - 1) * 100:+6.0f}%"
            tot_a += ta
            tot_b += tb
            both += 1
        else:
            change = f"{'-':>7}"
        sa, sb = answer(ra["dir"], q), answer(rb["dir"], q)
        if sa is None or sb is None:
            verdict = "-"
        elif sa == sb:
            verdict = "same"
        elif sorted(sa.splitlines()) == sorted(sb.splitlines()):
            # Same rows; ORDER BY ties (e.g. q11's equal values) came out in a
            # different order, which SQL permits.
            verdict = "same (tie order)"
        else:
            verdict = "DIFFER"
        ca = fmt(ta, 9) if ta is not None else f"{ra['status'][:9]:>9}"
        cb = fmt(tb, 9) if tb is not None else f"{rb['status'][:9]:>9}"
        comp = f"{cin:6.1f}->{cout:5.1f}" if cin else f"{'-':>14}"
        print(f"q{q:<3} {ca} {cb} {change}  {ha:6.1f}->{hb:6.1f} {da:6.1f}->{db:6.1f}  {comp:>14}  {verdict}")
    if both:
        print(f"\ntotal over the {both} queries that ran in both arms: "
              f"{tot_a:.1f}s -> {tot_b:.1f}s ({(tot_b / tot_a - 1) * 100:+.0f}%)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
