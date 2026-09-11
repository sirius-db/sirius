#!/usr/bin/env python3
"""Diff Sirius TSV results against the DuckDB oracle, every run of every query.

Compares row count, then cell by cell. Numeric cells compare with a relative tolerance
(decimal lowering drifts slightly); non-numeric cells must match exactly. Row ORDER is
compared as-is when the query has an ORDER BY, which every TPC-H query that cares does.

Every `<q>.rN.out` is compared, not only the last one: a query whose cold run returned zero
rows and whose warm runs matched is a flake, and a flake is a failure. The per-query verdict
is the worst verdict over its runs; the per-run verdicts are printed under it when they
differ.

Usage: compare.py <sirius_out_dir> <oracle_dir> [rel_tol]
  sirius_out_dir holds <q>.r<N>.out (mysql --batch), oracle_dir holds <q>.tsv
Exits 0 only when every run of every query is a MATCH, so a sweep can use it as a gate.
"""
import glob
import os
import re
import sys

sdir, odir = sys.argv[1], sys.argv[2]
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6

# Worst first. NO-ORACLE is not a run verdict; it is reported per query and never counts as
# a MATCH.
SEVERITY = ["NO-RESULT", "ERROR", "EMPTY", "ROWS-DIFFER", "VALUES-DIFFER", "MATCH"]


def num(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def read_tsv(path):
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip() != ""]
    if not lines:
        return None, []
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


def run_index(path):
    m = re.search(r"\.r(\d+)\.out$", path)
    return int(m.group(1)) if m else -1


def compare_run(spath, orows):
    """One run against the oracle rows: (verdict, detail)."""
    with open(spath) as f:
        head = f.readline()
    if head.startswith("ERROR"):
        return "ERROR", head.strip()[:90]
    shdr, srows = read_tsv(spath)
    if shdr is None:
        return "EMPTY", "no output"
    if len(srows) != len(orows):
        return "ROWS-DIFFER", f"sirius={len(srows)} oracle={len(orows)}"
    bad, worst = 0, 0.0
    for si, oi in zip(srows, orows):
        if len(si) != len(oi):
            bad += 1
            continue
        for a, b in zip(si, oi):
            fa, fb = num(a), num(b)
            if fa is not None and fb is not None:
                d = abs(fa - fb) / max(abs(fb), 1e-12) if fb != 0 else abs(fa)
                worst = max(worst, d)
                if d > TOL:
                    bad += 1
            elif a.strip() != b.strip():
                bad += 1
    verdict = "MATCH" if bad == 0 else "VALUES-DIFFER"
    detail = f"rows={len(srows)} maxreldiff={worst:.3e}" + (
        f" badcells={bad}" if bad else ""
    )
    return verdict, detail


summary = []  # (query, verdict, detail, [(run, verdict, detail), ...])
for q in sorted(
    {os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(sdir, "q*.out"))}
):
    cands = sorted(glob.glob(os.path.join(sdir, f"{q}.r*.out")), key=run_index)
    opath = os.path.join(odir, f"{q}.tsv")
    if not os.path.exists(opath):
        summary.append((q, "NO-ORACLE", "", []))
        continue
    _, orows = read_tsv(opath)
    runs = [(run_index(p),) + compare_run(p, orows) for p in cands]
    if not runs:
        summary.append((q, "NO-RESULT", "no runs", []))
        continue
    worst = min(runs, key=lambda r: SEVERITY.index(r[1]))
    detail = worst[2]
    if len({r[1] for r in runs}) > 1:
        bad_runs = ",".join(f"r{r[0]}" for r in runs if r[1] != "MATCH")
        detail = f"{detail} ({bad_runs} of {len(runs)} runs)"
    summary.append((q, worst[1], detail, runs))

w = max(len(r[1]) for r in summary)
print(f"{'query':6} {'verdict':{w}}  detail")
for q, v, d, runs in summary:
    print(f"{q:6} {v:{w}}  {d}")
    if len({r[1] for r in runs}) > 1:
        for run, rv, rd in runs:
            print(f"  r{run:<3} {rv:{w}}  {rd}")

nmatch = sum(1 for r in summary if r[1] == "MATCH")
nruns = sum(len(r[3]) for r in summary)
print(
    f"\n{nmatch}/{len(summary)} queries match the DuckDB oracle within rel tol {TOL:g} "
    f"on every run ({nruns} runs compared)"
)
sys.exit(0 if nmatch == len(summary) else 1)
