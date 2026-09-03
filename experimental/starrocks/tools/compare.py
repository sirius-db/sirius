#!/usr/bin/env python3
"""Diff Sirius TSV results against the DuckDB oracle.

Compares row count, then cell by cell. Numeric cells compare with a relative tolerance
(decimal lowering drifts slightly); non-numeric cells must match exactly. Row ORDER is
compared as-is when the query has an ORDER BY, which every TPC-H query that cares does.

Usage: compare.py <sirius_out_dir> <oracle_dir> [rel_tol]
  sirius_out_dir holds <q>.r<N>.out (mysql --batch), oracle_dir holds <q>.tsv
Exits 0 only when every query is a MATCH, so a sweep can use it as a gate.
"""
import glob
import os
import sys

sdir, odir = sys.argv[1], sys.argv[2]
TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6


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


rows = []
for q in sorted(
    {os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(sdir, "q*.out"))}
):
    # prefer the last warm run
    cands = sorted(glob.glob(os.path.join(sdir, f"{q}.r*.out")))
    opath = os.path.join(odir, f"{q}.tsv")
    if not os.path.exists(opath):
        rows.append((q, "NO-ORACLE", "", "", ""))
        continue

    ohdr, orows = read_tsv(opath)
    verdict, detail, sn = "NO-RESULT", "", ""
    for spath in reversed(cands):
        with open(spath) as f:
            head = f.readline()
        if head.startswith("ERROR"):
            verdict, detail = "ERROR", head.strip()[:90]
            continue
        shdr, srows = read_tsv(spath)
        if shdr is None:
            verdict, detail = "EMPTY", "no output"
            continue
        sn = len(srows)
        if len(srows) != len(orows):
            verdict = "ROWS-DIFFER"
            detail = f"sirius={len(srows)} oracle={len(orows)}"
            break
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
        break
    rows.append((q, verdict, detail, str(sn), str(len(orows))))

w = max(len(r[1]) for r in rows)
print(f"{'query':6} {'verdict':{w}}  detail")
for q, v, d, sn, on in rows:
    print(f"{q:6} {v:{w}}  {d}")

nmatch = sum(1 for r in rows if r[1] == "MATCH")
print(f"\n{nmatch}/{len(rows)} match the DuckDB oracle within rel tol {TOL:g}")
sys.exit(0 if nmatch == len(rows) else 1)
