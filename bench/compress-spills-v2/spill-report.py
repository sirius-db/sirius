#!/usr/bin/env python3
"""Attribute spilling to queries for one benchmark run, and diff two runs.

Reads the per-query Sirius logs a `performance_test.py` run leaves under
<benchmark_dir>/sirius/q<N>/sirius.log, plus <benchmark_dir>/csv/runtimes.csv.

Spill evidence comes from the downgrade_executor summary line (debug level):

  [downgrade] [GPU:0] request monitor done: 3 batches, 2112 bytes in 6.04 ms
  (0.3 MB/s) | repos: ... | to_host: 3/2112 batches/bytes, to_disk: 0/0 batches/bytes

so a run must have been made with SIRIUS_LOG_LEVEL=debug or nothing is reported.
`to_disk` bytes are the ones that actually reached /mnt/datasets/sirius_spill;
`to_host` stayed in the pinned host tier.

Usage:
  spill-report.py <benchmark_dir>                 # one harness run
  spill-report.py <arm>.tsv                       # one run-suite.sh arm
  spill-report.py <base>.tsv <cmp>.tsv            # A/B of two arms
  spill-report.py <baseline_dir> <compare_dir>    # A/B
"""

import csv
import os
import re
import sys
from collections import defaultdict

# downgrade_executor.cpp emits one of these per satisfied request. Kept loose on
# purpose (only the fields we aggregate are captured) so it survives minor
# format churn in the parts we do not read.
SUMMARY_RE = re.compile(
    r"\[downgrade\] \[(?P<tier>GPU|HOST|DISK):(?P<dev>-?\d+)\] "
    r"request (?:monitor )?done: (?P<batches>\d+) batches, (?P<bytes>\d+) bytes in "
    r"(?P<ms>[\d.]+) ms .*?"
    r"to_host: (?P<to_host_b>\d+)/(?P<to_host_bytes>\d+) batches/bytes, "
    r"to_disk: (?P<to_disk_b>\d+)/(?P<to_disk_bytes>\d+) batches/bytes"
)


def gb(n):
    return n / 1e9


def scan_log(path, acc=None):
    """Accumulate downgrade summary fields from one log file."""
    acc = acc if acc is not None else defaultdict(int)
    with open(path, errors="replace") as f:
        for line in f:
            if "[downgrade] [" not in line:
                continue
            sm = SUMMARY_RE.search(line)
            if not sm:
                continue
            acc["requests"] += 1
            acc["bytes"] += int(sm.group("bytes"))
            acc["to_host_bytes"] += int(sm.group("to_host_bytes"))
            acc["to_disk_bytes"] += int(sm.group("to_disk_bytes"))
            acc["ms"] += float(sm.group("ms"))
    return acc


def parse_run(bench_dir):
    """-> {qnum: {requests, bytes, to_host_bytes, to_disk_bytes, ms}}"""
    per_q = {}
    sirius_dir = os.path.join(bench_dir, "sirius")
    if not os.path.isdir(sirius_dir):
        sys.exit(f"no sirius/ dir under {bench_dir}")
    for entry in sorted(os.listdir(sirius_dir)):
        m = re.fullmatch(r"q(\d+)", entry)
        if not m:
            continue
        qnum = int(m.group(1))
        path = os.path.join(sirius_dir, entry, "sirius.log")
        if not os.path.isfile(path):
            continue
        per_q[qnum] = scan_log(path)
    return per_q


def parse_suite(tsv_path):
    """Aggregate a run-suite.sh arm: one benchmark dir per query.

    Each dir holds a single query, so the whole combined log under log_dir/
    belongs to it -- no need for the per-query split (which never runs when the
    harness dies mid-run anyway).

    -> ({qnum: acc}, {qnum: best_s}, {qnum: status})
    """
    per_q, times, status = {}, {}, {}
    with open(tsv_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            q = int(re.sub(r"\D", "", row["query"]))
            status[q] = row["status"]
            if row["best_s"] not in ("-", ""):
                times[q] = float(row["best_s"])
            d = row["dir"]
            acc = defaultdict(int)
            log_dir = os.path.join(d, "log_dir") if d not in ("-", "") else None
            if log_dir and os.path.isdir(log_dir):
                for name in sorted(os.listdir(log_dir)):
                    if name.endswith(".log"):
                        scan_log(os.path.join(log_dir, name), acc)
            per_q[q] = acc
    return per_q, times, status


def report_suite(tsv_path, label=None):
    per_q, times, status = parse_suite(tsv_path)
    print(f"\n=== {label or os.path.basename(tsv_path)} ===")
    print(f"{'q':>4} {'status':>9} {'best_s':>9} {'reqs':>7} {'evicted_GB':>11} "
          f"{'to_host_GB':>11} {'to_disk_GB':>11} {'dg_ms':>9}")
    for q in sorted(per_q):
        a = per_q[q]
        t = times.get(q)
        print(f"{q:>4} {status[q]:>9} {(f'{t:.4f}' if t else '-'):>9} "
              f"{a['requests']:>7} {gb(a['bytes']):>11.2f} "
              f"{gb(a['to_host_bytes']):>11.2f} {gb(a['to_disk_bytes']):>11.2f} "
              f"{a['ms']:>9.1f}")
    ok = [q for q in status if status[q] == "ok"]
    print(f"\nran: {len(ok)}/{len(status)}   "
          f"not-ok: {sorted(q for q in status if status[q] != 'ok')}")
    print(f"total (ok queries): {sum(times.values()):.4f}s")
    sp = sorted(q for q, a in per_q.items() if a["requests"])
    disk = sorted(q for q, a in per_q.items() if a["to_disk_bytes"])
    print(f"spilled at all : {sp or 'none'}")
    print(f"reached disk   : {disk or 'none'}")
    print(f"total evicted  : {gb(sum(a['bytes'] for a in per_q.values())):.2f} GB "
          f"({gb(sum(a['to_disk_bytes'] for a in per_q.values())):.2f} GB to disk)")
    return per_q, times, status


def report_suite_diff(base_tsv, cmp_tsv):
    bq, bt, bs = report_suite(base_tsv, "BASELINE")
    cq, ct, cs = report_suite(cmp_tsv, "SPILL COMPRESSION")
    interesting = sorted(
        q for q in set(bq) | set(cq)
        if bq.get(q, {}).get("requests") or cq.get(q, {}).get("requests")
        or bs.get(q) != "ok" or cs.get(q) != "ok"
    )
    print("\n=== A/B (queries that spilled, or failed in either arm) ===")
    if not interesting:
        print("neither arm spilled and both arms ran clean -- nothing to compare.")
        return
    print(f"{'q':>4} {'base':>10} {'comp':>10} {'delta%':>8} "
          f"{'base_ev_GB':>11} {'comp_ev_GB':>11} {'base_dg_ms':>11} {'comp_dg_ms':>11}")
    for q in interesting:
        b, c = bt.get(q), ct.get(q)
        bl = f"{b:.4f}" if b else bs.get(q, "-")
        cl = f"{c:.4f}" if c else cs.get(q, "-")
        pct = f"{100.0 * (c - b) / b:+.1f}%" if (b and c) else "-"
        print(f"{q:>4} {bl:>10} {cl:>10} {pct:>8} "
              f"{gb(bq.get(q, {}).get('bytes', 0)):>11.2f} "
              f"{gb(cq.get(q, {}).get('bytes', 0)):>11.2f} "
              f"{bq.get(q, {}).get('ms', 0):>11.1f} "
              f"{cq.get(q, {}).get('ms', 0):>11.1f}")
    common = sorted(set(bt) & set(ct))
    bsum, csum = sum(bt[q] for q in common), sum(ct[q] for q in common)
    if bsum:
        print(f"\ncommon-query total ({len(common)} queries): "
              f"{bsum:.3f}s -> {csum:.3f}s ({100.0 * (csum - bsum) / bsum:+.1f}%)")


def parse_runtimes(bench_dir):
    """-> {qnum: best-of-N seconds} from csv/runtimes.csv."""
    path = os.path.join(bench_dir, "csv", "runtimes.csv")
    best = {}
    if not os.path.isfile(path):
        return best
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("engine") not in (None, "", "sirius", "gpu"):
                continue
            try:
                q = int(re.sub(r"\D", "", row.get("query", "")))
                t = float(row.get("runtime_s") or row.get("runtime") or "nan")
            except (ValueError, TypeError):
                continue
            if t == t:  # not NaN
                best[q] = min(best.get(q, float("inf")), t)
    return best


def report_one(bench_dir):
    per_q = parse_run(bench_dir)
    times = parse_runtimes(bench_dir)
    spillers = {q: a for q, a in per_q.items() if a["requests"]}

    print(f"\n=== {os.path.basename(bench_dir)} ===")
    print(f"{'q':>4} {'best_s':>9} {'reqs':>7} {'evicted_GB':>11} "
          f"{'to_host_GB':>11} {'to_disk_GB':>11} {'dg_ms':>10}")
    for q in sorted(per_q):
        a = per_q[q]
        t = times.get(q)
        print(f"{q:>4} {(f'{t:.3f}' if t else '-'):>9} {a['requests']:>7} "
              f"{gb(a['bytes']):>11.2f} {gb(a['to_host_bytes']):>11.2f} "
              f"{gb(a['to_disk_bytes']):>11.2f} {a['ms']:>10.1f}")

    tot = sum(times.values()) if times else 0.0
    print(f"\nsuite best-of-N total: {tot:.3f}s over {len(times)} queries")
    if spillers:
        print(f"queries that spilled at all : {sorted(spillers)}")
        to_disk = sorted(q for q, a in spillers.items() if a["to_disk_bytes"])
        print(f"queries that reached disk   : {to_disk or 'none'}")
    else:
        print("NO query issued a downgrade request -- nothing spilled.")
        print("(if that is unexpected, check the run used SIRIUS_LOG_LEVEL=debug)")
    return per_q, times


def report_diff(base_dir, cmp_dir):
    base_q, base_t = report_one(base_dir)
    cmp_q, cmp_t = report_one(cmp_dir)

    interesting = sorted(
        q for q in set(base_q) | set(cmp_q)
        if base_q.get(q, {}).get("requests") or cmp_q.get(q, {}).get("requests")
    )
    print("\n=== A/B on queries that spilled in either arm ===")
    if not interesting:
        print("neither arm spilled -- spill compression cannot help here.")
        return
    print(f"{'q':>4} {'base_s':>9} {'comp_s':>9} {'delta':>9} {'delta%':>8} "
          f"{'base_disk_GB':>13} {'comp_disk_GB':>13}")
    for q in interesting:
        b, c = base_t.get(q), cmp_t.get(q)
        d = (c - b) if (b and c) else None
        pct = (100.0 * d / b) if (d is not None and b) else None
        print(f"{q:>4} {(f'{b:.3f}' if b else '-'):>9} "
              f"{(f'{c:.3f}' if c else '-'):>9} "
              f"{(f'{d:+.3f}' if d is not None else '-'):>9} "
              f"{(f'{pct:+.1f}%' if pct is not None else '-'):>8} "
              f"{gb(base_q.get(q, {}).get('to_disk_bytes', 0)):>13.2f} "
              f"{gb(cmp_q.get(q, {}).get('to_disk_bytes', 0)):>13.2f}")

    bt = sum(base_t.get(q, 0) for q in interesting)
    ct = sum(cmp_t.get(q, 0) for q in interesting)
    if bt:
        print(f"\nspilling-query subtotal: {bt:.3f}s -> {ct:.3f}s "
              f"({100.0 * (ct - bt) / bt:+.1f}%)")
    bs, cs = sum(base_t.values()), sum(cmp_t.values())
    if bs:
        print(f"whole suite            : {bs:.3f}s -> {cs:.3f}s "
              f"({100.0 * (cs - bs) / bs:+.1f}%)")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    # run-suite.sh arms are .tsv summaries; a single harness run is a directory.
    if all(a.endswith(".tsv") for a in args):
        if len(args) == 1:
            report_suite(args[0])
        elif len(args) == 2:
            report_suite_diff(args[0], args[1])
        else:
            sys.exit(__doc__)
    elif len(args) == 1:
        report_one(args[0])
    elif len(args) == 2:
        report_diff(args[0], args[1])
    else:
        sys.exit(__doc__)
