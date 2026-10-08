#!/usr/bin/env python3
"""Per query: how much device memory the spill-compression encoder really used.

    python3 bench/s3-sf1000/arena-report.py <arm>.tsv [<arm>.tsv ...]

Reads the QueryEnd lines SiriusContext logs after every query (run.sh's default
log level is enough), from the run directories a run-each.sh TSV points at:

  [compression_arena]              arena mode: capacity, and the peak bytes the
                                   encoders held in it during the query
  [compression_encode_reservation] no-arena mode: reservations granted/declined,
                                   the peak sum of in-flight reservations, and the
                                   most any one encode actually allocated
  [compression_alloc]              SIRIUS_COMPRESSION_ALLOC_STATS=1 (run.sh sets it
                                   for SPILL_COMPRESSION=1): the encoder's explicit
                                   allocations -- peak, largest, failures
  [gpu_pool]                       the query pool's peak, for scale

A directory holds every iteration of its query; the maximum over them is shown.
"""

import glob
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "ab_report", os.path.join(HERE, "ab-report.py")
)
ab_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab_report)

ARENA = re.compile(
    r"\[compression_arena\] QueryEnd .*capacity=(\d+) bytes used=(\d+) bytes peak=(\d+) bytes"
)
ENC = re.compile(
    r"\[compression_encode_reservation\] QueryEnd .*mode=(\w+) granted=(\d+) declined=(\d+) "
    r"outstanding=\d+ bytes peak_outstanding=(\d+) bytes largest_reserved=(\d+) bytes largest_used=(\d+) bytes"
)
ALLOC = re.compile(
    r"\[compression_alloc\] QueryEnd .*peak=(\d+)MiB largest=(\d+)MiB failures=(\d+)"
)
GPU = re.compile(r"\[gpu_pool\] GPU:\d+ QueryEnd .*peak=(\d+) bytes")


def scan(d):
    r = {
        "cap": None,
        "arena_peak": None,
        "mode": None,
        "granted": 0,
        "declined": 0,
        "res_peak": None,
        "res_used": None,
        "alloc_peak": None,
        "alloc_fail": 0,
        "gpu_peak": None,
    }

    def mx(k, v):
        r[k] = v if r[k] is None else max(r[k], v)

    for log in glob.glob(os.path.join(d, "log_dir", "*.log")) if d and d != "-" else []:
        with open(log, errors="replace") as f:
            for line in f:
                if "QueryEnd" not in line:
                    continue
                if m := ARENA.search(line):
                    r["cap"] = int(m.group(1))
                    mx("arena_peak", int(m.group(3)))
                elif m := ENC.search(line):
                    r["mode"] = m.group(1)
                    r["granted"] += int(m.group(2))
                    r["declined"] += int(m.group(3))
                    mx("res_peak", int(m.group(4)))
                    mx("res_used", int(m.group(6)))
                elif m := ALLOC.search(line):
                    mx("alloc_peak", int(m.group(1)) << 20)
                    r["alloc_fail"] += int(m.group(3))
                elif m := GPU.search(line):
                    mx("gpu_peak", int(m.group(1)))
    return r


def gib(x):
    return f"{x / 2**30:7.2f}" if x is not None else f"{'-':>7}"


def main(tsvs):
    for tsv in tsvs:
        print(f"== {os.path.basename(tsv)}")
        print(
            f"{'q':>4} {'status':>9}  {'arena':>7} {'peak':>7}  {'resv pk':>7} {'enc max':>7} "
            f"{'granted':>7} {'declin':>6}  {'alloc pk':>8} {'fails':>5}  {'gpu pk':>7}   (GiB)"
        )
        for q, row in sorted(ab_report.load(tsv).items()):
            r = scan(row["dir"])
            print(
                f"q{q:<3} {row['status'][:9]:>9}  {gib(r['cap'])} {gib(r['arena_peak'])}  "
                f"{gib(r['res_peak'])} {gib(r['res_used'])} {r['granted']:7d} {r['declined']:6d}  "
                f"{gib(r['alloc_peak']):>8} {r['alloc_fail']:5d}  {gib(r['gpu_peak'])}"
            )
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
