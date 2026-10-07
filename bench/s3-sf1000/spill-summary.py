#!/usr/bin/env python3
"""Per-query spill and OOM-retry summary from a Sirius debug log.

    python3 bench/s3-sf1000/spill-summary.py <sirius_*.log> <first-query> [...]

The harness runs each query as `SET gpu_execution = true;` followed by the query
statement, in the order given, so every QueryBegin whose SQL starts with SELECT
or WITH opens the next TPC-H query. Pass the query numbers in run order (e.g.
`1-17` or `1 2 3`). Sums the downgrade executor's `request done` lines (bytes
moved to the host and disk tiers) and counts OOM reschedules per query.
"""

import re
import sys

DONE = re.compile(
    r"request done: .*?to_host: (\d+)/(\d+) batches/bytes, to_disk: (\d+)/(\d+) batches/bytes"
)
BEGIN = re.compile(r"QueryBegin: .*? SQL: (.*)")


def parse_queries(args):
    out = []
    for a in args:
        if "-" in a:
            lo, hi = a.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(a))
    return out


def main():
    log, queries = sys.argv[1], parse_queries(sys.argv[2:])
    stats = {}
    current = None
    idx = 0
    for line in open(log, errors="replace"):
        m = BEGIN.search(line)
        if m:
            sql = m.group(1).lstrip().lower()
            if sql.startswith(("select", "with")):
                current = queries[idx] if idx < len(queries) else None
                idx += 1
                stats.setdefault(current, [0, 0, 0, 0, 0])
            continue
        if current is None:
            continue
        m = DONE.search(line)
        if m:
            s = stats[current]
            s[0] += int(m.group(1))
            s[1] += int(m.group(2))
            s[2] += int(m.group(3))
            s[3] += int(m.group(4))
        elif "reschedule (retry" in line:
            stats[current][4] += 1
    print(f"{'query':>5} {'to_host GB':>11} {'(batches)':>10} {'to_disk GB':>11} {'(batches)':>10} {'OOM retries':>12}")
    for q in queries:
        if q not in stats:
            continue
        hb, hB, db, dB, r = stats[q]
        print(f"q{q:<4} {hB / 1e9:11.2f} {hb:10d} {dB / 1e9:11.2f} {db:10d} {r:12d}")


if __name__ == "__main__":
    main()
