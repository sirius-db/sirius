#!/usr/bin/env python3
"""Summarize a run.sh NIC/CPU sample file: receive rate and CPU busy while active.

    python3 bench/s3-sf1000/nic-summary.py <nic_*.csv> [--series]
"""

import csv
import sys


def main():
    path = sys.argv[1]
    series = "--series" in sys.argv
    rows = list(csv.DictReader(open(path)))
    out = []
    for a, b in zip(rows, rows[1:]):
        dt = (int(b["epoch_s"]) - int(a["epoch_s"])) or 1
        gbps = (int(b["rx_bytes"]) - int(a["rx_bytes"])) * 8 / 1e9 / dt
        cpu = None
        if "cpu_busy" in a and a["cpu_busy"]:
            dtot = int(b["cpu_total"]) - int(a["cpu_total"])
            cpu = 100.0 * (int(b["cpu_busy"]) - int(a["cpu_busy"])) / dtot if dtot else 0.0
        mem = int(b["mem_avail_kb"]) / 1e6 if b.get("mem_avail_kb") else None
        out.append((int(b["epoch_s"]), gbps, cpu, mem))
    if series:
        for t, g, c, _ in out:
            print(f"{t}  {g:6.1f} Gb/s" + (f"  cpu {c:5.1f}%" if c is not None else ""))
    active = [(g, c) for _, g, c, _ in out if g > 1.0]
    mems = [m for *_, m in out if m is not None]
    if not active:
        print("no receive activity")
        return
    gs = [g for g, _ in active]
    line = (
        f"active {len(active)} s, NIC mean {sum(gs) / len(gs):.1f} Gb/s, "
        f"peak {max(gs):.1f} Gb/s, received {sum(gs) / 8:.1f} GB"
    )
    cs = [c for _, c in active if c is not None]
    if cs:
        line += f", CPU busy mean {sum(cs) / len(cs):.0f}% peak {max(cs):.0f}%"
    if mems:
        line += f", min host MemAvailable {min(mems):.1f} GB"
    print(line)


if __name__ == "__main__":
    main()
