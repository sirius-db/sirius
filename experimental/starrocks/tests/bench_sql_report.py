#!/usr/bin/env python3
"""Summarize a 2cn_bench.sh run: FE wall and CN round time per SQL case, cold and warm.

walls.tsv rows: case, run index, FE start_us, FE end_us. The CN round of one run is
min(start_us) .. max(end_us) over both CNs' `fragment timing` lines inside that FE window
(SIRIUS_CN_TIMING=1). `packed hop timing` lines in the window give bytes and wire busy.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

FIELD = re.compile(r"(\w+)=([^\s]+)")


def events(log: Path, marker: str):
    for line in log.read_text(errors="replace").splitlines():
        if marker in line:
            yield {k: v for k, v in FIELD.findall(line)}


def main() -> None:
    e2e = Path(sys.argv[1])
    frags, hops = [], []
    for name in ("cn0", "cn1"):
        frags += list(events(e2e / f"{name}.log", "fragment timing"))
        hops += list(events(e2e / f"{name}.log", "packed hop timing"))
    runs = defaultdict(list)
    for line in (e2e / "walls.tsv").read_text().splitlines():
        case, index, start, end = line.split("\t")
        runs[case].append((int(index), int(start), int(end)))

    print("case\trun\tfe_ms\tcn_round_ms\thop_bytes\thop_write_ms\thop_span_ms\twire_busy_pct\tfragments")
    summary = {}
    for case, items in runs.items():
        rows = []
        for index, start, end in sorted(items):
            inside = [f for f in frags if start <= int(f["start_us"]) and int(f["end_us"]) <= end]
            hop_in = [h for h in hops if start <= int(h["start_us"]) <= end]
            round_ms = (
                (max(int(f["end_us"]) for f in inside) - min(int(f["start_us"]) for f in inside))
                / 1e3
                if inside
                else float("nan")
            )
            hop_bytes = sum(int(h["bytes"]) for h in hop_in)
            write_ms = sum(int(h["write_us"]) for h in hop_in) / 1e3
            span_ms = sum(int(h["span_us"]) for h in hop_in) / 1e3
            busy = sum(int(h["wire_busy_us"]) for h in hop_in) / 1e3
            busy_pct = 100 * busy / span_ms if span_ms else 0.0
            fe_ms = (end - start) / 1e3
            rows.append((index, fe_ms, round_ms))
            print(
                f"{case}\t{index}\t{fe_ms:.1f}\t{round_ms:.1f}\t{hop_bytes}\t{write_ms:.2f}"
                f"\t{span_ms:.2f}\t{busy_pct:.0f}\t{len(inside)}"
            )
        cold = rows[0]
        warm = rows[1:] or rows
        summary[case] = (
            cold[1],
            cold[2],
            sum(r[1] for r in warm) / len(warm),
            sum(r[2] for r in warm) / len(warm),
        )
    print()
    print("case\tfe_cold_ms\tcn_cold_ms\tfe_warm_ms\tcn_warm_ms")
    for case, (fe_cold, cn_cold, fe_warm, cn_warm) in summary.items():
        print(f"{case}\t{fe_cold:.1f}\t{cn_cold:.1f}\t{fe_warm:.1f}\t{cn_warm:.1f}")


if __name__ == "__main__":
    main()
