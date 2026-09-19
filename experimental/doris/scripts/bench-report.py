#!/usr/bin/env python3
"""Flatten and report the benchmark runs of scripts/bench.sh (plan-doc experiments/sf10-bench).

    bench-report.py rounds --run DIR
        one bench.sh output directory -> DIR/rounds.csv: every (round, query) with the client
        wall time, the engine time (Sirius backend), the FE audit numbers (fe-audit.py), the
        validation verdict (summary.csv) and the process/GPU samples of the query's window
        (samples.csv: peak RSS, bytes read from disk, CPU cores busy, peak GPU memory).

    bench-report.py report --runs DIR [DIR...] [--baseline native] [--primary sirius-buffered]
                           [--price SYSTEM=DOLLARS_PER_HOUR ...] [--title TEXT] --out results.md
        the Markdown tables of plan §11 from several runs (one per system, same dataset):
        per query the cold (round 1) and hot (median of rounds 2..N, min in parentheses) wall
        times of every system, the speedup baseline/primary on hot medians with its geometric
        mean, the power totals, a resource table, validation notes and the environment
        snapshots. A query that did not validate in every round of a system is shown but left
        out of that system's speedups and totals.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROUND_COLUMNS = [
    "system", "sf", "round", "cold", "query", "status", "rows", "wall_ms", "engine_ms", "fe_ms", "plan_ms",
    "schedule_ms", "rpc1_ms", "cpu_ms", "peak_mem_bytes", "scan_bytes", "scan_rows", "query_id", "start_ms",
    "end_ms", "peak_rss_kb", "read_bytes", "cpu_cores", "gpu_peak_mib", "gpu_util_pct", "gpu_mem_util_pct",
]


def num(value: str | None) -> float | None:
    if value in (None, "", "-"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- rounds ---------------------------------------------------------------------------------


def window_samples(samples: list[dict[str, str]], start: float, end: float) -> dict[str, str]:
    """Process/GPU numbers of one query from the sampler's 500 ms ticks: the peaks inside
    [start, end] (widened by one tick so sub-tick queries still see a sample), the read_bytes
    and CPU deltas between the last tick before the query and the first tick after it."""
    out = {"peak_rss_kb": "-", "read_bytes": "-", "cpu_cores": "-", "gpu_peak_mib": "-", "gpu_util_pct": "-", "gpu_mem_util_pct": "-"}
    if not samples:
        return out
    inside = [s for s in samples if start - 600 <= num(s["ts_ms"]) <= end + 600]
    rss = [num(s["rss_kb"]) for s in inside if num(s["rss_kb"]) is not None]
    gpu = [num(s["gpu_used_mib"]) for s in inside if num(s["gpu_used_mib"]) is not None]
    if rss:
        out["peak_rss_kb"] = str(int(max(rss)))
    if gpu:
        out["gpu_peak_mib"] = str(int(max(gpu)))
    # GPU utilization: the mean over the samples strictly inside the window (nvidia-smi's
    # percentages already cover ~1 s each), so a query shorter than a sample shows "-".
    strictly = [s for s in samples if start <= num(s["ts_ms"]) <= end]
    for column in ("gpu_util_pct", "gpu_mem_util_pct"):
        values = [num(s.get(column)) for s in strictly if num(s.get(column)) is not None]
        if values:
            out[column] = f"{sum(values) / len(values):.0f}"
    before = [s for s in samples if num(s["ts_ms"]) <= start and num(s["read_bytes"]) is not None]
    after = [s for s in samples if num(s["ts_ms"]) >= end and num(s["read_bytes"]) is not None]
    if before and after:
        b, a = before[-1], after[0]
        if b.get("pid") == a.get("pid"):
            out["read_bytes"] = str(int(num(a["read_bytes"]) - num(b["read_bytes"])))
            dt = (num(a["ts_ms"]) - num(b["ts_ms"])) / 1000
            if dt > 0 and num(a["cpu_ticks"]) is not None and num(b["cpu_ticks"]) is not None:
                # clock ticks (100 Hz) over wall seconds = cores busy
                out["cpu_cores"] = f"{(num(a['cpu_ticks']) - num(b['cpu_ticks'])) / 100 / dt:.2f}"
    return out


def cmd_rounds(args) -> int:
    run = args.run
    system = (run / "system.txt").read_text().strip() if (run / "system.txt").exists() else run.name
    sf = "-"
    for line in (run / "env.txt").read_text().splitlines() if (run / "env.txt").exists() else []:
        if line.startswith("data:"):
            for token in line.split():
                if "sf" in token:
                    import re
                    m = re.search(r"sf\d+", token)
                    if m:
                        sf = m.group(0)
                        break
    rows = []
    round_dirs = [d for d in run.glob("round[0-9]*") if d.is_dir() and d.name[5:].isdigit()]
    for rdir in sorted(round_dirs, key=lambda p: int(p.name[5:])):
        k = int(rdir.name[5:])
        timings = read_csv(rdir / "timings.csv")
        verdicts = {r["query"]: r["status"] for r in read_csv(rdir / "summary.csv")}
        samples = read_csv(rdir / "samples.csv")
        for t in timings:
            status = verdicts.get(t["query"], "FAILED" if num(t.get("wall_ms")) is None else "UNVALIDATED")
            row = {c: "-" for c in ROUND_COLUMNS}
            row.update({"system": system, "sf": sf, "round": str(k), "cold": "1" if k == 1 else "0", "status": status})
            for c in ROUND_COLUMNS:
                if c in t and t[c] not in ("", None):
                    row[c] = t[c]
            start, end = num(t.get("start_ms")), num(t.get("end_ms"))
            if start is not None and end is not None:
                row.update(window_samples(samples, start, end))
            rows.append(row)
    with (run / "rounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROUND_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"bench-report: {len(rows)} row(s) -> {run / 'rounds.csv'}")
    return 0


# --- report ---------------------------------------------------------------------------------


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


class Run:
    def __init__(self, path: Path):
        self.path = path
        self.rows = read_csv(path / "rounds.csv")
        if not self.rows:
            raise SystemExit(f"{path}: no rounds.csv (run `bench-report.py rounds --run {path}` first)")
        self.system = self.rows[0]["system"]
        self.sf = self.rows[0]["sf"]
        self.env = (path / "env.txt").read_text() if (path / "env.txt").exists() else ""
        self.variables = (path / "variables.txt").read_text() if (path / "variables.txt").exists() else ""
        self.queries = []
        for r in self.rows:
            if r["query"] not in self.queries:
                self.queries.append(r["query"])
        self.rounds = sorted({int(r["round"]) for r in self.rows})

    def of(self, query: str, column: str, cold: bool | None = None) -> list[float]:
        return [
            v for r in self.rows
            if r["query"] == query and (cold is None or (r["cold"] == "1") == cold)
            for v in [num(r.get(column))] if v is not None
        ]

    def valid(self, query: str) -> bool:
        statuses = [r["status"] for r in self.rows if r["query"] == query]
        return bool(statuses) and all(s == "OK" for s in statuses)

    def statuses(self, query: str) -> str:
        bad = sorted({r["status"] for r in self.rows if r["query"] == query and r["status"] != "OK"})
        return ", ".join(bad)

    def hot(self, query: str, column: str = "wall_ms") -> float | None:
        values = self.of(query, column, cold=False)
        return median(values) if values else median(self.of(query, column))

    def hot_min(self, query: str, column: str = "wall_ms") -> float | None:
        values = self.of(query, column, cold=False) or self.of(query, column)
        return min(values) if values else None

    def cold(self, query: str, column: str = "wall_ms") -> float | None:
        values = self.of(query, column, cold=True)
        return values[0] if values else None

    def any_column(self, column: str) -> bool:
        return any(num(r.get(column)) is not None for r in self.rows)


def cmd_report(args) -> int:
    runs = [Run(p) for p in args.runs]
    by_system = {r.system: r for r in runs}
    baseline = by_system.get(args.baseline)
    primary = by_system.get(args.primary)
    queries = []
    for r in runs:
        for q in r.queries:
            if q not in queries:
                queries.append(q)
    prices = {}
    for spec in args.price or []:
        name, _, dollars = spec.partition("=")
        prices[name] = float(dollars)
    sf = runs[0].sf
    lines = []
    lines.append(f"# {args.title or 'Doris vs Doris + Sirius · TPC-H ' + sf}")
    lines.append("")
    lines.append(
        f"Systems: {', '.join(f'`{r.system}` ({len(r.rounds)} round(s))' for r in runs)}. "
        "Round 1 is cold (freshly started process, page cache evicted), the others hot; **hot** = median of "
        "the hot rounds, (min) in parentheses; ms of client round trip (`wall_ms`). Queries that did not "
        "validate against the DuckDB baseline in every round are marked and excluded from speedups and totals."
    )
    lines.append("")

    # --- main table: hot wall per system, speedup baseline/primary
    lines.append("## Hot runs (median wall ms, min in parentheses)")
    lines.append("")
    header = ["q"] + [r.system for r in runs]
    if baseline and primary and baseline is not primary:
        header.append(f"speedup {baseline.system}/{primary.system}")
    if primary and primary.any_column("engine_ms"):
        header.append(f"{primary.system} engine ms")
    header.append("notes")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    speedups = []
    totals = {r.system: 0.0 for r in runs}
    counted = {r.system: 0 for r in runs}
    for q in queries:
        cells = [q]
        notes = []
        for r in runs:
            hot, low = r.hot(q), r.hot_min(q)
            mark = "" if r.valid(q) else " ✗"
            cells.append(f"{fmt_ms(hot)} ({fmt_ms(low)}){mark}" if hot is not None else "-")
            if hot is not None and r.valid(q):
                totals[r.system] += hot
                counted[r.system] += 1
            if not r.valid(q):
                notes.append(f"{r.system}: {r.statuses(q) or 'missing'}")
        if baseline and primary and baseline is not primary:
            b, p = baseline.hot(q), primary.hot(q)
            if b and p and baseline.valid(q) and primary.valid(q):
                s = b / p
                speedups.append(s)
                cells.append(f"**{s:.2f}×**")
            else:
                cells.append("-")
        if primary and primary.any_column("engine_ms"):
            cells.append(fmt_ms(primary.hot(q, "engine_ms")))
        cells.append("; ".join(notes))
        lines.append("| " + " | ".join(cells) + " |")
    cells = ["**power total**"] + [f"**{fmt_ms(totals[r.system])}** ({counted[r.system]} q)" for r in runs]
    if baseline and primary and baseline is not primary:
        gm = geomean(speedups)
        tb, tp = totals[baseline.system], totals[primary.system]
        cells.append(f"**geomean {gm:.2f}×**, total {tb / tp:.2f}×" if gm and tp else "-")
    if primary and primary.any_column("engine_ms"):
        cells.append(fmt_ms(sum(primary.hot(q, 'engine_ms') or 0 for q in queries if primary.valid(q))))
    cells.append("")
    lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if prices and baseline and primary and baseline is not primary and baseline.system in prices and primary.system in prices:
        pb, pp = prices[baseline.system], prices[primary.system]
        per_dollar = [s * pb / pp for s in speedups]
        gm = geomean(per_dollar)
        lines.append(
            f"Cost-normalized (plan §7): {baseline.system} at ${pb}/h vs {primary.system} at ${pp}/h → "
            f"geomean speedup per dollar **{gm:.2f}×**." if gm else ""
        )
        lines.append("")

    # --- cold table
    lines.append("## Cold run (round 1, wall ms)")
    lines.append("")
    header = ["q"] + [r.system for r in runs]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for q in queries:
        lines.append("| " + " | ".join([q] + [fmt_ms(r.cold(q)) for r in runs]) + " |")
    lines.append("| **total** | " + " | ".join(fmt_ms(sum(r.cold(q) or 0 for q in queries)) for r in runs) + " |")
    lines.append("")

    # --- FE-side breakdown (Doris systems)
    fe_runs = [r for r in runs if r.any_column("fe_ms")]
    if fe_runs:
        lines.append("## FE audit (hot medians, ms): fe = FE end-to-end, plan = Nereids planning, rpc1 = first exec RPC")
        lines.append("")
        header = ["q"] + [f"{r.system} fe / plan / rpc1" for r in fe_runs]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for q in queries:
            cells = [q]
            for r in fe_runs:
                cells.append(f"{fmt_ms(r.hot(q, 'fe_ms'))} / {fmt_ms(r.hot(q, 'plan_ms'))} / {fmt_ms(r.hot(q, 'rpc1_ms'))}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # --- resources
    lines.append("## Resources (hot medians): peak RSS MiB / bytes read from disk MiB / CPU cores busy / GPU MiB in use / GPU busy % / GPU memory-controller busy %")
    lines.append("")
    header = ["q"] + [r.system for r in runs]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for q in queries:
        cells = [q]
        for r in runs:
            rss = r.hot(q, "peak_rss_kb")
            rb = r.hot(q, "read_bytes")
            cpu = r.hot(q, "cpu_cores")
            gpu = r.hot(q, "gpu_peak_mib")
            util = r.hot(q, "gpu_util_pct")
            mem_util = r.hot(q, "gpu_mem_util_pct")
            cells.append(
                f"{fmt_ms(rss / 1024 if rss is not None else None)} / {fmt_ms(rb / 2**20 if rb is not None else None)} / "
                f"{'-' if cpu is None else f'{cpu:.1f}'} / {fmt_ms(gpu)} / {fmt_ms(util)} / {fmt_ms(mem_util)}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Bytes read from disk = `/proc/<pid>/io read_bytes` delta over the query (0 on a page-cache hit; "
                 "O_DIRECT reads always count); sampled every 0.5 s, so sub-second queries are attributed approximately. "
                 "GPU MiB in use is the RMM pool (reserved up front, not a peak); GPU busy % = nvidia-smi `utilization.gpu` "
                 "(share of time a kernel was running) and memory-controller busy % = `utilization.memory`, both averaged "
                 "over the query's samples; `-` when the query was shorter than one sample.")
    lines.append("")

    # --- environment
    lines.append("## Environment")
    lines.append("")
    for r in runs:
        lines.append(f"### {r.system} (`{r.path}`)")
        lines.append("")
        lines.append("```")
        lines.append(r.env.rstrip())
        if r.variables.strip():
            lines.append("-- session variables (GLOBAL) --")
            lines.append(r.variables.rstrip())
        lines.append("```")
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"bench-report: {len(runs)} run(s), {len(queries)} querie(s) -> {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("rounds", help="flatten one bench.sh run into rounds.csv")
    p.add_argument("--run", type=Path, required=True)
    p.set_defaults(func=cmd_rounds)
    p = sub.add_parser("report", help="Markdown report over several runs")
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--baseline", default="native", help="system in the numerator of the speedup (default native)")
    p.add_argument("--primary", default="sirius-buffered", help="system in the denominator (default sirius-buffered)")
    p.add_argument("--price", action="append", help="SYSTEM=DOLLARS_PER_HOUR, for the per-dollar speedup")
    p.add_argument("--title")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_report)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
