#!/usr/bin/env python3
"""Generate a DuckDB CLI script for ONE TPC-H query in the multi-GPU 1k benchmark.

The emitted script:
  1. registers a view per TPC-H table over the parquet files (explicit file
     list, exactly like performance_test.py so pin path-matching works),
  2. enables transparent GPU execution (`SET gpu_execution = true`),
  3. for pinned scenarios, pins the query's referenced columns into the chosen
     cache tier (host/gpu) using the per-query column map in
     ``tpch_pin_columns.py``,
  4. runs the query `--iterations` times, each wrapped in `.timer on/off` with a
     unique marker line so the driver can attribute each "Run Time (s): real X"
     line to a specific (query, iteration),
  5. unpins.

Timing contract for the driver: `.timer` is ON *only* around each query
statement, so every "Run Time (s): real" line in the CLI output belongs to a
query iteration, immediately preceded by its `MARKER_PREFIX q<N> iter<k>` line.

This reuses the project's tested query text and per-query pin column lists; it
does not redefine them. Plain Python — no duckdb import — so it runs under any
interpreter.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

# Reuse the repo's query text and pin-column map. This script lives in
# scripts/nvl72/; queries.py and tpch_pin_columns.py live in test/tpch_performance/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_TPCH_PERF = os.path.join(_REPO, "test", "tpch_performance")
sys.path.insert(0, _TPCH_PERF)

from queries import QUERIES  # noqa: E402
import tpch_pin_columns as pin  # noqa: E402

TPCH_TABLES = [
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
]

MARKER_PREFIX = "###SIRIUS_BENCH"


def resolve_parquet_files(parquet_dir: str, table: str) -> list[str]:
    """Same resolution order as performance_test.py: <t>.parquet, <t>_*.parquet, <t>/*.parquet."""
    candidates: list[str] = []
    for pattern in (
        os.path.join(parquet_dir, f"{table}.parquet"),
        os.path.join(parquet_dir, f"{table}_*.parquet"),
        os.path.join(parquet_dir, table, "*.parquet"),
    ):
        candidates.extend(sorted(glob.glob(pattern)))
    return candidates


def emit(
    parquet_dir: str, qnum: int, tier: str, iterations: int, profile: bool = False
) -> str:
    lines: list[str] = []
    # 1. views over the parquet tables
    for table in TPCH_TABLES:
        files = resolve_parquet_files(parquet_dir, table)
        if not files:
            raise FileNotFoundError(
                f"No parquet files for table '{table}' in {parquet_dir}"
            )
        file_list = ",".join(f"'{f}'" for f in files)
        lines.append(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet([{file_list}]);"
        )

    # 2. transparent GPU execution
    lines.append("SET gpu_execution = true;")
    lines.append(".timer off")

    # 3. pin (host/gpu tiers only)
    if tier in ("host", "gpu"):
        os.environ["SIRIUS_PIN_TIER"] = tier
        lines.append(pin.emit_pin(qnum, parquet_dir).strip())

    # 4. timed iterations
    query = QUERIES[f"q{qnum}"].strip().rstrip(";")
    if profile:
        # Profiling mode: pin + views are already done (outside the capture
        # window). Bracket ONLY the query execution with cudaProfilerStart/Stop
        # so `nsys --capture-range=cudaProfilerApi` records query time only —
        # no pin population, no CUDA-context init. Single iteration.
        lines.append(f"SELECT '{MARKER_PREFIX} q{qnum} iter0' AS marker;")
        lines.append("CALL profiler_start();")
        lines.append(".timer on")
        lines.append(query + ";")
        lines.append(".timer off")
        lines.append("CALL profiler_stop();")
    else:
        for it in range(iterations):
            lines.append(f"SELECT '{MARKER_PREFIX} q{qnum} iter{it}' AS marker;")
            lines.append(".timer on")
            lines.append(query + ";")
            lines.append(".timer off")

    # 5. unpin
    if tier in ("host", "gpu"):
        lines.append(pin.emit_unpin(qnum).strip())

    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="TPC-H parquet directory")
    ap.add_argument("--query", required=True, type=int, help="query number 1..22")
    ap.add_argument(
        "--tier",
        default="none",
        choices=["none", "host", "gpu"],
        help="pin tier; 'none' reads from disk (no pinning)",
    )
    ap.add_argument("--iterations", default=2, type=int)
    ap.add_argument(
        "--profile",
        action="store_true",
        help="wrap the (single) query in CALL profiler_start()/profiler_stop() for nsys "
        "--capture-range=cudaProfilerApi, so only query execution is captured (pin excluded)",
    )
    args = ap.parse_args(argv)
    sys.stdout.write(
        emit(args.data, args.query, args.tier, args.iterations, args.profile)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
