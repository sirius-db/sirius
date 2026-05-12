# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

import argparse
import csv
import glob
import os
import subprocess
import threading
import time
from datetime import datetime

import duckdb
from queries import QUERIES


def log(msg):
    """Timestamped, flushed progress log so hangs are visible in real time."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


class QueryTimeout(Exception):
    pass


def execute_with_timeout(con, sql, timeout_s):
    """Run `sql` on `con`; if `timeout_s > 0` and exceeded, interrupt and raise.

    Returns (rows, elapsed_seconds). On timeout the connection is interrupted
    so the caller can continue using it for subsequent queries.
    """
    box = {}

    def target():
        try:
            box["rows"] = con.execute(sql).fetchall()
        except BaseException as e:  # noqa: BLE001 — capture interrupt too
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    start = time.perf_counter()
    t.start()
    t.join(timeout_s if timeout_s > 0 else None)
    if t.is_alive():
        log(f"  ⚠ TIMEOUT after {timeout_s}s — calling con.interrupt()")
        try:
            con.interrupt()
        except Exception as e:
            log(f"  interrupt() raised: {e}")
        t.join(30)
        if t.is_alive():
            log("  ⚠ thread still alive after interrupt — connection may be unusable")
        raise QueryTimeout(f"Query exceeded {timeout_s}s")
    elapsed = time.perf_counter() - start
    if "error" in box:
        raise box["error"]
    return box["rows"], elapsed


CACHE_MODES = ("grouped", "sequential", "isolated")
ENGINE_CHOICES = ("gpu", "cpu", "both")
TPCH_TABLES = (
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)

EXTENSION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "build/release/extension/sirius/sirius.duckdb_extension",
)


def _verify_results(duckdb_rows, sirius_rows, query_name):
    """Verify that DuckDB and Sirius results match."""
    try:
        if len(duckdb_rows) != len(sirius_rows):
            print(
                f"❌ {query_name}: Row count mismatch - DuckDB: {len(duckdb_rows)}, Sirius: {len(sirius_rows)}"
            )
            return False

        duckdb_rows_sorted = sorted(duckdb_rows, key=lambda x: str(x))
        sirius_rows_sorted = sorted(sirius_rows, key=lambda x: str(x))

        for i, (duck_row, sirius_row) in enumerate(
            zip(duckdb_rows_sorted, sirius_rows_sorted)
        ):
            if duck_row != sirius_row:
                print(f"❌ {query_name}: Row {i} mismatch")
                print(f"   DuckDB:  {duck_row}")
                print(f"   Sirius:  {sirius_row}")
                return False

        print(f"✓ {query_name}: Results match ({len(duckdb_rows)} rows)")
        return True
    except Exception as e:
        print(f"❌ {query_name}: Error during verification - {e}")
        return False


def parse_query_spec(spec):
    """Parse a query list spec like '1,3,6-10' into a list of ints. None → all."""
    if spec is None:
        return list(range(1, 23))
    nums = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            nums.extend(range(int(lo), int(hi) + 1))
        else:
            nums.append(int(token))
    return nums


def resolve_engine_modes(engine):
    if engine == "gpu":
        return [("sirius", True)]
    if engine == "cpu":
        return [("duckdb", False)]
    return [("duckdb", False), ("sirius", True)]


def default_output(cache, iterations):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"tpch_{ts}_{cache}_iter{iterations}.csv"


def drop_os_cache():
    """Drop OS filesystem cache. Requires passwordless sudo per CLAUDE.md."""
    proc = subprocess.run(
        ["sudo", "-n", "/usr/bin/tee", "/proc/sys/vm/drop_caches"],
        input="3\n",
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to drop OS cache. Set up passwordless sudo as described "
            f"in test/tpch_performance/CLAUDE.md (stderr: {proc.stderr.strip()})"
        )


def _resolve_parquet_files(parquet_dir, table):
    candidates = []
    for pattern in (
        os.path.join(parquet_dir, f"{table}.parquet"),
        os.path.join(parquet_dir, f"{table}_*.parquet"),
        os.path.join(parquet_dir, table, "*.parquet"),
    ):
        candidates.extend(sorted(glob.glob(pattern)))
    return candidates


def open_connection(source):
    """Open a DuckDB connection, load the Sirius extension, register tables.

    `source` may be a .duckdb file path or a parquet directory. When a parquet
    directory, each TPC-H table is registered as a view over its parquet files.
    """
    is_dir = os.path.isdir(source)
    db_path = ":memory:" if is_dir else source
    log(f"Opening DuckDB connection (db_path={db_path}, is_dir={is_dir})")
    con = duckdb.connect(db_path, config={"allow_unsigned_extensions": "true"})
    if is_dir:
        for table in TPCH_TABLES:
            files = _resolve_parquet_files(source, table)
            if not files:
                raise FileNotFoundError(
                    f"No parquet files found for table '{table}' in {source}"
                )
            log(f"  Registering view '{table}' over {len(files)} file(s)")
            file_list = ",".join(f"'{f}'" for f in files)
            con.execute(
                f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet([{file_list}])"
            )
        log("All TPC-H views registered")
    log(f"Loading Sirius extension from {EXTENSION_PATH}")
    con.execute(f"LOAD '{EXTENSION_PATH}'")
    log("Sirius extension loaded")
    return con


def time_query(con, qnum, use_gpu, timeout_s):
    engine_label = "GPU/sirius" if use_gpu else "CPU/duckdb"
    log(f"  SET gpu_execution = {str(use_gpu).lower()} ({engine_label})")
    con.execute(f"SET gpu_execution = {'true' if use_gpu else 'false'};")
    log(f"  Executing q{qnum} on {engine_label} (timeout={timeout_s}s)…")
    rows, elapsed = execute_with_timeout(con, QUERIES[f"q{qnum}"], timeout_s)
    log(f"  q{qnum} fetched {len(rows)} rows in {elapsed:.4f}s")
    return elapsed


def _record(writer, name, qnum, it, runtime):
    writer.writerow([name, f"q{qnum}", it, f"{runtime:.6f}"])
    log(f"[{name}] q{qnum} iter{it}: {runtime:.4f}s")


def _run_one(writer, con, name, qnum, it, use_gpu, timeout_s):
    try:
        _record(writer, name, qnum, it, time_query(con, qnum, use_gpu, timeout_s))
    except QueryTimeout:
        log(f"[{name}] q{qnum} iter{it}: TIMEOUT (>{timeout_s}s)")
        writer.writerow([name, f"q{qnum}", it, "TIMEOUT"])


def run_grouped(source, queries, engine_modes, iterations, writer, timeout_s):
    """For each query, run all iterations back-to-back (hot cache)."""
    log("Cache mode 'grouped': opening single connection")
    con = open_connection(source)
    try:
        for qnum in queries:
            for it in range(iterations):
                for name, use_gpu in engine_modes:
                    log(f"--- q{qnum} iter{it} engine={name} ---")
                    _run_one(writer, con, name, qnum, it, use_gpu, timeout_s)
    finally:
        log("Closing connection")
        con.close()


def run_sequential(source, queries, engine_modes, iterations, writer, timeout_s):
    """Run all queries once, then again, …"""
    log("Cache mode 'sequential': opening single connection")
    con = open_connection(source)
    try:
        for it in range(iterations):
            for qnum in queries:
                for name, use_gpu in engine_modes:
                    log(f"--- q{qnum} iter{it} engine={name} ---")
                    _run_one(writer, con, name, qnum, it, use_gpu, timeout_s)
    finally:
        log("Closing connection")
        con.close()


def run_isolated(source, queries, engine_modes, iterations, writer, timeout_s):
    """Drop OS cache and reopen the connection before every single run."""
    log("Cache mode 'isolated': reopening connection between every run")
    for qnum in queries:
        for it in range(iterations):
            for name, use_gpu in engine_modes:
                log(f"--- q{qnum} iter{it} engine={name} (dropping OS cache) ---")
                drop_os_cache()
                con = open_connection(source)
                try:
                    _run_one(writer, con, name, qnum, it, use_gpu, timeout_s)
                finally:
                    log("Closing connection")
                    con.close()


RUNNERS = {
    "grouped": run_grouped,
    "sequential": run_sequential,
    "isolated": run_isolated,
}


def validate(source, queries):
    """Compare CPU vs GPU result sets query-by-query."""
    con = open_connection(source)
    try:
        results = {}
        for qnum in queries:
            query_name = f"q{qnum}"
            print(f"Verifying {query_name}...")
            try:
                con.execute("SET gpu_execution = false;")
                cpu_rows = con.execute(QUERIES[query_name]).fetchall()
                con.execute("SET gpu_execution = true;")
                gpu_rows = con.execute(QUERIES[query_name]).fetchall()
                results[qnum] = _verify_results(cpu_rows, gpu_rows, query_name)
            except Exception as e:
                print(f"  {query_name}: Exception - {e}")
                results[qnum] = False

        passed = sum(1 for v in results.values() if v)
        print(f"\n{'=' * 60}")
        print(f"Validation Summary: {passed}/{len(results)} queries passed")
        print(f"{'=' * 60}")
        return results
    finally:
        con.close()


def parse_args():
    p = argparse.ArgumentParser(description="Run TPC-H performance tests")
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--sf",
        type=int,
        default=None,
        help="Scale factor — uses the default performance_test.duckdb file",
    )
    src.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a .duckdb database file or a parquet directory",
    )
    p.add_argument(
        "--cache",
        choices=CACHE_MODES,
        default="grouped",
        help="Cache mode: grouped (hot cache, per-query iterations), "
        "sequential (round-robin), isolated (drop OS cache between runs)",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations per query (default: 1)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: tpch_<timestamp>_<cache>_iter<N>.csv)",
    )
    p.add_argument(
        "--engine",
        choices=ENGINE_CHOICES,
        default="both",
        help="Which engine to benchmark (default: both)",
    )
    p.add_argument(
        "--validation",
        action="store_true",
        help="After timing, validate that GPU and CPU produce matching results",
    )
    p.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Comma-separated query list, e.g. '1,3,6-10' (default: all 22)",
    )
    p.add_argument(
        "--config",
        type=str,
        default=os.environ.get("SIRIUS_CONFIG_FILE", ""),
        help="Path to Sirius config YAML (default: $SIRIUS_CONFIG_FILE)",
    )
    p.add_argument(
        "--query-timeout",
        type=float,
        default=300.0,
        help="Per-query timeout in seconds; 0 disables (default: 300)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    source = args.input or "performance_test.duckdb"
    queries = parse_query_spec(args.queries)
    engine_modes = resolve_engine_modes(args.engine)
    output_path = args.output or default_output(args.cache, args.iterations)

    config_path = (args.config or "").strip()
    if config_path:
        os.environ["SIRIUS_CONFIG_FILE"] = config_path
    else:
        log(
            "SIRIUS_CONFIG_FILE not set and --config not provided — "
            "running with Sirius default configuration."
        )

    log(f"Source:      {source}")
    if args.sf is not None:
        log(f"Scale factor: {args.sf}")
    log(f"Cache mode:  {args.cache}")
    log(f"Iterations:  {args.iterations}")
    log(f"Engine:      {args.engine}")
    log(f"Queries:     {queries}")
    log(f"Config:      {config_path or '(default)'}")
    log(f"Output:      {output_path}")
    log(
        f"Query timeout: {args.query_timeout}s"
        if args.query_timeout > 0
        else "Query timeout: disabled"
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["engine", "query", "iteration", "runtime_s"])
        f.flush()
        RUNNERS[args.cache](
            source, queries, engine_modes, args.iterations, writer, args.query_timeout
        )

    log("Benchmark run complete")

    if args.validation:
        log("Starting validation")
        validate(source, queries)
        log("Validation complete")


if __name__ == "__main__":
    main()
