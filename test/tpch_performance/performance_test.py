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
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

import duckdb
from queries import QUERIES
from tpch_pin_columns import (
    emit_pin,
    emit_pin_all,
    emit_unpin,
    emit_unpin_all,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log(msg):
    """Timestamped, flushed progress log so hangs are visible in real time."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


MODES = ("grouped", "sequential", "isolated")
ENGINE_CHOICES = ("gpu", "cpu", "both")
PINNING_MODES = ("none", "per-query")
PIN_TIERS = ("gpu", "host")
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
    REPO_ROOT,
    "build/release/extension/sirius/sirius.duckdb_extension",
)


def _git_capture(args):
    try:
        out = subprocess.run(
            args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_info():
    return _git_capture(["git", "rev-parse", "HEAD"]), _git_capture(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    )


def setup_benchmark_dir(
    output_root,
    mode,
    iterations,
    engine,
    queries,
    config_path,
    pinning_mode,
    pin_tier,
):
    """Create the benchmark output directory and return its paths.

    Layout:
      <output_root>/
        <benchmark_name>/
          config.yml         (copy of Sirius config, if provided)
          metadata.json
          csv/runtimes.csv
          log_dir/                  (SIRIUS_LOG_DIR target)
          <engine>/q<N>/result.txt  (one repr(row) per line)
          sirius/q<N>/sirius.log    (post-run split of combined log)
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_name = f"tpch_{ts}_{mode}_{engine}_iter{iterations}"
    benchmark_dir = os.path.join(output_root, benchmark_name)
    csv_dir = os.path.join(benchmark_dir, "csv")
    log_dir = os.path.join(benchmark_dir, "log_dir")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    runtime_csv = os.path.join(csv_dir, "runtimes.csv")

    if config_path and os.path.isfile(config_path):
        shutil.copy2(config_path, os.path.join(benchmark_dir, "config.yml"))

    commit, branch = get_git_info()
    metadata = {
        "commit": commit,
        "branch_name": branch,
        "date": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "iterations": iterations,
        "engine": engine,
        "queries": [f"q{q}" for q in queries],
        "pinning_mode": pinning_mode,
        "pin_tier": pin_tier if pinning_mode != "none" else None,
        "runtime_file": os.path.relpath(runtime_csv, benchmark_dir),
    }
    with open(os.path.join(benchmark_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return benchmark_dir, runtime_csv, log_dir


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


DEFAULT_OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


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
    else:
        log("OS cache dropped successfully")


def _resolve_parquet_files(parquet_dir, table):
    candidates = []
    for pattern in (
        os.path.join(parquet_dir, f"{table}.parquet"),
        os.path.join(parquet_dir, f"{table}_*.parquet"),
        os.path.join(parquet_dir, table, "*.parquet"),
    ):
        candidates.extend(sorted(glob.glob(pattern)))
    return candidates


def open_connection(source, gpu_execution=False):
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
    if gpu_execution:
        log(f"Loading Sirius extension from {EXTENSION_PATH}")
        con.execute(f"LOAD '{EXTENSION_PATH}'")
        log("Sirius extension loaded")
    return con


def _execute_multi(con, sql):
    """Execute a multi-statement SQL string, splitting on ';' and consuming any rows."""
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        con.execute(stmt).fetchall()


def time_query(con, qnum, use_gpu):
    engine_label = "GPU/sirius" if use_gpu else "CPU/duckdb"
    if use_gpu:
        log("  SET gpu_execution = true (GPU/sirius)")
        con.execute("SET gpu_execution = true;")
    log(f"  Executing q{qnum} on {engine_label}…")
    start = time.perf_counter()
    rows = con.execute(QUERIES[f"q{qnum}"]).fetchall()
    elapsed = time.perf_counter() - start
    log(f"  q{qnum} fetched {len(rows)} rows in {elapsed:.4f}s")
    return elapsed, rows


def _query_dir(benchmark_dir, engine_name, qnum):
    qdir = os.path.join(benchmark_dir, engine_name, f"q{qnum}")
    os.makedirs(qdir, exist_ok=True)
    return qdir


def _write_result(benchmark_dir, engine_name, qnum, rows):
    path = os.path.join(_query_dir(benchmark_dir, engine_name, qnum), "result.txt")
    with open(path, "w") as f:
        for row in rows:
            f.write(repr(row) + "\n")


def _record(writer, name, qnum, it, runtime):
    writer.writerow([name, f"q{qnum}", it, f"{runtime:.6f}"])
    log(f"[{name}] q{qnum} iter{it}: {runtime:.4f}s")


def _run_one(writer, con, name, qnum, it, use_gpu, benchmark_dir):
    elapsed, rows = time_query(con, qnum, use_gpu)
    _record(writer, name, qnum, it, elapsed)
    _write_result(benchmark_dir, name, qnum, rows)


def run_grouped(
    source,
    queries,
    engine_modes,
    iterations,
    writer,
    *,
    benchmark_dir,
    parquet_dir,
    pinning_mode,
):
    """Per-query iterations back-to-back; one connection per engine. Pin per query."""
    log(
        "Mode 'grouped': single connection per engine, iterations back-to-back per query"
    )
    pin_enabled = pinning_mode == "per-query"
    for name, use_gpu in engine_modes:
        con = open_connection(source, gpu_execution=use_gpu)
        try:
            for qnum in queries:
                if pin_enabled and use_gpu:
                    log(f"  Pinning tables for q{qnum}")
                    _execute_multi(con, emit_pin(qnum, parquet_dir))
                try:
                    for it in range(iterations):
                        log(f"--- q{qnum} iter{it} engine={name} ---")
                        _run_one(writer, con, name, qnum, it, use_gpu, benchmark_dir)
                finally:
                    if pin_enabled and use_gpu:
                        log(f"  Unpinning tables for q{qnum}")
                        _execute_multi(con, emit_unpin(qnum))
        finally:
            log("Closing connection")
            con.close()


def run_sequential(
    source,
    queries,
    engine_modes,
    iterations,
    writer,
    *,
    benchmark_dir,
    parquet_dir,
    pinning_mode,
):
    """Round-robin iterations; one connection per engine. Single union-pin at session start."""
    log("Mode 'sequential': single connection per engine, round-robin iterations")
    pin_enabled = pinning_mode == "per-query"
    for name, use_gpu in engine_modes:
        con = open_connection(source, gpu_execution=use_gpu)
        try:
            if pin_enabled and use_gpu:
                log("  Union-pinning all referenced TPC-H tables once at session start")
                _execute_multi(con, emit_pin_all(parquet_dir))
            try:
                for it in range(iterations):
                    for qnum in queries:
                        log(f"--- q{qnum} iter{it} engine={name} ---")
                        _run_one(writer, con, name, qnum, it, use_gpu, benchmark_dir)
            finally:
                if pin_enabled and use_gpu:
                    log("  Union-unpinning all TPC-H tables")
                    _execute_multi(con, emit_unpin_all())
        finally:
            log("Closing connection")
            con.close()


def run_isolated(
    source,
    queries,
    engine_modes,
    iterations,
    writer,
    *,
    benchmark_dir,
    parquet_dir,
    pinning_mode,
):
    """Fresh connection + OS cache drop per (query, iteration). Pin per execution."""
    log("Mode 'isolated': renewing connection and dropping OS cache before every run")
    pin_enabled = pinning_mode == "per-query"
    for name, use_gpu in engine_modes:
        for qnum in queries:
            for it in range(iterations):
                log(f"--- q{qnum} iter{it} engine={name} (cold connection) ---")
                con = open_connection(source, gpu_execution=use_gpu)
                try:
                    drop_os_cache()
                    if pin_enabled and use_gpu:
                        log(f"  Pinning tables for q{qnum}")
                        _execute_multi(con, emit_pin(qnum, parquet_dir))
                    _run_one(writer, con, name, qnum, it, use_gpu, benchmark_dir)
                    if pin_enabled and use_gpu:
                        log(f"  Unpinning tables for q{qnum}")
                        _execute_multi(con, emit_unpin(qnum))
                finally:
                    log("Closing connection")
                    con.close()


RUNNERS = {
    "grouped": run_grouped,
    "sequential": run_sequential,
    "isolated": run_isolated,
}


def split_sirius_log(log_dir, benchmark_dir, queries, iterations, mode):
    """Split the combined Sirius spdlog into one log file per query.

    Mirrors the bash post-processor in run_tpch_parquet.sh:591-628: find QueryBegin
    markers that aren't pin/unpin/CREATE VIEW, then partition them by query based on
    the iteration mode's run ordering.
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, "sirius*.log")))
    if not log_files:
        log("No sirius*.log found; skipping per-query log split")
        return
    log_path = log_files[0]
    log(f"Splitting {log_path} per query")
    with open(log_path) as f:
        lines = f.readlines()

    begin_indices = []
    for i, line in enumerate(lines):
        if "QueryBegin:" not in line:
            continue
        if "pin_table" in line or "unpin_table" in line or "CREATE VIEW" in line:
            continue
        begin_indices.append(i)

    expected = len(queries) * iterations
    if len(begin_indices) != expected:
        log(
            f"WARNING: expected {expected} QueryBegin lines in {log_path}, "
            f"found {len(begin_indices)}; skipping per-query log split"
        )
        return

    spans = []
    for i, start in enumerate(begin_indices):
        end = begin_indices[i + 1] if i + 1 < len(begin_indices) else len(lines)
        spans.append((start, end))

    per_query_spans = {q: [] for q in queries}
    if mode in ("grouped", "isolated"):
        for qi, q in enumerate(queries):
            for it in range(iterations):
                per_query_spans[q].append(spans[qi * iterations + it])
    else:  # sequential
        for it in range(iterations):
            for qi, q in enumerate(queries):
                per_query_spans[q].append(spans[it * len(queries) + qi])

    for q, span_list in per_query_spans.items():
        qdir = os.path.join(benchmark_dir, "sirius", f"q{q}")
        os.makedirs(qdir, exist_ok=True)
        out_path = os.path.join(qdir, "sirius.log")
        with open(out_path, "w") as out:
            for start, end in span_list:
                out.writelines(lines[start:end])
    log(f"Per-query Sirius logs written under {os.path.join(benchmark_dir, 'sirius')}/")


def validate(source, queries):
    """Compare CPU vs GPU result sets query-by-query."""
    con = open_connection(source, gpu_execution=True)
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
        "--mode",
        choices=MODES,
        default="grouped",
        help="Iteration ordering: grouped (per-query iterations back-to-back, hot "
        "cache), sequential (round-robin across queries), isolated (renew "
        "connection + drop OS cache per run)",
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
        help=(
            "Root output directory. A timestamped benchmark subdirectory is "
            "created inside it containing config.yml, metadata.json, "
            "csv/runtimes.csv, log_dir/, <engine>/q<N>/result.txt, and "
            "sirius/q<N>/sirius.log "
            f"(default: {DEFAULT_OUTPUT_ROOT})"
        ),
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
        "--pinning-mode",
        choices=PINNING_MODES,
        default="none",
        help=(
            "Pin TPC-H tables into the Sirius cache. 'per-query' pins/unpins "
            "around each query in grouped/isolated mode; in sequential mode it "
            "becomes a single union-pin at session start. Requires --input to "
            "point at a parquet directory. (default: none)"
        ),
    )
    p.add_argument(
        "--pin-tier",
        choices=PIN_TIERS,
        default="gpu",
        help="Sirius cache tier when --pinning-mode is per-query (default: gpu)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    source = args.input or "performance_test.duckdb"
    queries = parse_query_spec(args.queries)
    engine_modes = resolve_engine_modes(args.engine)
    output_root = args.output or DEFAULT_OUTPUT_ROOT

    parquet_dir = source if os.path.isdir(source) else None
    if args.pinning_mode != "none":
        if args.engine == "cpu":
            raise SystemExit(
                "--pinning-mode is Sirius-only; cannot be combined with --engine cpu"
            )
        if parquet_dir is None:
            raise SystemExit(
                "--pinning-mode per-query requires --input to point at a parquet directory"
            )

    config_path = (args.config or "").strip()
    if config_path:
        os.environ["SIRIUS_CONFIG_FILE"] = config_path
    else:
        log(
            "SIRIUS_CONFIG_FILE not set and --config not provided — "
            "running with Sirius default configuration."
        )

    if args.pinning_mode != "none":
        os.environ["SIRIUS_PIN_TIER"] = args.pin_tier

    benchmark_dir, runtime_csv, log_dir = setup_benchmark_dir(
        output_root,
        args.mode,
        args.iterations,
        args.engine,
        queries,
        config_path,
        args.pinning_mode,
        args.pin_tier,
    )
    os.environ["SIRIUS_LOG_DIR"] = log_dir

    log(f"Source:        {source}")
    if args.sf is not None:
        log(f"Scale factor:  {args.sf}")
    log(f"Mode:          {args.mode}")
    log(f"Iterations:    {args.iterations}")
    log(f"Engine:        {args.engine}")
    log(f"Queries:       {queries}")
    log(f"Config:        {config_path or '(default)'}")
    log(f"Pinning mode:  {args.pinning_mode}")
    if args.pinning_mode != "none":
        log(f"Pin tier:      {args.pin_tier}")
    log(f"Benchmark dir: {benchmark_dir}")
    log(f"Runtime CSV:   {runtime_csv}")
    log(f"Log dir:       {log_dir}")

    with open(runtime_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["engine", "query", "iteration", "runtime_s"])
        f.flush()
        RUNNERS[args.mode](
            source,
            queries,
            engine_modes,
            args.iterations,
            writer,
            benchmark_dir=benchmark_dir,
            parquet_dir=parquet_dir,
            pinning_mode=args.pinning_mode,
        )

    log("Benchmark run complete")

    if any(use_gpu for _, use_gpu in engine_modes):
        split_sirius_log(log_dir, benchmark_dir, queries, args.iterations, args.mode)

    if args.validation:
        log("Starting validation")
        validate(source, queries)
        log("Validation complete")


if __name__ == "__main__":
    main()
