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
import math
import os
import shutil
import subprocess
import time
from datetime import date, datetime, time as dtime, timedelta
from decimal import Decimal

import duckdb
from queries import QUERIES
from tpch_pin_columns import (
    emit_pin,
    emit_pin_all,
    emit_unpin,
    emit_unpin_all,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log(msg):
    """Timestamped, flushed progress log so hangs are visible in real time."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


MODES = ("grouped", "sequential", "isolated")
ENGINE_CHOICES = ("gpu", "cpu", "both")
PIN_CHOICES = ("none", "gpu", "host")
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
    pin,
    name=None,
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

    If `name` is provided, the benchmark dir is suffixed:
    `<output_root>/tpch_<ts>_<mode>_<engine>_iter<N>-<name>`; otherwise the
    default `tpch_<ts>_<mode>_<engine>_iter<N>` is used.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_name = f"tpch_{ts}_{mode}_{engine}_iter{iterations}"
    if name:
        benchmark_name = f"{benchmark_name}-{name}"
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
        "pin": pin,
        "runtime_file": os.path.relpath(runtime_csv, benchmark_dir),
    }
    with open(os.path.join(benchmark_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return benchmark_dir, runtime_csv, log_dir


VALIDATION_ABS_TOL = 1e-10

_RESULT_EVAL_GLOBALS = {
    "__builtins__": {},
    "Decimal": Decimal,
    "datetime": datetime,
    "date": date,
    "time": dtime,
    "timedelta": timedelta,
}


def _parse_result_row(line):
    return eval(line.strip(), _RESULT_EVAL_GLOBALS)  # noqa: S307


def _load_result_file(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_parse_result_row(line))
    return rows


def _values_match(a, b, abs_tol):
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=0.0, abs_tol=abs_tol)
    return a == b


def _rows_match(a, b, abs_tol):
    if len(a) != len(b):
        return False
    return all(_values_match(av, bv, abs_tol) for av, bv in zip(a, b))


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


def open_connection(parquet_dir, gpu_execution=False):
    """Open an in-memory DuckDB, register TPC-H parquet views, optionally LOAD Sirius."""
    log(f"Opening DuckDB connection over parquet dir {parquet_dir}")
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    for table in TPCH_TABLES:
        files = _resolve_parquet_files(parquet_dir, table)
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for table '{table}' in {parquet_dir}"
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


def time_query(con, qnum, use_gpu, profile_path=None):
    engine_label = "GPU/sirius" if use_gpu else "CPU/duckdb"
    if use_gpu:
        log("  SET gpu_execution = true (GPU/sirius)")
        con.execute("SET gpu_execution = true;")
    elif profile_path is not None:
        # DuckDB CPU profiling: emit a per-operator JSON profile for the next
        # query. 'detailed' mode includes the physical operator tree (with
        # operator_timing / operator_cardinality) plus planner/optimizer phase
        # timings. NOTE: profiling adds overhead, so the runtime_s recorded for
        # profiled CPU runs is slightly inflated vs an unprofiled baseline.
        log(f"  Enabling DuckDB JSON profiling -> {profile_path}")
        con.execute("PRAGMA enable_profiling='json';")
        con.execute("PRAGMA profiling_mode='detailed';")
        con.execute(f"PRAGMA profiling_output='{profile_path}';")
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


def _record(collector, name, qnum, it, runtime):
    collector[(name, it, qnum)] = runtime
    log(f"[{name}] q{qnum} iter{it}: {runtime:.4f}s")


def _run_one(
    writer, con, name, qnum, it, use_gpu, benchmark_dir, duckdb_profiling=False
):
    profile_path = None
    if duckdb_profiling and not use_gpu:
        # Write one JSON profile per (query, iteration) so no iteration overwrites
        # another. The comparison tooling can select whichever iteration it wants.
        profile_path = os.path.join(
            _query_dir(benchmark_dir, name, qnum), f"profile_iter{it}.json"
        )
    elapsed, rows = time_query(con, qnum, use_gpu, profile_path=profile_path)
    _record(writer, name, qnum, it, elapsed)
    _write_result(benchmark_dir, name, qnum, rows)


CSV_ENGINE_ORDER = ("sirius", "duckdb")


def _csv_engine_sort_key(engine_name):
    try:
        return CSV_ENGINE_ORDER.index(engine_name)
    except ValueError:
        return len(CSV_ENGINE_ORDER)


def write_runtimes_csv(path, collector, queries, engine_modes, iterations):
    """Write wide-format runtimes CSV (one column per engine/iteration, msec)."""
    ordered_engines = sorted(
        (name for name, _ in engine_modes), key=_csv_engine_sort_key
    )
    col_keys = [(name, it) for name in ordered_engines for it in range(iterations)]
    header = ["query"] + [f"{name}_{it}_msec" for name, it in col_keys]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q in queries:
            row = [f"q{q}"]
            for name, it in col_keys:
                runtime = collector.get((name, it, q))
                row.append("" if runtime is None else f"{runtime * 1000:.3f}")
            writer.writerow(row)
        total_row = ["total"]
        for name, it in col_keys:
            total = sum(collector.get((name, it, q), 0.0) for q in queries)
            total_row.append(f"{total * 1000:.3f}")
        writer.writerow(total_row)


def run_grouped(
    source,
    queries,
    engine_modes,
    iterations,
    collector,
    *,
    benchmark_dir,
    parquet_dir,
    pin,
    duckdb_profiling=False,
):
    """Per-query iterations back-to-back; one connection per engine. Pin per query."""
    log(
        "Mode 'grouped': single connection per engine, iterations back-to-back per query"
    )
    pin_enabled = pin != "none"
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
                        _run_one(
                            writer,
                            con,
                            name,
                            qnum,
                            it,
                            use_gpu,
                            benchmark_dir,
                            duckdb_profiling,
                        )
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
    collector,
    *,
    benchmark_dir,
    parquet_dir,
    pin,
    duckdb_profiling=False,
):
    """Round-robin iterations; one connection per engine. Single union-pin at session start."""
    log("Mode 'sequential': single connection per engine, round-robin iterations")
    pin_enabled = pin != "none"
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
                        _run_one(
                            writer,
                            con,
                            name,
                            qnum,
                            it,
                            use_gpu,
                            benchmark_dir,
                            duckdb_profiling,
                        )
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
    collector,
    *,
    benchmark_dir,
    parquet_dir,
    pin,
    duckdb_profiling=False,
):
    """Fresh connection + OS cache drop per (query, iteration). Pin per execution."""
    log("Mode 'isolated': renewing connection and dropping OS cache before every run")
    pin_enabled = pin != "none"
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
                    _run_one(
                        writer,
                        con,
                        name,
                        qnum,
                        it,
                        use_gpu,
                        benchmark_dir,
                        duckdb_profiling,
                    )
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


def validate(benchmark_dir, queries):
    """Compare saved DuckDB vs Sirius result.txt files.

    Mirrors compare_results.py: byte-exact match first, then a tolerance-aware
    fallback (abs_tol=VALIDATION_ABS_TOL on Python float values only; strict
    equality on Decimal/int/str/date/etc.). No query re-execution, no DuckDB
    connection — safe to run after the GPU pool from the timed pass is still
    resident.
    """
    log(f"Validating saved results in {benchmark_dir}")
    results = {}
    for qnum in queries:
        qname = f"q{qnum}"
        duck_path = os.path.join(benchmark_dir, "duckdb", qname, "result.txt")
        sir_path = os.path.join(benchmark_dir, "sirius", qname, "result.txt")
        if not os.path.exists(duck_path) or not os.path.exists(sir_path):
            print(f"❌ {qname}: missing result.txt (duckdb or sirius)")
            results[qnum] = False
            continue
        with open(duck_path, "rb") as fd, open(sir_path, "rb") as fs:
            if fd.read() == fs.read():
                print(f"✓ {qname}: byte-exact match")
                results[qnum] = True
                continue
        try:
            duck_rows = _load_result_file(duck_path)
            sir_rows = _load_result_file(sir_path)
        except Exception as e:
            print(f"❌ {qname}: parse error - {e}")
            results[qnum] = False
            continue
        if len(duck_rows) != len(sir_rows):
            print(
                f"❌ {qname}: row count mismatch - "
                f"duckdb={len(duck_rows)} sirius={len(sir_rows)}"
            )
            results[qnum] = False
            continue
        duck_sorted = sorted(duck_rows, key=lambda x: str(x))
        sir_sorted = sorted(sir_rows, key=lambda x: str(x))
        mismatch = None
        for i, (d, s) in enumerate(zip(duck_sorted, sir_sorted)):
            if not _rows_match(d, s, VALIDATION_ABS_TOL):
                mismatch = (i, d, s)
                break
        if mismatch is None:
            print(
                f"✓ {qname}: within tolerance "
                f"(abs_tol={VALIDATION_ABS_TOL}, {len(duck_rows)} rows)"
            )
            results[qnum] = True
        else:
            i, d, s = mismatch
            print(f"❌ {qname}: row {i} mismatch")
            print(f"   duckdb: {d}")
            print(f"   sirius: {s}")
            results[qnum] = False

    passed = sum(1 for v in results.values() if v)
    failed = [f"q{q}" for q, ok in results.items() if not ok]
    print(f"\n{'=' * 60}")
    print(f"Validation Summary: {passed}/{len(results)} queries passed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'=' * 60}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Run TPC-H performance tests")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a TPC-H parquet directory (one .parquet file or subdir per table)",
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
        help=(
            "After timing, validate that GPU and CPU produce matching results "
            "by comparing the saved <engine>/q<N>/result.txt files. Byte-exact "
            f"match first, then abs_tol={VALIDATION_ABS_TOL} on float columns "
            "(strict equality elsewhere). Requires --engine both."
        ),
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
        "--pin",
        choices=PIN_CHOICES,
        default="none",
        help=(
            "Pin TPC-H tables into the Sirius cache (Sirius-only). 'gpu' or "
            "'host' selects the cache tier; 'none' disables pinning. Pin is "
            "per-query in grouped/isolated mode and a single union-pin at "
            "session start in sequential mode. (default: none)"
        ),
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Suffix appended to the default benchmark output subdirectory "
            "name, producing 'tpch_<ts>_<mode>_<engine>_iter<N>-<name>' "
            "under --output."
        ),
    )
    p.add_argument(
        "--duckdb-profiling",
        action="store_true",
        help=(
            "For CPU/DuckDB runs, enable DuckDB's JSON profiler (detailed mode) "
            "and write a per-operator profile to "
            "<benchmark_dir>/duckdb/q<N>/profile_iter<it>.json. Has no effect on "
            "GPU/Sirius runs (Sirius per-operator timing comes from its own logs). "
            "Profiling adds overhead, so profiled CPU runtime_s values are slightly "
            "inflated vs an unprofiled baseline."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    source = args.input
    if not os.path.isdir(source):
        raise SystemExit(
            f"--input must be a parquet directory; got {source!r}. "
            ".duckdb database files are not supported."
        )
    parquet_dir = source
    queries = parse_query_spec(args.queries)
    engine_modes = resolve_engine_modes(args.engine)
    output_root = args.output or DEFAULT_OUTPUT_ROOT

    if args.pin != "none" and args.engine == "cpu":
        raise SystemExit("--pin is Sirius-only; cannot be combined with --engine cpu")

    if args.validation and args.engine != "both":
        raise SystemExit(
            "--validation requires --engine both (needs both result sets to compare)"
        )

    config_path = (args.config or "").strip()
    if config_path:
        os.environ["SIRIUS_CONFIG_FILE"] = config_path
    else:
        log(
            "SIRIUS_CONFIG_FILE not set and --config not provided — "
            "running with Sirius default configuration."
        )

    if args.pin != "none":
        os.environ["SIRIUS_PIN_TIER"] = args.pin

    benchmark_dir, runtime_csv, log_dir = setup_benchmark_dir(
        output_root,
        args.mode,
        args.iterations,
        args.engine,
        queries,
        config_path,
        args.pin,
        name=args.name,
    )
    os.environ["SIRIUS_LOG_DIR"] = log_dir

    log(f"Source:        {source}")
    log(f"Mode:          {args.mode}")
    log(f"Iterations:    {args.iterations}")
    log(f"Engine:        {args.engine}")
    log(f"Queries:       {queries}")
    log(f"Config:        {config_path or '(default)'}")
    log(f"Pin:           {args.pin}")
    log(f"DuckDB profiling: {args.duckdb_profiling}")
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
            pin=args.pin,
            duckdb_profiling=args.duckdb_profiling,
        )

    write_runtimes_csv(runtime_csv, collector, queries, engine_modes, args.iterations)
    log("Benchmark run complete")

    if any(use_gpu for _, use_gpu in engine_modes):
        split_sirius_log(log_dir, benchmark_dir, queries, args.iterations, args.mode)

    if args.validation:
        log("Starting validation")
        validate(benchmark_dir, queries)
        log("Validation complete")


if __name__ == "__main__":
    main()
