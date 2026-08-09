# =============================================================================
# Copyright 2026, Sirius Contributors.
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

"""TPC-H power and throughput runs for Sirius with the RF1/RF2 refresh functions.

Power run (update set 1):
    [clean pass]  RF1  ->  22-query stream 0 (timed; feeds Power@Size)  ->  RF2
                  ->  post-RF2 pass (timed; delete mask active)

Throughput run (update sets 2..N+1):
    N concurrent query streams (spec permutations 1..N) + 1 refresh stream running
    N RF1/RF2 pairs. Sirius serializes all queries on one engine-wide lock, so this
    measures throughput of concurrent submission, not overlapped execution.

--mode selects power, throughput, or both. In both mode the throughput run
continues on the same pinned database right after the power run, with no unpin or
repin, so it sees the update-set-1 rows the power run left behind. The single
modes each pin a fresh copy of the input.

Every stream runs the same fixed substitution parameters by default, which keeps
validation and the clean/post-RF1/post-RF2 comparison meaningful.
--vary-predicates instead runs each stream's own qgen-generated parameters from
--query-dir (see generate_tpch_queries.sh), as an official run requires;
validation is not supported there.

Metrics:
    Power@Size      = 3600 * SF / geomean(22 stream-0 query times + T_RF1 + T_RF2)
    Throughput@Size = (N * 22 * 3600 / measurement_interval) * SF
    QphH@Size       = sqrt(Power@Size * Throughput@Size)

The input must be a file-backed .duckdb with native TPC-H tables. Only pinned
duckdb-native tables get the MVCC insert-delta/delete-mask path that makes RF1/RF2
visible on the GPU. Each phase copies the input and mutates the copy, never the
original. Refresh files come from generate_tpch_refresh.sh.

Example:
    export SIRIUS_CONFIG_FILE=$(pwd)/test/cpp/integration/integration.yaml
    pixi run python test/tpch_performance/tpch_power_throughput.py \
        --sf 1 --input test_datasets/tpch_sf1.duckdb \
        --refresh-dir test_datasets/tpch_refresh_sf1
"""

import argparse
import contextlib
import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import duckdb
from performance_test import (
    DEFAULT_OUTPUT_ROOT,
    EXTENSION_PATH,
    TPCH_TABLES,
    VALIDATION_ABS_TOL,
    _rows_match,
    get_git_info,
    log,
)
from queries import QUERIES
from tpch_pin_columns import QUERY_COLUMNS, union_columns_by_table
from tpch_query_streams import load_stream
from tpch_stream_permutations import default_streams, stream_order

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Queries that touch neither lineitem nor orders are unchanged by RF1/RF2, so
# refresh validation skips them.
REFRESH_INVARIANT_QUERIES = frozenset(
    q
    for q, tables in QUERY_COLUMNS.items()
    if not ({"lineitem", "orders"} & tables.keys())
)

_write_lock = threading.Lock()


def sql(cur, stmt):
    """Execute a statement and fetch all rows.

    Fetching fully matters: an open DuckDB result holds Sirius's engine-wide
    query lock and blocks the next query on any cursor.
    """
    return cur.execute(stmt).fetchall()


# ---------------------------------------------------------------------------
# Refresh functions
# ---------------------------------------------------------------------------


def refresh_file(refresh_dir, name):
    path = os.path.join(refresh_dir, name)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise SystemExit(
            f"Refresh file missing or empty: {path}\n"
            "Generate it with test/tpch_performance/generate_tpch_refresh.sh "
            "<SF> <num_sets> (num_sets >= streams + 1)."
        )
    return path


def rf1_statements(refresh_dir, n):
    orders = refresh_file(refresh_dir, f"orders.tbl.u{n}")
    lineitem = refresh_file(refresh_dir, f"lineitem.tbl.u{n}")
    return [
        f"COPY orders FROM '{orders}' (HEADER false, DELIMITER '|')",
        f"COPY lineitem FROM '{lineitem}' (HEADER false, DELIMITER '|')",
    ]


def rf2_statements(refresh_dir, n):
    delete = refresh_file(refresh_dir, f"delete.{n}")
    # dbgen delete files are "orderkey|" lines; column0 is the key.
    keys = f"SELECT column0 FROM read_csv('{delete}', delim='|', header=false)"
    return [
        f"DELETE FROM lineitem WHERE l_orderkey IN ({keys})",
        f"DELETE FROM orders WHERE o_orderkey IN ({keys})",
    ]


def check_refresh_matches_base(cur, refresh_dir, n):
    """Fail early if update set `n` does not belong to the input database.

    The base tables and the refresh sets come from different generators, so a
    set from the wrong scale factor still parses and inserts cleanly. Comparing
    keys is not enough to catch it: orderkeys are 75% sparse and delete sets
    start at the low end, so a larger set's keys all exist in a smaller
    database too. Sizes do catch it, since each set covers 0.1% of ORDERS.
    """
    keys = (
        f"SELECT column0 FROM read_csv("
        f"'{refresh_file(refresh_dir, f'delete.{n}')}', delim='|', header=false)"
    )
    rows = sql(cur, f"SELECT count(*) FROM ({keys}) t")[0][0]
    base = sql(cur, "SELECT count(*) FROM orders")[0][0]
    expected = base / 1000.0
    if not 0.5 * expected <= rows <= 2.0 * expected:
        raise SystemExit(
            f"Update set {n} does not match --input: delete.{n} holds {rows} "
            f"keys, but ORDERS has {base} rows and each set should cover about "
            f"{expected:.0f}. Check that the database and --refresh-dir are the "
            "same scale factor."
        )
    found = sql(cur, f"SELECT count(*) FROM orders WHERE o_orderkey IN ({keys})")[0][0]
    if found != rows:
        raise SystemExit(
            f"Update set {n} does not match --input: {found} of {rows} "
            f"delete.{n} keys exist in ORDERS."
        )


def run_refresh(cur, statements, label):
    """Run one refresh function as a single transaction and return its wall time."""
    start = time.perf_counter()
    sql(cur, "BEGIN TRANSACTION")
    try:
        for stmt in statements:
            sql(cur, stmt)
        sql(cur, "COMMIT")
    except Exception:
        sql(cur, "ROLLBACK")
        raise
    elapsed = time.perf_counter() - start
    log(f"  {label}: {elapsed:.4f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Query execution / validation helpers
# ---------------------------------------------------------------------------


def _is_select(stmt):
    return stmt.lstrip().lower().startswith("select")


def timed_query(cur, statements, timeout_s):
    """Run one query as a single transaction; return (elapsed_s, rows).

    A query is timed end to end including its transaction, matching the
    reference harness. Read-only queries take a READ ONLY transaction; older
    template sets write q15 as a view create/select/drop trio, which needs a
    writable one. Results are fetched in full because an open result holds
    Sirius's engine-wide query lock and would stall every other stream.
    """
    begin = (
        "BEGIN TRANSACTION READ ONLY"
        if all(_is_select(s) for s in statements)
        else "BEGIN TRANSACTION"
    )
    timer = None
    if timeout_s:
        timer = threading.Timer(timeout_s, cur.interrupt)
        timer.daemon = True
        timer.start()
    start = time.perf_counter()
    try:
        cur.execute(begin).fetchall()
        rows = []
        for stmt in statements:
            result = cur.execute(stmt).fetchall()
            if _is_select(stmt):
                rows = result
        cur.execute("COMMIT").fetchall()
    except Exception:
        try:
            cur.execute("ROLLBACK").fetchall()
        except Exception:  # noqa: BLE001 - the original error is what matters
            pass
        raise
    finally:
        if timer is not None:
            timer.cancel()
    return time.perf_counter() - start, rows


def stream_queries(stream, args):
    """[(qnum, [statements]), ...] in this stream's execution order.

    With fixed predicates every stream runs the same built-in SQL, permuted per
    the spec. With --vary-predicates the stream's qgen file supplies both the
    order and that stream's own substitution parameters.
    """
    if args.vary_predicates:
        return load_stream(args.query_dir, stream, args.sf)
    return [(q, [QUERIES[f"q{q}"]]) for q in stream_order(stream)]


def compare_rows(cpu_rows, gpu_rows):
    """Return None on match, else a description of the first difference."""
    if len(cpu_rows) != len(gpu_rows):
        return f"row count mismatch: cpu={len(cpu_rows)} gpu={len(gpu_rows)}"
    cpu_sorted = sorted(cpu_rows, key=str)
    gpu_sorted = sorted(gpu_rows, key=str)
    for i, (c, g) in enumerate(zip(cpu_sorted, gpu_sorted)):
        if not _rows_match(c, g, VALIDATION_ABS_TOL):
            return f"row {i} mismatch:\n      cpu: {c!r}\n      gpu: {g!r}"
    return None


def validate_pass(cpu, gpu_rows_by_q, plan, label, timeout_s):
    """Diff stored GPU rows against fresh DuckDB CPU runs on the current state."""
    failures = {}
    for q, statements in plan:
        if q in REFRESH_INVARIANT_QUERIES:
            continue
        _, cpu_rows = timed_query(cpu, statements, timeout_s)
        msg = compare_rows(cpu_rows, gpu_rows_by_q[q])
        if msg is None:
            log(f"  [{label}] q{q}: OK")
        else:
            failures[f"q{q}"] = msg
            log(f"  [{label}] q{q}: MISMATCH — {msg}")
    return failures


def _gpu_rows_path(run_dir, label):
    return os.path.join(run_dir, f"_gpu_rows_{label}.pickle")


def stash_gpu_rows(run_dir, label, rows_by_q):
    """Persist a GPU pass's rows for the deferred CPU comparison."""
    with open(_gpu_rows_path(run_dir, label), "wb") as f:
        pickle.dump(rows_by_q, f, protocol=pickle.HIGHEST_PROTOCOL)


def deferred_validation(args, run_dir):
    """Diff the stored GPU rows against pure DuckDB, in a fresh process.

    Validation cannot share a process with the pinned run. Sirius's host pool
    is a growing pool allocator, so unpinning returns blocks to the pool rather
    than to the OS, and DuckDB sizes its own memory_limit from total system RAM
    with no knowledge of what Sirius holds. At SF1000 that means a CPU-side q9
    allocates into a machine that is already 320 GB spoken for and gets
    OOM-killed. A fresh interpreter that never loads the extension starts with
    the machine to itself.

    The cost of leaving the process is that the refreshed state goes with it:
    RF1/RF2 are committed but never checkpointed, so a new process reading the
    same file would see the pre-refresh image. The worker therefore replays the
    refresh functions on its own copy of the base database, reproducing the
    post-RF1 and post-RF2 states the GPU was measured against. It runs after
    the pinned phases have deleted their scratch copy, so peak disk is
    unchanged.
    """
    spec = {
        "input": args.input,
        "refresh_dir": args.refresh_dir,
        "run_dir": run_dir,
        "scratch": os.path.join(run_dir, "bench_validate.duckdb"),
        "sf": args.sf,
        "vary_predicates": args.vary_predicates,
        "query_dir": args.query_dir,
        "query_timeout": args.query_timeout,
        "keep_scratch_db": args.keep_scratch_db,
    }
    spec_path = os.path.join(run_dir, "_validate_spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)

    log("=== Validation: pure DuckDB, refresh functions replayed (untimed) ===")
    env = dict(os.environ)
    env.pop("SIRIUS_CONFIG_FILE", None)  # nothing here should load the extension
    proc = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--validate-worker",
            spec_path,
            "--sf",  # required by the parser; the worker reads the spec instead
            str(args.sf),
        ],
        env=env,
    )
    verdict_path = os.path.join(run_dir, "_validate_verdict.json")
    if not os.path.exists(verdict_path):
        return {
            "after_rf1": {"worker": f"no verdict (exit {proc.returncode})"},
            "after_rf2": {},
        }
    with open(verdict_path) as f:
        return json.load(f)


def validation_worker(spec_path):
    """Child-process entry point: replay the refreshes, diff GPU rows vs CPU."""
    with open(spec_path) as f:
        spec = json.load(f)
    run_dir = spec["run_dir"]
    plan = stream_queries(
        0,
        argparse.Namespace(
            vary_predicates=spec["vary_predicates"],
            query_dir=spec["query_dir"],
            sf=spec["sf"],
        ),
    )
    scratch = spec["scratch"]
    copy_database(spec["input"], scratch)
    verdict = {}
    con = duckdb.connect(scratch)
    try:
        cur = con.cursor()
        limit = cur.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        log(f"  DuckDB memory_limit: {limit} (whole machine; no pinned pool held)")
        for label, refresh in (
            ("after_rf1", rf1_statements(spec["refresh_dir"], 1)),
            ("after_rf2", rf2_statements(spec["refresh_dir"], 1)),
        ):
            run_refresh(cur, refresh, label.upper().replace("AFTER_", ""))
            rows_path = _gpu_rows_path(run_dir, label)
            if not os.path.exists(rows_path):
                continue
            with open(rows_path, "rb") as f:
                gpu_rows = pickle.load(f)
            verdict[label] = validate_pass(
                cur, gpu_rows, plan, label.replace("_", " "), spec["query_timeout"]
            )
    finally:
        con.close()
        if not spec["keep_scratch_db"]:
            for f in (scratch, scratch + ".wal"):
                if os.path.exists(f):
                    os.remove(f)
    with open(os.path.join(run_dir, "_validate_verdict.json"), "w") as f:
        json.dump(verdict, f)


def table_counts(cpu):
    return {
        t: sql(cpu, f"SELECT count(*) FROM {t}")[0][0] for t in ("orders", "lineitem")
    }


def count_lines(path):
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def write_result(run_dir, phase, qnum, name, rows):
    qdir = os.path.join(run_dir, phase, f"q{qnum}")
    os.makedirs(qdir, exist_ok=True)
    with open(os.path.join(qdir, name), "w") as f:
        for row in rows:
            f.write(repr(row) + "\n")


# ---------------------------------------------------------------------------
# Connection / database handling
# ---------------------------------------------------------------------------


def copy_database(src, dst):
    log(f"Copying {src} -> {dst}")
    for stale in (dst, dst + ".wal"):
        if os.path.exists(stale):
            os.remove(stale)
    shutil.copy2(src, dst)
    _evict_page_cache(dst, dirty=True)
    _evict_page_cache(src, dirty=False)


def _evict_page_cache(path, dirty):
    """Drop a file's page cache before LOADing Sirius.

    On coherent-memory hosts (GB300) the copy's pages land partly on the GPU's
    NUMA node and block the RMM pool allocation at extension load. The data is
    re-read once for the pin and served from Sirius memory afterwards, so the
    cache buys nothing. Dirty pages can't be dropped, hence the fsync first.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            if dirty:
                os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError as e:
        log(f"  page-cache eviction skipped for {path}: {e}")


def open_benchmark_db(scratch_db, pin_tier, compression_plan_dir=None):
    """Connect writable, LOAD Sirius, CHECKPOINT, pin the TPC-H tables."""
    con = duckdb.connect(scratch_db, config={"allow_unsigned_extensions": "true"})
    log(f"Loading Sirius extension from {EXTENSION_PATH}")
    sql(con, f"LOAD '{EXTENSION_PATH}'")
    sql(con, "CHECKPOINT")
    if compression_plan_dir:
        log(f"Enabling Simpatico pin compression (plans: {compression_plan_dir})")
        sql(con, "SET pin_table_compression = true")
        sql(
            con,
            f"SET pin_table_input_compression_plan_dir = '{compression_plan_dir}'",
        )
    if pin_tier != "none":
        # Pin once up front, with only the columns the 22 queries actually read
        # (their union). Pinning whole tables also loads columns no TPC-H query
        # ever references — ps_comment, l_comment, o_clerk, p_comment — and at
        # SF1000 partsupp alone then needs >76 GB against a 471 GB host pool,
        # which does not fit. Refreshed rows land in the insert delta / delete
        # mask over these same columns, so RF1/RF2 stay visible on the GPU.
        for t, cols in union_columns_by_table().items():
            col_literals = ",".join(f"'{c}'" for c in cols)
            log(f"  Pinning {t} ({len(cols)} cols, tier={pin_tier})")
            sql(
                con,
                f"CALL pin_table(format='duckdb', name='{t}', "
                f"tier='{pin_tier}', cols=[{col_literals}])",
            )
    return con


def close_benchmark_db(con, pin_tier):
    try:
        if pin_tier != "none":
            for t in TPCH_TABLES:
                sql(con, f"CALL unpin_table('{t}')")
    finally:
        con.close()


def compressed_pin_count(log_dir, timeout_s=10.0):
    """Count pins the engine compressed, via the pin_table INFO log marker.

    The marker is written during CALL pin_table, well before the pins finish,
    but the log sink may flush late; poll briefly before concluding zero.
    """

    def scan():
        found = 0
        for name in os.listdir(log_dir):
            if not name.endswith(".log"):
                continue
            with open(os.path.join(log_dir, name), errors="replace") as f:
                found += sum("compressing with plan" in line for line in f)
        return found

    deadline = time.time() + timeout_s
    count = scan()
    while count == 0 and time.time() < deadline:
        time.sleep(0.5)
        count = scan()
    return count


@contextlib.contextmanager
def benchmark_db(args, run_dir, filename):
    """Copy the base DB into run_dir, open it with the TPC-H tables pinned, and
    unpin/close/delete the copy on exit."""
    scratch = os.path.join(run_dir, filename)
    copy_database(args.input, scratch)
    con = open_benchmark_db(
        scratch,
        args.pin,
        args.compression_plan_dir if args.pin_compression else None,
    )
    try:
        if args.pin_compression:
            compressed = compressed_pin_count(os.environ["SIRIUS_LOG_DIR"])
            if compressed == 0:
                raise SystemExit(
                    "--pin-compression was requested but no pinned table was "
                    "compressed; the run would silently measure uncompressed "
                    "data. Check the sirius log for '[pin_table]' warnings."
                )
            log(f"Simpatico engaged: {compressed} pinned table(s) compressed")
        yield con
    finally:
        close_benchmark_db(con, args.pin)
        if not args.keep_scratch_db:
            for f in (scratch, scratch + ".wal"):
                if os.path.exists(f):
                    os.remove(f)


# ---------------------------------------------------------------------------
# Power run
# ---------------------------------------------------------------------------


def geomean(values):
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def clamp_query_times(query_times):
    """Raise query times below slowest/1000 up to that floor (clause 5.4.1.4).

    Power@Size is a geometric mean, so one very fast query would otherwise pull
    it up without bound. The spec caps the spread at 1000:1. Returns the
    adjusted times and how many changed; below a 1000:1 spread nothing does.
    """
    if not query_times:
        return query_times, 0
    floor = max(query_times.values()) / 1000.0
    adjusted = {q: max(t, floor) for q, t in query_times.items()}
    changed = sum(1 for q, t in query_times.items() if adjusted[q] != t)
    return adjusted, changed


def power_run(con, args, run_dir, writer):
    plan = stream_queries(0, args)
    gpu = con.cursor()
    cpu = con.cursor()
    rf = con.cursor()
    try:
        sql(gpu, "SET gpu_execution = true")
        sql(cpu, "SET gpu_execution = false")
        sql(rf, "SET gpu_execution = false")

        def timed_pass(phase, result_name):
            times, rows_by_q = {}, {}
            for q, statements in plan:
                elapsed, rows = timed_query(gpu, statements, args.query_timeout)
                times[q], rows_by_q[q] = elapsed, rows
                with _write_lock:
                    writer.writerow([phase, 0, f"q{q}", f"{elapsed:.6f}"])
                write_result(run_dir, "power", q, result_name, rows)
                log(f"  q{q}: {elapsed:.4f}s ({len(rows)} rows)")
            return times, rows_by_q

        check_refresh_matches_base(cpu, args.refresh_dir, 1)
        counts_base = table_counts(cpu)
        log(f"Base counts: {counts_base}")

        t_clean = {}
        if args.baseline_pass:
            log("=== Power: clean pass (pre-refresh baseline; not in Power@Size) ===")
            t_clean, _ = timed_pass("power_clean", "result_clean.txt")

        log("=== Power: RF1 (insert update set 1) ===")
        t_rf1 = run_refresh(rf, rf1_statements(args.refresh_dir, 1), "RF1")
        with _write_lock:
            writer.writerow(["power", 0, "rf1", f"{t_rf1:.6f}"])

        counts_rf1 = table_counts(cpu)
        expected_orders = count_lines(refresh_file(args.refresh_dir, "orders.tbl.u1"))
        expected_lineitem = count_lines(
            refresh_file(args.refresh_dir, "lineitem.tbl.u1")
        )
        counts_ok = (
            counts_rf1["orders"] == counts_base["orders"] + expected_orders
            and counts_rf1["lineitem"] == counts_base["lineitem"] + expected_lineitem
        )
        log(f"Counts after RF1: {counts_rf1} (as expected: {counts_ok})")

        log("=== Power: query stream 0 (timed; feeds Power@Size) ===")
        t_p0, rows_p0 = timed_pass("power", "result_postrf1.txt")

        validation = {"counts_after_rf1_ok": counts_ok}
        if args.validation:
            stash_gpu_rows(run_dir, "after_rf1", rows_p0)

        log("=== Power: RF2 (delete update set 1) ===")
        t_rf2 = run_refresh(rf, rf2_statements(args.refresh_dir, 1), "RF2")
        with _write_lock:
            writer.writerow(["power", 0, "rf2", f"{t_rf2:.6f}"])

        counts_rf2 = table_counts(cpu)
        expected_deletes = count_lines(refresh_file(args.refresh_dir, "delete.1"))
        counts_ok = (
            counts_rf2["orders"] == counts_rf1["orders"] - expected_deletes
            and counts_rf2["lineitem"] < counts_rf1["lineitem"]
        )
        validation["counts_after_rf2_ok"] = counts_ok
        log(f"Counts after RF2: {counts_rf2} (as expected: {counts_ok})")

        log("=== Power: post-RF2 pass (delete mask active; not in Power@Size) ===")
        t_p2, rows_p2 = timed_pass("power_postrf2", "result_postrf2.txt")

        if args.validation:
            stash_gpu_rows(run_dir, "after_rf2", rows_p2)
            log(
                "Validation deferred to a pure-DuckDB pass after the pinned "
                "phases release the GPU and host pools"
            )

        metric_times, clamped = clamp_query_times(t_p0)
        if clamped:
            log(
                f"Clamped {clamped} query time(s) up to "
                f"{max(t_p0.values()) / 1000.0:.6f}s (1000:1 spread limit)"
            )
        power_at_size = (
            3600.0 * args.sf / geomean(list(metric_times.values()) + [t_rf1, t_rf2])
        )
        log(f"Power@Size = {power_at_size:.2f}")
        return {
            "power_at_size": power_at_size,
            "clamped_query_times": clamped,
            "t_rf1": t_rf1,
            "t_rf2": t_rf2,
            "query_times_stream0": {f"q{q}": t for q, t in t_p0.items()},
            "clean_times": {f"q{q}": t for q, t in t_clean.items()},
            "postrf2_times": {f"q{q}": t for q, t in t_p2.items()},
            "counts": {
                "base": counts_base,
                "after_rf1": counts_rf1,
                "after_rf2": counts_rf2,
            },
            "validation": validation,
        }
    finally:
        for c in (gpu, cpu, rf):
            c.close()


# ---------------------------------------------------------------------------
# Throughput run
# ---------------------------------------------------------------------------


def throughput_run(con, args, run_dir, writer, streams):
    check_cur = con.cursor()
    sql(check_cur, "SET gpu_execution = false")
    try:
        check_refresh_matches_base(check_cur, args.refresh_dir, 2)
    finally:
        check_cur.close()

    barrier = threading.Barrier(streams + 2)  # N query + 1 refresh + main
    errors = []
    stream_times = {i: {} for i in range(1, streams + 1)}
    stream_elapsed = {}
    refresh_times = []

    def fail(label, e):
        """Record a worker failure and release everyone waiting on the barrier.

        Setup runs inside the guard because a worker that dies before
        barrier.wait() never arrives, and a no-show does not break a barrier:
        the remaining parties would wait forever. Catch BaseException because
        stream_queries raises SystemExit, which is not an Exception.
        """
        errors.append(f"{label}: {e}")
        log(f"  [{label}] FAILED: {e}")
        barrier.abort()

    def query_stream(i):
        cur = None
        try:
            cur = con.cursor()
            sql(cur, "SET gpu_execution = true")
            plan = stream_queries(i, args)
            barrier.wait()
            start = time.perf_counter()
            for q, statements in plan:
                elapsed, _ = timed_query(cur, statements, args.query_timeout)
                stream_times[i][f"q{q}"] = elapsed
                with _write_lock:
                    writer.writerow(["throughput", i, f"q{q}", f"{elapsed:.6f}"])
                log(f"  [stream {i}] q{q}: {elapsed:.4f}s")
            stream_elapsed[i] = time.perf_counter() - start
        except threading.BrokenBarrierError:
            pass  # another worker failed and reported the cause
        except BaseException as e:  # noqa: BLE001 - reported after join
            fail(f"query stream {i}", e)
        finally:
            if cur is not None:
                cur.close()

    def refresh_stream():
        cur = None
        try:
            cur = con.cursor()
            sql(cur, "SET gpu_execution = false")
            barrier.wait()
            for pair in range(1, streams + 1):
                n = pair + 1  # set 1 belongs to the power run
                t1 = run_refresh(
                    cur, rf1_statements(args.refresh_dir, n), f"RF1(set {n})"
                )
                t2 = run_refresh(
                    cur, rf2_statements(args.refresh_dir, n), f"RF2(set {n})"
                )
                refresh_times.append({"set": n, "rf1": t1, "rf2": t2})
                with _write_lock:
                    writer.writerow(
                        ["throughput", "refresh", f"rf1_set{n}", f"{t1:.6f}"]
                    )
                    writer.writerow(
                        ["throughput", "refresh", f"rf2_set{n}", f"{t2:.6f}"]
                    )
        except threading.BrokenBarrierError:
            pass  # another worker failed and reported the cause
        except BaseException as e:  # noqa: BLE001 - reported after join
            fail("refresh stream", e)
        finally:
            if cur is not None:
                cur.close()

    threads = [
        threading.Thread(target=query_stream, args=(i,), name=f"stream-{i}")
        for i in range(1, streams + 1)
    ]
    threads.append(threading.Thread(target=refresh_stream, name="refresh"))
    log(f"=== Throughput: {streams} query streams + 1 refresh stream ===")
    for t in threads:
        t.start()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass  # a worker failed before starting; the error is raised after join
    start = time.perf_counter()
    for t in threads:
        t.join()
    interval = time.perf_counter() - start

    if errors:
        raise RuntimeError("throughput run failed:\n  " + "\n  ".join(errors))

    throughput_at_size = streams * 22 * 3600.0 / interval * args.sf
    log(
        f"Throughput@Size = {throughput_at_size:.2f} "
        f"(measurement interval {interval:.2f}s)"
    )
    return {
        "throughput_at_size": throughput_at_size,
        "streams": streams,
        "measurement_interval_s": interval,
        "per_stream_times": {str(i): stream_times[i] for i in stream_times},
        "per_stream_elapsed_s": {str(i): stream_elapsed.get(i) for i in stream_times},
        "refresh_times": refresh_times,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_summary(run_dir, args, streams, power, throughput, qphh):
    lines = []
    out = lines.append
    out("=" * 72)
    predicates = "qgen" if args.vary_predicates else "fixed"
    pin_label = args.pin + ("+simpatico" if args.pin_compression else "")
    out(
        f"  TPC-H Power / Throughput — Sirius   "
        f"(SF{args.sf:g}, pin={pin_label}, predicates={predicates})"
    )
    out("=" * 72)
    if power:
        out("")
        out(
            f"{'Query':<7} | {'clean':>9} | {'post-RF1':>9} | {'post-RF2':>9}"
            f" | {'delta oh':>9} | {'mask oh':>9}"
        )
        out("-" * 66)

        def fmt(v):
            return f"{v:.3f}s" if v is not None else "-"

        clean = power["clean_times"]
        p0 = power["query_times_stream0"]
        p2 = power["postrf2_times"]
        for q in stream_order(0):
            key = f"q{q}"
            c = clean.get(key)
            delta_oh = p0[key] - c if c is not None else None
            mask_oh = p2[key] - p0[key]
            out(
                f"{key:<7} | {fmt(c):>9} | {fmt(p0[key]):>9} | {fmt(p2[key]):>9}"
                f" | {fmt(delta_oh):>9} | {fmt(mask_oh):>9}"
            )
        out("-" * 66)
        total_clean = sum(clean.values()) if clean else None
        out(
            f"{'TOTAL':<7} | {fmt(total_clean):>9} | {fmt(sum(p0.values())):>9}"
            f" | {fmt(sum(p2.values())):>9} |"
        )
        out("")
        out(
            f"RF1 (insert): {power['t_rf1']:.3f}s    "
            f"RF2 (delete): {power['t_rf2']:.3f}s"
        )
        val = power["validation"]
        skipped = ", ".join(f"q{q}" for q in sorted(REFRESH_INVARIANT_QUERIES))
        for label in ("after_rf1", "after_rf2"):
            if label in val:
                n_fail = len(val[label])
                status = "PASS" if n_fail == 0 else f"FAIL ({n_fail} queries)"
                out(
                    f"Validation {label}: {status} "
                    f"({skipped} skipped: refresh-invariant)"
                )
        out("")
        if power.get("clamped_query_times"):
            out(
                f"Note: {power['clamped_query_times']} query time(s) raised to the "
                "1000:1 spread floor before Power@Size"
            )
        out(f"Power@Size       = {power['power_at_size']:.2f}")
    if throughput:
        out(
            f"Throughput@Size  = {throughput['throughput_at_size']:.2f}   "
            f"({streams} streams, interval "
            f"{throughput['measurement_interval_s']:.2f}s)"
        )
    if qphh is not None:
        out(f"QphH@Size        = {qphh:.2f}")
    out("=" * 72)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(text)
    print(text)


def write_run_info(run_dir, args, streams):
    commit, branch = get_git_info()
    info = [
        f"date: {datetime.now().isoformat(timespec='seconds')}",
        f"branch: {branch}",
        f"commit: {commit}",
        f"extension: {EXTENSION_PATH}",
        f"input: {args.input}",
        f"refresh_dir: {args.refresh_dir}",
        f"sf: {args.sf:g}",
        f"streams: {streams}",
        f"pin: {args.pin}",
        f"pin_compression: {args.pin_compression}",
        f"compression_plan_dir: "
        f"{args.compression_plan_dir if args.pin_compression else '(off)'}",
        f"mode: {args.mode}",
        f"vary_predicates: {args.vary_predicates}",
        f"query_dir: {args.query_dir if args.vary_predicates else '(fixed)'}",
        f"validation: {args.validation}",
        f"config: {os.environ.get('SIRIUS_CONFIG_FILE', '(default)')}",
    ]
    with open(os.path.join(run_dir, "run_info.txt"), "w") as f:
        f.write("\n".join(info) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="TPC-H power & throughput runs with RF1/RF2 refresh functions"
    )
    p.add_argument(
        "--sf", type=float, required=True, help="Scale factor (metric input)"
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Base .duckdb with native TPC-H tables (copied per phase; default "
        "test_datasets/tpch_sf<SF>.duckdb)",
    )
    p.add_argument(
        "--refresh-dir",
        type=str,
        default=None,
        help="Directory with dbgen -U refresh files (default "
        "test_datasets/tpch_refresh_sf<SF>)",
    )
    p.add_argument(
        "--streams",
        type=int,
        default=None,
        help="Throughput query streams (default: TPC-H spec minimum for SF)",
    )
    p.add_argument(
        "--mode",
        choices=("power", "throughput", "both"),
        default="both",
    )
    p.add_argument(
        "--pin",
        choices=("gpu", "host", "none"),
        default="gpu",
        help="Cache tier for the 8 TPC-H tables. 'none' disables pinning, which "
        "also disables GPU serving of refreshed tables (debug only).",
    )
    p.add_argument(
        "--pin-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compress the pinned tables with Simpatico; needs a pinned tier "
        "and per-table plan files (--compression-plan-dir)",
    )
    p.add_argument(
        "--compression-plan-dir",
        type=str,
        default=None,
        help="Directory of per-table Simpatico plan files (<table>.<ext>) for "
        "--pin-compression (default: the SF1000 plans under "
        "src/compression/simpatico_codegen/plans)",
    )
    p.add_argument(
        "--vary-predicates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run each stream's own qgen substitution parameters from "
        "--query-dir instead of the fixed built-in literals; not compatible "
        "with --validation",
    )
    p.add_argument(
        "--query-dir",
        type=str,
        default=None,
        help="Directory of qgen stream<N>.sql files for --vary-predicates "
        "(default test_datasets/tpch_queries_sf<SF>)",
    )
    p.add_argument(
        "--config",
        type=str,
        default=os.environ.get("SIRIUS_CONFIG_FILE", ""),
        help="Sirius config YAML (required; taken from $SIRIUS_CONFIG_FILE when "
        "set — there is no built-in default path)",
    )
    p.add_argument("--output", type=str, default=None, help="Output root directory")
    p.add_argument("--validate-worker", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--query-timeout",
        type=float,
        default=1200.0,
        help="Per-query timeout in seconds; 0 disables (default: 1200)",
    )
    p.add_argument(
        "--validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Diff GPU vs DuckDB CPU results after RF1 and RF2 (power run; "
        "default: on unless --vary-predicates)",
    )
    p.add_argument(
        "--baseline-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a clean pre-RF1 pass (timed, not in metric) to isolate delta/mask cost",
    )
    p.add_argument(
        "--keep-scratch-db",
        action="store_true",
        help="Keep the per-phase database copies instead of deleting them",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Re-entry as the validation child: no Sirius, no config, no pinning.
    if args.validate_worker:
        validation_worker(args.validate_worker)
        return

    sf_label = f"{args.sf:g}"

    if args.input is None:
        args.input = os.path.join(REPO_ROOT, f"test_datasets/tpch_sf{sf_label}.duckdb")
    if not os.path.isfile(args.input):
        raise SystemExit(f"--input not found: {args.input}")
    if args.refresh_dir is None:
        args.refresh_dir = os.path.join(
            REPO_ROOT, f"test_datasets/tpch_refresh_sf{sf_label}"
        )
    if args.query_dir is None:
        args.query_dir = os.path.join(
            REPO_ROOT, f"test_datasets/tpch_queries_sf{sf_label}"
        )

    streams = args.streams if args.streams is not None else default_streams(args.sf)

    # Validation diffs GPU vs CPU rows, so it needs the fixed default predicates.
    if args.vary_predicates and args.validation:
        raise SystemExit("--vary-predicates does not support --validation")
    if args.validation is None:
        args.validation = not args.vary_predicates

    # Simpatico compression happens at pin time, so it needs a pinned tier, and
    # it is a no-op without at least one plan file naming a TPC-H table.
    compression_tables = []
    if args.pin_compression:
        if args.pin == "none":
            raise SystemExit("--pin-compression needs a pinned tier (--pin gpu|host)")
        if args.compression_plan_dir is None:
            args.compression_plan_dir = os.path.join(
                REPO_ROOT, "src/compression/simpatico_codegen/plans/tpch_sf1000"
            )
        args.compression_plan_dir = os.path.abspath(args.compression_plan_dir)
        if not os.path.isdir(args.compression_plan_dir):
            raise SystemExit(
                f"--compression-plan-dir not found: {args.compression_plan_dir}"
            )
        plan_stems = {
            os.path.splitext(f)[0] for f in os.listdir(args.compression_plan_dir)
        }
        compression_tables = sorted(t for t in TPCH_TABLES if t in plan_stems)
        if not compression_tables:
            raise SystemExit(
                f"No plan file in {args.compression_plan_dir} names a TPC-H "
                "table; plan files are <table>.<ext>"
            )

    # Fail fast if any needed refresh set or query stream is missing.
    needed = []
    if args.mode in ("power", "both"):
        needed.append(1)
    if args.mode in ("throughput", "both"):
        needed.extend(range(2, streams + 2))
    for n in needed:
        rf1_statements(args.refresh_dir, n)
        rf2_statements(args.refresh_dir, n)
    if args.vary_predicates:
        wanted = [0] if args.mode in ("power", "both") else []
        if args.mode in ("throughput", "both"):
            wanted.extend(range(1, streams + 1))
        for n in wanted:
            load_stream(args.query_dir, n, args.sf)

    # Require an explicit config; it sets GPU/host memory sizing and a stale one
    # can fail extension init.
    config = (args.config or "").strip()
    if not config:
        raise SystemExit(
            "No Sirius config given: pass --config <yaml> or set SIRIUS_CONFIG_FILE "
            "(e.g. test/cpp/integration/integration.yaml)."
        )
    if not os.path.isfile(config):
        raise SystemExit(f"Sirius config not found: {config}")
    os.environ["SIRIUS_CONFIG_FILE"] = config

    output_root = args.output or DEFAULT_OUTPUT_ROOT
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"tpch_power_{ts}_sf{sf_label}_s{streams}")
    log_dir = os.path.join(run_dir, "log_dir")
    os.makedirs(log_dir, exist_ok=True)
    os.environ["SIRIUS_LOG_DIR"] = log_dir
    shutil.copy2(config, os.path.join(run_dir, "config.yml"))
    write_run_info(run_dir, args, streams)

    log(f"Run directory: {run_dir}")
    log(f"Input:         {args.input}")
    log(f"Refresh dir:   {args.refresh_dir}")
    log(f"Streams:       {streams}")
    log(f"Config:        {config}")
    log(f"Predicates:    {'qgen per stream' if args.vary_predicates else 'fixed'}")
    if args.vary_predicates:
        log(f"Query dir:     {args.query_dir}")
    if args.pin_compression:
        log(
            f"Compression:   simpatico ({len(compression_tables)} table plans: "
            f"{', '.join(compression_tables)})"
        )

    power = throughput = None
    # A failing throughput run must not discard a power run that already
    # succeeded: its rows are stashed and still worth validating, and its
    # metrics are still worth writing. The failure is carried to the exit
    # status at the end instead of aborting here.
    throughput_error = None
    with open(os.path.join(run_dir, "timings.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "stream", "element", "seconds"])
        if args.mode == "both":
            # One pinned database for both phases: throughput continues on the
            # post-power state (update set 1 applied), with no unpin/repin.
            with benchmark_db(args, run_dir, "bench.duckdb") as con:
                power = power_run(con, args, run_dir, writer)
                f.flush()
                try:
                    throughput = throughput_run(con, args, run_dir, writer, streams)
                except Exception as e:  # noqa: BLE001 - reported at exit
                    throughput_error = str(e)
                    log(f"Throughput run failed, continuing to validation: {e}")
        elif args.mode == "power":
            with benchmark_db(args, run_dir, "bench_power.duckdb") as con:
                power = power_run(con, args, run_dir, writer)
        elif args.mode == "throughput":
            with benchmark_db(args, run_dir, "bench_throughput.duckdb") as con:
                throughput = throughput_run(con, args, run_dir, writer, streams)

    # Deliberately outside the benchmark_db blocks above: the tables are
    # unpinned, the connection is closed and the pinned scratch copy is gone,
    # so the child process gets the machine to itself.
    if power and args.validation:
        power["validation"].update(deferred_validation(args, run_dir))

    qphh = None
    if power and throughput:
        qphh = math.sqrt(power["power_at_size"] * throughput["throughput_at_size"])

    commit, branch = get_git_info()
    metrics = {
        "benchmark": "tpch-power-throughput",
        "sf": args.sf,
        "streams": streams,
        "pin": args.pin,
        "pin_compression": args.pin_compression,
        "compression_plan_dir": (
            args.compression_plan_dir if args.pin_compression else None
        ),
        "vary_predicates": args.vary_predicates,
        "query_dir": args.query_dir if args.vary_predicates else None,
        "input": args.input,
        "refresh_dir": args.refresh_dir,
        "commit": commit,
        "branch": branch,
        "date": datetime.now().isoformat(timespec="seconds"),
        "power": power,
        "throughput": throughput,
        "throughput_error": throughput_error,
        "qphh_at_size": qphh,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    write_summary(run_dir, args, streams, power, throughput, qphh)
    log(f"Metrics written to {os.path.join(run_dir, 'metrics.json')}")

    failures = []
    if throughput_error:
        failures.append(f"throughput run: {throughput_error}")
    if power:
        val = power["validation"]
        if not val.get("counts_after_rf1_ok", True):
            failures.append("counts after RF1 unexpected")
        if not val.get("counts_after_rf2_ok", True):
            failures.append("counts after RF2 unexpected")
        for label in ("after_rf1", "after_rf2"):
            if val.get(label):
                failures.append(f"validation {label}: {sorted(val[label])}")
    if failures:
        raise SystemExit("FAILED: " + "; ".join(failures))


if __name__ == "__main__":
    main()
