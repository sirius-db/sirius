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
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Tuple

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


MODES = ("grouped", "sequential", "isolated", "nsys-profile", "ab")
ENGINE_CHOICES = ("gpu", "cpu", "both")
PIN_CHOICES = ("none", "gpu", "host")
DATA_SOURCE_CHOICES = ("parquet", "duckdb")
AB_CELL_CHOICES = ("disk_hot", "host_pinned", "cold_spotcheck", "clustered_stretch")
# TPC-H queries carrying a LIMIT: the Top-N dynamic filter's whole surface. They take the tighter
# resolution target (--target-resolution-limit); everything else takes --target-resolution-other.
LIMIT_QUERIES = (2, 3, 10, 18, 21)
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

BUILD_PATH = os.environ.get("SIRIUS_BUILD_PATH", "build/release")

EXTENSION_PATH = os.path.join(
    REPO_ROOT, BUILD_PATH, "extension/sirius/sirius.duckdb_extension"
)

DUCKDB_BIN = os.path.join(REPO_ROOT, BUILD_PATH, "duckdb")


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
    nsys_profile=False,
    data_source="parquet",
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

    If `name` is provided, the benchmark dir is `<output_root>/<name>` (no
    timestamp); otherwise the default `tpch_<ts>_<mode>_<engine>_iter<N>` is used.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_name = f"tpch_{ts}_{mode}_{engine}_iter{iterations}"
    if name:
        benchmark_name = f"{benchmark_name}_{name}"
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
        "data_source": data_source,
        "queries": [f"q{q}" for q in queries],
        "pin": pin,
        "nsys_profile": nsys_profile,
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


def _build_views_sql(parquet_dir):
    """Return CREATE OR REPLACE VIEW SQL for the 8 TPC-H tables.

    Each statement is terminated with `;\n`. Raises FileNotFoundError if any
    table has no parquet files in `parquet_dir`.
    """
    parts = []
    for table in TPCH_TABLES:
        files = _resolve_parquet_files(parquet_dir, table)
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for table '{table}' in {parquet_dir}"
            )
        file_list = ",".join(f"'{f}'" for f in files)
        parts.append(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet([{file_list}]);"
        )
    return "\n".join(parts) + "\n"


def open_connection(source, gpu_execution=False, data_source="parquet"):
    """Open a DuckDB connection over the benchmark source, optionally LOAD Sirius.

    parquet: in-memory DB with CREATE VIEW ... read_parquet over the directory.
    duckdb:  open the .duckdb file directly (read-only); its native TPC-H tables
             are already present in the `main` schema, so no views are registered.
             Read-only avoids write locks / accidental WAL and is correct for a
             read-only benchmark; it mirrors how run_tpch_duckdb.sh opens the file.
    """
    if data_source == "duckdb":
        log(f"Opening DuckDB database file {source} (read-only)")
        con = duckdb.connect(
            source, read_only=True, config={"allow_unsigned_extensions": "true"}
        )
    else:
        log(f"Opening DuckDB connection over parquet dir {source}")
        con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
        log("Registering TPC-H parquet views")
        for stmt in _build_views_sql(source).split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            con.execute(stmt)
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


def _record(writer, name, qnum, it, runtime):
    writer.writerow([name, f"q{qnum}", it, f"{runtime:.6f}"])
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


def run_grouped(
    source,
    queries,
    engine_modes,
    iterations,
    writer,
    *,
    benchmark_dir,
    pin,
    data_source="parquet",
    duckdb_profiling=False,
):
    """Per-query iterations back-to-back; one connection per engine. Pin per query."""
    log(
        "Mode 'grouped': single connection per engine, iterations back-to-back per query"
    )
    pin_enabled = pin != "none"
    for name, use_gpu in engine_modes:
        con = open_connection(source, gpu_execution=use_gpu, data_source=data_source)
        try:
            for qnum in queries:
                if pin_enabled and use_gpu:
                    log(f"  Pinning tables for q{qnum}")
                    _execute_multi(con, emit_pin(qnum, source, data_source))
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
    writer,
    *,
    benchmark_dir,
    pin,
    data_source="parquet",
    duckdb_profiling=False,
):
    """Round-robin iterations; one connection per engine. Single union-pin at session start."""
    log("Mode 'sequential': single connection per engine, round-robin iterations")
    pin_enabled = pin != "none"
    for name, use_gpu in engine_modes:
        con = open_connection(source, gpu_execution=use_gpu, data_source=data_source)
        try:
            if pin_enabled and use_gpu:
                log("  Union-pinning all referenced TPC-H tables once at session start")
                _execute_multi(con, emit_pin_all(source, data_source))
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
    writer,
    *,
    benchmark_dir,
    pin,
    data_source="parquet",
    duckdb_profiling=False,
):
    """Fresh connection + OS cache drop per (query, iteration). Pin per execution."""
    log("Mode 'isolated': renewing connection and dropping OS cache before every run")
    pin_enabled = pin != "none"
    for name, use_gpu in engine_modes:
        for qnum in queries:
            for it in range(iterations):
                log(f"--- q{qnum} iter{it} engine={name} (cold connection) ---")
                con = open_connection(
                    source, gpu_execution=use_gpu, data_source=data_source
                )
                try:
                    drop_os_cache()
                    if pin_enabled and use_gpu:
                        log(f"  Pinning tables for q{qnum}")
                        _execute_multi(con, emit_pin(qnum, source, data_source))
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


# =============================================================================
# A/B mode (Phase-5): interleaved flag-off/flag-on pairs in one session
# =============================================================================
#
# Ports the measurement discipline of test/cpp/utils/measurement_harness.hpp to the TPC-H runner:
# time-adjacent pairs with alternating lead arm, discarded warmup pairs, a resolution-driven stop
# (the pair count is an outcome, never a setting), per-pair nvidia-smi occupancy bracketing with
# discard-and-retry plus an idle park, and the paired geometric mean with its 95% CI in log space
# as the deciding statistic. The arm toggle is a per-connection `SET <ab_option>`; arming is
# asserted per execution from sirius_dynamic_filter_stats() counter deltas against a frozen
# expectations JSON, and the two arms' results must be byte-identical every pair.


class AbCellAbort(RuntimeError):
    """A (cell, query) is invalid -- e.g. an arming assertion failed, so the cell measured a
    feature that was not on. The query's cell is voided; the run stops so the configuration can
    be fixed."""


class AbRunAbort(RuntimeError):
    """The whole run is invalid -- the two arms returned different results, which is a
    correctness bug, not a measurement problem."""


# --- statistics, ported field-for-field from measurement_harness.hpp -----------------------------


def median_of(values: List[float]) -> float:
    """Median matching sample_series::median_of: middle element, or the mean of the two middle
    elements for an even count."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def log_ratio_stddev(ratios: List[float]) -> float:
    """Sample standard deviation of the per-pair log ratios (interleaved_ab_result::
    log_ratio_stddev): the query's own noise, independent of how many pairs were taken.
    """
    count = len(ratios)
    if count < 2:
        return 0.0
    logs = [math.log(r) for r in ratios]
    mean = sum(logs) / count
    variance = sum((x - mean) ** 2 for x in logs) / (count - 1)
    return math.sqrt(variance)


def paired_ratio_interval(ratios: List[float]) -> Tuple[float, float, float]:
    """Geometric mean of the per-pair ratios with its 95% CI, as (low, point, high), all 1.0 when
    fewer than two pairs survived (interleaved_ab_result::paired_ratio_interval). The interval,
    never the point estimate, decides a threshold: met when the whole interval sits below it,
    violated when the whole interval sits above, unresolved when it straddles."""
    count = len(ratios)
    if count < 2:
        return (1.0, 1.0, 1.0)
    mean = sum(math.log(r) for r in ratios) / count
    half_width = 1.96 * log_ratio_stddev(ratios) / math.sqrt(count)
    return (math.exp(mean - half_width), math.exp(mean), math.exp(mean + half_width))


def achieved_resolution(ratios: List[float]) -> float:
    """Half-width of the 95% interval, as a fraction -- a property of these samples, not of the
    schedule."""
    low, _, high = paired_ratio_interval(ratios)
    return 0.5 * (high - low)


def is_resolved(ratios: List[float], target_resolution: float) -> bool:
    """Whether the samples can distinguish an effect of target_resolution from zero."""
    return (
        len(ratios) >= 2
        and target_resolution > 0.0
        and achieved_resolution(ratios) <= target_resolution
    )


def pairs_needed_for_target(ratios: List[float], target_resolution: float) -> int:
    """Pairs the observed spread implies are needed to reach target_resolution
    (interleaved_ab_result::pairs_needed_for_target)."""
    count = len(ratios)
    if count < 2 or target_resolution <= 0.0:
        return 0
    half_width = 1.96 * log_ratio_stddev(ratios) / math.sqrt(count)
    if half_width <= 0.0:
        return count
    scale = half_width / math.log1p(target_resolution)
    return math.ceil(count * scale * scale)


def suite_geomean_interval(
    per_query_ratios: Dict[str, List[float]],
) -> Optional[Tuple[float, float, float]]:
    """Suite statistic: the unweighted mean of per-query log-geomeans, with its CI from the sum of
    the per-query variances of the mean divided by N^2 -- unweighted so a slow query cannot buy a
    regression in a fast one. Returns None until every query has >= 2 pairs."""
    contributing = {q: r for q, r in per_query_ratios.items() if len(r) >= 2}
    if len(contributing) != len(per_query_ratios) or not contributing:
        return None
    n_queries = len(contributing)
    log_means = []
    variance_of_means = []
    for ratios in contributing.values():
        logs = [math.log(r) for r in ratios]
        log_means.append(sum(logs) / len(logs))
        variance_of_means.append(log_ratio_stddev(ratios) ** 2 / len(ratios))
    suite_mean = sum(log_means) / n_queries
    suite_se = math.sqrt(sum(variance_of_means)) / n_queries
    return (
        math.exp(suite_mean - 1.96 * suite_se),
        math.exp(suite_mean),
        math.exp(suite_mean + 1.96 * suite_se),
    )


# --- GPU occupancy bracket (observe_gpu_occupancy semantics) --------------------------------------


@dataclass
class GpuOccupancy:
    """Device memory attributed to this process and to any other, as nvidia-smi reports it."""

    available: bool = False
    self_attributed: bool = False
    self_bytes: int = 0
    foreign_bytes: int = 0
    foreign_process_count: int = 0

    def describe(self) -> str:
        if not self.available:
            return "unavailable (nvidia-smi did not run)"
        if self.foreign_process_count == 0:
            if self.self_attributed:
                return f"quiet (self {self.self_bytes >> 20} MiB)"
            return "quiet, self not attributed"
        return (
            f"{self.foreign_process_count} foreign process(es) holding "
            f"{self.foreign_bytes >> 20} MiB"
            + (
                ""
                if self.self_attributed
                else " (self not attributed; attribution unreliable)"
            )
        )

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "self_attributed": self.self_attributed,
            "self_bytes": self.self_bytes,
            "foreign_bytes": self.foreign_bytes,
            "foreign_process_count": self.foreign_process_count,
        }


def observe_gpu_occupancy() -> GpuOccupancy:
    """Split compute-process device memory into this process's and everyone else's. Degrades to
    available=False (condition unknown) rather than failing when nvidia-smi cannot be reached.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return GpuOccupancy()
    if proc.returncode != 0:
        return GpuOccupancy()
    occupancy = GpuOccupancy(available=True)
    self_pid = os.getpid()
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mib = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        nbytes = mib << 20
        if pid == self_pid:
            occupancy.self_attributed = True
            occupancy.self_bytes += nbytes
        else:
            occupancy.foreign_bytes += nbytes
            occupancy.foreign_process_count += 1
    return occupancy


# --- pre-flight -----------------------------------------------------------------------------------


def file_sha256(path: str) -> Optional[str]:
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_meminfo() -> Dict[str, int]:
    """MemTotal/MemAvailable from /proc/meminfo, in kB -- recorded per cell so the from-disk
    host-RAM arithmetic (phase5_from_disk.yaml) is checked against the machine on every run.
    """
    values: Dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[f"{key}_kB"] = int(rest.split()[0])
    except OSError:
        pass
    return values


def probe_drop_caches_passwordless() -> bool:
    """Whether passwordless `sudo tee /proc/sys/vm/drop_caches` is configured, probed via
    `sudo -n -l` so the probe itself never drops the cache."""
    try:
        proc = subprocess.run(
            ["sudo", "-n", "-l", "/usr/bin/tee", "/proc/sys/vm/drop_caches"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def dataset_inventory(source: str, data_source: str) -> Dict[str, dict]:
    """Per-table file count and byte total, so the cell's metadata pins the dataset's identity and
    a missing or misshapen table fails before any timing."""
    inventory: Dict[str, dict] = {}
    if data_source == "duckdb":
        inventory["__file__"] = {"files": 1, "bytes": os.path.getsize(source)}
        return inventory
    for table in TPCH_TABLES:
        files = _resolve_parquet_files(source, table)
        if not files:
            raise SystemExit(
                f"preflight: no parquet files for table '{table}' in {source}"
            )
        inventory[table] = {
            "files": len(files),
            "bytes": sum(os.path.getsize(f) for f in files),
        }
    return inventory


def run_preflight(source: str, data_source: str, *, require_drop_caches: bool) -> dict:
    """AB pre-flight: extension identity, host RAM, GPU occupancy, drop_caches capability, and
    dataset inventory, all recorded into the cell's metadata so every number the cell produces
    carries the conditions it was produced under."""
    log("Running AB pre-flight checks")
    drop_caches_ok = probe_drop_caches_passwordless()
    if require_drop_caches and not drop_caches_ok:
        raise SystemExit(
            "preflight: --drop-caches-per-pair requires passwordless sudo for "
            "/usr/bin/tee /proc/sys/vm/drop_caches (see test/tpch_performance/CLAUDE.md)"
        )
    occupancy = observe_gpu_occupancy()
    if occupancy.available and occupancy.foreign_process_count > 0:
        log(f"WARNING: GPU not free at pre-flight: {occupancy.describe()}")
    preflight = {
        "extension_path": EXTENSION_PATH,
        "extension_sha256": file_sha256(EXTENSION_PATH),
        "meminfo": read_meminfo(),
        "drop_caches_passwordless": drop_caches_ok,
        "gpu_occupancy": occupancy.as_dict(),
        "gpu_occupancy_text": occupancy.describe(),
        "dataset_path": os.path.realpath(source),
        "dataset": dataset_inventory(source, data_source),
    }
    if preflight["extension_sha256"] is None:
        raise SystemExit(f"preflight: extension binary not found at {EXTENSION_PATH}")
    log(f"  extension sha256: {preflight['extension_sha256']}")
    log(f"  meminfo: {preflight['meminfo']}")
    log(f"  drop_caches passwordless: {drop_caches_ok}")
    log(f"  gpu: {preflight['gpu_occupancy_text']}")
    return preflight


# --- arming expectations --------------------------------------------------------------------------

ARMING_ARM_KEYS = ("off", "on")
ARMING_PREDICATE_KEYS = (
    "exact",  # field -> per-execution delta, exactly (plan-time counters)
    "nonzero",  # fields whose per-execution delta must be > 0
    "zero",  # fields whose per-execution delta must be == 0
    "zero_prefix",  # every field with this name prefix must have delta == 0
    "cell_nonzero",  # fields whose delta summed across the query's whole cell must be > 0
    "direction_only",  # informational: delivery-time fields asserted as >= 0 only
    "note",  # free-form calibration annotation
)


def load_arming_expectations(
    path: str,
    queries: List[int],
    extra_cell_nonzero: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """Load and structurally validate the frozen arming-expectations JSON; every query in the run
    must be covered (arming assertions are non-negotiable). `extra_cell_nonzero` merges
    additional per-query on-arm cell_nonzero requirements the calibration scale could not
    observe (e.g. q18's witness set never fills at SF1, where the qualifying groups sit below
    the LIMIT)."""
    with open(path) as f:
        doc = json.load(f)
    for qkey, counters in (extra_cell_nonzero or {}).items():
        spec = (
            doc["queries"].get(qkey) if isinstance(doc.get("queries"), dict) else None
        )
        if spec is None:
            raise SystemExit(
                f"--require-cell-nonzero: no expectations entry for {qkey}"
            )
        arm_spec = spec.setdefault("on", {})
        arm_spec["cell_nonzero"] = list(
            dict.fromkeys(list(arm_spec.get("cell_nonzero", [])) + list(counters))
        )
    if not isinstance(doc.get("queries"), dict):
        raise SystemExit(f"arming expectations {path}: missing 'queries' object")
    for qnum in queries:
        qkey = f"q{qnum}"
        spec = doc["queries"].get(qkey)
        if not isinstance(spec, dict):
            raise SystemExit(f"arming expectations {path}: no entry for {qkey}")
        for arm, arm_spec in spec.items():
            if arm == "note":
                continue
            if arm not in ARMING_ARM_KEYS:
                raise SystemExit(
                    f"arming expectations {path}: {qkey} has unknown arm '{arm}'"
                )
            for key, value in arm_spec.items():
                if key not in ARMING_PREDICATE_KEYS:
                    raise SystemExit(
                        f"arming expectations {path}: {qkey}.{arm} has unknown predicate '{key}'"
                    )
                if key == "exact":
                    if not isinstance(value, dict) or not all(
                        isinstance(v, int) for v in value.values()
                    ):
                        raise SystemExit(
                            f"arming expectations {path}: {qkey}.{arm}.exact must map "
                            "field -> integer delta"
                        )
                elif key != "note" and not isinstance(value, list):
                    raise SystemExit(
                        f"arming expectations {path}: {qkey}.{arm}.{key} must be a list"
                    )
    return doc


def validate_arming_field_names(expectations: dict, live_fields: List[str]) -> None:
    """Every field the expectations reference must exist in the live counter set, and every
    zero_prefix must match at least one field -- a typo would otherwise pass vacuously.
    """
    live = set(live_fields)
    for qkey, spec in expectations["queries"].items():
        for arm in ARMING_ARM_KEYS:
            arm_spec = spec.get(arm, {})
            named = list(arm_spec.get("exact", {}).keys())
            for key in ("nonzero", "zero", "cell_nonzero", "direction_only"):
                named.extend(arm_spec.get(key, []))
            for name in named:
                if name not in live:
                    raise SystemExit(
                        f"arming expectations: {qkey}.{arm} references unknown counter '{name}'"
                    )
            for prefix in arm_spec.get("zero_prefix", []):
                if not any(f.startswith(prefix) for f in live):
                    raise SystemExit(
                        f"arming expectations: {qkey}.{arm} zero_prefix '{prefix}' matches "
                        "no counter"
                    )


def evaluate_arming(deltas: Dict[str, int], arm_spec: dict) -> List[str]:
    """Per-execution arming predicates over one execution's counter deltas; returns violations."""
    violations = []
    for name, expected in arm_spec.get("exact", {}).items():
        if deltas.get(name, 0) != expected:
            violations.append(
                f"{name} delta {deltas.get(name, 0)} != expected {expected}"
            )
    for name in arm_spec.get("nonzero", []):
        if deltas.get(name, 0) == 0:
            violations.append(f"{name} delta expected > 0, got 0")
    for name in arm_spec.get("zero", []):
        if deltas.get(name, 0) != 0:
            violations.append(f"{name} delta expected 0, got {deltas.get(name, 0)}")
    for prefix in arm_spec.get("zero_prefix", []):
        for name, delta in deltas.items():
            if name.startswith(prefix) and delta != 0:
                violations.append(
                    f"{name} delta expected 0 (prefix {prefix}), got {delta}"
                )
    return violations


def evaluate_cell_arming(totals: Dict[str, int], arm_spec: dict) -> List[str]:
    """Cell-level direction-only predicates over the deltas accumulated across the whole query
    cell (e.g. Q18's witness set must have filled at least once across the cell)."""
    return [
        f"{name} cell total expected > 0, got 0"
        for name in arm_spec.get("cell_nonzero", [])
        if totals.get(name, 0) == 0
    ]


# --- pair schedule --------------------------------------------------------------------------------


@dataclass
class AbQueryPlan:
    """Sampling schedule for one (cell, query). pilot_pairs > 0 switches the stop rule from
    resolution-driven to a fixed kept-pair count (pilot mode)."""

    target_resolution: float = 0.02
    min_pairs: int = 21
    max_pairs: int = 400
    warmup_pairs: int = 3
    pilot_pairs: int = 0
    park_threshold_s: float = 600.0
    park_poll_s: float = 30.0


@dataclass
class ExecutionOutcome:
    """One arm execution: the timed interval, the captured result text, and the counter deltas
    taken (untimed) around the timed query."""

    ok: bool
    error: str = ""
    elapsed_s: float = 0.0
    result_text: str = ""
    deltas: Dict[str, int] = field(default_factory=dict)


@dataclass
class PairRecord:
    """One attempted pair, as written to pairs.csv."""

    pair_idx: int
    phase: str  # warmup | sample
    lead_arm: str
    off_s: float
    on_s: float
    ratio: float
    kept: bool
    discard_reason: str
    foreign_procs: int
    foreign_mib: int

    CSV_HEADER = (
        "pair_idx",
        "phase",
        "lead_arm",
        "off_s",
        "on_s",
        "ratio",
        "kept",
        "discard_reason",
        "foreign_procs",
        "foreign_mib",
    )

    def csv_row(self) -> list:
        return [
            self.pair_idx,
            self.phase,
            self.lead_arm,
            f"{self.off_s:.6f}",
            f"{self.on_s:.6f}",
            f"{self.ratio:.6f}",
            int(self.kept),
            self.discard_reason,
            self.foreign_procs,
            self.foreign_mib,
        ]


@dataclass
class QueryCellResult:
    """Everything one (cell, query) produced, including the conditions it was produced under."""

    query: str
    target_resolution: float
    off_s: List[float] = field(default_factory=list)
    on_s: List[float] = field(default_factory=list)
    paired_ratios: List[float] = field(default_factory=list)
    attempted_pairs: int = 0
    discard_notes: List[str] = field(default_factory=list)
    pairs_with_foreign: int = 0
    parked_s: float = 0.0
    before: GpuOccupancy = field(default_factory=GpuOccupancy)
    after: GpuOccupancy = field(default_factory=GpuOccupancy)
    worst: GpuOccupancy = field(default_factory=GpuOccupancy)
    arm_totals: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {"off": {}, "on": {}}
    )
    result_text: Dict[str, str] = field(default_factory=dict)
    arming_ok: bool = True

    def summary(self) -> dict:
        """cell_summary.json payload: all four statistics (their disagreement is information on a
        shared host), the CI, the achieved resolution, and the conditions."""
        low, point, high = paired_ratio_interval(self.paired_ratios)
        off_median = median_of(self.off_s)
        on_median = median_of(self.on_s)
        min_ratio = 0.0
        if self.off_s and self.on_s and min(self.off_s) > 0:
            min_ratio = min(self.on_s) / min(self.off_s)
        return {
            "query": self.query,
            "pairs_kept": len(self.paired_ratios),
            "pairs_attempted": self.attempted_pairs,
            "off_median_s": off_median,
            "on_median_s": on_median,
            "median_ratio": (on_median / off_median) if off_median > 0 else 0.0,
            "paired_median_ratio": median_of(self.paired_ratios),
            "min_ratio": min_ratio,
            "geomean_ratio": point,
            "ci_low": low,
            "ci_high": high,
            "log_ratio_stddev": log_ratio_stddev(self.paired_ratios),
            "target_resolution": self.target_resolution,
            "achieved_resolution": achieved_resolution(self.paired_ratios),
            "resolved": is_resolved(self.paired_ratios, self.target_resolution),
            "pairs_needed_for_target": pairs_needed_for_target(
                self.paired_ratios, self.target_resolution
            ),
            "arming": "pass" if self.arming_ok else "fail",
            "pairs_with_foreign": self.pairs_with_foreign,
            "parked_s": self.parked_s,
            "gpu_before": self.before.describe(),
            "gpu_after": self.after.describe(),
            "gpu_worst": self.worst.describe(),
            "discards": list(self.discard_notes),
        }


def park_until_quiet(
    occupancy_fn: Callable[[], GpuOccupancy],
    sleep_fn: Callable[[float], None],
    now_fn: Callable[[], float],
    poll_s: float,
    query_name: str,
) -> float:
    """Idle gate: poll until two consecutive quiet observations, returning the parked seconds.
    An unavailable observation counts as quiet so a host without nvidia-smi cannot park forever.
    """
    log(
        f"  [{query_name}] foreign GPU tenant persisted past the park threshold; parking"
    )
    start = now_fn()
    quiet_streak = 0
    while quiet_streak < 2:
        sleep_fn(poll_s)
        observed = occupancy_fn()
        if not observed.available or observed.foreign_process_count == 0:
            quiet_streak += 1
        else:
            quiet_streak = 0
    parked = now_fn() - start
    log(f"  [{query_name}] GPU quiet again after {parked:.0f}s parked; resuming")
    return parked


def run_ab_query_cell(
    query_name: str,
    executor: Callable[[str, int, str], ExecutionOutcome],
    plan: AbQueryPlan,
    arming_spec: Optional[dict],
    *,
    occupancy_fn: Callable[[], GpuOccupancy] = observe_gpu_occupancy,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.monotonic,
    drop_caches_fn: Optional[Callable[[], None]] = None,
    pair_log: Optional[Callable[[PairRecord], None]] = None,
) -> QueryCellResult:
    """Run the pair schedule for one query: warmup pairs, then sampled pairs with alternating lead
    arm until resolved (or pilot_pairs kept, or max_pairs attempted). `executor(arm, pair_idx,
    phase)` performs one execution of one arm; everything that must not be timed lives inside it,
    outside the interval it reports. Occupancy is sampled between pairs, never inside a timed
    interval. Raises AbCellAbort on an arming violation or an unrunnable query, AbRunAbort when
    the arms' results differ."""
    result = QueryCellResult(query=query_name, target_resolution=plan.target_resolution)

    def accumulate(arm: str, outcome: ExecutionOutcome) -> None:
        totals = result.arm_totals[arm]
        for name, delta in outcome.deltas.items():
            totals[name] = totals.get(name, 0) + delta

    def check_arming(
        arm: str, outcome: ExecutionOutcome, pair_idx: int, phase: str
    ) -> None:
        if arming_spec is None:
            return
        violations = evaluate_arming(outcome.deltas, arming_spec.get(arm, {}))
        if violations:
            result.arming_ok = False
            raise AbCellAbort(
                f"{query_name} {phase} pair {pair_idx} arm {arm}: arming violation "
                "(the cell measured a feature that was not on): "
                + "; ".join(violations)
            )

    def check_identity(
        outcomes: Dict[str, ExecutionOutcome], pair_idx: int, phase: str
    ) -> None:
        if not (outcomes["off"].ok and outcomes["on"].ok):
            return
        if outcomes["off"].result_text != outcomes["on"].result_text:
            # Distinguish a row-ORDER difference (ORDER BY ties resorted by an unstable GPU
            # sort under changed arrival order) from a row-CONTENT difference: both abort --
            # the arms must be byte-identical -- but the diagnostic decides where triage starts.
            off_sorted = sorted(outcomes["off"].result_text.splitlines())
            on_sorted = sorted(outcomes["on"].result_text.splitlines())
            kind = (
                "ordering-only difference (sorted contents identical; suspect ORDER BY ties)"
                if off_sorted == on_sorted
                else "content difference"
            )
            raise AbRunAbort(
                f"{query_name} {phase} pair {pair_idx}: flag-on result differs from flag-off "
                f"[{kind}] (correctness bug; aborting the whole run)"
            )

    def run_pair(
        pair_idx: int, phase: str, lead_arm: str
    ) -> Dict[str, ExecutionOutcome]:
        if drop_caches_fn is not None:
            drop_caches_fn()
        order = ("off", "on") if lead_arm == "off" else ("on", "off")
        outcomes: Dict[str, ExecutionOutcome] = {}
        for arm in order:
            outcomes[arm] = executor(arm, pair_idx, phase)
            if outcomes[arm].ok:
                accumulate(arm, outcomes[arm])
                check_arming(arm, outcomes[arm], pair_idx, phase)
        return outcomes

    def note_worst(observed: GpuOccupancy) -> None:
        if observed.available and observed.foreign_bytes >= result.worst.foreign_bytes:
            result.worst = observed

    # Warmup pairs absorb JIT, pool growth, and page-cache fill; they are discarded, but arming
    # and cross-arm identity hold on them too -- a violation is a configuration or correctness
    # error whenever it appears. An execution error here is fatal: an unrunnable query cannot be
    # measured, and burning max_pairs on it would only defer the same answer.
    for warmup_idx in range(plan.warmup_pairs):
        outcomes = run_pair(warmup_idx, "warmup", "off")
        for arm in ("off", "on"):
            if not outcomes[arm].ok:
                raise AbCellAbort(
                    f"{query_name} warmup pair {warmup_idx} arm {arm} failed: "
                    f"{outcomes[arm].error}"
                )
        check_identity(outcomes, warmup_idx, "warmup")
        if pair_log:
            off, on = outcomes["off"], outcomes["on"]
            pair_log(
                PairRecord(
                    warmup_idx,
                    "warmup",
                    "off",
                    off.elapsed_s,
                    on.elapsed_s,
                    on.elapsed_s / off.elapsed_s if off.elapsed_s > 0 else float("nan"),
                    False,
                    "warmup",
                    0,
                    0,
                )
            )

    result.before = occupancy_fn()
    result.worst = result.before
    foreign_since: Optional[float] = None

    def should_stop() -> bool:
        if plan.pilot_pairs > 0:
            return len(result.paired_ratios) >= plan.pilot_pairs
        # Stop once the interval is tight enough to decide the effect the cell was given, but
        # never before min_pairs KEPT ratios: an early interval is itself too noisy to trust as
        # a signal, and a discard-heavy cell must not quote a resolution over a handful of
        # survivors -- flooring on attempted pairs would let it.
        return len(result.paired_ratios) >= plan.min_pairs and is_resolved(
            result.paired_ratios, plan.target_resolution
        )

    pair = 0
    while pair < plan.max_pairs:
        if should_stop():
            break
        result.attempted_pairs += 1
        # The arm that runs first alternates, so the warmer second slot is shared evenly.
        lead_arm = "off" if pair % 2 == 0 else "on"
        outcomes = run_pair(pair, "sample", lead_arm)

        # Sampled between pairs, never inside a timed interval.
        observed = occupancy_fn()
        note_worst(observed)
        foreign = observed.available and observed.foreign_process_count > 0
        if foreign:
            result.pairs_with_foreign += 1

        discard_reason = ""
        for arm in ("off", "on"):
            if not outcomes[arm].ok:
                discard_reason = f"{arm} arm failed: {outcomes[arm].error}"
                break
        # Identity is checked whenever both arms produced results: a foreign-bracket discard
        # voids the pair's TIMING, never its correctness evidence.
        if not discard_reason:
            check_identity(outcomes, pair, "sample")
        if not discard_reason and foreign:
            discard_reason = (
                f"foreign GPU tenant in bracket: {observed.foreign_process_count} "
                f"process(es), {observed.foreign_bytes >> 20} MiB"
            )

        off_ok, on_ok = outcomes["off"].ok, outcomes["on"].ok
        off_s = outcomes["off"].elapsed_s if off_ok else float("nan")
        on_s = outcomes["on"].elapsed_s if on_ok else float("nan")
        ratio = on_s / off_s if (off_ok and on_ok and off_s > 0) else float("nan")
        if pair_log:
            pair_log(
                PairRecord(
                    pair,
                    "sample",
                    lead_arm,
                    off_s,
                    on_s,
                    ratio,
                    not discard_reason,
                    discard_reason,
                    observed.foreign_process_count,
                    observed.foreign_bytes >> 20,
                )
            )

        if discard_reason:
            result.discard_notes.append(f"pair {pair}: {discard_reason}")
        else:
            result.off_s.append(off_s)
            result.on_s.append(on_s)
            if off_s > 0:
                result.paired_ratios.append(on_s / off_s)
            result.result_text.setdefault("off", outcomes["off"].result_text)
            result.result_text.setdefault("on", outcomes["on"].result_text)

        now = now_fn()
        if foreign:
            if foreign_since is None:
                foreign_since = now
            elif now - foreign_since >= plan.park_threshold_s:
                result.parked_s += park_until_quiet(
                    occupancy_fn, sleep_fn, now_fn, plan.park_poll_s, query_name
                )
                foreign_since = None
        else:
            foreign_since = None
        pair += 1

    if arming_spec is not None:
        for arm in ("off", "on"):
            violations = evaluate_cell_arming(
                result.arm_totals[arm], arming_spec.get(arm, {})
            )
            if violations:
                result.arming_ok = False
                raise AbCellAbort(
                    f"{query_name} arm {arm}: cell-level arming violation: "
                    + "; ".join(violations)
                )

    result.after = occupancy_fn()
    note_worst(result.after)
    return result


# --- real executor and orchestration --------------------------------------------------------------


def read_dynamic_filter_counters(con) -> Dict[str, int]:
    """Counter snapshot via the sirius_dynamic_filter_stats() table function registered by the
    Sirius extension."""
    rows = con.execute(
        "SELECT name, value FROM sirius_dynamic_filter_stats();"
    ).fetchall()
    return {name: int(value) for name, value in rows}


class CounterCsvWriter:
    """Per-execution counter-delta CSV (counters_off.csv / counters_on.csv): one row per
    execution, one column per dynamic_filter_stats field, header written from the first row's
    field names so the column set tracks the engine's snapshot definition."""

    def __init__(self, path: str) -> None:
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._fields: Optional[List[str]] = None

    def write_row(self, pair_idx: int, phase: str, deltas: Dict[str, int]) -> None:
        if self._fields is None:
            self._fields = list(deltas.keys())
            self._writer.writerow(["pair_idx", "phase", *self._fields])
        self._writer.writerow([pair_idx, phase, *(deltas[f] for f in self._fields)])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def make_ab_executor(
    con,
    qnum: int,
    cell: str,
    ab_option: str,
    counter_writers: Dict[str, CounterCsvWriter],
) -> Callable[[str, int, str], ExecutionOutcome]:
    """One arm execution: SET the arm, label it for Quent attribution (a stored string; harmless
    with telemetry off), snapshot counters, run the timed query, snapshot again. Only the query
    itself sits inside the timed interval."""
    query_sql = QUERIES[f"q{qnum}"]

    def execute(arm: str, pair_idx: int, phase: str) -> ExecutionOutcome:
        label = f"phase5_{cell}_q{qnum}_{arm}_{pair_idx}"
        try:
            con.execute(f"SET {ab_option} = {'true' if arm == 'on' else 'false'};")
            con.execute(f"CALL sirius_set_query_label('{label}');").fetchall()
            before = read_dynamic_filter_counters(con)
            start = time.perf_counter()
            rows = con.execute(query_sql).fetchall()
            elapsed = time.perf_counter() - start
            after = read_dynamic_filter_counters(con)
        except duckdb.Error as e:
            return ExecutionOutcome(ok=False, error=str(e))
        deltas = {name: after[name] - before[name] for name in after}
        counter_writers[arm].write_row(pair_idx, phase, deltas)
        return ExecutionOutcome(
            ok=True,
            elapsed_s=elapsed,
            result_text="".join(repr(r) + "\n" for r in rows),
            deltas=deltas,
        )

    return execute


def open_ab_connection(source: str, data_source: str):
    """Open the single AB session, retrying once on an rmm out-of-memory at startup (GPU
    contention from a co-resident tenant, not a configuration error)."""
    try:
        return open_connection(source, gpu_execution=True, data_source=data_source)
    except (
        Exception
    ) as e:  # noqa: BLE001 -- rmm OOM surfaces as different exception types
        message = str(e).lower()
        if "out_of_memory" not in message and "rmm" not in message:
            raise
        log(
            "rmm out_of_memory at startup -- likely GPU contention; retrying once in 30s"
        )
        time.sleep(30)
        return open_connection(source, gpu_execution=True, data_source=data_source)


def setup_ab_dirs(output_root: str, cell: str) -> Tuple[str, str, str]:
    """AB deliverables layout: <output_root>/phase5/<YYYYMMDD>_<short-commit>/ with one
    subdirectory per cell. Returns (phase5_dir, cell_dir, log_dir)."""
    commit, _ = get_git_info()
    short = (commit or "unknown")[:8]
    phase5_dir = os.path.join(output_root, "phase5", f"{datetime.now():%Y%m%d}_{short}")
    cell_dir = os.path.join(phase5_dir, cell)
    log_dir = os.path.join(cell_dir, "log_dir")
    os.makedirs(log_dir, exist_ok=True)
    return phase5_dir, cell_dir, log_dir


def append_plan_entry(phase5_dir: str, entry: dict) -> None:
    """plan.json: the frozen schedule this run was taken under, one entry appended per cell run."""
    path = os.path.join(phase5_dir, "plan.json")
    doc = {"runs": []}
    if os.path.isfile(path):
        with open(path) as f:
            doc = json.load(f)
    doc.setdefault("runs", []).append(entry)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load_ab_schedule(path: str) -> dict:
    """Per-query schedule overrides frozen from a pilot: {"queries": {"qN": {"target_resolution":
    r, "max_pairs": n}}}."""
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc.get("queries"), dict):
        raise SystemExit(f"schedule {path}: missing 'queries' object")
    return doc


def build_pilot_summary(
    results: Dict[int, QueryCellResult], budget_minutes: float, min_pairs: int
) -> dict:
    """Pilot analysis: per-query noise, derived pair counts, projected wall-clock, and the abort
    criterion -- a query whose required pairs exceed its budget is flagged as
    unresolvable-at-target, never silently under-resolved."""
    per_query = {}
    for qnum, result in sorted(results.items()):
        s_q = log_ratio_stddev(result.paired_ratios)
        pair_time_s = median_of(result.off_s) + median_of(result.on_s)
        n_q = max(
            min_pairs,
            pairs_needed_for_target(result.paired_ratios, result.target_resolution),
        )
        projected_minutes = n_q * pair_time_s / 60.0
        pairs_in_budget = (
            int(budget_minutes * 60.0 / pair_time_s) if pair_time_s > 0 else 0
        )
        reachable = (
            math.expm1(1.96 * s_q / math.sqrt(pairs_in_budget))
            if pairs_in_budget >= 2
            else float("inf")
        )
        per_query[f"q{qnum}"] = {
            "pairs_sampled": len(result.paired_ratios),
            "s_q": s_q,
            "pair_time_s": pair_time_s,
            "target_resolution": result.target_resolution,
            "pairs_needed": n_q,
            "projected_minutes": projected_minutes,
            "budget_minutes": budget_minutes,
            "unresolvable_within_budget": projected_minutes > budget_minutes,
            "resolution_reachable_in_budget": reachable,
        }
    return {
        "per_query": per_query,
        "projected_total_minutes": sum(
            q["projected_minutes"] for q in per_query.values()
        ),
    }


def verify_pinned_cache_hits(cell_dir: str, queries: List[int]) -> Dict[str, str]:
    """Pinned-cache-hit verification per cell: every per-query log must show
    'using cached_split_provider' and never 'not all the columns are pinned' -- a silent
    fall-through would contaminate the pinned cell with the from-disk path and voids the query.
    """
    verdicts = {}
    for qnum in queries:
        log_path = os.path.join(cell_dir, f"q{qnum}", "sirius.log")
        if not os.path.isfile(log_path):
            verdicts[f"q{qnum}"] = "missing-log"
            continue
        with open(log_path, errors="replace") as f:
            text = f.read()
        if "not all the columns are pinned" in text:
            verdicts[f"q{qnum}"] = "fall-through"
        elif "using cached_split_provider" in text:
            verdicts[f"q{qnum}"] = "ok"
        else:
            verdicts[f"q{qnum}"] = "no-cache-hit"
    return verdicts


def run_ab(args, source: str, queries: List[int]) -> None:
    """Run one AB cell end to end: pre-flight, one connection, per-query pin -> warmup -> sampled
    pairs -> unpin, then per-query outputs, log split, pin verification, and the cell report.
    """
    output_root = args.output or DEFAULT_OUTPUT_ROOT
    phase5_dir, cell_dir, log_dir = setup_ab_dirs(output_root, args.cell)
    os.environ["SIRIUS_LOG_DIR"] = log_dir

    config_path = (args.config or "").strip()
    if config_path and os.path.isfile(config_path):
        shutil.copy2(config_path, os.path.join(cell_dir, "config.yml"))

    preflight = run_preflight(
        source, args.data_source, require_drop_caches=args.drop_caches_per_pair
    )
    extra_cell_nonzero: Dict[str, List[str]] = {}
    for item in args.require_cell_nonzero:
        qkey, sep, counter = item.partition(":")
        if not sep or not qkey or not counter:
            raise SystemExit(f"--require-cell-nonzero '{item}': expected QUERY:COUNTER")
        extra_cell_nonzero.setdefault(qkey, []).append(counter)
    expectations = load_arming_expectations(
        args.arming_expectations, queries, extra_cell_nonzero
    )
    schedule = load_ab_schedule(args.schedule) if args.schedule else {"queries": {}}

    pilot = args.pilot_pairs > 0
    commit, branch = get_git_info()
    append_plan_entry(
        phase5_dir,
        {
            "date": datetime.now().isoformat(timespec="seconds"),
            "commit": commit,
            "cell": args.cell,
            "mode": "pilot" if pilot else "full",
            "ab_option": args.ab_option,
            "queries": [f"q{q}" for q in queries],
            "target_resolution_limit": args.target_resolution_limit,
            "target_resolution_other": args.target_resolution_other,
            "min_pairs": args.min_pairs,
            "max_pairs": args.max_pairs,
            "warmup_pairs": args.warmup_pairs,
            "pilot_pairs": args.pilot_pairs,
            "pair_budget_minutes": args.pair_budget_minutes,
            "drop_caches_per_pair": args.drop_caches_per_pair,
            "arming_expectations": {
                "path": args.arming_expectations,
                "sha256": file_sha256(args.arming_expectations),
                "require_cell_nonzero": list(args.require_cell_nonzero),
            },
            "schedule": (
                {"path": args.schedule, "sha256": file_sha256(args.schedule)}
                if args.schedule
                else None
            ),
            "config": {"path": config_path or None, "sha256": file_sha256(config_path)},
            "extension_sha256": preflight["extension_sha256"],
        },
    )
    metadata = {
        "commit": commit,
        "branch_name": branch,
        "date": datetime.now().isoformat(timespec="seconds"),
        "mode": "ab",
        "cell": args.cell,
        "ab_option": args.ab_option,
        "engine": args.engine,
        "data_source": args.data_source,
        "queries": [f"q{q}" for q in queries],
        "pin": args.pin,
        "preflight": preflight,  # includes MemTotal/MemAvailable from /proc/meminfo
    }
    with open(os.path.join(cell_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    con = open_ab_connection(source, args.data_source)
    results: Dict[int, QueryCellResult] = {}
    aborted: Optional[str] = None
    pin_enabled = args.pin != "none"
    try:
        # Field-name validation needs the live counter set, i.e. an open connection.
        validate_arming_field_names(
            expectations, list(read_dynamic_filter_counters(con).keys())
        )
        for qnum in queries:
            qkey = f"q{qnum}"
            qdir = os.path.join(cell_dir, qkey)
            os.makedirs(qdir, exist_ok=True)
            query_schedule = schedule["queries"].get(qkey, {})
            target = query_schedule.get(
                "target_resolution",
                (
                    args.target_resolution_limit
                    if qnum in LIMIT_QUERIES
                    else args.target_resolution_other
                ),
            )
            plan = AbQueryPlan(
                target_resolution=target,
                min_pairs=args.min_pairs,
                max_pairs=query_schedule.get("max_pairs", args.max_pairs),
                warmup_pairs=args.warmup_pairs,
                pilot_pairs=args.pilot_pairs,
                park_threshold_s=args.park_threshold_minutes * 60.0,
            )
            log(
                f"--- {qkey} (ab, target +/-{plan.target_resolution:.3f}, min {plan.min_pairs}, "
                f"max {plan.max_pairs}, warmup {plan.warmup_pairs}"
                + (f", pilot {plan.pilot_pairs}" if pilot else "")
                + ") ---"
            )

            pairs_file = open(os.path.join(qdir, "pairs.csv"), "w", newline="")
            pairs_writer = csv.writer(pairs_file)
            pairs_writer.writerow(PairRecord.CSV_HEADER)
            counter_writers = {
                arm: CounterCsvWriter(os.path.join(qdir, f"counters_{arm}.csv"))
                for arm in ("off", "on")
            }

            def pair_log(record: PairRecord) -> None:
                pairs_writer.writerow(record.csv_row())
                pairs_file.flush()
                if record.phase == "sample":
                    log(
                        f"  [{qkey}] pair {record.pair_idx} lead={record.lead_arm} "
                        f"off={record.off_s:.4f}s on={record.on_s:.4f}s "
                        f"ratio={record.ratio:.4f} "
                        + (
                            "kept"
                            if record.kept
                            else f"DISCARDED ({record.discard_reason})"
                        )
                    )

            if pin_enabled:
                log(f"  Pinning tables for {qkey}")
                _execute_multi(con, emit_pin(qnum, source, args.data_source))
            try:
                executor = make_ab_executor(
                    con, qnum, args.cell, args.ab_option, counter_writers
                )
                result = run_ab_query_cell(
                    qkey,
                    executor,
                    plan,
                    expectations["queries"][qkey],
                    drop_caches_fn=drop_os_cache if args.drop_caches_per_pair else None,
                    pair_log=pair_log,
                )
            finally:
                if pin_enabled:
                    log(f"  Unpinning tables for {qkey}")
                    _execute_multi(con, emit_unpin(qnum))
                pairs_file.close()
                for writer in counter_writers.values():
                    writer.close()

            results[qnum] = result
            for arm in ("off", "on"):
                if arm in result.result_text:
                    with open(os.path.join(qdir, f"result_{arm}.txt"), "w") as f:
                        f.write(result.result_text[arm])
            with open(os.path.join(qdir, "cell_summary.json"), "w") as f:
                json.dump(result.summary(), f, indent=2)
            summary = result.summary()
            log(
                f"  [{qkey}] geomean {summary['geomean_ratio']:.4f} "
                f"CI [{summary['ci_low']:.4f}, {summary['ci_high']:.4f}] over "
                f"{summary['pairs_kept']} pairs -- "
                + ("RESOLVED" if summary["resolved"] else "NOT RESOLVED")
            )
    except (AbCellAbort, AbRunAbort) as e:
        aborted = str(e)
        log(f"ABORT: {aborted}")
    finally:
        log("Closing connection")
        con.close()

    split_sirius_log(log_dir, cell_dir, queries, iterations=None, engine_subdir="")

    pin_verification = (
        verify_pinned_cache_hits(cell_dir, list(results.keys()))
        if pin_enabled
        else None
    )
    if pin_verification:
        for qkey, verdict in pin_verification.items():
            if verdict != "ok":
                log(
                    f"WARNING: pinned-cache verification for {qkey}: {verdict} -- cell voided"
                )

    report = {
        "cell": args.cell,
        "mode": "pilot" if pilot else "full",
        "aborted": aborted,
        "per_query": {f"q{q}": r.summary() for q, r in sorted(results.items())},
        "pin_verification": pin_verification,
    }
    suite = suite_geomean_interval(
        {f"q{q}": r.paired_ratios for q, r in results.items()}
    )
    if suite is not None:
        report["suite_geomean"] = {
            "ci_low": suite[0],
            "point": suite[1],
            "ci_high": suite[2],
        }
    with open(os.path.join(cell_dir, "cell_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"Cell report written to {os.path.join(cell_dir, 'cell_report.json')}")

    if pilot and results:
        pilot_dir = os.path.join(phase5_dir, "pilot", args.cell)
        os.makedirs(pilot_dir, exist_ok=True)
        pilot_summary = build_pilot_summary(
            results, args.pair_budget_minutes, args.min_pairs
        )
        with open(os.path.join(pilot_dir, "pilot_summary.json"), "w") as f:
            json.dump(pilot_summary, f, indent=2)
        proposed = {
            "queries": {
                qkey: {
                    "target_resolution": entry["target_resolution"],
                    # Slack over the pilot's point estimate: without it, one bracket discard in
                    # the full run forces NOT RESOLVED and a rerun of the whole cell.
                    "max_pairs": entry["pairs_needed"]
                    + max(2, entry["pairs_needed"] // 2),
                }
                for qkey, entry in pilot_summary["per_query"].items()
            }
        }
        with open(os.path.join(pilot_dir, "schedule.json"), "w") as f:
            json.dump(proposed, f, indent=2)
        log(f"Pilot summary and proposed schedule written under {pilot_dir}")
        for qkey, entry in pilot_summary["per_query"].items():
            if entry["unresolvable_within_budget"]:
                log(
                    f"PILOT FLAG: {qkey} needs ~{entry['pairs_needed']} pairs "
                    f"(~{entry['projected_minutes']:.0f} min > budget "
                    f"{entry['budget_minutes']:.0f} min); reachable resolution in budget: "
                    f"+/-{entry['resolution_reachable_in_budget']:.4f}"
                )

    if aborted:
        raise SystemExit(f"AB run aborted: {aborted}")
    if pin_verification and any(v != "ok" for v in pin_verification.values()):
        # Artifacts above are already on disk for triage; the exit must still be fatal -- an
        # unattended wrapper would otherwise read a contaminated pinned cell as a clean run.
        raise SystemExit(
            "pinned-cache verification failed: "
            + ", ".join(
                f"{k}={v}" for k, v in sorted(pin_verification.items()) if v != "ok"
            )
        )


def _build_nsys_temp_sql(qnum, source, iterations, pin, qdir, data_source="parquet"):
    """Write the DuckDB SQL script for one nsys-profiled query.

    Produces a timings.csv with rows (views, iter_1, iter_2, ...). The
    cudaProfilerApi capture range brackets iterations 1..N, so the cold run
    (iter_1) is profiled along with every hot iteration; only the one-time
    process-startup GPU-pool init is left outside the captured window.
    """
    sql_path = os.path.join(qdir, "nsys.sql")
    timing_path = os.path.join(qdir, "timings.csv")

    # NOTE: the DuckDB CLI (build/release/duckdb) statically links the Sirius
    # extension, so gpu_execution is already registered at startup. An explicit
    # `LOAD '<ext>'` here throws "Table Function gpu_execution already exists".
    # (Only the non-nsys path, which uses the vanilla Python duckdb module,
    # needs LOAD.) The scaffolding (temp table, views, INSERTs, COPY) runs on
    # plain DuckDB; gpu_execution is toggled on only around the query iterations
    # so the LAG() window-function COPY is never routed through Sirius.
    parts = [
        "CREATE TEMP TABLE _timings (seq INTEGER, step VARCHAR, ts TIMESTAMP);",
        "INSERT INTO _timings VALUES (0, 'start', current_timestamp);",
    ]
    # parquet registers views over the directory; duckdb opens the .duckdb file
    # directly (its native tables are already present), so no view scaffolding is
    # needed. The 'views' timing marker is kept either way so timings.csv parsing
    # (which skips one 'views' row) stays uniform across sources.
    if data_source != "duckdb":
        parts.append(_build_views_sql(source).rstrip("\n"))
    parts.append("INSERT INTO _timings VALUES (1, 'views', current_timestamp);")

    if pin != "none":
        parts.append(emit_pin(qnum, source, data_source))

    parts.append("SET gpu_execution = true;")
    # Open the nsys capture range BEFORE the first (cold) iteration so the cold
    # run is profiled too. The one-time GPU memory-pool init happens at process
    # startup (the statically-linked extension inits before any SQL runs), so it
    # is still outside this range — only query execution (cold + every hot
    # iteration) is captured.
    parts.append("CALL profiler_start();")
    query_sql = QUERIES[f"q{qnum}"].rstrip().rstrip(";") + ";"
    for i in range(1, iterations + 1):
        parts.append(query_sql)
        parts.append(
            f"INSERT INTO _timings VALUES ({i + 1}, 'iter_{i}', current_timestamp);"
        )

    parts.append("CALL profiler_stop();")
    parts.append("SET gpu_execution = false;")

    if pin != "none":
        parts.append(emit_unpin(qnum))

    parts.append(
        f"""COPY (
    SELECT step, runtime_s FROM (
        SELECT
            seq,
            step,
            extract(epoch FROM (ts - LAG(ts) OVER (ORDER BY seq))) AS runtime_s
        FROM _timings
    )
    WHERE seq > 0
    ORDER BY seq
) TO '{timing_path}' (FORMAT CSV, HEADER);"""
    )

    with open(sql_path, "w") as f:
        f.write("\n".join(parts))
        f.write("\n")
    return sql_path


def run_nsys_profile(
    queries,
    source,
    iterations,
    writer,
    *,
    benchmark_dir,
    pin,
    config_path,
    query_timeout,
    data_source="parquet",
):
    """Profile each query with NVIDIA Nsight Systems: one DuckDB subprocess per query.

    Per query, builds a temp SQL file that registers views, optionally pins
    tables, then runs all N iterations (cold + hot) inside a single
    profiler_start / profiler_stop range so the cold run is profiled too, and
    emits a timings.csv via DuckDB's COPY. The subprocess is wrapped by
    `nsys profile --capture-range=cudaProfilerApi`, producing one
    .nsys-rep + .sqlite per query under <bench>/sirius/q<N>/.

    Iteration runtimes from timings.csv are written into csv/runtimes.csv with
    engine="sirius" and iteration=0..N-1.
    """
    if not os.path.isfile(DUCKDB_BIN):
        raise SystemExit(
            f"DuckDB binary not found at {DUCKDB_BIN}. "
            "Build with `pixi run -e clang make release` first."
        )
    if not shutil.which("nsys"):
        raise SystemExit("nsys (NVIDIA Nsight Systems) not found in PATH.")

    log(
        "Mode 'nsys-profile': one nsys-wrapped DuckDB subprocess per query "
        f"(iterations={iterations}, query_timeout={query_timeout}s)"
    )
    pin_enabled = pin != "none"

    for qnum in queries:
        qdir = _query_dir(benchmark_dir, "sirius", qnum)
        sub_log_dir = os.path.join(qdir, "log_dir")
        os.makedirs(sub_log_dir, exist_ok=True)

        sql_path = _build_nsys_temp_sql(
            qnum, source, iterations, pin, qdir, data_source
        )
        nsys_output = os.path.join(qdir, "nsys")
        stdout_path = os.path.join(qdir, "nsys_stdout.txt")

        nsys_cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cudabacktrace=none",
            # profiler_start/stop (in the temp SQL) bracket the cold + hot
            # iterations; capture only that range so the process-startup
            # GPU-pool init stays out of the trace.
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
        ]
        # For duckdb source, open the .duckdb file as the CLI's default database
        # (its native TPC-H tables become queryable by name) — mirroring
        # open_connection / run_tpch_duckdb.sh. For parquet, the in-script
        # CREATE VIEW read_parquet statements supply the tables, so no DB arg.
        duckdb_invocation = [DUCKDB_BIN]
        if data_source == "duckdb":
            duckdb_invocation.append(source)
        duckdb_invocation += [
            # -unsigned mirrors the Python runner's allow_unsigned_extensions
            # config (open_connection): without it, the DuckDB CLI rejects
            # locally-built (unsigned) Sirius extensions.
            "-unsigned",
            "-f",
            sql_path,
        ]
        nsys_cmd.extend(
            [
                "--output",
                nsys_output,
                "--force-overwrite=true",
                "--stats=false",
                "--export=sqlite",
                *duckdb_invocation,
            ]
        )

        env = os.environ.copy()
        if config_path:
            env["SIRIUS_CONFIG_FILE"] = config_path
        if pin_enabled:
            env["SIRIUS_PIN_TIER"] = pin
        env["SIRIUS_LOG_DIR"] = sub_log_dir

        log(f"--- q{qnum} (nsys, iterations={iterations}) ---")
        log(f"  output: {nsys_output}.nsys-rep / .sqlite")
        start = time.perf_counter()
        try:
            with open(stdout_path, "w") as out:
                proc = subprocess.run(
                    nsys_cmd,
                    timeout=query_timeout + 10,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - start
            log(
                f"  q{qnum} TIMED OUT after {wall:.1f}s "
                f"(timeout={query_timeout + 10}s)"
            )
            for it in range(iterations):
                _record(writer, "sirius", qnum, it, float("nan"))
            continue
        wall = time.perf_counter() - start
        log(f"  q{qnum} subprocess returned in {wall:.2f}s (exit={proc.returncode})")

        if proc.returncode != 0:
            log(f"  q{qnum} FAILED — last lines of {stdout_path}:")
            try:
                with open(stdout_path) as f:
                    tail = f.readlines()[-5:]
                for line in tail:
                    log(f"    > {line.rstrip()}")
            except OSError:
                pass
            for it in range(iterations):
                _record(writer, "sirius", qnum, it, float("nan"))
            continue

        # Parse the DuckDB-emitted timings.csv (rows: views, iter_1, iter_2, ...).
        # The 'views' row is skipped; subsequent rows map to iter=0,1,...
        timing_path = os.path.join(qdir, "timings.csv")
        if not os.path.isfile(timing_path):
            log(f"  q{qnum} WARNING: no timings.csv produced; recording NaN")
            for it in range(iterations):
                _record(writer, "sirius", qnum, it, float("nan"))
            continue
        with open(timing_path) as f:
            reader = csv.reader(f)
            next(reader, None)  # header: step,runtime_s
            next(reader, None)  # 'views' row
            it = 0
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    rt = float(row[1])
                except ValueError:
                    rt = float("nan")
                _record(writer, "sirius", qnum, it, rt)
                it += 1
            while it < iterations:
                _record(writer, "sirius", qnum, it, float("nan"))
                it += 1


def split_sirius_log(
    log_dir, benchmark_dir, queries, iterations, engine_subdir="sirius"
):
    """Split the combined Sirius spdlog into one log file per query.

    A query's segment runs from its `QueryBegin: SQL: <sql>` marker to the next such
    marker. Benchmarked-query begins are identified by matching the logged
    (whitespace-normalized) SQL against the known QUERIES text, so interleaved control
    statements (`SET gpu_execution`, `CALL pin_table`/`unpin_table`, `CREATE VIEW`,
    `LOAD`) are ignored and segments are grouped by query content. This is robust
    across data sources (parquet/duckdb), pinning on/off, and every iteration mode --
    it keys on query text, not on statement counts or run ordering.

    `iterations=None` skips the expected-count check (AB mode's per-query execution counts are an
    outcome of the schedule, not a setting); `engine_subdir=""` writes q<N>/sirius.log directly
    under `benchmark_dir` (the AB cell layout) instead of under an engine subdirectory.
    """
    log_files = sorted(glob.glob(os.path.join(log_dir, "sirius*.log")))
    if not log_files:
        log("No sirius*.log found; skipping per-query log split")
        return
    log_path = log_files[0]
    log(f"Splitting {log_path} per query")
    with open(log_path, errors="replace") as f:
        lines = f.readlines()

    def _norm(sql):
        # Mirror SiriusContext::QueryBegin whitespace collapsing, drop a trailing ';',
        # and lowercase so the match is exact but tolerant of formatting differences.
        return " ".join(sql.split()).rstrip(";").strip().lower()

    known = {_norm(QUERIES[f"q{q}"]): q for q in queries}

    begin_marker = "QueryBegin: "
    sql_marker = " SQL: "
    begins = []  # (qnum, line_index), in log order
    for i, line in enumerate(lines):
        pos = line.find(begin_marker)
        if pos == -1:
            continue
        sql_pos = line.find(sql_marker, pos)
        if sql_pos == -1:
            continue
        qnum = known.get(_norm(line[sql_pos + len(sql_marker) :]))
        if qnum is not None:
            begins.append((qnum, i))

    if not begins:
        log("No matched query begins in the Sirius log; skipping per-query split")
        return

    if iterations is not None:
        expected = len(queries) * iterations
        if len(begins) != expected:
            log(
                f"WARNING: expected {expected} query begins, matched {len(begins)} in "
                f"{log_path} (a query may have errored); writing what matched"
            )

    per_query_spans = {q: [] for q in queries}
    for k, (qnum, start) in enumerate(begins):
        end = begins[k + 1][1] if k + 1 < len(begins) else len(lines)
        per_query_spans[qnum].append((start, end))

    for q, span_list in per_query_spans.items():
        if not span_list:
            continue
        qdir = os.path.join(benchmark_dir, engine_subdir, f"q{q}")
        os.makedirs(qdir, exist_ok=True)
        out_path = os.path.join(qdir, "sirius.log")
        with open(out_path, "w") as out:
            for start, end in span_list:
                out.writelines(lines[start:end])
    log(
        f"Per-query Sirius logs written under {os.path.join(benchmark_dir, engine_subdir)}/"
    )


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
        help="TPC-H input: a parquet directory (--data-source parquet; one .parquet "
        "file or subdir per table) or a single .duckdb file (--data-source duckdb)",
    )
    p.add_argument(
        "--data-source",
        choices=DATA_SOURCE_CHOICES,
        default="parquet",
        help=(
            "Input data source/format: 'parquet' (a directory of TPC-H parquet "
            "files scanned via read_parquet -> GPU_PARQUET_SCAN) or 'duckdb' (a "
            "single .duckdb file whose native TPC-H tables are scanned via the "
            "GPU-native seq_scan). Pinning works for both. NOTE: this is a 2-value "
            "flag, distinct from the legacy benchmark_and_validate.sh --data-source "
            "(which also has a redundant 'duckdb-native' alias). (default: parquet)"
        ),
    )
    p.add_argument(
        "--mode",
        choices=MODES,
        default="grouped",
        help="Iteration ordering: grouped (per-query iterations back-to-back, hot "
        "cache), sequential (round-robin across queries), isolated (renew "
        "connection + drop OS cache per run), nsys-profile (one nsys-wrapped "
        "DuckDB subprocess per query; --engine gpu only)",
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
            "Name for the benchmark output subdirectory under --output. "
            "Overrides the default 'tpch_<ts>_<mode>_<engine>_iter<N>'. "
            "Re-runs with the same name will overwrite per-iteration outputs."
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
    p.add_argument(
        "--query-timeout",
        type=int,
        default=90,
        help=(
            "Per-query subprocess timeout in seconds for `--mode nsys-profile` "
            "(default: 90). Ignored in the other modes."
        ),
    )
    ab = p.add_argument_group(
        "ab mode",
        "Phase-5 A/B measurement (`--mode ab`): interleaved flag-off/flag-on pairs per query in "
        "one session, toggled per arm via `SET <--ab-option>`, with per-pair nvidia-smi "
        "bracketing, arming assertions from counter deltas, cross-arm result identity, and a "
        "resolution-driven pair count. Requires --engine gpu, --cell, and --arming-expectations.",
    )
    ab.add_argument(
        "--cell",
        choices=AB_CELL_CHOICES,
        default=None,
        help="Which Phase-5 cell this run is: names the output subdirectory and the Quent labels",
    )
    ab.add_argument(
        "--ab-option",
        type=str,
        default="enable_top_n_dynamic_filter",
        help="Boolean Sirius setting toggled per arm (default: enable_top_n_dynamic_filter)",
    )
    ab.add_argument(
        "--arming-expectations",
        type=str,
        default=None,
        help=(
            "Frozen arming-expectations JSON (calibrate with phase5_calibrate_arming.py): per "
            "query, per arm, the counter-delta predicates every execution must satisfy. A "
            "violation aborts the cell -- a mis-armed cell measures a feature that was not on."
        ),
    )
    ab.add_argument(
        "--require-cell-nonzero",
        action="append",
        default=[],
        metavar="QUERY:COUNTER",
        help=(
            "Additional per-query on-arm cell_nonzero arming requirement merged into the frozen "
            "expectations (repeatable). Use for predicates the calibration scale cannot "
            "observe, e.g. q18:top_n_group_witness_set_full at SF1000 -- at SF1 q18's "
            "qualifying groups sit below the LIMIT and the witness set never fills."
        ),
    )
    ab.add_argument(
        "--target-resolution-limit",
        type=float,
        default=0.02,
        help="CI half-width target for the LIMIT queries (Q2/Q3/Q10/Q18/Q21) (default: 0.02)",
    )
    ab.add_argument(
        "--target-resolution-other",
        type=float,
        default=0.03,
        help="CI half-width target for the non-LIMIT queries (default: 0.03)",
    )
    ab.add_argument(
        "--min-pairs",
        type=int,
        default=21,
        help="Pairs sampled before the interval is first consulted (default: 21)",
    )
    ab.add_argument(
        "--max-pairs",
        type=int,
        default=400,
        help="Attempted-pair cap; reaching it means the query did not resolve (default: 400)",
    )
    ab.add_argument(
        "--warmup-pairs",
        type=int,
        default=3,
        help="Discarded pairs run before sampling (default: 3)",
    )
    ab.add_argument(
        "--pilot-pairs",
        type=int,
        default=0,
        help=(
            "Run a pilot: stop each query after exactly this many kept pairs and emit "
            "pilot/<cell>/pilot_summary.json plus a proposed schedule.json. 0 = full run with "
            "the resolution-driven stop (default: 0)"
        ),
    )
    ab.add_argument(
        "--schedule",
        type=str,
        default=None,
        help=(
            "Per-query schedule JSON frozen from a pilot "
            "({'queries': {'qN': {'target_resolution': r, 'max_pairs': n}}}); overrides the "
            "class targets and the global --max-pairs per query"
        ),
    )
    ab.add_argument(
        "--pair-budget-minutes",
        type=float,
        default=30.0,
        help="Pilot abort criterion: per-query wall-clock budget in minutes (default: 30)",
    )
    ab.add_argument(
        "--drop-caches-per-pair",
        action="store_true",
        help=(
            "Drop the OS page cache before every pair (the cold_spotcheck cell); requires "
            "passwordless sudo, verified by the pre-flight"
        ),
    )
    ab.add_argument(
        "--park-threshold-minutes",
        type=float,
        default=10.0,
        help=(
            "Park (poll until quiet) once a foreign GPU tenant has been continuously present "
            "this long (default: 10)"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    source = args.input
    if args.data_source == "duckdb":
        if not os.path.isfile(source):
            raise SystemExit(
                f"--data-source duckdb requires --input to be a .duckdb file; "
                f"got {source!r}"
            )
    elif not os.path.isdir(source):
        raise SystemExit(
            f"--data-source parquet requires --input to be a parquet directory; "
            f"got {source!r}"
        )
    queries = parse_query_spec(args.queries)
    engine_modes = resolve_engine_modes(args.engine)
    output_root = args.output or DEFAULT_OUTPUT_ROOT

    if args.pin != "none" and args.engine == "cpu":
        raise SystemExit("--pin is Sirius-only; cannot be combined with --engine cpu")

    if args.validation and args.engine != "both":
        raise SystemExit(
            "--validation requires --engine both (needs both result sets to compare)"
        )

    nsys_profile = args.mode == "nsys-profile"
    if nsys_profile:
        if args.engine != "gpu":
            raise SystemExit("--mode nsys-profile requires --engine gpu")
        if args.validation:
            raise SystemExit("--mode nsys-profile is incompatible with --validation")
        if args.duckdb_profiling:
            raise SystemExit(
                "--mode nsys-profile is incompatible with --duckdb-profiling"
            )

    if args.mode == "ab":
        if args.engine != "gpu":
            raise SystemExit(
                "--mode ab requires --engine gpu (both arms run on Sirius; the flag is the only "
                "difference)"
            )
        if args.validation:
            raise SystemExit(
                "--mode ab is incompatible with --validation (cross-arm identity is checked "
                "every pair instead)"
            )
        if not args.cell:
            raise SystemExit("--mode ab requires --cell")
        if not args.arming_expectations:
            raise SystemExit(
                "--mode ab requires --arming-expectations (arming assertions are non-negotiable)"
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

    if args.mode == "ab":
        run_ab(args, source, queries)
        return

    benchmark_dir, runtime_csv, log_dir = setup_benchmark_dir(
        output_root,
        args.mode,
        args.iterations,
        args.engine,
        queries,
        config_path,
        args.pin,
        name=args.name,
        nsys_profile=nsys_profile,
        data_source=args.data_source,
    )
    os.environ["SIRIUS_LOG_DIR"] = log_dir

    log(f"Source:        {source}")
    log(f"Data source:   {args.data_source}")
    log(f"Mode:          {args.mode}")
    log(f"Iterations:    {args.iterations}")
    log(f"Engine:        {args.engine}")
    log(f"Queries:       {queries}")
    log(f"Config:        {config_path or '(default)'}")
    log(f"Pin:           {args.pin}")
    log(f"DuckDB profiling: {args.duckdb_profiling}")
    log(f"nsys-profile:  {nsys_profile}")
    log(f"Benchmark dir: {benchmark_dir}")
    log(f"Runtime CSV:   {runtime_csv}")
    log(f"Log dir:       {log_dir}")

    drop_os_cache()
    with open(runtime_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["engine", "query", "iteration", "runtime_s"])
        f.flush()
        if nsys_profile:
            run_nsys_profile(
                queries,
                source,
                args.iterations,
                writer,
                benchmark_dir=benchmark_dir,
                pin=args.pin,
                config_path=config_path,
                query_timeout=args.query_timeout,
                data_source=args.data_source,
            )
        else:
            RUNNERS[args.mode](
                source,
                queries,
                engine_modes,
                args.iterations,
                writer,
                benchmark_dir=benchmark_dir,
                pin=args.pin,
                data_source=args.data_source,
                duckdb_profiling=args.duckdb_profiling,
            )

    log("Benchmark run complete")

    # split_sirius_log post-processes the combined daily-sink log produced by
    # the long-running Python connection. In nsys-profile mode each query
    # runs in its own subprocess with its own SIRIUS_LOG_DIR, so the per-query
    # logs are already isolated under <bench>/sirius/q<N>/log_dir/.
    if not nsys_profile and any(use_gpu for _, use_gpu in engine_modes):
        split_sirius_log(log_dir, benchmark_dir, queries, args.iterations)

    if args.validation:
        log("Starting validation")
        validate(benchmark_dir, queries)
        log("Validation complete")


if __name__ == "__main__":
    main()
