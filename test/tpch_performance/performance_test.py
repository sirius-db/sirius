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


# `--execution` names a cache state to measure. Each profile fixes what Sirius
# caches, what retires it, the iteration ordering, and what is flushed between
# runs -- setting those independently is how a "cold" number ends up measured
# over a warm page cache. The OS page cache is dropped once at startup either
# way; the *_between flags are about doing it again between runs.
EXECUTION_PROFILES = {
    # The connection is deliberately NOT renewed: dropping the context would
    # also throw away the GPU context and compiled plans, which is not what
    # this is measuring.
    "cold": {
        "ordering": "sequential",
        "cache_mode": "sirius",
        "eviction": "idle",
        "drop_os_cache_between": True,
        "reset_cache_between": True,
        "summary": (
            "cold: per-query OS cache drop + reset_sirius_cache(), round-robin, "
            "one connection"
        ),
    },
    # Round-robin means a query's second iteration comes after every other query
    # has run, so it finds an LRU cache under real pressure rather than its own
    # leftovers.
    "lukewarm": {
        "ordering": "sequential",
        "cache_mode": "sirius",
        "eviction": "lru",
        "drop_os_cache_between": False,
        "reset_cache_between": False,
        "summary": (
            "lukewarm: OS cache dropped once at start, LRU retention, round-robin"
        ),
    },
    "hot": {
        "ordering": "grouped",
        "cache_mode": "sirius",
        "eviction": "idle",
        "drop_os_cache_between": False,
        "reset_cache_between": False,
        "summary": (
            "hot: OS cache dropped once at start, iterations back-to-back per query"
        ),
    },
}
EXECUTION_CHOICES = tuple(EXECUTION_PROFILES)

# Used when --execution is absent. Inert by design: None cache settings mean
# "override nothing", so the run measures what the user's own YAML asks for.
DEFAULT_PROFILE = {
    "ordering": "grouped",
    "cache_mode": None,
    "eviction": None,
    "drop_os_cache_between": False,
    "reset_cache_between": False,
    "summary": (
        "no execution profile: the Sirius config is used as given, iterations "
        "back-to-back per query"
    ),
}

CACHE_CONFIG_PATH = ("sirius", "executor", "scan_manager", "cache")

# --pin parquet pins undecoded column chunks into the prefetching cache, so it
# needs a cache that keeps them: 'sirius' to have one at all, 'lru' because
# 'idle' drops a chunk the moment nothing is reading it, and a threshold of 1.0
# so the evictor only starts once the pool is genuinely full. Without these the
# pin populates the cache and the evictor empties it again.
PARQUET_PIN_CACHE = {
    "mode": "sirius",
    "eviction": "lru",
    "eviction_threshold_fraction": 1.0,
}

ENGINE_CHOICES = ("gpu", "cpu", "both")
PIN_CHOICES = ("none", "gpu", "host", "parquet")
DATA_SOURCE_CHOICES = ("parquet", "duckdb")
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

S3_URI_PREFIX = "s3://"


def is_s3_source(source):
    """True when --input names an S3 prefix instead of a local directory."""
    return str(source).lower().startswith(S3_URI_PREFIX)


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
    execution,
    iterations,
    engine,
    queries,
    config_path,
    pin,
    name=None,
    nsys_profile=False,
    data_source="parquet",
    duckdb_results_source=None,
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
    timestamp); otherwise the default `tpch_<ts>_<execution>_<engine>_iter<N>` is used.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_name = f"tpch_{ts}_{execution}_{engine}_iter{iterations}"
    if name:
        benchmark_name = f"tpch_{ts}_{name}"
    else:
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
        "execution": execution,
        "iterations": iterations,
        "engine": engine,
        "data_source": data_source,
        "queries": [f"q{q}" for q in queries],
        "pin": pin,
        "pin_compression": PIN_COMPRESSION_PLAN_DIR is not None,
        "compression_plan_dir": PIN_COMPRESSION_PLAN_DIR,
        "nsys_profile": nsys_profile,
        "runtime_file": os.path.relpath(runtime_csv, benchmark_dir),
        "duckdb_results_source": duckdb_results_source,
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

# Set from --pin-compression/--compression-plan-dir in main(); when set, every
# Sirius connection enables Simpatico compression for its pin_table calls.
PIN_COMPRESSION_PLAN_DIR = None


def _load_yaml(path):
    """Parse a YAML file. Imported locally: PyYAML is in the repo-root pixi env,
    not test/tpch_performance's own, so a module-scope import would break every
    other invocation."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "--execution needs PyYAML to derive the effective Sirius config from "
            f"{path!r}. Run this script from the repo root via "
            "`pixi run python test/tpch_performance/performance_test.py ...`."
        ) from exc
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(doc, path):
    import yaml

    with open(path, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)


def _dig(doc, keys):
    """Nested mapping lookup; None if any level is missing or not a dict."""
    node = doc
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _plant(doc, keys, values):
    """Create the nested mapping path if needed and merge `values` into it."""
    node = doc
    for k in keys:
        child = node.get(k)
        if not isinstance(child, dict):
            child = {}
            node[k] = child
        node = child
    node.update(values)
    return node


def check_execution_sanity(execution, overrides, config_path, engine, pin):
    """Validate the run's inputs before anything runs.

    Every check here catches a mistake that would otherwise yield a plausible
    number rather than an error.
    """
    profile = (
        EXECUTION_PROFILES[execution] if execution is not None else DEFAULT_PROFILE
    )
    label = f"--execution {execution}" if execution is not None else f"--pin {pin}"
    problems = []
    warnings = []

    if config_path:
        if not os.path.isfile(config_path):
            problems.append(f"config file not found: {config_path}")
        else:
            try:
                doc = _load_yaml(config_path)
            except SystemExit:
                raise
            except Exception as exc:
                problems.append(f"could not parse config {config_path}: {exc}")
                doc = None
            if doc is not None and not isinstance(doc, dict):
                problems.append(f"config {config_path} is not a YAML mapping")
            elif doc is not None:
                # Say overrides out loud: a run that quietly ignored the file
                # is how two results become incomparable for reasons nobody can
                # reconstruct later.
                existing = _dig(doc, CACHE_CONFIG_PATH) or {}
                for key, wanted in overrides.items():
                    have = existing.get(key)
                    if have is not None and have != wanted:
                        warnings.append(
                            f"config sets cache.{key}={have!r}; "
                            f"{label} overrides it to {wanted!r}"
                        )
    else:
        warnings.append(
            "no Sirius config given (--config / $SIRIUS_CONFIG_FILE); the cache "
            f"settings for {label} will be written into a generated config over "
            "Sirius defaults"
        )

    # Fail now rather than after the first query was measured against a warm cache.
    if not can_drop_os_cache():
        detail = (
            "passwordless sudo for /usr/bin/tee /proc/sys/vm/drop_caches is not "
            "available, so the OS page cache cannot be dropped"
        )
        if profile["drop_os_cache_between"]:
            problems.append(
                f"--execution {execution} requires a cold page cache: {detail}"
            )
        else:
            warnings.append(
                f"{detail}; {label} wanted one drop at startup, so the first "
                "query may read warm"
            )

    if profile["reset_cache_between"] and engine == "cpu":
        warnings.append(
            f"--execution {execution} resets Sirius's cache between runs, which "
            "does nothing for --engine cpu"
        )

    if pin == "parquet" and engine == "cpu":
        problems.append("--pin parquet is Sirius-only; it cannot serve --engine cpu")

    if pin != "none" and pin != "parquet" and execution == "cold":
        warnings.append(
            "--pin keeps table data resident on the GPU, which is not something "
            "--execution cold flushes; the scan is cold but the pinned columns "
            "are not"
        )

    for w in warnings:
        log(f"  WARNING: {w}")
    if problems:
        raise SystemExit(
            "execution sanity check failed:\n  - " + "\n  - ".join(problems)
        )


def cache_overrides_for(execution, pin):
    """The cache settings this run needs, or {} to leave the config alone.

    Both inputs can ask for settings and --pin parquet wins where they disagree:
    a pin the evictor immediately undoes is not a pin, whereas an execution
    profile whose eviction policy shifted still measures something coherent.
    """
    overrides = {}
    if execution is not None:
        profile = EXECUTION_PROFILES[execution]
        overrides["mode"] = profile["cache_mode"]
        overrides["eviction"] = profile["eviction"]
    if pin == "parquet":
        for key, value in PARQUET_PIN_CACHE.items():
            if key in overrides and overrides[key] != value:
                log(
                    f"  WARNING: --execution {execution} wants cache.{key}="
                    f"{overrides[key]!r}, but --pin parquet requires {value!r}; "
                    "using the pin's value"
                )
            overrides[key] = value
    return overrides


def derive_execution_config(overrides, config_path, benchmark_dir):
    """Write the config this run will actually use and return its path.

    The user's file is the base and only the profile's cache settings are
    planted over it, so everything else survives untouched. Written beside the
    results as `effective_config.yml`; the original stays as `config.yml`.
    """
    doc = _load_yaml(config_path) if config_path and os.path.isfile(config_path) else {}
    if not isinstance(doc, dict):
        doc = {}

    _plant(doc, CACHE_CONFIG_PATH, overrides)

    effective_path = os.path.join(benchmark_dir, "effective_config.yml")
    _dump_yaml(doc, effective_path)
    settings = " ".join(f"cache.{k}={v}" for k, v in overrides.items())
    log(f"  {settings} -> {effective_path}")
    return effective_path


def can_drop_os_cache():
    """Whether drop_os_cache() would work, without dropping anything.

    Must name the exact command drop_os_cache runs: the sudoers rule is scoped
    to `/usr/bin/tee /proc/sys/vm/drop_caches`, so probing anything else reports
    "no sudo" on a correctly configured machine.
    """
    proc = subprocess.run(
        ["sudo", "-n", "-l", "/usr/bin/tee", "/proc/sys/vm/drop_caches"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


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
        files = []
        for table in TPCH_TABLES:
            files.extend(_resolve_parquet_files(source, table))

    os.sync()
    evicted = 0
    for path in files:
        try:
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                os.posix_fadvise(f.fileno(), 0, size, os.POSIX_FADV_DONTNEED)
                evicted += 1
        except OSError as e:
            log(f"WARNING: fadvise({path}): {e}")
    log(f"posix_fadvise(DONTNEED) applied to {evicted} file(s)")


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

    For an `s3://` prefix the files are not enumerated here: the view body keeps
    the glob and Sirius's `sirius_httpfs` expands it with ListObjectsV2 at bind
    time. Only the GPU path can serve `s3://` (no CPU fallback exists), which
    main() enforces.
    """
    if is_s3_source(parquet_dir):
        root = str(parquet_dir).rstrip("/")
        return (
            "\n".join(
                f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM "
                f"read_parquet('{root}/{table}/*.parquet');"
                for table in TPCH_TABLES
            )
            + "\n"
        )

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
    config = {"allow_unsigned_extensions": "true"}
    s3_source = data_source != "duckdb" and is_s3_source(source)
    if s3_source:
        # s3:// must resolve through Sirius's own sirius_httpfs, not DuckDB's
        # httpfs (which would autoload on the first s3:// bind and then serve the
        # scan on the CPU with its own credential chain, bypassing Sirius).
        config["autoinstall_known_extensions"] = "false"
        config["autoload_known_extensions"] = "false"

    if data_source == "duckdb":
        log(f"Opening DuckDB database file {source} (read-only)")
        con = duckdb.connect(source, read_only=True, config=config)
    else:
        log(f"Opening DuckDB connection over parquet dir {source}")
        con = duckdb.connect(":memory:", config=config)

    # Sirius must be loaded before the views are registered for an s3:// source:
    # registering sirius_httpfs is what makes the s3:// glob in the view body
    # bind at all. Load it first unconditionally -- for local sources the order
    # is immaterial.
    if gpu_execution:
        log(f"Loading Sirius extension from {EXTENSION_PATH}")
        con.execute(f"LOAD '{EXTENSION_PATH}'")
        log("Sirius extension loaded")

    if data_source != "duckdb":
        log("Registering TPC-H parquet views")
        for stmt in _build_views_sql(source).split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            con.execute(stmt)
        log("All TPC-H views registered")
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


class RuntimeCsv:
    """The runtimes.csv writer, which also totals what it writes.

    After the per-query rows it emits one TOTAL row per (engine, iteration) --
    the number you actually compare between runs. NaN runtimes (a query that
    failed or timed out) are left out rather than poisoning the total to NaN,
    so a total is logged with the count it covers whenever that is short of the
    queries asked for.
    """

    def __init__(self, writer, expected_queries):
        self._writer = writer
        self._expected = expected_queries
        self._totals = {}

    def writerow(self, row):
        self._writer.writerow(row)

    def record(self, name, qnum, it, runtime):
        self.writerow([name, f"q{qnum}", it, f"{runtime:.6f}"])
        log(f"[{name}] q{qnum} iter{it}: {runtime:.4f}s")
        if math.isfinite(runtime):
            total, n = self._totals.get((name, it), (0.0, 0))
            self._totals[(name, it)] = (total + runtime, n + 1)

    def write_totals(self):
        for (name, it), (total, n) in sorted(self._totals.items()):
            self.writerow([name, "TOTAL", it, f"{total:.6f}"])
            short = "" if n == self._expected else f" ({n}/{self._expected} queries)"
            log(f"[{name}] TOTAL iter{it}: {total:.4f}s{short}")


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
    writer.record(name, qnum, it, elapsed)
    _write_result(benchmark_dir, name, qnum, rows)


def _flush_between_runs(con, profile, use_gpu):
    """Flush whatever this profile wants gone before a query run, including the
    first -- otherwise the first run would be the odd one out."""
    if profile["drop_os_cache_between"]:
        log("  Dropping OS page cache")
        drop_os_cache()
    if profile["reset_cache_between"] and use_gpu:
        # Sirius's own prefetching cache survives an OS cache drop -- it holds
        # its chunks in pinned host memory, not the page cache -- so a cold run
        # has to ask the engine to let go of them too.
        log("  Resetting Sirius prefetching cache")
        con.execute("CALL reset_sirius_cache();").fetchall()


def run_grouped(
    source,
    queries,
    engine_modes,
    iterations,
    writer,
    *,
    benchmark_dir,
    pin,
    profile,
    data_source="parquet",
    duckdb_profiling=False,
    pin_after_iteration=0,
):
    """Per-query iterations back-to-back; one connection per engine. Pin per query.

    pin_after_iteration leading iterations run unpinned before pinning starts.
    """
    log(
        "Ordering 'grouped': single connection per engine, "
        "iterations back-to-back per query"
    )
    pin_enabled = pin != "none"
    for name, use_gpu in engine_modes:
        con = open_connection(source, gpu_execution=use_gpu, data_source=data_source)
        try:
            for qnum in queries:
                pinned = False
                try:
                    for it in range(iterations):
                        if (
                            pin_enabled
                            and use_gpu
                            and not pinned
                            and it >= pin_after_iteration
                        ):
                            log(f"  Pinning tables for q{qnum} (from iter{it})")
                            _execute_multi(con, emit_pin(qnum, source, data_source))
                            pinned = True
                        log(f"--- q{qnum} iter{it} engine={name} ---")
                        _flush_between_runs(con, profile, use_gpu)
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
                    if pinned:
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
    profile,
    data_source="parquet",
    duckdb_profiling=False,
    pin_after_iteration=0,
):
    """Round-robin iterations; one connection per engine. Single union-pin at session start."""
    log("Ordering 'sequential': single connection per engine, round-robin iterations")
    pin_enabled = pin != "none"
    for name, use_gpu in engine_modes:
        con = open_connection(source, gpu_execution=use_gpu, data_source=data_source)
        try:
            pinned = False
            try:
                for it in range(iterations):
                    if (
                        pin_enabled
                        and use_gpu
                        and not pinned
                        and it >= pin_after_iteration
                    ):
                        log(
                            f"  Union-pinning all referenced TPC-H tables (from iter{it})"
                        )
                        _execute_multi(con, emit_pin_all(source, data_source))
                        pinned = True
                    for qnum in queries:
                        log(f"--- q{qnum} iter{it} engine={name} ---")
                        _flush_between_runs(con, profile, use_gpu)
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
                if pinned:
                    log("  Union-unpinning all TPC-H tables")
                    _execute_multi(con, emit_unpin_all())
        finally:
            log("Closing connection")
            con.close()


# Keyed by an execution profile's "ordering", not by a user-facing choice --
# `--execution` picks the profile and the profile picks the runner.
RUNNERS = {
    "grouped": run_grouped,
    "sequential": run_sequential,
}


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

    pre_sql = os.environ.get("SIRIUS_PRE_SQL", "").strip()
    if pre_sql:
        parts.append(pre_sql.rstrip(";") + ";")

    if pin != "none":
        if PIN_COMPRESSION_PLAN_DIR:
            parts.append("SET pin_table_compression = true;")
            parts.append(
                "SET pin_table_input_compression_plan_dir = "
                f"'{PIN_COMPRESSION_PLAN_DIR}';"
            )
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
        "nsys-profile: one nsys-wrapped DuckDB subprocess per query "
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
                writer.record("sirius", qnum, it, float("nan"))
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
                writer.record("sirius", qnum, it, float("nan"))
            continue

        # Parse the DuckDB-emitted timings.csv (rows: views, iter_1, iter_2, ...).
        # The 'views' row is skipped; subsequent rows map to iter=0,1,...
        timing_path = os.path.join(qdir, "timings.csv")
        if not os.path.isfile(timing_path):
            log(f"  q{qnum} WARNING: no timings.csv produced; recording NaN")
            for it in range(iterations):
                writer.record("sirius", qnum, it, float("nan"))
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
                writer.record("sirius", qnum, it, rt)
                it += 1
            while it < iterations:
                writer.record("sirius", qnum, it, float("nan"))
                it += 1


def print_runtime_summary(runtime_csv):
    """Print a per-engine table of query runtimes with one column per iteration and a Total row."""
    data = {}
    iterations_seen = set()
    with open(runtime_csv, newline="") as f:
        for row in csv.DictReader(f):
            eng = row["engine"]
            qname = row["query"]
            it = int(row["iteration"])
            try:
                rt = float(row["runtime_s"])
            except (ValueError, KeyError):
                rt = float("nan")
            data.setdefault(eng, {}).setdefault(qname, {})[it] = rt
            iterations_seen.add(it)

    if not data:
        return

    n_iters = max(iterations_seen) + 1 if iterations_seen else 0
    iter_labels = [f"iter{i}" for i in range(n_iters)]
    q_w, col_w = 7, 10

    def _fmt(v):
        return f"{'nan':>{col_w}}" if math.isnan(v) else f"{v:{col_w}.4f}"

    def _qnum(name):
        try:
            return int(name.lstrip("q"))
        except ValueError:
            return 0

    print()
    print("=== Runtime Summary (seconds) ===")
    for eng in sorted(data):
        print(f"\n[{eng}]")
        header = f"{'Query':<{q_w}}" + "".join(
            f"  {lbl:>{col_w}}" for lbl in iter_labels
        )
        sep = "-" * len(header)
        print(header)
        print(sep)

        col_totals = [0.0] * n_iters
        for qname in sorted(data[eng], key=_qnum):
            cells = []
            for i in range(n_iters):
                rt = data[eng][qname].get(i, float("nan"))
                cells.append(_fmt(rt))
                if not math.isnan(rt):
                    col_totals[i] += rt
            print(f"{qname:<{q_w}}" + "".join(f"  {c}" for c in cells))

        print(sep)
        total_cells = [f"{t:{col_w}.4f}" for t in col_totals]
        print(f"{'Total':<{q_w}}" + "".join(f"  {c}" for c in total_cells))
    print()


def split_sirius_log(log_dir, benchmark_dir, queries, iterations):
    """Split the combined Sirius spdlog into one log file per query.

    A query's segment runs from its `QueryBegin: SQL: <sql>` marker to the next such
    marker. Benchmarked-query begins are identified by matching the logged
    (whitespace-normalized) SQL against the known QUERIES text, so interleaved control
    statements (`SET gpu_execution`, `CALL pin_table`/`unpin_table`, `CREATE VIEW`,
    `LOAD`) are ignored and segments are grouped by query content. This is robust
    across data sources (parquet/duckdb), pinning on/off, and every execution profile --
    it keys on query text, not on statement counts or run ordering.
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
        qdir = os.path.join(benchmark_dir, "sirius", f"q{q}")
        os.makedirs(qdir, exist_ok=True)
        out_path = os.path.join(qdir, "sirius.log")
        with open(out_path, "w") as out:
            for start, end in span_list:
                out.writelines(lines[start:end])
    log(f"Per-query Sirius logs written under {os.path.join(benchmark_dir, 'sirius')}/")


def validate(sirius_dir, duckdb_dir, queries):
    """Compare saved Sirius vs DuckDB result.txt files under two independent directories.

    Each is a directory of q<N>/result.txt files — sirius_dir and duckdb_dir
    need not share a parent, so a DuckDB reference captured anywhere (e.g. via
    --duckdb-results) can be validated in place, with nothing copied.

    Mirrors compare_results.py: byte-exact match first, then a tolerance-aware
    fallback (abs_tol=VALIDATION_ABS_TOL on Python float values only; strict
    equality on Decimal/int/str/date/etc.). No query re-execution, no DuckDB
    connection — safe to run after the GPU pool from the timed pass is still
    resident.

    Returns dict[qnum, {"status": "success"|"validation"|"error", "detail": str|None}].
    "error" covers missing/unparsable result files (structural failure, no
    comparison was possible); "validation" covers a comparison that ran but
    didn't match (row count or value mismatch); "success" is a match.
    """
    log(f"Validating {sirius_dir} against {duckdb_dir}")
    results = {}
    for qnum in queries:
        qname = f"q{qnum}"
        duck_path = os.path.join(duckdb_dir, qname, "result.txt")
        sir_path = os.path.join(sirius_dir, qname, "result.txt")
        if not os.path.exists(duck_path) or not os.path.exists(sir_path):
            print(f"❌ {qname}: missing result.txt (duckdb or sirius)")
            results[qnum] = {
                "status": "error",
                "detail": "missing result.txt (duckdb or sirius)",
            }
            continue
        with open(duck_path, "rb") as fd, open(sir_path, "rb") as fs:
            if fd.read() == fs.read():
                print(f"✓ {qname}: byte-exact match")
                results[qnum] = {"status": "success", "detail": None}
                continue
        try:
            duck_rows = _load_result_file(duck_path)
            sir_rows = _load_result_file(sir_path)
        except Exception as e:
            print(f"❌ {qname}: parse error - {e}")
            results[qnum] = {"status": "error", "detail": f"parse error - {e}"}
            continue
        if len(duck_rows) != len(sir_rows):
            detail = (
                f"row count mismatch - duckdb={len(duck_rows)} sirius={len(sir_rows)}"
            )
            print(f"❌ {qname}: {detail}")
            results[qnum] = {"status": "validation", "detail": detail}
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
            results[qnum] = {"status": "success", "detail": None}
        else:
            i, d, s = mismatch
            detail = f"row {i} mismatch: duckdb={d} sirius={s}"
            print(f"❌ {qname}: row {i} mismatch")
            print(f"   duckdb: {d}")
            print(f"   sirius: {s}")
            results[qnum] = {"status": "validation", "detail": detail}

    passed = sum(1 for v in results.values() if v["status"] == "success")
    failed = [f"q{q}" for q, v in results.items() if v["status"] != "success"]
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
        "file or subdir per table), an s3:// prefix holding one <table>/ subdir "
        "per table (--engine gpu only), or a single .duckdb file "
        "(--data-source duckdb)",
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
        "--execution",
        choices=EXECUTION_CHOICES,
        default=None,
        help=(
            "Cache state to measure. Each value fixes the Sirius cache mode, the "
            "eviction policy, the iteration ordering and what is flushed between "
            "runs, and OVERRIDES cache.mode/cache.eviction in --config. "
            "OMIT IT and the config is used exactly as given, with iterations "
            "back-to-back per query and nothing flushed between runs. "
            "cold (per-query OS cache drop + reset_sirius_cache(), round-robin; "
            "needs passwordless sudo), "
            "lukewarm (OS cache dropped once at start, LRU retention, round-robin), "
            "hot (OS cache dropped once at start, iterations back-to-back per "
            "query). (default: unset — change nothing)"
        ),
    )
    p.add_argument(
        "--nsys-profile",
        action="store_true",
        help=(
            "Run one nsys-wrapped DuckDB subprocess per query instead of the "
            "in-process runner (--engine gpu only). Has its own execution model, "
            "so --execution does not apply to it."
        ),
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
            "per-query under the hot profile and a single union-pin at "
            "session start under cold/lukewarm. 'parquet' pins the undecoded "
            "column-chunk bytes into the Sirius prefetching cache instead of "
            "materialising on the GPU, so it needs cache.mode='sirius' and "
            "--data-source parquet. (default: none)"
        ),
    )
    p.add_argument(
        "--pin-after-iteration",
        type=int,
        default=0,
        help=(
            "Number of leading iterations per query (grouped/isolated) or "
            "round-robin pass (sequential) to run unpinned before pinning "
            "kicks in for the remainder, e.g. cold+warm unpinned then hot "
            "pinned. Sirius-only; ignored with --pin none. (default: 0, pin "
            "immediately)"
        ),
    )
    p.add_argument(
        "--pin-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compress the pinned tables with Simpatico; needs --pin gpu|host "
            "and per-table plan files (--compression-plan-dir)"
        ),
    )
    p.add_argument(
        "--compression-plan-dir",
        type=str,
        default=None,
        help=(
            "Directory of per-table Simpatico plan files (<table>.<ext>) for "
            "--pin-compression (default: the SF1000 plans under "
            "src/compression/simpatico_codegen/plans)"
        ),
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Name for the benchmark output subdirectory under --output. "
            "Overrides the default 'tpch_<ts>_<execution>_<engine>_iter<N>'. "
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
            "Per-query subprocess timeout in seconds for `--nsys-profile` "
            "(default: 90). Ignored otherwise."
        ),
    )
    p.add_argument(
        "--duckdb-results",
        type=str,
        default=None,
        help=(
            "Reuse previously captured DuckDB reference results instead of "
            "running the duckdb engine. Accepts either a full benchmark "
            "directory (its duckdb/q<N>/result.txt files are used) or a "
            "duckdb/ directory itself; validated in place, nothing is "
            "copied. The requested --engine (must be 'gpu') still runs "
            "normally, and validation against the reused results runs "
            "automatically at the end (writing validation.csv), even "
            "without --validation."
        ),
    )
    return p.parse_args()


def _resolve_duckdb_results_dir(path):
    """Resolve --duckdb-results to a directory of q<N>/result.txt files."""
    candidate = path
    if os.path.isdir(os.path.join(path, "duckdb")):
        candidate = os.path.join(path, "duckdb")
    if not os.path.isdir(candidate):
        raise SystemExit(f"--duckdb-results directory not found: {candidate}")
    count = sum(
        1
        for entry in os.listdir(candidate)
        if entry.startswith("q")
        and os.path.isfile(os.path.join(candidate, entry, "result.txt"))
    )
    if count == 0:
        raise SystemExit(
            f"--duckdb-results: no query results (q*/result.txt) found in {candidate}"
        )
    return candidate


def main():
    args = parse_args()
    source = args.input
    if args.data_source == "duckdb":
        if not os.path.isfile(source):
            raise SystemExit(
                f"--data-source duckdb requires --input to be a .duckdb file; "
                f"got {source!r}"
            )
    elif is_s3_source(source):
        # S3 is GPU-only: DuckDB's CPU read_parquet has no S3 filesystem, and
        # Sirius deliberately refuses to serve s3:// to a CPU plan
        # (src/sirius_context.cpp, throw_if_s3_no_cpu_fallback). So there is no
        # CPU baseline to time against or validate with.
        if args.engine != "gpu":
            raise SystemExit(
                "an s3:// --input requires --engine gpu; S3 has no CPU fallback"
            )
        if args.pin != "none":
            raise SystemExit(
                "--pin is not supported for an s3:// --input; pin_table globs "
                "local files"
            )
    elif not os.path.isdir(source):
        raise SystemExit(
            f"--data-source parquet requires --input to be a parquet directory "
            f"or an s3:// prefix; got {source!r}"
        )
    queries = parse_query_spec(args.queries)
    engine_modes = resolve_engine_modes(args.engine)
    output_root = args.output or DEFAULT_OUTPUT_ROOT

    if args.pin != "none" and args.engine == "cpu":
        raise SystemExit("--pin is Sirius-only; cannot be combined with --engine cpu")

    duckdb_results_dir = None
    if args.duckdb_results:
        if args.engine != "gpu":
            raise SystemExit(
                "--duckdb-results reuses DuckDB results in place of running them; "
                "--engine must be 'gpu' (got: " + args.engine + ")"
            )
        duckdb_results_dir = _resolve_duckdb_results_dir(args.duckdb_results)

    # Simpatico compression happens at pin time, so it only applies to pinned
    # input, and it is a no-op without a plan file naming a TPC-H table.
    if args.pin_compression:
        if args.pin == "none":
            raise SystemExit("--pin-compression needs a pinned tier (--pin gpu|host)")
        plan_dir = args.compression_plan_dir or os.path.join(
            REPO_ROOT, "src/compression/simpatico_codegen/plans/tpch_sf1000"
        )
        plan_dir = os.path.abspath(plan_dir)
        if not os.path.isdir(plan_dir):
            raise SystemExit(f"--compression-plan-dir not found: {plan_dir}")
        plan_stems = {os.path.splitext(f)[0] for f in os.listdir(plan_dir)}
        if not any(t in plan_stems for t in TPCH_TABLES):
            raise SystemExit(
                f"No plan file in {plan_dir} names a TPC-H table; plan files "
                "are <table>.<ext>"
            )
        global PIN_COMPRESSION_PLAN_DIR
        PIN_COMPRESSION_PLAN_DIR = plan_dir

    if args.validation and args.engine != "both" and not duckdb_results_dir:
        raise SystemExit(
            "--validation requires --engine both, or --duckdb-results with --engine gpu "
            "(needs both result sets to compare)"
        )
    do_validate = args.validation or duckdb_results_dir is not None

    nsys_profile = args.nsys_profile
    if nsys_profile:
        if args.engine != "gpu":
            raise SystemExit("--nsys-profile requires --engine gpu")
        if args.validation:
            raise SystemExit("--nsys-profile is incompatible with --validation")
        if args.duckdb_profiling:
            raise SystemExit("--nsys-profile is incompatible with --duckdb-profiling")
        # It drives its own subprocesses with their own cache behaviour, so
        # silently applying an execution profile would be a lie in the metadata.
        if args.execution is not None:
            raise SystemExit(
                "--nsys-profile has its own execution model; --execution does not "
                "apply to it"
            )

    config_path = (args.config or "").strip()

    # No --execution means change nothing: the profile is inert and the config
    # below is left exactly as the user wrote it.
    profile = (
        EXECUTION_PROFILES[args.execution]
        if args.execution is not None
        else DEFAULT_PROFILE
    )
    cache_overrides = (
        {} if nsys_profile else cache_overrides_for(args.execution, args.pin)
    )
    if not nsys_profile:
        log(f"Execution:     {args.execution or '(unset)'} — {profile['summary']}")
        if cache_overrides:
            log("Checking execution sanity")
            check_execution_sanity(
                args.execution, cache_overrides, config_path, args.engine, args.pin
            )

    if args.pin != "none":
        os.environ["SIRIUS_PIN_TIER"] = args.pin

    # nsys drives its own subprocesses, so no execution profile was applied and
    # naming the run after one would misattribute it.
    execution_label = "nsys" if nsys_profile else (args.execution or "default")
    benchmark_dir, runtime_csv, log_dir = setup_benchmark_dir(
        output_root,
        execution_label,
        args.iterations,
        args.engine,
        queries,
        config_path,
        args.pin,
        name=args.name,
        nsys_profile=nsys_profile,
        data_source=args.data_source,
        duckdb_results_source=duckdb_results_dir,
    )
    os.environ["SIRIUS_LOG_DIR"] = log_dir

    # Set before any connection opens: Sirius reads SIRIUS_CONFIG_FILE at LOAD.
    if not cache_overrides:
        if config_path:
            os.environ["SIRIUS_CONFIG_FILE"] = config_path
        else:
            log(
                "SIRIUS_CONFIG_FILE not set and --config not provided — "
                "running with Sirius default configuration."
            )
    else:
        log("Deriving effective Sirius config")
        config_path = derive_execution_config(
            cache_overrides, config_path, benchmark_dir
        )
        os.environ["SIRIUS_CONFIG_FILE"] = config_path

    log(f"Source:        {source}")
    log(f"Data source:   {args.data_source}")
    log(f"Execution:     {execution_label}")
    log(f"Iterations:    {args.iterations}")
    log(f"Engine:        {args.engine}")
    log(f"Queries:       {queries}")
    log(f"Config:        {config_path or '(default)'}")
    log(f"Pin:           {args.pin}")
    if PIN_COMPRESSION_PLAN_DIR:
        log(f"Compression:   simpatico ({PIN_COMPRESSION_PLAN_DIR})")
    log(f"DuckDB profiling: {args.duckdb_profiling}")
    log(f"nsys-profile:  {nsys_profile}")
    log(f"Benchmark dir: {benchmark_dir}")
    log(f"Runtime CSV:   {runtime_csv}")
    log(f"Log dir:       {log_dir}")

    drop_os_cache(source, args.data_source)
    with open(runtime_csv, "w", newline="") as f:
        writer = RuntimeCsv(csv.writer(f), len(queries))
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
            RUNNERS[profile["ordering"]](
                source,
                queries,
                engine_modes,
                args.iterations,
                writer,
                benchmark_dir=benchmark_dir,
                pin=args.pin,
                profile=profile,
                data_source=args.data_source,
                duckdb_profiling=args.duckdb_profiling,
                pin_after_iteration=args.pin_after_iteration,
            )
        writer.write_totals()

    log("Benchmark run complete")

    # split_sirius_log post-processes the combined daily-sink log produced by
    # the long-running Python connection. In nsys-profile mode each query
    # runs in its own subprocess with its own SIRIUS_LOG_DIR, so the per-query
    # logs are already isolated under <bench>/sirius/q<N>/log_dir/.
    if not nsys_profile and any(use_gpu for _, use_gpu in engine_modes):
        split_sirius_log(log_dir, benchmark_dir, queries, args.iterations)

    if do_validate:
        log("Starting validation")
        duckdb_dir_for_validation = duckdb_results_dir or os.path.join(
            benchmark_dir, "duckdb"
        )
        results = validate(
            os.path.join(benchmark_dir, "sirius"), duckdb_dir_for_validation, queries
        )
        validation_csv = os.path.join(benchmark_dir, "validation.csv")
        with open(validation_csv, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["query", "status"])
            for qnum in queries:
                status = results.get(qnum, {}).get("status", "error")
                csv_writer.writerow([f"Q{qnum}", status])
        log(f"Wrote {validation_csv}")
        log("Validation complete")

    print_runtime_summary(runtime_csv)


if __name__ == "__main__":
    main()
