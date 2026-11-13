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

"""A/B benchmark of MERGE_SORT: sort_merge_mode x spill_compression x dataset.

Each run sorts a `sirius_generate(...)` dataset sized as a multiple of a
configured GPU memory limit and ends in the discard sink, in its own process
with its own Sirius config file. See README.md beside this script.
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DEFAULT_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
DEFAULT_SPILL_DIR = "/mnt/disk_2/sirius_spill"
DEFAULT_EXTENSION = "build/release/extension/sirius/sirius.duckdb_extension"
# Where `make` puts the extension when it is built through the sirius-duckdb
# wrapper; tried when --extension is left at its default and that is missing.
WRAPPER_EXTENSION = (
    "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
)

MODES = ("one_shot", "windowed", "auto")
SEED = 42
DEFAULT_ROWS_PER_BATCH = 8_000_000
DEFAULT_SPILL_MIN_BATCH_BYTES = 1 << 20

# Names under --out-dir. The repository's .gitignore ignores `runs/` and
# `*output.txt`; these names are not ignored, so a results directory can be
# committed as the harness wrote it.
PER_RUN_DIR = "per_run"
CHILD_STDOUT_FILE = "child_stdout.txt"
LOG_TAIL_FILE = "sirius_log_tail.txt"
# Written for a run that did not end `ok`: the first warnings and errors of
# its log, which the tail of a long failing run no longer holds.
FIRST_ERRORS_FILE = "sirius_log_first_errors.txt"
FIRST_ERRORS_LINES = 200
# A sizing that misses --scale x limit by more than this fraction is reported.
SIZING_TOLERANCE = 0.1

# Bytes per row by column type, used to size a dataset. VARCHAR adds its mean
# length (see row_width).
TYPE_WIDTH = {"BIGINT": 8, "INTEGER": 4, "DOUBLE": 8, "VARCHAR": 4}

# The dataset matrix. Each column is a `sirius_generate` column spec; `keys`
# is the ORDER BY list. For a VARCHAR column p1 and p2 are both the
# distribution's parameters and the length range, so `few_distinct` strings
# have D = p1 = their length: 18 distinct 18-character strings.
DATASETS = {
    "uniform-1key": {
        "description": "BIGINT uniform key + DOUBLE payload",
        "columns": [
            {"name": "k", "type": "BIGINT", "dist": "uniform", "p1": 0, "p2": 10**15},
            {"name": "v", "type": "DOUBLE", "dist": "uniform", "p1": 0, "p2": 1},
        ],
        "keys": ["k"],
    },
    "zipf-1key": {
        "description": "BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload",
        "columns": [
            {
                "name": "k",
                "type": "BIGINT",
                "dist": "zipf",
                "p1": 1.1,
                "cardinality": 10**9,
            },
            {"name": "v", "type": "DOUBLE", "dist": "uniform", "p1": 0, "p2": 1},
        ],
        "keys": ["k"],
    },
    "normal-multikey": {
        "description": "INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys",
        "columns": [
            {"name": "k1", "type": "INTEGER", "dist": "normal", "p1": 0, "p2": 10**6},
            {"name": "k2", "type": "BIGINT", "dist": "uniform", "p1": 0, "p2": 10**15},
            {"name": "s", "type": "VARCHAR", "dist": "uniform", "p1": 8, "p2": 16},
        ],
        "keys": ["k1", "k2", "s"],
    },
    "few_distinct-strings": {
        "description": "VARCHAR few_distinct key (18 distinct values, length 18)",
        "columns": [
            {
                "name": "s",
                "type": "VARCHAR",
                "dist": "few_distinct",
                "p1": 18,
                "p2": 18,
            },
        ],
        "keys": ["s"],
    },
    "sequential": {
        "description": "BIGINT sequential (already sorted) key + DOUBLE payload",
        "columns": [
            {"name": "k", "type": "BIGINT", "dist": "sequential", "p1": 0, "p2": 1},
            {"name": "v", "type": "DOUBLE", "dist": "uniform", "p1": 0, "p2": 1},
        ],
        "keys": ["k"],
    },
}

# Key order of the "[merge_sort][stats]" log line; keys the log adds later are
# appended to the CSV after these.
MERGE_STATS_KEYS = [
    "windowed_partitions",
    "one_shot_partitions",
    "windowed_tasks",
    "intermediate_tasks",
    "mergers_built",
    "rounds",
    "rows_out",
    "round_rows_min",
    "round_rows_max",
    "refill_rows",
    "refill_bytes",
    "refill_bytes_host",
    "refill_bytes_disk",
    "refill_bytes_compressed",
    "runs_read_compressed",
    "window_growths",
    "compacted_rows",
    "fallback_conversions",
    "shrinks",
]

CSV_COLUMNS = [
    "dataset",
    "mode",
    "compression",
    "repeat",
    "gpu_limit_bytes",
    "num_batches",
    "rows_per_batch",
    "rows",
    "bytes",
    "status",
    "wall_s",
    "load_s",
    "rows_counted",
    "peak_gpu_mib",
    "spill_bytes_host",
    "spill_bytes_disk",
    "spill_batches_host",
    "spill_batches_disk",
    "downgrade_requests",
    "oom_retries",
    "discard_rows",
    "discard_bytes",
    "order_violations",
]

# Statuses a `one_shot` run may end in without the harness calling it a failure.
ONE_SHOT_EXPECTED = {"ok", "oom", "error", "timeout", "crashed"}

_OOM_RE = re.compile(r"out[ _]of[ _]memory|bad_alloc|\boom\b|oom_", re.IGNORECASE)
_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "k": 10**3,
    "kb": 10**3,
    "m": 10**6,
    "mb": 10**6,
    "g": 10**9,
    "gb": 10**9,
    "t": 10**12,
    "tb": 10**12,
    "ki": 1 << 10,
    "kib": 1 << 10,
    "mi": 1 << 20,
    "mib": 1 << 20,
    "gi": 1 << 30,
    "gib": 1 << 30,
    "ti": 1 << 40,
    "tib": 1 << 40,
}


# `log`, `_load_yaml`, `_dump_yaml`, `_dig`, `_plant` and `_sql_literal` below
# are copies of the functions of the same names in
# test/tpch_performance/performance_test.py. The two copies must change
# together.


def log(msg):
    """Timestamped, flushed progress log so hangs are visible in real time."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def _load_yaml(path):
    """Parse a YAML file. Imported locally so the import error names the fix."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "PyYAML is needed to write the per-run Sirius config. Run this script "
            "from the repo root via "
            "`pixi run python test/sort_performance/sort_merge_bench.py ...`."
        ) from exc
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(doc, path=None):
    """Write `doc` to `path`, or return it as text when `path` is None."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "PyYAML is needed to write the per-run Sirius config. Run this script "
            "from the repo root via "
            "`pixi run python test/sort_performance/sort_merge_bench.py ...`."
        ) from exc
    if path is None:
        return yaml.safe_dump(doc, default_flow_style=False, sort_keys=False)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
    return None


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


def _sql_literal(text):
    return "'" + str(text).replace("'", "''") + "'"


def _parse_bytes(value):
    """A config byte count (int, or a string such as '25GB' / '64Mi'); None if unreadable."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*", str(value))
    if not m or m.group(2).lower() not in _BYTE_UNITS:
        return None
    return int(float(m.group(1)) * _BYTE_UNITS[m.group(2).lower()])


def _gib(n):
    return f"{n / (1 << 30):.2f}"


# -----------------------------------------------------------------------------
# Datasets and SQL
# -----------------------------------------------------------------------------


def row_width(dataset):
    """Bytes per row: 8 per BIGINT/DOUBLE, 4 per INTEGER, mean length + 4 per VARCHAR."""
    width = 0
    for col in DATASETS[dataset]["columns"]:
        width += TYPE_WIDTH[col["type"]]
        if col["type"] == "VARCHAR":
            width += (col["p1"] + col["p2"]) // 2
    return width


def size_dataset(dataset, target_bytes, rows_per_batch):
    """num_batches such that num_batches x rows_per_batch x row_width ~ target_bytes."""
    width = row_width(dataset)
    num_batches = max(1, round(target_bytes / (width * rows_per_batch)))
    rows = num_batches * rows_per_batch
    return {
        "num_batches": num_batches,
        "rows_per_batch": rows_per_batch,
        "rows": rows,
        "row_width": width,
        "bytes": rows * width,
    }


def _column_sql(col):
    parts = [f"name: {_sql_literal(col['name'])}"]
    parts.append(f"type: {_sql_literal(col['type'])}")
    parts.append(f"dist: {_sql_literal(col['dist'])}")
    for key in ("p1", "p2", "cardinality", "null_fraction"):
        if key in col:
            parts.append(f"{key}: {col[key]}")
    return "{" + ", ".join(parts) + "}"


def generate_call(dataset, sizing):
    columns = ",\n    ".join(_column_sql(c) for c in DATASETS[dataset]["columns"])
    return (
        "sirius_generate(\n"
        f"  num_batches := {sizing['num_batches']},\n"
        f"  rows_per_batch := {sizing['rows_per_batch']},\n"
        f"  seed := {SEED},\n"
        f"  columns := [\n    {columns}\n  ]\n)"
    )


def build_query(dataset, sizing, validate):
    """The benchmark statement: the full sort ending in the discard sink."""
    keys = ", ".join(DATASETS[dataset]["keys"])
    options = "FORMAT sirius_discard"
    if validate:
        options += ", check_order true"
    return (
        f"COPY (SELECT * FROM {generate_call(dataset, sizing)} ORDER BY {keys})\n"
        f"TO '/dev/null' ({options})"
    )


def session_settings(mode, compression):
    return [
        "SET gpu_execution = true",
        f"SET sort_merge_mode = {_sql_literal(mode)}",
        f"SET spill_compression = {'true' if compression else 'false'}",
    ]


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


def load_base_config(path):
    if not path:
        return {}
    if not os.path.isfile(path):
        raise SystemExit(f"--base-config: file not found: {path}")
    doc = _load_yaml(path)
    if not isinstance(doc, dict):
        raise SystemExit(f"--base-config: {path} is not a YAML mapping")
    space = _dig(doc, ["sirius", "space"])
    if isinstance(space, dict) and any(space.get(t) for t in ("gpu", "host", "disk")):
        raise SystemExit(
            f"--base-config: {path} configures memory through sirius.space; this "
            "harness sets sirius.memory.* and the loader rejects both together"
        )
    return doc


def derive_run_config(base, gpu_limit_bytes, spill_dir, compression, spill_min_bytes):
    """The config one run uses: the base plus the GPU limit, the spill directory
    and, for the compressed arm only, the lowered compression threshold."""
    doc = json.loads(json.dumps(base))  # deep copy of plain YAML data
    gpu = _plant(
        doc, ["sirius", "memory", "gpu"], {"usage_limit_bytes": int(gpu_limit_bytes)}
    )
    # usage_limit_fraction and usage_limit_bytes are mutually exclusive.
    gpu.pop("usage_limit_fraction", None)
    _plant(doc, ["sirius", "memory", "disk"], {"downgrade_root_dirs": spill_dir})
    if compression:
        _plant(
            doc,
            ["sirius", "compression"],
            {"spill_min_batch_bytes": int(spill_min_bytes)},
        )
    return doc


def host_tier_bytes(config):
    """Bytes the host tier keeps before it evicts to disk, per the run's config."""
    host = _dig(config, ["sirius", "memory", "host"]) or {}
    total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    capacity = _parse_bytes(host.get("capacity_bytes"))
    if capacity is None:
        fraction = host.get("capacity_fraction")
        capacity = int(total_ram * (fraction if fraction is not None else 0.9))
    capacity = min(capacity, total_ram)
    stop = host.get("downgrade_stop_fraction")
    return int(capacity * (stop if stop is not None else 0.8))


def disk_bytes_needed(data_bytes, config):
    """Disk the spilled inputs may take: what the host tier does not hold, plus 5%."""
    return int(max(0, data_bytes - host_tier_bytes(config)) * 1.05)


def _existing_ancestor(path):
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


# -----------------------------------------------------------------------------
# Child process: one connection, one COPY
# -----------------------------------------------------------------------------


def _write_json(doc, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)


def run_child(job_path):
    """Run one benchmark statement in this process and write its result JSON."""
    with open(job_path) as f:
        job = json.load(f)
    result = {"status": "error", "error": "", "wall_s": None, "load_s": None}
    try:
        import duckdb

        con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
        t0 = time.perf_counter()
        con.execute(f"LOAD {_sql_literal(job['extension'])}")
        for stmt in job["settings"]:
            con.execute(stmt)
        result["load_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        try:
            rows = con.execute(job["sql"]).fetchall()
        finally:
            result["wall_s"] = time.perf_counter() - t0
        result["rows_counted"] = int(rows[0][0])
        result["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001 - every failure is a recorded outcome
        result["error"] = f"{type(exc).__name__}: {exc}"
    # Written before the connection goes away: a teardown that hangs or crashes
    # must not lose the measurement.
    _write_json(result, job["result_path"])
    return 0 if result["status"] == "ok" else 1


# -----------------------------------------------------------------------------
# Parent: one run
# -----------------------------------------------------------------------------


def _kill_group(proc):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            continue


def run_log_paths(log_dir):
    """The Sirius log files the run wrote, oldest first (the log rotates daily)."""
    if not os.path.isdir(log_dir):
        return []
    names = sorted(n for n in os.listdir(log_dir) if n.startswith("sirius"))
    paths = [os.path.join(log_dir, n) for n in names]
    paths.sort(key=os.path.getmtime)
    return paths


def read_run_log(log_dir, bench_metrics):
    """The text of every Sirius log file the run wrote."""
    return "".join(bench_metrics.LogTail(p).read_new() for p in run_log_paths(log_dir))


def collect_metrics(log_text, bench_metrics):
    """Flatten what bench_metrics reads from the run's log into result columns."""
    out = {}
    stats = bench_metrics.parse_merge_sort_stats(log_text)
    out["merge_sort_operators"] = len(stats)
    out.update(bench_metrics.sum_merge_sort_stats(stats))
    totals = bench_metrics.parse_downgrade_totals(log_text)
    out["spill_bytes_host"] = totals.to_host_bytes
    out["spill_bytes_disk"] = totals.to_disk_bytes
    out["spill_batches_host"] = totals.to_host_batches
    out["spill_batches_disk"] = totals.to_disk_batches
    out["downgrade_requests"] = totals.requests
    out["oom_retries"] = bench_metrics.count_oom_retries(log_text)
    sink = bench_metrics.parse_discard_sink(log_text)
    if sink is not None:
        out["discard_rows"] = sink.get("rows")
        out["discard_bytes"] = sink.get("bytes")
        out["order_violations"] = sink.get("order_violations")
    return out


def classify_failure(error_text, log_text):
    """`oom` when the error (or, failing that, the log's last lines) says so."""
    if _OOM_RE.search(error_text or ""):
        return "oom"
    if not error_text and _OOM_RE.search("\n".join(log_text.splitlines()[-50:])):
        return "oom"
    return "error"


def execute_run(run, args, bench_metrics):
    """Run one matrix cell in a fresh process; returns its result row (a dict)."""
    row = dict(run["row"])
    run_dir = run["run_dir"]
    log_dir = os.path.join(run_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(run["spill_dir"], exist_ok=True)

    config_path = os.path.join(run_dir, "config.yaml")
    _dump_yaml(run["config"], config_path)
    with open(os.path.join(run_dir, "query.sql"), "w") as f:
        f.write(";\n".join(run["settings"] + [run["sql"]]) + ";\n")
    result_path = os.path.join(run_dir, "result.json")
    job_path = os.path.join(run_dir, "job.json")
    _write_json(
        {
            "extension": args.extension,
            "settings": run["settings"],
            "sql": run["sql"],
            "result_path": result_path,
        },
        job_path,
    )

    env = os.environ.copy()
    env["SIRIUS_CONFIG_FILE"] = config_path
    env["SIRIUS_LOG_DIR"] = log_dir
    env["SIRIUS_LOG_BACKEND"] = "spdlog"
    env["SIRIUS_LOG_LEVEL"] = "info"
    # nvidia-smi, which the sampler asks about index args.device, numbers the
    # GPUs by PCI bus. CUDA's default order is fastest first, so without this
    # the run and the sampler could be on different GPUs.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)

    sampler = bench_metrics.GpuMemorySampler(args.device)
    timed_out = False
    started = time.perf_counter()
    try:
        with open(os.path.join(run_dir, CHILD_STDOUT_FILE), "w") as child_out:
            proc = subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--_child", job_path],
                stdout=child_out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=REPO_ROOT,
                start_new_session=True,
            )
            sampler.start()
            try:
                proc.wait(timeout=args.timeout_s if args.timeout_s > 0 else None)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_group(proc)
            except BaseException:
                _kill_group(proc)
                raise
            finally:
                row["peak_gpu_mib"] = sampler.stop()
        elapsed = time.perf_counter() - started

        result = {}
        if os.path.isfile(result_path):
            with open(result_path) as f:
                result = json.load(f)
        log_text = read_run_log(log_dir, bench_metrics)
        try:
            row.update(collect_metrics(log_text, bench_metrics))
        except Exception as exc:  # noqa: BLE001 - keep the run's timing regardless
            log(f"  WARNING: could not parse the run's log: {exc}")
            row["metrics_error"] = f"{type(exc).__name__}: {exc}"

        row["wall_s"] = result.get("wall_s")
        row["load_s"] = result.get("load_s")
        row["rows_counted"] = result.get("rows_counted")
        error = result.get("error", "")
        if timed_out:
            row["status"] = "timeout"
            row["wall_s"] = row["wall_s"] or elapsed
            error = f"killed after --timeout-s {args.timeout_s}" + (
                f"; {error}" if error else ""
            )
        elif result.get("status") == "ok":
            row["status"], error = check_result(row, args.validate)
        elif result:
            row["status"] = classify_failure(error, log_text)
        else:
            # No result file: the process died before it could report (a signal,
            # the kernel's OOM killer, an abort inside the engine).
            row["status"] = "crashed"
            row["wall_s"] = elapsed
            error = f"process exited with code {proc.returncode} and no result"
            if classify_failure("", log_text) == "oom":
                error += "; the log's last lines mention out of memory"
        row["error"] = " ".join(error.split())[:2000]

        tail = log_text.splitlines()[-args.log_tail_lines :]
        with open(os.path.join(run_dir, LOG_TAIL_FILE), "w") as f:
            f.write("\n".join(tail) + ("\n" if tail else ""))
        if row["status"] != "ok":
            first = bench_metrics.first_problem_lines(
                run_log_paths(log_dir), FIRST_ERRORS_LINES
            )
            with open(os.path.join(run_dir, FIRST_ERRORS_FILE), "w") as f:
                f.write("\n".join(first) + ("\n" if first else ""))
        if not args.keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)
    finally:
        shutil.rmtree(run["spill_dir"], ignore_errors=True)
    _write_json(row, os.path.join(run_dir, "row.json"))
    return row


def check_result(row, validate):
    """Status and error text of a run whose COPY returned."""
    if row["rows_counted"] != row["rows"]:
        return (
            "row_mismatch",
            f"COPY counted {row['rows_counted']} rows, expected {row['rows']}",
        )
    if validate:
        if row.get("discard_rows") is None:
            return "error", "--validate: no [discard_sink] line in the run's log"
        if row["discard_rows"] != row["rows"]:
            return (
                "row_mismatch",
                f"discard sink saw {row['discard_rows']} rows, expected {row['rows']}",
            )
        if row.get("order_violations"):
            return (
                "order_violation",
                f"discard sink found {row['order_violations']} order violations",
            )
    return "ok", ""


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def csv_columns(rows):
    merge_keys = list(MERGE_STATS_KEYS)
    known = set(CSV_COLUMNS) | set(merge_keys) | {"error", "merge_sort_operators"}
    extra = []
    for row in rows:
        for key in row:
            if key not in known and key not in extra:
                extra.append(key)
    return CSV_COLUMNS + ["merge_sort_operators"] + merge_keys + extra + ["error"]


def write_csv(rows, path):
    columns = csv_columns(rows)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        for row in rows:
            out = {k: ("" if v is None else v) for k, v in row.items()}
            if isinstance(out.get("wall_s"), float):
                out["wall_s"] = f"{out['wall_s']:.3f}"
            if isinstance(out.get("load_s"), float):
                out["load_s"] = f"{out['load_s']:.3f}"
            writer.writerow(out)


def _md_cell(row, key, fmt=None):
    value = row.get(key)
    if value is None or value == "":
        return "-"
    return fmt(value) if fmt else str(value)


def write_markdown(rows, path, args, sizings):
    lines = ["# Sort merge benchmark", ""]
    lines.append(f"- Host: `{socket.gethostname()}`, GPU index {args.device}")
    lines.append(
        f"- GPU limit (`sirius.memory.gpu.usage_limit_bytes`): {args.gpu_limit_bytes} "
        f"bytes ({_gib(args.gpu_limit_bytes)} GiB); data: {args.scale:g}x the limit"
    )
    lines.append(
        f"- `spill_min_batch_bytes` in the compressed arm: {args.spill_min_batch_bytes}"
    )
    lines.append(
        f"- Validation (`check_order` + row counts): {'on' if args.validate else 'off'}"
    )
    lines.append(
        "- A `one_shot` run that ends in `oom`, `error`, `timeout` or `crashed` is "
        "an outcome of the comparison, not a harness failure."
    )
    lines.append("")
    header = (
        "| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB "
        "| to disk GiB | OOM retries | shrinks | rounds | intermediate tasks "
        "| rows/round min-max | runs read compressed | compressed refill GiB | error |"
    )
    for dataset in sizings:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue
        sizing = sizings[dataset]
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(
            f"{DATASETS[dataset]['description']}; `ORDER BY "
            f"{', '.join(DATASETS[dataset]['keys'])}`; {sizing['num_batches']} x "
            f"{sizing['rows_per_batch']} rows = {sizing['rows']} rows, "
            f"{sizing['row_width']} bytes/row, {_gib(sizing['bytes'])} GiB."
        )
        lines.append("")
        lines.append(header)
        lines.append("|" + "---|" * header.count("|", 1))
        for r in ds_rows:
            minmax = "-"
            if r.get("round_rows_max") not in (None, ""):
                minmax = f"{r.get('round_rows_min', 0)}-{r['round_rows_max']}"
            error = (r.get("error") or "").replace("|", "\\|")
            if len(error) > 160:
                error = error[:157] + "..."
            cells = [
                r["mode"],
                str(r["compression"]).lower(),
                str(r["repeat"]),
                r.get("status", "-"),
                _md_cell(r, "wall_s", lambda v: f"{v:.1f}"),
                _md_cell(r, "peak_gpu_mib"),
                _md_cell(r, "spill_bytes_host", _gib),
                _md_cell(r, "spill_bytes_disk", _gib),
                _md_cell(r, "oom_retries"),
                _md_cell(r, "shrinks"),
                _md_cell(r, "rounds"),
                _md_cell(r, "intermediate_tasks"),
                minmax,
                _md_cell(r, "runs_read_compressed"),
                _md_cell(r, "refill_bytes_compressed", _gib),
                error or "-",
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _csv_list(text):
    return [item.strip() for item in text.split(",") if item.strip()]


def _bool_list(text):
    out = []
    for item in _csv_list(text):
        if item.lower() not in ("true", "false"):
            raise argparse.ArgumentTypeError(
                f"expected true/false values, got {item!r}"
            )
        out.append(item.lower() == "true")
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="A/B benchmark of sort_merge_mode x spill_compression x dataset "
        "at a multiple of a configured GPU memory limit, ending in the discard sink."
    )
    p.add_argument(
        "--gpu-limit-bytes",
        type=int,
        required=True,
        help="GPU memory limit, written to sirius.memory.gpu.usage_limit_bytes",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="Data bytes as a multiple of the GPU limit; may be below 1 for data "
        "that fits the GPU (default: 10)",
    )
    p.add_argument(
        "--datasets",
        type=_csv_list,
        default=list(DATASETS),
        help=f"Comma-separated subset of: {', '.join(DATASETS)} (default: all)",
    )
    p.add_argument(
        "--modes",
        type=_csv_list,
        default=["one_shot", "windowed"],
        help="Comma-separated sort_merge_mode values (default: one_shot,windowed)",
    )
    p.add_argument(
        "--compression",
        type=_bool_list,
        default=[False, True],
        help="Comma-separated spill_compression values (default: false,true)",
    )
    p.add_argument(
        "--rows-per-batch",
        type=int,
        default=DEFAULT_ROWS_PER_BATCH,
        help=f"Rows per generated batch (default: {DEFAULT_ROWS_PER_BATCH})",
    )
    p.add_argument(
        "--spill-dir",
        default=DEFAULT_SPILL_DIR,
        help=f"Root of the per-run disk spill directories (default: {DEFAULT_SPILL_DIR})",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Results directory (default: test/sort_performance/output/<timestamp>)",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU index; sets CUDA_VISIBLE_DEVICES for each run (default: 0)",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=7200,
        help="Per-run timeout in seconds; 0 disables it (default: 7200)",
    )
    p.add_argument(
        "--repeat", type=int, default=1, help="Runs per matrix cell (default: 1)"
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Add `check_order true` to the discard sink and require its row count "
        "and zero order violations",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix, the configs and the SQL; run nothing",
    )
    p.add_argument(
        "--extension",
        default=DEFAULT_EXTENSION,
        help=f"Path to sirius.duckdb_extension (default: {DEFAULT_EXTENSION}, "
        f"else {WRAPPER_EXTENSION})",
    )
    p.add_argument(
        "--base-config",
        default="",
        help="Sirius config YAML each run's config starts from (default: none)",
    )
    p.add_argument(
        "--spill-min-batch-bytes",
        type=int,
        default=DEFAULT_SPILL_MIN_BATCH_BYTES,
        help="sirius.compression.spill_min_batch_bytes for the compressed arm "
        f"(default: {DEFAULT_SPILL_MIN_BATCH_BYTES})",
    )
    p.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Run even when the free space under --spill-dir looks too small",
    )
    p.add_argument(
        "--keep-logs",
        action="store_true",
        help="Keep each run's full sirius.log, not only its tail",
    )
    p.add_argument(
        "--log-tail-lines",
        type=int,
        default=200,
        help=f"Lines of each run's log kept in {LOG_TAIL_FILE} (default: 200)",
    )
    args = p.parse_args(argv)

    if args.gpu_limit_bytes <= 0:
        p.error("--gpu-limit-bytes must be positive")
    if args.scale <= 0:
        p.error("--scale must be positive")
    if args.repeat < 1:
        p.error("--repeat must be at least 1")
    if not 0 < args.rows_per_batch <= 2**31 - 1:
        p.error("--rows-per-batch must be in [1, 2147483647]")
    if args.log_tail_lines < 1:
        p.error("--log-tail-lines must be at least 1")
    unknown = [d for d in args.datasets if d not in DATASETS]
    if unknown:
        p.error(f"--datasets: unknown {unknown}; valid: {', '.join(DATASETS)}")
    unknown = [m for m in args.modes if m not in MODES]
    if unknown:
        p.error(f"--modes: unknown {unknown}; valid: {', '.join(MODES)}")
    if not args.datasets or not args.modes or not args.compression:
        p.error("--datasets, --modes and --compression must not be empty")
    return args


def resolve_extension(args):
    """Absolute path of the extension; falls back to the wrapper build's when the
    default is absent."""
    path = args.extension
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    if not os.path.isfile(path) and args.extension == DEFAULT_EXTENSION:
        wrapper = os.path.join(REPO_ROOT, WRAPPER_EXTENSION)
        if os.path.isfile(wrapper):
            path = wrapper
    return path


def build_matrix(args, base_config, out_dir, spill_root):
    """Every run of the matrix, in execution order, plus each dataset's sizing."""
    target_bytes = args.scale * args.gpu_limit_bytes
    sizings = {
        d: size_dataset(d, target_bytes, args.rows_per_batch) for d in args.datasets
    }
    runs = []
    for dataset in args.datasets:
        sizing = sizings[dataset]
        sql = build_query(dataset, sizing, args.validate)
        for mode in args.modes:
            for compression in args.compression:
                for rep in range(1, args.repeat + 1):
                    arm = "compressed" if compression else "uncompressed"
                    name = f"{dataset}__{mode}__{arm}__r{rep}"
                    spill_dir = os.path.join(spill_root, name)
                    runs.append(
                        {
                            "name": name,
                            "run_dir": os.path.join(out_dir, PER_RUN_DIR, name),
                            "spill_dir": spill_dir,
                            "sql": sql,
                            "settings": session_settings(mode, compression),
                            "config": derive_run_config(
                                base_config,
                                args.gpu_limit_bytes,
                                spill_dir,
                                compression,
                                args.spill_min_batch_bytes,
                            ),
                            "row": {
                                "dataset": dataset,
                                "mode": mode,
                                "compression": compression,
                                "repeat": rep,
                                "gpu_limit_bytes": args.gpu_limit_bytes,
                                "num_batches": sizing["num_batches"],
                                "rows_per_batch": sizing["rows_per_batch"],
                                "rows": sizing["rows"],
                                "bytes": sizing["bytes"],
                            },
                        }
                    )
    return runs, sizings


def print_matrix(args, runs, sizings):
    log(
        f"GPU limit {args.gpu_limit_bytes} bytes ({_gib(args.gpu_limit_bytes)} GiB), "
        f"data {args.scale:g}x the limit, GPU index {args.device}"
    )
    log(
        f"Matrix: {len(args.datasets)} datasets x {len(args.modes)} modes x "
        f"{len(args.compression)} compression x {args.repeat} repeat = {len(runs)} runs"
    )
    for dataset, sizing in sizings.items():
        log(
            f"  {dataset}: {sizing['num_batches']} x {sizing['rows_per_batch']} rows "
            f"= {sizing['rows']} rows, {sizing['row_width']} bytes/row, "
            f"{_gib(sizing['bytes'])} GiB"
        )
        target = args.scale * args.gpu_limit_bytes
        if abs(sizing["bytes"] - target) > SIZING_TOLERANCE * target:
            actual = sizing["bytes"] / args.gpu_limit_bytes
            log(
                f"  WARNING: {dataset} is {actual:.2f}x the limit, not "
                f"{args.scale:g}x: the data is a whole number of batches (at least "
                "one); lower --rows-per-batch to get closer"
            )
    for i, run in enumerate(runs, 1):
        row = run["row"]
        log(
            f"  [{i}/{len(runs)}] {row['dataset']}  mode={row['mode']}  "
            f"compression={str(row['compression']).lower()}  repeat={row['repeat']}"
        )


def print_dry_run(runs):
    for i, run in enumerate(runs, 1):
        needed = disk_bytes_needed(run["row"]["bytes"], run["config"])
        print(f"\n===== [{i}/{len(runs)}] {run['name']} =====")
        print(f"-- disk needed under the spill dir: about {_gib(needed)} GiB")
        print("-- config (SIRIUS_CONFIG_FILE):")
        print(_dump_yaml(run["config"]).rstrip())
        print("-- SQL:")
        print(";\n".join(run["settings"] + [run["sql"]]) + ";")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--_child":
        sys.exit(run_child(sys.argv[2]))

    args = parse_args()
    base_config = load_base_config(args.base_config)
    args.extension = resolve_extension(args)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(
        args.out_dir or os.path.join(DEFAULT_OUTPUT_ROOT, session)
    )
    spill_root = os.path.join(os.path.abspath(args.spill_dir), f"sort_bench_{session}")
    runs, sizings = build_matrix(args, base_config, out_dir, spill_root)

    print_matrix(args, runs, sizings)
    if args.dry_run:
        print_dry_run(runs)
        log("Dry run: nothing was executed")
        return 0

    if not os.path.isfile(args.extension):
        raise SystemExit(
            f"Sirius extension not found: {args.extension} (build it with "
            "`pixi run make`, or pass --extension)"
        )
    sys.path.insert(0, SCRIPT_DIR)
    import bench_metrics

    os.makedirs(os.path.join(out_dir, PER_RUN_DIR), exist_ok=True)
    os.makedirs(spill_root, exist_ok=True)
    if args.base_config:
        shutil.copyfile(args.base_config, os.path.join(out_dir, "base_config.yml"))
    _write_json(
        {
            "started": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "argv": sys.argv[1:],
            "extension": args.extension,
            "seed": SEED,
            "datasets": {d: {**DATASETS[d], **sizings[d]} for d in sizings},
        },
        os.path.join(out_dir, "run_info.json"),
    )
    log(f"Results: {out_dir}")

    csv_path = os.path.join(out_dir, "results.csv")
    md_path = os.path.join(out_dir, "results.md")
    rows = []
    try:
        for i, run in enumerate(runs, 1):
            log(f"--- [{i}/{len(runs)}] {run['name']} ---")
            needed = disk_bytes_needed(run["row"]["bytes"], run["config"])
            free = shutil.disk_usage(_existing_ancestor(run["spill_dir"])).free
            if needed > free and not args.skip_disk_check:
                row = dict(run["row"])
                row["status"] = "refused"
                row["error"] = (
                    f"needs about {_gib(needed)} GiB under {args.spill_dir}, "
                    f"{_gib(free)} GiB free"
                )
                log(f"  REFUSED: {row['error']}")
            else:
                row = execute_run(run, args, bench_metrics)
                wall = row.get("wall_s")
                log(
                    f"  {row['status']}"
                    + (f" in {wall:.1f}s" if wall is not None else "")
                    + (
                        f", peak GPU {row['peak_gpu_mib']} MiB"
                        if row.get("peak_gpu_mib")
                        else ""
                    )
                    + (f": {row['error'][:300]}" if row.get("error") else "")
                )
            rows.append(row)
            # Rewritten after every run so an interrupted matrix keeps its results.
            write_csv(rows, csv_path)
            write_markdown(rows, md_path, args, sizings)
    finally:
        shutil.rmtree(spill_root, ignore_errors=True)

    failed = [
        r
        for r in rows
        if r["status"] != "ok"
        and not (r["mode"] == "one_shot" and r["status"] in ONE_SHOT_EXPECTED)
    ]
    log(f"Wrote {csv_path}")
    log(f"Wrote {md_path}")
    if failed:
        for r in failed:
            log(
                f"FAILED: {r['dataset']} mode={r['mode']} "
                f"compression={str(r['compression']).lower()} repeat={r['repeat']}: "
                f"{r['status']}"
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
