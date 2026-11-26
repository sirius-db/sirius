#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Run TPC-H with the table path of GROUP BY off and on, compare the results and time both.

For every query and every mode the script starts one DuckDB CLI process, loads the Sirius
extension, runs the query a few times and saves the last result. Modes:

  off   SET agg_table_enabled = false   today's cudf::groupby -> PARTITION -> merge
  on    SET agg_table_enabled = true    the table path (src/cuda/agg_table)
  cpu   plain DuckDB (SIRIUS_DISABLE=1), only with --cpu

Both GPU modes run with `SET enable_duckdb_fallback = false`, so a query that would fall back to
the CPU fails instead of passing as a GPU result. The run reports, per query, whether `on` returned
the rows of `off` (and both the rows of `cpu`), and the timings of each mode.

Run it through pixi from the repository root, on a host with the GPU(s) to test:

    pixi run python scripts/agg_table_validate.py /data/tpch_sf100 --scale-factor 100 \\
        --config /path/to/sirius.yaml --output-dir /scratch/agg-table-sf100

    # a DuckDB database instead of Parquet, four queries, a CPU reference, five timed runs
    pixi run python scripts/agg_table_validate.py \\
        --duckdb-file test/cpp/integration/data/duckdb/integration.duckdb --scale-factor 0.1 \\
        --config test/cpp/integration/integration.yaml --output-dir /tmp/agg-table-sf01 \\
        --queries 13 15 17 18 --cpu --iterations 5

    # a table capped at 64 MiB in the `on` mode only, small scan batches in both.
    # scan_task_batch_size is a test-only setting: Sirius registers it, like the
    # sirius_test_* hooks, only when SIRIUS_ENABLE_TEST_OPTIONS=1 is in the environment.
    SIRIUS_ENABLE_TEST_OPTIONS=1 pixi run python scripts/agg_table_validate.py /data/tpch_sf100 \\
        --scale-factor 100 --config sirius.yaml --output-dir out \\
        --set scan_task_batch_size=268435456 --set-on agg_table_budget_bytes=67108864

Inputs. A directory with the eight TPC-H tables as `table.parquet`, `table/*.parquet` or
`table_<n>*.parquet` shards (the layouts test/scripts/classify_queries.py accepts), or
`--duckdb-file` with a database that holds them; it is attached read-only. The queries are the 22
of test/tpch_performance/queries.py, with Q11's fraction rendered for `--scale-factor`.

The CLI (`--duckdb`, default sirius-duckdb/build/release/duckdb) does not contain Sirius: every
GPU process starts it with `-unsigned` and LOADs `--extension` (default `$SIRIUS_EXTENSION_PATH`,
else the wrapper build's sirius.duckdb_extension). The two must come from the same build. The
Sirius configuration is `--config`, exported as SIRIUS_CONFIG_FILE; the number of GPUs, the memory
limits and the thread pools are whatever that file says.

Timing. Each process runs the query `--warmup` times (default 1, not reported), then
`--iterations` times (default 3) with the CLI's timer, discarding the rows, then once more for
the result. Reported times are the CLI's wall-clock "real" seconds of a statement, so they include
planning and result delivery; they are for comparing the two modes of one build on one host, not
for publishing. A fresh process per query and mode means no mode sees the other's caches, and
both pay the same cold start before the warm-up.

Comparison. Rows compare as multisets (duplicates count, order does not): FLOAT and DOUBLE cells
within `--float-rel-tolerance` or `--float-abs-tolerance`, every other cell exactly, including
DECIMAL sums and NULLs. A LIMIT over ties can select different valid rows; inspect a reported
mismatch on such a query before calling it a defect.

Output directory (must not exist):

    metadata.json            revision, binaries and their hashes, arguments
    sirius.yaml              copy of --config
    q<N>/<mode>/             input.sql, stdout.txt, stderr.txt, result.json, log/ (Sirius logs),
                             and whatever else Sirius writes to its working directory
    timings.csv              query, mode, iteration, seconds
    results.csv              query, per-mode status and min/median seconds, speedup, verdicts
    summary.txt              the table printed at the end

Exit status: 0 when every query ran in every mode and every comparison matched; 1 when a query
failed or a comparison did not match; 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "sirius-duckdb/build/release"
QUERIES_PY = ROOT / "test/tpch_performance/queries.py"
TABLES = "customer lineitem nation orders part partsupp region supplier".split()
# The one name to change if the engine's switch is renamed (also --setting).
DEFAULT_SETTING = "agg_table_enabled"
# Printed by the CLI around the two pieces of output the script reads.
SCHEMA_MARKER = "@@agg_table_validate_schema"
SCHEMA_END_MARKER = "@@agg_table_validate_schema_end"
RESULT_MARKER = "@@agg_table_validate_result"
TIMER_LINE = re.compile(r"^Run Time \(s\): real ([0-9.]+)", re.MULTILINE)
SETTING_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def quote(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_queries(scale_factor: float) -> dict[int, str]:
    """The 22 queries of test/tpch_performance/queries.py, keyed by number."""
    spec = importlib.util.spec_from_file_location("tpch_queries", QUERIES_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rendered = module.queries_for_scale_factor(scale_factor)
    return {int(name[1:]): sql.strip().rstrip(";") for name, sql in rendered.items()}


def table_setup(parquet_dir: Path | None, duckdb_file: Path | None) -> str:
    """SQL that makes the eight tables visible on a fresh connection."""
    if duckdb_file is not None:
        return f"ATTACH {quote(duckdb_file)} AS tpch (READ_ONLY);\nUSE tpch;\n"
    views = []
    for table in TABLES:
        files = sorted(
            p
            for pattern in (
                f"{table}.parquet",
                f"{table}/*.parquet",
                f"{table}_[0-9]*.parquet",
            )
            for p in parquet_dir.glob(pattern)
            if p.is_file()
        )
        if not files:
            raise ValueError(f"No parquet files for {table} in {parquet_dir}")
        views.append(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet(["
            + ",".join(quote(path) for path in files)
            + "]);\n"
        )
    return "".join(views)


def parse_assignment(text: str) -> tuple[str, str]:
    """`name=value` of --set / --set-on / --set-off as a checked (name, SQL literal)."""
    name, sep, value = text.partition("=")
    name, value = name.strip(), value.strip()
    if not sep or not SETTING_NAME.fullmatch(name) or not value:
        raise ValueError(f"expected name=value, got {text!r}")
    if re.fullmatch(r"-?[0-9]+(\.[0-9]+)?|true|false", value, re.IGNORECASE):
        return name, value
    return name, quote(value.strip("'"))


def json_results(output: str) -> list:
    """Consecutive JSON result sets of the CLI, keeping NULLs and exact decimals."""

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate result column: {key}")
            result[key] = value
        return result

    decoder = json.JSONDecoder(parse_float=Decimal, object_pairs_hook=unique_keys)
    results = []
    remaining = output.strip()
    while remaining:
        value, end = decoder.raw_decode(remaining)
        if not isinstance(value, list):
            raise ValueError("Expected a JSON array of result rows")
        results.append(value)
        remaining = remaining[end:].strip()
    return results


def compare_rows(schema, expected, actual, rel_tol: float, abs_tol: float):
    """Compare two row multisets; returns (match, detail).

    FLOAT / DOUBLE columns of @p schema compare within the tolerances, every other column
    exactly. The matching is a maximum bipartite matching, so sorting approximate floats cannot
    pair the wrong rows and one row cannot satisfy two duplicates.
    """
    names = tuple(column["column_name"] for column in schema)
    types = [column["column_type"].upper() for column in schema]
    floats = [i for i, typ in enumerate(types) if typ in {"FLOAT", "DOUBLE", "REAL"}]

    def normalize(rows):
        normalized = []
        for row in rows:
            if tuple(row) != names:
                raise ValueError(f"Result columns {tuple(row)!r} differ from {names!r}")
            values = list(row.values())
            for i, typ in enumerate(types):
                if values[i] is None:
                    continue
                if i in floats:
                    values[i] = float(values[i])
                elif typ.startswith("DECIMAL("):
                    values[i] = Decimal(str(values[i]))
            normalized.append(tuple(values))
        return normalized

    expected, actual = normalize(expected), normalize(actual)
    if len(expected) != len(actual):
        return False, f"row count {len(expected)} vs {len(actual)}"
    if collections.Counter(expected) == collections.Counter(actual):
        return True, "exact"
    if not floats:
        missing = next(
            (collections.Counter(expected) - collections.Counter(actual)).elements()
        )
        return False, f"no matching row for {missing!r}"

    def matches(lhs, rhs):
        for i, (a, b) in enumerate(zip(lhs, rhs)):
            if a == b:
                continue
            if i not in floats or a is None or b is None:
                return False
            if math.isnan(a) and math.isnan(b):
                continue
            if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        return True

    # Rows that are exactly equal pair off first; only the rest need the matching.
    left = list(
        (collections.Counter(expected) - collections.Counter(actual)).elements()
    )
    right = list(
        (collections.Counter(actual) - collections.Counter(expected)).elements()
    )
    edges = [[j for j, rhs in enumerate(right) if matches(lhs, rhs)] for lhs in left]
    assigned: dict[int, int] = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or assign(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(left)):
        if not assign(i, set()):
            return False, f"no matching row for {left[i]!r}"
    return True, f"within tolerance ({len(left)} rows differ in float digits)"


class Run:
    """Outcome of one CLI process: one query in one mode."""

    def __init__(self):
        self.status = "ok"  # ok | error | timeout
        self.error = ""
        self.seconds: list[float] = []
        self.schema = None
        self.rows = None


def run_mode(args, mode, number, sql, setup, directory: Path) -> Run:
    gpu = mode != "cpu"
    lines = []
    if gpu:
        lines.append(f"LOAD {quote(args.extension)};")
    lines.append(setup.rstrip("\n"))
    # The column types come from the binder, on the CPU, so the GPU plan is not involved.
    if gpu:
        lines.append("SET gpu_execution = false;")
    lines += [".mode json", f".print {SCHEMA_MARKER}", f"DESCRIBE {sql};"]
    lines.append(f".print {SCHEMA_END_MARKER}")
    if gpu:
        lines.append("SET gpu_execution = true;")
        if not args.allow_fallback:
            lines.append("SET enable_duckdb_fallback = false;")
        value = args.on_value if mode == "on" else args.off_value
        lines.append(f"SET {args.setting} = {value};")
        for name, literal in args.set + (args.set_on if mode == "on" else args.set_off):
            lines.append(f"SET {name} = {literal};")
    lines.append(".mode trash")
    lines += [f"{sql};"] * args.warmup
    lines.append(".timer on")
    lines += [f"{sql};"] * args.iterations
    lines += [".timer off", f".print {RESULT_MARKER}", ".mode json", f"{sql};"]
    text = "\n".join(lines) + "\n"

    directory.mkdir(parents=True)
    (directory / "input.sql").write_text(text)
    env = dict(os.environ)
    if gpu:
        env["SIRIUS_CONFIG_FILE"] = str(args.config)
        env["SIRIUS_LOG_DIR"] = str(directory / "log")
        env.pop("SIRIUS_DISABLE", None)
    else:
        env["SIRIUS_DISABLE"] = "1"
    run = Run()
    try:
        completed = subprocess.run(
            [str(args.duckdb), "-unsigned", "-bail", "-no-agent"],
            input=text,
            text=True,
            capture_output=True,
            env=env,
            # Sirius writes telemetry_data/ and cufile.log into the working directory.
            cwd=directory,
            timeout=args.timeout,
        )
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, code = exc.stdout or "", exc.stderr or "", None
        stdout = (
            stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        )
        stderr = (
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        )
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    if code is None:
        run.status, run.error = "timeout", f"exceeded {args.timeout:g}s"
        return run
    if code != 0:
        errors = [line for line in stderr.splitlines() if line.strip()]
        run.status, run.error = "error", (errors[0] if errors else f"exit code {code}")
        return run
    head, marker, tail = stdout.partition(RESULT_MARKER)
    try:
        if not marker or SCHEMA_MARKER not in head or SCHEMA_END_MARKER not in head:
            raise ValueError("the CLI stopped before the result run")
        schema_text = head.partition(SCHEMA_MARKER)[2].partition(SCHEMA_END_MARKER)[0]
        describe = json_results(schema_text)
        result = json_results(tail)
        if len(describe) != 1 or len(result) != 1:
            raise ValueError("expected one schema and one result set")
        run.schema, run.rows = describe[0], result[0]
    except (ValueError, TypeError) as exc:
        run.status, run.error = "error", f"cannot read the CLI output: {exc}"
        return run
    run.seconds = [float(s) for s in TIMER_LINE.findall(head)]
    if len(run.seconds) != args.iterations:
        run.status = "error"
        run.error = f"{len(run.seconds)} timer lines for {args.iterations} iterations"
        return run
    (directory / "result.json").write_text(
        json.dumps(run.rows, default=str, indent=0) + "\n"
    )
    return run


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("parquet_dir", type=Path, nargs="?")
    parser.add_argument(
        "--duckdb-file", type=Path, help="DuckDB database with the tables"
    )
    parser.add_argument("--scale-factor", type=float, required=True, help="for Q11")
    parser.add_argument("--config", type=Path, required=True, help="Sirius YAML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", nargs="+", type=int, help="default: all 22")
    parser.add_argument(
        "--iterations", type=int, default=3, help="timed runs (default 3)"
    )
    parser.add_argument(
        "--warmup", type=int, default=1, help="untimed runs (default 1)"
    )
    parser.add_argument("--cpu", action="store_true", help="also run plain DuckDB")
    parser.add_argument(
        "--timeout", type=float, default=1800, help="seconds per process"
    )
    parser.add_argument("--duckdb", type=Path, default=WRAPPER / "duckdb")
    parser.add_argument(
        "--extension",
        type=Path,
        default=os.environ.get("SIRIUS_EXTENSION_PATH")
        or WRAPPER / "extension/sirius/sirius.duckdb_extension",
    )
    parser.add_argument("--setting", default=DEFAULT_SETTING, help="the switch to flip")
    parser.add_argument("--off-value", default="false")
    parser.add_argument("--on-value", default="true")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="in both GPU modes",
    )
    parser.add_argument(
        "--set-on",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="only in `on`",
    )
    parser.add_argument(
        "--set-off",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="only in `off`",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="leave enable_duckdb_fallback on (a fallback then passes as a GPU result)",
    )
    parser.add_argument("--float-rel-tolerance", type=float, default=1e-9)
    parser.add_argument("--float-abs-tolerance", type=float, default=1e-10)
    args = parser.parse_args()

    if (args.parquet_dir is None) == (args.duckdb_file is None):
        parser.error("pass either a Parquet directory or --duckdb-file")
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be at least 1 and --warmup at least 0")
    if not args.scale_factor > 0:
        parser.error("--scale-factor must be positive")
    if not SETTING_NAME.fullmatch(args.setting):
        parser.error(f"--setting is not a setting name: {args.setting!r}")
    for name in ("off_value", "on_value"):
        value = getattr(args, name)
        if not re.fullmatch(r"-?[0-9]+(\.[0-9]+)?|true|false", value, re.IGNORECASE):
            setattr(args, name, quote(value.strip("'")))
    try:
        args.set = [parse_assignment(text) for text in args.set]
        args.set_on = [parse_assignment(text) for text in args.set_on]
        args.set_off = [parse_assignment(text) for text in args.set_off]
    except ValueError as exc:
        parser.error(str(exc))
    args.duckdb, args.extension = args.duckdb.resolve(), Path(args.extension).resolve()
    args.config = args.config.resolve()
    for path in (args.duckdb, args.extension, args.config):
        if not path.is_file():
            parser.error(f"Not found: {path}")
    data = (args.duckdb_file or args.parquet_dir).expanduser().resolve()
    if not data.exists():
        parser.error(f"Not found: {data}")
    try:
        queries = load_queries(args.scale_factor)
        setup = table_setup(
            None if args.duckdb_file else data, data if args.duckdb_file else None
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    numbers = args.queries or sorted(queries)
    if unknown := [n for n in numbers if n not in queries]:
        parser.error(f"unknown queries: {unknown}")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a fresh directory")
    output.mkdir(parents=True)
    (output / "sirius.yaml").write_bytes(args.config.read_bytes())

    def git(*command):
        try:
            return subprocess.check_output(
                ["git", *command], cwd=ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    modes = ["off", "on"] + (["cpu"] if args.cpu else [])
    metadata = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "argv": sys.argv,
        "data": str(data),
        "scale_factor": args.scale_factor,
        "queries": numbers,
        "modes": modes,
        "setting": {"name": args.setting, "off": args.off_value, "on": args.on_value},
        "set": {"both": args.set, "on": args.set_on, "off": args.set_off},
        "fallback_disabled": not args.allow_fallback,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "binary": str(args.duckdb),
        "binary_sha256": sha256(args.duckdb),
        "extension": str(args.extension),
        "extension_sha256": sha256(args.extension),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "float_rel_tolerance": args.float_rel_tolerance,
        "float_abs_tolerance": args.float_abs_tolerance,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def compare(reference: Run, candidate: Run) -> str:
        if reference.status != "ok" or candidate.status != "ok":
            return "not compared"
        if reference.schema != candidate.schema:
            return "MISMATCH: column names or types differ"
        try:
            match, detail = compare_rows(
                reference.schema,
                reference.rows,
                candidate.rows,
                args.float_rel_tolerance,
                args.float_abs_tolerance,
            )
        except (ValueError, TypeError, ArithmeticError) as exc:
            return f"MISMATCH: cannot compare: {exc}"
        return (
            ("match" if detail == "exact" else f"match ({detail})")
            if match
            else f"MISMATCH: {detail}"
        )

    header = ["query"]
    for mode in modes:
        header += [f"{mode}_status", f"{mode}_min_s", f"{mode}_median_s"]
    header += ["speedup_off_over_on", "on_vs_off"]
    if args.cpu:
        header += ["off_vs_cpu", "on_vs_cpu"]
    header.append("errors")
    rows = []
    failed = False
    with open(output / "timings.csv", "w", newline="") as timings_file:
        timings = csv.writer(timings_file)
        timings.writerow(["query", "mode", "iteration", "seconds"])
        for number in numbers:
            runs = {}
            for mode in modes:
                print(f"Q{number} {mode} ...", end=" ", flush=True)
                run = run_mode(
                    args,
                    mode,
                    number,
                    queries[number],
                    setup,
                    output / f"q{number}" / mode,
                )
                runs[mode] = run
                for i, seconds in enumerate(run.seconds, 1):
                    timings.writerow([number, mode, i, seconds])
                timings_file.flush()
                print(
                    (
                        f"{min(run.seconds):.3f}s"
                        if run.status == "ok"
                        else f"{run.status}: {run.error[:160]}"
                    ),
                    flush=True,
                )
            row = {"query": f"Q{number}"}
            for mode in modes:
                run = runs[mode]
                row[f"{mode}_status"] = run.status
                row[f"{mode}_min_s"] = f"{min(run.seconds):.3f}" if run.seconds else ""
                row[f"{mode}_median_s"] = (
                    f"{statistics.median(run.seconds):.3f}" if run.seconds else ""
                )
            off, on = runs["off"], runs["on"]
            row["speedup_off_over_on"] = (
                f"{min(off.seconds) / min(on.seconds):.2f}"
                if off.seconds and on.seconds and min(on.seconds) > 0
                else ""
            )
            verdicts = [("on_vs_off", compare(off, on))]
            if args.cpu:
                verdicts += [
                    ("off_vs_cpu", compare(runs["cpu"], off)),
                    ("on_vs_cpu", compare(runs["cpu"], on)),
                ]
            for name, verdict in verdicts:
                row[name] = verdict
                failed |= not verdict.startswith("match")
            row["errors"] = " | ".join(
                f"{mode}: {runs[mode].error}" for mode in modes if runs[mode].error
            )
            rows.append(row)
    with open(output / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    shown = [c for c in header if c != "errors" and not c.endswith("_median_s")]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in shown}
    lines = ["  ".join(c.ljust(widths[c]) for c in shown)]
    lines += ["  ".join(str(r[c]).ljust(widths[c]) for c in shown) for r in rows]
    for r in rows:
        if r["errors"]:
            lines.append(f"{r['query']}: {r['errors']}")
    matched = sum(r["on_vs_off"].startswith("match") for r in rows)
    lines.append(
        f"{matched}/{len(rows)} queries: `on` returned the rows of `off`. "
        f"Setting {args.setting}: off={args.off_value}, on={args.on_value}. "
        f"Times are the best of {args.iterations} runs after {args.warmup} warm-up, in seconds."
    )
    summary = "\n".join(lines) + "\n"
    (output / "summary.txt").write_text(summary)
    print("\n" + summary + f"Details in {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
