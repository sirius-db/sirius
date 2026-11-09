#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Compare per-key dynamic-filter strength between two Sirius builds.

Every enabled hash-join publication session logs one ``[dynamic_filter_key]`` DEBUG line per
admitted key that has a binding (see ``src/op/dynamic_filter/dynamic_filter_key_outcome.hpp``).

``run`` executes each ``*.sql`` file (one statement per file) in a fresh DuckDB CLI process with
the Sirius extension loaded, ``SIRIUS_LOG_LEVEL=debug`` and ``SIRIUS_LOG_DIR=<out>/<query>/``:
once on the GPU, then with ``gpu_execution = false``, and records whether the two results match
and whether the GPU statement fell back to the CPU at runtime. Data comes from a DuckDB database
(``--db``) or from Parquet (``--parquet``), read through one view per table as the TPC-H and
TPC-DS benchmark scripts create them.

``diff`` keys every line by (query, sig, occ, condition) and ranks strength as
none < bloom < small_in_list = hash_in_list. A key is ``stronger``, ``equal``, ``weaker``,
``extra`` (candidate only: an upgrade over no filter) or ``missing`` (baseline only: plan drift).
Zone-map presence is compared separately. The exit status is 1 when a result mismatches, either
side's GPU statement fell back to the CPU (its key lines would describe a partial execution), a
key is missing or logged twice, or a key is weaker or loses its zone map with a candidate reason
outside the declared deltas.

Run with ``pixi run python -I scripts/dynamic_filter_key_parity.py``. No third-party deps.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import pathlib
import subprocess
import sys

LINE_MARKER = "[dynamic_filter_key] "
LINE_VERSION = "1"
FIELDS = (
    "v",
    "sig",
    "occ",
    "join",
    "condition",
    "build_ordinal",
    "rows",
    "valid",
    "kind",
    "size",
    "zone_map",
    "pushed",
    "reason",
)
STRENGTH = {"none": 0, "bloom": 1, "small_in_list": 2, "hash_in_list": 2}
# Candidate reasons that explain a deliberate loss of strength (S5b declared deltas).
DECLARED_DELTAS = frozenset(
    {"bloom_cap", "declined_pipeline_barrier", "declined_compressed_arrival"}
)
RESULT_FILE = "result.json"
GPU_OUTPUT = "gpu.csv"
# Printed to stdout by both runtime fallback paths (transparent execution and the explicit
# gpu_execution call), whatever the log level or backend: print_cpu_fallback_banner().
FALLBACK_BANNER = "Error in Sirius GPU execution, fallback to DuckDB"
# Log messages that carry the GPU error on those two paths, matched for the error text.
FALLBACK_LOG_MARKERS = (
    "falling back to DuckDB CPU within the same transaction. GPU error: ",
    "SiriusExecuteQuery error: ",
)
CPU_OUTPUT = "cpu.csv"
DEFAULT_SETTINGS = {"enable_dynamic_zone_map_filter": "true"}
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DUCKDB = REPO_ROOT / "sirius-duckdb" / "build" / "release" / "duckdb"
FLOAT_REL_TOL = 1e-9
FLOAT_ABS_TOL = 1e-12


class ParityError(Exception):
    """A malformed input that makes the comparison meaningless."""


@dataclasses.dataclass(frozen=True)
class KeyLine:
    """One parsed ``[dynamic_filter_key]`` line."""

    sig: str
    occ: int
    join: str
    condition: int
    kind: str
    size: int
    zone_map: bool
    reason: str

    @property
    def strength(self) -> int:
        return STRENGTH[self.kind]


def parse_line(text: str) -> KeyLine | None:
    """Parses the key line in a log line, or returns None when the line holds none."""
    start = text.find(LINE_MARKER)
    if start < 0:
        return None
    tokens = text[start + len(LINE_MARKER) :].split()
    values = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep:
            raise ParityError(f"malformed token {token!r} in {text!r}")
        values[key] = value
    if tuple(values) != FIELDS:
        raise ParityError(f"unexpected fields {list(values)} in {text!r}")
    if values["v"] != LINE_VERSION:
        raise ParityError(f"unsupported line version {values['v']!r} in {text!r}")
    if values["kind"] not in STRENGTH:
        raise ParityError(f"unknown kind {values['kind']!r} in {text!r}")
    return KeyLine(
        sig=values["sig"],
        occ=int(values["occ"]),
        join=values["join"],
        condition=int(values["condition"]),
        kind=values["kind"],
        size=int(values["size"]),
        zone_map=values["zone_map"] == "1",
        reason=values["reason"],
    )


@dataclasses.dataclass
class QueryRun:
    """What one query's directory recorded."""

    keys: dict[tuple[str, int, int], KeyLine]
    duplicates: list[tuple[str, int, int]]
    result_match: bool | None
    gpu_fallback: bool | None = None


def load_query(directory: pathlib.Path) -> QueryRun:
    keys: dict[tuple[str, int, int], KeyLine] = {}
    duplicates = []
    for log in sorted(directory.glob("*.log")):
        for text in log.read_text(encoding="utf-8", errors="replace").splitlines():
            line = parse_line(text)
            if line is None:
                continue
            identity = (line.sig, line.occ, line.condition)
            if identity in keys:
                duplicates.append(identity)
            keys[identity] = line
    result_match = None
    gpu_fallback = None
    result = directory / RESULT_FILE
    if result.exists():
        recorded = json.loads(result.read_text())
        result_match = bool(recorded["result_match"])
        if "gpu_fallback" in recorded:
            gpu_fallback = bool(recorded["gpu_fallback"])
    return QueryRun(keys, duplicates, result_match, gpu_fallback)


def load_run(out: pathlib.Path) -> dict[str, QueryRun]:
    if not out.is_dir():
        raise ParityError(f"{out} is not a directory")
    return {
        child.name: load_query(child)
        for child in sorted(out.iterdir())
        if child.is_dir()
    }


@dataclasses.dataclass
class Row:
    status: str
    query: str
    identity: tuple[str, int, int]
    base: KeyLine | None
    candidate: KeyLine | None
    zone_map: str = "equal"
    failed: bool = False


def compare_keys(query: str, base: QueryRun, candidate: QueryRun) -> list[Row]:
    rows = []
    for identity in sorted(base.keys.keys() | candidate.keys.keys()):
        old = base.keys.get(identity)
        new = candidate.keys.get(identity)
        if new is None:
            rows.append(Row("missing", query, identity, old, None, failed=True))
            continue
        old_strength = old.strength if old else 0
        old_zone = old.zone_map if old else False
        if old is None:
            status = "extra"
        elif new.strength > old_strength:
            status = "stronger"
        elif new.strength < old_strength:
            status = "weaker"
        else:
            status = "equal"
        zone_map = "equal"
        if old_zone and not new.zone_map:
            zone_map = "lost"
        elif new.zone_map and not old_zone:
            zone_map = "gained"
        explained = new.reason in DECLARED_DELTAS
        failed = (status == "weaker" or zone_map == "lost") and not explained
        rows.append(Row(status, query, identity, old, new, zone_map, failed))
    for identity in sorted(set(base.duplicates) | set(candidate.duplicates)):
        rows.append(
            Row(
                "duplicate",
                query,
                identity,
                base.keys.get(identity),
                candidate.keys.get(identity),
                failed=True,
            )
        )
    return rows


def describe(line: KeyLine | None) -> str:
    if line is None:
        return "-"
    return f"{line.kind}/{line.size}/zm{int(line.zone_map)}/{line.reason}"


def diff(base_out: pathlib.Path, candidate_out: pathlib.Path, stream=sys.stdout) -> int:
    base = load_run(base_out)
    candidate = load_run(candidate_out)
    failures = 0
    rows: list[Row] = []
    for query in sorted(base.keys() | candidate.keys()):
        for side, runs in (("baseline", base), ("candidate", candidate)):
            if query not in runs:
                continue
            if not runs[query].result_match:
                state = "unknown" if runs[query].result_match is None else "mismatch"
                print(f"FAIL result {state}: {side} query={query}", file=stream)
                failures += 1
            if runs[query].gpu_fallback is not False:
                state = "unknown" if runs[query].gpu_fallback is None else "fell back"
                print(f"FAIL gpu {state}: {side} query={query}", file=stream)
                failures += 1
        empty = QueryRun({}, [], None)
        rows.extend(
            compare_keys(query, base.get(query, empty), candidate.get(query, empty))
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
        resized = (
            row.base is not None
            and row.candidate is not None
            and row.base.size != row.candidate.size
        )
        if row.status == "equal" and row.zone_map == "equal" and not resized:
            continue
        sig, occ, condition = row.identity
        marker = "FAIL" if row.failed else "info"
        print(
            f"{marker} {row.status} zone_map={row.zone_map} query={row.query} sig={sig} "
            f"occ={occ} condition={condition} base={describe(row.base)} "
            f"candidate={describe(row.candidate)}",
            file=stream,
        )
        failures += int(row.failed)
    summary = " ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(f"summary: {summary or 'no keys'} failures={failures}", file=stream)
    return 1 if failures else 0


def cells_equal(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        a, b = float(left), float(right)
    except ValueError:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)


def rows_equal(left: list[list[str]], right: list[list[str]]) -> bool:
    return len(left) == len(right) and all(
        len(a) == len(b) and all(map(cells_equal, a, b)) for a, b in zip(left, right)
    )


def results_match(gpu_csv: pathlib.Path, cpu_csv: pathlib.Path) -> bool:
    """Whether two CSV results hold the same rows, in order or else as multisets."""
    with gpu_csv.open(newline="") as f:
        gpu = list(csv.reader(f))
    with cpu_csv.open(newline="") as f:
        cpu = list(csv.reader(f))
    if not gpu or not cpu or gpu[0] != cpu[0]:
        return False
    if rows_equal(gpu[1:], cpu[1:]):
        return True
    return rows_equal(sorted(gpu[1:]), sorted(cpu[1:]))


def sql_string(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def parquet_tables(directory: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    """Each table's Parquet files: ``<table>.parquet``, or ``<table>/*.parquet``.

    Other files (metadata.json, schema.sql) are ignored.
    """
    if not directory.is_dir():
        raise ParityError(f"{directory} is not a directory")
    tables: dict[str, list[pathlib.Path]] = {}
    for entry in sorted(directory.iterdir()):
        if entry.is_dir():
            files = sorted(entry.glob("*.parquet"))
            name = entry.name
        elif entry.is_file() and entry.suffix == ".parquet":
            files = [entry]
            name = entry.stem
        else:
            continue
        if not files:
            continue
        if name in tables:
            raise ParityError(
                f"table {name!r} has both a file and a directory in {directory}"
            )
        tables[name] = [file.resolve() for file in files]
    if not tables:
        raise ParityError(f"no Parquet tables in {directory}")
    return tables


def view_statements(tables: dict[str, list[pathlib.Path]]) -> list[str]:
    """One view per table over ``read_parquet``, as the TPC-H and TPC-DS benchmarks create them."""
    statements = []
    for name, files in tables.items():
        file_list = ", ".join(sql_string(str(file)) for file in files)
        identifier = '"' + name.replace('"', '""') + '"'
        statements.append(
            f"CREATE VIEW {identifier} AS SELECT * FROM read_parquet([{file_list}]);"
        )
    return statements


def cli_script(
    extension: pathlib.Path,
    settings: dict[str, str],
    query: str,
    directory: pathlib.Path,
    views: list[str] | None = None,
) -> str:
    """The DuckDB CLI input that runs @p query on the GPU, then on the CPU."""
    lines = [f"LOAD '{extension}';"]
    if views:
        # Creating a view only binds it; disabling GPU execution keeps that certain.
        lines += ["SET gpu_execution = false;", *views, "RESET gpu_execution;"]
    lines += [f"SET {name} = {value};" for name, value in settings.items()]
    lines += [".mode csv", ".headers on"]
    statement = query.strip().rstrip(";") + ";"
    lines += [f'.once "{directory / GPU_OUTPUT}"', statement]
    lines += [
        "SET gpu_execution = false;",
        f'.once "{directory / CPU_OUTPUT}"',
        statement,
    ]
    return "\n".join(lines) + "\n"


def detect_fallback(stdout: str, directory: pathlib.Path) -> tuple[bool, str | None]:
    """Whether the GPU statement fell back to the CPU, and the GPU error the log names, if any.

    The CPU statement runs with GPU execution disabled, so any fallback belongs to the GPU one.
    """
    fell_back = FALLBACK_BANNER in stdout
    error = None
    for log in sorted(directory.glob("*.log")):
        for text in log.read_text(encoding="utf-8", errors="replace").splitlines():
            for marker in FALLBACK_LOG_MARKERS:
                at = text.find(marker)
                if at >= 0:
                    fell_back = True
                    error = error or text[at + len(marker) :].strip()
    return fell_back, error


def parse_settings(pairs: list[str]) -> dict[str, str]:
    settings = dict(DEFAULT_SETTINGS)
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name.strip():
            raise ParityError(f"--set expects name=value, got {pair!r}")
        settings[name.strip()] = value.strip()
    return settings


def run(args: argparse.Namespace) -> int:
    queries = sorted(pathlib.Path(args.queries).glob("*.sql"))
    if not queries:
        raise ParityError(f"no *.sql files in {args.queries}")
    settings = parse_settings(args.set)
    views = (
        view_statements(parquet_tables(pathlib.Path(args.parquet)))
        if args.parquet
        else None
    )
    out = pathlib.Path(args.out).resolve()
    extension = pathlib.Path(args.extension).resolve()
    failures = 0
    for path in queries:
        directory = out / path.stem
        if directory.exists() and any(directory.iterdir()):
            raise ParityError(f"{directory} is not empty; use a fresh --out directory")
        directory.mkdir(parents=True, exist_ok=True)
        # Settings such as enable_dynamic_zone_map_filter are test-only SET options.
        env = dict(
            os.environ,
            SIRIUS_LOG_LEVEL="debug",
            SIRIUS_LOG_DIR=str(directory),
            SIRIUS_ENABLE_TEST_OPTIONS="1",
        )
        # -no-agent keeps the plain output format when an AI agent environment is detected.
        command = [str(args.duckdb), "-no-agent", "-unsigned", "-bail"]
        if args.db:
            command += ["-readonly", str(args.db)]
        script = cli_script(extension, settings, path.read_text(), directory, views)
        try:
            completed = subprocess.run(
                command,
                input=script,
                text=True,
                capture_output=True,
                env=env,
                timeout=args.timeout,
            )
            returncode, stdout, stderr = (
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
        except subprocess.TimeoutExpired as expired:
            returncode, stderr = None, f"timed out after {args.timeout} s"
            stdout = expired.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
        fell_back, gpu_error = detect_fallback(stdout, directory)
        gpu, cpu = directory / GPU_OUTPUT, directory / CPU_OUTPUT
        match = (
            returncode == 0
            and gpu.exists()
            and cpu.exists()
            and results_match(gpu, cpu)
        )
        (directory / RESULT_FILE).write_text(
            json.dumps(
                {
                    "query": path.stem,
                    "returncode": returncode,
                    "result_match": match,
                    "gpu_fallback": fell_back,
                    "gpu_error": gpu_error,
                },
                indent=2,
            )
            + "\n"
        )
        status = "ok" if match else "MISMATCH" if returncode == 0 else "ERROR"
        if fell_back:
            status = f"FALLBACK({status})"
        print(f"{status} {path.stem}", flush=True)
        if fell_back:
            print(f"  GPU error: {gpu_error or 'unknown'}", flush=True)
        if returncode != 0:
            print(stderr.strip(), file=sys.stderr)
        failures += int(not match or fell_back)
    return 1 if failures else 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="run queries and capture key lines")
    run_cmd.add_argument(
        "--extension", required=True, help="sirius.duckdb_extension path"
    )
    source = run_cmd.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", help="DuckDB database file, opened read-only")
    source.add_argument(
        "--parquet",
        help="directory of <table>.parquet files or <table>/*.parquet subdirectories",
    )
    run_cmd.add_argument("--queries", required=True, help="directory of *.sql files")
    run_cmd.add_argument(
        "--out", required=True, help="output directory, one subdirectory per query"
    )
    run_cmd.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra SET statement (value is SQL); "
        "defaults include enable_dynamic_zone_map_filter=true",
    )
    run_cmd.add_argument(
        "--duckdb", default=str(DEFAULT_DUCKDB), help="DuckDB CLI binary"
    )
    run_cmd.add_argument(
        "--timeout", type=float, default=1800.0, help="seconds per query"
    )
    diff_cmd = commands.add_parser("diff", help="compare two run outputs")
    diff_cmd.add_argument("base_out", type=pathlib.Path)
    diff_cmd.add_argument("candidate_out", type=pathlib.Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        return diff(args.base_out, args.candidate_out)
    except ParityError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
