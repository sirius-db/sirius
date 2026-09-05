#!/usr/bin/env python3
"""Run the repository's 22 TPC-H queries through an FE and an exact CN count.

Use --prepare-only to generate SQL and CPU DuckDB answers without contacting the FE.
No services are started or stopped. Every query has an independent client deadline.
"""

import argparse
from collections import defaultdict, deque
import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
TABLES = (
    "customer",
    "lineitem",
    "nation",
    "orders",
    "part",
    "partsupp",
    "region",
    "supplier",
)
QUERY_SOURCE = ROOT / "test/tpch_performance/queries.py"
MYSQL = ROOT / "experimental/starrocks/.pixi/envs/default/bin/mysql"
SQL_ADAPTATIONS = {
    "q08": "Reorder the inner-join FROM list from lineitem,part,supplier to "
    "part,lineitem,supplier to use the branch's documented FILES join order. "
    "Apply identically to DuckDB; parameters and result semantics are unchanged.",
    "q09": "Reorder the inner-join FROM list from part,supplier,lineitem to "
    "part,lineitem,supplier to avoid the FILES planner's Cartesian intermediate. "
    "Apply identically to DuckDB; parameters and result semantics are unchanged.",
    "q11": "Use the TPC-H scale-dependent HAVING fraction 0.0001 / scale_factor "
    "identically on both engines. The default scale_factor=1 retains the original threshold.",
    "q22": "Replace the three substring(c_phone from 1 for 2) expressions with "
    "substring(c_phone, 1, 2) for StarRocks syntax; apply identically to DuckDB. "
    "Query parameters and substring semantics are unchanged.",
}


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def quote_sql(value):
    return "'" + str(value).replace("'", "''") + "'"


def adapt_query(name, sql, scale_factor=Decimal(1)):
    if name in ("q08", "q09"):
        pattern = (
            r"lineitem\s+l\s*,\s*part\s+p\s*,\s*supplier\s+s\s*,"
            if name == "q08"
            else r"part\s+p\s*,\s*supplier\s+s\s*,\s*lineitem\s+l\s*,"
        )
        adapted, count = re.subn(
            pattern,
            "part p,\n      lineitem l,\n      supplier s,",
            sql,
            flags=re.IGNORECASE,
        )
        if count != 1:
            raise RuntimeError(
                f"Expected one {name.upper()} inner-join FROM sequence, found {count}"
            )
        return adapted
    if name == "q11":
        adapted, count = re.subn(
            r"\b0\.0001000000\b", f"(0.0001000000 / {scale_factor})", sql
        )
        if count != 1:
            raise RuntimeError(f"Expected one Q11 HAVING fraction, found {count}")
        return adapted
    if name != "q22":
        return sql
    adapted, count = re.subn(
        r"substring\(\s*c_phone\s+from\s+1\s+for\s+2\s*\)",
        "substring(c_phone, 1, 2)",
        sql,
        flags=re.IGNORECASE,
    )
    if count != 3:
        raise RuntimeError(f"Expected three Q22 substring expressions, found {count}")
    return adapted


def parquet_ctes(sql, files, starrocks):
    definitions = []
    for table, path in files.items():
        source = (
            f"FILES('path'={quote_sql(path.as_uri().replace('%2A', '*'))}, 'format'='parquet')"
            if starrocks
            else f"read_parquet({quote_sql(path)})"
        )
        definitions.append(f"{table} AS (SELECT * FROM {source})")
    # Q15 already has a revenue CTE. Merge the preludes, preserving its query body.
    body = sql.strip().rstrip(";")
    with_match = re.match(r"WITH\s+", body, re.IGNORECASE)
    if with_match:
        return (
            "WITH\n"
            + ",\n".join(definitions)
            + ",\n"
            + body[with_match.end() :]
            + ";\n"
        )
    return "WITH\n" + ",\n".join(definitions) + "\n" + body + ";\n"


def encode_cell(value):
    if value is None:
        return {"kind": "null", "value": None}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": str(value)}
    if isinstance(value, (float, Decimal)):
        return {"kind": "number", "value": str(value)}
    return {"kind": "text", "value": str(value)}


def oracle_worker(query_dir, memory_limit, threads):
    # This worker deliberately loads no Sirius extension. Disabling extension auto-load
    # also prevents an installed extension from changing the CPU reference execution.
    import duckdb

    connection = duckdb.connect(
        config={
            "autoload_known_extensions": "false",
            "autoinstall_known_extensions": "false",
        }
    )
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET memory_limit={quote_sql(memory_limit)}")
    connection.execute(f"SET temp_directory={quote_sql(query_dir / 'oracle-temp')}")
    start = time.monotonic()
    try:
        result = connection.execute((query_dir / "duckdb.sql").read_text())
        columns = [column[0] for column in result.description]
        types = [str(column[1]) for column in result.description]
        rows = [[encode_cell(value) for value in row] for row in result.fetchall()]
        write_json(
            query_dir / "oracle.json",
            {
                "status": "OK",
                "duckdb_version": duckdb.__version__,
                "memory_limit": memory_limit,
                "threads": threads,
                "elapsed_seconds": time.monotonic() - start,
                "columns": columns,
                "types": types,
                "rows": rows,
            },
        )
    finally:
        connection.close()


def run_process(command, stdout_path, stderr_path, timeout, *, sql=None, env=None):
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if sql is not None else None,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        timed_out = False
        try:
            process.communicate(None if sql is None else sql.encode(), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.communicate()
            stderr.write(
                f"\nRunner client deadline exceeded ({timeout:g} seconds).\n".encode()
            )
    return {
        "status": (
            "TIMEOUT" if timed_out else ("OK" if process.returncode == 0 else "ERROR")
        ),
        "returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - start,
        "started_utc": started_utc,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def mysql_unescape(value):
    replacements = {"0": "\0", "n": "\n", "t": "\t", "r": "\r", "Z": "\x1a", "\\": "\\"}
    return re.sub(r"\\([0ntrZ\\])", lambda match: replacements[match[1]], value)


def read_mysql(path):
    # mysql --batch escapes embedded tabs/newlines, so rows and duplicate rows remain
    # unambiguous. Keep the exact client bytes on disk; decode only for comparison.
    lines = path.read_text().splitlines()
    if not lines:
        return [], []
    return lines[0].split("\t"), [
        [mysql_unescape(cell) for cell in line.split("\t")] for line in lines[1:]
    ]


def compare_ordered_rows(expected, actual_header, actual_rows, relative, absolute):
    expected_rows = expected["rows"]
    differences = []
    bad_cells = 0
    maximum_absolute = Decimal(0)
    maximum_relative = Decimal(0)
    if len(actual_rows) != len(expected_rows):
        differences.append(
            f"Row count differs: StarRocks={len(actual_rows)}, DuckDB={len(expected_rows)}"
        )
    # mysql does not emit column headers for some empty result sets.
    if actual_header and len(actual_header) != len(expected["columns"]):
        differences.append(
            f"Column count differs: StarRocks={len(actual_header)}, DuckDB={len(expected['columns'])}"
        )
    for row_index, (oracle_row, actual_row) in enumerate(
        zip(expected_rows, actual_rows), 1
    ):
        if len(oracle_row) != len(actual_row):
            bad_cells += 1
            if len(differences) < 20:
                differences.append(
                    f"Row {row_index} column count differs: {len(actual_row)} != {len(oracle_row)}"
                )
            continue
        for column_index, (reference, actual) in enumerate(
            zip(oracle_row, actual_row), 1
        ):
            wanted = reference["value"]
            if reference["kind"] == "null":
                matches = actual == "NULL"
            elif reference["kind"] == "bool":
                matches = actual.lower() in (
                    ("1", "true") if wanted else ("0", "false")
                )
            elif reference["kind"] in ("integer", "number"):
                try:
                    left, right = Decimal(actual), Decimal(wanted)
                    if not left.is_finite() or not right.is_finite():
                        matches = left == right or (left.is_nan() and right.is_nan())
                    else:
                        error = abs(left - right)
                        maximum_absolute = max(maximum_absolute, error)
                        if right:
                            maximum_relative = max(maximum_relative, error / abs(right))
                        matches = (
                            left == right
                            if reference["kind"] == "integer"
                            else error <= max(absolute, relative * abs(right))
                        )
                except InvalidOperation:
                    matches = False
            else:
                matches = wanted == actual
            if not matches:
                bad_cells += 1
                if len(differences) < 20:
                    differences.append(
                        f"Row {row_index}, column {column_index} ({expected['columns'][column_index - 1]}): "
                        f"StarRocks={actual!r}, DuckDB={wanted!r}"
                    )
    return {
        "match": not differences and bad_cells == 0,
        "expected_rows": len(expected_rows),
        "actual_rows": len(actual_rows),
        "bad_cells": bad_cells,
        "differences": differences,
        "max_absolute_numeric_error": str(maximum_absolute),
        "max_relative_numeric_error": str(maximum_relative),
    }


def comparison_key(row, types, *, oracle):
    """Use exact text, integer, Boolean, and NULL values to bound numeric matching."""
    key = []
    for index, cell in enumerate(row):
        value = cell["value"] if oracle else cell
        kind = cell["kind"] if oracle else None
        type_name = types[index].upper()
        if (oracle and kind == "null") or (not oracle and value == "NULL"):
            key.append(("null",))
        elif type_name.startswith(("DECIMAL", "FLOAT", "DOUBLE", "REAL")):
            key.append(("number",))
        elif "INT" in type_name:
            try:
                key.append(("integer", Decimal(value)))
            except InvalidOperation:
                key.append(("invalid_integer", value))
        elif type_name == "BOOLEAN":
            key.append(("bool", value if oracle else value.lower() in ("true", "1")))
        else:
            key.append(("text", value))
    return tuple(key)


def compare_rows(expected, actual_header, actual_rows, relative, absolute):
    """Compare multisets, retaining duplicates and allowing ORDER BY ties to reorder.

    Exact columns usually identify a unique TPC-H output row. Within ambiguous groups,
    a bipartite matching finds a one-to-one pairing of tolerance-compatible rows. A
    greedy pairing alone could incorrectly reject overlapping numeric tolerances.
    """
    ordered = compare_ordered_rows(
        expected, actual_header, actual_rows, relative, absolute
    )
    ordered.update(
        {"comparison_mode": "duplicate-preserving multiset", "order_verified": False}
    )
    if ordered["match"] or len(expected["rows"]) != len(actual_rows):
        return ordered
    if any(len(row) != len(expected["columns"]) for row in actual_rows):
        return ordered
    oracle_groups, actual_groups = defaultdict(list), defaultdict(list)
    for index, row in enumerate(expected["rows"]):
        oracle_groups[comparison_key(row, expected["types"], oracle=True)].append(index)
    for index, row in enumerate(actual_rows):
        actual_groups[comparison_key(row, expected["types"], oracle=False)].append(
            index
        )
    if set(oracle_groups) != set(actual_groups) or any(
        len(indices) != len(actual_groups[key])
        for key, indices in oracle_groups.items()
    ):
        ordered["differences"] = [
            "Row multiset differs in exact text/integer/Boolean/NULL values or duplicate counts."
        ] + ordered["differences"][:5]
        return ordered
    paired_rows = [None] * len(actual_rows)
    for key, oracle_indices in oracle_groups.items():
        candidates = actual_groups[key]
        compatible = {}
        for oracle_index in oracle_indices:
            single = {**expected, "rows": [expected["rows"][oracle_index]]}
            compatible[oracle_index] = [
                actual_index
                for actual_index in candidates
                if compare_ordered_rows(
                    single,
                    actual_header,
                    [actual_rows[actual_index]],
                    relative,
                    absolute,
                )["match"]
            ]
        # Augment the current matching with alternating paths, preserving duplicates.
        owner = {}
        for start in oracle_indices:
            pending = deque([start])
            parents = {start: None}
            visited_actual = set()
            endpoint = None
            while pending and endpoint is None:
                oracle_index = pending.popleft()
                for actual_index in compatible[oracle_index]:
                    if actual_index in visited_actual:
                        continue
                    visited_actual.add(actual_index)
                    if actual_index not in owner:
                        endpoint = (oracle_index, actual_index)
                        break
                    displaced = owner[actual_index]
                    if displaced not in parents:
                        parents[displaced] = (oracle_index, actual_index)
                        pending.append(displaced)
            if endpoint is None:
                ordered["differences"] = [
                    "No duplicate-preserving row pairing satisfies the numeric tolerance."
                ] + ordered["differences"][:5]
                return ordered
            oracle_index, actual_index = endpoint
            while True:
                owner[actual_index] = oracle_index
                previous = parents[oracle_index]
                if previous is None:
                    break
                oracle_index, actual_index = previous
        for actual_index, oracle_index in owner.items():
            paired_rows[oracle_index] = actual_rows[actual_index]
    matched = compare_ordered_rows(
        expected, actual_header, paired_rows, relative, absolute
    )
    matched.update(
        {"comparison_mode": "duplicate-preserving multiset", "order_verified": False}
    )
    return matched


def source_identity(path):
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    return {
        "path": str(path.resolve()),
        "git_head": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_changes": (
            status.stdout.splitlines() if status.returncode == 0 else None
        ),
    }


def stat_identity(path):
    status = path.stat()
    return {
        "device": status.st_dev,
        "inode": status.st_ino,
        "size": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
    }


def file_identity(path, cached=None):
    before = stat_identity(path)
    if (
        cached
        and cached.get("stat") == before
        and re.fullmatch(r"[0-9a-f]{64}", cached.get("sha256", ""))
    ):
        return {
            "path": str(path),
            "bytes": before["size"],
            "sha256": cached["sha256"],
            "stat": before,
        }
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if stat_identity(path) != before:
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {
        "path": str(path),
        "bytes": before["size"],
        "sha256": digest,
        "stat": before,
    }


def discover_inputs(data_dir, cache_path, generation_manifest):
    """Reuse hashes only when the inode and all change-sensitive stat fields match."""
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if generation_manifest.exists():
        generated = json.loads(generation_manifest.read_text())
        for table in generated.get("tables", {}).values():
            for item in table.get("files", []):
                path = (data_dir / item["path"]).resolve()
                cached.setdefault(str(path), item)
    files, inputs, updated = {}, {}, {}
    for table in TABLES:
        single = data_dir / f"{table}.parquet"
        shards = sorted((data_dir / table).glob("*.parquet"))
        if single.is_file() and shards:
            raise RuntimeError(
                f"Ambiguous input: both {single} and {table}/*.parquet exist"
            )
        paths = [single] if single.is_file() else shards
        if not paths:
            raise RuntimeError(f"Missing {single} or {data_dir / table}/*.parquet")
        files[table] = single if single.is_file() else data_dir / table / "*.parquet"
        identities = []
        for path in paths:
            print(f"Checking input {path}", flush=True)
            identity = file_identity(path, cached.get(str(path)))
            identities.append(identity)
            updated[str(path)] = identity
        inputs[table] = {"pattern": str(files[table]), "files": identities}
    write_json(cache_path, updated)
    return files, inputs


def verify_inputs(inputs):
    for table, source in inputs.items():
        pattern = Path(source["pattern"])
        paths = sorted(pattern.parent.glob(pattern.name))
        recorded = [Path(item["path"]) for item in source["files"]]
        if paths != recorded or any(
            stat_identity(Path(item["path"])) != item["stat"]
            for item in source["files"]
        ):
            raise RuntimeError(
                f"Input {table} changed since its manifest was recorded; refusing stale CPU references"
            )


def topology(client, run, timeout, expected_cns=1, prefix="topology"):
    counts = {}
    for label, statement in (
        ("compute-nodes", "SHOW COMPUTE NODES;"),
        ("backends", "SHOW BACKENDS;"),
    ):
        stdout, stderr = run / f"{prefix}-{label}.tsv", run / f"{prefix}-{label}.stderr"
        result = run_process(client, stdout, stderr, min(timeout, 10), sql=statement)
        if result["status"] != "OK":
            raise RuntimeError(f"{statement} failed; see {stderr}")
        header, rows = read_mysql(stdout)
        if not header and not rows and label == "backends":
            counts[label] = 0
            continue
        columns = {name.lower(): index for index, name in enumerate(header)}
        if "alive" not in columns:
            raise RuntimeError(f"Missing Alive column in {stdout}")
        counts[label] = sum(row[columns["alive"]].lower() == "true" for row in rows)
    if counts != {"compute-nodes": expected_cns, "backends": 0}:
        raise RuntimeError(
            f"Expected exactly {expected_cns} alive CN(s) and no alive BEs, found {counts}"
        )
    return counts


def write_report(run, manifest, results, prepared=False):
    write_json(run / "results.json", {"manifest": manifest, "results": results})
    passed = sum(result["status"] == "PASS" for result in results)
    lines = [
        f"# TPC-H through StarRocks with {manifest['expected_cns']} Sirius CN(s)",
        "",
        (
            f"Prepared {len(results)} CPU references; StarRocks was not contacted."
            if prepared
            else f"**{passed}/{len(results)} queries passed the CPU comparison.**"
        ),
        "",
        f"Query source: `{QUERY_SOURCE.relative_to(ROOT)}` (SHA256 `{manifest['query_source_sha256']}`).",
        "Query parameters come from that file, with Q11's fraction scaled as described below. FILES/read_parquet CTE preludes are added.",
        f"TPC-H scale factor for query parameters: {manifest['scale_factor']}.",
        "SQL syntax adaptations: "
        + (
            " ".join(
                f"{name.upper()}: {description}"
                for name, description in manifest["sql_adaptations"].items()
            )
            or "none."
        ),
        f"Data: `{manifest['data_dir']}`; single-file or sharded Parquet tables. No tables or databases are created.",
        f"Session: `{manifest['session_sql']}`",
        "",
        f"Engine source: `{manifest['engine_source']['path']}` at `{manifest['engine_source']['git_head']}`.",
        f"Numeric comparison uses the unrounded output: relative tolerance {manifest['relative_tolerance']}, "
        f"absolute tolerance {manifest['absolute_tolerance']} (the larger bound applies). "
        "Integer/count values, text, and NULLs must match exactly. "
        "Rows are compared as multisets, retaining duplicates and allowing ORDER BY ties to change position. "
        "Output ordering itself is not verified.",
        "Each query runs once. Timings include MySQL client startup and transfer; this is a correctness run, "
        "not a TPC-compliant performance benchmark.",
        "",
        "| Query | Status | StarRocks seconds | Actual / expected rows | Detail |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for result in results:
        timing = result.get("starrocks", {}).get("elapsed_seconds")
        comparison = result.get("comparison", {})
        rows = (
            f"{comparison.get('actual_rows', '-')} / {result.get('oracle_rows', '-')}"
        )
        detail = result.get("detail", "").replace("|", "\\|").replace("\n", " ")[:220]
        lines.append(
            f"| {result['query'].upper()} | {result['status']} | "
            f"{timing:.3f}" + f" | {rows} | {detail} |"
            if timing is not None
            else f"| {result['query'].upper()} | {result['status']} | - | {rows} | {detail} |"
        )
    lines.extend(
        [
            "",
            "Each query directory contains the SQL, EXPLAIN output, original MySQL output/stderr, "
            "typed DuckDB reference, comparison, and timing/status JSON. `manifest.json` records "
            "input file hashes and client settings.",
            "",
        ]
    )
    (run / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "test/cpp/integration/data/parquet"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "build/tpch-starrocks-1cn"
    )
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=ROOT,
        help="Engine checkout recorded in the report",
    )
    parser.add_argument(
        "--cn-binary", type=Path, help="Optional actual CN binary to hash and record"
    )
    parser.add_argument(
        "--mysql", type=Path, default=Path(os.environ.get("MYSQL_BIN", MYSQL))
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9030)
    parser.add_argument("--user", default="root")
    parser.add_argument("--expected-cns", type=int, default=1)
    parser.add_argument("--scale-factor", type=Decimal, default=Decimal(1))
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Per-query MySQL client deadline in seconds (3..3600)",
    )
    parser.add_argument(
        "--oracle-timeout",
        type=float,
        help="CPU reference deadline; defaults to --timeout",
    )
    parser.add_argument("--oracle-memory-limit", default="4GB")
    parser.add_argument("--oracle-threads", type=int, default=4)
    parser.add_argument(
        "--explain-costs",
        action="store_true",
        help="Save FE cost estimates before timing",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Generation manifest with stat-validated file hashes; defaults to DATA/generation-manifest.json",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after an execution failure so the operator can restart the cluster",
    )
    parser.add_argument("--relative-tolerance", type=Decimal, default=Decimal("1e-6"))
    parser.add_argument("--absolute-tolerance", type=Decimal, default=Decimal("1e-8"))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Additional session setting",
    )
    parser.add_argument(
        "--queries", nargs="+", default=[f"q{i:02}" for i in range(1, 23)]
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare SQL and CPU references; do not contact FE",
    )
    parser.add_argument(
        "--reuse-oracle",
        action="store_true",
        help="Reuse successful references only if SQL and input hashes match",
    )
    parser.add_argument("--oracle-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.oracle_threads < 1 or not re.fullmatch(
        r"[0-9]+(?:\.[0-9]+)?(?:KB|MB|GB|TB|KiB|MiB|GiB|TiB)", args.oracle_memory_limit
    ):
        parser.error(
            "Use positive --oracle-threads and a memory size such as --oracle-memory-limit 8GB"
        )
    if args.oracle_worker:
        oracle_worker(args.oracle_worker, args.oracle_memory_limit, args.oracle_threads)
        return 0
    if not 3 <= args.timeout <= 3600:
        parser.error("--timeout must be between 3 and 3600 seconds")
    if (
        args.expected_cns < 1
        or not args.scale_factor.is_finite()
        or args.scale_factor <= 0
    ):
        parser.error("--expected-cns and --scale-factor must be positive")
    args.oracle_timeout = args.oracle_timeout or args.timeout
    if not 3 <= args.oracle_timeout <= 3600:
        parser.error("--oracle-timeout must be between 3 and 3600 seconds")
    if any(
        not value.is_finite() or value < 0
        for value in (args.relative_tolerance, args.absolute_tolerance)
    ):
        parser.error("comparison tolerances must be nonnegative")
    for setting in args.set:
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*=[a-zA-Z_0-9.+-]+", setting):
            parser.error(f"Invalid --set {setting!r}; expected NAME=VALUE")
    query_names = []
    for name in args.queries:
        match = re.fullmatch(r"q?0*([1-9]|1[0-9]|2[0-2])", name.lower())
        if not match:
            parser.error(f"Invalid query name: {name}")
        normalized = f"q{int(match[1]):02}"
        if normalized not in query_names:
            query_names.append(normalized)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir.resolve()
    files, inputs = discover_inputs(
        data_dir,
        run / "input-hashes.json",
        (args.input_manifest or data_dir / "generation-manifest.json").resolve(),
    )
    queries = runpy.run_path(str(QUERY_SOURCE))["QUERIES"]
    settings = [f"query_timeout={int(args.timeout) - 2}", "pipeline_dop=1"] + args.set
    session = " ".join(f"SET {setting};" for setting in settings)
    client = [
        str(args.mysql),
        "--no-defaults",
        "--protocol=TCP",
        f"--host={args.host}",
        f"--port={args.port}",
        f"--user={args.user}",
        "--connect-timeout=3",
        "--ssl-mode=DISABLED",
        "--batch",
        "--column-names",
        "--default-character-set=utf8mb4",
    ]
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "expected_cns": args.expected_cns,
        "scale_factor": str(args.scale_factor),
        "inputs": inputs,
        "query_source_sha256": hashlib.sha256(QUERY_SOURCE.read_bytes()).hexdigest(),
        "sql_adaptations": {
            name: SQL_ADAPTATIONS[name]
            for name in query_names
            if name in SQL_ADAPTATIONS
        },
        "query_names": query_names,
        "session_sql": session,
        "mysql_command": client,
        "relative_tolerance": str(args.relative_tolerance),
        "absolute_tolerance": str(args.absolute_tolerance),
        "timeout_seconds": args.timeout,
        "oracle_timeout_seconds": args.oracle_timeout,
        "oracle_memory_limit": args.oracle_memory_limit,
        "oracle_threads": args.oracle_threads,
        "prepare_only": args.prepare_only,
        "explain_costs": args.explain_costs,
        "engine_source": source_identity(args.engine_root),
        "runner_source": source_identity(ROOT),
        "cn_binary": (
            file_identity(args.cn_binary.resolve()) if args.cn_binary else None
        ),
    }
    write_json(run / "manifest.json", manifest)
    if not args.prepare_only:
        manifest["topology"] = topology(client, run, args.timeout, args.expected_cns)
        write_json(run / "manifest.json", manifest)
    results = []
    for name in query_names:
        verify_inputs(inputs)
        query_dir = run / name
        query_dir.mkdir(exist_ok=True)
        source_query = queries[f"q{int(name[1:])}"]
        (query_dir / "source.sql").write_text(source_query.strip() + ";\n")
        query = adapt_query(name, source_query, args.scale_factor)
        starrocks_sql = parquet_ctes(query, files, True)
        cpu_sql = parquet_ctes(query, files, False)
        (query_dir / "starrocks.sql").write_text(session + "\n" + starrocks_sql)
        (query_dir / "duckdb.sql").write_text(cpu_sql)
        fingerprint = hashlib.sha256(
            (cpu_sql + json.dumps(inputs, sort_keys=True)).encode()
        ).hexdigest()
        oracle_path = query_dir / "oracle.json"
        fingerprint_path = query_dir / "oracle-fingerprint.txt"
        reusable = (
            args.reuse_oracle
            and oracle_path.exists()
            and fingerprint_path.exists()
            and fingerprint_path.read_text() == fingerprint
            and json.loads(oracle_path.read_text()).get("status") == "OK"
        )
        result = {"query": name, "status": "PREPARED"}
        print(
            f"{name.upper()}: {'Reusing' if reusable else 'Running'} DuckDB reference",
            flush=True,
        )
        write_json(
            run / "progress.json",
            {
                "query": name,
                "phase": "oracle",
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
        if not reusable:
            # A worker per query enforces the same bound for CPU oracle errors/hangs.
            oracle_result = run_process(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--oracle-worker",
                    str(query_dir),
                    "--oracle-memory-limit",
                    args.oracle_memory_limit,
                    "--oracle-threads",
                    str(args.oracle_threads),
                ],
                query_dir / "oracle.stdout",
                query_dir / "oracle.stderr",
                args.oracle_timeout,
                env=os.environ | {"SIRIUS_DISABLE": "1", "CUDA_VISIBLE_DEVICES": ""},
            )
            result["oracle_process"] = oracle_result
            if oracle_result["status"] != "OK":
                result["status"] = "ORACLE_" + oracle_result["status"]
                result["detail"] = (query_dir / "oracle.stderr").read_text()[-1500:]
                write_json(oracle_path, {"status": result["status"]})
            else:
                fingerprint_path.write_text(fingerprint)
        expected = json.loads(oracle_path.read_text())
        if expected["status"] == "OK":
            result["oracle_rows"] = len(expected["rows"])
            result["oracle_seconds"] = expected["elapsed_seconds"]
        else:
            result["status"] = expected["status"]
        if not args.prepare_only:
            try:
                verify_inputs(inputs)
                result["topology_before"] = topology(
                    client, query_dir, args.timeout, args.expected_cns
                )
                explain_sql = (
                    session
                    + ("\nEXPLAIN COSTS " if args.explain_costs else "\nEXPLAIN ")
                    + starrocks_sql
                )
                (query_dir / "explain.sql").write_text(explain_sql)
                result["explain"] = run_process(
                    client,
                    query_dir / "explain.tsv",
                    query_dir / "explain.stderr",
                    min(args.timeout, 20),
                    sql=explain_sql,
                )
                if args.explain_costs and result["explain"]["status"] != "OK":
                    raise RuntimeError(
                        "EXPLAIN COSTS failed; restart before measurement: "
                        + (query_dir / "explain.stderr").read_text()[-1000:]
                    )
                print(
                    f"{name.upper()}: Running StarRocks on {args.expected_cns} CN(s)",
                    flush=True,
                )
                write_json(
                    run / "progress.json",
                    {
                        "query": name,
                        "phase": "starrocks",
                        "started_utc": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    },
                )
                result["starrocks"] = run_process(
                    client,
                    query_dir / "starrocks.tsv",
                    query_dir / "starrocks.stderr",
                    args.timeout,
                    sql=session + "\n" + starrocks_sql,
                )
                if result["starrocks"]["status"] != "OK":
                    result["status"] = "STARROCKS_" + result["starrocks"]["status"]
                    result["detail"] = (query_dir / "starrocks.stderr").read_text()[
                        -1500:
                    ]
                elif expected["status"] == "OK":
                    header, actual = read_mysql(query_dir / "starrocks.tsv")
                    comparison = compare_rows(
                        expected,
                        header,
                        actual,
                        args.relative_tolerance,
                        args.absolute_tolerance,
                    )
                    result["comparison"] = comparison
                    write_json(query_dir / "comparison.json", comparison)
                    result["status"] = "PASS" if comparison["match"] else "MISMATCH"
                    result["detail"] = "; ".join(comparison["differences"][:2])
                result["topology_after"] = topology(
                    client,
                    query_dir,
                    args.timeout,
                    args.expected_cns,
                    prefix="topology-after",
                )
            except (OSError, RuntimeError, ValueError) as error:
                result["status"] = "RUNNER_ERROR"
                result["detail"] = str(error)
        results.append(result)
        write_json(query_dir / "status.json", result)
        write_report(run, manifest, results, args.prepare_only)
        print(
            f"{name.upper()}: {result['status']}"
            + (
                f" ({result['starrocks']['elapsed_seconds']:.3f}s)"
                if "starrocks" in result
                else ""
            )
            + (
                f" — {result['detail'][:180].replace(chr(10), ' ')}"
                if result.get("detail")
                else ""
            ),
            flush=True,
        )
        if args.stop_on_error and result["status"] not in (
            "PREPARED",
            "PASS",
            "MISMATCH",
        ):
            print(
                "Stopping after an execution failure; restart the cluster before continuing if fragments remain active.",
                flush=True,
            )
            break
    write_json(
        run / "progress.json",
        {
            "phase": "complete",
            "completed_queries": len(results),
            "requested_queries": len(query_names),
        },
    )
    print(f"Report: {run / 'report.md'}", flush=True)
    return (
        0
        if all(
            result["status"] == ("PREPARED" if args.prepare_only else "PASS")
            for result in results
        )
        else 1
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(str(error))
