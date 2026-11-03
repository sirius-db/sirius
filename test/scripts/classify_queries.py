#!/usr/bin/env python3
"""Classify TPC-H, TPC-Havoc and TPC-DS queries on Sirius with CPU fallback disabled.

Run through pixi; GPU access must be available to the process. No data generation.
Each query gets a fresh CLI process so a runtime failure cannot poison the next query.

Uses existing Parquet data or a DuckDB database; no database import or data
generation is needed. Run commands from the repository root. The CLI
(`--duckdb`, by default `sirius-duckdb/build/release/duckdb`) does not include
Sirius: GPU runs start it with `-unsigned` and `LOAD` the extension
(`--extension`, by default `$SIRIUS_EXTENSION_PATH` or the wrapper build's
`sirius.duckdb_extension`). Plan inventories and CPU reference runs use plain
DuckDB. Every run passes `-no-agent`, so the CLI's errors stay the same when
`AI_AGENT` or `CLAUDECODE` is set.

For a classification run:

```bash
pixi run python test/scripts/classify_queries.py \
  $PATH_TO_TPC_DS_TABLES \
  --config test/scripts/classification.yaml \
  --output-dir ...
```

`--benchmark` picks the tables and the queries:

- `tpcds` (the default): the canonical 99 SQL files from the local DuckDB submodule
  (`duckdb/extension/tpcds/dsdgen/queries/01.sql` through `99.sql`).
- `tpch`: the 22 SQL files in `duckdb/extension/tpch/dbgen/queries`.
- `tpch-havoc`: the TPC-Havoc rewrites of the TPC-H queries, which CMake downloads
  into `build/release/test/tpch_havoc` (see test/README.md; `--havoc-dir` reads
  another copy). It needs `--scale-factor`, which sets Q11's threshold the way
  BenchBox does. Identical rewrites run once, under the first label, such as
  `Q19_V01`.

Use `--query-dir` to select a different corpus for `tpch` or `tpcds`; it accepts
`q1.sql`, `q01.sql` or `01.sql` naming, and queries keep their file numbers.

To read the tables from a DuckDB database instead of Parquet, pass `--duckdb-file`
in place of the directory; the database is attached read-only. For example, to
classify TPC-Havoc against the C++ tests' database and validate the results:

```bash
pixi run python test/scripts/classify_queries.py --benchmark tpch-havoc \
  --duckdb-file test/cpp/integration/data/duckdb/integration.duckdb \
  --scale-factor 0.1 --validate \
  --config test/scripts/classification.yaml \
  --output-dir ...
```

The output directory must be new. Add `--queries 3 7 42` for a subset (with
`tpch-havoc`, base query numbers or labels such as `Q05_V03`) or `--timeout 60` to
limit each process to 60 seconds. Table inputs can be named
`table.parquet`, `table/*.parquet`, or `table_[0-9]*.parquet` shards. The shard
suffix must start with a digit. This avoids mistaking `store_returns` for a `store` shard or
`customer_address` for a `customer` shard. The example config uses
one GPU, 8 GiB of GPU memory and 4 GiB of pinned memory per host NUMA node.
It is a small baseline for SF1, not a memory recommendation for larger scales.

For each query, it saves:

- `plans.json`: DuckDB's unoptimized logical, optimized logical, and CPU physical
  trees, using Sirius's disabled optimizer mask.
- `plans/`: the exact EXPLAIN input, stdout and stderr.
- `gpu/`: the exact execution input, CSV results, stderr and Sirius logs.
- `classification.csv`: completed GPU execution, plan rejection, runtime failure,
  crash, timeout, or an unconfirmed/unclassified outcome, with the first exact
  error. With `tpch-havoc`, `duplicates` lists the labels of identical rewrites.
- `features.json`: query membership for plan operators, join types and function
  names visible in the EXPLAIN trees. This is an inventory, not a support verdict.
- `metadata.json` and `sirius.yaml`: source revision, dirty status, CLI and
  extension hashes, input location, optimizer mask, timeout and configuration.
- `summary.md`: counts by outcome (per base query for `tpch-havoc`) and the first
  error of each root cause, ready to paste into an issue.

GPU execution uses normal SQL with `SET enable_duckdb_fallback=false`. A fresh
process per query isolates runtime failures. Success requires both a zero exit
code and the transparent GPU completion log; timers alone do not establish GPU
coverage. Add `--validate` to compare each completed GPU query against a fresh
CPU DuckDB reference (`SIRIUS_DISABLE=1`). CPU execution uses its normal optimizer
settings. Rejected or failed GPU queries are marked `not_applicable` for validation.
Execution status stays separate from correctness status.

Validation compares row multisets, preserving duplicate multiplicities while
ignoring order, as the TPC-H runner does. FLOAT/DOUBLE values use absolute tolerance
1e-10 (override with `--float-tolerance`); integers, decimals, strings and NULLs
compare exactly. JSON output preserves NULLs and avoids truncated display tables.
This checks result contents, not ORDER BY behavior; LIMIT queries with unresolved
ties can select different valid rows and require inspection of reported mismatches.

With `--validate`, `gpu/stdout.txt` contains JSON instead of CSV. Each completed
query also has `cpu/` with its reference schema, results and diagnostics. Verdicts
(`pass`, `mismatch`, `error`, `cpu_error`, `cpu_timeout`, `not_applicable`) are saved
in `validation.csv`, `qN/validation.json`, and extra columns in `classification.csv`.
Without `--validate`, results remain CSV and no CPU reference queries are run.

Sirius can fall back at plan time and at runtime when fallback is enabled.
An error from a missing operator/expression is a capability gap; scan binding
errors, execution defects, resource limits and timeouts belong in separate
categories. A timeout is unresolved, not proof that a feature is unsupported.
Repeat classification at other scale factors when needed: statistics and types
can change optimized plan shapes. A run identifies behavior of the recorded
binary; the current source revision does not prove that binary was built from it.
"""

import argparse
import collections
import csv
import datetime
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]
# The wrapper build: a DuckDB CLI without Sirius, and the Sirius extension.
WRAPPER = ROOT / "sirius-duckdb/build/release"
TABLES = {
    "tpcds": """call_center catalog_page catalog_returns catalog_sales customer
customer_address customer_demographics date_dim household_demographics income_band
inventory item promotion reason ship_mode store store_returns store_sales time_dim
warehouse web_page web_returns web_sales web_site""".split(),
    "tpch": "customer lineitem nation orders part partsupp region supplier".split(),
}
# Default query directory and query count of the benchmarks with numbered SQL files.
QUERY_FILES = {
    "tpcds": (ROOT / "duckdb/extension/tpcds/dsdgen/queries", 99),
    "tpch": (ROOT / "duckdb/extension/tpch/dbgen/queries", 22),
}
MASK = "in_clause,compressed_materialization,late_materialization"
DEFAULT_FLOAT_TOLERANCE = 1e-10
# summary.md outcomes, in column order.
OUTCOMES = {
    "gpu": "GPU",
    "mismatch": "mismatch",
    "plan_rejected": "plan rejected",
    "runtime_failed": "runtime failed",
    "crash": "crash",
    "timeout": "timeout",
    "other": "other",
}
# `id` is the `query` value in the CSV files; `label` names the query in output.
Query = collections.namedtuple("Query", "id label directory base sql duplicates")


def json_results(output):
    """Read consecutive CLI JSON result sets without losing NULLs or decimals."""

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


def compare_result_rows(schema, expected, actual, float_tolerance):
    """Compare row multisets: exact non-floats and tolerance-aware float matching."""
    names = tuple(column["column_name"] for column in schema)
    types = [column["column_type"].upper() for column in schema]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate column names in reference schema")
    floats = [i for i, typ in enumerate(types) if typ in {"FLOAT", "DOUBLE", "REAL"}]

    def normalize(rows):
        normalized = []
        for row in rows:
            if tuple(row) != names:
                raise ValueError(f"Result columns {tuple(row)!r} differ from {names!r}")
            values = list(row.values())
            for i, typ in enumerate(types):
                if values[i] is not None:
                    if i in floats:
                        values[i] = float(values[i])
                    elif typ.startswith("DECIMAL("):
                        values[i] = Decimal(str(values[i]))
            normalized.append(tuple(values))
        return normalized

    expected, actual = normalize(expected), normalize(actual)
    if len(expected) != len(actual):
        return False, f"Row count mismatch: CPU={len(expected)}, GPU={len(actual)}"
    # Multiplicities matter, including duplicate rows.
    cpu, gpu = collections.Counter(expected), collections.Counter(actual)
    if cpu == gpu:
        return True, "Exact row multiset match"
    if not floats:
        missing = next((cpu - gpu).elements())
        return False, f"No matching GPU row for CPU row {missing!r}"

    def matches(lhs, rhs):
        for i, (a, b) in enumerate(zip(lhs, rhs)):
            if a == b:
                continue
            if i not in floats or a is None or b is None:
                return False
            if not (
                (math.isnan(a) and math.isnan(b))
                or math.isclose(a, b, rel_tol=0.0, abs_tol=float_tolerance)
            ):
                return False
        return True

    # A maximum bipartite matching avoids false mismatches from sorting approximate
    # floats and prevents one GPU row from satisfying multiple CPU duplicate rows.
    edges = [
        [j for j, rhs in enumerate(actual) if matches(lhs, rhs)] for lhs in expected
    ]
    assigned = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or assign(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(expected)):
        if not assign(i, set()):
            return (
                False,
                f"No matching GPU row for CPU row {expected[i]!r} (abs_tol={float_tolerance:g})",
            )
    return True, f"Row multiset match within float abs_tol={float_tolerance:g}"


def validate_result(binary, setup, sql, env, directory, timeout, gpu_output, tolerance):
    code, stdout, stderr = run_cli(
        binary,
        setup + ".mode json\nDESCRIBE " + sql + sql,
        env,
        directory,
        timeout,
    )
    if code == "timeout":
        return "cpu_timeout", f"CPU reference exceeded {timeout:g}s"
    if code != 0:
        return "cpu_error", stderr.strip()
    try:
        cpu = json_results(stdout)
        gpu = json_results(gpu_output)
        if len(cpu) != 2 or len(gpu) != 1:
            raise ValueError("Expected CPU schema plus one CPU/GPU query result set")
        match, detail = compare_result_rows(cpu[0], cpu[1], gpu[0], tolerance)
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        return "error", f"Cannot compare results: {exc}"
    return "pass" if match else "mismatch", detail


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def walk(nodes):
    for node in nodes:
        yield node
        yield from walk(node.get("children", []))


def explain_arrays(output):
    # DuckDB's shell renders EXPLAIN JSON specially, even with -json.
    decoder = json.JSONDecoder()
    arrays = []
    for match in re.finditer(r"^\[", output, re.MULTILINE):
        try:
            value, _ = decoder.raw_decode(output[match.start() :])
        except ValueError:
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            arrays.append(value)
    if len(arrays) != 3:
        raise ValueError(f"Expected three EXPLAIN trees, found {len(arrays)}")
    return dict(zip(("logical", "logical_optimized", "physical"), arrays))


def run_cli(binary, sql, env, directory, timeout, extension=None):
    if extension is not None:
        sql = f"LOAD {quote(extension)};\n" + sql
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.sql").write_text(sql)
    try:
        result = subprocess.run(
            [str(binary), "-unsigned", "-bail", "-no-agent"],
            input=sql,
            text=True,
            capture_output=True,
            env=env,
            cwd=ROOT,
            timeout=timeout,
        )
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout, stderr, code = exc.stdout or b"", exc.stderr or b"", "timeout"
        stdout = (
            stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        )
        stderr = (
            stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        )
    (directory / "stdout.txt").write_text(stdout)
    (directory / "stderr.txt").write_text(stderr)
    return code, stdout, stderr


def numbered_queries(query_dir, numbers):
    queries = []
    for q in numbers:
        names = (f"q{q}.sql", f"q{q:02}.sql", f"{q:02}.sql")
        path = next((query_dir / n for n in names if (query_dir / n).exists()), None)
        if path is None:
            raise ValueError(f"No SQL file for query {q} in {query_dir}")
        sql = path.read_text().strip().rstrip(";") + ";\n"
        queries.append(Query(q, f"Q{q}", f"q{q}", q, sql, []))
    return queries


def havoc_queries(havoc_dir, scale_factor):
    """Load the TPC-Havoc variants; identical later ones become duplicates."""
    # BenchBox renders Q11's threshold as f"{0.0001 / SF:.10f}".
    fraction = f"{0.0001 / scale_factor:.10f}"
    queries, first = [], {}
    for base in range(1, 23):
        document = yaml.safe_load((havoc_dir / f"q{base:02}.yaml").read_text())
        for variant in document["variants"]:
            label = f"Q{base:02}_V{int(variant['id']):02}"
            sql = variant["sql"].strip().replace("{q11_fraction}", fraction)
            if match := re.search(r"\{[A-Za-z_]\w*\}", sql):
                raise ValueError(
                    f"{label} has an unknown template parameter {match[0]}"
                )
            sql = sql.rstrip(";") + ";\n"
            if sql in first:
                first[sql].duplicates.append(label)
            else:
                first[sql] = Query(label, label, label, base, sql, [])
                queries.append(first[sql])
    return queries


def select_queries(queries, names):
    """Queries named by base query number or label; a duplicate names its query."""
    wanted = {str(int(name)) if name.isdigit() else name.upper() for name in names}
    keys = {q.id: {str(q.base), q.label, *q.duplicates} for q in queries}
    unknown = wanted.difference(*keys.values())
    if unknown:
        raise ValueError(f"Unknown queries: {' '.join(sorted(unknown))}")
    return [q for q in queries if wanted & keys[q.id]]


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def outcome(row):
    """Summary outcome; with validation, a completed query must also match CPU."""
    if row["status"] == "gpu_completed":
        validation = row.get("validation_status", "pass")
        return {"pass": "gpu", "mismatch": "mismatch"}.get(validation, "other")
    return row["status"] if row["status"] in OUTCOMES else "other"


def first_error(row):
    """A row's first error, without the prefix its whole outcome shares."""
    failed = row["status"] != "gpu_completed"
    text = (row["reason"] if failed else row["validation_reason"]).split(" | ")[0]
    text = re.sub(r"^.*?(GPU plan generation|Sirius GPU execution) failed: ", "", text)
    text = re.sub(r"CUDF failure at: \S*/", "cuDF failure at ", text)
    return text.strip()[:200] or row["status"]


def summary(results, context, per_base_query):
    """Markdown counts by outcome, then the first error of each root cause."""
    lines = [context, ""]
    variants = len(results) + sum(len(query.duplicates) for query, _ in results)
    if variants != len(results):
        lines += [
            f"Counts are distinct queries: {len(results)} queries cover "
            f"{variants} variants.",
            "",
        ]
    groups = [("all", results)]
    if per_base_query:
        bases = sorted({query.base for query, _ in results})
        groups = [
            (f"Q{base:02}", [r for r in results if r[0].base == base]) for base in bases
        ] + [("**all**", results)]
    lines.append("| query | " + " | ".join(OUTCOMES.values()) + " |")
    lines.append("|---" * (len(OUTCOMES) + 1) + "|")
    for name, members in groups:
        counts = collections.Counter(outcome(row) for _, row in members)
        cells = " | ".join(str(counts[kind]) for kind in OUTCOMES)
        lines.append(f"| {name} | {cells} |")
    # Errors that differ only in numbers, such as sizes or line numbers, share a cause.
    causes = collections.defaultdict(list)
    for query, row in results:
        if (kind := outcome(row)) != "gpu":
            error = first_error(row)
            causes[(kind, re.sub(r"\d+", "N", error))].append((query.label, error))
    if causes:
        lines += ["", "| outcome | # | queries | first error |", "|---|---|---|---|"]
        order = list(OUTCOMES)
        for (kind, _), members in sorted(
            causes.items(), key=lambda cause: (order.index(cause[0][0]), -len(cause[1]))
        ):
            labels = " ".join(label for label, _ in members)
            error = members[0][1].replace("`", "'").replace("|", "\\|")
            lines.append(
                f"| {OUTCOMES[kind]} | {len(members)} | {labels} | `{error}` |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_dir", type=Path, nargs="?")
    parser.add_argument(
        "--duckdb-file",
        type=Path,
        help="Read the tables from this DuckDB database instead of Parquet",
    )
    parser.add_argument(
        "--benchmark", choices=["tpcds", "tpch", "tpch-havoc"], default="tpcds"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Query numbers; for tpch-havoc, also labels such as Q05_V03",
    )
    parser.add_argument("--query-dir", type=Path)
    parser.add_argument(
        "--havoc-dir", type=Path, default=ROOT / "build/release/test/tpch_havoc"
    )
    parser.add_argument(
        "--scale-factor", type=float, help="Required for tpch-havoc (Q11's threshold)"
    )
    parser.add_argument("--duckdb", type=Path, default=WRAPPER / "duckdb")
    parser.add_argument(
        "--extension",
        type=Path,
        default=os.environ.get("SIRIUS_EXTENSION_PATH")
        or WRAPPER / "extension/sirius/sirius.duckdb_extension",
    )
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Compare completed GPU queries against CPU DuckDB results",
    )
    parser.add_argument(
        "--float-tolerance",
        type=float,
        default=DEFAULT_FLOAT_TOLERANCE,
        help="Absolute FLOAT/DOUBLE tolerance for validation (default: 1e-10)",
    )
    args = parser.parse_args()
    if not math.isfinite(args.float_tolerance) or args.float_tolerance < 0:
        parser.error("--float-tolerance must be finite and non-negative")
    if (args.parquet_dir is None) == (args.duckdb_file is None):
        parser.error("pass either a Parquet directory or --duckdb-file")
    havoc = args.benchmark == "tpch-havoc"
    if havoc and args.scale_factor is None:
        parser.error("--benchmark tpch-havoc needs --scale-factor")
    if args.scale_factor is not None and not args.scale_factor > 0:
        parser.error("--scale-factor must be positive")
    if havoc and args.query_dir:
        parser.error("--query-dir does not apply to tpch-havoc; use --havoc-dir")
    binary, extension = args.duckdb.resolve(), args.extension.resolve()
    for path in (binary, extension):
        if not path.is_file():
            parser.error(f"Not found: {path}")
    if havoc:
        query_dir = args.havoc_dir.resolve()
        if not (query_dir / "q01.yaml").is_file():
            parser.error(
                f"No TPC-Havoc queries in {query_dir}: configure a build with tests "
                "enabled, which downloads them, or pass --havoc-dir"
            )
    else:
        query_dir = (args.query_dir or QUERY_FILES[args.benchmark][0]).resolve()
    try:
        if havoc:
            queries = havoc_queries(query_dir, args.scale_factor)
            if args.queries:
                queries = select_queries(queries, args.queries)
        else:
            if args.queries and not all(name.isdigit() for name in args.queries):
                raise ValueError(f"--queries takes numbers for {args.benchmark}")
            numbers = (
                [int(name) for name in args.queries]
                if args.queries
                else range(1, QUERY_FILES[args.benchmark][1] + 1)
            )
            queries = numbered_queries(query_dir, numbers)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    family = "tpcds" if args.benchmark == "tpcds" else "tpch"
    if args.duckdb_file:
        data = args.duckdb_file.expanduser().resolve()
        if not data.is_file():
            parser.error(f"Database not found: {data}")
        setup = f"ATTACH {quote(data)} AS {family} (READ_ONLY);\nUSE {family};\n"
    else:
        data = args.parquet_dir.expanduser().resolve()
        views = []
        for table in TABLES[family]:
            files = sorted(
                p
                for pattern in (
                    f"{table}.parquet",
                    f"{table}/*.parquet",
                    f"{table}_[0-9]*.parquet",
                )
                for p in data.glob(pattern)
                if p.is_file()
            )
            if not files:
                parser.error(f"No parquet files for {table} in {data}")
            views.append(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet(["
                + ",".join(quote(path) for path in files)
                + "]);\n"
            )
        setup = "".join(views)
    if havoc:
        # Q01_V07 uses `x -> ...` lambdas, which DuckDB accepts only with this setting.
        setup += "SET lambda_syntax='ENABLE_SINGLE_ARROW';\n"
    config = args.config.resolve()
    if not config.is_file():
        parser.error(f"Config not found: {config}")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a fresh directory")
    output.mkdir(parents=True)
    (output / "sirius.yaml").write_bytes(config.read_bytes())
    metadata = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "scale_factor": args.scale_factor,
        "data": str(data),
        "query_dir": str(query_dir),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "binary": str(binary),
        "binary_mtime": binary.stat().st_mtime,
        "binary_sha256": sha256(binary),
        "extension": str(extension),
        "extension_mtime": extension.stat().st_mtime,
        "extension_sha256": sha256(extension),
        "optimizer_mask": MASK,
        "timeout_seconds": args.timeout,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "result_validation": (
            "CPU comparison of completed GPU queries"
            if args.validate
            else "not performed"
        ),
        "validation_float_abs_tol": args.float_tolerance if args.validate else None,
        "validation_row_order": (
            "ignored; duplicate multiplicity preserved" if args.validate else None
        ),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    rows = []
    inventory = collections.defaultdict(set)
    for query in queries:
        directory = output / query.directory
        sql = query.sql
        cpu_env = dict(os.environ, SIRIUS_DISABLE="1")
        code, stdout, stderr = run_cli(
            binary,
            setup
            + f"SET disabled_optimizers={quote(MASK)}; SET explain_output='all';\n"
            + "EXPLAIN (FORMAT JSON) "
            + sql,
            cpu_env,
            directory / "plans",
            args.timeout,
        )
        row = {"query": query.id}
        if havoc:
            row["duplicates"] = " ".join(query.duplicates)
        row.update(status="", reason="", plan_status=str(code))
        if code == 0:
            try:
                plans = explain_arrays(stdout)
                (directory / "plans.json").write_text(json.dumps(plans, indent=2))
                for kind, tree in plans.items():
                    for node in walk(tree):
                        inventory[(kind, "operator", node["name"])].add(query.id)
                        for key in (
                            "Join Type",
                            "Aggregates",
                            "Expressions",
                            "Projections",
                            "Order By",
                            "Filters",
                            "Conditions",
                            "Groups",
                        ):
                            value = node.get("extra_info", {}).get(key, [])
                            if not value:
                                continue
                            values = value if isinstance(value, list) else [value]
                            if key == "Join Type":
                                inventory[(kind, key, str(value))].add(query.id)
                            else:
                                for expr in values:
                                    for fn in re.findall(
                                        r"\b([a-zA-Z_]\w*)\(", str(expr)
                                    ):
                                        inventory[(kind, "function", fn.lower())].add(
                                            query.id
                                        )
            except ValueError as exc:
                row["plan_status"] = str(exc)
        gpu_env = dict(
            os.environ,
            SIRIUS_CONFIG_FILE=str(config),
            SIRIUS_LOG_DIR=str(directory / "gpu/log"),
            SIRIUS_LOG_LEVEL="info",
        )
        gpu_env.pop("SIRIUS_DISABLE", None)
        code, stdout, stderr = run_cli(
            binary,
            setup
            + "SET enable_duckdb_fallback=false;\n.mode "
            + ("json" if args.validate else "csv")
            + "\n"
            + sql,
            gpu_env,
            directory / "gpu",
            args.timeout,
            extension,
        )
        logs = "\n".join(
            p.read_text(errors="replace") for p in (directory / "gpu/log").glob("*.log")
        )
        errors = [line for line in stderr.splitlines() if re.search(r"\bError:", line)]
        row["reason"] = " | ".join(errors) or (stderr.strip() if code != 0 else "")
        if code == "timeout":
            row["status"] = "timeout"
            row["reason"] = (
                f"Exceeded {args.timeout:g}s; inspect GPU log for last progress"
            )
        elif code < 0:
            row["status"] = "crash"
            row["reason"] = " | ".join(
                [f"Killed by {signal.Signals(-code).name}"] + errors
            )
        elif code == 0 and "Transparent GPU execution: query completed" in logs:
            row["status"] = "gpu_completed"
        elif "GPU plan generation failed:" in stderr:
            row["status"] = "plan_rejected"
        elif (
            "Sirius GPU execution failed:" in stderr
            or "Sirius physical plan generated successfully" in logs
        ):
            row["status"] = "runtime_failed"
        else:
            row["status"] = "unclassified_error" if code != 0 else "gpu_not_confirmed"
        if args.validate:
            row["validation_status"], row["validation_reason"] = (
                "not_applicable",
                "GPU query did not complete",
            )
            if row["status"] == "gpu_completed":
                row["validation_status"], row["validation_reason"] = validate_result(
                    binary,
                    setup,
                    sql,
                    cpu_env,
                    directory / "cpu",
                    args.timeout,
                    stdout,
                    args.float_tolerance,
                )
            (directory / "validation.json").write_text(
                json.dumps(
                    {
                        "status": row["validation_status"],
                        "detail": row["validation_reason"],
                    },
                    indent=2,
                )
            )
        rows.append(row)
        with (output / "classification.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{query.label}: {row['status']} {row['reason'][:180]}", flush=True)
        if args.validate:
            with (output / "validation.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["query", "status", "detail"])
                writer.writeheader()
                writer.writerows(
                    {
                        "query": r["query"],
                        "status": r["validation_status"],
                        "detail": r["validation_reason"],
                    }
                    for r in rows
                )
            print(
                f"  validation: {row['validation_status']} {row['validation_reason'][:180]}",
                flush=True,
            )
    features = [
        {"plan": k[0], "kind": k[1], "feature": k[2], "queries": sorted(v)}
        for k, v in sorted(inventory.items())
    ]
    (output / "features.json").write_text(json.dumps(features, indent=2))
    context = f"`{args.benchmark}`"
    if args.scale_factor is not None:
        context += f" at SF {args.scale_factor:g}"
    context += f" on `{data.name}`, revision `{metadata['revision'][:8]}`, "
    context += "validated against DuckDB on CPU." if args.validate else "not validated."
    (output / "summary.md").write_text(
        summary(list(zip(queries, rows)), context, per_base_query=havoc)
    )
    print(dict(collections.Counter(row["status"] for row in rows)), flush=True)
    if args.validate:
        print(
            "Validation:",
            dict(collections.Counter(row["validation_status"] for row in rows)),
            flush=True,
        )
    print(f"Summary: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
