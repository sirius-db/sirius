#!/usr/bin/env python3
"""Inventory TPC-DS plans and run Sirius with CPU fallback disabled.

Run through pixi; GPU access must be available to the process. No data generation.
Each query gets a fresh CLI process so a runtime failure cannot poison the next query.

Uses existing Parquet data; no database import or data generation is
needed. Run commands from the repository root. The built
`build/release/duckdb` CLI already includes Sirius, so these scripts do not
need an explicit `LOAD` statement.

For a classification run:

```bash
pixi run python test/tpcds_performance/classify.py \
  $PATH_TO_TPC_DS_TABLES \
  --config test/tpcds_performance/classification.yaml \
  --output-dir ...
```

The output directory must be new. Add `--queries 3 7 42` for a subset or
`--timeout 60` to limit each process to 60 seconds. Table inputs can be named
`table.parquet`, `table/*.parquet`, or `table_[0-9]*.parquet` shards. The shard
suffix must start with a digit. This avoids mistaking `store_returns` for a `store` shard or
`customer_address` for a `customer` shard. The example config uses
one GPU, 8 GiB of GPU memory and 4 GiB of pinned memory per host NUMA node.
It is a small baseline for SF1, not a memory recommendation for larger scales.

The classifier reads the canonical 99 SQL files from the local DuckDB submodule
(`duckdb/extension/tpcds/dsdgen/queries/01.sql` through `99.sql`), without
changing their identifiers or generating data. Use `--query-dir` to select a
different corpus; it accepts either `q1.sql` or `01.sql` naming.

For each query, it saves:

- `plans.json`: DuckDB's unoptimized logical, optimized logical, and CPU physical
  trees, using Sirius's disabled optimizer mask.
- `plans/`: the exact EXPLAIN input, stdout and stderr.
- `gpu/`: the exact execution input, CSV results, stderr and Sirius logs.
- `classification.csv`: completed GPU execution, plan rejection, runtime failure,
  timeout, or an unconfirmed/unclassified outcome, with the first exact error.
- `features.json`: query membership for plan operators, join types and function
  names visible in the EXPLAIN trees. This is an inventory, not a support verdict.
- `metadata.json` and `sirius.yaml`: source revision, dirty status, binary hash,
  input location, optimizer mask, timeout and configuration.

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
import subprocess


ROOT = Path(__file__).resolve().parents[2]
TABLES = """call_center catalog_page catalog_returns catalog_sales customer
customer_address customer_demographics date_dim household_demographics income_band
inventory item promotion reason ship_mode store store_returns store_sales time_dim
warehouse web_page web_returns web_sales web_site""".split()
MASK = "in_clause,compressed_materialization,late_materialization"
DEFAULT_FLOAT_TOLERANCE = 1e-10


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


def run_cli(binary, sql, env, directory, timeout):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "input.sql").write_text(sql)
    try:
        result = subprocess.run(
            [str(binary), "-bail"],
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", nargs="+", type=int, default=list(range(1, 100)))
    parser.add_argument(
        "--query-dir", type=Path, default=ROOT / "duckdb/extension/tpcds/dsdgen/queries"
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
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory already exists; choose a fresh directory")
    output.mkdir(parents=True)
    views = []
    for table in TABLES:
        data = args.parquet_dir.expanduser().resolve()
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
    binary = ROOT / "build/release/duckdb"
    config = args.config.resolve()
    if not config.is_file():
        parser.error(f"Config not found: {config}")
    (output / "sirius.yaml").write_bytes(config.read_bytes())
    metadata = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data": str(data),
        "query_dir": str(args.query_dir.resolve()),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "binary_mtime": binary.stat().st_mtime,
        "binary_sha256": hashlib.file_digest(binary.open("rb"), "sha256").hexdigest(),
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
    for q in args.queries:
        directory = output / f"q{q}"
        path = args.query_dir / f"q{q}.sql"
        if not path.exists():
            path = args.query_dir / f"{q:02}.sql"
        sql = path.read_text().strip().rstrip(";") + ";\n"
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
        row = {"query": q, "status": "", "reason": "", "plan_status": str(code)}
        if code == 0:
            try:
                plans = explain_arrays(stdout)
                (directory / "plans.json").write_text(json.dumps(plans, indent=2))
                for kind, tree in plans.items():
                    for node in walk(tree):
                        inventory[(kind, "operator", node["name"])].add(q)
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
                                inventory[(kind, key, str(value))].add(q)
                            else:
                                for expr in values:
                                    for fn in re.findall(
                                        r"\b([a-zA-Z_]\w*)\(", str(expr)
                                    ):
                                        inventory[(kind, "function", fn.lower())].add(q)
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
        )
        logs = "\n".join(p.read_text() for p in (directory / "gpu/log").glob("*.log"))
        errors = [line for line in stderr.splitlines() if re.search(r"\bError:", line)]
        row["reason"] = " | ".join(errors) or (stderr.strip() if code != 0 else "")
        if code == "timeout":
            row["status"] = "timeout"
            row["reason"] = (
                f"Exceeded {args.timeout:g}s; inspect GPU log for last progress"
            )
        elif code == 0 and "Transparent GPU execution: query completed" in logs:
            row["status"] = "gpu_completed"
        elif "GPU plan generation failed:" in stderr:
            row["status"] = "plan_rejected"
        elif "Sirius physical plan generated successfully" in logs:
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
        print(f"Q{q}: {row['status']} {row['reason'][:180]}", flush=True)
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
    print(dict(collections.Counter(row["status"] for row in rows)), flush=True)
    if args.validate:
        print(
            "Validation:",
            dict(collections.Counter(row["validation_status"] for row in rows)),
            flush=True,
        )


if __name__ == "__main__":
    main()
