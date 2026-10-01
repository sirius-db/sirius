#!/usr/bin/env python3
"""Check committed Paimon tables against independent CPU expectations."""

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / "build/paimon-conformance"


class ProcessError(RuntimeError):
    """Keep the exit status separate from diagnostics when checking SQL errors."""

    def __init__(self, result):
        self.returncode = result.returncode
        self.stderr = result.stderr
        super().__init__(f"Process exited {self.returncode}: {self.stderr[-6000:]}")


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def execute(args, *, timeout, cwd, env=None, log=None):
    """Never turn a failed or timed-out reader into an empty successful result."""
    try:
        result = subprocess.run(
            list(map(str, args)),
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        if log:
            Path(str(log) + ".stderr").write_text(str(error))
        raise RuntimeError(f"Process timed out after {timeout}s: {args[0]}") from error
    if log:
        Path(str(log) + ".stdout").write_text(result.stdout)
        Path(str(log) + ".stderr").write_text(result.stderr)
    if result.returncode:
        raise ProcessError(result)
    return result.stdout


def decode_results(output):
    """DuckDB emits one JSON document per statement with a result."""
    decoder = json.JSONDecoder(parse_float=Decimal)
    results = []
    remaining = output.strip()
    while remaining:
        value, end = decoder.raw_decode(remaining)
        if not isinstance(value, list):
            raise ValueError("Expected an array of result rows")
        results.append(value)
        remaining = remaining[end:].strip()
    return results


def query_result(output, extension):
    """Require an explicit result document, including [] for zero rows."""
    results = decode_results(output)
    if len(results) != 5:
        raise AssertionError(
            "Expected exactly five SQL results: extension, version, CPU setting, "
            f"query, and liveness; got {len(results)}"
        )
    if results[0] != [{"extension_version": extension["version"]}]:
        raise AssertionError("Loaded extension version differs from qualified artifact")
    if results[1] != [{"version": extension["duckdb_version"]}]:
        raise AssertionError("Unexpected DuckDB version")
    if results[2] != [{"gpu": False}]:
        raise AssertionError("CPU setup was not verified")
    if results[4] != [{"probe": "LIVENESS_OK"}]:
        raise AssertionError("Second query on the same connection did not complete")
    return results[3]


def expect_planning_rejection(action):
    """Only DuckDB's normal SQL-error exit qualifies as the expected refusal."""
    try:
        action()
    except ProcessError as error:
        if error.returncode == 1 and (
            "GPU plan generation failed: Table function 'paimon_scan' is not supported in Sirius"
            in error.stderr
        ):
            return
        raise
    raise AssertionError("Expected planning-time rejection with CPU fallback disabled")


def normalize(value, type_name):
    if value is None:
        return None
    if type_name.startswith("DECIMAL("):
        if isinstance(value, float):
            raise ValueError("Binary floating point is not an exact decimal reference")
        return Decimal(value)
    if type_name == "BIGINT":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Expected integer, got {value!r}")
        return value
    if type_name in ("VARCHAR", "DATE"):
        if not isinstance(value, str):
            raise ValueError(f"Expected text for {type_name}, got {value!r}")
        return value
    raise ValueError(f"Unqualified reference type {type_name}")


def compare_rows(case, actual):
    names = [c[0] for c in case["columns"]]
    types = [c[1] for c in case["columns"]]
    if len(set(names)) != len(names):
        raise ValueError("Reference result columns must have unique names")
    expected = [
        tuple(normalize(v, t) for v, t in zip(row, types, strict=True))
        for row in case["rows"]
    ]
    observed = []
    for row in actual:
        if list(row) != names:
            raise AssertionError(
                f"Column names/order: expected {names}, got {list(row)}"
            )
        observed.append(tuple(normalize(row[n], t) for n, t in zip(names, types)))
    left, right = (
        (expected, observed)
        if case.get("ordered")
        else (Counter(expected), Counter(observed))
    )
    if left != right:
        raise AssertionError(f"Expected {expected!r}; observed {observed!r}")


def select_cases(cases, requested):
    ids = [c["id"] for c in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("Expected a nonempty case list with unique IDs")
    unknown = set(requested) - set(ids)
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    return [c for c in cases if not requested or c["id"] in requested]


def scan_sql(path, snapshot):
    argument = sql_literal(path)
    if snapshot is not None:
        if type(snapshot) is not int or snapshot <= 0:
            raise ValueError("Historical snapshot ID must be a positive integer")
        argument += f", snapshot_from_id={snapshot}"
    return f"paimon_scan({argument})"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?", default=HERE)
    parser.add_argument("--duckdb", type=Path, default=ROOT / "build/release/duckdb")
    parser.add_argument("--paimon-extension", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    corpus = args.corpus.resolve()
    spec = json.loads((corpus / "expectations.json").read_text())
    cases = select_cases(spec["cases"], args.case)
    manifest = json.loads((corpus / "manifest.json").read_text())
    extension = args.paimon_extension.resolve(strict=True)
    cli = args.duckdb.resolve(strict=True)
    if digest(extension) != spec["extension"]["sha256"]:
        raise ValueError(
            "Unqualified Paimon artifact: expected the version and SHA256 in expectations.json"
        )
    for relative, expected in manifest["files"].items():
        file = (corpus / relative).resolve(strict=True)
        if not file.is_relative_to(corpus) or digest(file) != expected:
            raise ValueError(f"Changed or external corpus file: {relative}")
    WORK.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=WORK))
    config = run_dir / "sirius.yaml"
    config.write_text(
        """sirius:
  topology: {num_gpus: 1}
  memory:
    gpu: {usage_limit_bytes: 1Gi}
    host: {capacity_bytes: 256Mi, initial_number_pools: 1, pool_size: 16, block_size: 1Mi}
    disk: {capacity_bytes: 1Gi, downgrade_root_dirs: "./spill"}
  telemetry: {enable_quent: false}
"""
    )
    report = {
        "state": "running",
        "corpus": str(corpus),
        "expected_sha256": digest(corpus / "expectations.json"),
        "manifest_sha256": digest(corpus / "manifest.json"),
        "duckdb": {"path": str(cli), "sha256": digest(cli)},
        "extension": spec["extension"],
        "planned_cases": len(cases) * 2 + 3,
        "cases": [],
    }
    print(f"Report directory: {run_dir}", flush=True)

    def command(sql, mode, label, *, rejection=False):
        env = os.environ.copy()
        env.update(SIRIUS_CONFIG_FILE=str(config), TZ="UTC")
        env["SIRIUS_DISABLE"] = "1" if mode == "disabled" else "0"
        setup = (
            "SET autoinstall_known_extensions=false; SET autoload_known_extensions=false;"
            f"LOAD {sql_literal(extension)}; SET gpu_execution=false;"
            "SELECT extension_version FROM duckdb_extensions() WHERE extension_name='paimon';"
            "SELECT version() AS version;"
            "SELECT current_setting('gpu_execution') AS gpu;"
        )
        if mode == "transparent":
            setup += "SET gpu_execution=true;"
        if rejection:
            setup += "SET enable_duckdb_fallback=false;"
        return execute(
            [
                cli,
                "-init",
                "/dev/null",
                "-json",
                "-batch",
                "-bail",
                ":memory:",
                "-c",
                setup + sql,
            ],
            timeout=args.timeout,
            cwd=run_dir,
            env=env,
            log=run_dir / label,
        )

    def query(sql, mode, label):
        return query_result(
            command(sql + "; SELECT 'LIVENESS_OK' AS probe;", mode, label),
            spec["extension"],
        )

    def record(case_id, mode, action, **metadata):
        item = {"id": case_id, "mode": mode, **metadata}
        try:
            action()
            item["passed"] = True
        except (RuntimeError, OSError, ValueError, AssertionError) as error:
            item.update(passed=False, error=str(error))
        report["cases"].append(item)
        print(f"{mode}/{case_id}: {'PASS' if item['passed'] else 'FAIL'}", flush=True)

    def case_sql(case):
        table = manifest["tables"][case["table"]]
        snapshot = table["snapshots"][case["snapshot"]] if case["snapshot"] else None
        return (
            case["sql"].format(scan=scan_sql(corpus / table["path"], snapshot)),
            snapshot,
        )

    try:
        for mode in ("disabled", "cpu"):
            for case in cases:
                sql, snapshot = case_sql(case)

                def check(case=case, sql=sql, mode=mode):
                    label = mode + "-" + case["id"]
                    desc = query("DESCRIBE " + sql, mode, label + "-types")
                    columns = [[r["column_name"], r["column_type"]] for r in desc]
                    if columns != case["columns"]:
                        raise AssertionError(
                            f"Expected {case['columns']}, observed {columns}"
                        )
                    compare_rows(case, query(sql, mode, label))

                record(case["id"], mode, check, snapshot_id=snapshot)

        smoke = next(c for c in spec["cases"] if c["id"] == "append_b")
        smoke_sql, _ = case_sql(smoke)
        record(
            "rows_and_liveness",
            "transparent",
            lambda: compare_rows(smoke, query(smoke_sql, "transparent", "transparent")),
        )

        def rejected():
            expect_planning_rejection(
                lambda: command(smoke_sql, "transparent", "rejection", rejection=True)
            )

        record("planning_rejection", "transparent", rejected)

        def attached():
            sql = (
                f"ATTACH {sql_literal(corpus / 'warehouse')} AS p (TYPE paimon, READ_ONLY);"
                "SELECT id, amount FROM p.reference.orders_pk"
            )
            case = next(c for c in spec["cases"] if c["id"] == "pk_d")
            compare_rows(case, query(sql, "cpu", "attached"))

        record("attached_catalog", "cpu", attached)
        if any(not case["passed"] for case in report["cases"]):
            raise RuntimeError("Conformance failures: inspect per-case errors and logs")
        if len(report["cases"]) != report["planned_cases"]:
            raise RuntimeError("Incomplete case execution")
        report["state"] = "passed"
    except Exception as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        report["executed_cases"] = len(report["cases"])
        report["failed_cases"] = sum(not c["passed"] for c in report["cases"])
        (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Paimon conformance failed: {error}", file=sys.stderr)
        sys.exit(1)
