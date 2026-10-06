#!/usr/bin/env python3
"""Check committed Paimon tables against independent CPU expectations."""

import argparse
from collections import Counter
from decimal import Decimal
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

from qualified_extension import select_qualification
from build_extension import validate_receipt

from corpus_checks import (
    digest,
    validate_inventory,
    validate_table_metadata,
    validate_oracle,
)

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
            # TimeoutExpired may carry bytes even when subprocess uses text=True.
            def text(value):
                return (
                    value.decode("utf-8", errors="replace")
                    if isinstance(value, bytes)
                    else value or ""
                )

            Path(str(log) + ".stdout").write_text(text(error.stdout))
            Path(str(log) + ".stderr").write_text(
                text(error.stderr) + "\n" + str(error) + "\n"
            )
        raise RuntimeError(f"Process timed out after {timeout}s: {args[0]}") from error
    if log:
        Path(str(log) + ".stdout").write_text(result.stdout)
        Path(str(log) + ".stderr").write_text(result.stderr)
    if result.returncode:
        raise ProcessError(result)
    return result.stdout


def decode_results(output):
    """DuckDB emits one JSON document per statement with a result."""

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON result column: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    decoder = json.JSONDecoder(
        parse_float=Decimal,
        object_pairs_hook=unique_object,
        parse_constant=invalid_constant,
    )
    results = []
    remaining = output.strip()
    while remaining:
        value, end = decoder.raw_decode(remaining)
        if not isinstance(value, list):
            raise ValueError("Expected an array of result rows")
        results.append(value)
        remaining = remaining[end:].strip()
    return results


def query_result(output, extension, columns, mode="cpu"):
    """Check the complete six-document protocol, including explicit empty rows."""
    results = decode_results(output)
    if len(results) != 6:
        raise AssertionError(
            "Expected exactly six SQL results: extension, version, GPU setting, "
            f"types, query, and liveness; got {len(results)}"
        )
    if results[0] != [{"extension_version": extension["version"]}]:
        raise AssertionError("Loaded extension version differs from qualified artifact")
    if results[1] != [{"version": extension["duckdb_version"]}]:
        raise AssertionError("Unexpected DuckDB version")
    enabled = mode == "transparent"
    if results[2] != [{"gpu": enabled}] or results[2][0]["gpu"] is not enabled:
        raise AssertionError("GPU execution setting was not verified")
    if results[5] != [{"probe": "LIVENESS_OK"}]:
        raise AssertionError("Second query on the same connection did not complete")
    observed = [[row["column_name"], row["column_type"]] for row in results[3]]
    if observed != columns:
        raise AssertionError(f"Expected types {columns}, observed {observed}")
    return results[4]


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
        if isinstance(value, (float, bool)):
            raise ValueError("Float/bool is not an exact decimal reference")
        result = Decimal(value)
        if not result.is_finite():
            raise ValueError("Non-finite decimal reference")
        return result
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
    if (
        not cases
        or any(
            not isinstance(i, str) or not re.fullmatch(r"[a-zA-Z0-9_]+", i) for i in ids
        )
        or len(set(ids)) != len(ids)
    ):
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


def probe_identity(cli, directory, timeout):
    env = os.environ.copy()
    env["SIRIUS_DISABLE"] = "1"
    output = execute(
        [
            cli,
            "-init",
            "/dev/null",
            "-json",
            "-batch",
            "-bail",
            ":memory:",
            "-c",
            "SELECT version() AS version; PRAGMA platform;",
        ],
        timeout=timeout,
        cwd=directory,
        env=env,
        log=directory / "reader-identity",
    )
    results = decode_results(output)
    if len(results) != 2 or len(results[0]) != 1 or len(results[1]) != 1:
        raise ValueError("Incomplete reader version/platform identity")
    version, platform = results[0][0]["version"], results[1][0]["platform"]
    if not isinstance(version, str) or not isinstance(platform, str):
        raise ValueError("Malformed reader version/platform identity")
    return {"version": version, "platform": platform}


def run_cases(args, corpus, spec, cases, manifest, extension, cli, run_dir, report):
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

    def command(sql, mode, label, *, rejection=False):
        env = os.environ.copy()
        env.update(SIRIUS_CONFIG_FILE=str(config), TZ="UTC", SIRIUS_LOG_LEVEL="warn")
        env["SIRIUS_DISABLE"] = "1" if mode == "disabled" else "0"
        enabled = "true" if mode == "transparent" else "false"
        setup = (
            "SET autoinstall_known_extensions=false; SET autoload_known_extensions=false;"
            f"LOAD {sql_literal(extension)}; SET gpu_execution={enabled};"
            "SELECT extension_version FROM duckdb_extensions() WHERE extension_name='paimon';"
            "SELECT version() AS version;"
            "SELECT current_setting('gpu_execution') AS gpu;"
        )
        if rejection:
            setup += "SET enable_duckdb_fallback=false;"
        return execute(
            [
                cli,
                *(
                    ["-unsigned"]
                    if spec["extension"].get("kind") == "source_build"
                    else []
                ),
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

    def query(case, sql, mode, label, setup=""):
        output = command(
            setup + "DESCRIBE " + sql + ";" + sql + "; SELECT 'LIVENESS_OK' AS probe;",
            mode,
            label,
        )
        compare_rows(
            case, query_result(output, spec["extension"], case["columns"], mode)
        )

    def record(case_id, mode, action):
        item = next(
            i for i in report["cases"] if i["id"] == case_id and i["mode"] == mode
        )
        try:
            action(item)
            item.update(state="PASS", passed=True)
        except Exception as error:
            item.update(
                state="FAIL",
                passed=False,
                error=str(error),
                error_type=type(error).__name__,
                traceback=traceback.format_exc(),
            )
        print(f"{mode}/{case_id}: {item['state']}", flush=True)

    def case_sql(case):
        table = manifest["tables"][case["table"]]
        snapshot = table["snapshots"][case["snapshot"]] if case["snapshot"] else None
        return (
            case["sql"].format(scan=scan_sql(corpus / table["path"], snapshot)),
            snapshot,
        )

    for mode in ("disabled", "cpu"):
        for case in cases:

            def check(item, case=case, mode=mode):
                sql, snapshot = case_sql(case)
                item["snapshot_id"] = snapshot
                query(case, sql, mode, mode + "-" + case["id"])

            record(case["id"], mode, check)

    def transparent(item):
        case = next(c for c in spec["cases"] if c["id"] == "append_b")
        sql, item["snapshot_id"] = case_sql(case)
        query(case, sql, "transparent", "transparent")

    record("rows_and_liveness", "transparent", transparent)

    def rejected(item):
        case = next(c for c in spec["cases"] if c["id"] == "append_b")
        sql, item["snapshot_id"] = case_sql(case)
        expect_planning_rejection(
            lambda: command(sql, "transparent", "rejection", rejection=True)
        )

    record("planning_rejection", "transparent", rejected)

    def attached(item):
        case = next(c for c in spec["cases"] if c["id"] == "pk_d")
        setup = (
            f"ATTACH {sql_literal(corpus / 'warehouse')} AS p (TYPE paimon, READ_ONLY);"
        )
        query(
            case,
            "SELECT id, amount FROM p.reference.orders_pk",
            "cpu",
            "attached",
            setup,
        )

    record("attached_catalog", "cpu", attached)

    validate_inventory(corpus, manifest)
    if any(c["state"] != "PASS" for c in report["cases"]):
        raise RuntimeError("Conformance failures: inspect per-case errors and logs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, nargs="?", default=HERE)
    parser.add_argument("--duckdb", type=Path, default=ROOT / "build/release/duckdb")
    parser.add_argument("--paimon-extension", type=Path, required=True)
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument(
        "--build-receipt",
        type=Path,
        help="Source build receipt (default: beside the extension)",
    )
    provenance.add_argument(
        "--registry",
        type=Path,
        help="Explicit historical byte-qualified registry instead of a source build receipt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORK,
        help="Parent directory for reports (outside the corpus)",
    )
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    if output.is_relative_to(corpus):
        parser.error("--output must be outside the corpus")
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    report = {
        "state": "running",
        "corpus": str(corpus),
        "cases": [],
        "requested_cases": args.case,
        "planned_cases": None,
    }
    print(f"Report directory: {run_dir}", flush=True)
    try:
        spec = json.loads((corpus / "expectations.json").read_text())
        cases = select_cases(spec["cases"], args.case)
        report["cases"] = [
            {"id": c["id"], "mode": mode, "state": "NOT RUN"}
            for mode in ("disabled", "cpu")
            for c in cases
        ]
        report["cases"] += [
            {"id": case_id, "mode": mode, "state": "NOT RUN"}
            for case_id, mode in (
                ("rows_and_liveness", "transparent"),
                ("planning_rejection", "transparent"),
                ("attached_catalog", "cpu"),
            )
        ]
        report["planned_cases"] = len(report["cases"])
        manifest = json.loads((corpus / "manifest.json").read_text())
        if manifest["format_version"] != 2 or not isinstance(manifest["tables"], dict):
            raise ValueError("Expected corpus manifest format 2")
        report.update(
            expected_sha256=digest(corpus / "expectations.json"),
            manifest_sha256=digest(corpus / "manifest.json"),
        )
        validate_inventory(corpus, manifest)
        validate_table_metadata(corpus, manifest)
        validate_oracle(spec, manifest, compare_rows)
        extension = args.paimon_extension.resolve(strict=True)
        cli = args.duckdb.resolve(strict=True)
        identity = probe_identity(cli, run_dir, args.timeout)
        if args.registry:
            spec["extension"] = select_qualification(args.registry, identity)
            report["registry_sha256"] = digest(args.registry)
            if digest(extension) != spec["extension"]["sha256"]:
                raise ValueError(
                    "Unqualified Paimon artifact: registry SHA256 mismatch"
                )
        else:
            receipt = args.build_receipt or extension.with_name("build-receipt.json")
            spec["extension"] = validate_receipt(receipt, extension, cli, identity)
        report.update(
            duckdb={"path": str(cli), "sha256": digest(cli), **identity},
            extension=spec["extension"],
        )
        run_cases(args, corpus, spec, cases, manifest, extension, cli, run_dir, report)
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=str(error), error_type=type(error).__name__)
        raise
    finally:
        report["executed_cases"] = sum(c["state"] != "NOT RUN" for c in report["cases"])
        report["failed_cases"] = sum(c["state"] == "FAIL" for c in report["cases"])
        report["not_run_cases"] = sum(c["state"] == "NOT RUN" for c in report["cases"])
        (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {run_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Paimon conformance failed: {error}", file=sys.stderr)
        sys.exit(1)
