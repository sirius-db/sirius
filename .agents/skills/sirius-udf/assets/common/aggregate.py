# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Validate aggregate SQL rewrites against an executable original reference."""

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


COUNTERS = ("executions", "plan_fallbacks", "runtime_fallbacks")


def table(value, label):
    if not isinstance(value, dict) or set(value) != {"types", "rows"}:
        raise ValueError(f"{label}: expected typed result table")
    types, rows = value["types"], value["rows"]
    if (
        not isinstance(types, list)
        or not types
        or any(not isinstance(t, str) or not t for t in types)
    ):
        raise ValueError(f"{label}: invalid logical types")
    if not isinstance(rows, list):
        raise ValueError(f"{label}: invalid rows")
    for row in rows:
        if not isinstance(row, list) or len(row) != len(types):
            raise ValueError(f"{label}: invalid row width")
        for cell, logical_type in zip(row, types):
            if cell is None:
                continue
            if logical_type in ("FLOAT", "DOUBLE"):
                width = 8 if logical_type == "FLOAT" else 16
                if (
                    not isinstance(cell, dict)
                    or set(cell) != {"ieee_bits"}
                    or not isinstance(cell["ieee_bits"], str)
                    or len(cell["ieee_bits"]) != width
                    or any(c not in "0123456789abcdef" for c in cell["ieee_bits"])
                ):
                    raise ValueError(f"{label}: invalid IEEE representation")
            elif not isinstance(cell, str):
                raise ValueError(f"{label}: non-floating cells must be strings")
    return value


def tolerances(policy, types):
    if not isinstance(policy, dict) or set(policy) - {"float_tolerances"}:
        raise ValueError("invalid comparison policy")
    configured = policy.get("float_tolerances", {})
    if not isinstance(configured, dict):
        raise ValueError("float_tolerances must be an object")
    for column, limits in configured.items():
        if (
            not isinstance(column, str)
            or not column.isdigit()
            or str(int(column)) != column
            or int(column) >= len(types)
            or types[int(column)] not in ("FLOAT", "DOUBLE")
            or not isinstance(limits, dict)
            or set(limits) != {"atol", "rtol"}
            or any(
                type(limit) not in (int, float) or not math.isfinite(limit) or limit < 0
                for limit in limits.values()
            )
        ):
            raise ValueError("invalid floating-point tolerance")
    return configured


def compare(reference, actual, label, policy=None):
    table(reference, "reference")
    table(actual, label)
    if reference["types"] != actual["types"]:
        raise AssertionError(f"{label}: logical result types differ")
    if len(reference["rows"]) != len(actual["rows"]):
        raise AssertionError(f"{label}: result row counts differ")
    configured = tolerances(policy if policy is not None else {}, reference["types"])
    for index, (left, right) in enumerate(zip(reference["rows"], actual["rows"])):
        for column, (lhs, rhs) in enumerate(zip(left, right)):
            if lhs == rhs:
                continue
            limits = configured.get(str(column))
            if limits is not None and lhs is not None and rhs is not None:
                width = "!f" if reference["types"][column] == "FLOAT" else "!d"
                a = struct.unpack(width, bytes.fromhex(lhs["ieee_bits"]))[0]
                b = struct.unpack(width, bytes.fromhex(rhs["ieee_bits"]))[0]
                if (
                    math.isfinite(a)
                    and math.isfinite(b)
                    and math.isclose(
                        a, b, rel_tol=limits["rtol"], abs_tol=limits["atol"]
                    )
                ):
                    continue
            raise AssertionError(
                f"{label}: value or NULL differs at row {index}, column {column}"
            )


def local_file(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("case SQL path must be a nonempty relative path")
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root.resolve() not in path.parents:
        raise ValueError("case SQL path must stay inside the external project")
    if not path.is_file():
        raise ValueError(f"missing case SQL file: {relative}")
    return path


def read_manifest(root):
    suite = json.loads((root / "cases.json").read_text())
    if (
        not isinstance(suite, dict)
        or type(suite.get("version")) is not int
        or suite["version"] != 1
    ):
        raise ValueError("unsupported aggregate suite version")
    oracle = suite.get("oracle")
    if not isinstance(oracle, dict):
        raise ValueError("aggregate suite requires an executable original reference")
    argv = oracle.get("argv")
    timeout = oracle.get("timeout_seconds", 60)
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or not arg for arg in argv)
        or type(timeout) not in (int, float)
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("invalid original-reference command or timeout")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("aggregate suite requires cases")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("invalid aggregate case")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case IDs must be nonempty and unique")
        ids.add(case_id)
        for kind in ("setup", "input", "candidate"):
            local_file(root, case.get(kind))
        if type(case.get("gpu_required")) is not bool:
            raise ValueError("each case must declare gpu_required")
        if not isinstance(case.get("parameters", {}), dict):
            raise ValueError("oracle parameters must be an object")
    if not any(case["gpu_required"] for case in cases):
        raise ValueError("aggregate suite must require real GPU execution")
    benchmark_case = suite.get("benchmark_case")
    if benchmark_case is not None and benchmark_case not in ids:
        raise ValueError("benchmark_case does not identify a validation case")
    return suite


def original_reference(root, oracle, case, inputs, work):
    request = {
        "version": 1,
        "case_id": case["id"],
        "input": table(inputs, "input"),
        "parameters": case.get("parameters", {}),
    }
    (work / "oracle-request.json").write_text(json.dumps(request) + "\n")
    argv = [sys.executable if arg == "{python}" else arg for arg in oracle["argv"]]
    process = subprocess.run(
        argv,
        input=json.dumps(request),
        capture_output=True,
        text=True,
        cwd=root,
        timeout=oracle.get("timeout_seconds", 60),
        check=False,
    )
    (work / "oracle-stdout.txt").write_text(process.stdout)
    (work / "oracle-stderr.txt").write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f"original reference exited {process.returncode}")
    response = json.loads(process.stdout)
    if (
        not isinstance(response, dict)
        or type(response.get("version")) is not int
        or response["version"] != 1
    ):
        raise ValueError("invalid original-reference response version")
    if "error" in response:
        raise RuntimeError(f"original reference reported an error: {response['error']}")
    return table(response.get("result"), "original reference")


def gpu_evidence(evidence, required):
    if (
        not isinstance(evidence, dict)
        or set(evidence) != set(COUNTERS)
        or any(
            type(evidence[name]) is not int or evidence[name] < 0 for name in COUNTERS
        )
    ):
        raise ValueError("invalid GPU execution counters")
    if evidence["plan_fallbacks"] or evidence["runtime_fallbacks"]:
        raise AssertionError("aggregate candidate fell back")
    if required and evidence["executions"] != 1:
        raise AssertionError(
            "aggregate candidate did not execute exactly once on the GPU"
        )
    if evidence["executions"] > 1:
        raise AssertionError("aggregate case supplied unexpected GPU executions")
    return evidence


def export_native(args, work, setup, registration, inputs, candidate):
    paths = []
    for name, sql in (
        ("setup", setup),
        ("registration", registration),
        ("input", inputs),
        ("candidate", candidate),
    ):
        path = work / f"{name}.sql"
        path.write_text(sql)
        paths.append(str(path))
    environment = os.environ.copy()
    environment["SIRIUS_LOG_DIR"] = str(work / "log")
    (work / "log").mkdir(mode=0o700)
    process = subprocess.run(
        [
            str(args.host.resolve()),
            "--export-case",
            str(work / "validation.db"),
            *paths,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )
    (work / "host-stdout.txt").write_text(process.stdout)
    (work / "host-stderr.txt").write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f"native aggregate validation exited {process.returncode}")
    reports = [
        line.removeprefix("SIRIUS_UDF_CASE=")
        for line in process.stdout.split("\n")
        if line.startswith("SIRIUS_UDF_CASE=")
    ]
    if len(reports) != 1:
        raise ValueError("native host must emit exactly one aggregate case result")
    result = json.loads(reports[0])
    if (
        not isinstance(result, dict)
        or type(result.get("version")) is not int
        or result["version"] != 1
    ):
        raise ValueError("unsupported native case report version")
    return result


def export_python(args, work, setup, registration, inputs, candidate, adapter_factory):
    previous = os.environ.get("SIRIUS_LOG_DIR")
    os.environ["SIRIUS_LOG_DIR"] = str(work / "log")
    (work / "log").mkdir(mode=0o700)
    adapter = None
    try:
        adapter = adapter_factory(work / "validation.db", args.extension)
        adapter.execute(setup)
        adapter.execute(registration)
        adapter.execute("CHECKPOINT")
        source = adapter.query(inputs)
        cpu = adapter.query(candidate)
        adapter.execute("SET enable_duckdb_fallback=false")
        adapter.execute("SET gpu_execution=true")
        before = adapter.snapshot()
        gpu = adapter.query(candidate)
        after = adapter.snapshot()
        return {
            "version": 1,
            "input": source,
            "candidate_cpu": cpu,
            "candidate_gpu": gpu,
            "gpu_counters": {name: after[name] - before[name] for name in COUNTERS},
        }
    finally:
        if adapter is not None:
            adapter.close()
        if previous is None:
            os.environ.pop("SIRIUS_LOG_DIR", None)
        else:
            os.environ["SIRIUS_LOG_DIR"] = previous


def save_report(args, report):
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


def run_project(root, args, project, adapter_factory, benchmark_runner):
    report = {
        "validation": "failed",
        "kind": "aggregate",
        "target": "sql",
        "reference_basis": project["reference"],
        "cases": [],
    }
    # Keep raw requests, candidate outputs and failure logs when a report is requested.
    with tempfile.TemporaryDirectory(
        prefix="sirius-aggregate-validation-"
    ) as temporary:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            evidence_root = Path(
                tempfile.mkdtemp(prefix="aggregate-evidence-", dir=args.report.parent)
            )
        else:
            evidence_root = Path(temporary)
        report["evidence_directory"] = str(evidence_root) if args.report else None
        try:
            suite = read_manifest(root)
            registration = (root / "registration.sql").read_text()
            selected = None
            for number, case in enumerate(suite["cases"]):
                report["active_case"] = case["id"]
                work = evidence_root / f"case-{number:03d}"
                work.mkdir(mode=0o700)
                sql = {
                    name: local_file(root, case[name])
                    .read_text()
                    .replace("@ROWS@", str(args.rows))
                    for name in ("setup", "input", "candidate")
                }
                result = (
                    export_native(
                        args,
                        work,
                        sql["setup"],
                        registration,
                        sql["input"],
                        sql["candidate"],
                    )
                    if args.host
                    else export_python(
                        args,
                        work,
                        sql["setup"],
                        registration,
                        sql["input"],
                        sql["candidate"],
                        adapter_factory,
                    )
                )
                (work / "candidate-results.json").write_text(json.dumps(result) + "\n")
                evidence = gpu_evidence(
                    result.get("gpu_counters"), case["gpu_required"]
                )
                expected = original_reference(
                    root, suite["oracle"], case, result.get("input"), work
                )
                policy = case.get("comparison", {})
                compare(
                    expected, result.get("candidate_cpu"), "CPU SQL candidate", policy
                )
                compare(
                    expected, result.get("candidate_gpu"), "GPU SQL candidate", policy
                )
                report["cases"].append(
                    {
                        "id": case["id"],
                        "validation": "passed",
                        "input_rows": len(result["input"]["rows"]),
                        "result_rows": len(expected["rows"]),
                        "gpu_required": case["gpu_required"],
                        "gpu_counters": evidence,
                        "comparison": policy,
                    }
                )
                if case["id"] == suite.get("benchmark_case"):
                    if not case["gpu_required"]:
                        raise ValueError(
                            "benchmark_case must require real GPU execution"
                        )
                    selected = (sql, work)
            report.pop("active_case", None)
            if not args.validate_only:
                if selected is None:
                    raise ValueError(
                        "benchmarking requires benchmark_case in cases.json"
                    )
                selected_sql, selected_work = selected
                measured = benchmark_runner(
                    args,
                    selected_work / "validation.db",
                    selected_sql["setup"],
                    registration,
                    selected_sql["candidate"],
                    selected_sql["candidate"],
                )
                report["cpu_baseline"] = (
                    "equivalent DuckDB CPU SQL; original reference is correctness-only"
                )
                report["timing"] = measured["timing"]
                report["cpu"] = measured["cpu"]
                report["gpu"] = measured["gpu"]
            report["validation"] = "passed"
            save_report(args, report)
            return report
        except Exception as error:
            report["error"] = str(error)
            save_report(args, report)
            raise


def measure_native(args, database, setup, registration, reference, candidate):
    """Measure the already validated database without replacing its comparison policy."""
    candidate_path = database.parent / "measurement-candidate.sql"
    candidate_path.write_text(candidate)
    environment = os.environ.copy()
    environment["SIRIUS_LOG_DIR"] = str(database.parent / "measurement-log")
    process = subprocess.run(
        [
            str(args.host.resolve()),
            "--measure-case",
            str(database),
            str(candidate_path),
            str(args.warmups),
            str(args.iterations),
        ],
        text=True,
        capture_output=True,
        env=environment,
        timeout=300,
        check=False,
    )
    (database.parent / "measurement-stdout.txt").write_text(process.stdout)
    (database.parent / "measurement-stderr.txt").write_text(process.stderr)
    if process.returncode:
        raise RuntimeError(f"aggregate measurement host exited {process.returncode}")
    reports = [
        line.removeprefix("SIRIUS_UDF_MEASUREMENT=")
        for line in process.stdout.split("\n")
        if line.startswith("SIRIUS_UDF_MEASUREMENT=")
    ]
    if len(reports) != 1:
        raise ValueError("host must emit exactly one aggregate measurement")
    result = json.loads(reports[0])
    if not isinstance(result, dict) or result.get("measurement") != "complete":
        raise ValueError("host omitted aggregate measurements")
    for mode in ("cpu", "gpu"):
        measurement = result.get(mode)
        if not isinstance(measurement, dict):
            raise ValueError("host omitted aggregate measurement samples")
        samples = measurement.get("samples_seconds")
        median = measurement.get("median_seconds")
        if (
            not isinstance(samples, list)
            or len(samples) != args.iterations
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
                for value in samples
            )
            or type(median) not in (int, float)
            or not math.isfinite(median)
            or median <= 0
        ):
            raise ValueError("invalid aggregate measurement samples")
    expected = {
        "executions": args.warmups + args.iterations,
        "plan_fallbacks": 0,
        "runtime_fallbacks": 0,
    }
    counters = result["gpu"].get("gpu_counters")
    if not isinstance(counters, dict) or any(
        type(counters.get(k)) is not int or counters[k] != v
        for k, v in expected.items()
    ):
        raise ValueError("measurement lacks GPU execution evidence for every query")
    return result


def measure_python(
    args,
    database,
    setup,
    registration,
    reference,
    candidate,
    *,
    adapter_factory,
    benchmark,
):
    previous = os.environ.get("SIRIUS_LOG_DIR")
    os.environ["SIRIUS_LOG_DIR"] = str(database.parent / "measurement-log")
    adapter = None
    try:
        adapter = adapter_factory(database, args.extension)
        adapter.execute("SET enable_duckdb_fallback=false")
        return {
            "measurement": "complete",
            "timing": "query wall time including result fetch",
            "cpu": benchmark(adapter, candidate, args.warmups, args.iterations, False),
            "gpu": benchmark(adapter, candidate, args.warmups, args.iterations, True),
        }
    finally:
        if adapter is not None:
            adapter.close()
        if previous is None:
            os.environ.pop("SIRIUS_LOG_DIR", None)
        else:
            os.environ["SIRIUS_LOG_DIR"] = previous
