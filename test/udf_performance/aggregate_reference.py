"""Full-input source-reference gate for the synthetic aggregate stage workloads."""

import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

from run_bench import AGGREGATE_KINDS, AGGREGATE_WORKLOADS, REPO, valid_gpu_counters

ASSETS = REPO / ".agents/skills/sirius-udf/assets/aggregate"


def typed_table(value, types, label):
    """Validate the exact typed cells emitted by these benchmark workloads."""
    if (
        not isinstance(value, dict)
        or value.get("types") != types
        or not isinstance(value.get("rows"), list)
    ):
        raise ValueError(f"{label}: invalid typed result table")
    for row in value["rows"]:
        if not isinstance(row, list) or len(row) != len(types):
            raise ValueError(f"{label}: invalid typed row width")
        for cell, logical_type in zip(row, types):
            if cell is None:
                continue
            if logical_type == "DOUBLE":
                if (
                    not isinstance(cell, dict)
                    or set(cell) != {"ieee_bits"}
                    or not isinstance(cell["ieee_bits"], str)
                    or not re.fullmatch(r"[0-9a-f]{16}", cell["ieee_bits"])
                ):
                    raise ValueError(f"{label}: invalid DOUBLE bit encoding")
            elif (
                not isinstance(cell, str)
                or not re.fullmatch(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)", cell)
                or not -(1 << (63 if logical_type == "BIGINT" else 31))
                <= int(cell)
                < (1 << (63 if logical_type == "BIGINT" else 31))
            ):
                raise ValueError(f"{label}: invalid integer encoding")
    return value


def verify(job, workload, rows, *, adapter=None):
    """Run the unchanged original source before the profiling process exists.

    Input/results retain the skill's versioned typed protocol. Integer values
    are strings, floats are IEEE bit objects, and JSON null represents SQL NULL.
    """
    if workload not in AGGREGATE_WORKLOADS:
        raise ValueError("unknown source aggregate")
    adapter = ASSETS / "oracle.py" if adapter is None else Path(adapter)
    document = json.loads((job / "aggregate-preflight.json").read_text())
    if (
        not isinstance(document, dict)
        or type(document.get("version")) is not int
        or document["version"] != 1
    ):
        raise ValueError("aggregate source input has incorrect version or types")
    source = typed_table(
        document.get("input"),
        ["BIGINT", "DOUBLE", "DOUBLE", "INTEGER", "INTEGER"],
        "aggregate source input",
    )
    if not isinstance(source.get("rows"), list) or len(source["rows"]) != rows:
        raise ValueError("aggregate source gate must inspect every input row")
    if any(
        not isinstance(row, list) or len(row) != 5 or row[0] != str(index)
        for index, row in enumerate(source["rows"])
    ):
        raise ValueError("aggregate source input is not complete and ordered")
    cases = document.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != len(AGGREGATE_KINDS)
        or any(not isinstance(case, dict) for case in cases)
        or any(not isinstance(case.get("query_kind"), str) for case in cases)
        or {case["query_kind"] for case in cases} != set(AGGREGATE_KINDS)
    ):
        raise ValueError("aggregate source gate has missing or duplicate query kinds")
    proofs = []
    for case in cases:
        kind = case["query_kind"]
        if not valid_gpu_counters(case.get("gpu_counters"), 1):
            raise ValueError("source gate lacks one GPU execution without fallback")
        result_types = ["DOUBLE"] if kind == "global" else ["INTEGER", "DOUBLE"]
        for engine in ("cpu", "gpu"):
            typed_table(
                case.get(f"candidate_{engine}"), result_types, f"candidate {engine}"
            )
        parameters = {
            "aggregate": workload,
            "value_columns": [1] if workload == "sum_squares" else [1, 2],
            "group_columns": (
                [] if kind == "global" else [3 if kind == "grouped_uniform" else 4]
            ),
            "merge_partitions": 7,
        }
        request = {
            "version": 1,
            "case_id": kind,
            "input": source,
            "parameters": parameters,
        }
        request_file = job / f"source-reference-{kind}.request.json"
        result_file = job / f"source-reference-{kind}.response.json"
        with request_file.open("w") as stream:
            json.dump(request, stream, allow_nan=False, separators=(",", ":"))
        with request_file.open() as input_stream, result_file.open(
            "w"
        ) as output_stream:
            process = subprocess.run(
                [sys.executable, str(adapter)],
                stdin=input_stream,
                stdout=output_stream,
                stderr=subprocess.PIPE,
                text=True,
                timeout=900,
                check=False,
            )
        (job / f"source-reference-{kind}.stderr.txt").write_text(process.stderr)
        if process.returncode:
            raise ValueError(f"original source adapter failed for {kind}")
        expected = json.loads(result_file.read_text())
        if (
            not isinstance(expected, dict)
            or type(expected.get("version")) is not int
            or expected["version"] != 1
            or "error" in expected
        ):
            raise ValueError(f"invalid original source response for {kind}")
        typed_table(expected.get("result"), result_types, f"original source {kind}")
        for engine in ("cpu", "gpu"):
            if expected["result"] != case.get(f"candidate_{engine}"):
                raise ValueError(
                    f"original source differs from {engine} SQL for {kind}"
                )
        proofs.append(
            {
                "query_kind": kind,
                "input_rows_compared": rows,
                "result_rows_compared": len(expected["result"]["rows"]),
                "comparison": "exact_logical_types_nulls_and_ieee_bits",
                "merge_partitions": 7,
                "gpu_counters": case["gpu_counters"],
            }
        )
    evidence = {
        "version": 1,
        "correctness_verified": True,
        "workload": workload,
        "input_rows": rows,
        "reference_basis": "independently_authored_executable_original",
        "sources": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ASSETS / "original.py", adapter)
            if path.is_relative_to(REPO)
        },
        "cases": proofs,
    }
    (job / "source-reference-verdict.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )
    return evidence
