#!/usr/bin/env python3
"""D9 private normalized causal model, not Quent wire or GPU qualification.

All tokens in this model require future trusted collector extraction/source
proof. In particular native-origin edges cannot be inferred from generic input
stamping. Even a consistent synthetic model never qualifies an actual execution.
"""

import argparse
from decimal import Decimal, DecimalException
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1] / "cpp/integration/data/paimon_conformance"
    ),
)
from run_conformance import compare_rows


class EvidenceError(ValueError):
    pass


def reject(code):
    raise EvidenceError(code)


def same(left, right):
    """Strict equality: JSON true must not impersonate integer one."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same(left[k], right[k]) for k in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def fields(value, expected, code):
    if not isinstance(value, dict) or any(
        k not in value or not same(value[k], v) for k, v in expected.items()
    ):
        reject(code)


def number(value, minimum=0):
    if type(value) is not int or value < minimum:
        reject("invalid_integer")
    return value


def token(value):
    if not isinstance(value, str) or not value:
        reject("missing_identity")
    return value


def check_model(expected, trace):
    scope, binding = expected["scope"], expected["binding"]
    for name in ("run", "runtime", "statement"):
        token(scope[name])
    number(scope["process"], 1)
    for name in ("sql", "oracle", "inputs", "artifacts"):
        if not isinstance(scope[name], str) or not re.fullmatch(
            "[0-9a-f]{64}", scope[name]
        ):
            reject("invalid_digest")
    window = number(expected["window"])
    for name in ("contract", "materialization_generation"):
        number(binding[name], 1)
    for name in ("occurrence", "scan_operator"):
        number(binding[name])
    for name in ("read_owner", "producer", "certified_view", "pipeline"):
        token(binding[name])
    columns = expected["result"]["columns"]
    if not isinstance(columns, list) or not columns:
        reject("invalid_oracle_columns")
    for column in columns:
        if (
            not isinstance(column, list)
            or len(column) != 2
            or not isinstance(column[0], str)
            or not column[0]
            or not isinstance(column[1], str)
        ):
            reject("invalid_oracle_columns")
        type_name = column[1]
        decimal = re.fullmatch(r"DECIMAL\(([1-9][0-9]?),([0-9]+)\)", type_name)
        if type_name not in ("BIGINT", "VARCHAR", "DATE") and not (
            decimal and 0 <= int(decimal[2]) <= int(decimal[1]) <= 38
        ):
            reject("invalid_oracle_columns")
    if not expected["result"]["rows"]:
        reject("nonempty_first_slice_required")
    if not same(trace["scope"], scope):
        reject("execution_identity")
    query = trace["query"]
    uuid, plan = token(query["uuid"]), token(query["plan"])
    fields(
        query,
        {"label": scope["statement"], "window": window},
        "execution_window",
    )
    number(query["finalize_generation"])
    fields(trace["mapping"], {"uuid": uuid, "window": window}, "window_uuid_mapping")
    if not same(
        trace["runtime"],
        {
            "scope": "whole_runtime",
            "before_active": [],
            "during_windows": [window],
            "after_active": [],
            "other_work": [],
            "drained": True,
        },
    ):
        reject("runtime_exclusivity")
    if not same(
        trace["settings"],
        {
            "gpu_execution": True,
            "enable_duckdb_fallback": False,
            "plan_fallbacks": 0,
            "runtime_fallbacks": 0,
        },
    ):
        reject("gpu_and_fallback")
    sources = trace["sources"]
    if not isinstance(sources, list) or len(sources) != 1:
        reject("single_source_only")
    fields(sources[0], {"query": uuid, "window": window, **binding}, "source_binding")
    # Separate actual-native acquisition/ownership relation from generic driver
    # stamps. Current 2e9/f139 cannot supply qualified evidence for these edges.
    native = trace["native_origin"]
    fields(
        native,
        {
            "query": uuid,
            "window": window,
            "certified_view": binding["certified_view"],
            "factory_view": binding["certified_view"],
            "sdk_read_view": binding["certified_view"],
            "read_owner": binding["read_owner"],
            "producer": binding["producer"],
        },
        "native_origin",
    )
    owners = native["batch_owners"]
    if (
        not isinstance(owners, list)
        or not owners
        or any(not isinstance(v, str) or not v for v in owners)
        or len(set(owners)) != len(owners)
    ):
        reject("native_batch_ownership")
    units, batch_units = {}, {}
    validations = trace["validations"]
    if not isinstance(validations, list) or not validations:
        reject("missing_actual_input")
    for item in validations:
        fields(
            item,
            {
                "query": uuid,
                "window": window,
                "contract": binding["contract"],
                "occurrence": binding["occurrence"],
                "generation": binding["materialization_generation"],
                "read_owner": binding["read_owner"],
                "producer": binding["producer"],
            },
            "input_binding",
        )
        unit, rows = number(item["unit"]), number(item["rows"])
        if type(item["rootless"]) is not bool:
            reject("invalid_rootless")
        owner = item.get("batch_owner")
        if item["rootless"]:
            if rows:
                reject("rootless_rows")
        elif owner not in owners:
            reject("unattributed_batch")
        else:
            # Private model identity denotes one immutable batch, not a pool or
            # lifetime shared by several batches. No production ID API implied.
            if owner in batch_units and batch_units[owner] != unit:
                reject("ambiguous_batch_unit")
            batch_units[owner] = unit
        canonical = (rows, item["rootless"], owner)
        if unit in units and not same(units[unit], canonical):
            reject("contradictory_unit")
        units[unit] = canonical
    if set(batch_units) != set(owners):
        reject("unvalidated_native_batch")
    data = [value for value in units.values() if not value[1]]
    if not data or sum(value[0] for value in data) <= 0:
        reject("no_nonrootless_data")
    before, after = trace["imports"]["before"], trace["imports"]["after"]
    batches = number(after["batches"]) - number(before["batches"])
    rows = number(after["rows"]) - number(before["rows"])
    # Under independently established whole-runtime/source/scan premises, this
    # establishes only some import activity. Same-unit OOM retries can count
    # again; aggregate deltas cannot establish per-unit successful consumption.
    if batches <= 0 or rows <= 0:
        reject("import_delta")
    tasks = trace["scan_tasks"]
    if not isinstance(tasks, list) or not tasks:
        reject("missing_gpu_scan_task")
    task_ids = set()
    for task in tasks:
        fields(
            task,
            {
                "query": uuid,
                "plan": plan,
                "pipeline": binding["pipeline"],
                "operator": binding["scan_operator"],
                "success": True,
            },
            "gpu_scan_task",
        )
        task_id = token(task["task"])
        if task_id in task_ids:
            reject("ambiguous_task")
        task_ids.add(task_id)
        number(task["gpu"])
        if number(task["finalizing"]) <= number(task["computing"]):
            reject("task_causal_order")
    if not same(
        trace["transport"],
        {"closed": True, "dropped": 0, "truncated": False, "ambiguous": False},
    ):
        reject("transport_incomplete")
    if not same(
        trace["process"], {"termination": "exited", "exit": 0, "timeout": False}
    ):
        reject("process_failure")
    result = trace["result"]
    fields(
        result,
        {
            "query": uuid,
            "complete": True,
            "error": None,
            "columns": expected["result"]["columns"],
        },
        "result_terminal",
    )
    try:
        compare_rows(expected["result"], result["rows"])
    except (AssertionError, ValueError, TypeError, KeyError, DecimalException):
        reject("result_oracle")
    fields(
        trace["cleanup"],
        {
            "query": uuid,
            "status": "ok",
            "drained": True,
            "probe": [{"probe": "LIVENESS_OK"}],
        },
        "cleanup_terminal",
    )


def read_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                reject("duplicate_json_key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=unique,
        parse_float=Decimal,
        parse_constant=lambda _: reject("non_json_number"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.expected.resolve(), args.trace.resolve()):
        parser.error("output cannot overwrite an input")
    report = {
        "decision": "REJECTED",
        "scope": "synthetic normalized causal model only",
        "actual_gpu_qualified": False,
    }
    try:
        output = args.output.open("x")
    except OSError:
        parser.error("output must be a new writable file")
    with output:
        try:
            inputs = {}
            for label, path in (("expected", args.expected), ("trace", args.trace)):
                raw = path.read_bytes()
                report[label + "_sha256"] = hashlib.sha256(raw).hexdigest()
                inputs[label] = read_json(raw)
            check_model(inputs["expected"], inputs["trace"])
            report["decision"] = "CONSISTENT_OFFLINE_MODEL"
        except EvidenceError as error:
            report["reason"] = str(error)
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            report["reason"] = "malformed_or_missing_evidence"
        report["checker_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        output.write(json.dumps(report, indent=2) + "\n")
    return 0 if report["decision"] == "CONSISTENT_OFFLINE_MODEL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
