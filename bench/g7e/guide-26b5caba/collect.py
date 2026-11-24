# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Read measured timings, GPU boundaries, immutable configs and saved answers."""
import csv
from datetime import datetime, timezone
import json
import math
import re

import yaml
from validation import sha256

STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\]")


def one(paths):
    values = list(paths)
    return values[0] if len(values) == 1 else None


def benchmark(worker):
    return one(worker.glob("benchmark/tpch_*")) if worker.is_dir() else None


def timings(worker):
    directory = benchmark(worker)
    path = directory / "csv/runtimes.csv" if directory else None
    answer = {}
    if path and path.is_file():
        for row in csv.DictReader(path.open()):
            if re.fullmatch(r"q\d+", row.get("query", "")):
                number = float(row["runtime_s"])
                if math.isfinite(number) and number >= 0:
                    key = int(row["query"][1:]), int(row["iteration"]) + 1
                    assert key not in answer
                    answer[key] = number
    return answer


def answer(worker, engine, query, iteration=1):
    directory = benchmark(worker)
    return (
        directory / f"iterations/iteration{iteration}/{engine}/q{query}/result.txt"
        if directory
        else worker / "missing-answer"
    )


def events(worker):
    path = worker / "events.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.is_file()
        else []
    )


def gpu_evidence(worker, query, iteration, journal):
    begins = [
        e
        for e in journal
        if e["event"] == "query_begin"
        and e.get("phase") == "measurement"
        and e.get("query") == query
        and e.get("iteration") == iteration
    ]
    ends = [
        e
        for e in journal
        if e["event"] == "query_end"
        and e.get("phase") == "measurement"
        and e.get("query") == query
        and e.get("iteration") == iteration
    ]
    if len(begins) != 1 or len(ends) != 1:
        return {"correct": False, "detail": "missing unique query execution boundaries"}
    start, finish = begins[0]["wall_ns"] - 1_000_000, ends[0]["wall_ns"] + 1_000_000
    directory = benchmark(worker)
    plans, completed = 0, 0
    for path in sorted(directory.glob("log_dir/*.log")) if directory else []:
        with path.open(errors="replace") as stream:
            for line in stream:
                if (
                    "physical plan replaced with GPU operator" not in line
                    and "Transparent GPU execution: query completed" not in line
                ):
                    continue
                stamp = STAMP.match(line)
                if not stamp:
                    continue
                moment = datetime.strptime(
                    stamp.group(1), "%Y-%m-%d %H:%M:%S.%f"
                ).replace(tzinfo=timezone.utc)
                when = int(moment.timestamp() * 1_000_000_000)
                if start <= when <= finish:
                    plans += "physical plan replaced with GPU operator" in line
                    completed += "Transparent GPU execution: query completed" in line
    return {
        "correct": plans >= 1 and completed >= 1,
        "gpu_plans": plans,
        "gpu_completions": completed,
        "query_begin_ns": begins[0]["wall_ns"],
        "query_end_ns": ends[0]["wall_ns"],
    }


def verify_config(worker, source):
    if not worker.is_dir():
        return False
    original = yaml.safe_load(source.read_text())
    if sha256(worker / "source-config.yaml") != sha256(source):
        return False
    effective = yaml.safe_load((worker / "effective-config.yaml").read_text())
    telemetry = effective["sirius"].pop("telemetry")
    assert telemetry == {
        "enable_quent": True,
        "output_directory": str(worker / "quent"),
    }
    assert effective == original
    saved = one(worker.glob("benchmark/*/config.yml"))
    if saved:
        captured = yaml.safe_load(saved.read_text())
        assert captured["sirius"].pop("telemetry") == telemetry
        assert captured == original
    traces = [p for p in (worker / "quent").rglob("*") if p.is_file()]
    return bool(saved and any(p.suffix == ".qmi" and p.stat().st_size for p in traces))


def pinned_evidence(worker, q, journal, it):
    from tpch_pin_columns import QUERY_COLUMNS

    expected = set(QUERY_COLUMNS[q])
    begins = [e for e in journal if e["event"] == "pin_begin"]
    ends = [e for e in journal if e["event"] == "pin_end"]
    pin_calls = bool(
        len(begins) == len(ends) == 1
        and "tier='host'" in begins[0]["sql"]
        and all(f"name='{t}'" in begins[0]["sql"] for t in expected)
    )
    bb = [e for e in journal if e["event"] == "query_begin" and e["iteration"] == it]
    ee = [e for e in journal if e["event"] == "query_end" and e["iteration"] == it]
    assigned = set()
    fallback = []
    if bb and ee:
        start, end = bb[0]["wall_ns"] - 1000000, ee[0]["wall_ns"] + 1000000
        for p in worker.glob("benchmark/*/log_dir/*.log"):
            for line in p.read_text(errors="replace").splitlines():
                stamp = STAMP.match(line)
                if not stamp:
                    continue
                when = int(
                    datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S.%f")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1e9
                )
                if not start <= when <= end:
                    continue
                match = re.search(r"assigned pinned entry '([^']+)'", line)
                if match:
                    assigned.add(match.group(1))
                if (
                    "error while trying to assign cached entries" in line
                    or "not all the columns are pinned" in line
                ):
                    fallback.append(line)
    return {
        "correct": bool(pin_calls and expected.issubset(assigned) and not fallback),
        "host_pin_calls_verified": pin_calls,
        "expected_tables": sorted(expected),
        "assigned_tables": sorted(assigned),
        "fallback_messages": fallback,
    }
