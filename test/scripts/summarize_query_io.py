#!/usr/bin/env python3
"""Join local file traces and query observations; reject incomplete I/O coverage."""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import re

from compare_query_cost import parse_backend_reads, parse_observations, file_sha256
from trace_query_io import joined_lines, parse_call, query_windows


PHASES = ("planning", "preparation", "execution")
DOMAINS = ("sql_filesystem", "datasource", "direct_puffin")


def decode_path(value):
    path = bytes.fromhex(value).decode()
    if path.startswith("file://"):
        path = path[7:]
    if not path.startswith("/"):
        raise ValueError("I/O coverage requires absolute local paths")
    return path


def summarize(observation, calls):
    if observation.get("datasource_backend") != ["uring"]:
        raise ValueError("local io_uring backend coverage is required")
    if observation.get("backend_io") != ["local_uring_pread_only"]:
        raise ValueError("missing local backend counters")
    result = {
        domain: {
            phase: {"logical": Counter(), "physical": Counter()} for phase in PHASES
        }
        for domain in DOMAINS
    }
    scopes, sql_paths, puffin_paths = [], set(), {}
    datasource_paths = set()
    for value in observation.get("datasource_open", []):
        phase, tid, start, end, path = value.split(":")
        path = decode_path(path)
        if phase not in PHASES or int(end) < int(start):
            raise ValueError("invalid datasource open scope")
        scopes.append((phase, tid, int(start), int(end), path))
        datasource_paths.add(path)
    for value in observation.get("sql_file", []):
        phase, path, opens, reads, size = value.split(":")
        if phase not in PHASES:
            raise ValueError(
                "SQL filesystem preparation/execution attribution is unresolved"
            )
        sql_paths.add(decode_path(path))
        result["sql_filesystem"][phase]["logical"].update(
            opens=int(opens), requests=int(reads), bytes=int(size)
        )
    for value in observation.get("puffin_file", []):
        phase, path, opens, reads, requested, returned, failures = value.split(":")
        path = decode_path(path)
        if phase not in PHASES or (
            path in puffin_paths and puffin_paths[path] != phase
        ):
            raise ValueError("ambiguous Puffin phase")
        puffin_paths[path] = phase
        result["direct_puffin"][phase]["logical"].update(
            opens=int(opens),
            requests=int(reads),
            bytes_requested=int(requested),
            bytes_returned=int(returned),
            failures=int(failures),
        )
    if not sql_paths or not puffin_paths or not scopes:
        raise ValueError("missing required DV workload domain observations")
    if (
        sql_paths & datasource_paths
        or sql_paths & puffin_paths.keys()
        or datasource_paths & puffin_paths.keys()
    ):
        raise ValueError("shared input paths prevent unambiguous domain attribution")
    backend = parse_backend_reads([observation])[0]["phases"]
    for phase in PHASES:
        if backend[phase] is None:
            raise ValueError("missing backend phase")
        result["datasource"][phase]["physical"].update(backend[phase])
        raw = observation.get(f"datasource_{phase}")
        if not raw or len(raw) != 1:
            raise ValueError("missing logical datasource phase")
        reads, requested, returned, failures = map(int, raw[0].split(":"))
        opens = int(observation.get(f"datasource_{phase}_open_attempts", ["0"])[0])
        result["datasource"][phase]["logical"].update(
            opens=opens,
            requests=reads,
            bytes_requested=requested,
            bytes_returned=returned,
            failures=failures,
        )
        if opens != sum(scope[0] == phase for scope in scopes):
            raise ValueError("incomplete datasource open scopes")
    backend_traced_reads = Counter()
    for call in calls:
        path = call["path"]
        candidates = [
            scope
            for scope in scopes
            if scope[1] == call["pid"]
            and scope[4] == path
            and scope[2] <= int(call["start_us"])
            and int(call["end_us"]) <= scope[3]
        ]
        if len(candidates) > 1:
            raise ValueError("overlapping datasource scopes")
        if candidates:
            domain, phase = "datasource", candidates[0][0]
            if call["call"] != "openat":
                raise ValueError(
                    "unexpected physical read inside local datasource open"
                )
        elif (
            path in puffin_paths
            and path not in sql_paths
            and path not in datasource_paths
        ):
            domain, phase = "direct_puffin", puffin_paths[path]
        elif path in sql_paths:
            domain, phase = "sql_filesystem", call["phase"]
            if phase not in PHASES:
                raise ValueError("unresolved SQL filesystem physical phase")
        elif path in datasource_paths and call["call"] != "openat":
            # pread is already included in the backend counters. Do not double count it.
            backend_traced_reads.update(
                requests=1, bytes_returned=int(call["returned_bytes"])
            )
            continue
        else:
            raise ValueError(f"unclassified input access: {call['call']} {path}")
        physical = result[domain][phase]["physical"]
        if call["call"] == "openat":
            physical["opens"] += 1
            physical["open_failures"] += call["status"] == "failed"
        else:
            requested, returned = int(call["requested_bytes"]), int(
                call["returned_bytes"]
            )
            physical.update(
                requests=1,
                bytes_requested=requested,
                bytes_returned=returned,
                failures=int(call["status"] == "failed"),
                short_reads=int(call["status"] == "ok" and returned < requested),
            )
    for key, value in backend_traced_reads.items():
        if value > sum(backend[phase][key] for phase in PHASES):
            raise ValueError("traced datasource reads exceed backend observations")
    for phase in PHASES:
        puffin = result["direct_puffin"][phase]
        if puffin["logical"]["opens"] != puffin["physical"]["opens"]:
            raise ValueError("Puffin open observations do not cover the physical opens")
    # Explicit zeros are valid only after every selected syscall has been attributed.
    for domain in DOMAINS:
        for phase in PHASES:
            for field in (
                "opens",
                "requests",
                "bytes_requested",
                "bytes_returned",
                "failures",
            ):
                result[domain][phase]["physical"].setdefault(field, 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = json.loads((args.trace / "summary.json").read_text())
    if trace.get("returncode") != 0:
        raise ValueError(
            "a failed or unverified traced query cannot establish coverage"
        )
    log = (args.trace / "run.log").read_text()
    observations = [
        row for row in parse_observations(log) if row.get("observations") == ["enabled"]
    ]
    if not observations:
        raise ValueError("no observed queries")
    if len({row["sample"][0] for row in observations}) != len(observations):
        raise ValueError("duplicate observed sample ids in one trace")
    with (args.trace / "calls.csv").open() as source:
        calls = list(csv.DictReader(source))
    # A selected input accessed through an unsupported syscall must not disappear.
    windows = [w for w in query_windows(log) if w[3] == "enabled"]
    source = Path(trace["trace_source"])
    raw_files = sorted(source.glob("trace.*"))
    if not raw_files:
        raise ValueError("raw traces are required for the coverage check")
    for file in raw_files:
        counts = Counter()
        for line in joined_lines(file.read_text().splitlines(), counts):
            if not re.search(
                r"\b(openat|read|pread64|readv|preadv|preadv2|mmap)\(", line
            ):
                continue
            if parse_call(line) is None and any(
                prefix in line for prefix in trace["path_prefixes"]
            ):
                timestamp = int(Decimal(line.split()[0]) * 1000000)
                if any(start <= timestamp <= end for start, end, *_ in windows):
                    raise ValueError("unsupported syscall accesses measured input")
        if counts["unparsed"]:
            raise ValueError(
                "unfinished or unmatched trace calls prevent complete coverage"
            )
    samples = []
    for call in calls:
        if call["observations"] == "unobserved" and any(
            int(call["start_us"]) <= end and int(call["end_us"]) >= start
            for start, end, *_ in windows
        ):
            raise ValueError("input access crosses a measured query boundary")
    for row in observations:
        selected = [
            call
            for call in calls
            if call["observations"] == "enabled" and call["sample"] == row["sample"][0]
        ]
        samples.append(
            {
                "sample": row["sample"][0],
                "cache": row["cache"][0],
                "domains": summarize(row, selected),
            }
        )
    report = {
        "coverage": "complete_for_observed_local_dv_queries",
        "samples": samples,
        "scope": trace["path_prefixes"],
        "run_log_sha256": file_sha256(args.trace / "run.log"),
        "calls_sha256": file_sha256(args.trace / "calls.csv"),
        "units": "OS read returns, not disk or controller traffic",
        "retry_accounting": "every syscall attempt is counted; local backend also reports retries separately",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified three I/O domains for {len(samples)} observed queries")


if __name__ == "__main__":
    main()
