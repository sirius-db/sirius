#!/usr/bin/env python3
"""Attribute successful Quent tier moves to sequential benchmark queries.

The input spec lists runs with their Quent session directories and matching
benchmark runtimes.csv files. Query labels are absent in these saved traces, so
the script pairs Quent query windows with runtime rows in execution order. It
rejects missing, overlapping, or unmatched windows and tier moves outside them.
"""

import argparse
import bisect
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

GIB = 1024**3
TIER = re.compile(r"tier=(GPU|HOST|DISK)")
MOVES = ("GPU->HOST", "HOST->DISK", "GPU->DISK")


def records(directory, kind):
    files = list((directory / kind).glob("*.ndjson"))
    if len(files) != 1:
        raise ValueError(f"Expected one {kind} file in {directory}, found {len(files)}")
    with files[0].open() as file:
        for line in file:
            yield json.loads(line)


def query_windows(directory, runtime_csv):
    events = {}
    for event in records(directory, "Query"):
        state = next(iter(event["data"]))
        if state in ("Init", "Finished"):
            events.setdefault(event["id"], {})[state] = event["timestamp"]
    if any(set(event) != {"Init", "Finished"} for event in events.values()):
        raise ValueError(f"Incomplete Quent query windows in {directory}")
    windows = sorted(events.values(), key=lambda event: event["Init"])
    with runtime_csv.open(newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["engine"] == "sirius" and re.fullmatch(r"q\d+", row["query"])
        ]
    if len(windows) != len(rows):
        raise ValueError(
            f"Quent/runtime query count differs for {directory}: "
            f"{len(windows)} versus {len(rows)}"
        )
    for index, (window, row) in enumerate(zip(windows, rows)):
        if window["Init"] >= window["Finished"]:
            raise ValueError(f"Invalid query window {index} in {directory}")
        if index and windows[index - 1]["Finished"] > window["Init"]:
            raise ValueError(f"Overlapping query windows in {directory}")
        duration = (window["Finished"] - window["Init"]) / 1e9
        if abs(float(row["runtime_s"]) - duration) > 1:
            raise ValueError(f"Quent/runtime duration mismatch at query {index}")
    return windows, rows


def memory_tiers(directory):
    tiers = {}
    for event in records(directory, "MemorySpace"):
        declaration = event["data"].get("Declaration")
        if declaration is None:
            continue
        match = TIER.search(declaration["label"])
        if match:
            tiers[event["id"]] = match.group(1)
    if not {"GPU", "HOST", "DISK"}.issubset(tiers.values()):
        raise ValueError(f"Missing Quent memory tier in {directory}")
    return tiers


def session_rows(directory, runtime_csv, scale_factor, profile):
    windows, timings = query_windows(directory, runtime_csv)
    tiers = memory_tiers(directory)
    starts = [window["Init"] for window in windows]
    moves = [defaultdict(int) for _ in windows]
    pending = {}
    for event in records(directory, "DataBatch"):
        batch = event["id"]
        state = event["data"]
        if "InTransit" in state:
            transit = state["InTransit"]
            pending[batch] = (
                event["timestamp"],
                transit["source_memory"]["target"],
                transit["dest_memory"]["target"],
            )
        elif "Stationary" in state and batch in pending:
            started, source, destination = pending.pop(batch)
            stationary = state["Stationary"]["memory"]
            if stationary["target"] != destination:
                continue  # A failed conversion returned to its source tier.
            direction = f"{tiers[source]}->{tiers[destination]}"
            if direction not in MOVES:
                continue
            finished = event["timestamp"]
            index = bisect.bisect_right(starts, finished) - 1
            if (
                index < 0
                or started < windows[index]["Init"]
                or finished > windows[index]["Finished"]
            ):
                raise ValueError(f"Tier move outside a query window in {directory}")
            moves[index][direction] += stationary["data"]["bytes"]
    if pending:
        raise ValueError(f"Unfinished batch conversions in {directory}: {len(pending)}")

    result = []
    for timing, counts in zip(timings, moves):
        host = counts["GPU->HOST"]
        disk = counts["HOST->DISK"] + counts["GPU->DISK"]
        result.append(
            {
                "scale_factor": scale_factor,
                "profile": profile,
                "query": timing["query"],
                "iteration": int(timing["iteration"]),
                "moved_into_host_bytes": host,
                "moved_into_disk_bytes": disk,
                "moved_into_host_gib": f"{host / GIB:.6f}",
                "moved_into_disk_gib": f"{disk / GIB:.6f}",
            }
        )
    return result


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    executions = []
    for run in spec["runs"]:
        for session in run["sessions"]:
            executions.extend(
                session_rows(
                    Path(session["trace_dir"]),
                    Path(session["runtime_csv"]),
                    run["scale_factor"],
                    run["profile"],
                )
            )
    profile_order = {"cold": 0, "hot": 1, "pin": 2}
    executions.sort(
        key=lambda row: (
            row["scale_factor"],
            profile_order[row["profile"]],
            int(row["query"][1:]),
            row["iteration"],
        )
    )
    totals = defaultdict(lambda: [0, 0, 0])
    for row in executions:
        key = (row["scale_factor"], row["profile"], row["query"])
        totals[key][0] += 1
        totals[key][1] += row["moved_into_host_bytes"]
        totals[key][2] += row["moved_into_disk_bytes"]
    per_query = []
    for (scale, profile, query), (count, host, disk) in sorted(
        totals.items(),
        key=lambda item: (
            item[0][0],
            profile_order[item[0][1]],
            int(item[0][2][1:]),
        ),
    ):
        per_query.append(
            {
                "scale_factor": scale,
                "profile": profile,
                "query": query,
                "executions": count,
                "moved_into_host_bytes": host,
                "moved_into_disk_bytes": disk,
                "moved_into_host_gib": f"{host / GIB:.6f}",
                "moved_into_disk_gib": f"{disk / GIB:.6f}",
            }
        )
    common = (
        "scale_factor",
        "profile",
        "query",
        "moved_into_host_bytes",
        "moved_into_disk_bytes",
        "moved_into_host_gib",
        "moved_into_disk_gib",
    )
    write_csv(
        args.output_dir / "per-execution.csv",
        executions,
        (*common[:3], "iteration", *common[3:]),
    )
    write_csv(
        args.output_dir / "per-query.csv",
        per_query,
        (*common[:3], "executions", *common[3:]),
    )
    print(f"Wrote {len(executions)} executions and {len(per_query)} query totals")


if __name__ == "__main__":
    main()
