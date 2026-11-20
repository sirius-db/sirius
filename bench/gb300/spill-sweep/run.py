#!/usr/bin/env python3
"""Run unpinned TPC-H queries across GB300 host-memory caps and count tier moves."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
BASE_CONFIG = ROOT / "bench/gb300/configs/cold.yaml"
BENCHMARK = ROOT / "test/tpch_performance/performance_test.py"
TIER = re.compile(r"tier=(GPU|HOST|DISK)")
GIB = 1024**3
FIELDS = (
    "scale_factor",
    "host_cap_gib",
    "query",
    "status",
    "runtime_s",
    "gpu_to_host_bytes",
    "host_to_disk_bytes",
    "gpu_to_disk_bytes",
    "error",
)
MOVE_FIELDS = ("gpu_to_host_bytes", "host_to_disk_bytes", "gpu_to_disk_bytes")


def parse_numbers(value, *, maximum=None):
    numbers = []
    for part in value.split(","):
        number = int(part)
        if number <= 0 or (maximum is not None and number > maximum):
            raise ValueError(f"invalid number: {number}")
        if number not in numbers:
            numbers.append(number)
    return numbers


def make_config(cap_gib, trace_dir):
    # The saved 450 GiB config starts 225 pools of 512 x 1 MiB blocks:
    # 112.5 GiB, one quarter of the host cap. Keep that ratio at smaller caps.
    initial_pools = max(1, cap_gib // 2)
    source = BASE_CONFIG.read_text()
    source, capacity_count = re.subn(
        r"(?m)^(\s*capacity_bytes:) 450Gi$",
        rf"\g<1> {cap_gib}Gi",
        source,
    )
    source, pools_count = re.subn(
        r"(?m)^(\s*initial_number_pools:) 225$",
        rf"\g<1> {initial_pools}",
        source,
    )
    if (capacity_count, pools_count) != (1, 1):
        raise RuntimeError("GB300 cold config changed; review the sweep substitutions")
    return source.rstrip() + (
        "\n  telemetry:\n"
        "    enable_quent: true\n"
        f"    output_directory: {trace_dir}\n"
    )


def records(session, kind):
    files = list((session / kind).glob("*.ndjson"))
    if len(files) != 1:
        raise ValueError(f"expected one {kind} trace in {session}, found {len(files)}")
    with files[0].open() as stream:
        for line in stream:
            yield json.loads(line)


def measure_moves(trace_dir, runtime_csv, query):
    sessions = [path for path in trace_dir.iterdir() if path.is_dir()]
    if len(sessions) != 1:
        raise ValueError(f"expected one Quent session, found {len(sessions)}")
    session = sessions[0]
    with runtime_csv.open(newline="") as stream:
        timings = [
            row
            for row in csv.DictReader(stream)
            if row["engine"] == "sirius" and row["query"] == f"q{query}"
        ]
    if len(timings) != 1:
        raise ValueError(f"expected one q{query} runtime, found {len(timings)}")
    windows = {}
    for event in records(session, "Query"):
        state = next(iter(event["data"]))
        if state in ("Init", "Finished"):
            windows.setdefault(event["id"], {})[state] = event["timestamp"]
    if len(windows) != 1 or set(next(iter(windows.values()))) != {"Init", "Finished"}:
        raise ValueError(f"expected one complete q{query} Quent window")
    window = next(iter(windows.values()))
    runtime_s = float(timings[0]["runtime_s"])
    if abs((window["Finished"] - window["Init"]) / 1e9 - runtime_s) > 1:
        raise ValueError(f"q{query} Quent/runtime duration mismatch")
    tiers = {}
    for event in records(session, "MemorySpace"):
        declaration = event["data"].get("Declaration")
        if declaration is not None:
            match = TIER.search(declaration["label"])
            if match:
                tiers[event["id"]] = match.group(1)
    if not {"GPU", "HOST", "DISK"}.issubset(tiers.values()):
        raise ValueError("Quent trace lacks a GPU, HOST, or DISK memory space")
    pending = {}
    moves = {("GPU", "HOST"): 0, ("HOST", "DISK"): 0, ("GPU", "DISK"): 0}
    for event in records(session, "DataBatch"):
        batch, state = event["id"], event["data"]
        if "InTransit" in state:
            transit = state["InTransit"]
            pending[batch] = (
                event["timestamp"],
                transit["source_memory"]["target"],
                transit["dest_memory"]["target"],
            )
        elif "Stationary" in state and batch in pending:
            started, source, dest = pending.pop(batch)
            stationary = state["Stationary"]["memory"]
            if stationary["target"] != dest:
                continue  # Failed conversion returned to its source.
            direction = (tiers[source], tiers[dest])
            if direction not in moves:
                continue
            if started < window["Init"] or event["timestamp"] > window["Finished"]:
                raise ValueError("completed tier move falls outside query window")
            size = stationary["data"]["bytes"]
            moves[direction] += size
    if pending:
        raise ValueError(f"{len(pending)} unfinished batch conversions")
    return runtime_s, *(moves[direction] for direction in moves)


def find_runtime_csv(benchmark_root):
    paths = list(benchmark_root.glob("*/csv/runtimes.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected one benchmark runtime CSV, found {len(paths)}")
    return paths[0]


def confirm_gpu_plan(benchmark_root, query):
    paths = list(benchmark_root.glob(f"*/sirius/q{query}/sirius.log"))
    if len(paths) != 1:
        raise ValueError(f"expected one q{query} Sirius log, found {len(paths)}")
    if "Transparent execution: physical plan replaced with GPU operator" not in paths[
        0
    ].read_text(errors="replace"):
        raise ValueError(f"q{query} did not confirm a GPU plan in the Sirius log")


def write_reports(output, sf, caps, queries):
    rows = []
    for cap in caps:
        for query in queries:
            path = output / f"sf{sf}" / f"host{cap}Gi" / f"q{query:02}" / "result.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
    report_dir = output / f"sf{sf}"
    report_dir.mkdir(parents=True, exist_ok=True)
    temporary = report_dir / "per-query.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(report_dir / "per-query.csv")
    totals = []
    for cap in caps:
        cap_rows = [row for row in rows if row["host_cap_gib"] == cap]
        successes = [row for row in cap_rows if row["status"] == "success"]
        totals.append(
            {
                "scale_factor": sf,
                "host_cap_gib": cap,
                "completed_queries": len(cap_rows),
                "successful_queries": len(successes),
                "requested_queries": len(queries),
                **{
                    field: sum(row[field] for row in successes) for field in MOVE_FIELDS
                },
            }
        )
    temporary = report_dir / "totals.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=totals[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(totals)
    temporary.replace(report_dir / "totals.csv")


def run_query(sf, cap, query, output, timeout):
    query_dir = output / f"sf{sf}" / f"host{cap}Gi" / f"q{query:02}"
    if query_dir.exists() and any(query_dir.iterdir()):
        query_dir.rename(
            query_dir.with_name(f"{query_dir.name}.previous-{time.time_ns()}")
        )
    query_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = query_dir / "quent"
    trace_dir.mkdir(exist_ok=True)
    config = query_dir / "config.yaml"
    config.write_text(make_config(cap, trace_dir))
    benchmark_root = query_dir / "benchmark"
    command = [
        sys.executable,
        str(BENCHMARK),
        "--input",
        str(ROOT / f"test_datasets/tpch_parquet_sf{sf}"),
        "--scale-factor",
        str(sf),
        "--engine",
        "gpu",
        "--pin",
        "none",
        "--iterations",
        "1",
        "--queries",
        str(query),
        "--config",
        str(config),
        "--output",
        str(benchmark_root),
        "--name",
        f"host{cap}Gi_q{query}",
    ]
    (query_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = {
        "scale_factor": sf,
        "host_cap_gib": cap,
        "query": f"q{query}",
        "status": "benchmark_failed",
        "runtime_s": None,
        **{field: None for field in MOVE_FIELDS},
        "error": "",
    }
    with (query_dir / "runner.log").open("w") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            if completed.returncode:
                result["error"] = f"benchmark exit {completed.returncode}"
            else:
                try:
                    confirm_gpu_plan(benchmark_root, query)
                    runtime, gpu_host, host_disk, gpu_disk = measure_moves(
                        trace_dir, find_runtime_csv(benchmark_root), query
                    )
                    result.update(
                        status="success",
                        runtime_s=runtime,
                        gpu_to_host_bytes=gpu_host,
                        host_to_disk_bytes=host_disk,
                        gpu_to_disk_bytes=gpu_disk,
                    )
                except (ValueError, KeyError, OSError) as exc:
                    result.update(status="telemetry_failed", error=str(exc))
        except subprocess.TimeoutExpired:
            result["error"] = f"benchmark exceeded {timeout}s timeout"
    (query_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def recount_from_traces(output, sf, caps, queries):
    """Rebuild saved counts from Quent traces, verifying any older totals."""
    for cap in caps:
        for query in queries:
            query_dir = output / f"sf{sf}" / f"host{cap}Gi" / f"q{query:02}"
            path = query_dir / "result.json"
            result = json.loads(path.read_text())
            if result["status"] != "success":
                raise ValueError(f"cannot recount unsuccessful {path}")
            runtime, gpu_host, host_disk, gpu_disk = measure_moves(
                query_dir / "quent", find_runtime_csv(query_dir / "benchmark"), query
            )
            if abs(runtime - result["runtime_s"]) > 1e-6:
                raise ValueError(f"runtime changed for {path}")
            if "moved_into_host_bytes" in result:
                if (
                    result["moved_into_host_bytes"] != gpu_host
                    or result["moved_into_disk_bytes"] != host_disk + gpu_disk
                ):
                    raise ValueError(f"old totals do not reconcile for {path}")
            result.pop("moved_into_host_bytes", None)
            result.pop("moved_into_disk_bytes", None)
            result.update(
                gpu_to_host_bytes=gpu_host,
                host_to_disk_bytes=host_disk,
                gpu_to_disk_bytes=gpu_disk,
            )
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(path)
        print(f"SF{sf} host={cap}Gi: recounted {len(queries)} queries", flush=True)
    write_reports(output, sf, caps, queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=int, required=True)
    parser.add_argument("--caps", default="32,64,128,256,450")
    parser.add_argument("--queries", default=",".join(map(str, range(1, 23))))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "test_datasets/gb300-spill-sweep"
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recount-from-traces", action="store_true")
    args = parser.parse_args()
    caps = parse_numbers(args.caps)
    queries = parse_numbers(args.queries, maximum=22)
    output = args.output.resolve()
    if args.recount_from_traces:
        recount_from_traces(output, args.sf, caps, queries)
        return
    dataset = ROOT / f"test_datasets/tpch_parquet_sf{args.sf}"
    if not dataset.is_dir():
        parser.error(f"dataset missing: {dataset}")
    if not (
        ROOT / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
    ).is_file():
        parser.error("built Sirius extension is missing")
    print(
        f"{datetime.now(timezone.utc).isoformat()} SF{args.sf}: caps={caps}, "
        f"queries={queries}, output={output}",
        flush=True,
    )
    if args.dry_run:
        for cap in caps:
            print(
                f"host {cap} GiB: initial pools={max(1, cap // 2)}; "
                f"initial allocation={max(1, cap // 2) / 2:.1f} GiB",
                flush=True,
            )
        return
    for cap in caps:
        for query in queries:
            result_path = (
                output
                / f"sf{args.sf}"
                / f"host{cap}Gi"
                / f"q{query:02}"
                / "result.json"
            )
            if result_path.exists() and (
                not args.retry_failed
                or json.loads(result_path.read_text())["status"] == "success"
            ):
                print(f"skip host={cap}Gi q{query}: already recorded", flush=True)
                continue
            print(
                f"{datetime.now(timezone.utc).isoformat()} start host={cap}Gi q{query}",
                flush=True,
            )
            result = run_query(args.sf, cap, query, output, args.timeout)
            write_reports(output, args.sf, caps, queries)
            print(
                f"{datetime.now(timezone.utc).isoformat()} done host={cap}Gi q{query} "
                f"{result['status']} runtime={result['runtime_s']} "
                f"GPU→HOST={result['gpu_to_host_bytes']} "
                f"HOST→DISK={result['host_to_disk_bytes']} "
                f"GPU→DISK={result['gpu_to_disk_bytes']} "
                f"{result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
