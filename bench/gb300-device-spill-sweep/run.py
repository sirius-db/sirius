#!/usr/bin/env python3
"""Sweep absolute GPU memory limits and count SF10k tier movements."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench/gb300-spill-sweep"))
from run import (
    confirm_gpu_plan,
    find_runtime_csv,
    measure_moves,
    parse_numbers,
)  # noqa: E402


BASE_CONFIG = ROOT / "bench/gb300-cold/config.yaml"
BENCHMARK = ROOT / "test/tpch_performance/performance_test.py"
MOVE_FIELDS = ("gpu_to_host_bytes", "host_to_disk_bytes", "gpu_to_disk_bytes")
FIELDS = (
    "scale_factor",
    "gpu_cap_gb",
    "host_cap_gib",
    "query",
    "status",
    "runtime_s",
    *MOVE_FIELDS,
    "error",
)


def prefer_query_for_oom_kill():
    # If a low GPU cap drives host RAM to exhaustion, preserve the sweep runner
    # so it can record the failed query and continue to the next one.
    fd = os.open("/proc/self/oom_score_adj", os.O_WRONLY)
    try:
        os.write(fd, b"500")
    finally:
        os.close(fd)


def make_config(cap_gb, host_cap_gib, trace_dir):
    source = BASE_CONFIG.read_text()
    source, count = re.subn(
        r"(?m)^(\s*)usage_limit_fraction: 0\.95$",
        rf"\g<1>usage_limit_bytes: {cap_gb * 1_000_000_000}",
        source,
    )
    if count != 1:
        raise RuntimeError("GB300 cold GPU config changed; review the substitution")
    source, host_count = re.subn(
        r"(?m)^(\s*capacity_bytes:) 450Gi$",
        rf"\g<1> {host_cap_gib}Gi",
        source,
    )
    source, pools_count = re.subn(
        r"(?m)^(\s*initial_number_pools:) 225$",
        rf"\g<1> {max(1, host_cap_gib // 2)}",
        source,
    )
    if (host_count, pools_count) != (1, 1):
        raise RuntimeError("GB300 cold host config changed; review the substitutions")
    return source.rstrip() + (
        "\n  telemetry:\n"
        "    enable_quent: true\n"
        f"    output_directory: {trace_dir}\n"
    )


def write_reports(output, sf, caps, host_cap_gib, queries):
    report_dir = output / f"host{host_cap_gib}Gi" / f"sf{sf}"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cap in caps:
        for query in queries:
            path = report_dir / f"gpu{cap}GB" / f"q{query:02}" / "result.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
    path = report_dir / "per-query.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    totals = []
    for cap in caps:
        cap_rows = [row for row in rows if row["gpu_cap_gb"] == cap]
        successful = [row for row in cap_rows if row["status"] == "success"]
        totals.append(
            {
                "scale_factor": sf,
                "gpu_cap_gb": cap,
                "host_cap_gib": host_cap_gib,
                "completed_queries": len(cap_rows),
                "successful_queries": len(successful),
                "requested_queries": len(queries),
                **{
                    field: sum(row[field] for row in successful)
                    for field in MOVE_FIELDS
                },
            }
        )
    path = report_dir / "totals.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=totals[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(totals)
    temporary.replace(path)


def run_query(sf, cap, host_cap_gib, query, output, timeout):
    query_dir = (
        output / f"host{host_cap_gib}Gi" / f"sf{sf}" / f"gpu{cap}GB" / f"q{query:02}"
    )
    if query_dir.exists() and any(query_dir.iterdir()):
        query_dir.rename(
            query_dir.with_name(f"{query_dir.name}.previous-{time.time_ns()}")
        )
    query_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = query_dir / "quent"
    trace_dir.mkdir()
    config = query_dir / "config.yaml"
    config.write_text(make_config(cap, host_cap_gib, trace_dir))
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
        f"gpu{cap}GB_q{query}",
    ]
    (query_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = {
        "scale_factor": sf,
        "gpu_cap_gb": cap,
        "host_cap_gib": host_cap_gib,
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
                preexec_fn=prefer_query_for_oom_kill,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=int, default=10000)
    parser.add_argument("--caps", default="16,32,64,128,256")
    parser.add_argument("--host-cap-gib", type=int, default=384)
    parser.add_argument("--queries", default=",".join(map(str, range(1, 23))))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "test_datasets/gb300-device-spill-sweep"
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    caps = parse_numbers(args.caps)
    if args.host_cap_gib <= 0:
        parser.error("--host-cap-gib must be positive")
    queries = parse_numbers(args.queries, maximum=22)
    output = args.output.resolve()
    dataset = ROOT / f"test_datasets/tpch_parquet_sf{args.sf}"
    if not dataset.is_dir():
        parser.error(f"dataset missing: {dataset}")
    if not (
        ROOT / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
    ).is_file():
        parser.error("built Sirius extension is missing")
    print(
        f"{datetime.now(timezone.utc).isoformat()} SF{args.sf}: GPU caps={caps} decimal GB, "
        f"host cap={args.host_cap_gib} GiB, queries={queries}, output={output}",
        flush=True,
    )
    if args.dry_run:
        for cap in caps:
            print(
                f"GPU usage_limit_bytes={cap * 1_000_000_000} ({cap} GB); "
                f"reservation=0.9, downgrade=0.8/0.6, host={args.host_cap_gib} GiB",
                flush=True,
            )
        return
    for cap in caps:
        for query in queries:
            result_path = (
                output
                / f"host{args.host_cap_gib}Gi"
                / f"sf{args.sf}"
                / f"gpu{cap}GB"
                / f"q{query:02}"
                / "result.json"
            )
            if result_path.exists() and (
                not args.retry_failed
                or json.loads(result_path.read_text())["status"] == "success"
            ):
                print(f"skip gpu={cap}GB q{query}: already recorded", flush=True)
                continue
            print(
                f"{datetime.now(timezone.utc).isoformat()} start gpu={cap}GB q{query}",
                flush=True,
            )
            result = run_query(
                args.sf, cap, args.host_cap_gib, query, output, args.timeout
            )
            write_reports(output, args.sf, caps, args.host_cap_gib, queries)
            print(
                f"{datetime.now(timezone.utc).isoformat()} done gpu={cap}GB q{query} "
                f"{result['status']} runtime={result['runtime_s']} "
                f"GPU→HOST={result['gpu_to_host_bytes']} "
                f"HOST→DISK={result['host_to_disk_bytes']} "
                f"GPU→DISK={result['gpu_to_disk_bytes']} "
                f"{result['error']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
