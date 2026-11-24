# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Adapt the pinned harness for fresh-process cold timings and durable answers.

Engine sources and YAML are untouched. No warmup is performed. The original
time_query measures SQL execute/materialization; startup and export are separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

PREFIX = "@@CORE_RC_EVENT "


def event(event_name, **data):
    print(
        PREFIX
        + json.dumps(
            {
                "event": event_name,
                "wall_ns": time.time_ns(),
                "utc": datetime.now(timezone.utc).isoformat(),
                **data,
            }
        ),
        flush=True,
    )


def expanded_query(sql, query):
    """Compute the independently checked LIMIT boundary, including all legal ties."""
    limit = {3: 10, 18: 100}[query]
    match = re.search(r"(?is)\border\s+by\b.+\blimit\s+(\d+)\s*;?\s*$", sql)
    if match is None or int(match.group(1)) != limit:
        raise ValueError(f"Unexpected Q{query} LIMIT/order shape")
    base = sql[: match.start()].strip().rstrip(";")
    order = (
        "revenue DESC, o_orderdate ASC"
        if query == 3
        else "o_totalprice DESC, o_orderdate ASC"
    )
    return (
        f"WITH full_result AS MATERIALIZED ({base}) SELECT full_result.*, "
        f"rank() OVER (ORDER BY {order}) AS __boundary_rank FROM full_result "
        f"QUALIFY __boundary_rank <= {limit} ORDER BY {order}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--sf", type=int, required=True)
    parser.add_argument(
        "--kind", choices=("reference", "cold", "hot", "pinned"), required=True
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cpu-temp", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from checks import verify_spill
    import yaml

    if args.kind == "reference":
        args.cpu_temp.mkdir(parents=True, exist_ok=True)
        storage = verify_spill(
            {
                "sirius": {
                    "memory": {
                        "disk": {
                            "capacity_bytes": "1",
                            "downgrade_root_dirs": str(args.cpu_temp),
                        }
                    }
                }
            }
        )
    else:
        startup = yaml.safe_load(args.config.read_text())
        disk = startup.get("sirius", {}).get("memory", {}).get("disk", {})
        storage = (
            verify_spill(startup)
            if disk.get("downgrade_root_dirs")
            else {
                "disk_spill_enabled": False,
                "reason": "guide preset leaves disk spilling disabled",
            }
        )
    (args.output / "storage-preflight.json").write_text(
        json.dumps(storage, indent=2) + "\n"
    )
    sys.path.insert(0, str(args.engine_root / "test/tpch_performance"))
    os.environ["SIRIUS_BUILD_PATH"] = str(
        args.engine_root / "sirius-duckdb/build/release"
    )
    m = importlib.import_module("performance_test")
    original_multi = m._execute_multi

    def observed_multi(con, sql):
        pin = "CALL pin_table(" in sql
        unpin = "CALL unpin_table(" in sql
        if pin or unpin:
            action = "pin" if pin else "unpin"
            if pin:
                assert args.kind == "pinned" and "tier='host'" in sql
            event(action + "_begin", sql=sql)
        value = original_multi(con, sql)
        if pin or unpin:
            event(action + "_end")
        return value

    m._execute_multi = observed_multi
    original_time = m.time_query
    original_open = m.open_connection
    original_record = m.RuntimeCsv.record
    original_write = m._write_result
    original_queries = m.queries_for_scale_factor
    original_csv_writer = m.csv.writer
    state = {"phase": "measurement", "iteration": {}, "last_record": {}}
    selected = m.parse_query_spec(args.queries)
    config_before = (
        hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config else None
    )

    class DurableWriter:
        def __init__(self, stream, *a, **kw):
            self.stream = stream
            self.writer = original_csv_writer(stream, *a, **kw)

        def writerow(self, row):
            self.writer.writerow(row)
            self.stream.flush()

    def record(writer, name, query, iteration, runtime):
        state["last_record"][query] = iteration
        original_record(writer, name, query, iteration, runtime)
        event("timing_saved", query=query, iteration=iteration + 1, seconds=runtime)

    def write_result(benchmark_dir, name, query, rows):
        iteration = state["last_record"][query] + 1
        directory = str(Path(benchmark_dir) / "iterations" / f"iteration{iteration}")
        original_write(directory, name, query, rows)
        event("answer_saved", query=query, iteration=iteration)

    def observed_time(con, query, sql, use_gpu, profile_path=None):
        phase = state["phase"]
        iteration = 0 if phase == "warmup" else state["iteration"].get(query, 0) + 1
        if phase != "warmup":
            state["iteration"][query] = iteration
        event("query_begin", phase=phase, query=query, iteration=iteration, gpu=use_gpu)
        try:
            elapsed, rows = original_time(
                con, query, sql, use_gpu, profile_path=profile_path
            )
        except BaseException as error:
            event(
                "query_error",
                phase=phase,
                query=query,
                iteration=iteration,
                error_type=type(error).__name__,
            )
            raise
        event(
            "query_end",
            phase=phase,
            query=query,
            iteration=iteration,
            seconds=elapsed,
            rows=len(rows),
            gpu=use_gpu,
        )
        if args.kind == "reference" and query in (3, 18):
            path = args.output / f"q{query}-expanded-ties.txt"
            with path.open("w") as stream:
                for row in rows:
                    stream.write(repr(row) + "\n")
            rows = [row[:-1] for row in rows[: {3: 10, 18: 100}[query]]]
        return elapsed, rows

    def queries(sf):
        result = original_queries(sf)
        if args.kind == "reference":
            for query in (3, 18):
                result[f"q{query}"] = expanded_query(result[f"q{query}"], query)
        return result

    def open_connection(
        source, gpu_execution=False, data_source="parquet", s3_input=None
    ):
        if not gpu_execution:
            con = original_open(
                source, gpu_execution=False, data_source=data_source, s3_input=s3_input
            )
            if args.cpu_temp:
                args.cpu_temp.mkdir(parents=True, exist_ok=True)
                free = os.statvfs(args.cpu_temp)
                budget = int(free.f_bavail * free.f_frsize * 0.85)
                con.execute(
                    f"SET temp_directory = {m._sql_literal(str(args.cpu_temp))}"
                )
                con.execute(f"SET max_temp_directory_size = '{budget}B'")
                event("reference_scratch", budget_bytes=budget)
            return con
        con = original_open(
            source, gpu_execution=True, data_source=data_source, s3_input=s3_input
        )
        event("engine_started", pid=os.getpid(), warmup=False)
        return con

    m.csv.writer = DurableWriter
    m.RuntimeCsv.record = record
    m._write_result = write_result
    m.time_query = observed_time
    m.queries_for_scale_factor = queries
    m.open_connection = open_connection
    sys.argv = [
        str(args.engine_root / "test/tpch_performance/performance_test.py"),
        "--input",
        args.input,
        "--scale-factor",
        str(args.sf),
        "--engine",
        "cpu" if args.kind == "reference" else "gpu",
        "--iterations",
        str(args.iterations),
        "--mode",
        "grouped" if args.kind in ("hot", "pinned") else "isolated",
        "--queries",
        args.queries,
        "--pin",
        "host" if args.kind == "pinned" else "none",
        "--output",
        str(args.output / "benchmark"),
    ]
    if args.config:
        sys.argv.extend(["--config", str(args.config)])
    event(
        "worker_start",
        pid=os.getpid(),
        kind=args.kind,
        scale_factor=args.sf,
        selected=selected,
        harness_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        argv=sys.argv,
        nofile=list(
            __import__("resource").getrlimit(__import__("resource").RLIMIT_NOFILE)
        ),
        memlock=list(
            __import__("resource").getrlimit(__import__("resource").RLIMIT_MEMLOCK)
        ),
    )
    try:
        m.main()
    finally:
        if args.config:
            assert (
                hashlib.sha256(args.config.read_bytes()).hexdigest() == config_before
            ), "YAML changed"
    event("worker_complete")


if __name__ == "__main__":
    main()
