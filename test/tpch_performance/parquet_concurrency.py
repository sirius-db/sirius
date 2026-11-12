#!/usr/bin/env python3
# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

"""Concurrent TPC-H query streams over Parquet, with qgen order and predicates.

This is a read-only concurrency experiment, not an official TPC-H throughput
score: there is no refresh stream or RF1/RF2. Each case gets a fresh Sirius
process, and each stream executes its 22 queries sequentially on its own cursor.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import duckdb
import yaml

from performance_test import EXTENSION_PATH, TPCH_TABLES, _build_views_sql
from tpch_pin_columns import emit_pin_all, emit_unpin_all
from tpch_power_throughput import timed_query
from tpch_query_streams import load_stream
from tpch_stream_permutations import stream_order

REPO = Path(__file__).resolve().parents[2]
GENERATOR = Path(__file__).with_name("generate_tpch_queries.sh")
PLAN_DIR = REPO / "bench/sf1000-repro/plans"
CONCURRENCIES = (1, 2, 4, 8, 10)
DEFAULT_CONCURRENCIES = (1, 2, 4, 8)
SEED_COMMENT = re.compile(r"using (\d+) as a seed")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_script(con, sql):
    for statement in sql.split(";"):
        if statement.strip():
            con.execute(statement).fetchall()


def max_overlap(records):
    events = []
    for row in records:
        events.extend(((row["start_s"], 1), (row["end_s"], -1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def feature_overrides(args, base_params):
    overrides = {}
    if args.dynamic_filter != "inherit":
        overrides["enable_dynamic_filter"] = args.dynamic_filter == "on"
    if args.type_narrowing != "inherit":
        overrides["enable_compressed_materialization"] = args.type_narrowing == "on"
    if args.adaptive_join == "off":
        # Both budgets must be zero to prevent BUILD_PROBE admission.
        overrides["max_build_hash_table_bytes"] = 0
        overrides["max_build_probe_resident_bytes"] = 0
    elif args.adaptive_join == "on":
        for key in ("max_build_hash_table_bytes", "max_build_probe_resident_bytes"):
            if base_params.get(key) == 0:
                raise ValueError(
                    f"Cannot enable adaptive join: base config sets {key}: 0"
                )
    return overrides


def write_case_config(base, target, concurrency, overrides):
    original = base.read_text()
    data = yaml.safe_load(original)
    if not isinstance(data, dict) or not isinstance(data.get("sirius"), dict):
        raise ValueError(f"Invalid Sirius YAML: {base}")
    if "max_concurrent_queries" in data["sirius"]:
        raise ValueError(f"Remove max_concurrent_queries from base config: {base}")
    scan = data["sirius"].get("executor", {}).get("scan_manager", {})
    if "max_concurrent_queries" in scan:
        raise ValueError(
            f"Remove deprecated scan_manager.max_concurrent_queries: {base}"
        )
    if overrides:
        data["sirius"].setdefault("operator_params", {}).update(overrides)
        data["sirius"]["max_concurrent_queries"] = concurrency
        modified = yaml.safe_dump(data, sort_keys=False)
    else:
        modified, count = re.subn(
            r"(?m)^sirius:\s*$",
            f"sirius:\n  max_concurrent_queries: {concurrency}",
            original,
        )
        if count != 1:
            raise ValueError(f"Expected one top-level sirius key in {base}")
    target.write_text(modified)
    effective = yaml.safe_load(modified)
    if overrides:
        if effective != data:
            raise ValueError(
                f"Generated config differs from requested settings: {target}"
            )
    elif (
        effective["sirius"].pop("max_concurrent_queries") != concurrency
        or effective != data
    ):
        raise ValueError(f"Config change affected more than concurrency: {target}")


def prepare(args):
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    source = args.input.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    _build_views_sql(str(source))  # Verify all eight tables before generating anything.
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    base_config = yaml.safe_load(args.config.read_text())
    if not isinstance(base_config, dict) or not isinstance(
        base_config.get("sirius"), dict
    ):
        raise ValueError(f"Invalid Sirius YAML: {args.config}")
    base_params = base_config["sirius"].get("operator_params", {})
    if not isinstance(base_params, dict):
        raise ValueError(f"Invalid operator_params in {args.config}")
    overrides = feature_overrides(args, base_params)
    plan_files = [PLAN_DIR / f"{table}.txt" for table in ("lineitem", "orders")]
    for plan_file in plan_files:
        if not plan_file.is_file() or plan_file.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty Simpatico plan: {plan_file}")
    args.output.mkdir(parents=True, exist_ok=True)

    query_dir = args.query_dir.resolve() if args.query_dir else args.output / "queries"
    seed = args.seed
    if args.query_dir is None:
        seed = (
            seed
            if seed is not None
            else int(datetime.now(timezone.utc).strftime("%m%d%H%M%S"))
        )
        subprocess.run(
            [
                "bash",
                str(GENERATOR),
                "--output-dir",
                str(query_dir),
                "--seed",
                str(seed),
                str(args.sf),
                str(max(args.concurrencies)),
            ],
            check=True,
        )
    plans = {}
    for stream in range(1, max(args.concurrencies) + 1):
        query_file = query_dir / f"stream{stream}.sql"
        plan = load_stream(str(query_dir), stream, args.sf)
        order = [q for q, _ in plan]
        if order != list(stream_order(stream)):
            raise ValueError(
                f"Stream {stream} does not have the throughput permutation"
            )
        seed_match = SEED_COMMENT.search(query_file.read_text()[:200])
        if seed_match is None:
            raise ValueError(f"Missing qgen seed comment: {query_file}")
        stream_seed = int(seed_match.group(1))
        if seed is None:
            seed = stream_seed - stream
        if stream_seed != seed + stream:
            raise ValueError(
                f"Stream {stream} has seed {stream_seed}; expected {seed + stream}"
            )
        plans[str(stream)] = {
            "order": order,
            "seed": stream_seed,
            "query_file": str(query_file),
            "sha256": sha256(query_file),
        }
    manifest = {
        "scale_factor": args.sf,
        "input": str(source),
        "base_config": str(args.config.resolve()),
        "base_config_sha256": sha256(args.config),
        "feature_switches": {
            "dynamic_filter": args.dynamic_filter,
            "adaptive_join": args.adaptive_join,
            "type_narrowing": args.type_narrowing,
        },
        "operator_param_overrides": overrides,
        "concurrencies": args.concurrencies,
        "seed": seed,
        "query_dir": str(query_dir),
        "pin_tier": "host",
        "pin_strategy": "union of columns used by Q1-Q22, before timed streams",
        "pin_compression": "simpatico",
        "compression_plan_dir": str(PLAN_DIR),
        "compression_plan_sha256": {
            plan_file.name: sha256(plan_file) for plan_file in plan_files
        },
        "streams": plans,
    }
    for concurrency in args.concurrencies:
        case = args.output / f"c{concurrency}"
        case.mkdir()
        write_case_config(args.config, case / "config.yaml", concurrency, overrides)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_case(output, concurrency):
    manifest = json.loads((output / "manifest.json").read_text())
    case = output / f"c{concurrency}"
    os.environ["SIRIUS_CONFIG_FILE"] = str((case / "config.yaml").resolve())
    os.environ["SIRIUS_LOG_DIR"] = str((case / "sirius-log").resolve())
    Path(os.environ["SIRIUS_LOG_DIR"]).mkdir(exist_ok=True)
    os.environ["SIRIUS_PIN_TIER"] = "host"
    for table in TPCH_TABLES:
        os.environ.pop(f"SIRIUS_PIN_TIER_{table.upper()}", None)

    source = manifest["input"]
    con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
    records = []
    errors = []
    stream_seconds = {}
    pin_seconds = None
    ready = threading.Barrier(concurrency + 1)
    stopped = threading.Event()
    pin_complete = False
    try:
        extension = str(Path(EXTENSION_PATH).resolve()).replace("'", "''")
        con.execute(f"LOAD '{extension}'")
        execute_script(con, _build_views_sql(source))
        plan_dir = manifest["compression_plan_dir"]
        for filename, expected in manifest["compression_plan_sha256"].items():
            if sha256(Path(plan_dir) / filename) != expected:
                raise RuntimeError(f"Simpatico plan changed during run: {filename}")
        con.execute("SET pin_table_compression = true")
        escaped_plan_dir = plan_dir.replace("'", "''")
        con.execute(f"SET pin_table_input_compression_plan_dir = '{escaped_plan_dir}'")
        start = time.perf_counter()
        pin_table_seconds = {}
        for statement in emit_pin_all(source).split(";"):
            if not statement.strip():
                continue
            match = re.search(r"\bname='([^']+)'", statement)
            table = match.group(1) if match else "unknown"
            table_start = time.perf_counter()
            print(f"c{concurrency}: pinning {table} on host", flush=True)
            con.execute(statement).fetchall()
            pin_table_seconds[table] = time.perf_counter() - table_start
            print(
                f"c{concurrency}: pinned {table} in {pin_table_seconds[table]:.2f}s",
                flush=True,
            )
        pin_complete = True
        pin_seconds = time.perf_counter() - start
        print(f"c{concurrency}: host pin finished in {pin_seconds:.2f}s", flush=True)

        def worker(stream):
            cur = None
            try:
                cur = con.cursor()
                cur.execute("SET gpu_execution = true")
                plan = load_stream(
                    manifest["query_dir"], stream, manifest["scale_factor"]
                )
                ready.wait()
                stream_start = time.perf_counter()
                for q, statements in plan:
                    if stopped.is_set():
                        break
                    begin = time.perf_counter()
                    elapsed, _ = timed_query(cur, statements, None, label=f"q{q:02d}")
                    end = time.perf_counter()
                    records.append(
                        {
                            "stream": stream,
                            "query": q,
                            "start_s": begin,
                            "end_s": end,
                            "elapsed_s": elapsed,
                        }
                    )
                    print(
                        f"c{concurrency} stream {stream} Q{q}: {elapsed:.3f}s",
                        flush=True,
                    )
                stream_seconds[str(stream)] = time.perf_counter() - stream_start
            except threading.BrokenBarrierError:
                pass
            except BaseException as error:
                errors.append(f"stream {stream}: {error}")
                stopped.set()
                ready.abort()
            finally:
                if cur is not None:
                    cur.close()

        threads = [
            threading.Thread(target=worker, args=(stream,))
            for stream in range(1, concurrency + 1)
        ]
        for thread in threads:
            thread.start()
        try:
            ready.wait()
            wall_start = time.perf_counter()
        except threading.BrokenBarrierError:
            wall_start = time.perf_counter()
        for thread in threads:
            thread.join()
        wall_seconds = time.perf_counter() - wall_start
        if errors:
            raise RuntimeError("; ".join(errors))
        if len(records) != 22 * concurrency:
            raise RuntimeError(
                f"Expected {22 * concurrency} queries; recorded {len(records)}"
            )

        with (case / "queries.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=("stream", "query", "start_s", "end_s", "elapsed_s")
            )
            writer.writeheader()
            writer.writerows(
                sorted(records, key=lambda row: (row["stream"], row["start_s"]))
            )
        summary = {
            "concurrency": concurrency,
            "scale_factor": manifest["scale_factor"],
            "wall_seconds": wall_seconds,
            "query_executions": len(records),
            "queries_per_second": len(records) / wall_seconds,
            "max_in_flight_queries": max_overlap(records),
            "stream_wall_seconds": stream_seconds,
            "pin_seconds": pin_seconds,
            "pin_table_seconds": pin_table_seconds,
            "config_sha256": sha256(case / "config.yaml"),
        }
        (case / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"c{concurrency} complete: {summary['queries_per_second']:.3f} queries/s",
            flush=True,
        )
    finally:
        try:
            if pin_complete:
                execute_script(con, emit_unpin_all())
        finally:
            con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sf", type=int, required=True, help="TPC-H scale factor")
    parser.add_argument(
        "--input", type=Path, required=True, help="TPC-H Parquet directory"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Base Sirius YAML; unchanged"
    )
    parser.add_argument(
        "--dynamic-filter", choices=("inherit", "on", "off"), default="inherit",
        help="Override runtime hash-join filtering (default: inherit base config)",
    )
    parser.add_argument(
        "--adaptive-join", choices=("inherit", "on", "off"), default="inherit",
        help="Allow BUILD_PROBE when eligible, or force standard hash join",
    )
    parser.add_argument(
        "--type-narrowing", choices=("inherit", "on", "off"), default="inherit",
        help="Override compressed materialization of eligible numeric columns",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New result directory"
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        choices=CONCURRENCIES,
        default=list(DEFAULT_CONCURRENCIES),
    )
    parser.add_argument(
        "--seed", type=int, help="qgen base seed; streams use seed + stream ID"
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        help="Use existing qgen-generated streams instead of generating",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate and validate query sets/configs only",
    )
    parser.add_argument(
        "--case", type=int, choices=CONCURRENCIES, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.sf <= 0 or args.seed is not None and args.seed < 0:
        parser.error("--sf must be positive and --seed must be non-negative")
    if args.case is not None:
        run_case(args.output.resolve(), args.case)
        return
    if len(set(args.concurrencies)) != len(args.concurrencies):
        parser.error("--concurrencies must not contain duplicates")
    manifest = prepare(args)
    print(
        f"Prepared SF{args.sf} streams with seed {manifest['seed']} in {args.output}",
        flush=True,
    )
    if args.prepare_only:
        return
    for concurrency in args.concurrencies:
        print(f"Starting {concurrency} stream(s)", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--sf",
                str(args.sf),
                "--input",
                str(args.input),
                "--config",
                str(args.config),
                "--output",
                str(args.output),
                "--case",
                str(concurrency),
            ],
            check=True,
        )
    with (args.output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "concurrency",
                "wall_seconds",
                "query_executions",
                "queries_per_second",
                "max_in_flight_queries",
                "pin_seconds",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        for concurrency in args.concurrencies:
            summary = json.loads(
                (args.output / f"c{concurrency}" / "summary.json").read_text()
            )
            writer.writerow(summary)


if __name__ == "__main__":
    main()
