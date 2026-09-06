#!/usr/bin/env python3
"""Run standalone Sirius against SQL and CPU references prepared by tpch-starrocks.py.

Uses the canonical performance runner's connection and timing functions. A parent
process enforces a deadline even if native GPU execution cannot be interrupted.
"""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
COMMON = runpy.run_path(str(Path(__file__).with_name("tpch-starrocks.py")))
write_json = COMMON["write_json"]


def progress(run, phase, query=None):
    write_json(
        run / "progress.json",
        {"phase": phase, "query": query, "started_monotonic": time.monotonic()},
    )


def log_offsets(directory):
    return {path: path.stat().st_size for path in directory.glob("*.log")}


def log_since(directory, offsets):
    chunks = []
    for path in sorted(directory.glob("*.log")):
        with path.open("rb") as source:
            source.seek(offsets.get(path, 0))
            chunks.append(source.read().decode(errors="replace"))
    return "".join(chunks)


def worker(args, manifest):
    sys.path.insert(0, str(ROOT / "test/tpch_performance"))
    import performance_test as canonical
    import duckdb

    canonical.EXTENSION_PATH = str(
        ROOT / "build/release/extension/sirius/sirius.duckdb_extension"
    )
    run = args.run_dir
    log_dir = run / "engine-log"
    log_dir.mkdir(exist_ok=True)
    engine_dir = run / "engine"
    (engine_dir / "spill").mkdir(parents=True, exist_ok=True)
    os.chdir(engine_dir)
    os.environ.pop("SIRIUS_DISABLE", None)
    os.environ.pop("SIRIUS_PRE_SQL", None)
    os.environ.update(
        SIRIUS_CONFIG_FILE=str(args.config),
        SIRIUS_LOG_BACKEND="spdlog",
        SIRIUS_LOG_LEVEL="info",
        SIRIUS_LOG_DIR=str(log_dir),
    )
    progress(run, "initializing")
    con = canonical.open_connection(manifest["data_dir"], gpu_execution=True)
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET enable_duckdb_fallback=false")
    con.execute("SET gpu_execution=true")
    try:
        for name in args.queries:
            COMMON["verify_inputs"](manifest["inputs"])
            source = args.reference_dir / name
            target = run / name
            target.mkdir(exist_ok=True)
            sql = (source / "duckdb.sql").read_text()
            (target / "query.sql").write_text(sql)
            fingerprint = hashlib.sha256(
                (sql + json.dumps(manifest["inputs"], sort_keys=True)).encode()
            ).hexdigest()
            if (source / "oracle-fingerprint.txt").read_text() != fingerprint:
                raise RuntimeError(f"Stale reference for {name}")
            oracle = json.loads((source / "oracle.json").read_text())
            if oracle["status"] != "OK":
                raise RuntimeError(f"Invalid CPU reference for {name}")
            canonical.QUERIES[f"q{int(name[1:])}"] = sql
            # Reinstalling the sink flushes the old sink, outside timed execution.
            con.execute("SET sirius_log_backend='spdlog'")
            offsets = log_offsets(log_dir)
            result = {
                "query": name,
                "status": "ERROR",
                "oracle_rows": len(oracle["rows"]),
            }
            progress(run, "query", name)
            started = time.monotonic()
            try:
                elapsed, rows = canonical.time_query(con, int(name[1:]), True)
                result["elapsed_seconds"] = elapsed
                columns = [column[0] for column in con.description]
                types = [str(column[1]) for column in con.description]
                progress(run, "comparison", name)
                raw_rows = [
                    ["NULL" if cell is None else str(cell) for cell in row]
                    for row in rows
                ]
                write_json(
                    target / "result.json",
                    {"columns": columns, "types": types, "rows": raw_rows},
                )
                comparison = COMMON["compare_rows"](
                    oracle, columns, raw_rows, Decimal("1e-6"), Decimal("1e-8")
                )
                # The shared comparator names the original engine in diagnostics.
                comparison["differences"] = [
                    item.replace("StarRocks=", "Sirius=")
                    for item in comparison["differences"]
                ]
                write_json(target / "comparison.json", comparison)
                result["comparison"] = comparison
                result["status"] = "PASS" if comparison["match"] else "MISMATCH"
            except Exception as error:
                result["elapsed_seconds"] = time.monotonic() - started
                result["error"] = str(error)
            con.execute("SET sirius_log_backend='spdlog'")
            engine_log = log_since(log_dir, offsets)
            (target / "sirius.log").write_text(engine_log)
            evidence = {
                "started": engine_log.count(
                    "Transparent GPU execution: executing query"
                ),
                "completed": engine_log.count(
                    "Transparent GPU execution: query completed"
                ),
                "fallback_enabled": False,
                "fallback_or_error_present": any(
                    marker in engine_log
                    for marker in (
                        "Transparent execution fallback",
                        "falling back to DuckDB CPU",
                        "Transparent GPU execution error",
                    )
                ),
            }
            evidence["verified"] = (
                evidence["started"] == 1
                and evidence["completed"] == 1
                and not evidence["fallback_or_error_present"]
            )
            result["gpu_execution"] = evidence
            if result["status"] == "PASS" and not evidence["verified"]:
                result["status"] = "GPU_UNVERIFIED"
            result["duckdb_version"] = duckdb.__version__
            write_json(target / "status.json", result)
            print(
                f"{name.upper()} {result['status']} {result['elapsed_seconds']:.3f}s",
                flush=True,
            )
    finally:
        con.close()
    progress(run, "finished")


def report(args, manifest, exit_code):
    results = []
    for name in args.queries:
        status = args.run_dir / name / "status.json"
        results.append(
            json.loads(status.read_text())
            if status.exists()
            else {"query": name, "status": "NOT_COMPLETED"}
        )
    passed = sum(item["status"] == "PASS" for item in results)
    elapsed = sum(item.get("elapsed_seconds", 0) for item in results)
    write_json(
        args.run_dir / "results.json",
        {
            "manifest": manifest,
            "worker_exit_code": exit_code,
            "passed": passed,
            "query_seconds_sum": elapsed,
            "results": results,
        },
    )
    with (args.run_dir / "timings.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query", "status", "seconds", "rows", "gpu_verified"])
        for item in results:
            writer.writerow(
                [
                    item["query"],
                    item["status"],
                    item.get("elapsed_seconds"),
                    item.get("comparison", {}).get("actual_rows"),
                    item.get("gpu_execution", {}).get("verified", False),
                ]
            )
    lines = [
        f"# TPC-H SF{manifest['scale_factor']} through standalone Sirius",
        "",
        f"**{passed}/{len(results)} queries passed the DuckDB comparison and GPU execution check.**",
        f"Sum of measured query times: **{elapsed:.3f} seconds**.",
        "",
        "One standalone Sirius process on one NVIDIA GB10; no StarRocks FE or CNs. Sirius GPU budget 48 GiB, host budget 16 GiB, disk spill enabled (1 TiB limit).",
        "Normal SQL executes through the Sirius DuckDB extension with `gpu_execution=true` and `enable_duckdb_fallback=false`. Each successful query must also have exactly one GPU start and completion in its engine log.",
        "Uses the canonical performance runner's connection and timing functions, with exactly the prepared StarRocks run's DuckDB SQL and input fingerprints. Q11 uses the SF-dependent threshold; Q8/Q9 and Q22 use the same equivalent SQL adaptations as that run.",
        f"Reference run: `{args.reference_dir}`. Source commit: `{manifest['engine_source']['git_head']}`.",
        "One measured execution per query in one connection, no pinning and no cold-cache reset. Times cover execute plus fetch, excluding extension startup and validation. StarRocks timings include MySQL client startup and transfer, so timing boundaries differ. This is not a formal TPC-H benchmark.",
        "Duplicate-preserving multiset comparison, relative tolerance 1e-6 / absolute 1e-8; integers, strings and NULLs exact. Output ordering is not verified.",
        "",
        "| Query | Status | Seconds | Rows | GPU verified |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['query'].upper()} | {item['status']} | {item.get('elapsed_seconds', 0):.3f} | {item.get('comparison', {}).get('actual_rows', '-')} | {item.get('gpu_execution', {}).get('verified', False)} |"
        )
    (args.run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return passed == len(results) and exit_code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("sirius-standalone.yaml"),
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--queries", default=",".join(f"q{i:02d}" for i in range(1, 23))
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.queries = [
        f"q{int(name.lower().lstrip('q')):02d}" for name in args.queries.split(",")
    ]
    if (
        args.timeout <= 0
        or len(args.queries) != len(set(args.queries))
        or any(name not in {f"q{i:02d}" for i in range(1, 23)} for name in args.queries)
    ):
        parser.error("Use a positive timeout and unique query numbers from 1 to 22")
    for key in ("reference_dir", "run_dir", "config"):
        setattr(args, key, getattr(args, key).resolve())
    manifest = json.loads((args.reference_dir / "manifest.json").read_text())
    if args.worker:
        worker(args, manifest)
        return 0
    if args.run_dir.exists():
        parser.error("Output directory already exists; choose a fresh --run-dir")
    COMMON["verify_inputs"](manifest["inputs"])
    args.run_dir.mkdir(parents=True)
    manifest = {
        key: manifest[key]
        for key in (
            "data_dir",
            "scale_factor",
            "inputs",
            "query_source_sha256",
            "sql_adaptations",
            "oracle_memory_limit",
            "oracle_threads",
        )
    }
    manifest.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "standalone",
            "reference_dir": str(args.reference_dir),
            "reference_manifest_sha256": hashlib.sha256(
                (args.reference_dir / "manifest.json").read_bytes()
            ).hexdigest(),
            "query_names": args.queries,
            "topology": {
                "standalone_processes": 1,
                "physical_gpus": 1,
                "compute_nodes": 0,
            },
            "session_sql": "SET memory_limit='8GB'; SET threads=4; SET enable_duckdb_fallback=false; SET gpu_execution=true;",
            "relative_tolerance": "1e-6",
            "absolute_tolerance": "1e-8",
            "config": str(args.config),
            "timeout_seconds": args.timeout,
            "engine_source": COMMON["source_identity"](ROOT),
            "engine_binary": COMMON["file_identity"](
                ROOT / "build/release/extension/sirius/sirius.duckdb_extension"
            ),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    shutil.copyfile(args.config, args.run_dir / "config.yaml")
    shutil.copyfile(
        args.reference_dir / "manifest.json", args.run_dir / "reference-manifest.json"
    )
    write_json(args.run_dir / "manifest.json", manifest)
    progress(args.run_dir, "starting")
    with (args.run_dir / "console.log").open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--worker",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        previous = None
        while process.poll() is None:
            current = json.loads((args.run_dir / "progress.json").read_text())
            label = (current["phase"], current["query"])
            if label != previous:
                print(f"{label[0]} {label[1] or ''}", flush=True)
                previous = label
            if time.monotonic() - current["started_monotonic"] > args.timeout:
                process.kill()
                process.wait()
                write_json(args.run_dir / "timeout.json", current)
                break
            time.sleep(0.5)
    return 0 if report(args, manifest, process.returncode) else 1


if __name__ == "__main__":
    sys.exit(main())
