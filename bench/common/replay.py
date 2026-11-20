# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Shared TPC-H replay planning, immutable configs, and complete/partial timing sums."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from checks import preflight

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "test/tpch_performance/performance_test.py"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def queries(value):
    result = []
    for part in value.split(","):
        bounds = [int(n) for n in part.split("-")]
        if len(bounds) > 2 or not 1 <= bounds[0] <= bounds[-1] <= 22:
            raise argparse.ArgumentTypeError("Queries must be in 1..22")
        for number in range(bounds[0], bounds[-1] + 1):
            if number in result:
                raise argparse.ArgumentTypeError("Duplicate queries are not allowed")
            result.append(number)
    return result


def schedule(hardware, profile, selected, iterations, sf):
    if hardware == "gb300" and sf == 10000 and profile == "pinned":
        raise ValueError("There is no successful saved GB300 SF10000 host-pin result")
    config = ROOT / "bench" / hardware / "configs" / f"{profile}.yaml"
    if profile == "cold":
        groups = [([q], [i]) for i in range(1, iterations + 1) for q in selected]
    elif hardware == "g7e":
        groups = [([q], list(range(1, iterations + 1))) for q in selected]
    else:
        groups = [(list(selected), list(range(1, iterations + 1)))]
    steps = [
        {"queries": qs, "iterations": its, "config": str(config)} for qs, its in groups
    ]
    if hardware == "gb300" and sf == 10000 and profile == "cold":
        for step in steps:
            if step["queries"] == [10]:
                step["config"] = str(config.with_name("sf10k-q10-cold.yaml"))
    elif hardware == "gb300" and sf == 10000:
        steps = []
        if others := [q for q in selected if q != 10]:
            steps.append(
                {
                    "queries": others,
                    "iterations": list(range(1, iterations + 1)),
                    "config": str(config),
                }
            )
        if 10 in selected:
            steps.append(
                {
                    "queries": [10],
                    "iterations": list(range(1, iterations + 1)),
                    "config": str(config.with_name(f"sf10k-q10-{profile}.yaml")),
                }
            )
    return steps


def child_env(hardware, profile):
    env = os.environ.copy()
    for key in (
        "SIRIUS_CONFIG_FILE",
        "SIRIUS_PRE_SQL",
        "SIRIUS_DISABLE",
        "SIRIUS_PIN_COMPRESSION_TABLES",
        "SIRIUS_PIN_TIER",
        "SIRIUS_EXP_LATE_MAT",
        "SIRIUS_EXP_FUSED_SCAN_FILTER",
        "SIRIUS_EXP_LOWCARD_GROUPBY",
    ):
        env.pop(key, None)
    env["SIRIUS_BUILD_PATH"] = str(ROOT / "sirius-duckdb/build/release")
    env["SIRIUS_PRE_SQL"] = "SET enable_duckdb_fallback = false;"
    env["SIRIUS_LOG_BACKEND"] = "spdlog"
    if env.get("SIRIUS_LOG_LEVEL") not in ("trace", "debug", "info"):
        env["SIRIUS_LOG_LEVEL"] = "info"
    if hardware == "gb300" and profile == "pinned":
        env["SIRIUS_PIN_COMPRESSION_TABLES"] = "lineitem,orders"
    return env


def command(args, step, directory, config):
    result = [
        sys.executable,
        str(BENCHMARK),
        "--input",
        args.input,
        "--scale-factor",
        str(args.sf),
        "--engine",
        "gpu",
        "--profile",
        "cold" if args.profile == "cold" else "hot",
        "--iterations",
        str(len(step["iterations"])),
        "--queries",
        ",".join(map(str, step["queries"])),
        "--config",
        str(config),
        "--output",
        str(directory / "benchmark"),
        "--name",
        directory.name,
    ]
    if args.profile == "pinned":
        result += ["--pin", args.pin_tier]
        if args.hardware == "gb300" or args.pin_compression:
            result += ["--pin-compression"]
    else:
        result += ["--pin", "none"]
    return result


def read_timings(directory, step):
    paths = list(directory.glob("benchmark/*/csv/runtimes.csv"))
    if len(paths) != 1:
        raise ValueError(f"Expected one runtime CSV under {directory}")
    measured = {}
    with paths[0].open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["query"] == "TOTAL":
                continue
            if row["engine"] != "sirius" or not row["query"].startswith("q"):
                raise ValueError(f"Unexpected measurement: {row}")
            q = int(row["query"][1:])
            iteration = int(row["iteration"])
            if q not in step["queries"] or not 0 <= iteration < len(step["iterations"]):
                raise ValueError(f"Unexpected query/iteration: {row}")
            key = (q, step["iterations"][iteration])
            if key in measured:
                raise ValueError(f"Duplicate measurement: {key}")
            elapsed = float(row["runtime_s"])
            measured[key] = elapsed if math.isfinite(elapsed) and elapsed >= 0 else None
    for q in step["queries"]:
        logs = list(directory.glob(f"benchmark/*/sirius/q{q}/sirius.log"))
        count = sum(
            p.read_text(errors="replace").count(
                "Transparent execution: physical plan replaced with GPU operator"
            )
            for p in logs
        )
        successes = sum(
            value is not None for (query, _), value in measured.items() if query == q
        )
        if count < successes:
            raise ValueError(
                f"Q{q}: {successes} timings but only {count} GPU plan confirmations"
            )
    return measured


def summarize(selected, iterations, timings):
    rows = []
    for q in selected:
        values = [timings.get((q, i)) for i in range(1, iterations + 1)]
        good = [value for value in values if value is not None]
        rows.append(
            {"query": q, "iterations_s": values, "best_s": min(good) if good else None}
        )
    totals = []
    for i in range(iterations):
        values = [
            row["iterations_s"][i] for row in rows if row["iterations_s"][i] is not None
        ]
        totals.append(
            {
                "iteration": i + 1,
                "sum_s": math.fsum(values) if values else None,
                "queries": len(values),
                "expected_queries": len(selected),
            }
        )
    best = [row["best_s"] for row in rows if row["best_s"] is not None]
    return {
        "per_query": rows,
        "iteration_totals": totals,
        "sum_best_s": math.fsum(best) if best else None,
        "queries_with_timings": len(best),
        "expected_queries": len(selected),
        "successful_measurements": sum(
            v is not None for row in rows for v in row["iterations_s"]
        ),
        "expected_measurements": len(selected) * iterations,
    }


def write_report(output, args, timings, errors):
    report = summarize(args.queries, args.iterations, timings)
    report["errors"] = errors
    report["complete"] = (
        not errors
        and report["successful_measurements"] == report["expected_measurements"]
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "per-query.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "query",
                *[f"iteration{i}_s" for i in range(1, args.iterations + 1)],
                "best_s",
            ]
        )
        for row in report["per_query"]:
            writer.writerow([row["query"], *row["iterations_s"], row["best_s"]])
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", choices=("gb300", "g7e"), required=True)
    parser.add_argument(
        "--sf", type=int, choices=(100, 300, 1000, 3000, 10000), required=True
    )
    parser.add_argument(
        "--profile", choices=("cold", "hot", "pin", "pinned"), required=True
    )
    parser.add_argument(
        "--input", help="Local Parquet directory or same-region s3://bucket/prefix"
    )
    parser.add_argument(
        "--output", type=Path, help="New run directory; refuses existing directories"
    )
    parser.add_argument("--queries", type=queries, default=list(range(1, 23)))
    parser.add_argument("--iterations", type=int)
    parser.add_argument(
        "--worker-timeout",
        type=int,
        default=3600,
        help="Seconds per fresh process, including setup; default 3600",
    )
    parser.add_argument("--pin-tier", choices=("host", "gpu"), default="host")
    parser.add_argument(
        "--pin-compression",
        action="store_true",
        help="Opt in for G7e pinned runs; always used for GB300 pinned replay",
    )
    parser.add_argument(
        "--allow-ebs-spill",
        action="store_true",
        help="Explicit exception to G7e's instance-store spill requirement",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan only; no GPU, AWS, filesystem writes, or dataset access",
    )
    args = parser.parse_args(argv)
    args.profile = "pinned" if args.profile == "pin" else args.profile
    args.iterations = (
        args.iterations
        if args.iterations is not None
        else (1 if args.profile == "pinned" else 2)
    )
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.worker_timeout < 1:
        parser.error("--worker-timeout must be positive")
    if args.hardware == "gb300" and args.pin_tier != "host":
        parser.error("GB300 saved replay uses host pinning")
    args.input = args.input or str(
        Path(os.environ.get("DATASET_ROOT", "/mnt/datasets"))
        / f"tpch_parquet_sf{args.sf}"
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    default_root = os.environ.get(
        f"{args.hardware.upper()}_REPLAY_OUTPUT",
        str(ROOT / f"test_datasets/{args.hardware}-replay-output"),
    )
    args.output = (
        args.output or Path(default_root) / f"sf{args.sf}-{args.profile}-{stamp}"
    ).resolve()
    steps = schedule(
        args.hardware, args.profile, args.queries, args.iterations, args.sf
    )
    for i, step in enumerate(steps):
        step["directory"] = str(args.output / f"worker{i + 1:03}")
        step["source_sha256"] = digest(step["config"])
        step["command"] = command(
            args,
            step,
            Path(step["directory"]),
            Path(step["directory"]) / "effective-config.yaml",
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "hardware": args.hardware,
                    "profile": args.profile,
                    "steps": steps,
                    "environment_overrides": {
                        k: v
                        for k, v in child_env(args.hardware, args.profile).items()
                        if k
                        in (
                            "SIRIUS_PRE_SQL",
                            "SIRIUS_PIN_COMPRESSION_TABLES",
                            "SIRIUS_BUILD_PATH",
                            "SIRIUS_LOG_BACKEND",
                            "SIRIUS_LOG_LEVEL",
                        )
                    },
                },
                indent=2,
            )
        )
        return 0

    import yaml

    if args.output.exists():
        parser.error("Output directory already exists; choose a new path")
    extension = (
        ROOT / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
    )
    if not extension.is_file():
        parser.error("Build Sirius from this checkout first: pixi run make")
    evidence = preflight(
        args.hardware,
        args.input,
        yaml.safe_load(Path(steps[0]["config"]).read_text()),
        args.allow_ebs_spill,
    )
    args.output.mkdir(parents=True)
    evidence.update(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        submodules=subprocess.check_output(
            ["git", "submodule", "status", "--recursive"], cwd=ROOT, text=True
        ),
        benchmark_sha256=digest(BENCHMARK),
        steps=steps,
        hardware=args.hardware,
        scale_factor=args.sf,
        profile=args.profile,
        input=args.input,
        cpu_fallback=False,
    )
    evidence["extension_sha256"] = digest(extension)
    evidence["source_status"] = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    evidence["worker_timeout_s"] = args.worker_timeout
    evidence["libraries_sha256"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (ROOT / "build/release/install").glob("lib*/libsirius.so.*")
        if p.is_file() and not p.is_symlink()
    }
    evidence["logging"] = {
        key: child_env(args.hardware, args.profile)[key]
        for key in ("SIRIUS_LOG_BACKEND", "SIRIUS_LOG_LEVEL")
    }
    (args.output / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
    timings, errors = {}, []
    try:
        for step in steps:
            source = Path(step["config"])
            if (
                digest(source) != step["source_sha256"]
                or digest(BENCHMARK) != evidence["benchmark_sha256"]
                or digest(extension) != evidence["extension_sha256"]
            ):
                raise ValueError("Source config or runner changed during the campaign")
            directory = Path(step["directory"])
            directory.mkdir()
            (directory / "source-config.yaml").write_bytes(source.read_bytes())
            config = yaml.safe_load(source.read_text())
            effective = copy.deepcopy(config)
            effective["sirius"].setdefault("telemetry", {}).update(
                enable_quent=True, output_directory=str(directory / "quent")
            )
            effective_path = directory / "effective-config.yaml"
            effective_path.write_text(yaml.safe_dump(effective, sort_keys=False))
            before = digest(effective_path)
            with (directory / "run.log").open("w") as log:
                print(shlex.join(step["command"]), flush=True)
                result = subprocess.run(
                    step["command"],
                    cwd=ROOT,
                    env=child_env(args.hardware, args.profile),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.worker_timeout,
                )
            if (
                digest(source) != step["source_sha256"]
                or digest(effective_path) != before
            ):
                raise ValueError("YAML changed during a measurement")
            timings.update(read_timings(directory, step))
            if result.returncode:
                raise ValueError(
                    f"Benchmark exited {result.returncode}; see {directory}/run.log"
                )
            expected = {(q, i) for q in step["queries"] for i in step["iterations"]}
            if any(timings.get(key) is None for key in expected):
                raise ValueError(
                    f"Incomplete or failed measurement; see {directory}/run.log"
                )
    except (Exception, KeyboardInterrupt) as error:
        errors.append(str(error) or "Interrupted")
    report = write_report(args.output, args, timings, errors)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
