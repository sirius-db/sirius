#!/usr/bin/env python3
"""Profile actual Sirius UDF query operators with bounded input and pool targets."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import run_bench


MIB = 1024 * 1024
STAGES = (
    "input_prepare_scan",
    "projection_expressions",
    "aggregation",
    "output_transfer_materialization",
    "other",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=run_bench.WORKLOADS,
        default=list(run_bench.WORKLOADS[:3]),
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[100_000, 1_000_000])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--q1-data", type=Path)
    parser.add_argument("--batch-mib", type=int, default=8)
    parser.add_argument("--gpu-budget-mib", type=int, default=512)
    parser.add_argument("--host-budget-mib", type=int, default=1024)
    options = parser.parse_args(argv)
    if options.batch_mib <= 0 or options.gpu_budget_mib <= 0:
        parser.error("--batch-mib and --gpu-budget-mib must be positive")
    if options.host_budget_mib < 2:
        parser.error("--host-budget-mib must be at least 2 for pinned pool blocks")

    # Share workload/build/path validation and exact SQL generation with the
    # end-to-end runner, while keeping the profiling experiment's own defaults.
    shared = [
        "--build-dir",
        str(options.build_dir),
        "--output-dir",
        str(options.output_dir),
        "--workloads",
        *options.workloads,
        "--sizes",
        *(str(size) for size in options.sizes),
        "--warmups",
        str(options.warmups),
        "--repetitions",
        str(options.repetitions),
        "--threads",
        str(options.threads),
        "--projection",
    ]
    if options.q1_data is not None:
        shared.extend(["--q1-data", str(options.q1_data)])
    args = run_bench.parse_args(shared)
    for name in ("batch_mib", "gpu_budget_mib", "host_budget_mib"):
        setattr(args, name, getattr(options, name))
    if args.threads > 1:
        print(
            "Warning: CPU operator timings sum wall durations across parallel "
            "workers; operator-work ratios are not query latency speedups.",
            file=sys.stderr,
        )
    return args


def device_info(build_dir):
    process = subprocess.run(
        [str(build_dir / "bench_host"), "--device-info"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    info = json.loads(process.stdout)
    if not isinstance(info, dict):
        raise ValueError("device probe did not return an object")
    for name in ("device_count", "total_bytes", "free_bytes"):
        if type(info.get(name)) is not int or info[name] <= 0:
            raise ValueError(f"device probe returned invalid {name}")
    if info["free_bytes"] > info["total_bytes"]:
        raise ValueError("device probe free memory exceeds total memory")
    return info


def configuration(args, device):
    """Derive a tracked-pool target from the selected visible CUDA device."""
    budget = args.gpu_budget_mib * MIB
    if budget > device["free_bytes"]:
        raise ValueError(
            "GPU pool budget exceeds currently free device memory; reduce it"
        )
    if not 0 < budget < device["total_bytes"]:
        raise ValueError("GPU pool budget must be smaller than total device memory")
    batch = args.batch_mib * MIB
    return (
        "sirius:\n"
        "  topology: { num_gpus: 1 }\n"
        "  memory:\n"
        f"    gpu: {{ usage_limit_bytes: {budget}, reservation_limit_fraction: 1.0 }}\n"
        f"    host: {{ capacity_bytes: {args.host_budget_mib * MIB}, initial_number_pools: 1, "
        f"pool_size: {min(128, args.host_budget_mib // 2)}, block_size: {MIB} }}\n"
        "  executor:\n"
        "    pipeline: { num_threads: 1 }\n"
        "    task_creator: { num_threads: 1 }\n"
        "    downgrade: { num_threads: 1, monitor_period: 10ms }\n"
        "  operator_params:\n"
        f"    scan_task_batch_size: {batch}\n"
        f"    concat_batch_bytes: {batch}\n"
        f"    hash_partition_bytes: {batch}\n"
        "    max_sort_partition_bytes: 0\n"
        f"    max_build_hash_table_bytes: {max(8 * MIB, batch * 2)}\n"
    )


def validate_manifest(manifest, args, workload, rows):
    """Reject missing correctness gates, variants, or per-query GPU proof."""
    if (
        not isinstance(manifest, dict)
        or manifest.get("correctness_verified") is not True
    ):
        raise ValueError("host did not confirm correctness comparisons")
    if manifest.get("input_rows", manifest.get("rows")) != rows:
        raise ValueError("host supplied the wrong input row count")
    validation = manifest.get("validation_records")
    expected_samples = {"prefix", "dispersed"}
    if workload == "udfbench_q1":
        expected_samples.add("q1_original_python_oracle")
    if not isinstance(validation, list) or any(
        not isinstance(v, dict) for v in validation
    ):
        raise ValueError("host did not supply per-query correctness evidence")
    if (
        len(validation) != len(expected_samples)
        or {v.get("sample") for v in validation} != expected_samples
    ):
        raise ValueError("host supplied missing or duplicate correctness gates")
    for proof in validation:
        if (
            proof.get("cache_mode") != "warm_storage"
            or proof.get("gpu_executions") != 1
            or proof.get("plan_fallbacks") != 0
            or proof.get("runtime_fallbacks") != 0
            or type(proof.get("rows_compared")) is not int
            or proof["rows_compared"] <= 0
        ):
            raise ValueError("correctness gate lacks value comparisons or GPU evidence")
        if (
            proof["sample"] == "q1_original_python_oracle"
            and proof["rows_compared"] != 50_000
        ):
            raise ValueError("Q1 correctness gate must compare all public oracle rows")
    if manifest.get("gpu_validation_queries") != len(validation):
        raise ValueError("GPU validation total disagrees with correctness gates")

    modes = run_bench.definitions(workload, rows, args)[1]
    labels = dict(modes)
    labels["gpu"] = "GPU"
    expected = {
        ("gpu" if mode == "gpu" else "cpu", mode, kind, iteration)
        for mode in labels
        for kind in ("aggregate", "projection")
        for iteration in range(args.warmups + args.repetitions)
    }
    profiles = manifest.get("profiles")
    if not isinstance(profiles, list) or any(not isinstance(p, dict) for p in profiles):
        raise ValueError("host did not supply per-query profiles")
    actual = set()
    sample_ids = set()
    for profile in profiles:
        if type(profile.get("iteration")) is not int:
            raise ValueError("profile has an invalid iteration")
        key = tuple(
            profile.get(k) for k in ("engine", "cpu_mode", "query_kind", "iteration")
        )
        if key not in expected or key in actual:
            raise ValueError("host supplied duplicate or unexpected profile variants")
        actual.add(key)
        sample_id = profile.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise ValueError("host supplied duplicate or invalid sample IDs")
        sample_ids.add(sample_id)
        if (
            type(profile.get("warmup")) is not bool
            or profile["warmup"] != (profile["iteration"] < args.warmups)
            or profile.get("cpu_baseline") != labels[profile["cpu_mode"]]
        ):
            raise ValueError("profile has incorrect warmup or CPU baseline labels")
        wall = profile.get("query_wall_ms")
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("profile has invalid query wall time")
        if not isinstance(profile.get("sql"), str) or not profile["sql"].strip():
            raise ValueError("profile is missing its exact query")
        if profile["engine"] == "gpu":
            counters = profile.get("gpu_counters")
            expected_counters = {
                "executions": 1,
                "plan_fallbacks": 0,
                "runtime_fallbacks": 0,
            }
            if not isinstance(counters, dict) or any(
                type(counters.get(k)) is not int or counters[k] != value
                for k, value in expected_counters.items()
            ):
                raise ValueError(
                    "profile did not confirm one GPU query without fallback"
                )
    if actual != expected:
        raise ValueError("host supplied incomplete profile variants")
    if manifest.get("gpu_counters") != {
        "executions": 2 * (args.warmups + args.repetitions),
        "plan_fallbacks": 0,
        "runtime_fallbacks": 0,
    }:
        raise ValueError(
            "profiled GPU total does not cover every warmup and repetition"
        )


def summarize(records):
    """Keep worker duration and query latency distinct, including unavailable stages."""
    groups = {}
    for record in records:
        if record["warmup"]:
            continue
        key = tuple(
            record[k]
            for k in (
                "workload",
                "rows",
                "engine",
                "cpu_mode",
                "cpu_baseline",
                "query_kind",
            )
        )
        groups.setdefault(key, []).append(record)
    summary = []
    for key, samples in sorted(groups.items()):
        common = dict(
            zip(
                (
                    "workload",
                    "rows",
                    "engine",
                    "cpu_mode",
                    "cpu_baseline",
                    "query_kind",
                ),
                key,
            )
        )
        for stage in STAGES:
            if not all(s["stage_available"].get(stage, False) for s in samples):
                continue
            values = [s["stages"][stage] for s in samples]
            summary.append(
                {
                    **common,
                    "stage": stage,
                    "samples": len(values),
                    "median_operator_work_ms": statistics.median(values),
                    "min_operator_work_ms": min(values),
                    "max_operator_work_ms": max(values),
                    "median_query_wall_ms": statistics.median(
                        s["query_wall_ms"] for s in samples
                    ),
                }
            )
    comparisons = []
    gpu = {
        (s["workload"], s["rows"], s["query_kind"], s["stage"]): s
        for s in summary
        if s["engine"] == "gpu"
    }
    for cpu in summary:
        if cpu["engine"] != "cpu":
            continue
        match = gpu.get((cpu["workload"], cpu["rows"], cpu["query_kind"], cpu["stage"]))
        if match is None:
            continue
        cpu_ms, gpu_ms = (
            cpu["median_operator_work_ms"],
            match["median_operator_work_ms"],
        )
        comparisons.append(
            {
                **{
                    k: cpu[k]
                    for k in ("workload", "rows", "query_kind", "stage", "cpu_baseline")
                },
                "cpu_operator_work_ms": cpu_ms,
                "gpu_operator_work_ms": gpu_ms,
                "operator_work_ratio": cpu_ms / gpu_ms if gpu_ms > 0 else None,
            }
        )
    return summary, comparisons


def save(output, directory):
    summary, comparisons = summarize(output["records"])
    output["summary"] = summary
    output["comparisons"] = comparisons
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "results.json")
    for name, rows, fields in (
        (
            "summary.csv",
            summary,
            [
                "workload",
                "rows",
                "engine",
                "cpu_mode",
                "cpu_baseline",
                "query_kind",
                "stage",
                "samples",
                "median_operator_work_ms",
                "min_operator_work_ms",
                "max_operator_work_ms",
                "median_query_wall_ms",
            ],
        ),
        (
            "comparison.csv",
            comparisons,
            [
                "workload",
                "rows",
                "query_kind",
                "stage",
                "cpu_baseline",
                "cpu_operator_work_ms",
                "gpu_operator_work_ms",
                "operator_work_ratio",
            ],
        ),
    ):
        with (directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = run_bench.metadata(args)
    metadata.update(
        {
            "title": "Sirius UDF query operator profile",
            "command": [
                sys.executable,
                str(Path(__file__).resolve()),
                *(sys.argv[1:] if argv is None else argv),
            ],
            "timing_scope": "Actual query operator wall durations, summed across calls/workers; query wall is retained separately. GPU execute durations synchronize their task stream. No device-only kernel timing.",
            "cache_modes": "Checkpointed native DuckDB storage with warm CPU/OS caches; input is not pinned to GPU.",
            "timing_model": "cumulative_worker_wall",
            "cpu_operator_threads": args.threads,
            "gpu_executor_threads": 1,
            "batch_target_bytes": args.batch_mib * MIB,
            "gpu_pool_target_bytes": args.gpu_budget_mib * MIB,
            "host_pool_target_bytes": args.host_budget_mib * MIB,
            "limitations": [
                "Operator duration includes framework launches, allocations and NULL handling; projection is not isolated callback or kernel time.",
                "CPU and GPU operator sums are worker wall durations, not CPU instruction time or query critical-path latency. Parallel CPU worker counts differ when --threads exceeds one.",
                "Input preparation/scan combines scan, decode and transfers; collector sink combines D2H and host materialization. Stage boundaries differ across engines.",
                "Pool budgets and scan batch sizes are targets, not a hard global VRAM cap. Runtime/context/library allocations are outside tracked pools.",
                "GPU log duration resolution is 0.01 ms; zero values cannot support an infinite speedup claim.",
                "Pool peaks are lifetime high-water marks; per-operator allocator peaks are not sums or globally simultaneous peaks.",
                "TRACE/profiling adds overhead. These are diagnostic samples separate from the end-to-end benchmark.",
                "GPU clocks and other machine activity remain uncontrolled.",
            ],
        }
    )
    output = {"metadata": metadata, "records": [], "jobs": [], "failures": []}
    save(output, args.output_dir)
    try:
        device = device_info(args.build_dir)
        yaml = configuration(args, device)
        metadata["selected_visible_device"] = {"ordinal": 0, **device}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        output["failures"].append({"phase": "device_probe", "error": str(error)})
        save(output, args.output_dir)
        print(f"Stage profiling failed: {error}", file=sys.stderr)
        return 1

    # Import only after writing the initial failure-preserving report.
    from operator_profiles import collect

    for workload in args.workloads:
        for rows in args.sizes:
            job = run_bench.make_job(workload, rows, args)
            with (job / "config.txt").open("a") as stream:
                stream.write("stage_profile=1\nstage_aggregate=1\n")
            config_file = job / "sirius.yaml"
            config_file.write_text(yaml)
            environment = os.environ.copy()
            environment.pop("SIRIUS_DISABLE", None)
            environment.update(
                {
                    "SIRIUS_CONFIG_FILE": str(config_file),
                    "SIRIUS_LOG_BACKEND": "spdlog",
                    "SIRIUS_LOG_LEVEL": "trace",
                    "SIRIUS_LOG_DIR": str(job / "engine-log"),
                }
            )
            started = time.perf_counter()
            try:
                with (job / "run.log").open("w") as log:
                    process = subprocess.run(
                        [str(args.build_dir / "bench_host"), str(job)],
                        cwd=job,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=1800,
                        check=False,
                    )
                if process.returncode:
                    raise RuntimeError(
                        f"native host exited {process.returncode}; inspect {job / 'run.log'}"
                    )
                manifest = json.loads((job / "stage-measurements.json").read_text())
                validate_manifest(manifest, args, workload, rows)
                records = collect(job, manifest, args.threads)
                if len(records) != len(manifest["profiles"]):
                    raise ValueError(
                        "operator collection did not cover every profiled query"
                    )
                for record in records:
                    record.update({"workload": workload, "rows": rows})
                output["records"].extend(records)
                output["jobs"].append(
                    {
                        "workload": workload,
                        "rows": rows,
                        "directory": str(job.relative_to(args.output_dir)),
                        "correctness_verified": True,
                        "gpu_validation_queries": manifest["gpu_validation_queries"],
                        "gpu_counters": manifest["gpu_counters"],
                        "validation_records": manifest["validation_records"],
                        "wall_seconds": time.perf_counter() - started,
                    }
                )
                print(
                    f"{workload} {rows}: stage profiles and correctness passed",
                    flush=True,
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as error:
                output["failures"].append(
                    {
                        "workload": workload,
                        "rows": rows,
                        "directory": str(job.relative_to(args.output_dir)),
                        "error": str(error),
                    }
                )
                print(f"{workload} {rows}: {error}", file=sys.stderr, flush=True)
            save(output, args.output_dir)
    return 1 if output["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
