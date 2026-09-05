#!/usr/bin/env python3
"""Reproduce the SF500 two-CN workload with immutable per-attempt evidence."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from execution_validation import validate_execution

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "build/tpch-starrocks-sf500-2cn"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--queries", nargs="+", default=[f"q{i:02}" for i in range(1, 23)]
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--transfer-window", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument(
        "--async-sender-dispatch",
        action="store_true",
        help="Queue sender execution before acknowledging FE deployment; record as a separate experiment",
    )
    parser.add_argument(
        "--log-filter",
        default="info",
        help="Rust log filter; use a separate output directory for diagnostic runs",
    )
    parser.add_argument(
        "--analyze-telemetry",
        action="store_true",
        help="Decode raw telemetry immediately after each stopped cluster (otherwise deferred until after the sweep)",
    )
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.repetitions < 1 or not set(args.queries) <= {
        f"q{i:02}" for i in range(1, 23)
    }:
        parser.error("positive repetitions and q01..q22 required")
    uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    if "\n" in uuid:
        raise RuntimeError("This local reproduction harness expects one physical GPU")
    gpu_lock = Path(f"/tmp/sirius-benchmark-{uuid}.lock").open("a+")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    foreign = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if foreign:
        raise RuntimeError("GPU workload already exists: " + foreign)
    binary_dir = ROOT / "build/multi-cn-throughput" / f"{args.arm}-bin"
    cn = binary_dir / "sirius-starrocks-cn"
    engine = binary_dir / "libsirius.so"
    for artifact in (cn, engine):
        if not artifact.is_file():
            raise RuntimeError(f"Missing frozen artifact {artifact}")
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["SIRIUS_EXCHANGE_OPTIMIZED"] = "1" if args.arm == "optimized" else "0"
    env["SIRIUS_CN_ASYNC_SENDER_DISPATCH"] = "1" if args.async_sender_dispatch else "0"
    env["SIRIUS_QUERY_WATCHDOG_SECS"] = "300"
    env["SIRIUS_CN_RPC_TIMEOUT_SECS"] = "900"
    env["SIRIUS_EXCHANGE_STAGING_BYTES"] = "2GiB"
    env["SIRIUS_CN_NIXL_TRANSFER_WINDOW"] = str(args.transfer_window)
    env["RUST_LOG"] = args.log_filter
    config = output / "sirius-config.yaml"
    shutil.copy2(ROOT / "scripts/local-gb10/sirius-sf500-2cn.yaml", config)
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arm": args.arm,
        "artifact_source": (
            {
                "commit": "95bec853b684e1510c7ddb3d9becc9b73374e983",
                "description": "Original all22/integration binaries frozen before implementation",
            }
            if args.arm == "baseline"
            else {
                "description": "Current workspace implementation; see source.patch and untracked_source_sha256"
            }
        ),
        "source_snapshot_scope": "Current harness/workspace provenance; baseline artifact provenance is recorded separately",
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu_uuid": uuid,
        "topology": "two CNs on physical GPU 0",
        "budgets_per_cn": {
            "gpu_gib": 24,
            "host_gib": 8,
            "staging_gib": 2,
            "disk_gib": 512,
        },
        "settings": ["pipeline_dop=1", "cbo_cte_reuse_rate=0"],
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "warmup": env.get("SIRIUS_CN_NIXL_WARMUP", "on"),
        "log_filter": args.log_filter,
        "async_sender_dispatch": args.async_sender_dispatch,
        "telemetry_analysis": (
            "after each block"
            if args.analyze_telemetry
            else "deferred until after timed sweep"
        ),
        "binary_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (cn, engine)
        },
        "queries": args.queries,
        "repetitions": args.repetitions,
        "transfer_window": args.transfer_window if args.arm == "optimized" else 1,
        "cold_definition": "fresh application cluster; OS data cache uncontrolled; transport warmup="
        + env.get("SIRIUS_CN_NIXL_WARMUP", "on"),
        "reference": str(REFERENCE),
        "runs": [],
    }
    source_patch = subprocess.check_output(["git", "diff", "--binary"], text=True)
    (output / "source.patch").write_text(source_patch)
    manifest["source_patch_sha256"] = hashlib.sha256(source_patch.encode()).hexdigest()
    untracked = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "src",
                "rust",
                "experimental",
                "scripts",
                "docs",
                "STATUS.md",
            ]
        )
        .decode()
        .split("\0")
    )
    manifest["untracked_source_sha256"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in untracked
        if path and (ROOT / path).is_file()
    }
    save(output / "manifest.json", manifest)
    cluster = None
    launcher = None
    launcher_log = None
    generation = 0
    archived_clusters = set()

    def cleanup_owned_processes():
        identity_file = cluster / "process-start-ticks.json"
        if not identity_file.exists():
            return
        identities = json.loads(identity_file.read_text())

        def still_owned(pid, expected):
            try:
                fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                return fields[19] == expected and fields[0] != "Z"
            except FileNotFoundError:
                return False

        for pid, ticks in identities.items():
            if still_owned(pid, ticks):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 20
        while (
            any(still_owned(pid, ticks) for pid, ticks in identities.items())
            and time.monotonic() < deadline
        ):
            time.sleep(0.2)
        for pid, ticks in identities.items():
            if still_owned(pid, ticks):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def stop():
        nonlocal launcher, launcher_log
        try:
            if launcher and launcher.poll() is None:
                launcher.terminate()
                try:
                    launcher.wait(timeout=75)
                except subprocess.TimeoutExpired:
                    launcher.kill()
                    launcher.wait(timeout=10)
        finally:
            try:
                if launcher_log:
                    launcher_log.close()
            finally:
                if cluster:
                    cleanup_owned_processes()
        if cluster and cluster not in archived_clusters:
            with (cluster / "activity.json").open("w") as activity:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/local-gb10/cn-activity.py",
                        "--dir",
                        str(cluster),
                        "--json",
                    ],
                    stdout=activity,
                    check=False,
                )
            for engine_dir in (cluster / "engine").glob("cn*"):
                if (engine_dir / "telemetry_data").is_dir() and not (
                    engine_dir / "telemetry"
                ).exists():
                    (engine_dir / "telemetry").symlink_to(
                        "telemetry_data", target_is_directory=True
                    )
            if args.analyze_telemetry:
                with (cluster / "telemetry-distribution.json").open("w") as telemetry:
                    subprocess.run(
                        [
                            sys.executable,
                            "experimental/starrocks/scripts/cn-distribution.py",
                            "--dir",
                            str(cluster / "engine"),
                            "--prefix",
                            "cn",
                            "--all-runs",
                            "--json",
                        ],
                        stdout=telemetry,
                        check=False,
                    )
            archived_clusters.add(cluster)
        launcher = None
        launcher_log = None

    def start():
        nonlocal cluster, launcher, launcher_log, generation
        generation += 1
        cluster = output / f"cluster-{generation:03}"
        launcher_log = (output / f"cluster-{generation:03}.log").open("w")
        launcher = subprocess.Popen(
            [
                sys.executable,
                "scripts/local-gb10/stack.py",
                "--cn-count",
                "2",
                "--cn-binary",
                str(cn),
                "--engine-library-dir",
                str(binary_dir),
                "--sirius-config",
                str(config),
                "--run-dir",
                str(cluster),
                "--timeout",
                "600",
            ],
            env=env,
            stdout=launcher_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 610
        while not (cluster / "ready-compute-nodes.tsv").exists():
            if launcher.poll() is not None:
                raise RuntimeError(f"Cluster startup failed: {cluster}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"Cluster startup timed out: {cluster}")
            time.sleep(1)
        processes = json.loads((cluster / "processes.json").read_text())
        loaded = {}
        for name, pid in processes.items():
            if not name.startswith("cn"):
                continue
            maps = Path(f"/proc/{pid}/maps").read_text()
            if str(engine) not in maps:
                raise RuntimeError(f"{name} did not load the expected engine {engine}")
            (cluster / f"{name}-maps.txt").write_text(maps)
            loaded[name] = {
                "pid": pid,
                "executable": str(Path(f"/proc/{pid}/exe").resolve()),
                "engine": str(engine),
            }
        save(cluster / "runtime-identities.json", loaded)
        # Topology is also checked before and after every timed query by tpch-starrocks.py.
        save(
            output / "status.json",
            {"phase": "ready", "cluster": str(cluster), "launcher_pid": launcher.pid},
        )

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for query in args.queries:
            stop()
            start()
            failed = False
            for repetition in range(args.repetitions):
                run = {
                    "query": query,
                    "repetition": repetition,
                    "phase": "cold" if repetition == 0 else "warm",
                    "cluster": str(cluster),
                }
                if failed:
                    run["status"] = "SKIPPED_AFTER_FAILURE"
                    manifest["runs"].append(run)
                    save(output / "manifest.json", manifest)
                    continue
                attempt = output / query / f"r{repetition:02}"
                query_dir = attempt / query
                query_dir.mkdir(parents=True)
                for name in ("oracle.json", "oracle-fingerprint.txt"):
                    shutil.copy2(REFERENCE / query / name, query_dir / name)
                command = [
                    sys.executable,
                    "scripts/local-gb10/tpch-starrocks.py",
                    "--data-dir",
                    "test_datasets/tpch_parquet_sf500",
                    "--run-dir",
                    str(attempt),
                    "--expected-cns",
                    "2",
                    "--scale-factor",
                    "500",
                    "--timeout",
                    str(args.timeout),
                    "--oracle-memory-limit",
                    "8GB",
                    "--oracle-threads",
                    "4",
                    "--set",
                    "cbo_cte_reuse_rate=0",
                    "--cn-binary",
                    str(cn),
                    "--reuse-oracle",
                    "--explain-costs",
                    "--stop-on-error",
                    "--queries",
                    query,
                ]
                save(
                    output / "status.json",
                    {"phase": "query", **run, "attempt": str(attempt)},
                )
                started = time.monotonic()
                run["started_utc"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                try:
                    with (attempt / "runner.log").open("w") as log:
                        result = subprocess.run(
                            command,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=args.timeout + 90,
                        )
                    run["runner_returncode"] = result.returncode
                    rows = json.loads((attempt / "results.json").read_text())["results"]
                    if len(rows) != 1 or rows[0]["query"] != query:
                        raise RuntimeError(f"Incomplete benchmark output: {attempt}")
                    run.update(rows[0])
                    if result.returncode and run.get("status") == "PASS":
                        run.update(
                            status="RUNNER_ERROR",
                            detail=f"Runner returned {result.returncode} despite a PASS result row",
                        )
                except (
                    OSError,
                    ValueError,
                    KeyError,
                    RuntimeError,
                    subprocess.TimeoutExpired,
                ) as error:
                    run.update(status="RUNNER_ERROR", detail=str(error))
                run.update(
                    attempt=str(attempt),
                    attempt_elapsed_seconds=time.monotonic() - started,
                )
                run["finished_utc"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                execution = run.get("starrocks", {})
                validation = validate_execution(
                    cluster,
                    execution.get("started_utc"),
                    execution.get("finished_utc"),
                )
                run["execution_validation"] = validation
                execution_ineligible = validation["status"] == "INELIGIBLE" or (
                    validation["status"] == "UNKNOWN" and validation["detected_retry"]
                )
                run["benchmark_eligible"] = (
                    run["status"] == "PASS" and not execution_ineligible
                )
                run["benchmark_failure_class"] = (
                    validation["failure_class"] if execution_ineligible else None
                )
                manifest["runs"].append(run)
                save(output / "manifest.json", manifest)
                timing = run.get("starrocks", {}).get("elapsed_seconds")
                measured = (
                    f" {timing:.3f}s" if timing is not None else " (no query timing)"
                )
                print(
                    f"{args.arm} {query} r{repetition}: {run['status']}{measured}"
                    + (
                        f" [{validation['failure_class']}]"
                        if execution_ineligible
                        else ""
                    ),
                    flush=True,
                )
                failed = run["status"] != "PASS" or execution_ineligible
                if failed:
                    # No next query or warm sample may use a cluster that has failed.
                    stop()
            stop()
        save(
            output / "status.json",
            {"phase": "complete", "runs": len(manifest["runs"]), "services": "stopped"},
        )
    finally:
        stop()


if __name__ == "__main__":
    main()
