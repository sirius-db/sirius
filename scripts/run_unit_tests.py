#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the LICENSE file at the repo root for the full text.
"""Run the Sirius C++ unit tests.

The run has three steps:

  shards     Two Catch2 shards per visible GPU, run in parallel. Each shard sees
             one GPU and uses integration-shard.yaml. [multi_gpu] and hidden
             tests are excluded.
  multi_gpu  The [multi_gpu] tests with all GPUs visible. Skipped when fewer
             than two GPUs are visible.
  late_mat   The late-materialization tests with SIRIUS_EXP_LATE_MAT=1. The
             gate is read once per process, so these tests need a process of
             their own.

The run stops after a step that fails. Arguments after -- are Catch2 options
passed to every process. Every process writes its console output (unittest.log)
and the Sirius logs to its own subdirectory of
<build-dir>/test/cpp/log.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
UNITTEST_DIR = Path("test/cpp")
SHARD_CONFIG = REPO_ROOT / "test/cpp/integration/integration-shard.yaml"
STEPS = ("shards", "multi_gpu", "late_mat")
STEP_SPECS = {
    "shards": "~[.]~[multi_gpu]",
    "multi_gpu": "[multi_gpu]~[.]",
    "late_mat": "[late_mat],[deferred_query],[native_filter]",
}
TIMEOUT_MIN = {"shards": 45, "multi_gpu": 20, "late_mat": 20}
SUMMARY_PREFIXES = ("All tests passed", "test cases:", "No tests ran")

_print_lock = threading.Lock()


class Job(NamedTuple):
    """One test process."""

    label: str
    cmd: list[str]
    env: dict[str, str]
    log_dir: Path
    prefix: str = ""


class Result(NamedTuple):
    """Outcome of one test process."""

    label: str
    status: int
    seconds: float
    summary: str


class Launcher:
    """Starts test processes and terminates all of them on cancel()."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: list[subprocess.Popen] = []
        self._cancelled = False

    def popen(self, *args, **kwargs) -> subprocess.Popen | None:
        """Start a process, or return None after cancel()."""
        with self._lock:
            if self._cancelled:
                return None
            proc = subprocess.Popen(*args, **kwargs)
            self._procs.append(proc)
            return proc

    def cancel(self) -> None:
        """Terminate every started process and keep new ones from starting."""
        with self._lock:
            self._cancelled = True
            for proc in self._procs:
                proc.terminate()


_launcher = Launcher()


def visible_gpus() -> list[str]:
    """GPU ids from CUDA_VISIBLE_DEVICES when set, otherwise from nvidia-smi."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        try:
            ids = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            ).stdout.split()
        except OSError:
            ids = []
    return [i.strip() for i in ids if i.strip()]


def run_job(job: Job) -> Result:
    """Run one test process and tee its output to its log directory."""
    shutil.rmtree(job.log_dir, ignore_errors=True)
    job.log_dir.mkdir(parents=True)
    summary = "no Catch2 summary"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="sirius-unittest-") as tmp:
        env = {
            **os.environ,
            **job.env,
            "SIRIUS_TEST_LOG_DIR": str(job.log_dir),
            "TMPDIR": tmp,
        }
        proc = _launcher.popen(
            job.cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if proc is None:
            return Result(label=job.label, status=130, seconds=0.0, summary="")
        with proc, open(job.log_dir / "unittest.log", "wb") as log:
            for raw in proc.stdout:
                log.write(raw)
                line = raw.decode(errors="replace").rstrip("\n")
                if line.startswith(SUMMARY_PREFIXES):
                    summary = line
                if line or not job.prefix:
                    with _print_lock:
                        print(job.prefix + line, flush=True)
    return Result(
        label=job.label,
        status=proc.returncode,
        seconds=time.monotonic() - started,
        summary=summary,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build/release"),
        help="build directory (default: build/release)",
    )
    parser.add_argument(
        "--steps",
        default=",".join(STEPS),
        help=f"comma list of {', '.join(STEPS)} (default: all)",
    )
    parser.add_argument(
        "--shards", type=int, help="number of shards (default: 2 per visible GPU)"
    )
    parser.add_argument(
        "--require-gpus",
        type=int,
        default=0,
        metavar="N",
        help="fail when fewer than N GPUs are visible",
    )
    parser.add_argument("catch2_args", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()

    steps = args.steps.split(",")
    if unknown := [s for s in steps if s not in STEPS]:
        parser.error(f"unknown steps: {', '.join(unknown)}")

    build_dir = args.build_dir.resolve()
    binary = build_dir / UNITTEST_DIR / "sirius_unittest"
    log_root = build_dir / UNITTEST_DIR / "log"
    gpus = visible_gpus()
    print(f"Visible GPUs: {', '.join(gpus) or 'none'}", flush=True)
    needed = max(args.require_gpus, 1)
    if len(gpus) < needed:
        print(f"error: needs {needed} GPUs, {len(gpus)} visible", file=sys.stderr)
        return 1

    defaults = [
        *(["--order", "decl"] if "--order" not in args.catch2_args else []),
        *(["--durations", "yes"] if "--durations" not in args.catch2_args else []),
    ]
    results: list[Result] = []
    for step in steps:
        if step == "multi_gpu" and len(gpus) < 2:
            print("Skipping multi_gpu: needs at least 2 GPUs", flush=True)
            continue
        cmd = ["timeout", f"{TIMEOUT_MIN[step]}m", str(binary)]
        cmd += [*defaults, *args.catch2_args, STEP_SPECS[step]]
        match step:
            case "shards":
                count = args.shards or 2 * len(gpus)
                jobs = [
                    Job(
                        label=f"shard-{i}",
                        cmd=cmd
                        + ["--shard-count", str(count), "--shard-index", str(i)],
                        env={
                            "CUDA_VISIBLE_DEVICES": gpus[i % len(gpus)],
                            "SIRIUS_TEST_SINGLE_GPU": "1",
                            "SIRIUS_TEST_INTEGRATION_CONFIG": str(SHARD_CONFIG),
                        },
                        log_dir=log_root / f"shard-{i}",
                        prefix=f"[shard {i}] ",
                    )
                    for i in range(count)
                ]
            case "multi_gpu":
                jobs = [Job(label=step, cmd=cmd, env={}, log_dir=log_root / step)]
            case "late_mat":
                jobs = [
                    Job(
                        label=step,
                        cmd=cmd,
                        env={"SIRIUS_EXP_LATE_MAT": "1"},
                        log_dir=log_root / step,
                    )
                ]
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            try:
                step_results = list(pool.map(run_job, jobs))
            except KeyboardInterrupt:
                # Ctrl-C does not reach the tests in timeout's process group, but timeout
                # passes SIGTERM on to them.
                _launcher.cancel()
                print("Interrupted", file=sys.stderr)
                return 130
        results += step_results
        if any(r.status != 0 for r in step_results):
            print(f"Stopping after the failed {step} step", flush=True)
            break

    print(f"\nUnit test summary (logs in {log_root}/<process>/):")
    for r in results:
        print(f"{r.label:<10} exit {r.status:<4} {r.seconds:5.0f}s  {r.summary}")
    return 0 if all(r.status == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
