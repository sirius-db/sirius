#!/usr/bin/env python3
"""Alternate frozen baseline/optimized SF500 blocks on exactly two CNs."""

import argparse
import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--queries", nargs="+", default=[f"q{i:02}" for i in range(1, 23)]
    )
    args = parser.parse_args()
    if (
        args.repetitions < 1
        or len(set(args.queries)) != len(args.queries)
        or not set(args.queries) <= {f"q{i:02}" for i in range(1, 23)}
    ):
        parser.error("Positive repetitions and unique q01..q22 required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    schedule = [
        {"query": query, "arm": arm}
        for index, query in enumerate(args.queries)
        for arm in (
            ("baseline", "optimized") if index % 2 == 0 else ("optimized", "baseline")
        )
    ]
    (output / "schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
    child = None

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        for block in schedule:
            arm, query = block["arm"], block["query"]
            print(
                f"Starting {arm} {query} at {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
                flush=True,
            )
            (output / "status.json").write_text(
                json.dumps({"phase": "running", **block}, indent=2) + "\n"
            )
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "scripts/local-gb10/benchmark-multi-cn.py"),
                    "--arm",
                    arm,
                    "--output",
                    str(output / arm / query),
                    "--queries",
                    query,
                    "--repetitions",
                    str(args.repetitions),
                ],
                cwd=ROOT,
            )
            result = child.wait()
            child = None
            if result:
                raise RuntimeError(
                    f"Benchmark setup or harness failed for {arm} {query}: {result}"
                )
        (output / "status.json").write_text(
            json.dumps({"phase": "complete", "blocks": len(schedule)}, indent=2) + "\n"
        )
    finally:
        if child and child.poll() is None:
            child.terminate()
            child.wait(timeout=90)


if __name__ == "__main__":
    main()
