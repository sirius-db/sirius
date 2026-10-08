#!/usr/bin/env python3
"""Run alternating process blocks around the C++ query-cost measurement entry."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def parse_samples(log):
    groups = {mode: [] for mode in ("disabled", "enabled")}
    summaries = {}
    for line in log.splitlines():
        for kind in ("SAMPLE", "SUMMARY"):
            prefix = f"PREPARATION_COST_{kind} "
            if prefix not in line:
                continue
            raw = line.split(prefix, 1)[1]
            fields = dict(token.split("=", 1) for token in raw.split())
            mode = fields["observations"]
            if mode not in groups or fields["cache"] != "warm":
                raise ValueError("unexpected observation mode or cache state")
            if kind == "SUMMARY":
                if mode in summaries:
                    raise ValueError("duplicate summary")
                summaries[mode] = fields
            else:
                sample, total = int(fields["sample"]), int(fields["total_us"])
                if sample != len(groups[mode]) or total < 0:
                    raise ValueError("invalid sample sequence or duration")
                groups[mode].append(total)
    for mode, samples in groups.items():
        if mode not in summaries:
            raise ValueError(
                f"missing {mode} summary; a skipped test is not a measurement"
            )
        summary = summaries[mode]
        if not samples or len(samples) != int(summary["samples"]):
            raise ValueError("missing samples; a skipped test is not a measurement")
        if summary["percentile"] != "nearest_rank" or any(
            percentile(samples, fraction) != int(summary[f"total_p{rank}_us"])
            for rank, fraction in ((50, 0.50), (95, 0.95))
        ):
            raise ValueError("sample/summary percentile mismatch")
    return groups, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--sql", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4, help="process blocks per side")
    parser.add_argument("--baseline-cwd", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--table", help="optional exact Iceberg table root for SQL file logs"
    )
    args = parser.parse_args()
    if args.blocks < 2 or args.blocks % 2:
        parser.error(
            "--blocks must be positive and even for balanced A/B and B/A order"
        )
    sql = args.sql.resolve(strict=True)
    commands = {
        side: [str(getattr(args, side).resolve(strict=True)), "[preparation_cost]"]
        for side in ("baseline", "candidate")
    }
    directories = {
        side: str(getattr(args, f"{side}_cwd").resolve(strict=True))
        for side in commands
    }
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, SIRIUS_TEST_PREPARATION_COST_SQL_FILE=str(sql))
    env.pop("SIRIUS_TEST_PREPARATION_COST_TABLE", None)
    if args.table:
        env["SIRIUS_TEST_PREPARATION_COST_TABLE"] = args.table
    manifest = {
        "commands": commands,
        "cwd": directories,
        "sql": str(sql),
        "table": args.table,
        "sql_sha256": hashlib.sha256(sql.read_bytes()).hexdigest(),
        "blocks": args.blocks,
        "runs": [],
    }
    pooled = {side: {mode: [] for mode in ("disabled", "enabled")} for side in commands}
    shape = None
    for block in range(args.blocks):
        for side in (
            ("baseline", "candidate") if block % 2 == 0 else ("candidate", "baseline")
        ):
            stem = args.output / f"{block:02d}-{side}"
            with stem.with_suffix(".log").open("w") as log:
                result = subprocess.run(
                    commands[side],
                    cwd=directories[side],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            run = {"block": block, "side": side, "returncode": result.returncode}
            manifest["runs"].append(run)
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            if result.returncode:
                raise RuntimeError(f"{side} block {block} failed; see {stem}.log")
            groups, summaries = parse_samples(stem.with_suffix(".log").read_text())
            run["summaries"] = summaries
            current = [
                (mode, len(samples), int(summaries[mode]["warmups"]))
                for mode, samples in groups.items()
            ]
            if shape is not None and current != shape:
                raise ValueError("A/B blocks have different sample or warmup counts")
            shape = current
            for mode, samples in groups.items():
                pooled[side][mode].extend(samples)
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    report = {}
    for mode in ("disabled", "enabled"):
        report[mode] = {
            side: {
                "samples": len(values[mode]),
                "p50_us": percentile(values[mode], 0.50),
                "p95_us": percentile(values[mode], 0.95),
            }
            for side, values in pooled.items()
        }
        baseline = report[mode]["baseline"]["p95_us"]
        report[mode]["p95_ratio"] = (
            report[mode]["candidate"]["p95_us"] / baseline if baseline else None
        )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
