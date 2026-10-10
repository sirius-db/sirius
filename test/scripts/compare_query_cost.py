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


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def parse_observations(log):
    """Keep every per-scan record; repeated keys must not overwrite each other."""
    observations = []
    for line in log.splitlines():
        prefix = "PREPARATION_COST_SAMPLE "
        if prefix not in line:
            continue
        fields = {}
        for token in line.split(prefix, 1)[1].split():
            key, value = token.split("=", 1)
            fields.setdefault(key, []).append(value)
        observations.append(fields)
    return observations


def require_deferred_routes(observations, expected):
    observed = [row for row in observations if row.get("observations") == ["enabled"]]
    if not observed:
        raise ValueError("no enabled observations to verify candidate routes")
    for row in observed:
        routes = row.get("scan_route", [])
        if (
            row.get("legacy_scans") != ["0"]
            or row.get("deferred_scans") != [str(expected)]
            or len(routes) != expected
            or len(set(routes)) != expected
            or any(not route.endswith(":deferred") for route in routes)
        ):
            raise ValueError(
                f"candidate sample {row.get('sample')} did not use the required routes"
            )


def parse_backend_reads(observations):
    columns = (
        "requests",
        "completions",
        "bytes_requested",
        "bytes_returned",
        "failures",
        "short_reads",
        "retries",
        "unobserved_completions",
    )
    result = []
    for row in observations:
        phases = {}
        for phase in ("planning", "preparation", "execution"):
            value = row.get(f"local_backend_{phase}")
            if value is None:
                phases[phase] = None
                continue
            if len(value) != 1 or len(value[0].split(":")) != len(columns):
                raise ValueError("invalid local backend read counters")
            counters = dict(zip(columns, map(int, value[0].split(":"))))
            if any(n < 0 for n in counters.values()) or (
                counters["requests"] != counters["completions"]
                or counters["unobserved_completions"] != 0
                or counters["bytes_returned"] > counters["bytes_requested"]
            ):
                raise ValueError("incomplete local backend read counters")
            phases[phase] = counters
        result.append(
            {
                "sample": row.get("sample"),
                "observations": row.get("observations"),
                "coverage": row.get("backend_io", ["unobserved"]),
                "phases": phases,
            }
        )
    return result


def parse_samples(log, cache="warm", modes=("disabled", "enabled")):
    groups = {mode: [] for mode in modes}
    summaries = {}
    for line in log.splitlines():
        for kind in ("SAMPLE", "SUMMARY"):
            prefix = f"PREPARATION_COST_{kind} "
            if prefix not in line:
                continue
            raw = line.split(prefix, 1)[1]
            fields = {}
            for token in raw.split():
                key, value = token.split("=", 1)
                if key in fields and key not in (
                    "scan_route",
                    "admission",
                    "puffin",
                    "sql_file",
                    "datasource_open",
                    "puffin_file",
                ):
                    raise ValueError(f"duplicate scalar field: {key}")
                fields[key] = value
            mode = fields["observations"]
            if mode not in groups or fields["cache"] != cache:
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
        if cache == "cold" and (len(samples) != 1 or int(summary["warmups"]) != 0):
            raise ValueError(
                "cold samples require one query per fresh process and no warmups"
            )
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
    parser.add_argument(
        "--blocks",
        type=int,
        default=4,
        help="process blocks per side (cold: samples per observation mode)",
    )
    parser.add_argument(
        "--cold-files",
        type=Path,
        help="file list covering all local table metadata, Parquet and Puffin files; one absolute path per line",
    )
    parser.add_argument("--baseline-cwd", type=Path, default=Path.cwd())
    parser.add_argument("--candidate-cwd", type=Path, default=Path.cwd())
    parser.add_argument(
        "--candidate-deferred-scans",
        type=int,
        help="require this many deferred scans and no legacy scans in every observed candidate sample",
    )
    parser.add_argument(
        "--table", help="optional exact Iceberg table root for SQL file logs"
    )
    args = parser.parse_args()
    if args.candidate_deferred_scans is not None and args.candidate_deferred_scans < 0:
        parser.error("--candidate-deferred-scans must be nonnegative")
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
    cold_entries = []
    if args.cold_files:
        paths = args.cold_files.read_text().splitlines()
        if not paths or len(set(paths)) != len(paths):
            parser.error("cold file list must be nonempty with no duplicates")
        for name in paths:
            path = Path(name)
            if not path.is_absolute() or not path.is_file() or path.is_symlink():
                parser.error(
                    f"cold input must be an absolute regular local file: {name}"
                )
            cold_entries.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, SIRIUS_TEST_PREPARATION_COST_SQL_FILE=str(sql))
    env.pop("SIRIUS_TEST_PREPARATION_COST_TABLE", None)
    env.pop("SIRIUS_TEST_PREPARATION_COST_COLD_FILES", None)
    env.pop("SIRIUS_TEST_PREPARATION_COST_OBSERVATIONS", None)
    if cold_entries:
        frozen_files = args.output.resolve() / "cold-files.txt"
        frozen_files.write_text("".join(entry["path"] + "\n" for entry in cold_entries))
        env["SIRIUS_TEST_PREPARATION_COST_COLD_FILES"] = str(frozen_files)
    if args.table:
        env["SIRIUS_TEST_PREPARATION_COST_TABLE"] = args.table
    manifest = {
        "cache": "cold" if cold_entries else "warm",
        "cold_files": cold_entries,
        "commands": commands,
        "binary_sha256": {side: file_sha256(cmd[0]) for side, cmd in commands.items()},
        "cwd": directories,
        "sql": str(sql),
        "table": args.table,
        "sql_sha256": hashlib.sha256(sql.read_bytes()).hexdigest(),
        "blocks": args.blocks,
        "candidate_deferred_scans": args.candidate_deferred_scans,
        "runs": [],
    }
    config = env.get("SIRIUS_TEST_INTEGRATION_CONFIG")
    if config:
        manifest["integration_config"] = {
            side: {"path": str(path), "sha256": file_sha256(path)}
            for side, cwd in directories.items()
            for path in [(Path(cwd) / config).resolve(strict=True)]
        }
    pooled = {side: {mode: [] for mode in ("disabled", "enabled")} for side in commands}
    shape = None
    for block in range(args.blocks):
        sides = (
            ("baseline", "candidate") if block % 2 == 0 else ("candidate", "baseline")
        )
        modes = ("disabled", "enabled") if cold_entries else (None,)
        for side, selected_mode in ((side, mode) for mode in modes for side in sides):
            suffix = f"-{selected_mode}" if selected_mode else ""
            stem = args.output / f"{block:02d}-{side}{suffix}"
            if selected_mode:
                env["SIRIUS_TEST_PREPARATION_COST_OBSERVATIONS"] = selected_mode
            with stem.with_suffix(".log").open("w") as log:
                result = subprocess.run(
                    commands[side],
                    cwd=directories[side],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            run = {
                "block": block,
                "side": side,
                "observations": selected_mode,
                "returncode": result.returncode,
            }
            manifest["runs"].append(run)
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
            if result.returncode:
                raise RuntimeError(f"{side} block {block} failed; see {stem}.log")
            log_text = stem.with_suffix(".log").read_text()
            groups, summaries = parse_samples(
                log_text,
                cache=manifest["cache"],
                modes=(selected_mode,) if selected_mode else ("disabled", "enabled"),
            )
            observations = parse_observations(log_text)
            stem.with_suffix(".observations.json").write_text(
                json.dumps(observations, indent=2) + "\n"
            )
            stem.with_suffix(".backend-reads.json").write_text(
                json.dumps(parse_backend_reads(observations), indent=2) + "\n"
            )
            if cold_entries:
                if len(observations) != 1 or any(
                    observations[0].get(key) != [str(value)]
                    for key, value in {
                        "cold_files": len(cold_entries),
                        "cold_bytes": sum(e["bytes"] for e in cold_entries),
                        "cold_resident_pages": 0,
                    }.items()
                ):
                    raise ValueError("missing or inconsistent cold-cache verification")
            if (
                side == "candidate"
                and args.candidate_deferred_scans is not None
                and selected_mode != "disabled"
            ):
                require_deferred_routes(observations, args.candidate_deferred_scans)
            run["summaries"] = summaries
            current = [
                (mode, len(samples), int(summaries[mode]["warmups"]))
                for mode, samples in groups.items()
            ]
            if not cold_entries and shape is not None and current != shape:
                raise ValueError("A/B blocks have different sample or warmup counts")
            shape = current
            for mode, samples in groups.items():
                pooled[side][mode].extend(samples)
            (args.output / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    for entry in cold_entries:
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"cold input changed during measurement: {entry['path']}")
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
