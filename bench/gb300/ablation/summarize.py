#!/usr/bin/env python3
"""Summarize a completed GB300 SF1000 host-pin ablation output directory."""

import argparse
import csv
from pathlib import Path


STAGES = (
    "all_off",
    "type_narrow",
    "compressed_pin",
    "fused_decode",
    "build_probe",
    "dynamic_filter",
)
QUERIES = tuple(f"q{i}" for i in range(1, 23))


def read_stage(root: Path, stage: int, label: str) -> dict[str, float] | None:
    directory = root / f"stage{stage}-{label}"
    exit_code = directory / "exit-code"
    if not exit_code.is_file() or exit_code.read_text().strip() != "0":
        print(f"Stage {stage} ({label}): incomplete or failed")
        return None
    files = list(directory.glob("tpch_*/csv/runtimes.csv"))
    if len(files) != 1:
        raise ValueError(
            f"Stage {stage}: expected one runtimes.csv, found {len(files)}"
        )
    with files[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        row
        for row in rows
        if row["engine"] == "sirius"
        and row["query"] in QUERIES
        and row["iteration"] == "0"
    ]
    if len(selected) != 22 or {row["query"] for row in selected} != set(QUERIES):
        raise ValueError(f"Stage {stage}: expected one timing for every Q1–Q22")
    return {row["query"]: float(row["runtime_s"]) for row in selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--baseline-vs-simpatico",
        action="store_true",
        help="summarize the all-off versus Simpatico-only comparison",
    )
    comparison.add_argument(
        "--all-on-vs-simpatico",
        action="store_true",
        help="summarize all-on runs with host pin compression off and on",
    )
    comparison.add_argument(
        "--join-first-order",
        action="store_true",
        help="summarize the BUILD_PROBE, dynamic filter, narrowing, compression, fused-decode order",
    )
    comparison.add_argument(
        "--narrow-first-order",
        action="store_true",
        help="summarize the narrowing, BUILD_PROBE, dynamic filter, compression, fused-decode order",
    )
    comparison.add_argument(
        "--compressed-first-order",
        action="store_true",
        help="summarize the compression, narrowing, BUILD_PROBE, dynamic filter, fused-decode order",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    if not root.is_dir():
        parser.error(f"missing output directory: {root}")

    if args.baseline_vs_simpatico:
        stages = ((0, "all_off"), (6, "simpatico_only"))
    elif args.all_on_vs_simpatico:
        stages = ((7, "all_on_uncompressed"), (8, "all_on_compressed"))
    elif args.join_first_order:
        stages = (
            (9, "all_off"),
            (10, "build_probe"),
            (11, "dynamic_filter"),
            (12, "type_narrow"),
            (13, "compressed_pin"),
            (14, "fused_decode"),
        )
    elif args.narrow_first_order:
        stages = (
            (15, "all_off"),
            (16, "type_narrow"),
            (17, "build_probe"),
            (18, "dynamic_filter"),
            (19, "compressed_pin"),
            (20, "fused_decode"),
        )
    elif args.compressed_first_order:
        stages = (
            (21, "all_off"),
            (22, "compressed_pin"),
            (23, "type_narrow"),
            (24, "build_probe"),
            (25, "dynamic_filter"),
            (26, "fused_decode"),
        )
    else:
        stages = tuple(enumerate(STAGES))
    all_times = [read_stage(root, stage, label) for stage, label in stages]
    summary = root / "summary.csv"
    with summary.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "stage",
                "optimization_added",
                "status",
                "query_total_s",
                "change_from_previous_s",
                "change_from_previous_pct",
            )
        )
        for position, ((stage, label), timings) in enumerate(zip(stages, all_times)):
            if timings is None:
                writer.writerow((stage, label, "incomplete", "", "", ""))
                continue
            total = sum(timings.values())
            previous = all_times[position - 1] if position else None
            previous_total = sum(previous.values()) if previous else None
            delta = total - previous_total if previous_total is not None else None
            pct = 100 * delta / previous_total if delta is not None else None
            writer.writerow(
                (
                    stage,
                    label,
                    "success",
                    f"{total:.6f}",
                    f"{delta:.6f}" if delta is not None else "",
                    f"{pct:.3f}" if pct is not None else "",
                )
            )
            print(
                f"Stage {stage} ({label}): {total:.3f}s"
                + (f", {pct:+.2f}% vs previous" if pct is not None else "")
            )

    per_query = root / "per-query.csv"
    with per_query.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "query",
                *[f"stage{stage}_{label}_s" for stage, label in stages],
            )
        )
        for query in QUERIES:
            writer.writerow(
                (
                    query,
                    *[
                        f"{timings[query]:.6f}" if timings else ""
                        for timings in all_times
                    ],
                )
            )
    print(f"Saved {summary} and {per_query}")


if __name__ == "__main__":
    main()
