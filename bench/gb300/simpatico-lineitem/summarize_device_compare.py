#!/usr/bin/env python3
"""Aggregate the device-resident Parquet versus Simpatico SF1000 run."""

import argparse
import csv
from pathlib import Path


PARTITIONS = 60
COLUMNS = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    order = None
    for partition in range(PARTITIONS):
        path = args.input_dir / f"part.{partition}.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        names = [row["column"] for row in rows]
        if len(rows) != COLUMNS or len(set(names)) != COLUMNS:
            raise ValueError(f"unexpected columns in {path}")
        if order is None:
            order = names
        elif names != order:
            raise ValueError(f"column order differs in {path}")
        if any(row["verify_ok"] != "1" for row in rows):
            raise ValueError(f"round-trip verification failed in {path}")
        all_rows.extend({"partition": partition, **row} for row in rows)

    with (args.output_dir / "device-compare-per-partition.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(all_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(all_rows)

    fields = (
        "column",
        "rows",
        "decoded_bytes",
        "simpatico_payload_bytes",
        "parquet_ms_sum_of_partition_medians",
        "parquet_gbps",
        "simpatico_ms_sum_of_partition_medians",
        "simpatico_gbps",
        "simpatico_speedup",
    )
    with (args.output_dir / "device-compare-summary.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in order:
            selected = [row for row in all_rows if row["column"] == name]
            decoded = sum(int(row["decoded_bytes"]) for row in selected)
            parquet_ms = sum(float(row["parquet_ms_median"]) for row in selected)
            simpatico_ms = sum(float(row["simpatico_ms_median"]) for row in selected)
            writer.writerow(
                {
                    "column": name,
                    "rows": sum(int(row["rows"]) for row in selected),
                    "decoded_bytes": decoded,
                    "simpatico_payload_bytes": sum(
                        int(row["simpatico_payload_bytes"]) for row in selected
                    ),
                    "parquet_ms_sum_of_partition_medians": f"{parquet_ms:.6f}",
                    "parquet_gbps": f"{decoded / (1e6 * parquet_ms):.6f}",
                    "simpatico_ms_sum_of_partition_medians": f"{simpatico_ms:.6f}",
                    "simpatico_gbps": f"{decoded / (1e6 * simpatico_ms):.6f}",
                    "simpatico_speedup": f"{parquet_ms / simpatico_ms:.6f}",
                }
            )


if __name__ == "__main__":
    main()
