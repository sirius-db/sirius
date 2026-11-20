#!/usr/bin/env python3
"""Aggregate Simpatico's per-column SF1000 lineitem partition benchmarks."""

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
        if len(rows) != COLUMNS + 1 or len(set(names)) != COLUMNS + 1:
            raise ValueError(f"unexpected columns in {path}")
        if order is None:
            order = names
        elif names != order:
            raise ValueError(f"column order differs in {path}")
        if names[-1] != "TOTAL" or any(row["verify_ok"] != "1" for row in rows):
            raise ValueError(f"round-trip verification failed in {path}")
        all_rows.extend({"partition": partition, **row} for row in rows)

    with (args.output_dir / "per-partition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    fields = ("column", "rows", "input_bytes", "compressed_bytes", "ratio",
              "compress_ms_sum_of_partition_medians", "compress_gbps",
              "decompress_ms_sum_of_partition_medians", "decompress_gbps")
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name in order:
            selected = [row for row in all_rows if row["column"] == name]
            rows = sum(int(row["rows"]) for row in selected)
            input_bytes = sum(int(row["input_bytes"]) for row in selected)
            compressed_bytes = sum(int(row["compressed_bytes"]) for row in selected)
            compress_ms = sum(float(row["compress_ms_median"]) for row in selected)
            decompress_ms = sum(float(row["decompress_ms_median"]) for row in selected)
            writer.writerow({
                "column": name,
                "rows": rows,
                "input_bytes": input_bytes,
                "compressed_bytes": compressed_bytes,
                "ratio": f"{input_bytes / compressed_bytes:.6f}",
                "compress_ms_sum_of_partition_medians": f"{compress_ms:.6f}",
                "compress_gbps": f"{input_bytes / (1e6 * compress_ms):.6f}",
                "decompress_ms_sum_of_partition_medians": f"{decompress_ms:.6f}",
                "decompress_gbps": f"{input_bytes / (1e6 * decompress_ms):.6f}",
            })


if __name__ == "__main__":
    main()
