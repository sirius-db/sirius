#!/usr/bin/env python3
"""Verify every TPC-H Parquet page and save reproducible generation metadata."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MULTIPLIERS = {
    "customer": 150_000,
    "lineitem": 6_000_000,
    "nation": 0,
    "orders": 1_500_000,
    "part": 200_000,
    "partsupp": 800_000,
    "region": 0,
    "supplier": 10_000,
}
COLUMNS = {
    "customer": "c_custkey c_name c_address c_nationkey c_phone c_acctbal c_mktsegment c_comment",
    "lineitem": "l_orderkey l_partkey l_suppkey l_linenumber l_quantity l_extendedprice l_discount l_tax l_returnflag l_linestatus l_shipdate l_commitdate l_receiptdate l_shipinstruct l_shipmode l_comment",
    "nation": "n_nationkey n_name n_regionkey n_comment",
    "orders": "o_orderkey o_custkey o_orderstatus o_totalprice o_orderdate o_orderpriority o_clerk o_shippriority o_comment",
    "part": "p_partkey p_name p_mfgr p_brand p_type p_size p_container p_retailprice p_comment",
    "partsupp": "ps_partkey ps_suppkey ps_availqty ps_supplycost ps_comment",
    "region": "r_regionkey r_name r_comment",
    "supplier": "s_suppkey s_name s_address s_nationkey s_phone s_acctbal s_comment",
}
DECIMALS = set(
    "c_acctbal l_quantity l_extendedprice l_discount l_tax o_totalprice p_retailprice ps_supplycost s_acctbal".split()
)
INTS = set("l_linenumber o_shippriority p_size ps_availqty".split())
DATES = set("l_shipdate l_commitdate l_receiptdate o_orderdate".split())
PIN = "cdcf74def0072f94bf1886667e8d2ac51feb8721"


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def file_stat(path):
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def expected_counts(generator, scale):
    """Replay only the pinned generator's RNG to independently count lineitems."""
    counter_source = Path(__file__).with_name("tpch-lineitem-count.rs")
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "-p", "tpchgen", "-j", "4"],
        cwd=generator,
        env=os.environ | {"RUSTFLAGS": "-C target-cpu=native"},
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="tpch-row-counter-") as directory:
        counter_binary = Path(directory) / "lineitem-count"
        subprocess.run(
            [
                "rustc",
                "--edition=2021",
                "-O",
                str(counter_source),
                "--extern",
                f"tpchgen={generator / 'target/release/libtpchgen.rlib'}",
                "-o",
                str(counter_binary),
            ],
            check=True,
        )
        # Known SF100 count verifies this independent count path against prior data.
        if int(command(str(counter_binary), "100")) != 600_037_902:
            raise ValueError("Lineitem RNG counter disagrees with known SF100 count")
        lineitems = int(command(str(counter_binary), str(scale)))
    expected = {table: scale * factor for table, factor in MULTIPLIERS.items()}
    expected |= {"nation": 25, "region": 5, "lineitem": lineitems}
    return expected, {
        "method": "Replay OrderGenerator::create_line_count_random for every order using the pinned Rust generator library; independent of Parquet metadata.",
        "source_file": str(counter_source),
        "source_sha256": sha256(counter_source),
        "known_sf100_count_verified": 600_037_902,
        "orders": expected["orders"],
        "lineitems": lineitems,
    }


def verify_file(root, table, path):
    started = time.monotonic()
    before = file_stat(path)
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    if schema.names != COLUMNS[table].split():
        raise ValueError(f"{path}: unexpected column names/order: {schema.names}")
    for field in schema:
        if field.name in DECIMALS:
            valid = field.type == pa.decimal128(15, 2)
        elif field.name in DATES:
            valid = field.type == pa.date32()
        elif field.name in {"l_orderkey", "o_orderkey"}:
            valid = field.type == pa.int64()
        elif field.name.endswith("key") or field.name in INTS:
            valid = field.type == pa.int32()
        else:
            valid = (
                pa.types.is_string(field.type)
                or pa.types.is_large_string(field.type)
                or pa.types.is_string_view(field.type)
            )
        if not valid or field.nullable:
            raise ValueError(f"{path}: unexpected column type or nullability: {field}")
    row_groups = [
        metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)
    ]
    if not row_groups or any(rows <= 0 for rows in row_groups):
        raise ValueError(f"{path}: missing or empty row groups")
    if sum(row_groups) != metadata.num_rows:
        raise ValueError(f"{path}: inconsistent row group counts")
    decoded_rows = 0
    for batch in parquet.iter_batches(batch_size=131072, use_threads=False):
        batch.validate(full=True)
        if any(column.null_count for column in batch.columns):
            raise ValueError(f"{path}: unexpected null values")
        decoded_rows += batch.num_rows
    if decoded_rows != metadata.num_rows:
        raise ValueError(f"{path}: decoded row count disagrees with metadata")
    digest = sha256(path)
    if before != file_stat(path):
        raise ValueError(f"{path}: file changed during verification")
    result = {
        "path": str(path.relative_to(root)),
        "bytes": before["size"],
        "sha256": digest,
        "stat": before,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "row_group_rows": row_groups,
        "decoded_rows": decoded_rows,
        "decoded_columns": len(schema),
        "null_count": 0,
        "schema": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ],
        "parquet_format_version": metadata.format_version,
        "created_by": metadata.created_by,
        "verification_seconds": time.monotonic() - started,
    }
    print(
        f"Verified {result['path']}: {decoded_rows:,} rows, "
        f"{result['bytes'] / 2**30:.3f} GiB, SHA256 {digest}",
        flush=True,
    )
    return table, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--scale-factor", type=int, default=100)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.scale_factor < 1:
        parser.error("--scale-factor must be positive")
    root, generator = args.dataset.resolve(), args.generator.resolve()
    head = command("git", "-C", str(generator), "rev-parse", "HEAD")
    if head != PIN:
        raise ValueError(f"Generator HEAD {head} does not match expected {PIN}")
    if command("git", "-C", str(generator), "diff", "HEAD", "--"):
        raise ValueError("Generator has tracked source changes")
    options_path = root / "metadata.json"
    if not options_path.exists():
        raise ValueError(
            "Generator metadata.json is missing; generation did not finish"
        )
    metadata = json.loads(options_path.read_text())
    options = metadata.get("options", metadata)
    for option, required in {
        "scale_factor": args.scale_factor,
        "format": "parquet",
        "decimal_column_type": "decimal128",
        "date_column_type": "date32",
        "nationkey_type": "i32",
        "regionkey_type": "i32",
    }.items():
        if options.get(option) != required:
            raise ValueError(
                f"Generator metadata {option}={options.get(option)!r}, expected {required!r}"
            )
    expected, count_evidence = expected_counts(generator, args.scale_factor)
    partitions = {
        table: max(1, (args.scale_factor * factor + 99_999_999) // 100_000_000)
        for table, factor in MULTIPLIERS.items()
    }
    jobs = []
    for table in expected:
        files = sorted((root / table).glob("*.parquet"))
        expected_names = {f"part.{index}.parquet" for index in range(partitions[table])}
        if {path.name for path in files} != expected_names:
            raise ValueError(f"{table}: expected files {sorted(expected_names)}")
        jobs.extend((root, table, path) for path in files)
    if set(root.rglob("*.parquet")) != {job[2] for job in jobs}:
        raise ValueError("Unexpected Parquet files outside the eight table directories")
    started = time.monotonic()
    tables = {
        table: {"expected_rows": count, "rows": 0, "bytes": 0, "files": []}
        for table, count in expected.items()
    }
    pa.set_cpu_count(args.jobs)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(verify_file, *job) for job in jobs]
        for future in as_completed(futures):
            table, result = future.result()
            tables[table]["files"].append(result)
            tables[table]["rows"] += result["rows"]
            tables[table]["bytes"] += result["bytes"]
    for table, info in tables.items():
        if info["rows"] != info["expected_rows"]:
            raise ValueError(
                f"{table}: count {info['rows']} != {info['expected_rows']}"
            )
        table_metadata = metadata["tables"][table]
        if table_metadata["row_count"] != info["rows"] or table_metadata[
            "partition_count"
        ] != len(info["files"]):
            raise ValueError(
                f"{table}: generator metadata disagrees with verified files"
            )
        info["files"].sort(key=lambda item: item["path"])
    lock_copy = root / "generator-Cargo.lock"
    if lock_copy.exists() and sha256(lock_copy) != sha256(generator / "Cargo.lock"):
        raise ValueError("Generator dependency lock changed since dataset generation")
    shutil.copyfile(generator / "Cargo.lock", lock_copy)
    manifest = {
        "scale_factor": args.scale_factor,
        "status": "PASS",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "verification_scope": "All rows and columns decoded and validated; exact per-table counts; no null values; exact decimal/date/key schemas; SHA256 of every Parquet file.",
        "verification_seconds": time.monotonic() - started,
        "lineitem_count_evidence": count_evidence,
        "pyarrow_version": pa.__version__,
        "generator": {
            "repository": "https://github.com/sirius-db/tpchgen-rs.git",
            "head": head,
            "source_clean": True,
            "lockfile_sha256": sha256(generator / "Cargo.lock"),
            "lockfile_copy": lock_copy.name,
            "binary_sha256": sha256(generator / "target/release/tpchgen-cli"),
            "rustc": command("rustc", "--version"),
            "cargo": command("cargo", "--version"),
            "build_flags": "RUSTFLAGS='-C target-cpu=native' CARGO_BUILD_JOBS=4 cargo build --release -p tpchgen-cli",
        },
        "options": options,
        "tables": tables,
        "total_bytes": sum(info["bytes"] for info in tables.values()),
        "total_rows": sum(info["rows"] for info in tables.values()),
    }
    destination = root / "generation-manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    print(
        f"SF{args.scale_factor} PASS: {manifest['total_bytes'] / 2**30:.3f} GiB; {destination}"
    )


if __name__ == "__main__":
    main()
