#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Write local Iceberg V2 tables from the existing TPC-H Parquet dataset.

Run from the project root:
    pixi run python test/tpch_performance/generate_tpch_iceberg.py \
        test_datasets/tpch_parquet_sf1 test_datasets/tpch_iceberg_sf1

PyIceberg rewrites the data with field IDs and creates the manifests and snapshots.
The output owns its data files and contains create_views.sql with exact metadata
paths and snapshot IDs. Input files are read in batches, never a whole table at once.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import sys

TABLES = (
    "region",
    "nation",
    "supplier",
    "customer",
    "part",
    "partsupp",
    "orders",
    "lineitem",
)
DEFAULT_BATCH_ROWS = 1_000_000


@dataclass
class InputTable:
    files: list[Path]
    schema: object
    rows: int


def parquet_files(root: Path, table: str) -> list[Path]:
    """Accept the flat and partitioned layouts used by the TPC-H generators."""
    layouts = [
        [root / f"{table}.parquet"],
        sorted((root / table).glob("*.parquet")),
        sorted(root.glob(f"{table}_*.parquet")),
        sorted(root.glob(f"{table}.[0-9]*.parquet")),
    ]
    layouts = [[path for path in files if path.is_file()] for files in layouts]
    layouts = [files for files in layouts if files]
    if not layouts:
        raise ValueError(f"Missing Parquet data for {table} under {root}")
    if len(layouts) != 1:
        raise ValueError(f"Ambiguous Parquet layouts for {table} under {root}")
    files = layouts[0]
    if len({path.resolve() for path in files}) != len(files):
        raise ValueError(f"Duplicate Parquet files for {table}")
    return files


def input_tables(root: Path) -> dict[str, InputTable]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    inputs = {}
    for table in TABLES:
        files = parquet_files(root, table)
        schema = None
        rows = 0
        for path in files:
            with pq.ParquetFile(path) as parquet:
                # The new Iceberg table assigns its own IDs. Do not propagate IDs
                # or pandas metadata from an unrelated source table.
                actual = pa.schema(
                    [
                        pa.field(field.name, field.type, nullable=field.nullable)
                        for field in parquet.schema_arrow
                    ]
                )
                if schema is not None and not schema.equals(actual):
                    raise ValueError(f"Inconsistent schema for {table}: {path}")
                schema = actual
                rows += parquet.metadata.num_rows
        inputs[table] = InputTable(files, schema, rows)
    return inputs


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def generate(parquet_dir: Path, output_dir: Path, batch_rows: int = DEFAULT_BATCH_ROWS):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyiceberg.catalog.sql import SqlCatalog
    from sqlalchemy.engine import URL

    source = parquet_dir.resolve()
    output = output_dir.resolve()
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if not source.is_dir():
        raise ValueError(f"Parquet input directory does not exist: {source}")
    if output.exists():
        raise ValueError(f"Output already exists; choose a fresh directory: {output}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Input and output directories must not contain one another")

    # Validate all eight tables before creating the warehouse. A missing table
    # or inconsistent shard must not leave a dataset that looks complete.
    inputs = input_tables(source)
    output.mkdir(parents=True)
    catalog = SqlCatalog(
        "tpch",
        uri=URL.create(
            "sqlite", database=str(output / "catalog.sqlite")
        ).render_as_string(),
        warehouse=str(output),
        **{"py-io-impl": "pyiceberg.io.pyarrow.PyArrowFileIO"},
    )
    manifest = {
        "format_version": 2,
        "source_directory": str(source),
        "batch_rows": batch_rows,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("pyiceberg", "pyarrow", "sqlalchemy")
        },
        "tables": {},
    }
    views = [
        "-- Generated local Iceberg views; metadata paths and snapshots are fixed.",
        "LOAD iceberg;",
    ]
    try:
        catalog.create_namespace("tpch")
        for name, data in inputs.items():
            print(
                f"Writing {name}: {data.rows:,} rows from {len(data.files)} file(s)",
                flush=True,
            )
            table = catalog.create_table(
                ("tpch", name),
                schema=data.schema,
                location=str(output / name),
                properties={"format-version": "2"},
            )
            written = 0
            for path in data.files:
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(batch_size=batch_rows):
                        arrow = pa.Table.from_arrays(batch.columns, schema=data.schema)
                        table.append(arrow)
                        written += batch.num_rows
            if written == 0:
                table.append(pa.Table.from_batches([], schema=data.schema))
            if written != data.rows:
                raise ValueError(f"{name}: read {written} rows, expected {data.rows}")

            snapshot = table.current_snapshot()
            if snapshot is None:
                raise ValueError(f"{name}: Iceberg writer produced no snapshot")
            # Verify the committed inventory, including the physical field IDs
            # required by Sirius. The warehouse must stand alone from its input.
            field_ids = {field.name: field.field_id for field in table.schema().fields}
            data_files = []
            committed_rows = 0
            for task in table.scan(snapshot_id=snapshot.snapshot_id).plan_files():
                path = Path(task.file.file_path)
                if not path.resolve().is_relative_to(output):
                    raise ValueError(
                        f"{name}: data file is outside the warehouse: {path}"
                    )
                with pq.ParquetFile(path) as parquet:
                    for field in parquet.schema_arrow:
                        actual = (field.metadata or {}).get(b"PARQUET:field_id")
                        if actual != str(field_ids[field.name]).encode():
                            raise ValueError(
                                f"{name}.{field.name}: missing or incorrect field ID"
                            )
                    committed_rows += parquet.metadata.num_rows
                    if parquet.metadata.num_rows != task.file.record_count:
                        raise ValueError(
                            f"{name}: manifest row count differs from {path}"
                        )
                data_files.append(str(path))
            if committed_rows != data.rows:
                raise ValueError(
                    f"{name}: committed {committed_rows} rows, expected {data.rows}"
                )
            manifest["tables"][name] = {
                "metadata_file": table.metadata_location,
                "snapshot_id": snapshot.snapshot_id,
                "row_count": committed_rows,
                "source_files": [str(path) for path in data.files],
                "data_files": sorted(data_files),
                "field_ids": field_ids,
            }
            views.append(
                f"CREATE VIEW {name} AS SELECT * FROM iceberg_scan("
                f"{sql_literal(table.metadata_location)}, snapshot_from_id={snapshot.snapshot_id});"
            )
    finally:
        catalog.engine.dispose()

    # Publish entry points only after all tables have been written and checked.
    (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "create_views.sql").write_text("\n".join(views) + "\n")
    print(f"Created {output}\nViews: {output / 'create_views.sql'}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parquet_dir", type=Path, help="Existing TPC-H Parquet directory"
    )
    parser.add_argument(
        "output_dir", type=Path, help="New local Iceberg warehouse directory"
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help="Rows per write batch (default: %(default)s); controls memory and output file sizes",
    )
    args = parser.parse_args()
    try:
        generate(args.parquet_dir, args.output_dir, args.batch_rows)
    except ImportError as error:
        print(
            f"Missing generation dependency: {error}. Run with `pixi run python`.",
            file=sys.stderr,
        )
        return 1
    except Exception as error:
        print(f"Iceberg generation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
