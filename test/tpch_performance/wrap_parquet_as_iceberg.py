#!/usr/bin/env python3
"""
Wrap existing TPC-H parquet files as Iceberg V2 tables.

Takes a directory of TPC-H parquet files (e.g. tpch_parquet_sf10/) and creates
Iceberg metadata (metadata.json, avro manifests) around them so they can be
read via iceberg_scan().

Optionally injects equality-delete files for specified tables to test the
equality-delete hook under load.

Usage (from sirius-dev root):
    python3 test/tpch_performance/wrap_parquet_as_iceberg.py \
        test_datasets/tpch_parquet_sf10 \
        test_datasets/tpch_iceberg_sf10 \
        [--delete-pct 5 --delete-tables lineitem]
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import fastavro
import pyarrow.parquet as pq
import pyarrow as pa

# ---------------------------------------------------------------------------
# Avro schemas (same as gen_iceberg_test_data.py)
# ---------------------------------------------------------------------------

MANIFEST_LIST_SCHEMA = {
    "type": "record",
    "name": "manifest_file",
    "fields": [
        {"field-id": 500, "name": "manifest_path", "type": "string"},
        {"field-id": 501, "name": "manifest_length", "type": "long"},
        {"field-id": 502, "name": "partition_spec_id", "type": "int"},
        {"field-id": 517, "name": "content", "type": "int"},
        {"field-id": 515, "name": "sequence_number", "type": "long"},
        {"field-id": 516, "name": "min_sequence_number", "type": "long"},
        {"field-id": 503, "name": "added_snapshot_id", "type": "long"},
        {"field-id": 504, "name": "added_files_count", "type": "int"},
        {"field-id": 505, "name": "existing_files_count", "type": "int"},
        {"field-id": 506, "name": "deleted_files_count", "type": "int"},
        {"field-id": 512, "name": "added_rows_count", "type": "long"},
        {"field-id": 513, "name": "existing_rows_count", "type": "long"},
        {"field-id": 514, "name": "deleted_rows_count", "type": "long"},
        {
            "field-id": 507,
            "name": "partitions",
            "type": [
                "null",
                {
                    "element-id": 508,
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "r508",
                        "fields": [
                            {
                                "field-id": 509,
                                "name": "contains_null",
                                "type": "boolean",
                            },
                            {
                                "field-id": 518,
                                "name": "contains_nan",
                                "type": ["null", "boolean"],
                                "default": None,
                            },
                            {
                                "field-id": 510,
                                "name": "lower_bound",
                                "type": ["null", "bytes"],
                                "default": None,
                            },
                            {
                                "field-id": 511,
                                "name": "upper_bound",
                                "type": ["null", "bytes"],
                                "default": None,
                            },
                        ],
                    },
                },
            ],
            "default": None,
        },
        {
            "field-id": 519,
            "name": "key_metadata",
            "type": ["null", "bytes"],
            "default": None,
        },
    ],
}


def make_manifest_schema(sequence_number: int, snapshot_id: int):
    return {
        "type": "record",
        "name": "manifest_entry",
        "fields": [
            {"field-id": 0, "name": "status", "type": "int"},
            {
                "field-id": 1,
                "name": "snapshot_id",
                "type": ["null", "long"],
                "default": None,
            },
            {
                "field-id": 3,
                "name": "sequence_number",
                "type": ["null", "long"],
                "default": None,
            },
            {
                "field-id": 4,
                "name": "file_sequence_number",
                "type": ["null", "long"],
                "default": None,
            },
            {
                "field-id": 2,
                "name": "data_file",
                "type": {
                    "type": "record",
                    "name": "r2",
                    "fields": [
                        {"field-id": 134, "name": "content", "type": "int"},
                        {"field-id": 100, "name": "file_path", "type": "string"},
                        {"field-id": 101, "name": "file_format", "type": "string"},
                        {
                            "field-id": 102,
                            "name": "partition",
                            "type": {"type": "record", "name": "r102", "fields": []},
                        },
                        {"field-id": 103, "name": "record_count", "type": "long"},
                        {"field-id": 104, "name": "file_size_in_bytes", "type": "long"},
                        {
                            "field-id": 108,
                            "name": "column_sizes",
                            "type": [
                                "null",
                                {
                                    "logicalType": "map",
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k108_v",
                                        "fields": [
                                            {
                                                "field-id": 117,
                                                "name": "key",
                                                "type": "int",
                                            },
                                            {
                                                "field-id": 118,
                                                "name": "value",
                                                "type": "long",
                                            },
                                        ],
                                    },
                                },
                            ],
                            "default": None,
                        },
                        {
                            "field-id": 109,
                            "name": "value_counts",
                            "type": [
                                "null",
                                {
                                    "logicalType": "map",
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k109_v",
                                        "fields": [
                                            {
                                                "field-id": 119,
                                                "name": "key",
                                                "type": "int",
                                            },
                                            {
                                                "field-id": 120,
                                                "name": "value",
                                                "type": "long",
                                            },
                                        ],
                                    },
                                },
                            ],
                            "default": None,
                        },
                        {
                            "field-id": 110,
                            "name": "null_value_counts",
                            "type": [
                                "null",
                                {
                                    "logicalType": "map",
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k110_v",
                                        "fields": [
                                            {
                                                "field-id": 121,
                                                "name": "key",
                                                "type": "int",
                                            },
                                            {
                                                "field-id": 122,
                                                "name": "value",
                                                "type": "long",
                                            },
                                        ],
                                    },
                                },
                            ],
                            "default": None,
                        },
                        {
                            "field-id": 137,
                            "name": "nan_value_counts",
                            "type": [
                                "null",
                                {
                                    "logicalType": "map",
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k137_v",
                                        "fields": [
                                            {
                                                "field-id": 138,
                                                "name": "key",
                                                "type": "int",
                                            },
                                            {
                                                "field-id": 139,
                                                "name": "value",
                                                "type": "long",
                                            },
                                        ],
                                    },
                                },
                            ],
                            "default": None,
                        },
                        {
                            "field-id": 125,
                            "name": "lower_bounds",
                            "type": [
                                "null",
                                {
                                    "logicalType": "map",
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k125_v",
                                        "fields": [
                                            {
                                                "field-id": 126,
                                                "name": "key",
                                                "type": "int",
                                            },
                                            {
                                                "field-id": 127,
                                                "name": "value",
                                                "type": "bytes",
                                            },
                                        ],
                                    },
                                },
                            ],
                            "default": None,
                        },
                        {
                            "field-id": 128,
                            "name": "upper_bounds",
                            "type": [
                                "null",
                                {
                                    "logicalType": "map",
                                    "type": "array",
                                    "items": {
                                        "type": "record",
                                        "name": "k128_v",
                                        "fields": [
                                            {
                                                "field-id": 129,
                                                "name": "key",
                                                "type": "int",
                                            },
                                            {
                                                "field-id": 130,
                                                "name": "value",
                                                "type": "bytes",
                                            },
                                        ],
                                    },
                                },
                            ],
                            "default": None,
                        },
                        {
                            "field-id": 131,
                            "name": "key_metadata",
                            "type": ["null", "bytes"],
                            "default": None,
                        },
                        {
                            "field-id": 132,
                            "name": "split_offsets",
                            "type": [
                                "null",
                                {"element-id": 133, "type": "array", "items": "long"},
                            ],
                            "default": None,
                        },
                        {
                            "field-id": 135,
                            "name": "equality_ids",
                            "type": [
                                "null",
                                {"element-id": 136, "type": "array", "items": "int"},
                            ],
                            "default": None,
                        },
                        {
                            "field-id": 140,
                            "name": "sort_order_id",
                            "type": ["null", "int"],
                            "default": None,
                        },
                    ],
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Map pyarrow types to Iceberg type strings
def arrow_to_iceberg_type(arrow_type):
    if pa.types.is_boolean(arrow_type):
        return "boolean"
    if pa.types.is_int32(arrow_type):
        return "int"
    if pa.types.is_int64(arrow_type):
        return "long"
    if pa.types.is_float32(arrow_type):
        return "float"
    if pa.types.is_float64(arrow_type):
        return "double"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_date32(arrow_type):
        return "date"
    if pa.types.is_decimal(arrow_type):
        return f"decimal({arrow_type.precision},{arrow_type.scale})"
    # fallback
    return "string"


def parquet_schema_to_iceberg(pq_schema):
    """Convert a pyarrow schema (from parquet) to an Iceberg schema dict."""
    fields = []
    for i, field in enumerate(pq_schema):
        fields.append(
            {
                "id": i + 1,
                "name": field.name,
                "required": not field.nullable,
                "type": arrow_to_iceberg_type(field.type),
            }
        )
    return {
        "type": "struct",
        "schema-id": 0,
        "fields": fields,
    }


def write_avro(path: Path, schema: dict, records: list) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    parsed = fastavro.parse_schema(schema)
    with open(path, "wb") as f:
        fastavro.writer(f, parsed, records)
    return path.stat().st_size


def make_data_file_entry(content, file_path, record_count, file_size):
    return {
        "content": content,
        "file_path": file_path,
        "file_format": "PARQUET",
        "partition": {},
        "record_count": record_count,
        "file_size_in_bytes": file_size,
        "column_sizes": None,
        "value_counts": None,
        "null_value_counts": None,
        "nan_value_counts": None,
        "lower_bounds": None,
        "upper_bounds": None,
        "key_metadata": None,
        "split_offsets": [4],
        "equality_ids": None,
        "sort_order_id": 0,
    }


def make_manifest_entry(snapshot_id, sequence_number, data_file):
    return {
        "status": 1,
        "snapshot_id": snapshot_id,
        "sequence_number": sequence_number,
        "file_sequence_number": sequence_number,
        "data_file": data_file,
    }


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

TPCH_TABLES = [
    "nation",
    "region",
    "part",
    "supplier",
    "partsupp",
    "customer",
    "orders",
    "lineitem",
]


def wrap_table(
    table_name,
    parquet_files,
    out_dir,
    rel_out,
    snap_id,
    delete_pct=0,
    delete_key_cols=None,
):
    """Wrap parquet file(s) as a single Iceberg table."""

    print(f"  Wrapping {table_name}: {len(parquet_files)} data file(s) ...", end="")

    # Read schema from first parquet file
    pq_file = pq.ParquetFile(str(parquet_files[0]))
    arrow_schema = pq_file.schema_arrow
    iceberg_schema = parquet_schema_to_iceberg(arrow_schema)

    seq_num = 1

    # Data manifest entries
    data_entries = []
    total_rows = 0
    for pf in parquet_files:
        pq_meta = pq.read_metadata(str(pf))
        nrows = pq_meta.num_rows
        fsize = pf.stat().st_size
        # Use relative path from project root
        rel_path = str(pf.resolve().relative_to(Path.cwd()))
        data_entries.append(
            make_manifest_entry(
                snap_id, seq_num, make_data_file_entry(0, rel_path, nrows, fsize)
            )
        )
        total_rows += nrows

    # Write data manifest
    data_mf_path = out_dir / "metadata" / "m0-data.avro"
    data_mf_rel = f"{rel_out}/metadata/m0-data.avro"
    data_mf_size = write_avro(
        data_mf_path, make_manifest_schema(seq_num, snap_id), data_entries
    )

    manifest_list_entries = [
        {
            "manifest_path": data_mf_rel,
            "manifest_length": data_mf_size,
            "partition_spec_id": 0,
            "content": 0,
            "sequence_number": seq_num,
            "min_sequence_number": seq_num,
            "added_snapshot_id": snap_id,
            "added_files_count": len(parquet_files),
            "existing_files_count": 0,
            "deleted_files_count": 0,
            "added_rows_count": total_rows,
            "existing_rows_count": 0,
            "deleted_rows_count": 0,
            "partitions": [],
            "key_metadata": None,
        }
    ]

    # Equality deletes (if requested)
    delete_rows = 0
    if delete_pct > 0 and delete_key_cols:
        print(f" + {delete_pct}% equality deletes on {delete_key_cols} ...", end="")
        # Read a sample of key columns, pick rows to delete
        key_col_indices = []
        for kc in delete_key_cols:
            for i, f in enumerate(arrow_schema):
                if f.name == kc:
                    key_col_indices.append(i)
                    break

        # Read all key columns from all files and sample delete_pct%
        import pyarrow.parquet as pq_mod

        key_tables = []
        for pf in parquet_files:
            t = pq_mod.read_table(str(pf), columns=delete_key_cols)
            key_tables.append(t)
        all_keys = pa.concat_tables(key_tables)

        n_delete = max(1, int(all_keys.num_rows * delete_pct / 100))
        # Deterministic: take every Nth row
        step = max(1, all_keys.num_rows // n_delete)
        indices = list(range(0, all_keys.num_rows, step))[:n_delete]
        delete_table = all_keys.take(indices)
        delete_rows = delete_table.num_rows

        # Add iceberg metadata to schema
        iceberg_del_schema = json.dumps(
            {
                "type": "struct",
                "schema-id": 0,
                "fields": [
                    {
                        "id": i + 1,
                        "name": f.name,
                        "required": False,
                        "type": arrow_to_iceberg_type(f.type),
                    }
                    for i, f in enumerate(arrow_schema)
                    if f.name in delete_key_cols
                ],
            }
        )
        new_fields = []
        for f in delete_table.schema:
            # Find field id in original schema
            for orig_i, orig_f in enumerate(arrow_schema):
                if orig_f.name == f.name:
                    new_fields.append(
                        f.with_metadata({"PARQUET:field_id": str(orig_i + 1)})
                    )
                    break
        delete_arrow_schema = pa.schema(
            new_fields,
            metadata={
                "iceberg.schema": iceberg_del_schema,
                "iceberg.delete.content": "EQUALITY",
            },
        )
        delete_table = delete_table.cast(delete_arrow_schema)

        # Write delete parquet
        del_pq_path = out_dir / "data" / "equality-deletes.parquet"
        del_pq_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(delete_table, str(del_pq_path), compression="snappy")
        del_pq_size = del_pq_path.stat().st_size
        del_pq_rel = f"{rel_out}/data/equality-deletes.parquet"

        # Equality IDs = field IDs of key columns
        eq_ids = [
            i + 1 for i, f in enumerate(arrow_schema) if f.name in delete_key_cols
        ]

        del_entry = make_data_file_entry(2, del_pq_rel, delete_rows, del_pq_size)
        del_entry["equality_ids"] = eq_ids

        del_mf_path = out_dir / "metadata" / "m1-deletes.avro"
        del_mf_rel = f"{rel_out}/metadata/m1-deletes.avro"
        del_mf_size = write_avro(
            del_mf_path,
            make_manifest_schema(seq_num, snap_id),
            [make_manifest_entry(snap_id, seq_num, del_entry)],
        )

        manifest_list_entries.append(
            {
                "manifest_path": del_mf_rel,
                "manifest_length": del_mf_size,
                "partition_spec_id": 0,
                "content": 2,
                "sequence_number": seq_num,
                "min_sequence_number": seq_num,
                "added_snapshot_id": snap_id,
                "added_files_count": 1,
                "existing_files_count": 0,
                "deleted_files_count": 0,
                "added_rows_count": delete_rows,
                "existing_rows_count": 0,
                "deleted_rows_count": 0,
                "partitions": [],
                "key_metadata": None,
            }
        )

    # Manifest list
    snap_avro_path = out_dir / "metadata" / "snap.avro"
    snap_avro_rel = f"{rel_out}/metadata/snap.avro"
    write_avro(snap_avro_path, MANIFEST_LIST_SCHEMA, manifest_list_entries)

    # Build name-mapping so readers can resolve field IDs for parquet files
    # that lack PARQUET:field_id metadata (e.g. files generated by tpchgen-rs).
    name_mapping = json.dumps(
        [{"field-id": f["id"], "names": [f["name"]]} for f in iceberg_schema["fields"]]
    )

    # metadata.json
    metadata = {
        "format-version": 2,
        "table-uuid": f"tpch-{table_name}-00000000-0000-0000",
        "location": rel_out,
        "last-sequence-number": seq_num,
        "last-updated-ms": 1700000000000,
        "last-column-id": len(iceberg_schema["fields"]),
        "current-schema-id": 0,
        "schemas": [iceberg_schema],
        "default-spec-id": 0,
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "last-partition-id": 999,
        "default-sort-order-id": 0,
        "sort-orders": [{"order-id": 0, "fields": []}],
        "properties": {"schema.name-mapping.default": name_mapping},
        "current-snapshot-id": snap_id,
        "refs": {"main": {"snapshot-id": snap_id, "type": "branch"}},
        "snapshots": [
            {
                "sequence-number": seq_num,
                "snapshot-id": snap_id,
                "timestamp-ms": 1700000000000,
                "summary": {"operation": "append"},
                "manifest-list": snap_avro_rel,
                "schema-id": 0,
            }
        ],
        "snapshot-log": [{"timestamp-ms": 1700000000000, "snapshot-id": snap_id}],
        "metadata-log": [],
    }
    meta_path = out_dir / "metadata" / "v1.metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(out_dir / "metadata" / "version-hint.text", "w") as f:
        f.write("1")

    print(f" {total_rows} rows, {delete_rows} deletes")


def main():
    parser = argparse.ArgumentParser(description="Wrap TPC-H parquet as Iceberg")
    parser.add_argument("parquet_dir", help="Input parquet directory")
    parser.add_argument("output_dir", help="Output iceberg directory")
    parser.add_argument(
        "--delete-pct",
        type=int,
        default=0,
        help="Percentage of rows to equality-delete (0=none)",
    )
    parser.add_argument(
        "--delete-tables",
        nargs="+",
        default=[],
        help="Tables to add equality deletes to (default: none)",
    )
    parser.add_argument(
        "--delete-key-cols",
        nargs="+",
        default=None,
        help="Key columns for equality deletes (default: first 2 columns)",
    )
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not parquet_dir.exists():
        print(f"Error: {parquet_dir} does not exist")
        sys.exit(1)

    print(f"Wrapping {parquet_dir} -> {output_dir}")
    print(f"  Delete: {args.delete_pct}% on tables: {args.delete_tables or 'none'}")

    for i, table_name in enumerate(TPCH_TABLES):
        # Find parquet files for this table
        candidates = (
            list(parquet_dir.glob(f"{table_name}.parquet"))
            + list(parquet_dir.glob(f"{table_name}/*.parquet"))
            + list(parquet_dir.glob(f"{table_name}_*.parquet"))
        )
        if not candidates:
            print(f"  Skipping {table_name} (no parquet files found)")
            continue

        table_out = output_dir / table_name
        rel_out = str(table_out.relative_to(Path.cwd()))

        snap_id = 9000000000000000001 + i

        del_pct = args.delete_pct if table_name in args.delete_tables else 0

        # Default key columns: first two columns of the table
        if args.delete_key_cols and table_name in args.delete_tables:
            key_cols = args.delete_key_cols
        elif table_name in args.delete_tables:
            pf = pq.ParquetFile(str(candidates[0]))
            schema = pf.schema_arrow
            key_cols = [schema.field(j).name for j in range(min(2, len(schema)))]
        else:
            key_cols = None

        wrap_table(
            table_name,
            sorted(candidates),
            table_out,
            rel_out,
            snap_id,
            del_pct,
            key_cols,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
