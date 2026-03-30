#!/usr/bin/env python3
"""
Generate Iceberg test data for Sirius iceberg_scan tests.

Creates three tables under substrait/data/:
  iceberg_v1/  - V2-format table, no delete files (3 rows)
  iceberg_v2_delete/ - V2-format table, positional deletes (5 rows - 2 deleted = 3 rows)
  iceberg_v2_equality_delete/ - V2-format table, equality deletes (5 rows - 2 deleted = 3 rows)

All paths in manifest files are relative to the project root (sirius-dev/),
so tests can be run as: iceberg_scan('substrait/data/iceberg_v1') from there.

Run from the sirius-dev/ project root:
  python3 substrait/data/gen_iceberg_test_data.py
"""

import json
import os
import struct
import uuid
from decimal import Decimal
from pathlib import Path

import fastavro
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Iceberg parquet schema metadata helpers
# ---------------------------------------------------------------------------

ICEBERG_SCHEMA_FRUIT_COUNT = json.dumps({
    "type": "struct",
    "schema-id": 0,
    "fields": [
        {"id": 1, "name": "fruit", "required": False, "type": "string"},
        {"id": 2, "name": "count", "required": False, "type": "long"},
    ],
})

# Arrow schema for the data files (field IDs embedded as metadata)
def make_data_arrow_schema():
    return pa.schema([
        pa.field("fruit", pa.string(), nullable=True,
                 metadata={"PARQUET:field_id": "1"}),
        pa.field("count", pa.int64(), nullable=True,
                 metadata={"PARQUET:field_id": "2"}),
    ], metadata={"iceberg.schema": ICEBERG_SCHEMA_FRUIT_COUNT})


# Arrow schema for positional delete files
ICEBERG_SCHEMA_POSDELETE = json.dumps({
    "type": "struct",
    "schema-id": 0,
    "fields": [
        {"id": 2147483546, "name": "file_path", "required": True, "type": "string"},
        {"id": 2147483545, "name": "pos", "required": True, "type": "long"},
    ],
})

def make_delete_arrow_schema():
    return pa.schema([
        pa.field("file_path", pa.string(), nullable=False,
                 metadata={"PARQUET:field_id": "2147483546"}),
        pa.field("pos", pa.int64(), nullable=False,
                 metadata={"PARQUET:field_id": "2147483545"}),
    ], metadata={
        "iceberg.schema": ICEBERG_SCHEMA_POSDELETE,
        "iceberg.delete.content": "POSITION",
    })


# ---------------------------------------------------------------------------
# Avro schema definitions (reused from existing iceberg table format)
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
            "type": ["null", {
                "element-id": 508,
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "r508",
                    "fields": [
                        {"field-id": 509, "name": "contains_null", "type": "boolean"},
                        {"field-id": 518, "name": "contains_nan", "type": ["null", "boolean"], "default": None},
                        {"field-id": 510, "name": "lower_bound", "type": ["null", "bytes"], "default": None},
                        {"field-id": 511, "name": "upper_bound", "type": ["null", "bytes"], "default": None},
                    ],
                },
            }],
            "default": None,
        },
        {"field-id": 519, "name": "key_metadata", "type": ["null", "bytes"], "default": None},
    ],
}


def make_manifest_schema(sequence_number: int, snapshot_id: int):
    """Manifest entry schema for V2 manifests (data or delete files)."""
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
                            "type": ["null", {
                                "logicalType": "map",
                                "type": "array",
                                "items": {
                                    "type": "record", "name": "k108_v",
                                    "fields": [
                                        {"field-id": 117, "name": "key", "type": "int"},
                                        {"field-id": 118, "name": "value", "type": "long"},
                                    ],
                                },
                            }],
                            "default": None,
                        },
                        {
                            "field-id": 109,
                            "name": "value_counts",
                            "type": ["null", {
                                "logicalType": "map",
                                "type": "array",
                                "items": {
                                    "type": "record", "name": "k109_v",
                                    "fields": [
                                        {"field-id": 119, "name": "key", "type": "int"},
                                        {"field-id": 120, "name": "value", "type": "long"},
                                    ],
                                },
                            }],
                            "default": None,
                        },
                        {
                            "field-id": 110,
                            "name": "null_value_counts",
                            "type": ["null", {
                                "logicalType": "map",
                                "type": "array",
                                "items": {
                                    "type": "record", "name": "k110_v",
                                    "fields": [
                                        {"field-id": 121, "name": "key", "type": "int"},
                                        {"field-id": 122, "name": "value", "type": "long"},
                                    ],
                                },
                            }],
                            "default": None,
                        },
                        {
                            "field-id": 137,
                            "name": "nan_value_counts",
                            "type": ["null", {
                                "logicalType": "map",
                                "type": "array",
                                "items": {
                                    "type": "record", "name": "k137_v",
                                    "fields": [
                                        {"field-id": 138, "name": "key", "type": "int"},
                                        {"field-id": 139, "name": "value", "type": "long"},
                                    ],
                                },
                            }],
                            "default": None,
                        },
                        {
                            "field-id": 125,
                            "name": "lower_bounds",
                            "type": ["null", {
                                "logicalType": "map",
                                "type": "array",
                                "items": {
                                    "type": "record", "name": "k125_v",
                                    "fields": [
                                        {"field-id": 126, "name": "key", "type": "int"},
                                        {"field-id": 127, "name": "value", "type": "bytes"},
                                    ],
                                },
                            }],
                            "default": None,
                        },
                        {
                            "field-id": 128,
                            "name": "upper_bounds",
                            "type": ["null", {
                                "logicalType": "map",
                                "type": "array",
                                "items": {
                                    "type": "record", "name": "k128_v",
                                    "fields": [
                                        {"field-id": 129, "name": "key", "type": "int"},
                                        {"field-id": 130, "name": "value", "type": "bytes"},
                                    ],
                                },
                            }],
                            "default": None,
                        },
                        {"field-id": 131, "name": "key_metadata", "type": ["null", "bytes"], "default": None},
                        {
                            "field-id": 132,
                            "name": "split_offsets",
                            "type": ["null", {"element-id": 133, "type": "array", "items": "long"}],
                            "default": None,
                        },
                        {
                            "field-id": 135,
                            "name": "equality_ids",
                            "type": ["null", {"element-id": 136, "type": "array", "items": "int"}],
                            "default": None,
                        },
                        {"field-id": 140, "name": "sort_order_id", "type": ["null", "int"], "default": None},
                    ],
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_parquet(path: Path, table: pa.Table):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path), compression="snappy")
    return path.stat().st_size


def write_avro(path: Path, schema: dict, records: list) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    parsed = fastavro.parse_schema(schema)
    with open(path, "wb") as f:
        fastavro.writer(f, parsed, records)
    return path.stat().st_size


def long_le(v: int) -> bytes:
    """Encode an int64 as 8-byte little-endian (Iceberg lower/upper bound encoding)."""
    return struct.pack("<q", v)


def str_bytes(s: str) -> bytes:
    return s.encode("utf-8")


def make_data_file_entry(content: int, file_path: str, record_count: int, file_size: int,
                         split_offset: int = 4):
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
        "split_offsets": [split_offset],
        "equality_ids": None,
        "sort_order_id": 0,
    }


def make_manifest_entry(snapshot_id: int, sequence_number: int, data_file: dict):
    return {
        "status": 1,  # ADDED
        "snapshot_id": snapshot_id,
        "sequence_number": sequence_number,
        "file_sequence_number": sequence_number,
        "data_file": data_file,
    }


def write_metadata_json(path: Path, table_location: str, snapshot_id: int,
                        manifest_list_path: str, sequence_number: int = 1):
    """Write a minimal V2 metadata.json with one snapshot."""
    metadata = {
        "format-version": 2,
        "table-uuid": str(uuid.uuid4()),
        "location": table_location,
        "last-sequence-number": sequence_number,
        "last-updated-ms": 1700000000000,
        "last-column-id": 2,
        "current-schema-id": 0,
        "schemas": [{
            "type": "struct",
            "schema-id": 0,
            "fields": [
                {"id": 1, "name": "fruit", "required": False, "type": "string"},
                {"id": 2, "name": "count", "required": False, "type": "long"},
            ],
        }],
        "default-spec-id": 0,
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "last-partition-id": 999,
        "default-sort-order-id": 0,
        "sort-orders": [{"order-id": 0, "fields": []}],
        "current-snapshot-id": snapshot_id,
        "refs": {"main": {"snapshot-id": snapshot_id, "type": "branch"}},
        "snapshots": [{
            "sequence-number": sequence_number,
            "snapshot-id": snapshot_id,
            "timestamp-ms": 1700000000000,
            "summary": {
                "operation": "append",
                "added-data-files": "1",
            },
            "manifest-list": manifest_list_path,
            "schema-id": 0,
        }],
        "snapshot-log": [{"timestamp-ms": 1700000000000, "snapshot-id": snapshot_id}],
        "metadata-log": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Generate iceberg_v1 (V2 format, no deletes, 3 rows)
# ---------------------------------------------------------------------------

def gen_v1(base: Path, rel_base: str):
    print(f"Generating {rel_base} ...")
    snap_id = 1000000000000000001
    snap_uuid = "a1b2c3d4-0001-0001-0001-000000000001"
    manifest_uuid = "a1b2c3d4-0001-0001-0001-000000000002"
    data_file_name = "00000-0-a1b2c3d4-0001-0001-0001-000000000003-00001.parquet"

    # Data parquet
    schema = make_data_arrow_schema()
    table = pa.table(
        {"fruit": ["apple", "banana", "cherry"], "count": [1, 2, 3]},
        schema=schema,
    )
    data_path = base / "data" / data_file_name
    data_size = write_parquet(data_path, table)
    data_rel = f"{rel_base}/data/{data_file_name}"

    # Data manifest
    manifest_path = base / "metadata" / f"{manifest_uuid}-m0.avro"
    manifest_rel = f"{rel_base}/metadata/{manifest_uuid}-m0.avro"
    manifest_schema = make_manifest_schema(1, snap_id)
    manifest_records = [
        make_manifest_entry(snap_id, 1, make_data_file_entry(0, data_rel, 3, data_size))
    ]
    manifest_size = write_avro(manifest_path, manifest_schema, manifest_records)

    # Manifest list
    snap_avro_name = f"snap-{snap_id}-1-{snap_uuid}.avro"
    snap_avro_path = base / "metadata" / snap_avro_name
    snap_avro_rel = f"{rel_base}/metadata/{snap_avro_name}"
    snap_records = [{
        "manifest_path": manifest_rel,
        "manifest_length": manifest_size,
        "partition_spec_id": 0,
        "content": 0,
        "sequence_number": 1,
        "min_sequence_number": 1,
        "added_snapshot_id": snap_id,
        "added_files_count": 1,
        "existing_files_count": 0,
        "deleted_files_count": 0,
        "added_rows_count": 3,
        "existing_rows_count": 0,
        "deleted_rows_count": 0,
        "partitions": [],
        "key_metadata": None,
    }]
    write_avro(snap_avro_path, MANIFEST_LIST_SCHEMA, snap_records)

    # metadata.json + version-hint
    write_metadata_json(
        base / "metadata" / "v1.metadata.json",
        table_location=rel_base,
        snapshot_id=snap_id,
        manifest_list_path=snap_avro_rel,
    )
    with open(base / "metadata" / "version-hint.text", "w") as f:
        f.write("1")

    print(f"  {rel_base}: 3 rows, no deletes -> apple/1, banana/2, cherry/3")


# ---------------------------------------------------------------------------
# Generate iceberg_v2_delete (V2 format, positional deletes, 5 rows -> 3)
# ---------------------------------------------------------------------------

def gen_v2_delete(base: Path, rel_base: str):
    print(f"Generating {rel_base} ...")
    snap_id = 2000000000000000001
    data_uuid = "b2c3d4e5-0002-0002-0002-000000000001"
    delete_uuid = "b2c3d4e5-0002-0002-0002-000000000002"
    data_manifest_uuid = "b2c3d4e5-0002-0002-0002-000000000003"
    delete_manifest_uuid = "b2c3d4e5-0002-0002-0002-000000000004"
    snap_uuid = "b2c3d4e5-0002-0002-0002-000000000005"

    # Data parquet (5 rows: apple/1, banana/2, cherry/3, date/4, elderberry/5)
    data_file_name = f"00000-0-{data_uuid}-00001.parquet"
    data_schema = make_data_arrow_schema()
    data_table = pa.table(
        {
            "fruit": ["apple", "banana", "cherry", "date", "elderberry"],
            "count": [1, 2, 3, 4, 5],
        },
        schema=data_schema,
    )
    data_path = base / "data" / data_file_name
    data_size = write_parquet(data_path, data_table)
    data_rel = f"{rel_base}/data/{data_file_name}"

    # Positional delete file: delete pos 1 (banana) and pos 3 (date)
    delete_file_name = f"delete-00000-0-{delete_uuid}-00001.parquet"
    delete_schema = make_delete_arrow_schema()
    delete_table = pa.table(
        {"file_path": [data_rel, data_rel], "pos": [1, 3]},
        schema=delete_schema,
    )
    delete_path = base / "data" / delete_file_name
    delete_size = write_parquet(delete_path, delete_table)
    delete_rel = f"{rel_base}/data/{delete_file_name}"

    seq_num = 1

    # Data manifest (content=0)
    data_manifest_path = base / "metadata" / f"{data_manifest_uuid}-m0.avro"
    data_manifest_rel = f"{rel_base}/metadata/{data_manifest_uuid}-m0.avro"
    data_manifest_schema = make_manifest_schema(seq_num, snap_id)
    data_manifest_records = [
        make_manifest_entry(snap_id, seq_num, make_data_file_entry(0, data_rel, 5, data_size))
    ]
    data_manifest_size = write_avro(data_manifest_path, data_manifest_schema, data_manifest_records)

    # Delete manifest (content=1)
    delete_manifest_path = base / "metadata" / f"{delete_manifest_uuid}-m1.avro"
    delete_manifest_rel = f"{rel_base}/metadata/{delete_manifest_uuid}-m1.avro"
    delete_manifest_schema = make_manifest_schema(seq_num, snap_id)
    delete_manifest_records = [
        make_manifest_entry(snap_id, seq_num, make_data_file_entry(1, delete_rel, 2, delete_size))
    ]
    delete_manifest_size = write_avro(delete_manifest_path, delete_manifest_schema, delete_manifest_records)

    # Manifest list
    snap_avro_name = f"snap-{snap_id}-1-{snap_uuid}.avro"
    snap_avro_path = base / "metadata" / snap_avro_name
    snap_avro_rel = f"{rel_base}/metadata/{snap_avro_name}"
    snap_records = [
        {
            "manifest_path": data_manifest_rel,
            "manifest_length": data_manifest_size,
            "partition_spec_id": 0,
            "content": 0,
            "sequence_number": seq_num,
            "min_sequence_number": seq_num,
            "added_snapshot_id": snap_id,
            "added_files_count": 1,
            "existing_files_count": 0,
            "deleted_files_count": 0,
            "added_rows_count": 5,
            "existing_rows_count": 0,
            "deleted_rows_count": 0,
            "partitions": [],
            "key_metadata": None,
        },
        {
            "manifest_path": delete_manifest_rel,
            "manifest_length": delete_manifest_size,
            "partition_spec_id": 0,
            "content": 1,
            "sequence_number": seq_num,
            "min_sequence_number": seq_num,
            "added_snapshot_id": snap_id,
            "added_files_count": 1,
            "existing_files_count": 0,
            "deleted_files_count": 0,
            "added_rows_count": 2,
            "existing_rows_count": 0,
            "deleted_rows_count": 0,
            "partitions": [],
            "key_metadata": None,
        },
    ]
    write_avro(snap_avro_path, MANIFEST_LIST_SCHEMA, snap_records)

    # metadata.json + version-hint
    write_metadata_json(
        base / "metadata" / "v1.metadata.json",
        table_location=rel_base,
        snapshot_id=snap_id,
        manifest_list_path=snap_avro_rel,
    )
    with open(base / "metadata" / "version-hint.text", "w") as f:
        f.write("1")

    print(f"  {rel_base}: 5 rows, delete pos 1+3 -> apple/1, cherry/3, elderberry/5")


# ---------------------------------------------------------------------------
# Generate iceberg_v2_equality_delete (V2 format, equality deletes, 5 rows -> 3)
# ---------------------------------------------------------------------------

def make_equality_delete_arrow_schema():
    """Arrow schema for equality-delete files (same columns as data)."""
    return pa.schema([
        pa.field("fruit", pa.string(), nullable=True,
                 metadata={"PARQUET:field_id": "1"}),
        pa.field("count", pa.int64(), nullable=True,
                 metadata={"PARQUET:field_id": "2"}),
    ], metadata={
        "iceberg.schema": ICEBERG_SCHEMA_FRUIT_COUNT,
        "iceberg.delete.content": "EQUALITY",
    })


ICEBERG_SCHEMA_FRUIT_ONLY = json.dumps({
    "type": "struct",
    "schema-id": 0,
    "fields": [
        {"id": 1, "name": "fruit", "required": False, "type": "string"},
    ],
})


def make_equality_delete_single_col_schema():
    """Equality-delete schema keyed on fruit only (field_id=1)."""
    return pa.schema([
        pa.field("fruit", pa.string(), nullable=True,
                 metadata={"PARQUET:field_id": "1"}),
    ], metadata={
        "iceberg.schema": ICEBERG_SCHEMA_FRUIT_ONLY,
        "iceberg.delete.content": "EQUALITY",
    })


def gen_v2_equality_delete(base: Path, rel_base: str):
    print(f"Generating {rel_base} ...")
    snap_id = 3000000000000000001
    data_uuid   = "c3d4e5f6-0003-0003-0003-000000000001"
    delete_uuid = "c3d4e5f6-0003-0003-0003-000000000002"
    data_manifest_uuid   = "c3d4e5f6-0003-0003-0003-000000000003"
    delete_manifest_uuid = "c3d4e5f6-0003-0003-0003-000000000004"
    snap_uuid   = "c3d4e5f6-0003-0003-0003-000000000005"

    # Data parquet (5 rows: apple/1, banana/2, cherry/3, date/4, elderberry/5)
    data_file_name = f"00000-0-{data_uuid}-00001.parquet"
    data_schema = make_data_arrow_schema()
    data_table = pa.table(
        {
            "fruit": ["apple", "banana", "cherry", "date", "elderberry"],
            "count": [1, 2, 3, 4, 5],
        },
        schema=data_schema,
    )
    data_path = base / "data" / data_file_name
    data_size = write_parquet(data_path, data_table)
    data_rel = f"{rel_base}/data/{data_file_name}"

    # Equality delete file: delete rows where (fruit='banana', count=2) and (fruit='date', count=4)
    delete_file_name = f"delete-eq-00000-0-{delete_uuid}-00001.parquet"
    eq_delete_schema = make_equality_delete_arrow_schema()
    eq_delete_table = pa.table(
        {"fruit": ["banana", "date"], "count": [2, 4]},
        schema=eq_delete_schema,
    )
    delete_path = base / "data" / delete_file_name
    delete_size = write_parquet(delete_path, eq_delete_table)
    delete_rel = f"{rel_base}/data/{delete_file_name}"

    seq_num = 1

    # Data manifest (content=0, data files)
    data_manifest_path = base / "metadata" / f"{data_manifest_uuid}-m0.avro"
    data_manifest_rel  = f"{rel_base}/metadata/{data_manifest_uuid}-m0.avro"
    data_manifest_schema  = make_manifest_schema(seq_num, snap_id)
    data_manifest_records = [
        make_manifest_entry(snap_id, seq_num, make_data_file_entry(0, data_rel, 5, data_size))
    ]
    data_manifest_size = write_avro(data_manifest_path, data_manifest_schema, data_manifest_records)

    # Equality-delete manifest (content=2 for EQUALITY_DELETES)
    # equality_ids = [1, 2] (field IDs for fruit and count)
    def make_eq_delete_file_entry(file_path: str, record_count: int, file_size: int):
        entry = make_data_file_entry(2, file_path, record_count, file_size)
        entry["equality_ids"] = [1, 2]
        return entry

    delete_manifest_path = base / "metadata" / f"{delete_manifest_uuid}-m1.avro"
    delete_manifest_rel  = f"{rel_base}/metadata/{delete_manifest_uuid}-m1.avro"
    delete_manifest_schema  = make_manifest_schema(seq_num, snap_id)
    delete_manifest_records = [
        make_manifest_entry(snap_id, seq_num, make_eq_delete_file_entry(delete_rel, 2, delete_size))
    ]
    delete_manifest_size = write_avro(delete_manifest_path, delete_manifest_schema, delete_manifest_records)

    # Manifest list
    snap_avro_name = f"snap-{snap_id}-1-{snap_uuid}.avro"
    snap_avro_path = base / "metadata" / snap_avro_name
    snap_avro_rel  = f"{rel_base}/metadata/{snap_avro_name}"
    snap_records = [
        {
            "manifest_path": data_manifest_rel,
            "manifest_length": data_manifest_size,
            "partition_spec_id": 0,
            "content": 0,
            "sequence_number": seq_num,
            "min_sequence_number": seq_num,
            "added_snapshot_id": snap_id,
            "added_files_count": 1,
            "existing_files_count": 0,
            "deleted_files_count": 0,
            "added_rows_count": 5,
            "existing_rows_count": 0,
            "deleted_rows_count": 0,
            "partitions": [],
            "key_metadata": None,
        },
        {
            "manifest_path": delete_manifest_rel,
            "manifest_length": delete_manifest_size,
            "partition_spec_id": 0,
            "content": 2,  # EQUALITY_DELETES
            "sequence_number": seq_num,
            "min_sequence_number": seq_num,
            "added_snapshot_id": snap_id,
            "added_files_count": 1,
            "existing_files_count": 0,
            "deleted_files_count": 0,
            "added_rows_count": 2,
            "existing_rows_count": 0,
            "deleted_rows_count": 0,
            "partitions": [],
            "key_metadata": None,
        },
    ]
    write_avro(snap_avro_path, MANIFEST_LIST_SCHEMA, snap_records)

    # metadata.json + version-hint
    write_metadata_json(
        base / "metadata" / "v1.metadata.json",
        table_location=rel_base,
        snapshot_id=snap_id,
        manifest_list_path=snap_avro_rel,
    )
    with open(base / "metadata" / "version-hint.text", "w") as f:
        f.write("1")

    print(f"  {rel_base}: 5 rows, equality-delete banana/2 + date/4 -> apple/1, cherry/3, elderberry/5")


# ---------------------------------------------------------------------------
# Generate iceberg_v2_eq_single_col  (single-column equality key on fruit)
# ---------------------------------------------------------------------------

def gen_v2_eq_single_col(base: Path, rel_base: str):
    print(f"Generating {rel_base} ...")
    snap_id          = 4000000000000000001
    data_uuid        = "d4e5f6a7-0004-0004-0004-000000000001"
    delete_uuid      = "d4e5f6a7-0004-0004-0004-000000000002"
    data_mf_uuid     = "d4e5f6a7-0004-0004-0004-000000000003"
    delete_mf_uuid   = "d4e5f6a7-0004-0004-0004-000000000004"
    snap_uuid        = "d4e5f6a7-0004-0004-0004-000000000005"
    seq_num = 1

    # Data: 5 rows
    data_file_name = f"00000-0-{data_uuid}-00001.parquet"
    data_table = pa.table(
        {"fruit": ["apple", "banana", "cherry", "date", "elderberry"],
         "count": [1, 2, 3, 4, 5]},
        schema=make_data_arrow_schema(),
    )
    data_path = base / "data" / data_file_name
    data_size = write_parquet(data_path, data_table)
    data_rel  = f"{rel_base}/data/{data_file_name}"

    # Equality delete: key on fruit only — delete "banana" and "date"
    del_file_name = f"delete-eq-{delete_uuid}-00001.parquet"
    del_table = pa.table({"fruit": ["banana", "date"]},
                         schema=make_equality_delete_single_col_schema())
    del_path = base / "data" / del_file_name
    del_size = write_parquet(del_path, del_table)
    del_rel  = f"{rel_base}/data/{del_file_name}"

    def eq_entry(fp, rc, fs):
        e = make_data_file_entry(2, fp, rc, fs)
        e["equality_ids"] = [1]   # fruit field id only
        return e

    data_mf_path = base / "metadata" / f"{data_mf_uuid}-m0.avro"
    data_mf_rel  = f"{rel_base}/metadata/{data_mf_uuid}-m0.avro"
    data_mf_size = write_avro(data_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, make_data_file_entry(0, data_rel, 5, data_size))])

    del_mf_path = base / "metadata" / f"{delete_mf_uuid}-m1.avro"
    del_mf_rel  = f"{rel_base}/metadata/{delete_mf_uuid}-m1.avro"
    del_mf_size = write_avro(del_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, eq_entry(del_rel, 2, del_size))])

    snap_name = f"snap-{snap_id}-1-{snap_uuid}.avro"
    write_avro(base / "metadata" / snap_name, MANIFEST_LIST_SCHEMA, [
        {"manifest_path": data_mf_rel, "manifest_length": data_mf_size,
         "partition_spec_id": 0, "content": 0, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 5, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
        {"manifest_path": del_mf_rel, "manifest_length": del_mf_size,
         "partition_spec_id": 0, "content": 2, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 2, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
    ])
    write_metadata_json(base / "metadata" / "v1.metadata.json", rel_base, snap_id,
                        f"{rel_base}/metadata/{snap_name}")
    with open(base / "metadata" / "version-hint.text", "w") as f:
        f.write("1")
    print(f"  {rel_base}: 5 rows, single-col eq-delete fruit IN (banana,date) -> apple/1, cherry/3, elderberry/5")


# ---------------------------------------------------------------------------
# Generate iceberg_v2_eq_multi_delete_file  (two separate equality-delete files)
# ---------------------------------------------------------------------------

def gen_v2_eq_multi_delete_file(base: Path, rel_base: str):
    print(f"Generating {rel_base} ...")
    snap_id         = 5000000000000000001
    data_uuid       = "e5f6a7b8-0005-0005-0005-000000000001"
    del1_uuid       = "e5f6a7b8-0005-0005-0005-000000000002"
    del2_uuid       = "e5f6a7b8-0005-0005-0005-000000000003"
    data_mf_uuid    = "e5f6a7b8-0005-0005-0005-000000000004"
    del_mf_uuid     = "e5f6a7b8-0005-0005-0005-000000000005"
    snap_uuid       = "e5f6a7b8-0005-0005-0005-000000000006"
    seq_num = 1

    # Data: 5 rows
    data_file_name = f"00000-0-{data_uuid}-00001.parquet"
    data_table = pa.table(
        {"fruit": ["apple", "banana", "cherry", "date", "elderberry"],
         "count": [1, 2, 3, 4, 5]},
        schema=make_data_arrow_schema(),
    )
    data_path = base / "data" / data_file_name
    data_size = write_parquet(data_path, data_table)
    data_rel  = f"{rel_base}/data/{data_file_name}"

    # Delete file 1: banana/2
    del1_name = f"delete-eq-1-{del1_uuid}-00001.parquet"
    del1_table = pa.table({"fruit": ["banana"], "count": [2]},
                          schema=make_equality_delete_arrow_schema())
    del1_path = base / "data" / del1_name
    del1_size = write_parquet(del1_path, del1_table)
    del1_rel  = f"{rel_base}/data/{del1_name}"

    # Delete file 2: date/4
    del2_name = f"delete-eq-2-{del2_uuid}-00001.parquet"
    del2_table = pa.table({"fruit": ["date"], "count": [4]},
                          schema=make_equality_delete_arrow_schema())
    del2_path = base / "data" / del2_name
    del2_size = write_parquet(del2_path, del2_table)
    del2_rel  = f"{rel_base}/data/{del2_name}"

    def eq_entry(fp, rc, fs):
        e = make_data_file_entry(2, fp, rc, fs)
        e["equality_ids"] = [1, 2]
        return e

    data_mf_path = base / "metadata" / f"{data_mf_uuid}-m0.avro"
    data_mf_rel  = f"{rel_base}/metadata/{data_mf_uuid}-m0.avro"
    data_mf_size = write_avro(data_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, make_data_file_entry(0, data_rel, 5, data_size))])

    # Both delete files in one delete manifest
    del_mf_path = base / "metadata" / f"{del_mf_uuid}-m1.avro"
    del_mf_rel  = f"{rel_base}/metadata/{del_mf_uuid}-m1.avro"
    del_mf_size = write_avro(del_mf_path, make_manifest_schema(seq_num, snap_id), [
        make_manifest_entry(snap_id, seq_num, eq_entry(del1_rel, 1, del1_size)),
        make_manifest_entry(snap_id, seq_num, eq_entry(del2_rel, 1, del2_size)),
    ])

    snap_name = f"snap-{snap_id}-1-{snap_uuid}.avro"
    write_avro(base / "metadata" / snap_name, MANIFEST_LIST_SCHEMA, [
        {"manifest_path": data_mf_rel, "manifest_length": data_mf_size,
         "partition_spec_id": 0, "content": 0, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 5, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
        {"manifest_path": del_mf_rel, "manifest_length": del_mf_size,
         "partition_spec_id": 0, "content": 2, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 2, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 2, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
    ])
    write_metadata_json(base / "metadata" / "v1.metadata.json", rel_base, snap_id,
                        f"{rel_base}/metadata/{snap_name}")
    with open(base / "metadata" / "version-hint.text", "w") as f:
        f.write("1")
    print(f"  {rel_base}: 5 rows, 2 delete files (banana/2, date/4) -> apple/1, cherry/3, elderberry/5")


# ---------------------------------------------------------------------------
# Generate iceberg_v2_eq_all_deleted  (equality deletes remove every row)
# ---------------------------------------------------------------------------

def gen_v2_eq_all_deleted(base: Path, rel_base: str):
    print(f"Generating {rel_base} ...")
    snap_id       = 6000000000000000001
    data_uuid     = "f6a7b8c9-0006-0006-0006-000000000001"
    del_uuid      = "f6a7b8c9-0006-0006-0006-000000000002"
    data_mf_uuid  = "f6a7b8c9-0006-0006-0006-000000000003"
    del_mf_uuid   = "f6a7b8c9-0006-0006-0006-000000000004"
    snap_uuid     = "f6a7b8c9-0006-0006-0006-000000000005"
    seq_num = 1

    # Data: 3 rows
    data_file_name = f"00000-0-{data_uuid}-00001.parquet"
    data_table = pa.table(
        {"fruit": ["apple", "banana", "cherry"], "count": [1, 2, 3]},
        schema=make_data_arrow_schema(),
    )
    data_path = base / "data" / data_file_name
    data_size = write_parquet(data_path, data_table)
    data_rel  = f"{rel_base}/data/{data_file_name}"

    # Delete all 3 rows
    del_file_name = f"delete-eq-{del_uuid}-00001.parquet"
    del_table = pa.table(
        {"fruit": ["apple", "banana", "cherry"], "count": [1, 2, 3]},
        schema=make_equality_delete_arrow_schema(),
    )
    del_path = base / "data" / del_file_name
    del_size = write_parquet(del_path, del_table)
    del_rel  = f"{rel_base}/data/{del_file_name}"

    def eq_entry(fp, rc, fs):
        e = make_data_file_entry(2, fp, rc, fs)
        e["equality_ids"] = [1, 2]
        return e

    data_mf_path = base / "metadata" / f"{data_mf_uuid}-m0.avro"
    data_mf_rel  = f"{rel_base}/metadata/{data_mf_uuid}-m0.avro"
    data_mf_size = write_avro(data_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, make_data_file_entry(0, data_rel, 3, data_size))])

    del_mf_path = base / "metadata" / f"{del_mf_uuid}-m1.avro"
    del_mf_rel  = f"{rel_base}/metadata/{del_mf_uuid}-m1.avro"
    del_mf_size = write_avro(del_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, eq_entry(del_rel, 3, del_size))])

    snap_name = f"snap-{snap_id}-1-{snap_uuid}.avro"
    write_avro(base / "metadata" / snap_name, MANIFEST_LIST_SCHEMA, [
        {"manifest_path": data_mf_rel, "manifest_length": data_mf_size,
         "partition_spec_id": 0, "content": 0, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 3, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
        {"manifest_path": del_mf_rel, "manifest_length": del_mf_size,
         "partition_spec_id": 0, "content": 2, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 3, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
    ])
    write_metadata_json(base / "metadata" / "v1.metadata.json", rel_base, snap_id,
                        f"{rel_base}/metadata/{snap_name}")
    with open(base / "metadata" / "version-hint.text", "w") as f:
        f.write("1")
    print(f"  {rel_base}: 3 rows, all deleted -> 0 rows")


# ---------------------------------------------------------------------------
# Generate iceberg_v2_eq_pos_combined  (equality + positional deletes together)
# ---------------------------------------------------------------------------

def gen_v2_eq_pos_combined(base: Path, rel_base: str):
    print(f"Generating {rel_base} ...")
    snap_id        = 7000000000000000001
    data_uuid      = "a7b8c9d0-0007-0007-0007-000000000001"
    pos_del_uuid   = "a7b8c9d0-0007-0007-0007-000000000002"
    eq_del_uuid    = "a7b8c9d0-0007-0007-0007-000000000003"
    data_mf_uuid   = "a7b8c9d0-0007-0007-0007-000000000004"
    pos_mf_uuid    = "a7b8c9d0-0007-0007-0007-000000000005"
    eq_mf_uuid     = "a7b8c9d0-0007-0007-0007-000000000006"
    snap_uuid      = "a7b8c9d0-0007-0007-0007-000000000007"
    seq_num = 1

    # Data: 5 rows
    data_file_name = f"00000-0-{data_uuid}-00001.parquet"
    data_table = pa.table(
        {"fruit": ["apple", "banana", "cherry", "date", "elderberry"],
         "count": [1, 2, 3, 4, 5]},
        schema=make_data_arrow_schema(),
    )
    data_path = base / "data" / data_file_name
    data_size = write_parquet(data_path, data_table)
    data_rel  = f"{rel_base}/data/{data_file_name}"

    # Positional delete: row 0 (apple/1)
    pos_del_name = f"delete-pos-{pos_del_uuid}-00001.parquet"
    pos_del_table = pa.table(
        {"file_path": [data_rel], "pos": [0]},
        schema=make_delete_arrow_schema(),
    )
    pos_del_path = base / "data" / pos_del_name
    pos_del_size = write_parquet(pos_del_path, pos_del_table)
    pos_del_rel  = f"{rel_base}/data/{pos_del_name}"

    # Equality delete: banana/2
    eq_del_name = f"delete-eq-{eq_del_uuid}-00001.parquet"
    eq_del_table = pa.table(
        {"fruit": ["banana"], "count": [2]},
        schema=make_equality_delete_arrow_schema(),
    )
    eq_del_path = base / "data" / eq_del_name
    eq_del_size = write_parquet(eq_del_path, eq_del_table)
    eq_del_rel  = f"{rel_base}/data/{eq_del_name}"

    def eq_entry(fp, rc, fs):
        e = make_data_file_entry(2, fp, rc, fs)
        e["equality_ids"] = [1, 2]
        return e

    data_mf_path = base / "metadata" / f"{data_mf_uuid}-m0.avro"
    data_mf_rel  = f"{rel_base}/metadata/{data_mf_uuid}-m0.avro"
    data_mf_size = write_avro(data_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, make_data_file_entry(0, data_rel, 5, data_size))])

    pos_mf_path = base / "metadata" / f"{pos_mf_uuid}-m1.avro"
    pos_mf_rel  = f"{rel_base}/metadata/{pos_mf_uuid}-m1.avro"
    pos_mf_size = write_avro(pos_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, make_data_file_entry(1, pos_del_rel, 1, pos_del_size))])

    eq_mf_path = base / "metadata" / f"{eq_mf_uuid}-m2.avro"
    eq_mf_rel  = f"{rel_base}/metadata/{eq_mf_uuid}-m2.avro"
    eq_mf_size = write_avro(eq_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, eq_entry(eq_del_rel, 1, eq_del_size))])

    snap_name = f"snap-{snap_id}-1-{snap_uuid}.avro"
    write_avro(base / "metadata" / snap_name, MANIFEST_LIST_SCHEMA, [
        {"manifest_path": data_mf_rel, "manifest_length": data_mf_size,
         "partition_spec_id": 0, "content": 0, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 5, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
        {"manifest_path": pos_mf_rel, "manifest_length": pos_mf_size,
         "partition_spec_id": 0, "content": 1, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 1, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
        {"manifest_path": eq_mf_rel, "manifest_length": eq_mf_size,
         "partition_spec_id": 0, "content": 2, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 1, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
    ])
    write_metadata_json(base / "metadata" / "v1.metadata.json", rel_base, snap_id,
                        f"{rel_base}/metadata/{snap_name}")
    with open(base / "metadata" / "version-hint.text", "w") as f:
        f.write("1")
    print(f"  {rel_base}: 5 rows, pos-del row0 + eq-del banana/2 -> cherry/3, date/4, elderberry/5")


# ---------------------------------------------------------------------------
# Generate iceberg_v2_eq_multi_data_file  (two data files, one delete file)
# ---------------------------------------------------------------------------

def gen_v2_eq_multi_data_file(base: Path, rel_base: str):
    print(f"Generating {rel_base} ...")
    snap_id       = 8000000000000000001
    data1_uuid    = "b8c9d0e1-0008-0008-0008-000000000001"
    data2_uuid    = "b8c9d0e1-0008-0008-0008-000000000002"
    del_uuid      = "b8c9d0e1-0008-0008-0008-000000000003"
    data_mf_uuid  = "b8c9d0e1-0008-0008-0008-000000000004"
    del_mf_uuid   = "b8c9d0e1-0008-0008-0008-000000000005"
    snap_uuid     = "b8c9d0e1-0008-0008-0008-000000000006"
    seq_num = 1

    # Data file 1: apple/1, banana/2, cherry/3
    data1_name = f"00000-0-{data1_uuid}-00001.parquet"
    data1_table = pa.table(
        {"fruit": ["apple", "banana", "cherry"], "count": [1, 2, 3]},
        schema=make_data_arrow_schema(),
    )
    data1_path = base / "data" / data1_name
    data1_size = write_parquet(data1_path, data1_table)
    data1_rel  = f"{rel_base}/data/{data1_name}"

    # Data file 2: date/4, elderberry/5, fig/6
    data2_name = f"00001-0-{data2_uuid}-00001.parquet"
    data2_table = pa.table(
        {"fruit": ["date", "elderberry", "fig"], "count": [4, 5, 6]},
        schema=make_data_arrow_schema(),
    )
    data2_path = base / "data" / data2_name
    data2_size = write_parquet(data2_path, data2_table)
    data2_rel  = f"{rel_base}/data/{data2_name}"

    # Equality delete: banana/2 (from file1) and elderberry/5 (from file2)
    del_file_name = f"delete-eq-{del_uuid}-00001.parquet"
    del_table = pa.table(
        {"fruit": ["banana", "elderberry"], "count": [2, 5]},
        schema=make_equality_delete_arrow_schema(),
    )
    del_path = base / "data" / del_file_name
    del_size = write_parquet(del_path, del_table)
    del_rel  = f"{rel_base}/data/{del_file_name}"

    def eq_entry(fp, rc, fs):
        e = make_data_file_entry(2, fp, rc, fs)
        e["equality_ids"] = [1, 2]
        return e

    # Data manifest: both data files
    data_mf_path = base / "metadata" / f"{data_mf_uuid}-m0.avro"
    data_mf_rel  = f"{rel_base}/metadata/{data_mf_uuid}-m0.avro"
    data_mf_size = write_avro(data_mf_path, make_manifest_schema(seq_num, snap_id), [
        make_manifest_entry(snap_id, seq_num, make_data_file_entry(0, data1_rel, 3, data1_size)),
        make_manifest_entry(snap_id, seq_num, make_data_file_entry(0, data2_rel, 3, data2_size)),
    ])

    del_mf_path = base / "metadata" / f"{del_mf_uuid}-m1.avro"
    del_mf_rel  = f"{rel_base}/metadata/{del_mf_uuid}-m1.avro"
    del_mf_size = write_avro(del_mf_path, make_manifest_schema(seq_num, snap_id),
        [make_manifest_entry(snap_id, seq_num, eq_entry(del_rel, 2, del_size))])

    snap_name = f"snap-{snap_id}-1-{snap_uuid}.avro"
    write_avro(base / "metadata" / snap_name, MANIFEST_LIST_SCHEMA, [
        {"manifest_path": data_mf_rel, "manifest_length": data_mf_size,
         "partition_spec_id": 0, "content": 0, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 2, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 6, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
        {"manifest_path": del_mf_rel, "manifest_length": del_mf_size,
         "partition_spec_id": 0, "content": 2, "sequence_number": seq_num,
         "min_sequence_number": seq_num, "added_snapshot_id": snap_id,
         "added_files_count": 1, "existing_files_count": 0, "deleted_files_count": 0,
         "added_rows_count": 2, "existing_rows_count": 0, "deleted_rows_count": 0,
         "partitions": [], "key_metadata": None},
    ])
    write_metadata_json(base / "metadata" / "v1.metadata.json", rel_base, snap_id,
                        f"{rel_base}/metadata/{snap_name}")
    with open(base / "metadata" / "version-hint.text", "w") as f:
        f.write("1")
    print(f"  {rel_base}: 6 rows across 2 files, eq-delete banana/2+elderberry/5 -> apple/1, cherry/3, date/4, fig/6")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    gen_v1(
        base=project_root / "substrait/data/iceberg_v1",
        rel_base="substrait/data/iceberg_v1",
    )
    gen_v2_delete(
        base=project_root / "substrait/data/iceberg_v2_delete",
        rel_base="substrait/data/iceberg_v2_delete",
    )
    gen_v2_equality_delete(
        base=project_root / "substrait/data/iceberg_v2_equality_delete",
        rel_base="substrait/data/iceberg_v2_equality_delete",
    )
    gen_v2_eq_single_col(
        base=project_root / "substrait/data/iceberg_v2_eq_single_col",
        rel_base="substrait/data/iceberg_v2_eq_single_col",
    )
    gen_v2_eq_multi_delete_file(
        base=project_root / "substrait/data/iceberg_v2_eq_multi_delete_file",
        rel_base="substrait/data/iceberg_v2_eq_multi_delete_file",
    )
    gen_v2_eq_all_deleted(
        base=project_root / "substrait/data/iceberg_v2_eq_all_deleted",
        rel_base="substrait/data/iceberg_v2_eq_all_deleted",
    )
    gen_v2_eq_pos_combined(
        base=project_root / "substrait/data/iceberg_v2_eq_pos_combined",
        rel_base="substrait/data/iceberg_v2_eq_pos_combined",
    )
    gen_v2_eq_multi_data_file(
        base=project_root / "substrait/data/iceberg_v2_eq_multi_data_file",
        rel_base="substrait/data/iceberg_v2_eq_multi_data_file",
    )

    print("\nDone. Verify with:")
    print("  build/release/duckdb -c \"SET autoload_known_extensions=1; SET autoinstall_known_extensions=1; SELECT * FROM iceberg_scan('substrait/data/iceberg_v1');\"")
    print("  build/release/duckdb -c \"SET autoload_known_extensions=1; SET autoinstall_known_extensions=1; SELECT * FROM iceberg_metadata('substrait/data/iceberg_v2_delete');\"")
    print("  build/release/duckdb -c \"SET autoload_known_extensions=1; SET autoinstall_known_extensions=1; SELECT * FROM iceberg_metadata('substrait/data/iceberg_v2_equality_delete');\"")
