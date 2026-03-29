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

    print("\nDone. Verify with:")
    print("  build/release/duckdb -c \"SET autoload_known_extensions=1; SET autoinstall_known_extensions=1; SELECT * FROM iceberg_scan('substrait/data/iceberg_v1');\"")
    print("  build/release/duckdb -c \"SET autoload_known_extensions=1; SET autoinstall_known_extensions=1; SELECT * FROM iceberg_metadata('substrait/data/iceberg_v2_delete');\"")
    print("  build/release/duckdb -c \"SET autoload_known_extensions=1; SET autoinstall_known_extensions=1; SELECT * FROM iceberg_metadata('substrait/data/iceberg_v2_equality_delete');\"")
