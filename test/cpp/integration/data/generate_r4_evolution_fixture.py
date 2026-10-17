#!/usr/bin/env python3
"""R4: evolved flat fields alongside live positional deletes.

Run with pixi run python test/cpp/integration/data/generate_r4_evolution_fixture.py.
The snapshot retains old physical names/IDs, a retired same-name field, an INT32
and a FLOAT32. The bound schema reorders/renames/promotes them and adds a default.
"""
import json
import shutil

import fastavro
import pyarrow as pa
import pyarrow.parquet as pq

from generate_schema_gate_fixtures import DATA, repoint, write_metadata

SOURCE = "iceberg_v2_delete"
DEST = "iceberg_r4_evolution"
target = DATA / DEST
if target.exists():
    shutil.rmtree(target)
shutil.copytree(DATA / SOURCE, target)

for path in (target / "data").glob("*.parquet"):
    table = pq.read_table(path)
    if "pos" in table.column_names:
        # Delete payload paths must change as well as the manifest paths.
        paths = [
            value.replace(SOURCE, DEST) for value in table["file_path"].to_pylist()
        ]
        table = table.set_column(0, table.schema.field(0), pa.array(paths))
    else:
        schema = pa.schema(
            [
                pa.field("fruit", pa.string(), metadata={b"PARQUET:field_id": b"1"}),
                pa.field("count", pa.int32(), metadata={b"PARQUET:field_id": b"2"}),
                pa.field(
                    "old_score", pa.float32(), metadata={b"PARQUET:field_id": b"4"}
                ),
            ]
        )
        table = pa.Table.from_arrays(
            [
                table["fruit"],
                table["count"].cast(pa.int32()),
                pa.array([1.5, 2.5, None, 4.5, 5.5], type=pa.float32()),
            ],
            schema=schema,
        )
    pq.write_table(table, path, row_group_size=2)

repoint(target, SOURCE, DEST)

meta = target / "metadata/v1.metadata.json"
doc = json.loads(meta.read_text())
doc["format-version"] = 3
doc["next-row-id"] = 5
doc["snapshots"][0]["first-row-id"] = 0
doc["snapshots"][0]["added-rows"] = 5
doc["last-column-id"] = 5
doc["schemas"][0]["fields"] = [
    {"id": 4, "name": "score", "required": False, "type": "double"},
    {"id": 2, "name": "count", "required": False, "type": "long"},
    # Old fruit #1 was dropped; the reused name must NOT recover its values.
    {"id": 3, "name": "fruit", "required": False, "type": "string"},
    {
        "id": 5,
        "name": "added",
        "required": False,
        "type": "long",
        "initial-default": 7,
        "write-default": 9,
    },
]
write_metadata(meta, doc)

# Data rewrites invalidate file sizes, per-column statistics and split offsets.
# Leave optional statistics absent instead of manufacturing inconsistent bounds.
for path in (target / "metadata").glob("*.avro"):
    with path.open("rb") as handle:
        reader = fastavro.reader(handle)
        schema = reader.writer_schema
        records = list(reader)
    for record in records:
        data = record.get("data_file")
        if not data:
            continue
        data["file_size_in_bytes"] = (
            (DATA.parents[3] / data["file_path"]).stat().st_size
        )
        for key in (
            "column_sizes",
            "value_counts",
            "null_value_counts",
            "nan_value_counts",
            "lower_bounds",
            "upper_bounds",
            "split_offsets",
        ):
            if key in data:
                data[key] = None
    with path.open("wb") as handle:
        fastavro.writer(
            handle, schema, records, codec="null", sync_marker=b"r4-evolution-001"
        )

# A valid migration whose absent field comes from identity-partition metadata.
# It must remain refused until the adapter exports per-file partition constants.
source_name = "iceberg_v1"
dest_name = "iceberg_r4_partition_missing"
target = DATA / dest_name
if target.exists():
    shutil.rmtree(target)
shutil.copytree(DATA / source_name, target)
repoint(target, source_name, dest_name)
for path in (target / "data").glob("*.parquet"):
    table = pq.read_table(path).select(["fruit"])
    pq.write_table(table, path)
meta = target / "metadata/v1.metadata.json"
doc = json.loads(meta.read_text())
spec = [{"source-id": 2, "field-id": 1000, "name": "count", "transform": "identity"}]
doc["partition-spec"] = spec
doc["partition-specs"][0]["fields"] = spec
doc["last-partition-id"] = 1000
write_metadata(meta, doc)
for path in (target / "metadata").glob("*.avro"):
    with path.open("rb") as handle:
        reader = fastavro.reader(handle)
        schema = reader.writer_schema
        records = list(reader)
        avro_metadata = dict(reader.metadata)
    if not records or "data_file" not in records[0]:
        for record in records:
            record["partitions"] = [
                {
                    "contains_null": False,
                    "contains_nan": False,
                    "lower_bound": (42).to_bytes(8, "little"),
                    "upper_bound": (42).to_bytes(8, "little"),
                }
            ]
        with path.open("wb") as handle:
            fastavro.writer(handle, schema, records, sync_marker=b"r4-partition-001")
        continue
    data_schema = next(
        field["type"] for field in schema["fields"] if field["name"] == "data_file"
    )
    partition = next(
        field["type"] for field in data_schema["fields"] if field["name"] == "partition"
    )
    partition["fields"] = [
        {"name": "count", "type": ["null", "long"], "default": None, "field-id": 1000}
    ]
    for record in records:
        data = record["data_file"]
        data["partition"] = {"count": 42}
        data["file_size_in_bytes"] = (
            (DATA.parents[3] / data["file_path"]).stat().st_size
        )
        for key in (
            "column_sizes",
            "value_counts",
            "null_value_counts",
            "nan_value_counts",
            "lower_bounds",
            "upper_bounds",
            "split_offsets",
        ):
            if key in data:
                data[key] = None
    avro_metadata.pop("avro.schema", None)
    avro_metadata.pop("avro.codec", None)
    avro_metadata["partition-spec"] = json.dumps(spec)
    with path.open("wb") as handle:
        fastavro.writer(
            handle,
            schema,
            records,
            metadata=avro_metadata,
            sync_marker=b"r4-partition-001",
        )
