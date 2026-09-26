#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
"""Generate Iceberg inventory edge cases in a supplied scratch directory (requires fastavro and pyarrow)."""
import copy
import json
from pathlib import Path
import sys
import shutil

import fastavro
import pyarrow.parquet as pq

repo = Path.cwd()
source = repo / "test/cpp/integration/data/iceberg_v3_deletion_vector"
out = Path(sys.argv[1]).resolve()


def read(path):
    with path.open("rb") as f:
        reader = fastavro.reader(f)
        return reader.writer_schema, dict(reader.metadata), list(reader)


def write(path, schema, metadata, rows):
    metadata.pop("avro.schema", None)
    with path.open("wb") as f:
        fastavro.writer(f, schema, rows, metadata=metadata)


for mode in ("empty", "equality_bad_dv"):
    dest = out / mode
    (dest / "metadata").mkdir(parents=True, exist_ok=True)
    meta = json.loads((source / "metadata/v1.metadata.json").read_text())
    meta["location"] = str(dest)
    for snapshot in meta["snapshots"]:
        snapshot["manifest-list"] = str(
            dest / "metadata" / Path(snapshot["manifest-list"]).name
        )
    (dest / "metadata/v1.metadata.json").write_text(json.dumps(meta, indent=2))
    (dest / "metadata/version-hint.text").write_text("1")
    for path in (source / "metadata").glob("*.avro"):
        schema, metadata, rows = read(path)
        if "manifest_path" in rows[0]:
            for row in rows:
                row["manifest_path"] = str(
                    dest / "metadata" / Path(row["manifest_path"]).name
                )
                if mode == "empty":
                    row["added_files_count"] = row["existing_files_count"] = 0
                    row["deleted_files_count"] = 1
                    row["deleted_rows_count"] = row["added_rows_count"]
                    row["added_rows_count"] = row["existing_rows_count"] = 0
                elif row["content"] == 1:
                    row["added_files_count"] = 2
        else:
            for row in rows:
                if mode == "empty":
                    row["status"] = 2
                elif row["data_file"]["content"] == 1:
                    row["data_file"]["content_offset"] = -1
                    row["data_file"]["content_size_in_bytes"] = -1
            if mode != "empty" and rows[0]["data_file"]["content"] == 1:
                eq = copy.deepcopy(rows[0])
                eq["data_file"].update(
                    content=2,
                    file_format="PARQUET",
                    equality_ids=[1, 2],
                    file_path=str(
                        repo
                        / "test/cpp/integration/data/iceberg_v2_equality_delete/data/delete-eq-00000-0-c3d4e5f6-0003-0003-0003-000000000002-00001.parquet"
                    ),
                )
                rows.append(eq)
        write(dest / "metadata" / path.name, schema, metadata, rows)
    for path in (dest / "metadata").glob("snap-*.avro"):
        schema, metadata, rows = read(path)
        for row in rows:
            row["manifest_length"] = Path(row["manifest_path"]).stat().st_size
        write(path, schema, metadata, rows)
# One manifest fixes the data-file order for the late schema-failure rendezvous.
# Copy the data so metadata/cache state from the original conformance cases cannot mask decode.
rename = (
    repo / "test/cpp/integration/data/iceberg_conformance/rename_col/conf/rename_col"
)
dest = out / "schema_later"
(dest / "metadata").mkdir(parents=True, exist_ok=True)
(dest / "data").mkdir(parents=True, exist_ok=True)
meta = json.loads(sorted((rename / "metadata").glob("*.metadata.json"))[-1].read_text())
snapshot = next(
    s for s in meta["snapshots"] if s["snapshot-id"] == meta["current-snapshot-id"]
)
list_schema, list_metadata, manifest_rows = read(repo / snapshot["manifest-list"])
entries = []
for marker, target in (("571c62d9", "a_good.parquet"), ("1f276fa2", "z_bad.parquet")):
    manifest = next(
        p for p in (rename / "metadata").glob("*-m0.avro") if marker in p.name
    )
    schema, metadata, rows = read(manifest)
    row = rows[0]
    shutil.copyfile(repo / row["data_file"]["file_path"], dest / "data" / target)
    if target == "a_good.parquet":
        # Two row groups force a split to be emitted before the next file's footer is read.
        data = pq.read_table(dest / "data" / target)
        pq.write_table(
            data, dest / "data" / target, row_group_size=1, compression="snappy"
        )
    row["data_file"]["file_size_in_bytes"] = (dest / "data" / target).stat().st_size
    row["data_file"]["file_path"] = str(dest / "data" / target)
    row["snapshot_id"] = snapshot["snapshot-id"]
    row["sequence_number"] = snapshot["sequence-number"]
    row["status"] = 1
    entries.append(row)
manifest = dest / "metadata" / "schema.avro"
write(manifest, schema, metadata, entries)
manifest_row = manifest_rows[0]
manifest_row.update(
    manifest_path=str(manifest),
    manifest_length=manifest.stat().st_size,
    added_files_count=2,
    existing_files_count=0,
    deleted_files_count=0,
    added_rows_count=4,
    existing_rows_count=0,
    deleted_rows_count=0,
)
manifest_list = dest / "metadata" / "snap-schema.avro"
write(manifest_list, list_schema, list_metadata, [manifest_row])
snapshot["manifest-list"] = str(manifest_list)
meta["snapshots"] = [snapshot]
meta["snapshot-log"] = [
    {"snapshot-id": snapshot["snapshot-id"], "timestamp-ms": snapshot["timestamp-ms"]}
]
meta["metadata-log"] = []
meta["location"] = str(dest)
(dest / "metadata" / "v1.metadata.json").write_text(json.dumps(meta, indent=2))
(dest / "metadata" / "version-hint.text").write_text("1")

print(out)
