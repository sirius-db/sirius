#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
"""Relocate a committed Iceberg table fixture to a remote root (requires fastavro).

Usage: generate_remote_table_fixture.py <source table dir> <out dir> <remote root>
Writes the table under <out dir> with every path rewritten to <remote root>, plus
metadata/empty.metadata.json: the same table before its first snapshot, and
metadata/detached.metadata.json: the same table with its snapshots retained and none of them
current, so they are readable by version only. A table with deletes also gets
metadata/replaced.metadata.json: another table for the same name, whose snapshot of the same
ID lists the data manifests only.
"""
import copy
import json
from pathlib import Path
import shutil
import sys

import fastavro

source, out, remote = (
    Path(sys.argv[1]),
    Path(sys.argv[2]).resolve(),
    sys.argv[3].rstrip("/"),
)
meta = json.loads((source / "metadata/v1.metadata.json").read_text())
old = meta["location"]


def relocate(value):
    if isinstance(value, str):
        return (
            remote + value[len(old) :]
            if value.startswith(old + "/") or value == old
            else value
        )
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, dict):
        return {key: relocate(item) for key, item in value.items()}
    return value


def read_avro(path):
    with path.open("rb") as f:
        reader = fastavro.reader(f)
        metadata = dict(reader.metadata)
        metadata.pop("avro.schema", None)
        return reader.writer_schema, metadata, list(reader)


def write_avro(path, schema, metadata, rows):
    with path.open("wb") as f:
        fastavro.writer(f, schema, rows, metadata=metadata)


shutil.copytree(source, out, dirs_exist_ok=True)
# Data manifests first: a manifest list records the rewritten manifest's length.
for path in sorted(
    (out / "metadata").glob("*.avro"), key=lambda p: p.name.startswith("snap-")
):
    schema, metadata, rows = read_avro(path)
    rows = relocate(rows)
    for row in rows:
        if "manifest_path" in row:
            name = Path(row["manifest_path"]).name
            row["manifest_length"] = (out / "metadata" / name).stat().st_size
    write_avro(path, schema, metadata, rows)
meta = relocate(meta)
(out / "metadata/v1.metadata.json").write_text(json.dumps(meta))
for snapshot in meta.get("snapshots", []):
    if snapshot["snapshot-id"] != meta.get("current-snapshot-id"):
        continue
    listing = out / "metadata" / Path(snapshot["manifest-list"]).name
    schema, metadata, rows = read_avro(listing)
    data_only = [row for row in rows if row.get("content", 0) == 0]
    if len(data_only) == len(rows):
        continue
    write_avro(out / "metadata/snap-replaced.avro", schema, metadata, data_only)
    replaced = copy.deepcopy(meta)
    replaced["table-uuid"] = "00000000-0000-4000-8000-0000000000ff"
    for other in replaced["snapshots"]:
        if other["snapshot-id"] == snapshot["snapshot-id"]:
            other["manifest-list"] = remote + "/metadata/snap-replaced.avro"
    (out / "metadata/replaced.metadata.json").write_text(json.dumps(replaced))
detached = copy.deepcopy(meta)
for key in ("current-snapshot-id", "refs"):
    detached.pop(key, None)
(out / "metadata/detached.metadata.json").write_text(json.dumps(detached))
for key in ("current-snapshot-id", "snapshots", "snapshot-log", "refs"):
    meta.pop(key, None)
(out / "metadata/empty.metadata.json").write_text(json.dumps(meta))
