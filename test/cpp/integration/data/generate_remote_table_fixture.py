#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. Licensed under the Apache License, Version 2.0.
"""Relocate a committed Iceberg table fixture to a remote root (requires fastavro).

Usage: generate_remote_table_fixture.py <source table dir> <out dir> <remote root>
Writes the table under <out dir> with every path rewritten to <remote root>, plus
metadata/empty.metadata.json: the same table before its first snapshot.
"""
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


shutil.copytree(source, out, dirs_exist_ok=True)
# Data manifests first: a manifest list records the rewritten manifest's length.
for path in sorted(
    (out / "metadata").glob("*.avro"), key=lambda p: p.name.startswith("snap-")
):
    with path.open("rb") as f:
        reader = fastavro.reader(f)
        schema, metadata, rows = (
            reader.writer_schema,
            dict(reader.metadata),
            list(reader),
        )
    rows = relocate(rows)
    for row in rows:
        if "manifest_path" in row:
            name = Path(row["manifest_path"]).name
            row["manifest_length"] = (out / "metadata" / name).stat().st_size
    metadata.pop("avro.schema", None)
    with path.open("wb") as f:
        fastavro.writer(f, schema, rows, metadata=metadata)
meta = relocate(meta)
(out / "metadata/v1.metadata.json").write_text(json.dumps(meta))
for key in ("current-snapshot-id", "snapshots", "snapshot-log", "refs"):
    meta.pop(key, None)
(out / "metadata/empty.metadata.json").write_text(json.dumps(meta))
