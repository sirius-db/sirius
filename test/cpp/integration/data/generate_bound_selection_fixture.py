#!/usr/bin/env python3
"""Create/append a tiny local Iceberg table and publish immutable metadata via a hint.

Uses the already provisioned pyiceberg/pyarrow/sqlalchemy environment; never installs
packages or connects to a service. The C++ caller owns a fresh temporary directory.
"""

import argparse
import json
from pathlib import Path
import shutil

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import IntegerType, NestedField


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "append", "append_expire"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    catalog_file = root / "catalog.sqlite"
    if args.action == "create" and catalog_file.exists():
        raise RuntimeError("create requires a fresh fixture directory")
    if args.action != "create" and not catalog_file.exists():
        raise RuntimeError("append requires the original fixture catalog")
    catalog = SqlCatalog(
        "bound_selection",
        uri=f"sqlite:///{catalog_file}",
        warehouse=str(root / "warehouse"),
    )
    if args.action == "create":
        catalog.create_namespace("ns")
        table = catalog.create_table(
            "ns.t", schema=Schema(NestedField(1, "id", IntegerType(), required=False))
        )
        ids, version, expected = [1, 2, 3], 1, [1, 2, 3]
    else:
        table = catalog.load_table("ns.t")
        ids, version, expected = [4, 5], 2, [1, 2, 3, 4, 5]
    table.append(pa.table({"id": pa.array(ids, type=pa.int32())}))
    rows = sorted(table.scan().to_arrow().column("id").to_pylist())
    if rows != expected:
        raise RuntimeError(f"pyiceberg oracle differs: {rows} != {expected}")

    # All manifest/data paths remain absolute, pointing into this fixture's warehouse.
    # The published JSON files never change, including after the next snapshot is appended.
    metadata = root / "table" / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    target = metadata / f"v{version}.metadata.json"
    if target.exists():
        raise RuntimeError(f"refusing to overwrite selected metadata: {target}")
    if args.action == "append_expire":
        # Expire the old snapshot from the newly published metadata only. The bound
        # v1 JSON and its files stay readable; resolving the directory with the old
        # snapshot ID must now fail, unlike querying the captured v1 metadata URI.
        contents = json.loads(Path(table.metadata_location).read_text())
        selected_id = table.current_snapshot().snapshot_id
        contents["snapshots"] = [
            snapshot
            for snapshot in contents["snapshots"]
            if snapshot["snapshot-id"] == selected_id
        ]
        contents["snapshot-log"] = [
            entry
            for entry in contents["snapshot-log"]
            if entry["snapshot-id"] == selected_id
        ]
        target.write_text(json.dumps(contents) + "\n")
    else:
        shutil.copyfile(table.metadata_location, target)
    (root / f"oracle_v{version}.json").write_text(
        json.dumps(
            {
                "snapshot_id": table.current_snapshot().snapshot_id,
                "metadata_file": str(target),
                "rows": rows,
            }
        )
        + "\n"
    )
    pending = metadata / "version-hint.next"
    # The qualified reader treats the hint as an exact version string, without trimming.
    pending.write_text(str(version))
    pending.replace(metadata / "version-hint.text")


if __name__ == "__main__":
    main()
