"""Offline validation of the exact writer corpus and recorded oracle provenance."""

from datetime import date
import hashlib
import json
from pathlib import Path, PurePosixPath


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_json(value):
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def inventory_digest(files):
    return hashlib.sha256(canonical_json(files).encode()).hexdigest()


def validate_inventory(corpus, manifest):
    files = manifest["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Expected a nonempty warehouse file inventory")
    for relative in files:
        path = PurePosixPath(relative)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or path.parts[0] != "warehouse"
            or str(path) != relative
        ):
            raise ValueError(f"Invalid warehouse inventory path: {relative}")
    warehouse = corpus / "warehouse"
    if warehouse.is_symlink() or not warehouse.is_dir():
        raise ValueError("Warehouse must be a real directory")
    actual = set()
    for path in warehouse.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Warehouse symlinks are not allowed: {path}")
        if path.is_file():
            actual.add(path.relative_to(corpus).as_posix())
        elif not path.is_dir():
            raise ValueError(f"Unexpected warehouse entry: {path}")
    if actual != set(files):
        raise ValueError(
            f"Warehouse inventory differs: extra={sorted(actual - set(files))}, missing={sorted(set(files) - actual)}"
        )
    for relative, expected in files.items():
        if digest(corpus / relative) != expected:
            raise ValueError(f"Changed corpus file: {relative}")


def table_schema(corpus, table):
    # PR1 has fixed schema 0; schema evolution is deliberately out of scope.
    return json.loads((corpus / table["path"] / "schema/schema-0").read_text())


def validate_table_metadata(corpus, manifest):
    for name, table in manifest["tables"].items():
        if table["path"] != f"warehouse/reference.db/{name}":
            raise ValueError(f"Unexpected table path: {name}")
        if table["path"] + "/schema/schema-0" not in manifest["files"]:
            raise ValueError(f"Unlisted table schema: {name}")
        schema = table_schema(corpus, table)
        columns = [
            [
                f["name"],
                f["type"]
                .replace(" NOT NULL", "")
                .replace(", ", ",")
                .replace("STRING", "VARCHAR"),
            ]
            for f in schema["fields"]
        ]
        if table["schema"] != columns or table["options"] != schema["options"]:
            raise ValueError(f"Schema/options differ from persisted metadata: {name}")
        partition_columns = [
            [key, dict(columns)[key]] for key in schema["partitionKeys"]
        ]
        if table["partition_columns"] != partition_columns:
            raise ValueError(f"Partition schema differs: {name}")
        # Verify the summary against recorded live file locations. This does not
        # re-plan Avro manifests or claim to independently prove file liveness.
        locations = set()
        for filename in table["live_files"]:
            matches = [
                PurePosixPath(p).relative_to(table["path"])
                for p in manifest["files"]
                if p.startswith(table["path"] + "/")
                and PurePosixPath(p).name == filename
            ]
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous live file: {name}/{filename}")
            directories = matches[0].parts[
                :-2
            ]  # partition directories before bucket/file
            if len(directories) != len(partition_columns):
                raise ValueError(f"Partition layout differs: {name}/{filename}")
            values = []
            for directory, (key, _) in zip(directories, partition_columns, strict=True):
                if not directory.startswith(key + "="):
                    raise ValueError(f"Partition key differs: {name}/{filename}")
                values.append(directory[len(key) + 1 :])
            locations.add(tuple(values))
        if table["live_partitions"] != [list(v) for v in sorted(locations)]:
            raise ValueError(
                f"Partition values differ from live file locations: {name}"
            )
        for values in table["live_partitions"]:
            if not isinstance(values, list) or len(values) != len(partition_columns):
                raise ValueError(f"Invalid structured partition: {name}")
            # All partitioned fixtures in this corpus use DATE keys.
            for value, (_, typ) in zip(values, partition_columns, strict=True):
                if (
                    typ != "DATE"
                    or not isinstance(value, str)
                    or date.fromisoformat(value).isoformat() != value
                ):
                    raise ValueError(f"Invalid DATE partition: {name}")
        ids = list(table["snapshots"].values())
        if any(type(i) is not int or i <= 0 for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid snapshot identities: {name}")
        for snapshot in ids:
            stored = json.loads(
                (corpus / table["path"] / f"snapshot/snapshot-{snapshot}").read_text()
            )
            if stored["id"] != snapshot or stored["schemaId"] != 0:
                raise ValueError(
                    f"Unexpected snapshot/schema identity: {name}/{snapshot}"
                )


def oracle_identity(case, manifest):
    table = manifest["tables"][case["table"]]
    snapshot = (
        table["snapshots"][case["snapshot"]]
        if case["snapshot"]
        else max(table["snapshots"].values(), default=None)
    )
    prefix = table["path"]
    return {
        "table": case["table"],
        "snapshot_label": case["snapshot"],
        "snapshot_id": snapshot,
        "snapshot_sha256": (
            manifest["files"][f"{prefix}/snapshot/snapshot-{snapshot}"]
            if snapshot is not None
            else None
        ),
        "schema_sha256": manifest["files"][f"{prefix}/schema/schema-0"],
        "columns": case["columns"],
        "sql": case["sql"],
        "warehouse_inventory_sha256": inventory_digest(manifest["files"]),
    }


def validate_oracle(spec, manifest, compare_rows):
    cases = {c["id"]: c for c in spec["cases"] if c["sql"] == "SELECT * FROM {scan}"}
    checks = manifest["independent_checks"]
    ids = [entry["case"] for entry in checks]
    if len(ids) != len(set(ids)) or set(ids) != set(cases):
        raise ValueError("Independent oracle coverage differs from full-row cases")
    for entry in checks:
        case = cases[entry["case"]]
        if entry["passed"] is not True or entry["identity"] != oracle_identity(
            case, manifest
        ):
            raise ValueError(
                f"Independent oracle identity/status differs: {case['id']}"
            )
        names = [c[0] for c in case["columns"]]
        rows = []
        for row in entry["rows"]:
            if set(row) != set(names):
                raise ValueError(f"Independent oracle columns differ: {case['id']}")
            # Canonical JSON sorts object keys; SQL column order is explicit.
            rows.append({name: row[name] for name in names})
        compare_rows(case, rows)
