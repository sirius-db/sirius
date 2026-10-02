#!/usr/bin/env python3
"""Explicitly regenerate the small corpus through real Paimon write/commit APIs."""

import argparse
from datetime import date
from decimal import Decimal
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess

import pyarrow as pa
from pypaimon import CatalogFactory, Schema

from run_conformance import compare_rows
from corpus_checks import (
    canonical_json,
    digest,
    oracle_identity,
    table_schema,
    validate_inventory,
    validate_table_metadata,
    validate_oracle,
)

HERE = Path(__file__).resolve().parent
OPTIONS = {
    "file.format": "parquet",
    "manifest.format": "avro",
    "write-only": "true",
    "snapshot.num-retained.min": "100",
    "snapshot.num-retained.max": "100",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="New directory; never overwrite an existing corpus"
    )
    parser.add_argument("--delete-writer", type=Path, required=True)
    args = parser.parse_args()
    if version("pypaimon") != "2.0.0" or pa.__version__ != "19.0.1":
        parser.error("Regeneration requires pypaimon==2.0.0 and pyarrow==19.0.1")
    writer = args.delete_writer.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "generator-input"
    inputs.mkdir()
    spec = json.loads((HERE / "expectations.json").read_text())
    warehouse = output / "warehouse"
    catalog = CatalogFactory.create({"warehouse": str(warehouse)})
    catalog.create_database("reference", False)
    manifest = {
        "format_version": 2,
        "generator": {
            "pypaimon": version("pypaimon"),
            "pyarrow": pa.__version__,
            "delete_writer_sha256": digest(writer),
            "native_commit": "53f9c86d45aabb0a6f1a379271da07d7a9f27a3d",
        },
        "tables": {},
    }
    types = {
        "BIGINT": pa.int64(),
        "DECIMAL(12,2)": pa.decimal128(12, 2),
        "VARCHAR": pa.string(),
        "DATE": pa.date32(),
    }
    for name, recipe in spec["recipes"].items():
        schema = pa.schema(
            [pa.field(n, types[t], nullable=n != "id") for n, t in recipe["schema"]]
        )

        def batch(rows):
            columns = []
            for i, (_, typ) in enumerate(recipe["schema"]):
                values = [r[i] for r in rows]
                if typ.startswith("DECIMAL"):
                    values = [None if v is None else Decimal(v) for v in values]
                if typ == "DATE":
                    values = [
                        None if v is None else date.fromisoformat(v) for v in values
                    ]
                columns.append(pa.array(values, type=schema[i].type))
            return pa.record_batch(columns, schema=schema)

        def native(command, rows, label, kind="insert"):
            ipc = inputs / f"{name}-{label}.arrow"
            with pa.OSFile(str(ipc), "wb") as sink:
                with pa.ipc.new_file(sink, schema) as stream:
                    stream.write_batch(batch(rows))
            argv = [str(writer), command, str(warehouse), name, str(ipc)]
            if command == "write":
                argv += [kind, label]
            result = subprocess.run(
                argv, check=False, capture_output=True, text=True, timeout=90
            )
            (inputs / f"{name}-{label}.stderr").write_text(result.stderr)
            (inputs / f"{name}-{label}.stdout").write_text(result.stdout)
            result.check_returncode()
            return json.loads(result.stdout)

        options = {**OPTIONS, "bucket": "1" if recipe["mode"] == "pk" else "-1"}
        if recipe["mode"] == "pk":
            options["merge-engine"] = "deduplicate"
        if recipe.get("partitioned"):
            # PyPaimon writes DATE paths as ISO dates, not legacy epoch-day integers.
            options["partition.legacy-name"] = "false"
        if name == "orders_pk":
            native("create", [], "schema")
        else:
            definition = Schema.from_pyarrow_schema(
                schema,
                partition_keys=["day"] if recipe.get("partitioned") else [],
                primary_keys=["id"] if recipe["mode"] == "pk" else [],
                options=options,
            )
            catalog.create_table(f"reference.{name}", definition, False)
        snapshots = {}
        for commit in recipe["commits"]:
            if name == "orders_pk":
                result = native(
                    "write",
                    commit["rows"],
                    commit["label"],
                    commit.get("kind", "insert"),
                )
                snapshot = result["snapshot_id"]
            else:
                table = catalog.get_table(f"reference.{name}")
                builder = table.new_batch_write_builder()
                write, finish = builder.new_write(), builder.new_commit()
                try:
                    write.write_arrow_batch(batch(commit["rows"]))
                    finish.commit(write.prepare_commit())
                finally:
                    write.close()
                    finish.close()
                snapshot = (
                    catalog.get_table(f"reference.{name}")
                    .new_read_builder()
                    .new_scan()
                    .plan()
                    .snapshot_id
                )
            if (
                type(snapshot) is not int
                or snapshot <= 0
                or snapshot in snapshots.values()
            ):
                raise AssertionError(
                    f"{name}: commit did not publish a distinct snapshot"
                )
            snapshots[commit["label"]] = snapshot
        table = catalog.get_table(f"reference.{name}")
        plan = table.new_read_builder().new_scan().plan()
        files = sorted({f.file_name for split in plan.splits() for f in split.files})
        partitions = sorted(
            {
                tuple(
                    v.isoformat() if isinstance(v, date) else v
                    for v in split.partition.values
                )
                for split in plan.splits()
            }
        )
        persisted = table_schema(
            output, {"path": str(Path("warehouse/reference.db") / name)}
        )
        if recipe.get("partitioned") and (len(files) < 2 or len(partitions) != 2):
            raise AssertionError("Need two live partitions and at least two live files")
        if not recipe["commits"] and (plan.snapshot_id is not None or plan.splits()):
            raise AssertionError(
                "Empty fixture unexpectedly contains a snapshot or splits"
            )
        manifest["tables"][name] = {
            "path": str(Path("warehouse/reference.db") / name),
            "snapshots": snapshots,
            "schema": recipe["schema"],
            "options": persisted["options"],
            "partition_columns": [
                [key, dict(recipe["schema"])[key]] for key in persisted["partitionKeys"]
            ],
            "writer": "paimon-cpp" if name == "orders_pk" else "pypaimon",
            "live_files": files,
            "live_partitions": [list(values) for values in partitions],
        }
        print(f"Generated {name}: {snapshots}", flush=True)

    # After ALL writes, independently read historical complete rows using Python.
    # SQL aggregate/filter expectations remain separately hand-authored.
    manifest["independent_checks"] = []
    for case in spec["cases"]:
        if case["sql"] != "SELECT * FROM {scan}":
            continue
        table = catalog.get_table(f"reference.{case['table']}")
        if case["snapshot"]:
            snapshot = manifest["tables"][case["table"]]["snapshots"][case["snapshot"]]
            table = table.copy({"scan.snapshot-id": str(snapshot)})
        read = table.new_read_builder()
        plan = read.new_scan().plan()
        data = read.new_read().to_arrow(plan.splits(), parallelism=1)
        rows = [] if data is None else data.to_pylist()
        for row in rows:
            for key, value in row.items():
                if isinstance(value, date):
                    row[key] = value.isoformat()
        compare_rows(case, rows)
        manifest["independent_checks"].append(
            {"case": case["id"], "rows": rows, "passed": True}
        )
    # These are task-owned temporary input files, not Paimon table files.
    shutil.rmtree(inputs)
    shutil.copy2(HERE / "expectations.json", output / "expectations.json")
    manifest["files"] = {
        str(p.relative_to(output)): digest(p)
        for p in sorted(warehouse.rglob("*"))
        if p.is_file()
    }
    for entry in manifest["independent_checks"]:
        case = next(c for c in spec["cases"] if c["id"] == entry["case"])
        entry["identity"] = oracle_identity(case, manifest)
    validate_inventory(output, manifest)
    validate_table_metadata(output, manifest)
    validate_oracle(spec, manifest, compare_rows)
    (output / "manifest.json").write_text(canonical_json(manifest))
    print(f"Validated corpus: {output}", flush=True)


if __name__ == "__main__":
    main()
