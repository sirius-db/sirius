#!/usr/bin/env python3
"""Build the Iceberg equality-delete fixtures the GPU scan path is tested against.

The fixtures are not checked in: test_gpu_execution_multi_format.cpp runs this script once per
test process and reads them from a scratch directory. To build and check them by hand:
    python3 test/cpp/integration/data/generate_equality_delete_fixtures.py --out DIR
    python3 test/cpp/integration/data/generate_equality_delete_fixtures.py --out DIR NAME...

Each fixture isolates one rule a reader gets silently wrong. The rows a WRONG reader returns
are listed beside the right ones, so a failure says which rule broke:

  iceberg_v2_eq_wide_delete_file    The delete file stores the whole row; equality_ids=[1].
                                    Matching on every column deletes nothing.
  iceberg_v2_eq_with_positional     A position delete (row 3, date) and an equality delete
                                    (banana) apply to the same data file. Positions name rows
                                    by where they sit in the file, so they must be applied
                                    before the equality delete removes a row above them: a
                                    reader that runs them the other way round deletes
                                    elderberry instead of date.
  iceberg_v2_eq_manifest_rewrite    A rewritten manifest: its own sequence number is 3, but one
                                    entry keeps data sequence number 1. Comparing the MANIFEST
                                    number skips a delete that applies.
  iceberg_v2_eq_null_key            Composite key with NULL members. NULL matches NULL.
  iceberg_v2_eq_partitioned         One delete stored in partition A, one stored under an
                                    unpartitioned spec. The first must not reach partition B;
                                    the second must reach both.
  iceberg_v2_eq_partition_shapes    Deletes stored under an older spec, a two-field spec with an
                                    int and a NULL partition value, a spec whose only field is
                                    void, and an unpartitioned spec. The first two stay inside
                                    their spec and partition; the last two reach every file.
  iceberg_v2_eq_partition_bounds    Two data files of one partition with disjoint key bounds, a
                                    delete stored in that partition and one stored under an
                                    unpartitioned spec. Both deletes apply to the same data
                                    files and each matches one of them: one group, and the data
                                    files share a batch.
  iceberg_v2_eq_v1_data_manifest    The partitioned table as an upgraded table carries it: format
                                    version 1 wrote the data manifest, which has no `content`
                                    field and no sequence numbers. A reader that needs `content`
                                    cannot plan the partition-scoped delete. The manifest has no
                                    bounds fields either, while the delete in partition A records
                                    bounds.
  iceberg_v2_eq_float_partition     Data files in partitions -0.0, +0.0 and 1.5 of a float field,
                                    and two sibling snapshots. A delete stored in 1.5 is applied
                                    on the GPU. A delete stored in +0.0 is declined: DuckDB takes
                                    the two zeros for one partition and also deletes in -0.0,
                                    where the spec's bit patterns keep them apart.
  iceberg_v2_eq_upsert_same_commit  One commit rewrites key 2 twice: the equality delete removes
                                    the OLDER commit's row, a position delete removes this
                                    commit's first version. Applying the equality delete to its
                                    own commit loses the row just written.
  iceberg_v2_eq_renamed_key         The key column was renamed between the data file and the
                                    delete file. Only the field ID still links them.
  iceberg_v2_eq_promoted_key        The key was promoted int -> long. Old data and an old delete
                                    file store INT32; newer files store INT64.
  iceberg_v2_eq_double_key          A DOUBLE key. The GPU path declines it; the fixture pins the
                                    refusal and the rows DuckDB returns.
  iceberg_v2_eq_typed_keys          One snapshot per key type beyond int, long and string that the
                                    GPU path reads: date, timestamp and boolean. The snapshots are
                                    siblings, so each carries exactly one equality delete.
  iceberg_v3_eq_timestamp_ns_key    A timestamp_ns key. The type exists from format version 3 on,
                                    and the extension plans a V3 table only with row lineage, so
                                    this table is V3 and carries first-row-id and next-row-id.
  iceberg_v2_eq_unsupported_keys    The same layout for the keys the GPU path declines: decimal,
                                    float, a struct member and a column the table has since
                                    dropped.
  iceberg_v3_eq_no_row_lineage      A V3 table without first-row-id and next-row-id, as older
                                    writers left them. Deletes do not depend on row lineage, so
                                    the table is read like any other.
  iceberg_v2_eq_remote_delete       The equality delete is recorded under an http:// URI; data
                                    and metadata are local. No Sirius backend reads that URI, so
                                    the GPU path must decline before opening it.
  iceberg_v2_eq_avro_delete         The equality delete is recorded as an Avro file, which the
                                    spec allows. The GPU path reads Parquet delete files and
                                    must decline before opening it.
  iceberg_v2_eq_avro_delete_older   The same Avro delete, committed before the only data file. A
                                    delete applies to data with a strictly lower sequence
                                    number, so this one applies to nothing: no reader opens it,
                                    and its format must not decline the scan.
  iceberg_v2_eq_key_ranges          Two data files whose manifest records the key's bounds and
                                    null count, and one sibling snapshot per delete: a key range
                                    beside both files, one that meets a file at its endpoint, a
                                    NULL key facing a file with NULLs, a delete without bounds, a
                                    composite key disjoint in one column, two deletes that each
                                    meet one file, a delete manifest whose schema has no bounds
                                    fields, a NULL key whose delete records no null count,
                                    string bounds truncated to 16 characters, a delete beside
                                    both files next to one that meets both, a delete whose own
                                    bounds are 2000 characters long, six deletes with
                                    61-character bounds, a composite key with bounds for one
                                    column only, and one whose first column cannot decide for
                                    the file with NULLs. Skipping a delete whose range still
                                    meets a file, or whose NULL key does, returns deleted rows.
                                    One more manifest, listed by no snapshot, has a literal %20
                                    in its name and no live entry.
  iceberg_v2_eq_manifest_shapes     Two data manifests of different shape. One records every
                                    metric map. The other has no null_value_counts field and a
                                    zero NaN count, and its data file holds a NULL key outside
                                    the bounds of a delete that has a NULL key. Taking another
                                    map for the missing null counts skips the delete and returns
                                    the NULL row.
  iceberg_v2_eq_manifest_shapes_bare_first
                                    The same table with the two manifests in the other order.
  iceberg_v2_eq_duplicate_entry     For the plan only. The data manifest lists its data file
                                    twice; the second entry's bounds are beside a delete that
                                    matches the file.
  iceberg_v2_eq_own_file_index      The data manifest has a column of its own named file_index.
                                    It holds 0, the position read_avro gives the only manifest
                                    of a list, and the entry's bounds are beside the delete.
  iceberg_v2_eq_mixed_manifests     Two data manifests, each with one data file whose bounds
                                    are beside the delete. The one listed first is written as
                                    format version 2 writes it, the other as version 1 did,
                                    without the sequence-number columns.
  iceberg_v2_eq_repeated_bound      For the plan only. The bounds maps of the data file list
                                    the key twice, and both pairs are beside a delete that
                                    matches the file. Taking either skips the delete.
  iceberg_v2_eq_repeated_delete_bound
                                    The same with the repeat in the delete file's entry.
  iceberg_v2_eq_bounds_stream       For the plan only: its data files are not written. A data
                                    manifest large enough that a streamed read hands out entries
                                    before it ends, and a delete beside their bounds.

Hand-forged, like the other delete fixtures: pyiceberg does not write delete files, and the only
independent writer of equality deletes is Flink. Avro schemas are copied from an existing
fixture so they stay byte-compatible with what the Iceberg extension already reads. The format
version 1 manifest has the fields of one written by pyiceberg.
"""

import argparse
import copy
import json
import pathlib
import shutil
import struct
import sys

DATA = pathlib.Path(__file__).resolve().parent
REL = pathlib.Path("test/cpp/integration/data")
TEMPLATE = DATA / "iceberg_eq_seq_number" / "metadata"
POS_PATH_ID, POS_POS_ID = 2147483546, 2147483545
BOUNDS_FIELDS = ("lower_bounds", "upper_bounds", "null_value_counts")


def _avro(path):
    import fastavro

    with open(path, "rb") as fh:
        reader = fastavro.reader(fh)
        return dict(reader.metadata), reader.writer_schema, list(reader)


def _as_v1_manifest(schema, meta, rows):
    """Removes what format version 2 added to a data manifest; v1 requires block_size_in_bytes."""
    schema["fields"] = [
        f for f in schema["fields"] if not f["name"].endswith("sequence_number")
    ]
    for field in schema["fields"]:
        if field["name"] != "data_file":
            continue
        fields = [
            f
            for f in field["type"]["fields"]
            if f["name"] not in ("content", "equality_ids")
        ]
        size = [f["name"] for f in fields].index("file_size_in_bytes")
        fields.insert(
            size + 1, {"name": "block_size_in_bytes", "type": "long", "field-id": 105}
        )
        field["type"]["fields"] = fields
    del meta["content"]
    meta["format-version"] = "1"
    for row in rows:
        del row["sequence_number"], row["file_sequence_number"]
        del row["data_file"]["content"], row["data_file"]["equality_ids"]
        row["data_file"]["block_size_in_bytes"] = 64 * 1024 * 1024


class Table:
    """One fixture: accumulates commits, then writes metadata that describes all of them."""

    def __init__(
        self,
        name,
        schemas,
        partition_specs=None,
        format_version=2,
        row_lineage=True,
        partition_types=None,
    ):
        # partition_types: Avro type of a partition field by name, where it is not a string.
        self.partition_types = partition_types or {}
        self.name, self.root, self.rel = name, DATA / name, REL / name
        self.format_version, self.next_row_id = format_version, 0
        self.row_lineage = row_lineage and format_version >= 3
        self.schemas = schemas  # list of [(field_id, name, iceberg_type)]
        self.specs = partition_specs or [[]]
        self.snapshots, self.manifests, self.seq = [], [], 0
        self.v1_manifests = set()  # data manifests to write as format version 1 did
        self.file_formats = (
            {}
        )  # file_format of an entry by path, where it is not PARQUET
        self.bare_manifests = {}  # manifest -> the metric fields its schema leaves out
        self.indexed_manifests = (
            {}
        )  # manifest -> the value of a `file_index` column of its own
        if self.root.exists():
            shutil.rmtree(self.root)
        (self.root / "data").mkdir(parents=True)
        (self.root / "metadata").mkdir()
        self.entry_meta, self.entry_schema, _ = _avro(TEMPLATE / "manifest-delete.avro")
        _, self.list_schema, rows = _avro(TEMPLATE / "snap-9600000000000002-2-ml.avro")
        self.list_row = rows[0]

    def parquet(self, file_name, columns):
        """columns: [(field_id, name, arrow_type, values)]. Returns (relative path, rows, size)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        fields = [
            pa.field(n, t, metadata={"PARQUET:field_id": str(i)})
            for i, n, t, _ in columns
        ]
        table = pa.table(
            [pa.array(v, t) for _, _, t, v in columns], schema=pa.schema(fields)
        )
        pq.write_table(table, self.root / "data" / file_name)
        return (
            str(self.rel / "data" / file_name),
            table.num_rows,
            (self.root / "data" / file_name).stat().st_size,
        )

    def _manifest(self, name, content, spec_id, entries):
        """entries: [(status, explicit_sequence_or_None, file, content, partition, equality_ids)],
        optionally followed by the file's column metrics: a dict with any of `lower_bounds`,
        `upper_bounds` ({field_id: serialized value}) and the count maps ({field_id: n}). A
        map given as a list of (field_id, value) pairs is written as it is, repeats included.
        """
        import fastavro

        entries = [e if len(e) == 7 else (*e, None) for e in entries]

        spec = self.specs[spec_id]
        schema = copy.deepcopy(self.entry_schema)
        for field in schema["fields"]:
            if field["name"] != "data_file":
                continue
            for sub in field["type"]["fields"]:
                if sub["name"] == "partition":
                    sub["type"]["fields"] = [
                        {
                            "field-id": f["field-id"],
                            "name": f["name"],
                            "type": [
                                "null",
                                self.partition_types.get(f["name"], "string"),
                            ],
                            "default": None,
                        }
                        for f in spec
                    ]
        meta = dict(self.entry_meta)
        meta["content"] = "data" if content == 0 else "deletes"
        meta["partition-spec"] = json.dumps(spec)
        meta["partition-spec-id"] = str(spec_id)
        meta["schema"] = json.dumps(self._schema_json(len(self.schemas) - 1))
        rows = []
        for (
            status,
            explicit_seq,
            (path, count, size),
            kind,
            partition,
            ids,
            metrics,
        ) in entries:
            metrics = {
                name: [
                    {"key": k, "value": v}
                    for k, v in (
                        sorted(values.items()) if isinstance(values, dict) else values
                    )
                ]
                for name, values in (metrics or {}).items()
            }
            rows.append(
                {
                    "status": status,
                    "snapshot_id": self.snapshot_id(self.seq),
                    "sequence_number": explicit_seq,
                    "file_sequence_number": explicit_seq,
                    "data_file": {
                        "content": kind,
                        "file_path": path,
                        "file_format": self.file_formats.get(path, "PARQUET"),
                        "partition": partition or {},
                        "record_count": count,
                        "file_size_in_bytes": size,
                        "column_sizes": None,
                        "value_counts": metrics.get("value_counts"),
                        "null_value_counts": metrics.get("null_value_counts"),
                        "nan_value_counts": metrics.get("nan_value_counts"),
                        "lower_bounds": metrics.get("lower_bounds"),
                        "upper_bounds": metrics.get("upper_bounds"),
                        "key_metadata": None,
                        "split_offsets": [4],
                        "equality_ids": ids,
                        "sort_order_id": 0 if kind != 1 else None,
                    },
                }
            )
        v1 = name in self.v1_manifests
        if v1:
            _as_v1_manifest(schema, meta, rows)
        if name in self.bare_manifests:
            bounds = self.bare_manifests[name]
            for field in schema["fields"]:
                if field["name"] == "data_file":
                    field["type"]["fields"] = [
                        f for f in field["type"]["fields"] if f["name"] not in bounds
                    ]
            for row in rows:
                for bound in bounds:
                    del row["data_file"][bound]
        if name in self.indexed_manifests:
            schema["fields"].append(
                {"name": "file_index", "type": "long", "field-id": 9000}
            )
            for row in rows:
                row["file_index"] = self.indexed_manifests[name]
        out = self.root / "metadata" / name
        with open(out, "wb") as fh:
            fastavro.writer(fh, schema, rows, codec="null", metadata=meta)
        added = sum(1 for r in rows if r["status"] == 1)
        existing = sum(1 for r in rows if r["status"] == 0)

        def summary(field):
            values = [e[4][field["name"]] for e in entries if e[4]]
            present = sorted(v for v in values if v is not None)

            def bound(v):
                if isinstance(v, float):
                    return struct.pack("<f", v)
                return v.encode() if isinstance(v, str) else v.to_bytes(4, "little")

            return {
                "contains_null": len(present) != len(values),
                "contains_nan": False,
                "lower_bound": bound(present[0]) if present else None,
                "upper_bound": bound(present[-1]) if present else None,
            }

        row = dict(self.list_row)
        row.update(
            manifest_path=str(self.rel / "metadata" / name),
            manifest_length=out.stat().st_size,
            partition_spec_id=spec_id,
            content=content,
            sequence_number=0 if v1 else self.seq,
            min_sequence_number=(
                0
                if v1
                else min([self.seq] + [s for _, s, *_ in entries if s is not None])
            ),
            added_snapshot_id=self.snapshot_id(self.seq),
            added_files_count=added,
            existing_files_count=existing,
            deleted_files_count=0,
            added_rows_count=sum(
                r["data_file"]["record_count"] for r in rows if r["status"] == 1
            ),
            existing_rows_count=sum(
                r["data_file"]["record_count"] for r in rows if r["status"] == 0
            ),
            deleted_rows_count=0,
            partitions=(
                [summary(f) for f in spec] if any(e[4] for e in entries) else []
            ),
        )
        return row

    def snapshot_id(self, seq):
        return 8800000000000000 + seq

    def commit(self, operation, manifests, replace=False, sibling=False):
        """manifests: [(file name, content, spec_id, entries)]. replace drops earlier DATA
        manifests, as a manifest rewrite does. A sibling snapshot lists its manifests without
        handing them on to later commits."""
        import fastavro

        self.seq += 1
        if replace:
            self.manifests = [m for m in self.manifests if m["content"] != 0]
        listed = self.manifests + [self._manifest(*m) for m in manifests]
        if not sibling:
            self.manifests = listed
        name = f"snap-{self.snapshot_id(self.seq)}-{self.seq}-ml.avro"
        with open(self.root / "metadata" / name, "wb") as fh:
            fastavro.writer(fh, self.list_schema, listed, codec="null")
        added_rows = sum(
            count
            for _, _, _, entries in manifests
            for status, _, (_, count, _), kind, *_ in entries
            if status == 1 and kind == 0
        )
        lineage = {"first-row-id": self.next_row_id, "added-rows": added_rows}
        self.next_row_id += added_rows
        self.snapshots.append(
            {
                "snapshot-id": self.snapshot_id(self.seq),
                "sequence-number": self.seq,
                "timestamp-ms": 1700000000000 + 1000 * self.seq,
                "manifest-list": str(self.rel / "metadata" / name),
                "summary": {"operation": operation},
                "schema-id": len(self.schemas) - 1,
                **(lineage if self.row_lineage else {}),
                **(
                    {"parent-snapshot-id": self.snapshot_id(self.seq - 1)}
                    if self.seq > 1
                    else {}
                ),
            }
        )

    def _schema_json(self, schema_id):
        return {
            "type": "struct",
            "schema-id": schema_id,
            "fields": [
                {"id": i, "name": n, "required": False, "type": t}
                for i, n, t in self.schemas[schema_id]
            ],
        }

    def finish(self):
        meta = json.loads((TEMPLATE / "v1.metadata.json").read_text())
        meta.update(
            {
                "location": str(self.rel),
                "table-uuid": "e0e0e0e0-0000-4000-8000-"
                + f"{abs(hash(self.name)) % 16**12:012x}",
                "last-sequence-number": self.seq,
                "last-updated-ms": self.snapshots[-1]["timestamp-ms"],
                "last-column-id": max(i for s in self.schemas for i, _, _ in s),
                "current-schema-id": len(self.schemas) - 1,
                "schemas": [self._schema_json(i) for i in range(len(self.schemas))],
                "partition-specs": [
                    {"spec-id": i, "fields": spec} for i, spec in enumerate(self.specs)
                ],
                "default-spec-id": 0,
                "last-partition-id": max(
                    [999] + [f["field-id"] for s in self.specs for f in s]
                ),
                "current-snapshot-id": self.snapshots[-1]["snapshot-id"],
                "refs": {
                    "main": {
                        "snapshot-id": self.snapshots[-1]["snapshot-id"],
                        "type": "branch",
                    }
                },
                "snapshots": self.snapshots,
                "snapshot-log": [
                    {"snapshot-id": s["snapshot-id"], "timestamp-ms": s["timestamp-ms"]}
                    for s in self.snapshots
                ],
            }
        )
        if self.format_version >= 3:
            meta["format-version"] = 3
        if self.row_lineage:
            meta["next-row-id"] = self.next_row_id
        # A fixed UUID per fixture: hash() is salted per process.
        meta["table-uuid"] = "e0e0e0e0-0000-4000-8000-%012x" % (
            sum(ord(c) * 131**k for k, c in enumerate(self.name)) % 16**12
        )
        (self.root / "metadata" / "v1.metadata.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n"
        )
        (self.root / "metadata" / "version-hint.text").write_text("1")


def fruit_schema():
    return [[(1, "fruit", "string"), (2, "count", "long")]]


def build_wide_delete_file():
    import pyarrow as pa

    t = Table("iceberg_v2_eq_wide_delete_file", fruit_schema())
    data = t.parquet(
        "data-1.parquet",
        [
            (
                1,
                "fruit",
                pa.string(),
                ["apple", "banana", "cherry", "date", "elderberry"],
            ),
            (2, "count", pa.int64(), [1, 2, 3, 4, 5]),
        ],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    # The whole deleted row, with counts that no longer match the data: only fruit is a key.
    # Columns are stored count-first so position cannot stand in for the field ID either.
    delete = t.parquet(
        "delete-eq-2.parquet",
        [
            (2, "count", pa.int64(), [999, 0]),
            (1, "fruit", pa.string(), ["banana", "date"]),
        ],
    )
    t.commit("delete", [("m-delete-2.avro", 1, 0, [(1, None, delete, 2, None, [1])])])
    t.finish()


def build_with_positional():
    import pyarrow as pa

    t = Table("iceberg_v2_eq_with_positional", fruit_schema())
    data = t.parquet(
        "data-1.parquet",
        [
            (
                1,
                "fruit",
                pa.string(),
                ["apple", "banana", "cherry", "date", "elderberry"],
            ),
            (2, "count", pa.int64(), [1, 2, 3, 4, 5]),
        ],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    # Reserved field IDs of a position delete file.
    positional = t.parquet(
        "delete-pos-2.parquet",
        [
            (2147483546, "file_path", pa.string(), [data[0]]),
            (2147483545, "pos", pa.int64(), [3]),
        ],
    )
    equality = t.parquet("delete-eq-2.parquet", [(1, "fruit", pa.string(), ["banana"])])
    t.commit(
        "delete",
        [
            (
                "m-delete-2.avro",
                1,
                0,
                [
                    (1, None, positional, 1, None, None),
                    (1, None, equality, 2, None, [1]),
                ],
            )
        ],
    )
    t.finish()


def build_manifest_rewrite():
    import pyarrow as pa

    t = Table("iceberg_v2_eq_manifest_rewrite", fruit_schema())
    data1 = t.parquet(
        "data-1.parquet",
        [
            (1, "fruit", pa.string(), ["apple", "banana", "cherry"]),
            (2, "count", pa.int64(), [1, 2, 3]),
        ],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data1, 0, None, None)])])
    delete = t.parquet("delete-eq-2.parquet", [(1, "fruit", pa.string(), ["banana"])])
    t.commit("delete", [("m-delete-2.avro", 1, 0, [(1, None, delete, 2, None, [1])])])
    data2 = t.parquet(
        "data-3.parquet",
        [
            (1, "fruit", pa.string(), ["banana", "date"]),
            (2, "count", pa.int64(), [4, 5]),
        ],
    )
    # The append also rewrites the data manifest: data-1 becomes EXISTING and keeps its own
    # sequence number 1 inside a manifest whose sequence number is 3.
    t.commit(
        "append",
        [
            (
                "m-data-3-merged.avro",
                0,
                0,
                [(0, 1, data1, 0, None, None), (1, None, data2, 0, None, None)],
            )
        ],
        replace=True,
    )
    t.finish()


def build_null_key():
    import pyarrow as pa

    t = Table("iceberg_v2_eq_null_key", fruit_schema())
    data = t.parquet(
        "data-1.parquet",
        [
            (1, "fruit", pa.string(), ["apple", None, "cherry", None, "cherry"]),
            (2, "count", pa.int64(), [1, 2, 3, 4, None]),
        ],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    delete = t.parquet(
        "delete-eq-2.parquet",
        [
            (1, "fruit", pa.string(), [None, "cherry"]),
            (2, "count", pa.int64(), [2, None]),
        ],
    )
    t.commit(
        "delete", [("m-delete-2.avro", 1, 0, [(1, None, delete, 2, None, [1, 2])])]
    )
    t.finish()


def build_partitioned(name="iceberg_v2_eq_partitioned", v1_data_manifest=False):
    import pyarrow as pa

    spec = [
        {"name": "region", "transform": "identity", "source-id": 2, "field-id": 1000}
    ]
    t = Table(
        name,
        [[(1, "id", "long"), (2, "region", "string"), (3, "v", "long")]],
        [spec, []],
    )
    in_a_metrics = None
    if v1_data_manifest:
        # Written before the table recorded bounds, while the delete in partition A records them.
        t.v1_manifests.add("m-data-1.avro")
        t.bare_manifests["m-data-1.avro"] = BOUNDS_FIELDS
        one = struct.pack("<q", 1)
        in_a_metrics = {
            "lower_bounds": {1: one},
            "upper_bounds": {1: one},
            "null_value_counts": {1: 0},
        }

    def data(region, values):
        return t.parquet(
            f"data-1-{region}.parquet",
            [
                (1, "id", pa.int64(), [1, 2, 3]),
                (2, "region", pa.string(), [region] * 3),
                (3, "v", pa.int64(), values),
            ],
        )

    a, b = data("A", [10, 20, 50]), data("B", [30, 40, 60])
    t.commit(
        "append",
        [
            (
                "m-data-1.avro",
                0,
                0,
                [
                    (1, None, a, 0, {"region": "A"}, None),
                    (1, None, b, 0, {"region": "B"}, None),
                ],
            )
        ],
    )
    in_a = t.parquet("delete-eq-2-A.parquet", [(1, "id", pa.int64(), [1])])
    t.commit(
        "delete",
        [
            (
                "m-delete-2.avro",
                1,
                0,
                [(1, None, in_a, 2, {"region": "A"}, [1], in_a_metrics)],
            )
        ],
    )
    everywhere = t.parquet("delete-eq-3-global.parquet", [(1, "id", pa.int64(), [2])])
    t.commit(
        "delete", [("m-delete-3.avro", 1, 1, [(1, None, everywhere, 2, None, [1])])]
    )
    t.finish()


def build_partition_shapes():
    import pyarrow as pa

    region = {"name": "region", "source-id": 2, "field-id": 1000}
    t = Table(
        "iceberg_v2_eq_partition_shapes",
        [
            [
                (1, "id", "long"),
                (2, "region", "string"),
                (3, "k", "int"),
                (4, "v", "long"),
            ]
        ],
        [
            [{**region, "transform": "identity"}],
            [
                {**region, "transform": "identity"},
                {
                    "name": "k",
                    "transform": "identity",
                    "source-id": 3,
                    "field-id": 1001,
                },
            ],
            [{**region, "transform": "void"}],
            [],
        ],
        partition_types={"k": "int"},
    )

    def data(n, region, k):
        return t.parquet(
            f"data-{n}.parquet",
            [
                (1, "id", pa.int64(), [1, 2, 3, 4, 5]),
                (2, "region", pa.string(), [region] * 5),
                (3, "k", pa.int32(), [k] * 5),
                (4, "v", pa.int64(), [10 * n + i for i in range(1, 6)]),
            ],
        )

    def delete(name, key):
        return t.parquet(f"delete-eq-{name}.parquet", [(1, "id", pa.int64(), [key])])

    def added(file, kind, partition, ids=None):
        return (1, None, file, kind, partition, ids)

    # Files 1 and 2 are written under the first spec, files 3 to 5 under the second.
    t.commit(
        "append",
        [
            (
                "m-data-1.avro",
                0,
                0,
                [
                    added(data(1, "A", 1), 0, {"region": "A"}),
                    added(data(2, "B", 1), 0, {"region": "B"}),
                ],
            )
        ],
    )
    t.commit(
        "append",
        [
            (
                "m-data-2.avro",
                0,
                1,
                [
                    added(data(3, "A", 1), 0, {"region": "A", "k": 1}),
                    added(data(4, "A", 2), 0, {"region": "A", "k": 2}),
                    added(data(5, None, 1), 0, {"region": None, "k": 1}),
                ],
            )
        ],
    )
    t.commit(
        "delete",
        [
            # region=A under the first spec: file 1 only, not the second spec's region A.
            (
                "m-delete-3a.avro",
                1,
                0,
                [added(delete("3a", 1), 2, {"region": "A"}, [1])],
            ),
            # (A, 1) and (NULL, 1) under the second spec: files 3 and 5.
            (
                "m-delete-3b.avro",
                1,
                1,
                [
                    added(delete("3b", 2), 2, {"region": "A", "k": 1}, [1]),
                    added(delete("3c", 3), 2, {"region": None, "k": 1}, [1]),
                ],
            ),
            # A spec whose only field is void does not partition: every file.
            (
                "m-delete-3d.avro",
                1,
                2,
                [added(delete("3d", 4), 2, {"region": None}, [1])],
            ),
            # An unpartitioned spec: every file.
            ("m-delete-3e.avro", 1, 3, [added(delete("3e", 5), 2, None, [1])]),
        ],
    )
    t.finish()


# The partition each sibling snapshot of iceberg_v2_eq_float_partition stores its delete in.
FLOAT_PARTITION_DELETES = {2: 1.5, 3: 0.0}


def build_float_partition():
    import pyarrow as pa

    spec = [{"name": "f", "transform": "identity", "source-id": 2, "field-id": 1000}]
    t = Table(
        "iceberg_v2_eq_float_partition",
        [[(1, "id", "long"), (2, "f", "float"), (3, "v", "long")]],
        [spec],
        partition_types={"f": "float"},
    )

    def data(n, f):
        file = t.parquet(
            f"data-{n}.parquet",
            [
                (1, "id", pa.int64(), [1, 2, 3]),
                (2, "f", pa.float32(), [f] * 3),
                (3, "v", pa.int64(), [10 * n + i for i in (1, 2, 3)]),
            ],
        )
        return (1, None, file, 0, {"f": f}, None)

    t.commit(
        "append",
        [("m-data-1.avro", 0, 0, [data(1, -0.0), data(2, 0.0), data(3, 1.5)])],
    )
    # Sibling snapshots, each deleting id 1 in one partition.
    for seq, f in FLOAT_PARTITION_DELETES.items():
        file = t.parquet(f"delete-eq-{seq}.parquet", [(1, "id", pa.int64(), [1])])
        entry = (1, None, file, 2, {"f": f}, [1])
        t.commit("delete", [(f"m-delete-{seq}.avro", 1, 0, [entry])], sibling=True)
    t.finish()


def build_upsert_same_commit():
    import pyarrow as pa

    t = Table(
        "iceberg_v2_eq_upsert_same_commit", [[(1, "id", "long"), (2, "v", "string")]]
    )
    data1 = t.parquet(
        "data-1.parquet",
        [(1, "id", pa.int64(), [1, 2, 3]), (2, "v", pa.string(), ["a", "b", "c"])],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data1, 0, None, None)])])
    data2 = t.parquet(
        "data-2.parquet",
        [(1, "id", pa.int64(), [2, 2, 4]), (2, "v", pa.string(), ["x", "y", "d"])],
    )
    eq = t.parquet("delete-eq-2.parquet", [(1, "id", pa.int64(), [2])])
    pos = t.parquet(
        "delete-pos-2.parquet",
        [
            (POS_PATH_ID, "file_path", pa.string(), [data2[0]]),
            (POS_POS_ID, "pos", pa.int64(), [0]),
        ],
    )
    t.commit(
        "overwrite",
        [
            ("m-data-2.avro", 0, 0, [(1, None, data2, 0, None, None)]),
            (
                "m-delete-2.avro",
                1,
                0,
                [(1, None, eq, 2, None, [1]), (1, None, pos, 1, None, None)],
            ),
        ],
    )
    t.finish()


def build_renamed_key():
    import pyarrow as pa

    t = Table(
        "iceberg_v2_eq_renamed_key",
        [
            [(1, "fruit", "string"), (2, "count", "long")],
            [(1, "produce", "string"), (2, "count", "long")],
        ],
    )
    data = t.parquet(
        "data-1.parquet",
        [
            (1, "fruit", pa.string(), ["apple", "banana", "cherry"]),
            (2, "count", pa.int64(), [1, 2, 3]),
        ],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    delete = t.parquet("delete-eq-2.parquet", [(1, "produce", pa.string(), ["banana"])])
    t.commit("delete", [("m-delete-2.avro", 1, 0, [(1, None, delete, 2, None, [1])])])
    t.finish()


def build_promoted_key():
    import pyarrow as pa

    t = Table(
        "iceberg_v2_eq_promoted_key",
        [
            [(1, "id", "int"), (2, "v", "string")],
            [(1, "id", "long"), (2, "v", "string")],
        ],
    )
    data1 = t.parquet(
        "data-1-int.parquet",
        [
            (1, "id", pa.int32(), [1, 2, 3, 4]),
            (2, "v", pa.string(), ["a", "b", "c", "d"]),
        ],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data1, 0, None, None)])])
    old = t.parquet("delete-eq-2-int.parquet", [(1, "id", pa.int32(), [2])])
    t.commit("delete", [("m-delete-2.avro", 1, 0, [(1, None, old, 2, None, [1])])])
    data2 = t.parquet(
        "data-3-long.parquet",
        [(1, "id", pa.int64(), [2, 5, 6]), (2, "v", pa.string(), ["b2", "e", "f"])],
    )
    t.commit("append", [("m-data-3.avro", 0, 0, [(1, None, data2, 0, None, None)])])
    new = t.parquet("delete-eq-4-long.parquet", [(1, "id", pa.int64(), [4, 5])])
    t.commit("delete", [("m-delete-4.avro", 1, 0, [(1, None, new, 2, None, [1])])])
    t.finish()


def build_double_key():
    import pyarrow as pa

    t = Table(
        "iceberg_v2_eq_double_key", [[(1, "fruit", "string"), (2, "weight", "double")]]
    )
    data = t.parquet(
        "data-1.parquet",
        [
            (1, "fruit", pa.string(), ["apple", "banana", "cherry"]),
            (2, "weight", pa.float64(), [1.5, 2.5, 3.5]),
        ],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    delete = t.parquet("delete-eq-2.parquet", [(2, "weight", pa.float64(), [2.5])])
    t.commit("delete", [("m-delete-2.avro", 1, 0, [(1, None, delete, 2, None, [2])])])
    t.finish()


# What each sibling snapshot of iceberg_v2_eq_unsupported_keys deletes row id=2 by:
# snapshot sequence -> (label, equality field id).
UNSUPPORTED_KEYS = {
    2: ("decimal", 4),
    3: ("float", 6),
    4: ("struct member", 8),
    5: ("dropped column", 9),
}
# The same for iceberg_v2_eq_typed_keys.
TYPED_KEYS = {2: ("date", 2), 3: ("timestamp", 3), 4: ("boolean", 4)}


def build_typed_keys():
    import datetime

    import pyarrow as pa

    t = Table(
        "iceberg_v2_eq_typed_keys",
        [
            [
                (1, "id", "long"),
                (2, "day", "date"),
                (3, "ts", "timestamp"),
                (4, "flag", "boolean"),
            ]
        ],
    )
    day = datetime.date(2024, 1, 1)
    second = datetime.datetime(2024, 1, 1, 0, 0, 0)
    columns = {
        1: ("id", pa.int64(), [1, 2, 3]),
        2: ("day", pa.date32(), [day + datetime.timedelta(days=d) for d in range(3)]),
        3: (
            "ts",
            pa.timestamp("us"),
            [second + datetime.timedelta(seconds=d) for d in range(3)],
        ),
        4: ("flag", pa.bool_(), [True, False, True]),
    }
    data = t.parquet(
        "data-1.parquet", [(i, n, ty, v) for i, (n, ty, v) in columns.items()]
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    for seq, (_, key) in TYPED_KEYS.items():
        name, arrow_type, values = columns[key]
        delete = t.parquet(
            f"delete-eq-{seq}.parquet", [(key, name, arrow_type, [values[1]])]
        )
        t.commit(
            "delete",
            [(f"m-delete-{seq}.avro", 1, 0, [(1, None, delete, 2, None, [key])])],
            sibling=True,
        )
    t.finish()


def build_timestamp_ns_key():
    import datetime

    import pyarrow as pa

    t = Table(
        "iceberg_v3_eq_timestamp_ns_key",
        [[(1, "id", "long"), (2, "ts", "timestamp_ns")]],
        format_version=3,
    )
    second = datetime.datetime(2024, 1, 1, 0, 0, 0)
    stamps = [second + datetime.timedelta(seconds=d) for d in range(3)]
    data = t.parquet(
        "data-1.parquet",
        [(1, "id", pa.int64(), [1, 2, 3]), (2, "ts", pa.timestamp("ns"), stamps)],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    delete = t.parquet(
        "delete-eq-2.parquet", [(2, "ts", pa.timestamp("ns"), [stamps[1]])]
    )
    t.commit("delete", [("m-delete-2.avro", 1, 0, [(1, None, delete, 2, None, [2])])])
    t.finish()


def build_unsupported_keys():
    import datetime
    import decimal

    import pyarrow as pa

    point = {
        "type": "struct",
        "fields": [{"id": 8, "name": "x", "required": False, "type": "long"}],
    }
    live = [
        (1, "id", "long"),
        (2, "day", "date"),
        (3, "ts", "timestamp"),
        (4, "amount", "decimal(9, 2)"),
        (5, "flag", "boolean"),
        (6, "ratio", "float"),
        (7, "point", point),
    ]
    # The first schema still has `gone`; the current one dropped it. Its field ID stays in the
    # data file and in the delete file keyed on it.
    t = Table("iceberg_v2_eq_unsupported_keys", [live + [(9, "gone", "long")], live])
    x = pa.field("x", pa.int64(), metadata={"PARQUET:field_id": "8"})
    day = datetime.date(2024, 1, 1)
    second = datetime.datetime(2024, 1, 1, 0, 0, 0)
    columns = {
        1: ("id", pa.int64(), [1, 2, 3]),
        2: ("day", pa.date32(), [day + datetime.timedelta(days=d) for d in range(3)]),
        3: (
            "ts",
            pa.timestamp("us"),
            [second + datetime.timedelta(seconds=d) for d in range(3)],
        ),
        4: (
            "amount",
            pa.decimal128(9, 2),
            [decimal.Decimal(v) for v in ("1.50", "2.50", "3.50")],
        ),
        5: ("flag", pa.bool_(), [True, False, True]),
        6: ("ratio", pa.float32(), [1.5, 2.5, 3.5]),
        7: ("point", pa.struct([x]), [{"x": 10}, {"x": 20}, {"x": 30}]),
        9: ("gone", pa.int64(), [100, 200, 300]),
    }
    data = t.parquet(
        "data-1.parquet", [(i, n, ty, v) for i, (n, ty, v) in columns.items()]
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    for seq, (_, key) in UNSUPPORTED_KEYS.items():
        column = 7 if key == 8 else key  # the struct member is stored inside its struct
        name, arrow_type, values = columns[column]
        delete = t.parquet(
            f"delete-eq-{seq}.parquet", [(column, name, arrow_type, [values[1]])]
        )
        t.commit(
            "delete",
            [(f"m-delete-{seq}.avro", 1, 0, [(1, None, delete, 2, None, [key])])],
            sibling=True,
        )
    t.finish()


def build_unread_delete(name, delete_path, file_format="PARQUET", applies=True):
    """applies=False commits the delete before the data file, so it applies to nothing."""
    import pyarrow as pa

    t = Table(name, fruit_schema())
    t.file_formats[delete_path] = file_format

    def append():
        seq = t.seq + 1
        data = t.parquet(
            f"data-{seq}.parquet",
            [
                (1, "fruit", pa.string(), ["apple", "banana", "cherry"]),
                (2, "count", pa.int64(), [1, 2, 3]),
            ],
        )
        entry = (1, None, data, 0, None, None)
        t.commit("append", [(f"m-data-{seq}.avro", 0, 0, [entry])])

    def delete():
        # Recorded in the manifest and never written: nothing may open it.
        entry = (1, None, (delete_path, 1, 540), 2, None, [1])
        t.commit("delete", [(f"m-delete-{t.seq + 1}.avro", 1, 0, [entry])])

    for commit in (append, delete) if applies else (delete, append):
        commit()
    t.finish()


def build_no_row_lineage():
    import pyarrow as pa

    t = Table(
        "iceberg_v3_eq_no_row_lineage",
        fruit_schema(),
        format_version=3,
        row_lineage=False,
    )
    data = t.parquet(
        "data-1.parquet",
        [
            (1, "fruit", pa.string(), ["apple", "banana", "cherry"]),
            (2, "count", pa.int64(), [1, 2, 3]),
        ],
    )
    t.commit("append", [("m-data-1.avro", 0, 0, [(1, None, data, 0, None, None)])])
    delete = t.parquet("delete-eq-2.parquet", [(1, "fruit", pa.string(), ["banana"])])
    t.commit("delete", [("m-delete-2.avro", 1, 0, [(1, None, delete, 2, None, [1])])])
    t.finish()


# (file number, region, k, the id its partition-scoped delete removes); ids 4 and 5 go everywhere.
PARTITION_SHAPES = [
    (1, "A", 1, 1),
    (2, "B", 1, None),
    (3, "A", 1, 2),
    (4, "A", 2, None),
    (5, None, 1, 3),
]

# Sequence number of each iceberg_v2_eq_key_ranges snapshot -> the ids it leaves.
KEY_RANGES = {
    2: [1, 2, 3, 4, 5, 6],  # k in [100, 110]: beside both data files
    3: [1, 2, 4, 5, 6],  # k in [30, 35]: meets data-a at its upper bound
    4: [1, 2, 3, 4, 5],  # k = NULL and 100: data-b holds a NULL
    5: [1, 3, 4, 5, 6],  # k = 20, no bounds recorded
    6: [1, 2, 3, 4, 5, 6],  # (k, s) = (20, "zz"): s is beside both data files
    7: [1, 3, 4, 5, 6],  # k = 20 and k = 45 in two delete files, one for each data file
    8: [1, 3, 4, 5, 6],  # k = 20, in a manifest without the bounds fields
    9: [1, 2, 3, 4, 5],  # k = NULL and 100, bounds recorded but no null count
    10: [
        1,
        2,
        3,
        4,
        5,
        6,
    ],  # s = 20 times "z", bounds truncated to 16 as Iceberg Java does
    11: [
        1,
        3,
        4,
        6,
    ],  # k in [100, 110] beside both files, and k = 20 and 50 in a second file
    12: [
        1,
        2,
        3,
        4,
        5,
        6,
    ],  # s = 2000 times "z": bounds longer than a plan reads by default
    13: [1, 2, 3, 4, 5, 6],  # six delete files of s, each with 61-byte bounds
    14: [1, 2, 3, 4, 5, 6],  # (k, s) = (20, "zz") with bounds recorded for s alone
    15: [
        1,
        2,
        3,
        4,
        5,
        6,
    ],  # (k, s) = (NULL, "zz") and (20, "zz"): k cannot decide for data-b
}


def build_key_ranges():
    import pyarrow as pa

    def metrics(k=None, s=None, k_nulls=0):
        """Bounds as a manifest serializes them: long little-endian, string UTF-8."""
        lower, upper = {}, {}
        if k:
            lower[2], upper[2] = (struct.pack("<q", v) for v in k)
        if s:
            lower[3], upper[3] = (v.encode() for v in s)
        nulls = {3: 0} if k_nulls is None else {2: k_nulls, 3: 0}
        return {
            "lower_bounds": lower,
            "upper_bounds": upper,
            "null_value_counts": nulls,
        }

    t = Table(
        "iceberg_v2_eq_key_ranges",
        [[(1, "id", "long"), (2, "k", "long"), (3, "s", "string")]],
    )
    columns = lambda ids, ks, ss: [  # noqa: E731
        (1, "id", pa.int64(), ids),
        (2, "k", pa.int64(), ks),
        (3, "s", pa.string(), ss),
    ]
    a = t.parquet("data-a.parquet", columns([1, 2, 3], [10, 20, 30], ["a", "b", "c"]))
    b = t.parquet("data-b.parquet", columns([4, 5, 6], [40, 50, None], ["d", "e", "f"]))
    t.commit(
        "append",
        [
            (
                "m-data-1.avro",
                0,
                0,
                [
                    (1, None, a, 0, None, None, metrics((10, 30), ("a", "c"))),
                    (
                        1,
                        None,
                        b,
                        0,
                        None,
                        None,
                        metrics((40, 50), ("d", "f"), k_nulls=1),
                    ),
                ],
            )
        ],
    )

    def delete(seq, files):
        """files: [(key columns, equality ids, metrics or None)], one delete file each."""
        entries = []
        for n, (keys, ids, file_metrics) in enumerate(files):
            file = t.parquet(f"delete-eq-{seq}-{n}.parquet", keys)
            entries.append((1, None, file, 2, None, ids, file_metrics))
        t.commit("delete", [(f"m-delete-{seq}.avro", 1, 0, entries)], sibling=True)

    k = lambda values: [(2, "k", pa.int64(), values)]  # noqa: E731
    delete(2, [(k([100, 110]), [2], metrics((100, 110)))])
    delete(3, [(k([30, 35]), [2], metrics((30, 35)))])
    delete(4, [(k([None, 100]), [2], metrics((100, 100), k_nulls=1))])
    delete(5, [(k([20]), [2], None)])
    delete(
        6,
        [
            (
                k([20]) + [(3, "s", pa.string(), ["zz"])],
                [2, 3],
                metrics((20, 20), ("zz", "zz")),
            )
        ],
    )
    delete(
        7,
        [(k([20]), [2], metrics((20, 20))), (k([45]), [2], metrics((45, 45)))],
    )
    t.bare_manifests["m-delete-8.avro"] = BOUNDS_FIELDS
    delete(8, [(k([20]), [2], None)])
    delete(9, [(k([None, 100]), [2], metrics((100, 100), k_nulls=None))])
    # truncate(16) keeps a prefix as the lower bound and rounds the upper bound's prefix up.
    delete(
        10,
        [
            (
                [(3, "s", pa.string(), ["z" * 20])],
                [3],
                metrics(s=("z" * 16, "z" * 15 + "{")),
            )
        ],
    )
    delete(
        11,
        [
            (k([100, 110]), [2], metrics((100, 110))),
            (k([20, 50]), [2], metrics((20, 50))),
        ],
    )
    delete(
        12,
        [
            (
                [(3, "s", pa.string(), ["z" * 2000])],
                [3],
                metrics(s=("z" * 2000, "z" * 2000)),
            )
        ],
    )
    delete(
        13,
        [
            (
                [(3, "s", pa.string(), ["z" * 60 + str(n)])],
                [3],
                metrics(s=("z" * 60 + str(n), "z" * 60 + str(n))),
            )
            for n in range(6)
        ],
    )
    delete(
        14,
        [
            (
                k([20]) + [(3, "s", pa.string(), ["zz"])],
                [2, 3],
                metrics(s=("zz", "zz")),
            )
        ],
    )
    delete(
        15,
        [
            (
                k([None, 20]) + [(3, "s", pa.string(), ["zz", "zz"])],
                [2, 3],
                metrics((20, 20), ("zz", "zz"), k_nulls=1),
            )
        ],
    )
    # A manifest no snapshot lists, whose only entry is deleted. Its name holds a literal %20:
    # a reader that decodes it takes the name for that of a manifest with a space.
    t._manifest("m%20data.avro", 0, 0, [(2, None, a, 0, None, None)])
    t.finish()


def build_partition_bounds():
    """Two data files of partition A; a delete stored in A and one stored unpartitioned."""
    import pyarrow as pa

    spec = [
        {"name": "region", "transform": "identity", "source-id": 2, "field-id": 1000}
    ]
    t = Table(
        "iceberg_v2_eq_partition_bounds",
        [[(1, "id", "long"), (2, "region", "string")]],
        [spec, []],
    )

    def bounds(low, high):
        return {
            "lower_bounds": {1: struct.pack("<q", low)},
            "upper_bounds": {1: struct.pack("<q", high)},
            "null_value_counts": {1: 0},
        }

    def data(name, ids):
        return t.parquet(
            name,
            [(1, "id", pa.int64(), ids), (2, "region", pa.string(), ["A"] * 3)],
        )

    a, b = data("data-a.parquet", [1, 2, 3]), data("data-b.parquet", [4, 5, 6])
    t.commit(
        "append",
        [
            (
                "m-data-1.avro",
                0,
                0,
                [
                    (1, None, a, 0, {"region": "A"}, None, bounds(1, 3)),
                    (1, None, b, 0, {"region": "A"}, None, bounds(4, 6)),
                ],
            )
        ],
    )
    in_a = t.parquet("delete-eq-2-A.parquet", [(1, "id", pa.int64(), [2])])
    t.commit(
        "delete",
        [
            (
                "m-delete-2.avro",
                1,
                0,
                [(1, None, in_a, 2, {"region": "A"}, [1], bounds(2, 2))],
            )
        ],
    )
    everywhere = t.parquet("delete-eq-3-global.parquet", [(1, "id", pa.int64(), [5])])
    t.commit(
        "delete",
        [
            (
                "m-delete-3.avro",
                1,
                1,
                [(1, None, everywhere, 2, None, [1], bounds(5, 5))],
            )
        ],
    )
    t.finish()


def build_manifest_shapes(name="iceberg_v2_eq_manifest_shapes", bare_first=False):
    """Data file a in a manifest with every metric map, data file b in one without null counts."""
    import pyarrow as pa

    t = Table(name, [[(1, "id", "long"), (2, "k", "long")]])
    long = lambda value: struct.pack("<q", value)  # noqa: E731

    def data(file_name, ids, ks):
        return t.parquet(
            file_name, [(1, "id", pa.int64(), ids), (2, "k", pa.int64(), ks)]
        )

    full = {
        "lower_bounds": {2: long(10)},
        "upper_bounds": {2: long(30)},
        "value_counts": {2: 3},
        "null_value_counts": {2: 0},
        "nan_value_counts": {2: 0},
    }
    # Read by position, the zero NaN count sits where the other manifest keeps its null count.
    bare = {
        "lower_bounds": {2: long(40)},
        "upper_bounds": {2: long(50)},
        "value_counts": {2: 3},
        "nan_value_counts": {2: 0},
    }
    files = [
        (data("data-a.parquet", [1, 2, 3], [10, 20, 30]), full),
        (data("data-b.parquet", [4, 5, 6], [40, 50, None]), bare),
    ]
    for n, (file, metrics) in enumerate(reversed(files) if bare_first else files, 1):
        manifest = f"m-data-{n}.avro"
        if metrics is bare:
            t.bare_manifests[manifest] = ("null_value_counts",)
        t.commit(
            "append", [(manifest, 0, 0, [(1, None, file, 0, None, None, metrics)])]
        )
    delete = t.parquet("delete-eq-3.parquet", [(2, "k", pa.int64(), [None, 100])])
    recorded = {
        "lower_bounds": {2: long(100)},
        "upper_bounds": {2: long(100)},
        "null_value_counts": {2: 1},
    }
    t.commit(
        "delete",
        [("m-delete-3.avro", 1, 0, [(1, None, delete, 2, None, [2], recorded)])],
    )
    t.finish()


def _long_bounds(low, high):
    """Bounds of the long field 2 without NULLs, as a manifest records them."""
    return {
        "lower_bounds": {2: struct.pack("<q", low)},
        "upper_bounds": {2: struct.pack("<q", high)},
        "null_value_counts": {2: 0},
    }


def build_duplicate_entry():
    """One data file listed twice; the second entry's bounds are not the file's."""
    import pyarrow as pa

    t = Table("iceberg_v2_eq_duplicate_entry", [[(1, "id", "long"), (2, "k", "long")]])
    a = t.parquet(
        "data-a.parquet",
        [(1, "id", pa.int64(), [1, 2, 3]), (2, "k", pa.int64(), [10, 20, 30])],
    )
    entries = [
        (1, None, a, 0, None, None, _long_bounds(10, 30)),
        (1, None, a, 0, None, None, _long_bounds(200, 300)),
    ]
    t.commit("append", [("m-data-1.avro", 0, 0, entries)])
    # The first delete matches the file. The file's own bounds rule the second one out.
    deletes = [
        (t.parquet("delete-eq-2-0.parquet", [(2, "k", pa.int64(), [20])]), (20, 20)),
        (t.parquet("delete-eq-2-1.parquet", [(2, "k", pa.int64(), [100])]), (100, 110)),
    ]
    t.commit(
        "delete",
        [
            (
                "m-delete-2.avro",
                1,
                0,
                [
                    (1, None, file, 2, None, [2], _long_bounds(*bounds))
                    for file, bounds in deletes
                ],
            )
        ],
    )
    t.finish()


# Entries of the data manifest of iceberg_v2_eq_bounds_stream. With 1024-byte bounds of an
# unrelated column they come to about 13 MiB, past the 10 MiB a streamed result buffers.
STREAM_ENTRIES = 6144


def build_bounds_stream():
    """A data manifest that a streamed read hands out in parts."""
    import pyarrow as pa

    t = Table(
        "iceberg_v2_eq_bounds_stream",
        [[(1, "id", "long"), (2, "k", "long"), (3, "s", "string")]],
    )
    recorded = _long_bounds(10, 30)
    recorded["lower_bounds"][3] = b"a" * 1024
    recorded["upper_bounds"][3] = b"c" * 1024

    def unwritten(name):
        return (str(t.rel / "data" / name), 3, 1000)

    t.commit(
        "append",
        [
            (
                "m-data-1.avro",
                0,
                0,
                [
                    (
                        1,
                        None,
                        unwritten(f"data-{n:04d}.parquet"),
                        0,
                        None,
                        None,
                        recorded,
                    )
                    for n in range(STREAM_ENTRIES)
                ],
            )
        ],
    )
    delete = t.parquet("delete-eq-2.parquet", [(2, "k", pa.int64(), [100])])
    t.commit(
        "delete",
        [
            (
                "m-delete-2.avro",
                1,
                0,
                [(1, None, delete, 2, None, [2], _long_bounds(100, 110))],
            )
        ],
    )
    t.finish()


def build_own_file_index():
    """One data file in a manifest with a column of its own named file_index."""
    import pyarrow as pa

    t = Table("iceberg_v2_eq_own_file_index", [[(1, "id", "long"), (2, "k", "long")]])
    a = t.parquet(
        "data-a.parquet",
        [(1, "id", pa.int64(), [1, 2, 3]), (2, "k", pa.int64(), [10, 20, 30])],
    )
    t.indexed_manifests["m-data-1.avro"] = 0
    t.commit(
        "append",
        [("m-data-1.avro", 0, 0, [(1, None, a, 0, None, None, _long_bounds(10, 30))])],
    )
    delete = t.parquet("delete-eq-2.parquet", [(2, "k", pa.int64(), [100])])
    entry = (1, None, delete, 2, None, [2], _long_bounds(100, 110))
    t.commit("delete", [("m-delete-2.avro", 1, 0, [entry])])
    t.finish()


def build_mixed_manifests():
    """A data manifest of format version 1 next to one of version 2 whose name sorts first."""
    import pyarrow as pa

    t = Table("iceberg_v2_eq_mixed_manifests", [[(1, "id", "long"), (2, "k", "long")]])

    def data(name, ids, ks):
        return t.parquet(name, [(1, "id", pa.int64(), ids), (2, "k", pa.int64(), ks)])

    files = [
        ("m-data-2.avro", data("data-b.parquet", [4, 5, 6], [40, 50, 60]), (40, 60)),
        ("m-data-1.avro", data("data-a.parquet", [1, 2, 3], [10, 20, 30]), (10, 30)),
    ]
    t.v1_manifests.add("m-data-2.avro")
    for manifest, file, bounds in files:
        entry = (1, None, file, 0, None, None, _long_bounds(*bounds))
        t.commit("append", [(manifest, 0, 0, [entry])])
    delete = t.parquet("delete-eq-3.parquet", [(2, "k", pa.int64(), [100])])
    entry = (1, None, delete, 2, None, [2], _long_bounds(100, 110))
    t.commit("delete", [("m-delete-3.avro", 1, 0, [entry])])
    t.finish()


def build_repeated_bound(name="iceberg_v2_eq_repeated_bound", in_delete=False):
    """Bounds maps that list the key twice, in the data file's entry or in the delete's. No
    pair is the file's: whichever a reader takes, it skips a delete that matches."""
    import pyarrow as pa

    t = Table(name, [[(1, "id", "long"), (2, "k", "long")]])
    long = lambda value: struct.pack("<q", value)  # noqa: E731
    a = t.parquet(
        "data-a.parquet",
        [(1, "id", pa.int64(), [1, 2, 3]), (2, "k", pa.int64(), [10, 20, 30])],
    )
    repeated = {
        "lower_bounds": [(2, long(200)), (2, long(400))],
        "upper_bounds": [(2, long(300)), (2, long(500))],
        "null_value_counts": {2: 0},
    }
    data_metrics = _long_bounds(10, 30) if in_delete else repeated
    delete_metrics = repeated if in_delete else _long_bounds(20, 20)
    t.commit(
        "append", [("m-data-1.avro", 0, 0, [(1, None, a, 0, None, None, data_metrics)])]
    )
    delete = t.parquet("delete-eq-2.parquet", [(2, "k", pa.int64(), [20])])
    t.commit(
        "delete",
        [("m-delete-2.avro", 1, 0, [(1, None, delete, 2, None, [2], delete_metrics)])],
    )
    t.finish()


# name -> (builder, expectation). The expectation is the rows a correct reader returns for the
# current snapshot, a {snapshot sequence: rows} map for a table of sibling snapshots, or None
# for a table that is not checked against DuckDB: one it cannot read from this machine, or one
# only a plan reads.
FIXTURES = {
    "iceberg_v2_eq_wide_delete_file": (
        build_wide_delete_file,
        [("apple", 1), ("cherry", 3), ("elderberry", 5)],
    ),
    "iceberg_v2_eq_with_positional": (
        build_with_positional,
        [("apple", 1), ("cherry", 3), ("elderberry", 5)],
    ),
    "iceberg_v2_eq_manifest_rewrite": (
        build_manifest_rewrite,
        [("apple", 1), ("banana", 4), ("cherry", 3), ("date", 5)],
    ),
    "iceberg_v2_eq_null_key": (
        build_null_key,
        [("apple", 1), ("cherry", 3), (None, 4)],
    ),
    "iceberg_v2_eq_partitioned": (
        build_partitioned,
        [(1, "B", 30), (3, "A", 50), (3, "B", 60)],
    ),
    "iceberg_v2_eq_partition_bounds": (
        build_partition_bounds,
        [(1, "A"), (3, "A"), (4, "A"), (6, "A")],
    ),
    "iceberg_v2_eq_partition_shapes": (
        build_partition_shapes,
        [
            (i, r, k, 10 * n + i)
            for n, r, k, gone in PARTITION_SHAPES
            for i in (1, 2, 3)
            if i != gone
        ],
    ),
    "iceberg_v2_eq_v1_data_manifest": (
        lambda: build_partitioned("iceberg_v2_eq_v1_data_manifest", True),
        [(1, "B", 30), (3, "A", 50), (3, "B", 60)],
    ),
    # Sequence 3 lists DuckDB's rows: it deletes id 1 in -0.0 as well as in +0.0.
    "iceberg_v2_eq_float_partition": (
        build_float_partition,
        {
            2: [(1,), (2,), (3,), (1,), (2,), (3,), (2,), (3,)],
            3: [(2,), (3,), (2,), (3,), (1,), (2,), (3,)],
        },
    ),
    "iceberg_v2_eq_upsert_same_commit": (
        build_upsert_same_commit,
        [(1, "a"), (2, "y"), (3, "c"), (4, "d")],
    ),
    "iceberg_v2_eq_renamed_key": (build_renamed_key, [("apple", 1), ("cherry", 3)]),
    "iceberg_v2_eq_promoted_key": (
        build_promoted_key,
        [(1, "a"), (2, "b2"), (3, "c"), (6, "f")],
    ),
    "iceberg_v2_eq_double_key": (build_double_key, [("apple", 1.5), ("cherry", 3.5)]),
    "iceberg_v2_eq_typed_keys": (
        build_typed_keys,
        {seq: [(1,), (3,)] for seq in TYPED_KEYS},
    ),
    "iceberg_v3_eq_timestamp_ns_key": (build_timestamp_ns_key, {2: [(1,), (3,)]}),
    "iceberg_v2_eq_unsupported_keys": (
        build_unsupported_keys,
        {seq: [(1,), (3,)] for seq in UNSUPPORTED_KEYS},
    ),
    "iceberg_v3_eq_no_row_lineage": (
        build_no_row_lineage,
        [("apple", 1), ("cherry", 3)],
    ),
    "iceberg_v2_eq_remote_delete": (
        lambda: build_unread_delete(
            "iceberg_v2_eq_remote_delete",
            "http://sirius-equality-delete.invalid/delete-eq-2.parquet",
        ),
        None,
    ),
    "iceberg_v2_eq_avro_delete": (
        lambda: build_unread_delete(
            "iceberg_v2_eq_avro_delete",
            str(REL / "iceberg_v2_eq_avro_delete/data/delete-eq-2.avro"),
            "AVRO",
        ),
        None,
    ),
    "iceberg_v2_eq_avro_delete_older": (
        lambda: build_unread_delete(
            "iceberg_v2_eq_avro_delete_older",
            str(REL / "iceberg_v2_eq_avro_delete_older/data/delete-eq-1.avro"),
            "AVRO",
            applies=False,
        ),
        [("apple", 1), ("banana", 2), ("cherry", 3)],
    ),
    "iceberg_v2_eq_key_ranges": (
        build_key_ranges,
        {seq: [(i,) for i in ids] for seq, ids in KEY_RANGES.items()},
    ),
    "iceberg_v2_eq_manifest_shapes": (
        build_manifest_shapes,
        [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)],
    ),
    "iceberg_v2_eq_manifest_shapes_bare_first": (
        lambda: build_manifest_shapes(
            "iceberg_v2_eq_manifest_shapes_bare_first", bare_first=True
        ),
        [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)],
    ),
    "iceberg_v2_eq_duplicate_entry": (build_duplicate_entry, None),
    "iceberg_v2_eq_mixed_manifests": (
        build_mixed_manifests,
        [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)],
    ),
    "iceberg_v2_eq_own_file_index": (
        build_own_file_index,
        [(1, 10), (2, 20), (3, 30)],
    ),
    # Not checked: DuckDB's reader takes one of the pairs and skips the delete.
    "iceberg_v2_eq_repeated_bound": (build_repeated_bound, None),
    "iceberg_v2_eq_repeated_delete_bound": (
        lambda: build_repeated_bound("iceberg_v2_eq_repeated_delete_bound", True),
        None,
    ),
    "iceberg_v2_eq_bounds_stream": (build_bounds_stream, None),
}


def verify(name, expected):
    """Check the fixture against DuckDB's Iceberg reader -- the oracle the GPU path is held to."""
    import duckdb

    if expected is None:
        print(f"  {name}: skipped - not checked against DuckDB")
        return True
    con = duckdb.connect()
    con.execute("LOAD iceberg")
    key = lambda row: tuple((v is None, v) for v in row)  # noqa: E731
    scans = {"": ("*", expected)}
    if isinstance(expected, dict):
        scans = {
            f", snapshot_from_id = {8800000000000000 + seq}": ("id", rows)
            for seq, rows in expected.items()
        }
    ok = True
    for selector, (columns, want) in scans.items():
        try:
            rows = con.sql(
                f"SELECT {columns} FROM iceberg_scan('{REL / name}'{selector})"
            ).fetchall()
        except duckdb.Error as error:
            rows = [("ERROR", str(error).splitlines()[0])]
        if sorted(rows, key=key) != sorted(want, key=key):
            print(
                f"  {name}{selector}: FAIL - got {sorted(rows, key=key)}",
                file=sys.stderr,
            )
            ok = False
        else:
            print(f"  {name}{selector}: OK - {len(rows)} rows")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify", action="store_true", help="validate only, write nothing"
    )
    ap.add_argument(
        "--out", required=True, help="directory the fixtures are written to"
    )
    ap.add_argument(
        "--no-verify", action="store_true", help="skip the check against DuckDB"
    )
    ap.add_argument("names", nargs="*", help="fixtures to build (default: all)")
    args = ap.parse_args()
    if unknown := [name for name in args.names if name not in FIXTURES]:
        ap.error(f"unknown fixtures: {', '.join(unknown)}")
    selected = {n: f for n, f in FIXTURES.items() if not args.names or n in args.names}
    # Paths in the metadata are absolute, so the tables read from any working directory.
    global DATA, REL
    DATA = REL = pathlib.Path(args.out).resolve()
    DATA.mkdir(parents=True, exist_ok=True)
    if not args.verify:
        for name, (build, _) in selected.items():
            build()
            print(f"wrote {name}")
        print("validating:")
    if args.no_verify:
        return 0
    ok = True
    for name, (_, expected) in selected.items():
        ok &= verify(name, expected)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
