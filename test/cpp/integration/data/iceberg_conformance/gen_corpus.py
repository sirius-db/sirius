#!/usr/bin/env python3
"""Generate the Iceberg conformance corpus with the Apache reference implementation.

Every table here is written by pyiceberg (apache/iceberg-python). The expected
results in `expectations.json` are recorded from **pyiceberg's own scan**, never
from Sirius and never from DuckDB. That independence is the entire point: the 13
hand-built fixtures in the tree were green through three real bugs, including
silent wrong results, because they encoded the implementation's own assumption
that Iceberg columns resolve by name.

Regenerate with:

    python3 gen_corpus.py <output-dir>

Requires: pip install pyiceberg pyarrow sqlalchemy

Cases marked expect_gpu=False are ones the GPU scan path must DECLINE (cleanly, at
plan time) rather than answer. Cases marked expect_gpu=True must run on the GPU and
match the reference exactly.
"""
import json
import os
import shutil
import sys

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.types import IntegerType

OUT = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/iceberg_conformance"

# A RELATIVE output path makes pyiceberg record relative paths in both the metadata
# JSON and inside the binary .avro manifests - which is what lets the corpus be
# committed and resolved against the repo root, exactly like the 13 hand-built
# fixtures. Run from the repo root when generating in-tree:
#
#   python3 gen_corpus.py test/cpp/integration/data/iceberg_conformance
#
# The `file_uri` case is the one exception: its whole point is a manifest carrying an
# absolute file:// URI, which cannot be expressed relatively. It is generated only for
# an absolute OUT, and is covered in-tree by the strip_file_scheme unit tests instead.
IN_TREE = not os.path.isabs(OUT)

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

cases = []


def catalog_for(name, uri_style=False):
    """A catalog per case, so one case's warehouse layout can't affect another's."""
    root = os.path.join(OUT, name)
    os.makedirs(root, exist_ok=True)
    warehouse = f"file://{root}" if uri_style else root
    cat = SqlCatalog(name, **{"uri": f"sqlite:///{root}/catalog.db", "warehouse": warehouse})
    cat.create_namespace("conf")
    return cat, root


def record(name, table, root, expect_gpu, why):
    """Record pyiceberg's OWN result as the expectation. Never our engine's."""
    ref = table.scan().to_arrow().to_pydict()
    # Column order from the current schema, so the runner can compare positionally.
    cols = [f.name for f in table.schema().fields]
    rows = [[ref[c][i] for c in cols] for i in range(len(ref[cols[0]]))] if cols else []
    cases.append(
        {
            "name": name,
            "table_dir": os.path.join(root, "conf.db", name)
            if os.path.isdir(os.path.join(root, "conf.db", name))
            else os.path.join(root, "conf", name),
            "columns": cols,
            "field_ids": {f.name: f.field_id for f in table.schema().fields},
            "expected_rows": rows,
            "expect_gpu": expect_gpu,
            "why": why,
        }
    )
    print(f"  {name}: {len(rows)} rows, fields={ {f.name: f.field_id for f in table.schema().fields} }")


# ---------------------------------------------------------------- append_only
# Baseline: no evolution, no deletes. Must run on the GPU and match exactly.
# If THIS case ever declines, something in the gate is over-refusing.
cat, root = catalog_for("append_only")
s = pa.schema([("id", pa.int32()), ("name", pa.string())])
t = cat.create_table("conf.append_only", schema=s)
t.append(pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}, schema=s))
record("append_only", cat.load_table("conf.append_only"), root, True,
       "no schema evolution, no deletes - the GPU path's happy case")

# ---------------------------------------------------------------- drop_readd
# The silent-wrong-results case. `y` is dropped and re-added under the SAME name,
# so it is a NEW field id (4). The old data file holds the ORIGINAL y at field id 3.
# Per spec the new column must read NULL for those rows. Name resolution finds the
# old column and returns 111/222 - wrong, with no error and no fallback.
cat, root = catalog_for("drop_readd")
s = pa.schema([("id", pa.int32()), ("x", pa.int32()), ("y", pa.int32())])
t = cat.create_table("conf.drop_readd", schema=s)
t.append(pa.table({"id": [1, 2], "x": [10, 20], "y": [111, 222]}, schema=s))
with t.update_schema() as u:
    u.delete_column("y")
with t.update_schema() as u:
    u.add_column("y", IntegerType())
record("drop_readd", cat.load_table("conf.drop_readd"), root, False,
       "dropped and re-added column: old file's y is a DIFFERENT field id, must read NULL")

# ---------------------------------------------------------------- rename_col
# Renaming keeps the field id and changes the name. The old data file carries the
# ORIGINAL name. Name resolution fails loudly here ("Projected column 'value' not
# found") -> runtime fallback -> the engine deadlock.
cat, root = catalog_for("rename_col")
s1 = pa.schema([("id", pa.int32()), ("val", pa.string())])
t = cat.create_table("conf.rename_col", schema=s1)
t.append(pa.table({"id": [1, 2], "val": ["old_a", "old_b"]}, schema=s1))
with t.update_schema() as u:
    u.rename_column("val", "value")
s2 = pa.schema([("id", pa.int32()), ("value", pa.string())])
t.append(pa.table({"id": [3, 4], "value": ["new_c", "new_d"]}, schema=s2))
record("rename_col", cat.load_table("conf.rename_col"), root, False,
       "renamed column: same field id, old file carries the original name")

# ---------------------------------------------------------------- add_column
# The case a max(field_id) > column_count pre-filter MISSES. A column added after
# the first data file was written is absent from that file and must read NULL.
cat, root = catalog_for("add_column")
s1 = pa.schema([("id", pa.int32()), ("a", pa.int32())])
t = cat.create_table("conf.add_column", schema=s1)
t.append(pa.table({"id": [1, 2], "a": [10, 20]}, schema=s1))
with t.update_schema() as u:
    u.add_column("b", IntegerType())
s2 = pa.schema([("id", pa.int32()), ("a", pa.int32()), ("b", pa.int32())])
t.append(pa.table({"id": [3, 4], "a": [30, 40], "b": [300, 400]}, schema=s2))
record("add_column", cat.load_table("conf.add_column"), root, False,
       "column added after the first data file: absent from that file, must read NULL")

# ---------------------------------------------------------------- file_uri
# Warehouse configured as a URI, so manifests record `file:///...` data-file paths -
# what Java/Spark writers emit. Un-stripped, this reaches the local reactor's
# "unsupported path" throw -> RUNTIME fallback -> next GPU query on the connection
# deadlocks. Schema is trivial on purpose: this case is about the PATH, not the schema.
if not IN_TREE:
    cat, root = catalog_for("file_uri", uri_style=True)
    s = pa.schema([("id", pa.int32()), ("name", pa.string())])
    t = cat.create_table("conf.file_uri", schema=s)
    t.append(pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}, schema=s))
    record("file_uri", cat.load_table("conf.file_uri"), root, True,
           "manifests carry file:// URIs (Java/Spark shape); must run on GPU, not fall back")
else:
    print("  file_uri: SKIPPED for in-tree generation (needs an absolute file:// URI;"
          " covered in-tree by the strip_file_scheme unit tests)")

with open(os.path.join(OUT, "expectations.json"), "w") as f:
    json.dump({"cases": cases}, f, indent=2)

print(f"\ncorpus: {OUT}")
print(f"expectations: {os.path.join(OUT, 'expectations.json')} ({len(cases)} cases)")
