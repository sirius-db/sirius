#!/usr/bin/env python3
"""Build fixtures for the three holes the (name, field id) schema probe used to have.

Run from the repo root:
    python3 test/cpp/integration/data/generate_schema_gate_fixtures.py
    python3 test/cpp/integration/data/generate_schema_gate_fixtures.py --verify

Both are `iceberg_v1` with only its DATA FILE rewritten -- same path, same row count, so every
manifest and metadata byte stays valid and the table still reads. Only the parquet footer differs,
which is the whole point: these are the two ways a data file can disagree with the table's schema
without any manifest saying so.

  iceberg_v1_no_field_ids   data file written with NO parquet field ids at all.
      Iceberg permits this through name mapping -- it is what tables migrated with `add_files`
      look like. The probe used to filter `WHERE field_id IS NOT NULL`, so such a file returned
      ZERO rows and was never checked; a file that contributed no rows is indistinguishable from
      one that was never probed, and it passed as GPU-safe. The scan then resolves its columns by
      NAME, which either throws at scan time (poisoning the connection) or silently reads a
      different current column that happens to share a physical name.

  iceberg_v1_promoted_type  `count` stored as INT32 while the table declares it `long`.
      Iceberg allows int -> long promotion, and it keeps BOTH the name and the field id, so a
      comparison of (name, field id) pairs reports the file as matching. Nothing throws: the GPU
      scan reads the file's own physical type and hands back a column narrower than the plan
      declared. The probe must therefore compare the TYPE as well.

  iceberg_v1_reordered      the same two fields, in the opposite physical order.
      Iceberg identifies fields by id, so a file may store them in any order and the table is
      entirely valid -- DuckDB reads it with BY_FIELD_ID and returns the snapshot's order. The
      probe compared MEMBERSHIP in a (name, field id) map, which a permutation satisfies exactly.
      For a full `SELECT *` the GPU path installs no reader projection, so cuDF emits columns in
      the FILE's order while the rest of the plan expects the bound snapshot's: the values come
      back under each other's names, and because the types here are castable nothing throws.

Hand-forged rather than produced by pyiceberg for the same reason as the retired-entry fixtures:
pyiceberg writes neither an id-less data file for a table whose schema has ids, nor a table left
mid-promotion. Both are states real engines produce and readers must handle.
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

DATA = pathlib.Path(__file__).resolve().parent
SOURCE = "iceberg_v1"

# Candidates in preference order. `duckdb/build/release/duckdb` is the trap: it is written by
# DuckDB's OWN build, survives every `/var/tmp` wipe because it lives inside the repo, and so goes
# stale by whole DuckDB versions while everything else moves on. A fixture written by one DuckDB
# version and read by a suite linked against another is not a fixture, it is a coincidence -- and
# the same stale binary, used as an oracle for whether a table needs a setting, will answer for a
# different iceberg extension build than the tests resolve. Hence require_duckdb() below.
DUCKDB_CANDIDATES = [
    pathlib.Path("build/release/duckdb"),
    pathlib.Path("duckdb/build/release/duckdb"),
]


def submodule_version() -> str:
    """The version the suite links against, from the duckdb submodule's own tag."""
    out = subprocess.run(
        ["git", "-C", "duckdb", "describe", "--tags"], capture_output=True, text=True
    )
    if out.returncode != 0:
        sys.exit(f"cannot read the duckdb submodule version:\n{out.stderr}")
    return out.stdout.strip().split("-")[0]


def require_duckdb() -> pathlib.Path:
    """Pick a CLI and REFUSE one whose version differs from the submodule's."""
    wanted = submodule_version()
    tried = []
    for candidate in DUCKDB_CANDIDATES:
        if not candidate.exists():
            tried.append(f"  {candidate} -- not built")
            continue
        out = subprocess.run(
            [str(candidate), "-noheader", "-list", "-c", "SELECT version();"],
            capture_output=True,
            text=True,
        )
        found = out.stdout.strip()
        if out.returncode == 0 and found == wanted:
            return candidate
        tried.append(
            f"  {candidate} -- reports {found or out.stderr.strip()!r}, want {wanted}"
        )
    sys.exit(
        "no DuckDB CLI matching the submodule ("
        + wanted
        + ") was found:\n"
        + "\n".join(tried)
        + "\n\nBuild one (`pixi run make release`) rather than using a stale binary: these "
        "fixtures are read by a suite linked against " + wanted + "."
    )


# (destination, SELECT list rewriting the data file, COPY options)
# Iceberg's own mechanism for reading a file that carries no field ids: the table property maps
# each field id to the physical names it may go by. Without it the spec answer for such a file is
# all-NULL, which is a degenerate table rather than the `add_files` migration this reproduces --
# and it would let the fixture pass for the wrong reason, since NULL rows differ from what a
# name-resolving scan returns.
NAME_MAPPING = json.dumps(
    [{"field-id": 1, "names": ["fruit"]}, {"field-id": 2, "names": ["count"]}]
)

# (destination, SELECT list rewriting the data file, COPY options, extra table properties)
FIXTURES = [
    (
        "iceberg_v1_no_field_ids",
        "fruit, count",
        "FORMAT PARQUET",
        {"schema.name-mapping.default": NAME_MAPPING},
    ),
    (
        "iceberg_v1_promoted_type",
        "fruit, count::INT AS count",
        "FORMAT PARQUET, FIELD_IDS {fruit: 1, count: 2}",
        {},
    ),
    (
        "iceberg_v1_reordered",
        "count, fruit",
        "FORMAT PARQUET, FIELD_IDS {fruit: 1, count: 2}",
        {},
    ),
]


def data_file(table: pathlib.Path) -> pathlib.Path:
    files = sorted((table / "data").glob("*.parquet"))
    if len(files) != 1:
        sys.exit(
            f"expected exactly one data file under {table}/data, found {len(files)}"
        )
    return files[0]


def duckdb_sql(sql: str) -> str:
    out = subprocess.run(
        [str(require_duckdb()), "-noheader", "-list", "-c", sql],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        sys.exit(f"duckdb failed:\n{out.stderr}")
    return out.stdout


def rewrite(obj, src_name: str, dst_name: str):
    """Repoint every repo-rooted path string from the source fixture to this one."""
    if isinstance(obj, dict):
        return {k: rewrite(v, src_name, dst_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, src_name, dst_name) for v in obj]
    if isinstance(obj, str):
        return obj.replace(src_name, dst_name)
    return obj


def write_metadata(path: pathlib.Path, doc) -> None:
    """Matches pre-commit's pretty-format-json --autofix, so regenerating stays lint-clean."""
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def repoint(dst: pathlib.Path, src_name: str, dst_name: str) -> None:
    """Rewrite the copied metadata so it names THIS fixture's files.

    Skipping this is the trap: `copytree` leaves every manifest and the metadata JSON pointing at
    the SOURCE fixture's data file, so the scan quietly reads the original — which has field ids
    and the wide type — and the fixture reports a healthy table instead of the defect it exists to
    reproduce. Caught here only because the tests assert the ROUTE; the rows were correct.
    """
    import fastavro

    for avro_path in sorted((dst / "metadata").glob("*.avro")):
        with avro_path.open("rb") as fh:
            reader = fastavro.reader(fh)
            schema = reader.writer_schema
            records = [rewrite(r, src_name, dst_name) for r in reader]
        with avro_path.open("wb") as fh:
            fastavro.writer(fh, schema, records, codec="null")

    for meta in sorted((dst / "metadata").glob("*.metadata.json")):
        write_metadata(meta, rewrite(json.loads(meta.read_text()), src_name, dst_name))

    # `src_name + "/"`, not `src_name`: the destination name has the source name as a prefix, so a
    # bare substring test matches every correctly-rewritten path too.
    stale = (src_name + "/").encode()
    leaked = [
        str(p)
        for p in sorted(dst.rglob("*"))
        if p.is_file() and stale in p.read_bytes()
    ]
    if leaked:
        sys.exit(f"{dst_name}: metadata still references {src_name}: {leaked}")


def build(dst_name: str, select: str, copy_opts: str, properties: dict) -> None:
    src = DATA / SOURCE
    dst = DATA / dst_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    repoint(dst, SOURCE, dst_name)

    if properties:
        for meta in sorted((dst / "metadata").glob("*.metadata.json")):
            doc = json.loads(meta.read_text())
            doc.setdefault("properties", {}).update(properties)
            write_metadata(meta, doc)

    target = data_file(dst)
    source = data_file(src)
    # Written beside the target and moved into place: COPY cannot read a file it is writing.
    staged = target.with_suffix(".staged.parquet")
    duckdb_sql(
        f"COPY (SELECT {select} FROM read_parquet('{source}')) "
        f"TO '{staged}' ({copy_opts});"
    )
    staged.replace(target)

    # The manifest records this file's record_count; a rewrite that changed it would make the
    # table inconsistent in a way that has nothing to do with what the fixture is testing.
    before = duckdb_sql(f"SELECT count(*) FROM read_parquet('{source}');").strip()
    after = duckdb_sql(f"SELECT count(*) FROM read_parquet('{target}');").strip()
    if before != after:
        sys.exit(f"{dst_name}: row count changed {before} -> {after}")
    print(f"built {dst_name} ({after} rows)")


def verify(dst_name: str) -> bool:
    target = data_file(DATA / dst_name)
    rows = duckdb_sql(
        "SELECT name, coalesce(field_id::VARCHAR, 'NULL'), duckdb_type "
        f"FROM parquet_schema('{target}') WHERE duckdb_type IS NOT NULL;"
    ).strip()
    print(f"{dst_name}:\n{rows}")
    if dst_name == "iceberg_v1_no_field_ids":
        ok = "NULL" in rows and "|1|" not in rows
    elif dst_name == "iceberg_v1_promoted_type":
        ok = "count|2|INTEGER" in rows
    else:
        # Order is the whole fixture, so assert the ORDER of the rows: parquet_schema() returns
        # the footer's flattened schema in its own preorder. Membership is deliberately unchanged.
        names = [line.split("|")[0] for line in rows.splitlines()]
        ok = names == ["count", "fruit"]
    print("  OK" if ok else "  UNEXPECTED")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify", action="store_true", help="inspect the built fixtures instead"
    )
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if all(verify(name) for name, *_ in FIXTURES) else 1)
    for name, select, opts, props in FIXTURES:
        build(name, select, opts, props)
