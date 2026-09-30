#!/usr/bin/env python3
"""Build fixtures whose ONLY delete entry has been retired by a later commit.

Run from the repo root:
    python3 test/cpp/integration/data/generate_retired_entry_fixtures.py
    python3 test/cpp/integration/data/generate_retired_entry_fixtures.py --verify

Produces two fixtures, both by copying an existing one and flipping its delete entry's
manifest `status` from 1 (ADDED) to 2 (DELETED):

  iceberg_v2_pos_delete_retired  <- iceberg_v2_delete
      A positional-delete file that a later commit retired. Nothing deletes any more, so the
      scan must return all 5 rows. A reader that ignores status still applies the retired
      file and returns 3.

  iceberg_v2_eq_delete_retired   <- iceberg_v2_equality_delete
      The table's only equality-delete file, retired. Equality deletes are the one thing the
      GPU path refuses outright, so the interesting assertion is the ROUTE: with no LIVE
      equality delete the table must stay GPU-eligible. A status-blind gate refuses it
      forever over a delete that no longer applies -- a silent performance cliff rather than
      a wrong answer, which is why nothing would otherwise catch it.

Hand-forged for the same reason as iceberg_v3_dv_replaced: pyiceberg refuses to write delete
files at all ("Merge on read is not yet supported, falling back to copy-on-write"), so the
conformance corpus cannot produce a table with any delete entry, live or retired.
"""

import argparse
import json
import pathlib
import shutil
import sys

DATA = pathlib.Path(__file__).resolve().parent

STATUS_DELETED = 2

# (destination, source, rows the table holds, rows the retired file would delete)
FIXTURES = [
    ("iceberg_v2_pos_delete_retired", "iceberg_v2_delete", 5, 2),
    ("iceberg_v2_eq_delete_retired", "iceberg_v2_equality_delete", 5, 2),
]


def rewrite(obj, src_name, dst_name):
    if isinstance(obj, dict):
        return {k: rewrite(v, src_name, dst_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, src_name, dst_name) for v in obj]
    if isinstance(obj, str):
        return obj.replace(src_name, dst_name)
    return obj


def build(dst_name, src_name):
    import fastavro

    src, dst = DATA / src_name, DATA / dst_name
    if not src.is_dir():
        raise SystemExit(f"source fixture missing: {src}")

    keep = dst / "README.md"
    saved = keep.read_text() if keep.exists() else None
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for stray in list(dst.rglob("*.bak")) + list(dst.rglob("*.py")):
        stray.unlink()
    if saved is not None:
        keep.write_text(saved)

    # Every avro carries repo-rooted paths -- the DATA manifest too. Missing one leaves the
    # delete entry pointing at the source fixture's files, which reads as a healthy table
    # rather than a broken fixture.
    for avro_path in sorted(dst.glob("metadata/*.avro")):
        with avro_path.open("rb") as fh:
            reader = fastavro.reader(fh)
            schema, records = reader.writer_schema, [
                rewrite(r, src_name, dst_name) for r in reader
            ]
        retired = 0
        for record in records:
            # content 0 is a DATA file; anything else is a delete entry.
            if "data_file" in record and record["data_file"]["content"] != 0:
                record["status"] = STATUS_DELETED
                retired += 1
        with avro_path.open("wb") as fh:
            fastavro.writer(fh, schema, records, codec="null")
        if retired:
            print(f"  {avro_path.name}: retired {retired} delete entry/entries")

    meta = dst / "metadata" / "v1.metadata.json"
    meta.write_text(
        json.dumps(rewrite(json.loads(meta.read_text()), src_name, dst_name), indent=2)
    )
    print(f"wrote {dst_name} from {src_name}")


def verify(dst_name, total_rows, would_delete):
    import fastavro
    from pyiceberg.table import StaticTable

    dst = DATA / dst_name
    statuses = []
    for avro_path in sorted(dst.glob("metadata/*.avro")):
        with avro_path.open("rb") as fh:
            for record in fastavro.reader(fh):
                if "data_file" in record and record["data_file"]["content"] != 0:
                    statuses.append(record["status"])
    if not statuses or any(s != STATUS_DELETED for s in statuses):
        print(
            f"  {dst_name}: FAIL - delete entries not all retired: {statuses}",
            file=sys.stderr,
        )
        return False

    rows = (
        StaticTable.from_metadata(str(dst / "metadata" / "v1.metadata.json"))
        .scan()
        .to_arrow()
        .num_rows
    )
    if rows != total_rows:
        blind = total_rows - would_delete
        hint = (
            "the retired entry was still applied"
            if rows == blind
            else "unexpected - check the fixture's paths were repointed"
        )
        print(
            f"  {dst_name}: FAIL - {rows} rows, expected {total_rows} ({hint})",
            file=sys.stderr,
        )
        return False

    print(
        f"  {dst_name}: OK - {rows} rows, retired entry ignored "
        f"(a status-blind reader would return {total_rows - would_delete})"
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify", action="store_true", help="validate only, write nothing"
    )
    args = ap.parse_args()

    if not args.verify:
        for dst_name, src_name, _, _ in FIXTURES:
            build(dst_name, src_name)
        print("validating:")

    ok = True
    for dst_name, _, total_rows, would_delete in FIXTURES:
        ok &= verify(dst_name, total_rows, would_delete)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
