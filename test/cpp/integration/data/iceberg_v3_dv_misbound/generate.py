#!/usr/bin/env python3
"""Build a fixture whose manifest points each deletion vector at the OTHER file's vector.

Run from the repo root:
    python3 test/cpp/integration/data/iceberg_v3_dv_misbound/generate.py
    python3 test/cpp/integration/data/iceberg_v3_dv_misbound/generate.py --verify

What it encodes
---------------
One Puffin file holds a deletion vector for each of two data files -- the normal shape, since a
commit writes one Puffin and puts every vector it produced inside. Locating a vector therefore
means trusting `content_offset`, and this fixture is the case where that trust is misplaced: the
two manifest entries carry each other's offsets.

Every check that does not read the Puffin footer passes:

    Puffin container magic   both offsets land inside a real Puffin file
    blob magic + CRC-32      both blobs are well-formed deletion vectors
    record_count             both vectors delete exactly ONE position, so the counts agree

The vectors delete different positions from data files with different contents, so the wrong
answer is a different SET of rows rather than a different number of them -- there is no count
anywhere that disagrees. Only the footer's `referenced-data-file` reveals the swap, which is why
the Iceberg spec requires `content_offset`/`content_size_in_bytes` to match the footer descriptor
exactly rather than treating the manifest as sufficient on its own.

    correct  A drops 'banana' (pos 1), B drops 'jackfruit' (pos 4)  -> 8 rows
    misbound A drops 'elderberry' (pos 4), B drops 'grape' (pos 1)  -> 8 rows, a DIFFERENT eight

Both vectors reference DIFFERENT data files, so unlike the two-vector case in
iceberg_v3_dv_replaced they can share one container without colliding in readers that key
decoded vectors by `referenced-data-file`.
"""

import argparse
import json
import pathlib
import shutil
import struct
import sys
import zlib

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "iceberg_v3_deletion_vector"
SRC_NAME = SRC.name
DST_NAME = HERE.name

PUFFIN_MAGIC = b"PFA1"
DV_MAGIC = bytes([0xD1, 0xD3, 0x39, 0x64])

# Second data file's contents. Distinct from the first file's so that applying the wrong vector
# changes WHICH rows survive; identical contents would make the swap unobservable in principle.
FILE_B_FRUIT = ["fig", "grape", "honeydew", "indian fig", "jackfruit"]
FILE_B_COUNT = [6, 7, 8, 9, 10]

# One position each: equal cardinality is what makes record_count blind to the swap.
POSITIONS_A = [1]  # 'banana'
POSITIONS_B = [4]  # 'jackfruit'

STATUS_ADDED = 1


def roaring64(positions):
    """Serialize a portable 64-bit Roaring bitmap; see iceberg_v3_dv_replaced/generate.py."""
    by_high = {}
    for p in sorted(set(positions)):
        by_high.setdefault(p >> 32, []).append(p & 0xFFFFFFFF)

    out = struct.pack("<q", len(by_high))
    for high, values in sorted(by_high.items()):
        out += struct.pack("<I", high)
        containers = {}
        for v in values:
            containers.setdefault(v >> 16, []).append(v & 0xFFFF)
        keys = sorted(containers)

        out += struct.pack("<I", 12346) + struct.pack("<I", len(keys))
        for k in keys:
            out += struct.pack("<HH", k, len(containers[k]) - 1)

        offset = 4 + 4 + 4 * len(keys) + 4 * len(keys)
        for k in keys:
            out += struct.pack("<I", offset)
            offset += 2 * len(containers[k])

        for k in keys:
            for v in containers[k]:
                out += struct.pack("<H", v)
    return out


def dv_blob(positions):
    """deletion-vector-v1: [4B BE length of magic+vector][magic][vector][4B BE CRC-32]."""
    body = DV_MAGIC + roaring64(positions)
    return (
        struct.pack(">I", len(body))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def descriptor_for(offset, blob, referenced_data_file, cardinality):
    """`snapshot-id`/`sequence-number` are -1 and `cardinality` is a string, both per the Puffin
    spec's deletion-vector-v1 rules."""
    return {
        "type": "deletion-vector-v1",
        "fields": [],
        "snapshot-id": -1,
        "sequence-number": -1,
        "offset": offset,
        "length": len(blob),
        "properties": {
            "referenced-data-file": referenced_data_file,
            "cardinality": str(cardinality),
        },
    }


def build_puffin(blobs, descriptors):
    footer = json.dumps(
        {"blobs": descriptors, "properties": {}}, separators=(",", ":")
    ).encode()
    return (
        PUFFIN_MAGIC
        + b"".join(blobs)
        + PUFFIN_MAGIC
        + footer
        + struct.pack("<i", len(footer))
        + b"\x00\x00\x00\x00"  # flags: bit 0 clear = uncompressed footer
        + PUFFIN_MAGIC
    )


def retarget(value):
    if isinstance(value, str):
        return value.replace(SRC_NAME, DST_NAME)
    return value


def rewrite_paths(obj):
    if isinstance(obj, dict):
        return {k: rewrite_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite_paths(v) for v in obj]
    return retarget(obj)


def verify():
    """Prove the fixture is caught ONLY by the footer: every other check must pass on it."""
    import fastavro
    from pyiceberg.table.puffin import PuffinFile

    ok = True
    puffins = sorted((HERE / "data").glob("*.puffin"))
    if len(puffins) != 1:
        print(f"  expected ONE Puffin file, found {len(puffins)}", file=sys.stderr)
        return False

    raw = puffins[0].read_bytes()
    blobs = PuffinFile(raw).footer.blobs
    if len(blobs) != 2:
        print(
            f"  expected 2 blobs in the container, found {len(blobs)}", file=sys.stderr
        )
        return False
    print(f"  pyiceberg: OK - {puffins[0].name}: 2 blob(s), type={blobs[0].type}")

    # pyiceberg decodes by referenced-data-file, i.e. it trusts the FOOTER. Its answer is the
    # correct pairing, which is what the manifest is supposed to agree with and does not.
    decoded = {k: sorted(v) for k, v in PuffinFile(raw).to_vector().items()}
    if len(decoded) != 2:
        print(
            f"  the two vectors must name different data files; got {decoded.keys()}",
            file=sys.stderr,
        )
        return False

    manifest = next((HERE / "metadata").glob("*-m1.avro"))
    with manifest.open("rb") as fh:
        entries = [
            r
            for r in fastavro.reader(fh)
            if (r["data_file"].get("file_format") or "").upper() == "PUFFIN"
        ]
    if len(entries) != 2:
        print(
            f"  expected 2 PUFFIN manifest entries, found {len(entries)}",
            file=sys.stderr,
        )
        return False

    counts = {e["data_file"]["record_count"] for e in entries}
    if counts != {1}:
        print(
            f"  cardinalities must be EQUAL or record_count catches the swap; got {counts}",
            file=sys.stderr,
        )
        ok = False
    else:
        print(
            "  record_count: OK - both entries claim 1, so the count check cannot fire"
        )

    # The swap itself: each entry's (offset, length) must be the OTHER blob's.
    by_offset = {b.offset: b for b in blobs}
    for entry in entries:
        df = entry["data_file"]
        blob = by_offset.get(df["content_offset"])
        if blob is None:
            print(
                f"  entry offset {df['content_offset']} matches no blob",
                file=sys.stderr,
            )
            ok = False
            continue
        pointed_at = blob.properties["referenced-data-file"]
        if pointed_at == df["referenced_data_file"]:
            print(
                f"  entry for {df['referenced_data_file']} points at its OWN vector; "
                "the fixture is not misbound",
                file=sys.stderr,
            )
            ok = False
        else:
            print(
                f"  misbound: entry for .../{pathlib.Path(df['referenced_data_file']).name} "
                f"points at the vector for .../{pathlib.Path(pointed_at).name}"
            )
        if blob.length != df["content_size_in_bytes"]:
            print(
                "  lengths must match the blob they point at, or the size check fires first",
                file=sys.stderr,
            )
            ok = False

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify", action="store_true", help="validate only, write nothing"
    )
    args = ap.parse_args()

    if args.verify:
        print(f"verifying {DST_NAME}")
        return 0 if verify() else 1

    import fastavro
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not SRC.is_dir():
        print(f"source fixture missing: {SRC}", file=sys.stderr)
        return 1

    # Rebuild from the sibling fixture each run so the two cannot drift apart.
    for child in HERE.iterdir():
        if child.name != pathlib.Path(__file__).name:
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    for sub in ("data", "metadata"):
        shutil.copytree(SRC / sub, HERE / sub)
    for stray in list((HERE / "data").glob("*.bak")) + list(
        (HERE / "metadata").glob("*.bak")
    ):
        stray.unlink()
    for stray in list((HERE / "data").glob("*.py")) + list(
        (HERE / "metadata").glob("*.py")
    ):
        stray.unlink()

    meta_path = HERE / "metadata" / "v1.metadata.json"
    meta = rewrite_paths(json.loads(meta_path.read_text()))
    meta_path.write_text(json.dumps(meta, indent=2))

    for avro_path in sorted((HERE / "metadata").glob("*.avro")):
        with avro_path.open("rb") as fh:
            reader = fastavro.reader(fh)
            schema, records = reader.writer_schema, [rewrite_paths(r) for r in reader]
        with avro_path.open("wb") as fh:
            fastavro.writer(fh, schema, records, codec="null")

    # --- a second data file, reusing the first file's schema so the field ids survive --------
    data_dir = HERE / "data"
    file_a = next(data_dir.glob("*.parquet"))
    table_a = pq.read_table(file_a)
    file_b = data_dir / file_a.name.replace("00000-0-", "00001-0-")
    pq.write_table(
        pa.table(
            [pa.array(FILE_B_FRUIT), pa.array(FILE_B_COUNT, type=pa.int64())],
            schema=table_a.schema,
        ),
        file_b,
    )

    rel = (
        lambda p: f"{HERE.relative_to(pathlib.Path.cwd())}/data/{p.name}"
    )  # noqa: E731

    # --- the data manifest gains an entry for the second file --------------------------------
    data_manifest = next((HERE / "metadata").glob("*-m0.avro"))
    with data_manifest.open("rb") as fh:
        reader = fastavro.reader(fh)
        m0_schema, m0_records = reader.writer_schema, list(reader)
    entry_b = json.loads(json.dumps(m0_records[0]))  # deep copy through plain types
    entry_b["data_file"]["file_path"] = rel(file_b)
    entry_b["data_file"]["file_size_in_bytes"] = file_b.stat().st_size
    m0_records.append(entry_b)
    with data_manifest.open("wb") as fh:
        fastavro.writer(fh, m0_schema, m0_records, codec="null")
    print(f"wrote {data_manifest.name}: {len(m0_records)} data files")

    # --- one container, both vectors ---------------------------------------------------------
    blob_a, blob_b = dv_blob(POSITIONS_A), dv_blob(POSITIONS_B)
    offset_a = len(PUFFIN_MAGIC)
    offset_b = offset_a + len(blob_a)
    desc_a = descriptor_for(offset_a, blob_a, rel(file_a), len(POSITIONS_A))
    desc_b = descriptor_for(offset_b, blob_b, rel(file_b), len(POSITIONS_B))

    old_puffin = next(data_dir.glob("*.puffin"))
    puffin = data_dir / old_puffin.name
    puffin.write_bytes(build_puffin([blob_a, blob_b], [desc_a, desc_b]))
    print(f"wrote {puffin.name}: 2 blobs at offsets {offset_a} and {offset_b}")

    # --- the delete manifest, with the two entries' offsets SWAPPED --------------------------
    delete_manifest = next((HERE / "metadata").glob("*-m1.avro"))
    with delete_manifest.open("rb") as fh:
        reader = fastavro.reader(fh)
        m1_schema, m1_records = reader.writer_schema, list(reader)
    template = next(
        r
        for r in m1_records
        if (r["data_file"].get("file_format") or "").upper() == "PUFFIN"
    )

    def dv_entry(referenced, descriptor):
        entry = json.loads(json.dumps(template))
        entry["status"] = STATUS_ADDED
        entry["data_file"]["file_path"] = rel(puffin)
        entry["data_file"]["file_size_in_bytes"] = puffin.stat().st_size
        entry["data_file"]["referenced_data_file"] = referenced
        entry["data_file"]["content_offset"] = descriptor["offset"]
        entry["data_file"]["content_size_in_bytes"] = descriptor["length"]
        entry["data_file"]["record_count"] = int(
            descriptor["properties"]["cardinality"]
        )
        return entry

    # Each entry gets the OTHER descriptor. Same length, same cardinality, wrong vector.
    m1_records = [
        r
        for r in m1_records
        if (r["data_file"].get("file_format") or "").upper() != "PUFFIN"
    ] + [dv_entry(rel(file_a), desc_b), dv_entry(rel(file_b), desc_a)]
    with delete_manifest.open("wb") as fh:
        fastavro.writer(fh, m1_schema, m1_records, codec="null")
    print(f"wrote {delete_manifest.name}: 2 PUFFIN entries, offsets swapped")

    print("validating:")
    return 0 if verify() else 1


if __name__ == "__main__":
    raise SystemExit(main())
