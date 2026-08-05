#!/usr/bin/env python3
"""Rebuild this fixture's deletion vector as a spec-valid Puffin file.

The original fixture stored a bare deletion-vector blob with no Puffin container:
no PFA1 magic, no footer. Our reader accepted it because it only validates the
blob's own magic and CRC, and DuckDB's iceberg reader accepted it until it began
checking the container's trailing magic. It was never a valid Puffin file.

Puffin layout (https://iceberg.apache.org/puffin-spec/):

    Magic(4) | Blob... | Footer
    Footer = Magic(4) FooterPayload FooterPayloadSize(4, LE) Flags(4) Magic(4)

`offset` in the footer's blob metadata is absolute from the start of the file, so
wrapping the blob shifts it from 0 to 4 and the manifest's `content_offset` has to
move with it -- which is why this script rewrites the manifest too.

Run from the repo root:
    python3 test/cpp/integration/data/iceberg_v3_deletion_vector/regenerate_puffin.py

Then verify (both must pass -- see the module docstring's note on validators):
    python3 .../regenerate_puffin.py --verify
"""

import argparse
import json
import pathlib
import shutil
import struct
import sys

MAGIC = b"PFA1"
HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
METADATA = HERE / "metadata"

# The manifest that carries the deletion-vector entry (m1); m0 is the data manifest.
MANIFEST = METADATA / "d7e8f9a0-0007-0007-0007-000000000004-m1.avro"
PUFFIN = DATA / "dv-00000-0-d7e8f9a0-0007-0007-0007-000000000002-00001.puffin"
REFERENCED_DATA_FILE = (
    "test/cpp/integration/data/iceberg_v3_deletion_vector/data/"
    "00000-0-d7e8f9a0-0007-0007-0007-000000000001-00001.parquet"
)


def extract_blob(raw: bytes) -> bytes:
    """Return the deletion-vector blob, whether or not `raw` is already wrapped.

    Makes the script idempotent: re-running it must not nest one container inside
    another, which would still start and end with the right magic and so pass a
    naive check while pointing `offset` at garbage.
    """
    if raw[:4] == MAGIC and raw[-4:] == MAGIC:
        footer_size = struct.unpack("<i", raw[-12:-8])[0]
        footer = json.loads(raw[-(footer_size + 12) : -12])
        blob = footer["blobs"][0]
        return raw[blob["offset"] : blob["offset"] + blob["length"]]
    return raw


def build_puffin(blob: bytes, snapshot_id: int, sequence_number: int) -> bytes:
    offset = len(MAGIC)
    footer_payload = json.dumps(
        {
            "blobs": [
                {
                    "type": "deletion-vector-v1",
                    "fields": [],
                    "snapshot-id": snapshot_id,
                    "sequence-number": sequence_number,
                    "offset": offset,
                    "length": len(blob),
                    "properties": {"referenced-data-file": REFERENCED_DATA_FILE},
                }
            ],
            "properties": {},
        },
        separators=(",", ":"),
    ).encode()

    return (
        MAGIC
        + blob
        + MAGIC
        + footer_payload
        + struct.pack("<i", len(footer_payload))
        + b"\x00\x00\x00\x00"  # flags: bit 0 clear = uncompressed footer
        + MAGIC
    )


def verify(puffin_path: pathlib.Path) -> bool:
    """Validate against a reader we did not write. Self-consistency proves nothing."""
    raw = puffin_path.read_bytes()
    ok = True
    try:
        from pyiceberg.table.puffin import PuffinFile

        pf = PuffinFile(raw)
        blobs = pf.footer.blobs
        print(f"  pyiceberg: OK - {len(blobs)} blob(s), type={blobs[0].type}")
    except Exception as exc:  # noqa: BLE001 - report whatever the reader says
        print(f"  pyiceberg: FAIL - {exc}")
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--verify", action="store_true", help="validate only, write nothing"
    )
    args = ap.parse_args()

    if args.verify:
        print(f"verifying {PUFFIN.name}")
        return 0 if verify(PUFFIN) else 1

    import fastavro

    with MANIFEST.open("rb") as fh:
        reader = fastavro.reader(fh)
        schema = reader.writer_schema
        records = list(reader)

    dv_entries = [
        r for r in records if r["data_file"].get("file_format", "").upper() == "PUFFIN"
    ]
    if len(dv_entries) != 1:
        print(
            f"expected exactly one PUFFIN entry, found {len(dv_entries)}",
            file=sys.stderr,
        )
        return 1
    entry = dv_entries[0]

    snapshot_id = entry.get("snapshot_id") or 7000000000000000001
    sequence_number = entry.get("sequence_number") or 2

    blob = extract_blob(PUFFIN.read_bytes())
    puffin = build_puffin(blob, snapshot_id, sequence_number)

    shutil.copy2(PUFFIN, PUFFIN.with_suffix(".puffin.bak"))
    PUFFIN.write_bytes(puffin)
    print(f"wrote {PUFFIN.name}: {len(blob)}B blob -> {len(puffin)}B container")

    # The blob no longer starts at 0, so the manifest has to agree.
    entry["data_file"]["content_offset"] = len(MAGIC)
    entry["data_file"]["content_size_in_bytes"] = len(blob)

    shutil.copy2(MANIFEST, MANIFEST.with_suffix(".avro.bak"))
    with MANIFEST.open("wb") as fh:
        fastavro.writer(fh, schema, records, codec="null")
    print(f"wrote {MANIFEST.name}: content_offset={len(MAGIC)} size={len(blob)}")

    print("validating:")
    return 0 if verify(PUFFIN) else 1


if __name__ == "__main__":
    raise SystemExit(main())
