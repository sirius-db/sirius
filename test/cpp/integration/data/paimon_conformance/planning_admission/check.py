# Copyright 2026, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0

"""Caller-side stock SDK experiment for a sealed, unpartitioned local fixture."""

import argparse
import ctypes as C
import ctypes.util
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys

SNAPSHOT_CAP = 256 * 1024 - 1
FILE_CAP = 16 * 1024
BLOCK_CAP = 128 * 1024
TOTAL_DECODED_CAP = 1024 * 1024
MANIFEST_CAP = 64
ENTRY_CAP = 256
LIST_SCHEMA = "c9a49903a1ec4f40f9de34e222d102a4111e232874c9985bbfb7efd490dc1cc2"
ZSTD_UNKNOWN = (1 << 64) - 1
ZSTD_ERROR = ZSTD_UNKNOWN - 1


class Refused(Exception):
    pass


def need(ok, reason):
    if not ok:
        raise Refused(reason)


class Cursor:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def get(self, size):
        end = self.offset + size
        need(0 <= size and end <= len(self.data), "invalid-length")
        result = self.data[self.offset : end]
        self.offset = end
        return result

    def long(self):
        value = 0
        for shift in range(0, 70, 7):
            b = self.get(1)[0]
            need(shift < 63 or b <= 1, "invalid-long")
            value |= (b & 127) << shift
            if b < 128:
                return (value >> 1) ^ -(value & 1)
        raise Refused("invalid-long")

    def blob(self):
        return self.get(self.long())

    def done(self):
        return self.offset == len(self.data)


class FrameHeader(C.Structure):
    _fields_ = [
        ("content_size", C.c_ulonglong),
        ("window_size", C.c_ulonglong),
        ("block_size_max", C.c_uint),
        ("frame_type", C.c_uint),
        ("header_size", C.c_uint),
        ("dict_id", C.c_uint),
        ("checksum", C.c_uint),
        ("reserved1", C.c_uint),
        ("reserved2", C.c_uint),
    ]


def zstd_library():
    name = os.environ.get("PAIMON_PROBE_ZSTD") or ctypes.util.find_library("zstd")
    if not name:
        raise OSError("libzstd was not found; set PAIMON_PROBE_ZSTD to its path")
    library = C.CDLL(name)
    if not hasattr(library, "ZSTD_getFrameHeader"):
        raise OSError("libzstd does not export ZSTD_getFrameHeader")
    library.ZSTD_getFrameHeader.argtypes = [
        C.POINTER(FrameHeader),
        C.c_void_p,
        C.c_size_t,
    ]
    library.ZSTD_getFrameHeader.restype = C.c_size_t
    library.ZSTD_findFrameCompressedSize.argtypes = [C.c_void_p, C.c_size_t]
    library.ZSTD_findFrameCompressedSize.restype = C.c_size_t
    library.ZSTD_decompress.argtypes = [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]
    library.ZSTD_decompress.restype = C.c_size_t
    return library


def read_bounded(path, cap=FILE_CAP):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as opened:
        info = os.fstat(opened.fileno())
        need(stat.S_ISREG(info.st_mode), "unsupported-type")
        need(info.st_size <= cap, "encoded-file-budget")
        data = opened.read(cap + 1)
        need(len(data) == info.st_size, "extent-mismatch")
        return data


def simple_name(data, prefix):
    name = data.decode("ascii")
    need(
        name.startswith(prefix) and all(c.isalnum() or c in "-_" for c in name),
        "unsupported-name",
    )
    return name


def inspect_ocf(path, zstd, record_cap, expected_schema=None):
    raw = read_bounded(path)
    cursor = Cursor(raw)
    need(cursor.get(4) == b"Obj\x01", "unsupported-container")
    attrs = {}
    while (count := cursor.long()) != 0:
        need(count > 0 and count <= 8 - len(attrs), "unsupported-header")
        for _ in range(count):
            key, value = cursor.blob(), cursor.blob()
            need(key not in attrs, "duplicate-header-key")
            attrs[key] = value
    need(attrs.get(b"avro.codec") == b"zstandard", "unsupported-codec")
    if expected_schema:
        schema = json.dumps(
            json.loads(attrs[b"avro.schema"]), sort_keys=True, separators=(",", ":")
        )
        need(
            hashlib.sha256(schema.encode()).hexdigest() == expected_schema,
            "unsupported-list-schema",
        )
    sync = cursor.get(16)
    records = 0
    blocks = []
    while not cursor.done():
        count = cursor.long()
        need(
            0 < count <= record_cap - records,
            "manifest-count-budget" if expected_schema else "entry-count-budget",
        )
        encoded = cursor.blob()
        need(cursor.get(16) == sync, "invalid-sync")
        header = FrameHeader()
        need(
            zstd.ZSTD_getFrameHeader(C.byref(header), encoded, len(encoded)) == 0
            and header.frame_type == 0
            and header.dict_id == 0,
            "unknown-frame",
        )
        need(
            header.content_size not in (ZSTD_UNKNOWN, ZSTD_ERROR),
            "unknown-decoded-size",
        )
        need(header.content_size <= BLOCK_CAP, "decoded-block-budget")
        need(header.window_size <= BLOCK_CAP, "decoder-window-budget")
        need(
            zstd.ZSTD_findFrameCompressedSize(encoded, len(encoded)) == len(encoded),
            "not-one-frame",
        )
        records += count
        blocks.append((count, header.content_size, encoded))
    return len(raw), records, blocks


def read_list(blocks, zstd):
    for count, size, encoded in blocks:
        destination = C.create_string_buffer(size)
        need(
            zstd.ZSTD_decompress(destination, size, encoded, len(encoded)) == size,
            "invalid-list",
        )
        cursor = Cursor(destination.raw)
        for _ in range(count):
            need(cursor.long() == 2, "unsupported-list-version")
            name = simple_name(cursor.blob(), "manifest-")
            length, added, deleted = cursor.long(), cursor.long(), cursor.long()
            cursor.blob()
            cursor.blob()
            tag = cursor.long()
            need(
                tag == 0 or (tag == 1 and cursor.long() == 0),
                "unsupported-partition-stats",
            )
            need(
                tuple(cursor.long() for _ in range(3)) == (0, 0, 0),
                "unsupported-list-fields",
            )
            yield name, length, added, deleted
        need(cursor.done(), "list-size-mismatch")


def admit(root, zstd):
    snapshot = json.loads(read_bounded(root / "snapshot/snapshot-2", SNAPSHOT_CAP))
    need(
        snapshot["id"] == 2
        and snapshot["schemaId"] == 0
        and snapshot["commitKind"] == "APPEND",
        "unsupported-snapshot",
    )
    # The experiment uses an isolated, sealed clone. Stock SDK reopens these paths.
    lists = [
        simple_name(snapshot[key].encode("ascii"), "manifest-list-")
        for key in ("baseManifestList", "deltaManifestList")
    ]
    manifests, entries, decoded = 0, 0, 0
    for name in lists:
        _, records, blocks = inspect_ocf(
            root / "manifest" / name, zstd, MANIFEST_CAP - manifests, LIST_SCHEMA
        )
        manifests += records
        decoded += sum(size for _, size, _ in blocks)
        need(decoded <= TOTAL_DECODED_CAP, "aggregate-decoded-budget")
        for filename, listed_size, added, deleted in read_list(blocks, zstd):
            need(
                0 <= added and 0 <= deleted and added + deleted <= ENTRY_CAP - entries,
                "entry-count-budget",
            )
            size, rows, content = inspect_ocf(
                root / "manifest" / filename, zstd, ENTRY_CAP - entries
            )
            need(
                size == listed_size and rows == added + deleted,
                "manifest-metadata-mismatch",
            )
            entries += rows
            decoded += sum(amount for _, amount, _ in content)
            need(decoded <= TOTAL_DECODED_CAP, "aggregate-decoded-budget")
    return {"manifest_refs": manifests, "entry_rows": entries, "decoded_bytes": decoded}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=pathlib.Path, help="owned append_basic fixture")
    parser.add_argument(
        "reader",
        nargs=argparse.REMAINDER,
        help="optional reader command invoked after admission",
    )
    args = parser.parse_args(argv)
    table = args.table.resolve()
    try:
        result = admit(table, zstd_library())
    except Refused as exc:
        print(json.dumps({"admission": "refused", "reason": str(exc)}), flush=True)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"admission": "error", "reason": str(exc)}), flush=True)
        return 3
    print(json.dumps({"admission": "allowed", **result}), flush=True)
    if args.reader:
        command = args.reader[1:] if args.reader[0] == "--" else args.reader
        if command:
            try:
                return subprocess.run(
                    command,
                    env=os.environ | {"PAIMON_PROBE_TABLE": str(table)},
                    timeout=45,
                ).returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(str(exc), file=sys.stderr)
                return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
