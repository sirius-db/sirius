# Copyright 2026, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import fastavro
import zstandard

import check

FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1] / "warehouse/reference.db/append_basic"
)
CHECK = pathlib.Path(check.__file__).resolve()


def read_ocf(path):
    with path.open("rb") as stream:
        reader = fastavro.reader(stream)
        return reader.writer_schema, list(reader)


def write_ocf(path, schema, records):
    with path.open("wb") as stream:
        fastavro.writer(stream, schema, records, codec="zstandard")


def avro_long(value):
    value <<= 1
    encoded = bytearray()
    while value >= 128:
        encoded.append((value & 127) | 128)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = pathlib.Path(scratch.name) / "table"
        shutil.copytree(FIXTURE, self.root)
        snapshot = json.loads((self.root / "snapshot/snapshot-2").read_text())
        self.delta = self.root / "manifest" / snapshot["deltaManifestList"]
        self.calls = pathlib.Path(scratch.name) / "reader-calls"

    def run_check(self, expected_status, **environment):
        # The sentinel observes whether the caller enters the reader stage.
        code = "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()"
        result = subprocess.run(
            [
                sys.executable,
                str(CHECK),
                str(self.root),
                "--",
                sys.executable,
                "-c",
                code,
                str(self.calls),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ | environment,
        )
        self.assertEqual(
            result.returncode, expected_status, result.stdout + result.stderr
        )
        self.assertEqual(self.calls.exists(), expected_status == 0)
        return json.loads(result.stdout)

    def test_healthy_and_reencoded_list_can_read(self):
        self.assertEqual(self.run_check(0)["manifest_refs"], 2)
        self.calls.unlink()
        schema, rows = read_ocf(self.delta)
        write_ocf(self.delta, schema, rows)
        self.assertEqual(self.run_check(0)["entry_rows"], 2)

    def test_small_file_large_decoded_block_refused(self):
        schema, rows = read_ocf(self.delta)
        entry = self.delta.parent / rows[0]["_FILE_NAME"]
        entry_schema, entries = read_ocf(entry)
        entries[0]["_FILE"]["_EXTRA_FILES"] = ["aux-" + "x" * 250] * 32768
        write_ocf(entry, entry_schema, entries)
        rows[0]["_FILE_SIZE"] = entry.stat().st_size
        self.assertLess(entry.stat().st_size, check.FILE_CAP)
        write_ocf(self.delta, schema, rows)
        self.assertEqual(self.run_check(2)["reason"], "decoded-block-budget")

    def test_repeated_manifest_references_are_counted_before_reader(self):
        schema, rows = read_ocf(self.delta)
        write_ocf(self.delta, schema, rows * (check.MANIFEST_CAP + 1))
        self.assertLess(self.delta.stat().st_size, check.FILE_CAP)
        self.assertEqual(self.run_check(2)["reason"], "manifest-count-budget")

    def test_oversized_snapshot_refused(self):
        snapshot = self.root / "snapshot/snapshot-2"
        with snapshot.open("ab") as stream:
            stream.write(b" " * (270337 - snapshot.stat().st_size))
        self.assertEqual(self.run_check(2)["reason"], "encoded-file-budget")

    def test_frame_without_a_decoded_bound_refused(self):
        schema, rows = read_ocf(self.delta)
        entry = self.delta.parent / rows[0]["_FILE_NAME"]
        with entry.open("rb") as stream:
            (block,) = fastavro.block_reader(stream)
        cursor = check.Cursor(entry.read_bytes())
        cursor.get(4)
        while (count := cursor.long()) != 0:
            for _ in range(count):
                cursor.blob()
                cursor.blob()
        sync = cursor.get(16)
        compressed = zstandard.ZstdCompressor(write_content_size=False).compress(
            block.bytes_.getvalue()
        )
        entry.write_bytes(
            cursor.data[: cursor.offset]
            + avro_long(block.num_records)
            + avro_long(len(compressed))
            + compressed
            + sync
        )
        rows[0]["_FILE_SIZE"] = entry.stat().st_size
        write_ocf(self.delta, schema, rows)
        self.assertEqual(self.run_check(2)["reason"], "unknown-decoded-size")

    def test_missing_zstd_is_an_error_before_reader(self):
        result = self.run_check(3, PAIMON_PROBE_ZSTD=str(self.root / "missing-libzstd"))
        self.assertEqual(result["admission"], "error")


if __name__ == "__main__":
    unittest.main()
