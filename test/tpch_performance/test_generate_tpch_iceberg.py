# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Local writer checks: pixi run python -m unittest discover \
    -s test/tpch_performance -p test_generate_tpch_iceberg.py
"""

from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq
from pyiceberg.table import StaticTable

from generate_tpch_iceberg import TABLES, generate

FIXTURES = Path(__file__).resolve().parents[1] / "cpp/integration/data/parquet"


class IcebergGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tpch-iceberg-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "parquet"
        self.source.mkdir()
        self.output = self.root / "warehouse's data"
        self.expected = {}
        for name in TABLES:
            with pq.ParquetFile(FIXTURES / f"{name}.parquet") as parquet:
                batch = next(parquet.iter_batches(batch_size=4))
            import pyarrow as pa

            data = pa.Table.from_batches([batch])
            self.expected[name] = data
            pq.write_table(data, self.source / f"{name}.parquet")

    def generate(self, batch_rows=2):
        with redirect_stdout(StringIO()):
            return generate(self.source, self.output, batch_rows)

    def test_split_inputs_round_trip_without_original_files(self):
        # Exercise both partitioned layouts, multiple shards and write batches.
        for name, nested in (("lineitem", True), ("orders", False)):
            (self.source / f"{name}.parquet").unlink()
            parent = self.source / name if nested else self.source
            parent.mkdir(exist_ok=True)
            for index, data in enumerate(
                (self.expected[name].slice(0, 2), self.expected[name].slice(2))
            ):
                pq.write_table(data, parent / f"{name}_{index}.parquet")
        manifest = self.generate()
        self.source.rename(self.root / "input-no-longer-at-original-location")
        self.assertEqual(set(manifest["tables"]), set(TABLES))
        self.assertEqual(
            json.loads((self.output / "dataset.json").read_text()), manifest
        )
        views = (self.output / "create_views.sql").read_text()
        self.assertNotIn("snapshot_from_id", views)
        for name, entry in manifest["tables"].items():
            with self.subTest(table=name):
                directory = self.output / name
                quoted = str(directory).replace("'", "''")
                self.assertIn(f"iceberg_scan('{quoted}');", views)
                hint = (directory / "metadata/version-hint.text").read_text()
                self.assertEqual(
                    directory / "metadata" / f"{hint}.metadata.json",
                    Path(entry["metadata_file"]),
                )
                table = StaticTable.from_metadata(entry["metadata_file"])
                actual = table.scan(snapshot_id=entry["snapshot_id"]).to_arrow()
                self.assertEqual(actual.column_names, self.expected[name].column_names)
                self.assertEqual(
                    Counter(tuple(row.values()) for row in actual.to_pylist()),
                    Counter(
                        tuple(row.values()) for row in self.expected[name].to_pylist()
                    ),
                )
                self.assertEqual(entry["row_count"], len(self.expected[name]))
                for path in entry["data_files"]:
                    for field in pq.read_schema(path):
                        self.assertEqual(
                            int(field.metadata[b"PARQUET:field_id"]),
                            entry["field_ids"][field.name],
                        )

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing data")
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            self.generate()
        self.assertEqual(marker.read_text(), "existing data")

    def test_missing_table_is_rejected_before_writing(self):
        (self.source / "lineitem.parquet").unlink()
        with self.assertRaisesRegex(ValueError, "Missing Parquet data for lineitem"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_ambiguous_layout_is_rejected_before_writing(self):
        pq.write_table(self.expected["region"], self.source / "region_0.parquet")
        with self.assertRaisesRegex(ValueError, "Ambiguous Parquet layouts for region"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_inconsistent_shard_is_rejected_before_writing(self):
        original = self.source / "lineitem.parquet"
        original.rename(self.source / "lineitem_0.parquet")
        pq.write_table(
            self.expected["lineitem"].drop(["l_shipdate"]),
            self.source / "lineitem_1.parquet",
        )
        with self.assertRaisesRegex(ValueError, "Inconsistent schema for lineitem"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_empty_table_has_a_readable_snapshot(self):
        self.expected["supplier"] = self.expected["supplier"].slice(0, 0)
        pq.write_table(self.expected["supplier"], self.source / "supplier.parquet")
        entry = self.generate(batch_rows=10)["tables"]["supplier"]
        table = StaticTable.from_metadata(entry["metadata_file"])
        self.assertEqual(
            table.scan(snapshot_id=entry["snapshot_id"]).to_arrow().num_rows, 0
        )
        self.assertEqual(entry["row_count"], 0)


if __name__ == "__main__":
    unittest.main()
