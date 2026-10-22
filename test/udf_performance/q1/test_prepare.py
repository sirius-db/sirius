# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the public Q1 input preparation contract."""

import csv
from pathlib import Path
import tempfile
import unittest

import prepare


class PreparationTests(unittest.TestCase):
    def test_restricted_domain(self):
        for value in (None, "", "2024-1-01", "２０２４-01-01", "2024-01-01\n"):
            self.assertEqual(
                [prepare.restricted_extract(value, a, b) for a, b in prepare.SLICES],
                [None, None, None],
            )
        # The original extraction does no calendar validation on this domain.
        for value, expected in (
            ("2023-02-31", [2023, 2, 31]),
            ("0000-00-00", [0, 0, 0]),
            ("9999-99-99", [9999, 99, 99]),
        ):
            self.assertEqual(
                [prepare.restricted_extract(value, a, b) for a, b in prepare.SLICES],
                expected,
            )

    def test_source_modification_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "LICENSE"
            path.write_bytes(b"not the pinned original")
            with self.assertRaisesRegex(ValueError, "pinned source hash/size mismatch"):
                prepare.verify_source(path, "LICENSE")
            path.write_bytes(b"x" * prepare.SOURCES["LICENSE"][1])
            with self.assertRaisesRegex(ValueError, "pinned source hash/size mismatch"):
                prepare.verify_source(path, "LICENSE")
            output = Path(temporary) / "output"
            with self.assertRaisesRegex(ValueError, "pinned source hash/size mismatch"):
                prepare.fetch_sources(output, Path(temporary))
            self.assertFalse((output / "sources/LICENSE").exists())
            self.assertEqual(list((output / "sources").iterdir()), [])

    def test_projection_distinguishes_empty_and_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "projection.csv"
            prepare.write_csv(
                path,
                [
                    [0, "empty", "", None, None, None],
                    [1, "null", None, None, None, None],
                ],
            )
            with path.open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], list(prepare.FIELDS))
            self.assertEqual(
                rows[1][2:], ["", prepare.NULL, prepare.NULL, prepare.NULL]
            )
            self.assertEqual(rows[2][2:], [prepare.NULL] * 4)


if __name__ == "__main__":
    unittest.main()
