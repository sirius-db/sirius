# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. See the LICENSE file at the
# repo root for the full text.
# =============================================================================

import math
from pathlib import Path
import subprocess
import sys
import unittest

import duckdb

from tpch_schema import schema_for_scale_factor


class AdaptiveTpchSchemaTests(unittest.TestCase):
    def key_types(self, connection):
        return dict(
            connection.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND column_name LIKE '%key'"
            ).fetchall()
        )

    def assert_schema(self, scale, wide_families):
        with duckdb.connect() as connection:
            connection.execute(schema_for_scale_factor(scale))
            keys = self.key_types(connection)
            self.assertEqual(len(keys), 15)
            for column, actual in keys.items():
                family = column.split("_", 1)[1]
                expected = "BIGINT" if family in wide_families else "INTEGER"
                self.assertEqual(actual, expected, (scale, column))

    def test_integer_scale_boundaries(self):
        for scale, wide in [
            (0.01, set()),
            (1, set()),
            (357, set()),
            (358, {"orderkey"}),
            (1000, {"orderkey"}),
            (10737, {"orderkey"}),
            (10738, {"orderkey", "partkey"}),
            (14316, {"orderkey", "partkey"}),
            (14317, {"orderkey", "partkey", "custkey"}),
            (214748, {"orderkey", "partkey", "custkey"}),
            (214749, {"orderkey", "partkey", "custkey", "suppkey"}),
        ]:
            with self.subTest(scale=scale):
                self.assert_schema(scale, wide)

    def test_fractional_scale_boundaries(self):
        # Exact last-safe / first-overflow row counts, including sparse order keys.
        cases = [
            ("orderkey", 1_500_000, 536_870_911, set()),
            ("partkey", 200_000, 2_147_483_647, {"orderkey"}),
            ("custkey", 150_000, 2_147_483_647, {"orderkey", "partkey"}),
            ("suppkey", 10_000, 2_147_483_647, {"orderkey", "partkey", "custkey"}),
        ]
        for family, base, last_safe, already_wide in cases:
            for offset in (0, 1):
                scale = (last_safe + offset + 0.25) / base
                with self.subTest(family=family, offset=offset):
                    self.assert_schema(
                        scale, already_wide | ({family} if offset else set())
                    )

    def test_wide_keys_round_trip_and_join(self):
        with duckdb.connect() as connection:
            connection.execute(schema_for_scale_factor(300000))
            # These values would overflow the previous fixed INTEGER columns.
            key = 2_147_483_648
            rows = {
                "part": ("p_partkey",),
                "supplier": ("s_suppkey",),
                "customer": ("c_custkey",),
                "partsupp": ("ps_partkey", "ps_suppkey"),
                "orders": ("o_orderkey", "o_custkey"),
                "lineitem": ("l_orderkey", "l_partkey", "l_suppkey"),
            }
            for table, columns in rows.items():
                placeholders = ", ".join("?" for _ in columns)
                connection.execute(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                    [key] * len(columns),
                )
            result = connection.execute(
                "SELECT l_orderkey, l_partkey, l_suppkey, o_custkey FROM lineitem "
                "JOIN orders ON l_orderkey = o_orderkey "
                "JOIN customer ON o_custkey = c_custkey "
                "JOIN partsupp ON l_partkey = ps_partkey AND l_suppkey = ps_suppkey "
                "JOIN part ON ps_partkey = p_partkey "
                "JOIN supplier ON ps_suppkey = s_suppkey"
            ).fetchall()
            self.assertEqual(result, [(key, key, key, key)])

    def test_invalid_scales_produce_no_schema(self):
        for scale in (0, -1, math.nan, math.inf, -math.inf, 1e308, 2e12):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    schema_for_scale_factor(scale)
        script = Path(__file__).with_name("tpch_schema.py")
        result = subprocess.run(
            [sys.executable, str(script), "nan"], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("finite and positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
