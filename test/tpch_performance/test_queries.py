# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import performance_test
import tpch_power_throughput
from queries import QUERIES, queries_for_scale_factor, q11_fraction


class TestScaleAwareQueries(unittest.TestCase):
    def test_q11_fraction_matches_qgen_format(self):
        expected = {
            1: "0.0001000000",
            10: "0.0000100000",
            30: "0.0000033333",
            50: "0.0000020000",
            100: "0.0000010000",
            500: "0.0000002000",
            1000: "0.0000001000",
        }
        for scale_factor, fraction in expected.items():
            with self.subTest(scale_factor=scale_factor):
                self.assertEqual(q11_fraction(scale_factor), fraction)
                self.assertIn(
                    f"* {fraction}", queries_for_scale_factor(scale_factor)["q11"]
                )

    def test_only_q11_changes_with_scale_factor(self):
        sf1 = queries_for_scale_factor(1)
        sf100 = queries_for_scale_factor(100)
        self.assertEqual(set(sf1), set(sf100))
        for query_name in sf1.keys() - {"q11"}:
            with self.subTest(query=query_name):
                self.assertEqual(sf1[query_name], sf100[query_name])
        self.assertNotEqual(sf1["q11"], sf100["q11"])

    def test_default_queries_remain_sf1_compatible(self):
        self.assertEqual(QUERIES, queries_for_scale_factor(1))

    def test_invalid_scale_factors_are_rejected(self):
        for scale_factor in (0, -1, "nan", "inf", "not-a-number"):
            with self.subTest(scale_factor=scale_factor):
                with self.assertRaises(ValueError):
                    queries_for_scale_factor(scale_factor)

    def test_benchmark_artifacts_record_scale_and_effective_sql(self):
        query_texts = queries_for_scale_factor(100)
        with tempfile.TemporaryDirectory() as output_root:
            with patch.object(
                performance_test, "get_git_info", return_value=("abc", "branch")
            ):
                benchmark_dir, _, _ = performance_test.setup_benchmark_dir(
                    output_root,
                    "grouped",
                    1,
                    "cpu",
                    [11],
                    "",
                    "none",
                    100,
                    query_texts,
                )
            with open(os.path.join(benchmark_dir, "metadata.json")) as f:
                metadata = json.load(f)
            with open(os.path.join(benchmark_dir, "queries", "q11.sql")) as f:
                q11_sql = f.read()

        self.assertEqual(metadata["scale_factor"], 100)
        self.assertIn("* 0.0000010000", q11_sql)

    def test_nsys_script_uses_rendered_query(self):
        q11_sql = queries_for_scale_factor(500)["q11"]
        with tempfile.TemporaryDirectory() as qdir:
            script_path = performance_test._build_nsys_temp_sql(
                11, q11_sql, "unused.duckdb", 1, "none", qdir, "duckdb"
            )
            with open(script_path) as f:
                script = f.read()
        self.assertIn("* 0.0000002000", script)
        self.assertNotIn("* 0.0001000000", script)

    def test_power_throughput_fixed_queries_use_run_scale(self):
        args = SimpleNamespace(
            vary_predicates=False,
            query_texts=queries_for_scale_factor(1000),
        )
        stream = dict(tpch_power_throughput.stream_queries(0, args))

        self.assertIn("* 0.0000001000", stream[11][0])


if __name__ == "__main__":
    unittest.main()
