# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Benchmark compression options must override the extension/session default."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import performance_test as benchmark
import tpch_power_throughput as power


class PinCompressionTests(unittest.TestCase):
    def test_python_connection_sets_both_compression_states(self):
        for plan_dir in (None, "/plans"):
            with self.subTest(plan_dir=plan_dir):
                con = Mock()
                with (
                    patch.object(benchmark.duckdb, "connect", return_value=con),
                    patch.object(benchmark, "PIN_COMPRESSION_PLAN_DIR", plan_dir),
                    patch.dict(os.environ, {"SIRIUS_PRE_SQL": ""}),
                ):
                    benchmark.open_connection(
                        "/tpch.duckdb", gpu_execution=True, data_source="duckdb"
                    )
                statements = [call.args[0] for call in con.execute.call_args_list]
                intended = (
                    f"SET pin_table_compression = {'true' if plan_dir else 'false'};"
                )
                self.assertIn(intended, statements)
                self.assertLess(
                    next(
                        i
                        for i, statement in enumerate(statements)
                        if statement.startswith("LOAD ")
                    ),
                    statements.index(intended),
                )

    def test_power_runner_sets_compression_before_pinning(self):
        for plan_dir in (None, "/plans"):
            with self.subTest(plan_dir=plan_dir):
                con = Mock()
                with (
                    patch.object(power.duckdb, "connect", return_value=con),
                    patch.dict(os.environ, {"SIRIUS_PRE_SQL": ""}),
                ):
                    power.open_benchmark_db(
                        "/tpch.duckdb", "gpu", compression_plan_dir=plan_dir
                    )
                statements = [call.args[0] for call in con.execute.call_args_list]
                intended = (
                    f"SET pin_table_compression = {'true' if plan_dir else 'false'}"
                )
                self.assertIn(intended, statements)
                first_pin = next(
                    i
                    for i, statement in enumerate(statements)
                    if statement.startswith("CALL pin_table(")
                )
                self.assertLess(statements.index(intended), first_pin)

    def test_power_runner_compresses_with_automatic_plans_without_a_plan_dir(self):
        con = Mock()
        with (
            patch.object(power.duckdb, "connect", return_value=con),
            patch.dict(os.environ, {"SIRIUS_PRE_SQL": ""}),
        ):
            power.open_benchmark_db("/tpch.duckdb", "gpu", pin_compression=True)
        statements = [call.args[0] for call in con.execute.call_args_list]
        self.assertIn("SET pin_table_compression = true", statements)
        self.assertFalse(
            any("pin_table_input_compression_plan_dir" in s for s in statements)
        )

    def test_compressed_pin_count_counts_only_pins_with_compressed_chunks(self):
        with tempfile.TemporaryDirectory() as log_dir:
            Path(log_dir, "sirius.log").write_text(
                "[info] [pin_table] 'part' tier=gpu: automatically selecting "
                "compression plans for 4 column(s)\n"
                "[info] [materialize_all_batches_compressed] pin '/d/part.parquet': "
                "0/1 chunk(s) compressed\n"
            )
            self.assertEqual(power.pin_compression_counts(log_dir, timeout_s=0), (0, 0))
            with Path(log_dir, "sirius.log").open("a") as log:
                log.write(
                    "[info] [materialize_pin_to_host] pin '/d/lineitem.parquet': "
                    "3/13 chunk(s) compressed — remainder pinned UNCOMPRESSED\n"
                )
            self.assertEqual(power.pin_compression_counts(log_dir, timeout_s=0), (1, 0))

    def test_external_runner_script_sets_compression_before_pinning(self):
        for plan_dir in (None, "/plans"):
            with (
                self.subTest(plan_dir=plan_dir),
                tempfile.TemporaryDirectory() as directory,
            ):
                with (
                    patch.object(benchmark, "PIN_COMPRESSION_PLAN_DIR", plan_dir),
                    patch.dict(os.environ, {"SIRIUS_PRE_SQL": ""}),
                ):
                    path = benchmark._build_precmd_temp_sql(
                        1,
                        "SELECT 42",
                        "/tpch.duckdb",
                        1,
                        "gpu",
                        directory,
                        "nsys",
                        "duckdb",
                    )
                statements = Path(path).read_text()
                intended = (
                    f"SET pin_table_compression = {'true' if plan_dir else 'false'};"
                )
                self.assertIn(intended, statements)
                self.assertLess(
                    statements.index(intended), statements.index("CALL pin_table(")
                )


if __name__ == "__main__":
    unittest.main()
