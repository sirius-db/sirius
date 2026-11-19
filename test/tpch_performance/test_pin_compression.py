# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Benchmark compression options must override the extension/session default."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import performance_test as benchmark
import tpch_power_throughput as power
from tpch_pin_columns import emit_pin_all


class PinCompressionTests(unittest.TestCase):
    def test_pin_compression_table_allowlist(self):
        with patch.dict(
            os.environ, {"SIRIUS_PIN_COMPRESSION_TABLES": "lineitem,orders"}
        ):
            statements = emit_pin_all("/tpch.duckdb", data_source="duckdb")
        for table in ("lineitem", "orders"):
            self.assertRegex(
                statements,
                rf"name='{table}'[^;]*compression=true\);",
            )
        for table in ("customer", "nation", "part", "partsupp", "region", "supplier"):
            self.assertRegex(
                statements,
                rf"name='{table}'[^;]*compression=false\);",
            )

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

    def test_external_runner_script_sets_compression_before_pinning(self):
        for plan_dir in (None, "/plans"):
            with self.subTest(
                plan_dir=plan_dir
            ), tempfile.TemporaryDirectory() as directory:
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
