# Copyright 2026, Sirius Contributors.
# Licensed under the Apache License, Version 2.0. See LICENSE for details.

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import performance_test as benchmark


class ProfileConfigTests(unittest.TestCase):
    def test_profiles_preserve_yaml_and_keep_ordering_and_cache_flushes(self):
        for profile in ("cold", "hot", "lukewarm"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / "custom.yaml"
                original = (
                    "# Preserve spelling, comments, and settings.\n"
                    "sirius:\n"
                    "  executor:\n"
                    "    scan_manager:\n"
                    "      max_readahead_scans: 8\n"
                    "      cache: {mode: none, eviction: idle}\n"
                )
                config.write_text(original)
                output = root / "output"
                order = []
                con = Mock()

                def run_one(writer, connection, name, q, iteration, *args):
                    order.append((q, iteration))
                    self.assertEqual(os.environ["SIRIUS_CONFIG_FILE"], str(config))
                    writer.record(name, q, iteration, 1.0)

                argv = [
                    "performance_test.py",
                    "--input",
                    str(root),
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                    "--scale-factor",
                    "1000",
                    "--engine",
                    "gpu",
                    "--queries",
                    "1-2",
                    "--iterations",
                    "2",
                    "--pin",
                    "none",
                    "--profile",
                    profile,
                ]
                with (
                    patch.dict(os.environ),
                    patch.object(benchmark.sys, "argv", argv),
                    patch.object(benchmark, "open_connection", return_value=con),
                    patch.object(benchmark, "_run_one", side_effect=run_one),
                    patch.object(benchmark, "drop_os_cache") as drop,
                    patch.object(benchmark, "split_sirius_log"),
                    patch.object(benchmark, "print_runtime_summary"),
                    patch.object(benchmark, "log"),
                    patch.object(
                        benchmark, "get_git_info", return_value=("test", "test")
                    ),
                ):
                    benchmark.main()

                expected = (
                    [(1, 0), (1, 1), (2, 0), (2, 1)]
                    if profile == "hot"
                    else [(1, 0), (2, 0), (1, 1), (2, 1)]
                )
                self.assertEqual(order, expected)
                self.assertEqual(drop.call_count, 5 if profile == "cold" else 1)
                resets = [
                    c
                    for c in con.execute.call_args_list
                    if c.args == ("CALL reset_sirius_cache();",)
                ]
                self.assertEqual(len(resets), 4 if profile == "cold" else 0)
                self.assertEqual(config.read_text(), original)
                result = next(output.iterdir())
                self.assertEqual(
                    (result / "config.yml").read_bytes(), config.read_bytes()
                )
                self.assertFalse((result / "effective_config.yml").exists())

    def test_explicit_parquet_pinning_retains_its_required_cache_settings(self):
        self.assertEqual(
            benchmark.cache_overrides_for("parquet"),
            {"mode": "cucs", "eviction": "lru", "eviction_threshold_fraction": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
