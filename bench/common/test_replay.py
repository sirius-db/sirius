# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Scheduling and failure-path tests; never loads Sirius or accesses AWS."""

from contextlib import redirect_stdout
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import checks
import replay


class ReplayTests(unittest.TestCase):
    def test_fake_child_processes_preserve_configs_and_partial_results(self):
        # This is a tiny stdlib-only stand-in, never a GPU benchmark.
        fake = """import argparse, csv, os
from pathlib import Path
p=argparse.ArgumentParser()
for name in ("input","scale-factor","engine","profile","iterations","queries","config","output","name","pin"):
    p.add_argument("--"+name)
a=p.parse_args()
assert os.environ["SIRIUS_PRE_SQL"]=="SET enable_duckdb_fallback = false;"
assert a.profile=="cold" and a.iterations=="1"
r=Path(a.output)/"fake"
(r/"csv").mkdir(parents=True)
(r/"pid").write_text(str(os.getpid()))
with (r/"csv/runtimes.csv").open("w") as f:
    w=csv.writer(f); w.writerow(["engine","query","iteration","runtime_s"])
    for q in a.queries.split(","):
        for i in range(int(a.iterations)):
            w.writerow(["sirius","q"+q,i,"nan" if q==os.environ.get("FAKE_FAIL_QUERY") else "1.25"])
        log=r/"sirius"/("q"+q)/"sirius.log"
        log.parent.mkdir(parents=True)
        log.write_text("Transparent execution: physical plan replaced with GPU operator\\n"*int(a.iterations))
"""
        for hardware, failed in (
            (h, f) for h in ("gb300", "g7e") for f in (False, True)
        ):
            with self.subTest(
                hardware=hardware, failed=failed
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = root / "bench" / hardware / "configs/cold.yaml"
                config.parent.mkdir(parents=True)
                original = "sirius:\n  executor:\n    pipeline: {num_threads: 4}\n"
                config.write_text(original)
                runner = root / "fake.py"
                runner.write_text(fake)
                extension = (
                    root
                    / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
                )
                extension.parent.mkdir(parents=True)
                extension.write_text("not an actual extension")
                output = root / "output"
                with patch.object(replay, "ROOT", root), patch.object(
                    replay, "BENCHMARK", runner
                ), patch.object(replay, "preflight", return_value={}), patch.object(
                    replay.subprocess, "check_output", return_value="fake-source"
                ), patch.dict(
                    os.environ, {"FAKE_FAIL_QUERY": "6" if failed else ""}
                ), redirect_stdout(
                    io.StringIO()
                ):
                    code = replay.main(
                        [
                            "--hardware",
                            hardware,
                            "--sf",
                            "1000",
                            "--profile",
                            "cold",
                            "--queries",
                            "3,6",
                            "--iterations",
                            "2",
                            "--output",
                            str(output),
                        ]
                    )
                report = json.loads((output / "summary.json").read_text())
                self.assertEqual(code, 1 if failed else 0)
                self.assertEqual(report["complete"], not failed)
                self.assertEqual(report["successful_measurements"], 1 if failed else 4)
                self.assertEqual(config.read_text(), original)
                pids = [
                    p.read_text() for p in output.glob("worker*/benchmark/fake/pid")
                ]
                self.assertEqual(len(pids), 2 if failed else 4)
                self.assertEqual(len(set(pids)), len(pids))

    def test_cold_restarts_every_measurement_on_both_hardware_types(self):
        for hardware in ("gb300", "g7e"):
            with self.subTest(hardware=hardware):
                steps = replay.schedule(hardware, "cold", [3, 6], 2, 1000)
                self.assertEqual(
                    [(s["queries"], s["iterations"]) for s in steps],
                    [([3], [1]), ([6], [1]), ([3], [2]), ([6], [2])],
                )

    def test_gb300_cold_q10_override_preserves_fresh_process_order(self):
        steps = replay.schedule("gb300", "cold", [3, 10, 18], 2, 10000)
        self.assertEqual(
            [(s["queries"], s["iterations"]) for s in steps],
            [
                ([3], [1]),
                ([10], [1]),
                ([18], [1]),
                ([3], [2]),
                ([10], [2]),
                ([18], [2]),
            ],
        )
        for step in steps:
            expected = "sf10k-q10-cold.yaml" if step["queries"] == [10] else "cold.yaml"
            self.assertEqual(Path(step["config"]).name, expected)

    def test_g7e_hot_and_pinned_reuse_only_within_query(self):
        for profile in ("hot", "pinned"):
            steps = replay.schedule("g7e", profile, [3, 6], 2, 1000)
            self.assertEqual(
                [(s["queries"], s["iterations"]) for s in steps],
                [([3], [1, 2]), ([6], [1, 2])],
            )

    def test_gb300_hot_preserves_session_and_q10_override(self):
        steps = replay.schedule("gb300", "hot", [3, 6], 2, 1000)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["queries"], [3, 6])
        steps = replay.schedule("gb300", "hot", [3, 10, 18], 2, 10000)
        self.assertEqual([s["queries"] for s in steps], [[3, 18], [10]])
        self.assertTrue(steps[1]["config"].endswith("sf10k-q10-hot.yaml"))
        with self.assertRaises(ValueError):
            replay.schedule("gb300", "pinned", [3], 1, 10000)

    def test_dry_run_does_not_access_aws_gpu_dataset_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must-not-exist"
            with patch.object(
                replay, "preflight", side_effect=AssertionError("preflight in dry run")
            ), patch.object(
                replay.subprocess,
                "run",
                side_effect=AssertionError("launched benchmark"),
            ), redirect_stdout(
                io.StringIO()
            ):
                code = replay.main(
                    [
                        "--hardware",
                        "g7e",
                        "--sf",
                        "1000",
                        "--profile",
                        "cold",
                        "--input",
                        "s3://not-contacted/data",
                        "--output",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())

    def test_environment_disables_fallback_and_clears_stale_overrides(self):
        with patch.dict(
            replay.os.environ,
            {
                "SIRIUS_PRE_SQL": "bad SQL",
                "SIRIUS_CONFIG_FILE": "old.yml",
                "SIRIUS_DISABLE": "1",
                "SIRIUS_PIN_COMPRESSION_TABLES": "old",
            },
        ):
            env = replay.child_env("g7e", "cold")
        self.assertNotIn("SIRIUS_DISABLE", env)
        self.assertNotIn("SIRIUS_CONFIG_FILE", env)
        self.assertNotIn("SIRIUS_PIN_COMPRESSION_TABLES", env)
        self.assertEqual(env["SIRIUS_PRE_SQL"], "SET enable_duckdb_fallback = false;")
        self.assertEqual(
            replay.child_env("gb300", "pinned")["SIRIUS_PIN_COMPRESSION_TABLES"],
            "lineitem,orders",
        )

    def test_separate_engine_checkout_does_not_change_config_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Path(tmp) / "pinned-engine"
            captured = io.StringIO()
            with redirect_stdout(captured):
                replay.main(
                    [
                        "--hardware",
                        "g7e",
                        "--sf",
                        "10000",
                        "--profile",
                        "cold",
                        "--queries",
                        "3",
                        "--source-dir",
                        str(engine),
                        "--dry-run",
                    ]
                )
            plan = json.loads(captured.getvalue())
            self.assertEqual(
                plan["environment_overrides"]["SIRIUS_BUILD_PATH"],
                str(engine / "sirius-duckdb/build/release"),
            )
            self.assertEqual(
                Path(plan["steps"][0]["config"]),
                replay.ROOT / "bench/g7e/configs/cold.yaml",
            )
            self.assertFalse(engine.exists())

    def test_explicit_config_is_preserved_for_every_query_and_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "custom.yaml"
            original = "sirius:\n  executor:\n    pipeline: {num_threads: 4}\n"
            config.write_text(original)
            for hardware, profile in (
                ("g7e", "cold"),
                ("g7e", "hot"),
                ("gb300", "cold"),
                ("gb300", "hot"),
            ):
                with self.subTest(hardware=hardware, profile=profile):
                    captured = io.StringIO()
                    with redirect_stdout(captured):
                        replay.main(
                            [
                                "--hardware",
                                hardware,
                                "--sf",
                                "10000",
                                "--profile",
                                profile,
                                "--queries",
                                "3,10",
                                "--config",
                                str(config),
                                "--dry-run",
                            ]
                        )
                    plan = json.loads(captured.getvalue())
                    self.assertTrue(
                        all(s["config"] == str(config) for s in plan["steps"])
                    )
                    self.assertEqual(config.read_text(), original)

    def test_sums_keep_missing_measurements_missing(self):
        report = replay.summarize([3, 6], 2, {(3, 1): 4.0, (3, 2): 2.0, (6, 1): 7.0})
        self.assertEqual(report["sum_best_s"], 9.0)
        self.assertEqual(report["iteration_totals"][0]["sum_s"], 11.0)
        self.assertEqual(report["iteration_totals"][1]["queries"], 1)
        self.assertEqual(report["per_query"][1]["iterations_s"], [7.0, None])
        empty = replay.summarize([17], 2, {})
        self.assertIsNone(empty["sum_best_s"])
        self.assertIsNone(empty["iteration_totals"][0]["sum_s"])

    def runtime_tree(self, root, rows, gpu_plans=2):
        directory = root / "benchmark/run/csv"
        directory.mkdir(parents=True)
        with (directory / "runtimes.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["engine", "query", "iteration", "runtime_s"])
            writer.writerows(rows)
        log = root / "benchmark/run/sirius/q3/sirius.log"
        log.parent.mkdir(parents=True)
        log.write_text(
            "Transparent execution: physical plan replaced with GPU operator\n"
            * gpu_plans
        )

    def test_child_iterations_map_to_campaign_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.runtime_tree(
                root, [("sirius", "q3", 0, 5.0), ("sirius", "TOTAL", 0, 5.0)]
            )
            self.assertEqual(
                replay.read_timings(root, {"queries": [3], "iterations": [2]}),
                {(3, 2): 5.0},
            )

    def test_untrusted_or_duplicate_timings_fail(self):
        cases = [
            ([("duckdb", "q3", 0, 5)], 1),
            ([("sirius", "q3", 0, 5)] * 2, 2),
            ([("sirius", "q3", 0, 5)], 0),
        ]
        for rows, gpu_plans in cases:
            with self.subTest(
                rows=rows, gpu_plans=gpu_plans
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.runtime_tree(root, rows, gpu_plans)
                with self.assertRaises(ValueError):
                    replay.read_timings(root, {"queries": [3], "iterations": [1]})

    def test_nan_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.runtime_tree(root, [("sirius", "q3", 0, "nan")], 0)
            self.assertIsNone(
                replay.read_timings(root, {"queries": [3], "iterations": [1]})[(3, 1)]
            )

    def test_cross_region_s3_rejected_before_inventory(self):
        client = Mock()
        client.get_bucket_location.return_value = {"LocationConstraint": "eu-west-2"}
        with patch.dict(
            sys.modules, {"boto3": SimpleNamespace(client=Mock(return_value=client))}
        ):
            with self.assertRaisesRegex(ValueError, "Cross-region"):
                checks.verify_s3("s3://example/data", "us-east-1")
        client.get_paginator.assert_not_called()

    def test_same_region_s3_checks_all_tables(self):
        client = Mock()
        client.get_bucket_location.return_value = {"LocationConstraint": None}
        pages = client.get_paginator.return_value
        pages.paginate.side_effect = lambda **kwargs: [
            {"Contents": [{"Key": kwargs["Prefix"] + "/part.parquet"}]}
        ]
        with patch.dict(
            sys.modules, {"boto3": SimpleNamespace(client=Mock(return_value=client))}
        ):
            evidence = checks.verify_s3("s3://example/data", "us-east-1")
        self.assertEqual(evidence["region"], "us-east-1")
        self.assertEqual(pages.paginate.call_count, 8)

    def test_head_bucket_region_is_required_when_location_is_denied(self):
        for location in ("us-east-1", "eu-west-2", None):
            with self.subTest(location=location):
                client = Mock()
                error = RuntimeError("GetBucketLocation denied")
                error.response = {"Error": {"Code": "AccessDenied"}}
                client.get_bucket_location.side_effect = error
                client.head_bucket.return_value = {"BucketRegion": location}
                pages = client.get_paginator.return_value
                pages.paginate.side_effect = lambda **kwargs: [
                    {"Contents": [{"Key": kwargs["Prefix"] + "/part.parquet"}]}
                ]
                with patch.dict(
                    sys.modules,
                    {"boto3": SimpleNamespace(client=Mock(return_value=client))},
                ):
                    if location == "us-east-1":
                        self.assertEqual(
                            checks.verify_s3("s3://example/data", "us-east-1")[
                                "region"
                            ],
                            location,
                        )
                    else:
                        with self.assertRaises(ValueError):
                            checks.verify_s3("s3://example/data", "us-east-1")
                        client.get_paginator.assert_not_called()

    def test_location_network_errors_do_not_fall_back_to_assumed_region(self):
        client = Mock()
        client.get_bucket_location.side_effect = RuntimeError("network failure")
        with patch.dict(
            sys.modules, {"boto3": SimpleNamespace(client=Mock(return_value=client))}
        ):
            with self.assertRaises(RuntimeError):
                checks.verify_s3("s3://example/data", "us-east-1")
        client.head_bucket.assert_not_called()

    def test_saved_g7e_snapshot_hashes_and_sums(self):
        root = replay.ROOT / "bench/g7e/results/baseline-20261111"
        manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["checksums_sha256"].items():
            self.assertEqual(replay.digest(root / name), expected, name)
        with (root / "per-query.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        for summary in manifest["summaries"]:
            group = [r for r in rows if r["stage"] == summary["stage"]]
            self.assertEqual(len(group), 22)
            best = [
                float(r["best_available_s"]) for r in group if r["best_available_s"]
            ]
            self.assertAlmostEqual(math.fsum(best), summary["sum_best_s"], places=8)
        self.assertEqual(manifest["summaries"][-1]["queries_with_timings"], 21)
        self.assertEqual(
            (replay.ROOT / "bench/g7e/configs/hot.yaml").read_bytes(),
            (replay.ROOT / "bench/g7e/configs/pinned.yaml").read_bytes(),
        )

    def test_ebs_is_rejected_by_default_and_requires_explicit_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "sirius": {
                    "memory": {
                        "disk": {"downgrade_root_dirs": tmp, "capacity_bytes": "1"}
                    }
                }
            }
            mount = {"filesystems": [{"source": "/dev/nvme0n1p1", "target": "/"}]}
            ebs = {
                "blockdevices": [
                    {
                        "type": "part",
                        "children": [
                            {"type": "disk", "model": "Amazon Elastic Block Store"}
                        ],
                    }
                ]
            }
            with patch.object(checks, "command_json", side_effect=[mount, ebs]):
                with self.assertRaisesRegex(ValueError, "instance storage"):
                    checks.verify_spill(config)
            with patch.object(checks, "command_json", side_effect=[mount, ebs]):
                self.assertTrue(
                    checks.verify_spill(config, allow_ebs=True)[0]["allow_ebs"]
                )

    def test_query_selection_rejects_duplicates_and_bad_ranges(self):
        self.assertEqual(replay.queries("3,6,9,18"), [3, 6, 9, 18])
        for value in ("0", "23", "8-4", "1,1", "1-3,2", "1-2-3"):
            with self.subTest(value=value), self.assertRaises(
                replay.argparse.ArgumentTypeError
            ):
                replay.queries(value)


if __name__ == "__main__":
    unittest.main()
