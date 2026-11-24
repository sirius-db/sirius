# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for pinned recipe selection and measurement boundaries."""

import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("guide_reproduction", HERE / "replay.py")
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


class Recipes(unittest.TestCase):
    def test_sf3k_uses_sf10k_config_and_threads(self):
        cold, session = reproduction.recipe(3000, "cold")
        self.assertEqual(cold, reproduction.recipe(10000, "cold")[0])
        self.assertIn("threads = 24", session)
        self.assertIn("fallback = false", session)
        self.assertTrue(
            all(
                s["config"] == str(cold)
                for s in reproduction.plan(3000, "cold", [1, 2], 1)["steps"]
            )
        )

    def test_hot_changes_only_eviction(self):
        cold, _ = reproduction.recipe(3000, "cold")
        hot, _ = reproduction.recipe(3000, "hot")
        self.assertEqual(
            cold.read_bytes().replace(b"eviction: idle", b"eviction: lru"),
            hot.read_bytes(),
        )

    def test_small_presets_keep_implicit_engine_defaults(self):
        cold, session = reproduction.recipe(100, "cold")
        self.assertNotIn("threads =", session)
        self.assertNotIn("operator_params", cold.read_text())
        self.assertEqual(cold.name, "cpu-budget-cold.yaml")
        self.assertEqual(
            reproduction.recipe(100, "pinned")[0].read_bytes(),
            reproduction.recipe(100, "hot")[0].read_bytes(),
        )

    def test_cold_is_one_process_per_query_iteration(self):
        p = reproduction.plan(3000, "cold", [1, 3], 2)
        self.assertEqual(
            [(s["queries"], s["iterations"]) for s in p["steps"]],
            [([1], [1]), ([3], [1]), ([1], [2]), ([3], [2])],
        )
        self.assertEqual(p["worker_mode"], "isolated")
        self.assertIsNone(p["harness_profile"])

    def test_hot_is_one_process_per_query_with_paired_iterations(self):
        p = reproduction.plan(3000, "hot", [1, 3], 2)
        self.assertEqual(
            [(s["queries"], s["iterations"]) for s in p["steps"]],
            [([1], [1, 2]), ([3], [1, 2])],
        )
        self.assertEqual(p["worker_mode"], "grouped")
        self.assertFalse(p["warmup"])

    def test_incomplete_iteration_never_gets_full_suite_sum(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = reproduction.report(
                Path(tmp), [1, 2], 2, {(1, 1): 3, (2, 1): 4, (1, 2): 1}, [], []
            )
        self.assertEqual(s["iteration_totals"][0]["sum_s"], 7)
        self.assertIsNone(s["iteration_totals"][1]["sum_s"])
        self.assertEqual(s["iteration_totals"][1]["available_sum_s"], 1)
        self.assertEqual(s["best_complete_iteration_sum_s"], 7)
        self.assertFalse(s["cpu_validated"])
        self.assertFalse(s["complete"])

    def test_dry_run_does_not_create_output_or_require_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must-not-exist"
            result = subprocess.run(
                [
                    "bash",
                    str(HERE / "replay.sh"),
                    "--sf",
                    "3000",
                    "--profile",
                    "hot",
                    "--source-dir",
                    "/nonexistent-engine",
                    "--input",
                    "s3://unused/data/",
                    "--output",
                    str(output),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            p = json.loads(result.stdout)
            self.assertEqual(len(p["steps"]), 22)
            self.assertFalse(output.exists())

    def test_zero_iterations_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            reproduction.main(
                [
                    "--sf",
                    "3000",
                    "--profile",
                    "hot",
                    "--source-dir",
                    "/unused",
                    "--input",
                    "s3://unused/data/",
                    "--iterations",
                    "0",
                    "--dry-run",
                ]
            )

    def test_public_entrypoints_select_the_same_guide_recipe(self):
        repo = HERE.parents[2]
        common = [
            "--source-dir",
            "/unused-engine",
            "--input",
            "s3://unused/data/",
            "--dry-run",
        ]
        commands = [
            ["bash", str(repo / "bench/g7e/replay.sh"), "3000", "hot", *common],
            ["bash", str(repo / "bench/replay.sh"), "g7e", "3000", "hot", *common],
        ]
        plans = [
            json.loads(subprocess.check_output(cmd, text=True)) for cmd in commands
        ]
        self.assertEqual(plans[0], plans[1])
        self.assertEqual(Path(plans[0]["config"]).name, "balanced-hot.yaml")
        self.assertEqual(len(plans[0]["steps"]), 22)
        self.assertTrue(
            all(s["config"] == plans[0]["config"] for s in plans[0]["steps"])
        )

    def test_custom_config_override_is_rejected(self):
        command = [
            "bash",
            str(HERE.parent / "replay.sh"),
            "3000",
            "cold",
            "--source-dir",
            "/unused-engine",
            "--input",
            "s3://unused/data/",
            "--config",
            "/unused-custom.yaml",
            "--dry-run",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments: --config", result.stderr)

    def test_no_competing_g7e_config_or_result_folders(self):
        self.assertFalse((HERE.parent / "configs").exists())
        self.assertEqual(
            {p.name for p in (HERE.parent / "results").iterdir()},
            {"guide-ktls-20261123"},
        )

    def test_selected_results_match_all_22_rows_and_exact_config_hashes(self):
        snapshot = HERE.parent / "results/guide-ktls-20261123"
        manifest = json.loads((snapshot / "manifest.json").read_text())
        with (snapshot / "per-query.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 506)
        for cell in manifest["selected"]:
            selected = [
                r
                for r in rows
                if int(r["scale_factor"]) == cell["scale_factor"]
                and r["profile"] == cell["profile"]
                and int(r["iteration"]) == cell["iteration"]
            ]
            self.assertEqual({int(r["query"]) for r in selected}, set(range(1, 23)))
            self.assertTrue(all(r["cpu_validated"] == "True" for r in selected))
            self.assertAlmostEqual(
                sum(float(r["seconds"]) for r in selected), cell["sum_s"]
            )
            self.assertEqual(
                hashlib.sha256(
                    (snapshot / cell["configuration"]).read_bytes()
                ).hexdigest(),
                cell["config_sha256"],
            )

    def test_gpu_evidence_uses_each_iteration_window(self):
        from collect import gpu_evidence

        epoch = int(datetime(2026, 11, 23, tzinfo=timezone.utc).timestamp() * 1e9)
        journal = [
            {
                "event": event,
                "phase": "measurement",
                "query": 1,
                "iteration": i,
                "wall_ns": epoch + ms * 1_000_000,
            }
            for i, start in [(1, 0), (2, 20)]
            for event, ms in [("query_begin", start), ("query_end", start + 10)]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            worker = Path(tmp)
            log = worker / "benchmark/tpch_run/log_dir/native.log"
            log.parent.mkdir(parents=True)
            lines = [
                f"[2026-11-23 00:00:00.{ms:03}] {text}\n"
                for ms, text in [
                    (1, "physical plan replaced with GPU operator"),
                    (9, "Transparent GPU execution: query completed"),
                    (21, "physical plan replaced with GPU operator"),
                    (29, "Transparent GPU execution: query completed"),
                ]
            ]
            log.write_text("".join(lines))
            self.assertTrue(gpu_evidence(worker, 1, 1, journal)["correct"])
            self.assertTrue(gpu_evidence(worker, 1, 2, journal)["correct"])
            log.write_text("".join(lines[:2]))
            self.assertFalse(gpu_evidence(worker, 1, 2, journal)["correct"])

    def test_ktls_requires_new_connections_and_no_fallback(self):
        from ktls_guard import validate

        before = {"module_loaded": True, "counters": {"TlsTxSw": 100, "TlsRxSw": 100}}
        after = {"module_loaded": True, "counters": {"TlsTxSw": 101, "TlsRxSw": 101}}
        with tempfile.TemporaryDirectory() as tmp:
            worker = Path(tmp)
            log = worker / "benchmark/run/log_dir/native.log"
            log.parent.mkdir(parents=True)
            log.write_text("native connection\n")
            self.assertFalse(validate(worker, before, before)["correct"])
            self.assertTrue(validate(worker, before, after)["correct"])
            log.write_text("kTLS not engaged\n")
            self.assertFalse(validate(worker, before, after)["correct"])

    def test_interrupt_cleans_up_the_real_child_process(self):
        import supervisor

        children = []
        real_popen = subprocess.Popen

        def launch(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            children.append(process)
            return process

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = root / "engine"
            engine.mkdir()
            (root / "session.sql").write_text("SET enable_duckdb_fallback=false;\n")
            worker = root / "fake.py"
            worker.write_text("import time; time.sleep(30)\n")
            with patch.multiple(
                supervisor,
                ROOT=root,
                ENGINE=engine,
                REGION="eu-west-2",
                INPUT="s3://unused/data/",
                WORKER=worker,
            ), patch.object(supervisor, "storage_for", return_value={}), patch.object(
                supervisor, "tls_snapshot", return_value={}
            ), patch.object(
                supervisor.os, "sync"
            ), patch.object(
                subprocess, "check_output", return_value=""
            ), patch.object(
                subprocess, "run"
            ), patch.object(
                subprocess, "Popen", side_effect=launch
            ), patch.object(
                supervisor.queue.Queue, "get", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    supervisor.run_worker(
                        3000, "hot", [1], HERE / "configs/balanced-hot.yaml"
                    )
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())


if __name__ == "__main__":
    unittest.main()
