"""Developer-facing contracts: portable evidence, exact replay and process isolation."""

import argparse
import json
import os
import pathlib
import signal
import tempfile
import time
import unittest
from dataclasses import asdict
from unittest.mock import patch

from . import conftest_path  # noqa: F401
from .test_evaluator import FakeSession, gpu_variant_wrong
from siriusfuzz.artifacts import runtime_info, write_json
from siriusfuzz.cli import (
    build_parser,
    cmd_recheck,
    cmd_replay,
    cmd_selftest,
    validate_limits,
)
from siriusfuzz.config import load_config
from siriusfuzz.isolation import supervise
from siriusfuzz.report import QueryRecord, Report
from siriusfuzz.runner import Evaluator, Mailbox, Orchestrator, OrchestratorOptions
from siriusfuzz.session import RunResult, Session, SessionError


def crash_child(payload, work):
    os.kill(os.getpid(), signal.SIGKILL)


def hang_child(payload, work):
    time.sleep(30)


def ok_child(payload, work):
    write_json(pathlib.Path(work) / "result.json", {"status": "ok"})


def completed_then_crash_child(payload, work):
    write_json(pathlib.Path(work) / "result.json", {"status": "ok"})
    os.kill(os.getpid(), signal.SIGKILL)


def completed_then_hang_child(payload, work):
    write_json(pathlib.Path(work) / "result.json", {"status": "ok"})
    time.sleep(30)


def restarting_worker(args, cfg, out, stop):
    """Two incarnations produce mismatches against different datasets."""
    stem = f"w0-s{args.spawn_id}-d0"
    dataset = pathlib.Path(args.run_dir) / "datasets" / f"{stem}.sql"
    dataset.write_text(
        f"CREATE TABLE t(k INTEGER); INSERT INTO t VALUES ({args.spawn_id});"
    )

    def send(message):
        out.put({**message, "worker": args.worker_id, "spawn": args.spawn_id})

    record = QueryRecord(0, stem, args.seed, "SELECT k FROM t", "mismatch")
    send({"type": "begin", "sql": record.sql, "dataset": stem})
    send({"type": "result", "record": asdict(record)})
    send({"type": "idle"})
    if args.spawn_id == 1:
        send({"type": "begin", "sql": "SELECT k + 1 FROM t", "dataset": stem})
        os._exit(9)
    send({"type": "done", "reason": "finished"})


def startup_dead_worker(args, cfg, out, stop):
    os._exit(9)


class ToolTests(unittest.TestCase):
    def test_cpu_only_session_does_not_select_ambient_sirius_configuration(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"SIRIUS_CONFIG_FILE": "ambient.yaml"}
        ), patch("duckdb.connect"):
            session = Session(None, None, pathlib.Path(tmp), 0)
            session.open()
            self.assertEqual(runtime_info(session)["sirius_config_mode"], "cpu_only")
            self.assertEqual(os.environ["SIRIUS_CONFIG_FILE"], "ambient.yaml")
            session.close()

    def test_no_yaml_session_rejects_ambient_configuration_before_connecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cwd, home = root / "cwd", root / "home"
            cwd.mkdir()
            (home / ".sirius").mkdir(parents=True)
            for location in ("env", "empty_env", "cwd", "home"):
                with self.subTest(location=location), patch.dict(
                    os.environ, {"HOME": str(home)}
                ), patch("pathlib.Path.cwd", return_value=cwd), patch(
                    "duckdb.connect"
                ) as connect:
                    os.environ.pop("SIRIUS_CONFIG_FILE", None)
                    if location in ("env", "empty_env"):
                        os.environ["SIRIUS_CONFIG_FILE"] = (
                            str(root / "ambient.yaml") if location == "env" else ""
                        )
                    else:
                        config = (
                            cwd / "sirius.yaml"
                            if location == "cwd"
                            else home / ".sirius/sirius.yaml"
                        )
                        config.write_text("# ambient configuration\n")
                    session = Session("synthetic-extension", None, root / "db", 0)
                    with self.assertRaisesRegex(
                        SessionError, "ambient Sirius configuration"
                    ):
                        session.open()
                    connect.assert_not_called()
                    if location in ("cwd", "home"):
                        config.unlink()

    def test_session_records_defaults_or_explicit_yaml_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            selected = root / "selected.yaml"
            selected.write_text("# explicitly selected snapshot\n")
            for explicit in (False, True):
                with self.subTest(explicit=explicit), patch.dict(
                    os.environ, {"HOME": str(root)}
                ), patch("pathlib.Path.cwd", return_value=root), patch(
                    "duckdb.connect"
                ):
                    os.environ.pop("SIRIUS_CONFIG_FILE", None)
                    if explicit:
                        os.environ["SIRIUS_CONFIG_FILE"] = str(root / "ambient.yaml")
                        (root / "sirius.yaml").write_text("# not selected\n")
                    session = Session(
                        "synthetic-extension",
                        str(selected) if explicit else None,
                        root / "db",
                        0,
                    )
                    session.open()
                    info = runtime_info(session)
                    self.assertEqual(
                        info["sirius_config_mode"],
                        "explicit_yaml" if explicit else "builtin_defaults",
                    )
                    if explicit:
                        self.assertEqual(
                            os.environ["SIRIUS_CONFIG_FILE"], str(selected.resolve())
                        )
                        self.assertEqual(info["sirius_config"], str(selected.resolve()))
                    else:
                        self.assertNotIn("SIRIUS_CONFIG_FILE", os.environ)
                        self.assertIsNone(info["sirius_config"])
                    session.close()

    def test_selftest_bounds_default_data_and_preserves_explicit_overrides(self):
        defaults = build_parser().parse_args(["selftest", "--queries", "100"])
        with patch("siriusfuzz.cli.cmd_run", return_value=0):
            self.assertEqual(cmd_selftest(defaults), 0)
        default_cfg = load_config(None, defaults.set)
        self.assertEqual(default_cfg.data.rows, [8, 16])
        self.assertEqual(default_cfg.features.joins.max_tables, 2)
        self.assertEqual(default_cfg.features.subqueries.max_depth, 1)

        args = build_parser().parse_args(
            ["selftest", "--queries", "100", "--set", "data.rows=[20,30]"]
        )
        with patch("siriusfuzz.cli.cmd_run", return_value=0) as run:
            self.assertEqual(cmd_selftest(args), 0)
        run.assert_called_once_with(args)
        cfg = load_config(None, args.set)
        self.assertEqual(cfg.data.rows, [20, 30])
        self.assertEqual(cfg.features.joins.max_tables, 2)
        self.assertEqual(cfg.features.subqueries.max_depth, 1)

    def test_interrupt_during_process_start_records_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("siriusfuzz.isolation.mp.get_context") as context:
                process = context.return_value.Process.return_value
                process.start.side_effect = KeyboardInterrupt
                process.pid = None
                result = supervise({}, pathlib.Path(tmp), 5)
                self.assertEqual(result["status"], "cancelled")
                self.assertEqual(
                    json.loads((pathlib.Path(tmp) / "outcome.json").read_text())[
                        "status"
                    ],
                    "cancelled",
                )
                process.kill.assert_not_called()

    def test_startup_crash_does_not_exhaust_query_restart_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(None)
            report = Report(pathlib.Path(tmp), cfg, [], 1)
            runner = Orchestrator(
                cfg,
                report,
                1,
                OrchestratorOptions(max_queries=1, max_respawns=200, quiet=True),
            )
            with patch("siriusfuzz.runner.worker_main", startup_dead_worker):
                summary = runner.run()
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(runner.spawn_count, 1)
            self.assertEqual(summary["queries"], 0)

    def test_worker_restart_preserves_both_datasets_and_completed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cfg = load_config(None)
            report = Report(root, cfg, [], 1)
            runner = Orchestrator(
                cfg,
                report,
                1,
                OrchestratorOptions(max_queries=3, max_respawns=1, quiet=True),
            )
            with patch("siriusfuzz.runner.worker_main", restarting_worker):
                summary = runner.run()
            self.assertEqual(summary["status"], "complete", summary)
            self.assertEqual(summary["counts"], {"mismatch": 2, "crash": 1})
            bundles = [
                p
                for p in (root / "findings").rglob("meta.json")
                if json.loads(p.read_text())["verdict"] == "mismatch"
            ]
            self.assertEqual(len(bundles), 2)
            self.assertNotEqual(
                (bundles[0].parent / "dataset.sql").read_text(),
                (bundles[1].parent / "dataset.sql").read_text(),
            )

    def test_reduction_does_not_accept_an_unrelated_runtime_error(self):
        cfg = load_config(None)
        session = FakeSession(
            lambda *args: RunResult(
                "error", error="Sirius GPU execution failed: unrelated operator failure"
            )
        )
        ev = Evaluator(cfg, session, lambda message: None)
        record = QueryRecord(
            0,
            "dataset",
            0,
            "SELECT 1",
            "gpu_error",
            reason="original expression failure",
        )
        predicate = ev._make_still_fails(record)
        self.assertIsNotNone(predicate)
        self.assertFalse(predicate("SELECT 2"))

    def test_mailbox_ignores_interrupted_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = Mailbox(pathlib.Path(tmp))
            (pathlib.Path(tmp) / "partial.json.tmp").write_text('{"unfinished":')
            box.put({"type": "result", "record": "complete"})
            self.assertEqual(box.get_nowait()["record"], "complete")

    def test_cancellation_preserves_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(None)
            report = Report(pathlib.Path(tmp), cfg, [], 1)
            report.add(QueryRecord(0, "missing", 1, "SELECT 1", "mismatch"))
            runner = Orchestrator(cfg, report, 1, OrchestratorOptions())
            with patch.object(runner, "_run", side_effect=KeyboardInterrupt):
                summary = runner.run()
            self.assertEqual(summary["status"], "cancelled")
            self.assertEqual(summary["counts"]["mismatch"], 1)
            self.assertTrue((pathlib.Path(tmp) / "summary.json").is_file())

    def test_replay_accepts_decimal_lists_and_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "config.toml").write_text(load_config(None).to_toml())
            (root / "dataset.sql").write_text(
                "CREATE TABLE t(k INT); INSERT INTO t VALUES (1);"
            )
            (root / "query.sql").write_text(
                "SELECT 1.25::DECIMAL(12,2) AS scalar_decimal, "
                "[1.25::DECIMAL(12,2), NULL] AS decimal_list, "
                "[1.25, NULL]::DECIMAL(12,2)[2] AS decimal_array, "
                "[[1.25, NULL], NULL]::DECIMAL(12,2)[][] AS nested_list FROM t"
            )
            outcome = supervise(
                {
                    "operation": "replay",
                    "query": str(root / "query.sql"),
                    "dataset": str(root / "dataset.sql"),
                    "config": str(root / "config.toml"),
                },
                root,
                30,
            )
            self.assertEqual(outcome["status"], "ok", outcome)
            self.assertEqual(outcome["record"]["verdict"], "ok")
            columns = outcome["record"]["evidence"]["cpu"]["columns"]
            self.assertEqual(
                [col["type"] for col in columns],
                [
                    "DECIMAL(12,2)",
                    "DECIMAL(12,2)[]",
                    "DECIMAL(12,2)[2]",
                    "DECIMAL(12,2)[][]",
                ],
            )

    def test_replay_parses_comments_and_quoted_semicolons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "dataset.sql").write_text(
                "-- header\nCREATE TABLE t(k VARCHAR);\nINSERT INTO t VALUES ('a;\nb'), ('--literal');\nCHECKPOINT;\n"
            )
            (root / "query.sql").write_text(
                "-- query comment\nSELECT k FROM t ORDER BY k;"
            )
            (root / "config.toml").write_text(load_config(None).to_toml())
            outcome = supervise(
                {
                    "operation": "replay",
                    "query": str(root / "query.sql"),
                    "dataset": str(root / "dataset.sql"),
                    "config": str(root / "config.toml"),
                    "comparison": "ordered",
                },
                root,
                30,
            )
            self.assertEqual(outcome["status"], "ok", outcome)
            self.assertEqual(outcome["record"]["verdict"], "ok")
            self.assertEqual(outcome["record"]["evidence"]["cpu"]["row_count"], 2)

    def test_supervision_handles_native_death_and_deadline(self):
        for target, expected, timeout in (
            (ok_child, "ok", 10),
            (crash_child, "crash", 10),
            (hang_child, "timeout", 0.5),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                result = supervise({}, pathlib.Path(tmp), timeout, target)
                self.assertEqual(result["status"], expected)
                self.assertTrue((pathlib.Path(tmp) / "outcome.json").exists())
                self.assertLess(result["elapsed_seconds"], timeout + 6)

    def test_supervision_preserves_completed_result_after_cleanup_failure(self):
        for target, timeout in (
            (completed_then_crash_child, 10),
            (completed_then_hang_child, 1),
        ):
            with self.subTest(
                target=target.__name__
            ), tempfile.TemporaryDirectory() as tmp:
                result = supervise({}, pathlib.Path(tmp), timeout, target)
                self.assertEqual(result["status"], "cleanup_error")
                self.assertEqual(result["query_result"]["status"], "ok")
                self.assertEqual(
                    result["cleanup"]["status"],
                    "crash" if target == completed_then_crash_child else "timeout",
                )
                self.assertEqual(
                    json.loads((pathlib.Path(tmp) / "outcome.json").read_text())[
                        "status"
                    ],
                    "cleanup_error",
                )

    def test_supervision_does_not_reuse_stale_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_json(pathlib.Path(tmp) / "result.json", {"status": "ok"})
            result = supervise({}, pathlib.Path(tmp), 10, crash_child)
            self.assertEqual(result["status"], "crash")

    def test_replay_restores_bundle_and_does_not_modify_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            bundle = root / "copied-bundle"
            bundle.mkdir()
            (bundle / "query.sql").write_text("SELECT k FROM t;")
            (bundle / "dataset.sql").write_text(
                "CREATE TABLE t(k INT); INSERT INTO t VALUES (1);"
            )
            cfg = load_config(None, ["oracle.float64_rel_tol=0.0007"])
            (bundle / "config.toml").write_text(cfg.to_toml())
            write_json(
                bundle / "meta.json",
                {
                    "comparison": "ordered",
                    "variant": {"hash_partition_bytes": 123},
                    "execution": {"cpu_only": True},
                },
            )
            before = {p.name: p.read_bytes() for p in bundle.iterdir()}
            args = build_parser().parse_args(
                ["replay", str(bundle), "--out", str(root / "out")]
            )
            with patch(
                "siriusfuzz.cli.supervise",
                return_value={"status": "ok", "record": {"verdict": "ok"}},
            ) as probe, patch("siriusfuzz.cli.provenance", return_value={}):
                self.assertEqual(cmd_replay(args), 0)
            payload = probe.call_args.args[0]
            self.assertEqual(payload["variant"], {"hash_partition_bytes": 123})
            self.assertEqual(payload["comparison"], "ordered")
            self.assertEqual(
                load_config(payload["config"]).oracle.float64_rel_tol, 0.0007
            )
            self.assertEqual(before, {p.name: p.read_bytes() for p in bundle.iterdir()})

    def test_replay_does_not_invent_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = pathlib.Path(tmp)
            (bundle / "query.sql").write_text("SELECT k FROM t")
            args = build_parser().parse_args(["replay", str(bundle), "--cpu-only"])
            with self.assertRaisesRegex(ValueError, "missing dataset.sql"):
                cmd_replay(args)

    def test_gpu_bundle_replay_respects_recorded_yaml_requirement(self):
        for required, override in (
            (False, False),
            (True, False),
            ("legacy", False),
            (True, True),
            ("legacy", True),
        ):
            with self.subTest(
                required=required, override=override
            ), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                bundle = root / "bundle"
                bundle.mkdir()
                (bundle / "query.sql").write_text("SELECT k FROM t;")
                (bundle / "dataset.sql").write_text(
                    "CREATE TABLE t(k INT); INSERT INTO t VALUES (1);"
                )
                cfg = load_config(None, ["sirius.configs=[]"])
                (bundle / "config.toml").write_text(cfg.to_toml())
                execution = {"cpu_only": False}
                if required != "legacy":
                    execution["sirius_config_required"] = required
                write_json(bundle / "meta.json", {"execution": execution})
                before = {p.name: p.read_bytes() for p in bundle.iterdir()}
                extension = root / "synthetic-extension"
                extension.write_text("unit test; never loaded")
                options = [
                    "replay",
                    str(bundle),
                    "--extension",
                    str(extension),
                    "--out",
                    str(root / "out"),
                ]
                yaml = root / "override.yaml"
                if override:
                    yaml.write_text("# synthetic config\n")
                    options += ["--sirius-config", str(yaml)]
                args = build_parser().parse_args(options)
                with patch(
                    "siriusfuzz.cli.supervise",
                    return_value={"status": "ok", "record": {"verdict": "ok"}},
                ) as probe, patch("siriusfuzz.cli.provenance", return_value={}):
                    if required is not False and not override:
                        with self.assertRaisesRegex(
                            ValueError, "missing saved Sirius YAML"
                        ):
                            cmd_replay(args)
                        probe.assert_not_called()
                    else:
                        self.assertEqual(cmd_replay(args), 0)
                        payload, work, _ = probe.call_args.args
                        self.assertEqual(payload["extension"], str(extension))
                        self.assertEqual(bool(payload["sirius_config"]), override)
                        if override:
                            self.assertEqual(
                                pathlib.Path(payload["sirius_config"]).read_bytes(),
                                yaml.read_bytes(),
                            )
                        else:
                            self.assertEqual(
                                load_config(payload["config"]).sirius.configs, []
                            )
                            self.assertIsNone(payload["sirius_config"])
                        recorded = json.loads((work / "meta.json").read_text())[
                            "execution"
                        ]
                        self.assertFalse(recorded["cpu_only"])
                        self.assertEqual(recorded["sirius_config_required"], override)
                self.assertEqual(
                    before, {p.name: p.read_bytes() for p in bundle.iterdir()}
                )

    def test_recheck_replays_every_finding_and_reports_what_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            run = root / "run"
            for name, verdict in (
                ("000-mismatch-aa", "mismatch"),
                ("001-gpu_error-bb", "gpu_error"),
            ):
                bundle = run / "findings" / name
                bundle.mkdir(parents=True)
                (bundle / "query.sql").write_text("SELECT k FROM t;")
                (bundle / "reduced.sql").write_text("SELECT k FROM t WHERE k = 1;")
                (bundle / "dataset.sql").write_text(
                    "CREATE TABLE t(k INT); INSERT INTO t VALUES (1);"
                )
                (bundle / "config.toml").write_text(load_config(None).to_toml())
                write_json(
                    bundle / "meta.json",
                    {"verdict": verdict, "execution": {"cpu_only": True}},
                )
            (run / "findings" / "000-mismatch-aa" / "additional").mkdir()
            outcomes = iter(
                [
                    {"status": "ok", "record": {"verdict": "ok", "reason": ""}},
                    {
                        "status": "ok",
                        "record": {"verdict": "gpu_error", "reason": "still broken"},
                    },
                ]
            )
            args = build_parser().parse_args(
                ["recheck", str(run), "--cpu-only", "--out", str(root / "out")]
            )
            with patch(
                "siriusfuzz.cli.supervise", side_effect=lambda *a, **k: next(outcomes)
            ) as probe, patch("siriusfuzz.cli.provenance", return_value={}):
                self.assertEqual(cmd_recheck(args), 1)
            self.assertEqual(probe.call_count, 2)
            # reduced.sql is what gets replayed, each bundle on its own copy of args.
            for call in probe.call_args_list:
                payload = call.args[0]
                self.assertEqual(
                    pathlib.Path(payload["query"]).read_text(),
                    "SELECT k FROM t WHERE k = 1;",
                )
            report = json.loads(
                next((root / "out").glob("run-*/recheck.json")).read_text()
            )
            self.assertEqual(report["cleared"], 1)
            self.assertEqual(
                [(r["recorded"], r["now"]) for r in report["findings"]],
                [("mismatch", "ok"), ("gpu_error", "gpu_error")],
            )
            with patch(
                "siriusfuzz.cli.supervise",
                return_value={
                    "status": "ok",
                    "record": {"verdict": "ok", "reason": ""},
                },
            ), patch("siriusfuzz.cli.provenance", return_value={}):
                self.assertEqual(cmd_recheck(args), 0)

    def test_forced_variant_ignores_random_variant_budget(self):
        cfg = load_config(
            None, ["variants.per_query=0", "oracle.ambiguity_filter=false"]
        )
        session = FakeSession(gpu_variant_wrong)
        ev = Evaluator(cfg, session, lambda message: None)
        ev.forced_variant = {"hash_partition_bytes": 1024}
        ev.variants = {}
        record = ev.evaluate(None, "SELECT 1", 0, "dataset", 0)
        self.assertEqual(record.verdict, "variant_mismatch")
        self.assertEqual(record.variant, ev.forced_variant)
        self.assertEqual(session.settings, {})

    def test_metadata_override_is_explicit(self):
        class Connection:
            calls = []

            def execute(self, sql):
                self.calls.append(sql)
                if len(self.calls) == 1:
                    raise RuntimeError("built specifically for DuckDB version abc")

        s = Session("extension", None, pathlib.Path("unused"), 0)
        s.con = Connection()
        with self.assertRaises(SessionError):
            s._load_extension()
        self.assertEqual(len(s.con.calls), 1)

    def test_invalid_limits(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                validate_limits(argparse.Namespace(timeout=value))


if __name__ == "__main__":
    unittest.main()
