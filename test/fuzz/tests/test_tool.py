"""Developer-facing contracts: the shell session, exact replay, recheck and the orchestrator."""

import argparse
import json
import os
import pathlib
import signal
import tempfile
import threading
import unittest
from dataclasses import asdict
from unittest.mock import patch

from . import conftest_path
from .test_evaluator import FakeSession, gpu_variant_wrong
from siriusfuzz.artifacts import write_json
from siriusfuzz.cli import (
    _engine,
    build_parser,
    cmd_recheck,
    cmd_replay,
    cmd_selftest,
    validate_limits,
)
from siriusfuzz.config import load_config
from siriusfuzz.probe import execute_probe
from siriusfuzz.report import QueryRecord, Report
from siriusfuzz.runner import Evaluator, Mailbox, Orchestrator, OrchestratorOptions
from siriusfuzz.session import (
    RunResult,
    Session,
    _strip_fallback_banners,
    discover_sirius_yaml,
    single_select,
)

SHELL = conftest_path.available_shell()


class ShellOutputTests(unittest.TestCase):
    def test_sirius_fallback_banner_does_not_corrupt_json_result(self):
        output, banners = _strip_fallback_banners(
            [
                '[{"1":1}]\n',
                "=============================================\n",
                "Error in Sirius GPU execution, fallback to DuckDB\n",
                "=============================================\n",
            ]
        )
        self.assertEqual(json.loads("".join(output)), [{"1": 1}])
        self.assertIn("fallback to DuckDB", "".join(banners))

    def test_other_stdout_is_not_discarded(self):
        output, banners = _strip_fallback_banners(
            ['[{"1":1}]\n', "unexpected output\n"]
        )
        self.assertEqual(output, ['[{"1":1}]\n', "unexpected output\n"])
        self.assertEqual(banners, [])


class fake_shell:
    """A shell path that exists but is never run, for CLI tests that patch the subprocess."""

    def __init__(self, root):
        self.path = root / "duckdb"
        self.path.write_text("unit test; never run")
        self.patcher = patch("siriusfuzz.cli.check_shell", return_value="fake")

    def __enter__(self):
        self.patcher.start()
        return str(self.path)

    def __exit__(self, *exc):
        self.patcher.stop()


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
    def test_cpu_only_session_ignores_sirius_yaml(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"SIRIUS_CONFIG_FILE": "ambient.yaml"}
        ):
            session = Session("duckdb", None, None, pathlib.Path(tmp), 0, cpu_only=True)
            session._configure_sirius()
            self.assertEqual(session.sirius_config_mode, "cpu_only")
            self.assertEqual(os.environ["SIRIUS_CONFIG_FILE"], "ambient.yaml")

    def test_sirius_yaml_discovery_follows_sirius_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            cwd, home = root / "cwd", root / "home"
            cwd.mkdir()
            (home / ".sirius").mkdir(parents=True)
            for location, source in (
                ("env", "SIRIUS_CONFIG_FILE"),
                ("cwd", "current directory"),
                ("home", "home directory"),
                ("none", "none"),
            ):
                with self.subTest(location=location), patch.dict(
                    os.environ, {"HOME": str(home)}
                ), patch("pathlib.Path.cwd", return_value=cwd):
                    os.environ.pop("SIRIUS_CONFIG_FILE", None)
                    expected = {
                        "env": str(root / "ambient.yaml"),
                        "cwd": str(cwd / "sirius.yaml"),
                        "home": str(home / ".sirius/sirius.yaml"),
                        "none": None,
                    }[location]
                    if location == "env":
                        os.environ["SIRIUS_CONFIG_FILE"] = expected
                    elif expected:
                        pathlib.Path(expected).write_text("# ambient\n")
                    self.assertEqual(discover_sirius_yaml(), (expected, source))
                    session = Session("duckdb", "synthetic-extension", None, root, 0)
                    session._configure_sirius()
                    if expected:
                        self.assertEqual(
                            session.sirius_config_mode, f"ambient ({source})"
                        )
                        self.assertEqual(session.sirius_config, expected)
                        self.assertEqual(os.environ["SIRIUS_CONFIG_FILE"], expected)
                    else:
                        self.assertEqual(session.sirius_config_mode, "builtin_defaults")
                        self.assertNotIn("SIRIUS_CONFIG_FILE", os.environ)
                    if location in ("cwd", "home"):
                        pathlib.Path(expected).unlink()
            # An empty variable counts as unset; an explicit file wins over everything.
            with patch.dict(
                os.environ, {"SIRIUS_CONFIG_FILE": "", "HOME": str(home)}
            ), patch("pathlib.Path.cwd", return_value=cwd):
                self.assertEqual(discover_sirius_yaml(), (None, "none"))
            selected = root / "selected.yaml"
            selected.write_text("# selected\n")
            with patch.dict(
                os.environ, {"SIRIUS_CONFIG_FILE": str(root / "ambient.yaml")}
            ):
                session = Session(
                    "duckdb", "synthetic-extension", str(selected), root, 0
                )
                session._configure_sirius()
                self.assertEqual(session.sirius_config_mode, "explicit_yaml")
                self.assertEqual(
                    os.environ["SIRIUS_CONFIG_FILE"], str(selected.resolve())
                )

    def test_engine_snapshots_the_ambient_yaml_when_the_configuration_names_none(self):
        with tempfile.TemporaryDirectory() as tmp, fake_shell(
            pathlib.Path(tmp)
        ) as shell:
            root = pathlib.Path(tmp)
            extension = root / "sirius.duckdb_extension"
            extension.write_text("unit test; never loaded")
            yaml = root / "ambient.yaml"
            yaml.write_text("# ambient\n")
            cfg = load_config(None, ["sirius.configs=[]"])
            args = argparse.Namespace(
                cpu_only=False,
                shell=shell,
                extension=str(extension),
                sirius_config=None,
            )
            with patch.dict(os.environ, {"SIRIUS_CONFIG_FILE": str(yaml)}):
                engine = _engine(args, cfg)
                self.assertEqual(
                    (engine.shell, engine.extension), (shell, str(extension))
                )
                self.assertEqual(engine.configs, [str(yaml)])
            with patch.dict(
                os.environ, {"SIRIUS_CONFIG_FILE": "", "HOME": str(root)}
            ), patch("pathlib.Path.cwd", return_value=root):
                self.assertEqual(_engine(args, cfg).configs, [])
            args.cpu_only = True
            self.assertEqual(_engine(args, cfg).extension, None)

    def test_single_select_accepts_one_query_with_comments(self):
        self.assertTrue(single_select("-- note\nSELECT 'a;b' FROM t; "))
        self.assertTrue(single_select("WITH c AS (SELECT 1) /* x; */ SELECT * FROM c"))
        self.assertFalse(single_select("SELECT 1; SELECT 2"))
        self.assertFalse(single_select("CREATE TABLE t(k INT)"))

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

    @unittest.skipUnless(SHELL, "no DuckDB shell available")
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
            outcome = execute_probe(
                {
                    "operation": "replay",
                    "timeout": 60,
                    "shell": SHELL,
                    "cpu_only": True,
                    "query": str(root / "query.sql"),
                    "dataset": str(root / "dataset.sql"),
                    "config": str(root / "config.toml"),
                },
                root,
            )
            self.assertEqual(outcome["status"], "ok", outcome)
            self.assertEqual(outcome["record"]["verdict"], "ok")
            self.assertEqual(outcome["record"]["evidence"]["cpu"]["row_count"], 1)

    @unittest.skipUnless(SHELL, "no DuckDB shell available")
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
            outcome = execute_probe(
                {
                    "operation": "replay",
                    "timeout": 60,
                    "shell": SHELL,
                    "cpu_only": True,
                    "query": str(root / "query.sql"),
                    "dataset": str(root / "dataset.sql"),
                    "config": str(root / "config.toml"),
                    "comparison": "ordered",
                },
                root,
            )
            self.assertEqual(outcome["status"], "ok", outcome)
            self.assertEqual(outcome["record"]["verdict"], "ok")
            self.assertEqual(outcome["record"]["evidence"]["cpu"]["row_count"], 2)

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
            with fake_shell(root) as shell, patch(
                "siriusfuzz.cli.execute_probe",
                return_value={"status": "ok", "record": {"verdict": "ok"}},
            ) as probe, patch("siriusfuzz.cli.provenance", return_value={}):
                args = build_parser().parse_args(
                    [
                        "replay",
                        str(bundle),
                        "--shell",
                        shell,
                        "--out",
                        str(root / "out"),
                    ]
                )
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
                with fake_shell(root) as shell, patch(
                    "siriusfuzz.cli.execute_probe",
                    return_value={"status": "ok", "record": {"verdict": "ok"}},
                ) as probe, patch("siriusfuzz.cli.provenance", return_value={}):
                    args = build_parser().parse_args(options + ["--shell", shell])
                    if required is not False and not override:
                        with self.assertRaisesRegex(
                            ValueError, "missing saved Sirius YAML"
                        ):
                            cmd_replay(args)
                        probe.assert_not_called()
                    else:
                        self.assertEqual(cmd_replay(args), 0)
                        payload, work = probe.call_args.args
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
            (run / "findings" / "000-mismatch-aa" / "more").mkdir()
            outcomes = iter(
                [
                    {"status": "ok", "record": {"verdict": "ok", "reason": ""}},
                    {
                        "status": "ok",
                        "record": {"verdict": "gpu_error", "reason": "still broken"},
                    },
                ]
            )
            with fake_shell(root) as shell:
                args = build_parser().parse_args(
                    ["recheck", str(run), "--cpu-only", "--shell", shell]
                    + ["--out", str(root / "out")]
                )
                with patch(
                    "siriusfuzz.cli.execute_probe",
                    side_effect=lambda *a, **k: next(outcomes),
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
                    self.assertEqual(payload["shell"], shell)
                report = json.loads(
                    next((root / "out").glob("run-*/recheck.json")).read_text()
                )
                self.assertEqual(report["cleared"], 1)
                self.assertEqual(
                    [(r["recorded"], r["now"]) for r in report["findings"]],
                    [("mismatch", "ok"), ("gpu_error", "gpu_error")],
                )
                with patch(
                    "siriusfuzz.cli.execute_probe",
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

    def test_invalid_limits(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                validate_limits(argparse.Namespace(timeout=value))


@unittest.skipUnless(SHELL, "no DuckDB shell available")
class ShellTests(unittest.TestCase):
    def test_statements_results_errors_restarts_and_crashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            session = Session(SHELL, None, None, root / "db", 0, cpu_only=True)
            session.open()
            try:
                ok = session.run(
                    "SELECT 1 AS a, NULL AS b, 'x' AS c, 1.5::DOUBLE AS d, 2.5::DECIMAL(10,2) AS e",
                    gpu=False,
                    timeout=60,
                )
                self.assertEqual(ok.status, "ok", ok.error)
                self.assertEqual([c.name for c in ok.result.columns], list("abcde"))
                self.assertEqual(ok.result.rows, [(1, None, "x", 1.5, "2.50")])
                empty = session.run("SELECT 1 AS a WHERE false", gpu=False, timeout=60)
                self.assertEqual((empty.status, empty.result.rows), ("ok", []))
                described = session.describe("SELECT 1 AS a WHERE false")
                self.assertEqual(
                    [(c.name, c.type.name) for c in described], [("a", "INTEGER")]
                )
                dup = session.run("SELECT 1 AS k, 2 AS k", gpu=False, timeout=60)
                self.assertEqual(dup.result.rows, [(1, 2)])
                err = session.run("SELECT nope", gpu=False, timeout=60)
                self.assertEqual(err.status, "error")
                self.assertIn("Binder Error", err.error)
                data = root / "d.duckdb"
                session.execute(f"ATTACH '{data}' AS d")
                session.attached["d"] = data
                session.use("d")
                session.execute(
                    "CREATE TABLE t(k INT); INSERT INTO t VALUES (1), (2); CHECKPOINT"
                )
                self.assertEqual(session.scalar("SELECT count(*) FROM t"), 2)
                # A statement that overruns its timeout costs the shell; the next
                # statement gets a fresh one with the datasets re-attached.
                slow = session.run(
                    "SELECT count(*) FROM range(100000000000)", gpu=False, timeout=0.5
                )
                self.assertEqual(slow.status, "timeout")
                self.assertFalse(session.shell.alive)
                again = session.run(
                    "SELECT count(*) AS n FROM t", gpu=False, timeout=60
                )
                self.assertEqual((again.status, again.result.rows), ("ok", [(2,)]))
                self.assertEqual(session.restarts, 1)
                # A process that dies mid-statement is a crash with its exit status.
                threading.Timer(
                    0.3, session.shell.proc.send_signal, args=(signal.SIGSEGV,)
                ).start()
                dead = session.run(
                    "SELECT count(*) FROM range(100000000000)", gpu=False, timeout=60
                )
                self.assertEqual(dead.status, "crash")
                self.assertEqual(dead.exitcode, -signal.SIGSEGV)
                self.assertEqual(session.scalar("SELECT 1"), 1)
                self.assertEqual(session.restarts, 2)
            finally:
                session.close()
            self.assertFalse(session.shell.alive)


if __name__ == "__main__":
    unittest.main()
