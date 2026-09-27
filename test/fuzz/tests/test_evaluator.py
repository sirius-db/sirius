"""Evaluator + Report with a fake session: findings, dedup, known issues, artifacts."""

import json
import pathlib
import random
import tempfile
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from . import conftest_path  # noqa: F401
from siriusfuzz import sqltypes as st
from siriusfuzz.artifacts import verify
from siriusfuzz.classify import Verdict, uses_plan_fallback
from siriusfuzz.cli import build_parser, cmd_run
from siriusfuzz.compare import ColumnInfo, ResultSet
from siriusfuzz.config import load_config
from siriusfuzz.report import KnownIssue, QueryRecord, Report, catch2_snippet, signature
from siriusfuzz.runner import Evaluator, Orchestrator, OrchestratorOptions
from siriusfuzz.session import RunResult, Session
from siriusfuzz.sqlast import Alias, ColumnRef, OrderItem, Select, SelectItem, TableRef


class FakeSession:
    """Scripted stand-in for Session: `gpu_behaviour` decides what the GPU run returns."""

    def __init__(self, gpu_behaviour):
        self.gpu_available = True
        self.gpu_behaviour = gpu_behaviour
        self.settings = {}
        self.current_alias = "main"
        self.con = self
        self.calls = []

    # Session API used by Evaluator
    def setting_supported(self, name):
        return name == "hash_partition_bytes"

    def set(self, name, value):
        self.settings[name] = value

    def restore(self, name):
        self.settings.pop(name, None)

    def execute(self, sql, *a):
        self.calls.append(sql)
        return self

    def begin_fallback_retry(self, reason):
        self.plan_fallback_reason = reason

    def end_fallback_retry(self):
        self.plan_fallback_reason = None

    def mark_auxiliary(self, operation, input_sql):
        self.auxiliary = {"operation": operation, "input_sql": input_sql}

    def fetchall(self):
        return []

    def load_dataset(self, ds, alias, permutation_seed=None):
        pass

    def use(self, alias):
        self.current_alias = alias

    def load_sqlsmith(self):
        return False

    def describe(self, sql):
        return [ColumnInfo("c0", st.INTEGER), ColumnInfo("c1", st.DOUBLE)]

    def run(self, sql, gpu, timeout):
        cols = [ColumnInfo("c0", st.INTEGER), ColumnInfo("c1", st.DOUBLE)]
        cpu_rows = [(1, 1.5), (2, 2.5)]
        if not gpu:
            return RunResult("ok", ResultSet(cols, list(cpu_rows)))
        return self.gpu_behaviour(sql, cols, cpu_rows, self)


def gpu_ok(sql, cols, rows, s):
    return RunResult("ok", ResultSet(cols, list(rows)))


def gpu_float_noise(sql, cols, rows, s):
    return RunResult("ok", ResultSet(cols, [(1, 1.5 + 1e-13), (2, 2.5)]))


def gpu_wrong(sql, cols, rows, s):
    return RunResult("ok", ResultSet(cols, [(1, 1.5), (3, 2.5)]))


def gpu_plan_fallback(sql, cols, rows, s):
    return RunResult(
        "error",
        error="Not implemented Error: GPU plan generation failed: Window not supported",
    )


def gpu_plan_then_fallback(sql, cols, rows, s):
    if s.calls and s.calls[-1] == "SET enable_duckdb_fallback = true":
        return gpu_ok(sql, cols, rows, s)
    return gpu_plan_fallback(sql, cols, rows, s)


def failing_fallback(result):
    def behaviour(sql, cols, rows, session):
        if session.calls and session.calls[-1] == "SET enable_duckdb_fallback = true":
            return result
        return gpu_plan_fallback(sql, cols, rows, session)

    return behaviour


def gpu_variant_wrong(sql, cols, rows, s):
    if s.settings.get("hash_partition_bytes") is not None:
        return RunResult("ok", ResultSet(cols, [(1, 1.5)] * 2))
    return RunResult("ok", ResultSet(cols, list(rows)))


def gpu_count_distinct_error(sql, cols, rows, s):
    return RunResult(
        "error",
        error="Sirius GPU execution failed: count(DISTINCT x) not supported ungrouped",
    )


def evaluate(behaviour, sql="SELECT 1 AS c0, 1.5 AS c1", **over):
    cfg = load_config(
        None, ["oracle.ambiguity_filter=false", *[f"{k}={v}" for k, v in over.items()]]
    )
    session = FakeSession(behaviour)
    ev = Evaluator(cfg, session, lambda m: None)
    ev.rng = random.Random(1)
    return ev.evaluate(None, sql, 0, "w0-d0", 0), ev


class EvaluatorTests(unittest.TestCase):
    def test_fallback_retry_marker_survives_before_and_during_native_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            active_path = pathlib.Path(tmp) / "worker.active.json"
            session = Session(
                None, None, pathlib.Path(tmp), 0, evidence_path=active_path
            )
            session.evidence["gpu"] = {
                "sql": "SELECT 1",
                "phase": "gpu",
                "stage": "evaluation",
                "status": "error",
            }
            session.begin_fallback_retry("Window not supported")
            self.assertEqual(
                json.loads(active_path.read_text())["plan_fallback_reason"],
                "Window not supported",
            )

            class Cursor:
                description = []

                def execute(self, sql):
                    self.active_during_query = json.loads(active_path.read_text())
                    return self

                def fetchall(self):
                    return []

            session.con = Cursor()
            with patch.object(session, "set_gpu"), patch.dict(
                "sys.modules", {"duckdb": SimpleNamespace(InterruptException=Exception)}
            ):
                session.run("SELECT 1", gpu=True, timeout=0)
            self.assertEqual(
                session.con.active_during_query["plan_fallback_reason"],
                "Window not supported",
            )
            session.end_fallback_retry()
            self.assertIsNone(session.plan_fallback_reason)

    def test_ok_and_float_noise(self):
        rec, _ = evaluate(gpu_ok)
        self.assertEqual(rec.verdict, Verdict.OK.value)
        rec, _ = evaluate(gpu_float_noise)
        self.assertEqual(rec.verdict, Verdict.OK.value)

    def test_mismatch(self):
        rec, _ = evaluate(gpu_wrong)
        self.assertEqual(rec.verdict, Verdict.MISMATCH.value)
        self.assertTrue(rec.diffs)

    def test_plan_fallback_strict(self):
        rec, _ = evaluate(gpu_plan_fallback)
        self.assertEqual(rec.verdict, Verdict.PLAN_FALLBACK.value)
        self.assertEqual(rec.reason, "Window not supported")

    def test_skip_policy_does_not_execute_fallback(self):
        rec, ev = evaluate(
            gpu_plan_then_fallback, **{"oracle.on_plan_fallback": "skip"}
        )
        self.assertEqual(rec.verdict, "plan_fallback")
        self.assertEqual(ev.s.calls, [])

    def test_fallback_failures_record_actual_verdict_and_reason(self):
        for status, error, verdict, reason in (
            (
                "error",
                "Invalid Input Error: fallback conversion failed",
                "gpu_error",
                "fallback conversion failed",
            ),
            (
                "error",
                "CUDA internal error",
                "gpu_internal_error",
                "CUDA internal error",
            ),
            ("error", "out of memory", "gpu_oom", "out of memory"),
            ("timeout", "interrupted", "timeout", "fallback run exceeded 60.0s"),
        ):
            with self.subTest(verdict=verdict):
                rec, ev = evaluate(
                    failing_fallback(RunResult(status, error=error)),
                    **{"oracle.on_plan_fallback": "count"},
                )
                self.assertEqual(rec.verdict, verdict)
                self.assertEqual(rec.reason, reason)
                self.assertIn(error, rec.detail)
                self.assertEqual(
                    rec.context["plan_fallback_reason"], "Window not supported"
                )
                self.assertEqual(ev.s.calls[-1], "SET enable_duckdb_fallback = false")
                if status == "timeout":
                    self.assertIsNone(ev._make_still_fails(rec))

    def test_fallback_error_reduction_preserves_execution_path_and_failure(self):
        failure = RunResult(
            "error", error="Invalid Input Error: fallback conversion failed"
        )
        rec, ev = evaluate(
            failing_fallback(failure), **{"oracle.on_plan_fallback": "count"}
        )
        check = ev._make_still_fails(rec)
        self.assertIsNotNone(check)
        self.assertTrue(check("SELECT 1"))
        for replacement in (
            RunResult("error", error="fallback storage failed"),
            RunResult("timeout", error="interrupted"),
            RunResult("ok", ResultSet([ColumnInfo("c0", st.INTEGER)], [(1,)])),
        ):
            ev.s.gpu_behaviour = failing_fallback(replacement)
            self.assertFalse(check("SELECT 1"))
            self.assertEqual(ev.s.calls[-1], "SET enable_duckdb_fallback = false")
        # An error before reaching fallback is not the same execution path.
        ev.s.gpu_behaviour = lambda *args: failure
        self.assertFalse(check("SELECT 1"))

    def test_fallback_mismatch_reduction_requires_strict_plan_rejection(self):
        result = RunResult(
            "ok",
            ResultSet(
                [ColumnInfo("c0", st.INTEGER), ColumnInfo("c1", st.DOUBLE)], [(1, 1.5)]
            ),
        )
        rec, ev = evaluate(
            failing_fallback(result), **{"oracle.on_plan_fallback": "count"}
        )
        self.assertEqual(rec.verdict, "fallback_mismatch")
        check = ev._make_still_fails(rec)
        self.assertTrue(check("SELECT 1"))
        ev.s.gpu_behaviour = gpu_wrong
        self.assertFalse(check("SELECT 1"))
        self.assertEqual(ev.s.calls[-1], "SET enable_duckdb_fallback = false")

    def test_reduction_fallback_checks_mark_the_active_retry(self):
        for result in (
            RunResult("error", error="Invalid Input Error: fallback conversion failed"),
            RunResult(
                "ok",
                ResultSet(
                    [ColumnInfo("c0", st.INTEGER), ColumnInfo("c1", st.DOUBLE)],
                    [(1, 1.5)],
                ),
            ),
        ):
            with self.subTest(status=result.status):
                rec, ev = evaluate(
                    failing_fallback(result), **{"oracle.on_plan_fallback": "count"}
                )
                check = ev._make_still_fails(rec)
                self.assertIsNotNone(check)
                with patch.object(
                    ev.s, "begin_fallback_retry", wraps=ev.s.begin_fallback_retry
                ) as begin:
                    self.assertTrue(check("SELECT 1"))
                begin.assert_called_once_with("Window not supported")
                self.assertIsNone(ev.s.plan_fallback_reason)

    def test_sqlsmith_call_marks_nonquery_reducer_work(self):
        _, ev = evaluate(gpu_wrong)
        with patch.object(ev.s, "mark_auxiliary", wraps=ev.s.mark_auxiliary) as mark:
            ev._sqlsmith_candidates("SELECT candidate")
        mark.assert_called_once_with("sqlsmith_reduction", "SELECT candidate")

    def test_auxiliary_marker_replaces_completed_fallback_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            active_path = pathlib.Path(tmp) / "worker.active.json"
            active_path.write_text(
                json.dumps(
                    {
                        "sql": "SELECT old_candidate",
                        "plan_fallback_reason": "Window not supported",
                        "status": "ok",
                    }
                )
            )
            session = Session(
                None, None, pathlib.Path(tmp), 0, evidence_path=active_path
            )
            session.stage = "reduction"
            session.mark_auxiliary("sqlsmith_reduction", "SELECT candidate")
            active = json.loads(active_path.read_text())
            self.assertEqual(active["operation"], "sqlsmith_reduction")
            self.assertEqual(active["input_sql"], "SELECT candidate")
            self.assertNotIn("sql", active)
            self.assertNotIn("plan_fallback_reason", active)

    def test_fallback_replay_preserves_comparison_mode(self):
        def reversed_fallback(sql, cols, rows, session):
            result = gpu_plan_then_fallback(sql, cols, rows, session)
            if result.result:
                result.result.rows.reverse()
            return result

        ordered_query = Select(
            [
                SelectItem(ColumnRef("t", name, typ), name)
                for name, typ in (("c0", st.INTEGER), ("c1", st.DOUBLE))
            ],
            TableRef("t", "t"),
            order_by=[OrderItem(Alias("c0")), OrderItem(Alias("c1"))],
        )
        for mode, verdict in (
            ("ordered", "fallback_mismatch"),
            ("multiset", "plan_fallback"),
            (None, "fallback_mismatch"),
        ):
            with self.subTest(mode=mode):
                cfg = load_config(None, ["oracle.on_plan_fallback=count"])
                session = FakeSession(reversed_fallback)
                evaluator = Evaluator(cfg, session, lambda message: None)
                evaluator.replay_mode = mode
                rec = evaluator.evaluate(
                    ordered_query if mode is None else None,
                    ordered_query.sql(),
                    0,
                    "synthetic",
                    0,
                )
                self.assertEqual(rec.verdict, verdict)
                self.assertEqual(rec.comparison, mode or "ordered")
                self.assertEqual(
                    session.calls[-1], "SET enable_duckdb_fallback = false"
                )

    def test_variant_mismatch(self):
        rec, _ = evaluate(gpu_variant_wrong, **{"variants.per_query": "1"})
        self.assertEqual(rec.verdict, Verdict.VARIANT_MISMATCH.value)
        self.assertEqual(list(rec.variant), ["hash_partition_bytes"])

    def test_variant_reduction_requires_cpu_matching_baseline(self):
        for base_value, variant_value, accepted in (
            (1.0, 2.0, False),
            (0.0, 2.0, True),
            (0.0, 0.0, False),
            (5e-13, 2.0, True),
        ):
            with self.subTest(baseline=base_value, variant=variant_value):
                session = FakeSession(gpu_ok)
                evaluator = Evaluator(load_config(None), session, lambda message: None)
                record = QueryRecord(
                    worker=0,
                    dataset="synthetic",
                    seed=0,
                    sql="SELECT 0",
                    verdict=Verdict.VARIANT_MISMATCH.value,
                    variant={"hash_partition_bytes": 123},
                )
                check = evaluator._make_still_fails(record)
                self.assertIsNotNone(check)
                results = [
                    RunResult("ok", ResultSet([ColumnInfo("c0", typ)], [(value,)]))
                    for value, typ in (
                        (0.0, st.DOUBLE),
                        (base_value, st.INTEGER),
                        (variant_value, st.INTEGER),
                    )
                ]
                with patch.object(session, "run", side_effect=results) as run:
                    self.assertEqual(check("SELECT 0"), accepted)
                self.assertEqual(run.call_count, 2 if base_value == 1.0 else 3)
                self.assertEqual(session.settings, {})


class ReportTests(unittest.TestCase):
    def test_finding_dedup_preserves_distinct_inputs_and_evidence(self):
        for verdict in ("mismatch", "fallback_mismatch", "variant_mismatch", "timeout"):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as tmp:
                report = Report(pathlib.Path(tmp), load_config(None), [], 1)
                original = QueryRecord(
                    0,
                    "d0",
                    1,
                    "SELECT k FROM t",
                    verdict,
                    labels=["Select"],
                    variant={"hash_partition_bytes": 123},
                )
                records = [original] + [
                    replace(original, **change)
                    for change in (
                        {"sql": "SELECT k + 1 FROM t"},
                        {"dataset": "d1"},
                        {"comparison": "ordered"},
                        {"variant": {"hash_partition_bytes": 456}},
                        {"detail": "row count 1 vs 2"},
                        {"reason": "GPU run exceeded 120s"},
                        {"context": {"phase": "gpu", "stage": "reduction"}},
                        {"diffs": ["row 0 col 0: 1 vs 2"]},
                        {
                            "evidence": {
                                "operations": [
                                    {
                                        "phase": "gpu",
                                        "fingerprint_multiset": "different",
                                    }
                                ]
                            }
                        },
                    )
                ]
                for record in records:
                    name = report.add(record)
                    self.assertIsNotNone(name)
                    verify(report.run_dir / "findings" / name)
                # Reduction is additional evidence, not a new observation or key.
                reduced = replace(
                    original, reduced_sql="SELECT k", reduced_labels=["ColumnRef"]
                )
                self.assertEqual(signature(original), signature(reduced))
                self.assertIsNone(report.add(reduced))
                self.assertEqual(len(report.finish()["findings"]), len(records))
                if verdict == "timeout":
                    fallback = replace(
                        original,
                        context={"plan_fallback_reason": "Window not supported"},
                    )
                    self.assertNotEqual(signature(original), signature(fallback))

    def test_catch2_reproducer_applies_and_restores_variant(self):
        for variant in (
            None,
            {"hash_partition_bytes": 123},
            {"expression_evaluator_strategy": "ast_jit"},
            {"enable_operator": False},
        ):
            for mode in ("ordered", "multiset"):
                with self.subTest(variant=variant, mode=mode):
                    rec = QueryRecord(
                        0,
                        "d0",
                        1,
                        "SELECT k FROM t",
                        "variant_mismatch" if variant else "mismatch",
                        variant=variant,
                        comparison=mode,
                    )
                    snippet = catch2_snippet("test", rec, None, rec.sql)
                    comparator = (
                        "compare_gpu_vs_cpu_ordered"
                        if mode == "ordered"
                        else "compare_gpu_vs_cpu"
                    )
                    self.assertIn(f"{comparator}(query)", snippet)
                    if variant:
                        setting, value = next(iter(variant.items()))
                        literal = (
                            f"'{value}'"
                            if isinstance(value, str)
                            else str(value).lower()
                        )
                        stages = [
                            snippet.index(part)
                            for part in (
                                "auto baseline =",
                                f"SET {setting} = {literal}",
                                "auto variant =",
                                "ToSQLString()",
                                "REQUIRE_FALSE(variant->HasError())",
                            )
                        ]
                        self.assertEqual(stages, sorted(stages))
                        self.assertIn("REQUIRE(baseline_rows == variant_rows)", snippet)
                        self.assertIn(
                            f"Result>(), {'false' if mode == 'ordered' else 'true'})",
                            snippet,
                        )

    def test_catch2_reproducer_compares_the_recorded_fallback_path(self):
        for verdict, context in (
            ("fallback_mismatch", {}),
            ("gpu_error", {"plan_fallback_reason": "Window not supported"}),
        ):
            for mode in ("ordered", "multiset"):
                with self.subTest(verdict=verdict, mode=mode):
                    rec = QueryRecord(
                        0,
                        "d0",
                        1,
                        "SELECT k FROM t",
                        verdict,
                        comparison=mode,
                        context=context,
                    )
                    snippet = catch2_snippet("test", rec, None, rec.sql)
                    self.assertIn("#include <utils/scoped_sirius_setting.hpp>", snippet)
                    self.assertIn(
                        'scoped_sirius_setting fallback(*con, "enable_duckdb_fallback", true)',
                        snippet,
                    )
                    self.assertIn(
                        f"expect_plan_fallback_matches_cpu(query, {'true' if mode == 'ordered' else 'false'})",
                        snippet,
                    )
                    self.assertNotIn("compare_gpu_vs_cpu", snippet)

    def test_supervisor_does_not_turn_reducer_work_into_query_finding(self):
        for failure in ("crash", "hang"):
            for active_state in (
                {
                    "stage": "reduction",
                    "operation": "sqlsmith_reduction",
                    "input_sql": "SELECT candidate",
                    "status": "running",
                },
                {
                    "stage": "reduction",
                    "phase": "gpu",
                    "sql": "SELECT old_candidate",
                    "plan_fallback_reason": "Window not supported",
                    "status": "ok",
                },
            ):
                with self.subTest(
                    failure=failure, state=active_state
                ), tempfile.TemporaryDirectory() as tmp:
                    root = pathlib.Path(tmp)
                    cfg = load_config(None)
                    report = Report(root, cfg, [], 1)
                    active_path = root / "worker.active.json"
                    active_path.write_text(json.dumps(active_state))
                    runner = Orchestrator(cfg, report, 1, OrchestratorOptions())
                    runner.active_paths[0] = active_path
                    runner.inflight[0] = (
                        "SELECT original",
                        "synthetic",
                        time.time(),
                        [],
                    )
                    runner._say = lambda message: None
                    process = SimpleNamespace(
                        exitcode=-11, kill=lambda: None, join=lambda timeout: None
                    )
                    with patch.object(runner, "_maybe_respawn") as respawn:
                        if failure == "crash":
                            runner._worker_died(0, process, set())
                        else:
                            runner._worker_hung(0, process, runner.inflight[0], set())
                    respawn.assert_not_called()
                    self.assertEqual(report.status, "incomplete")
                    self.assertEqual(report.queries, 0)
                    self.assertEqual(list((root / "findings").iterdir()), [])
                    report.finish()

    def test_supervisor_crash_and_hang_keep_fallback_path_in_finding(self):
        for verdict in ("crash", "timeout"):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                cfg = load_config(None, ["oracle.on_plan_fallback=count"])
                report = Report(root, cfg, [], 1)
                active_path = root / "worker.active.json"
                active_path.write_text(
                    json.dumps(
                        {
                            "sql": "SELECT 1",
                            "phase": "gpu",
                            "status": "running",
                            "plan_fallback_reason": "Window not supported",
                        }
                    )
                )
                runner = Orchestrator(cfg, report, 1, OrchestratorOptions())
                runner.active_paths[0] = active_path
                runner.inflight[0] = ("SELECT 1", "synthetic", time.time(), [])
                runner._say = lambda text: None
                process = SimpleNamespace(
                    exitcode=-11, kill=lambda: None, join=lambda timeout: None
                )
                with patch.object(runner, "_maybe_respawn"):
                    if verdict == "crash":
                        runner._worker_died(0, process, set())
                    else:
                        runner._worker_hung(0, process, runner.inflight[0], set())
                record = json.loads(report.log_path.read_text().splitlines()[0])
                self.assertEqual(record["verdict"], verdict)
                self.assertTrue(uses_plan_fallback(verdict, record["context"]))
                finding = next((root / "findings").iterdir())
                repro = (finding / "repro.sql").read_text()
                self.assertIn("-- Recorded fallback path", repro)
                report.finish()

    def test_standalone_repro_reaches_the_recorded_fallback_failure(self):
        for behaviour in (
            failing_fallback(RunResult("error", error="fallback conversion failed")),
            failing_fallback(RunResult("timeout", error="interrupted")),
            failing_fallback(
                RunResult(
                    "ok",
                    ResultSet(
                        [ColumnInfo("c0", st.INTEGER), ColumnInfo("c1", st.DOUBLE)],
                        [(1, 1.5)],
                    ),
                )
            ),
            gpu_wrong,
        ):
            with self.subTest(
                behaviour=behaviour
            ), tempfile.TemporaryDirectory() as tmp:
                cfg = load_config(None, ["oracle.on_plan_fallback=count"])
                rec, _ = evaluate(behaviour, **{"oracle.on_plan_fallback": "count"})
                report = Report(pathlib.Path(tmp), cfg, [], 0)
                name = report.add(rec)
                report.finish()
                script = (
                    pathlib.Path(tmp) / "findings" / name / "repro.sql"
                ).read_text()
                # Interpret the generated setting/query sequence against the scripted
                # session, including a shell configured to stop on the first error.
                session = FakeSession(behaviour)
                gpu, bail, results = False, True, []
                for line in script.splitlines():
                    if line == ".bail off":
                        bail = False
                    elif line.startswith("SET gpu_execution = "):
                        gpu = line.endswith("true;")
                    elif line.startswith("SET enable_duckdb_fallback = "):
                        session.execute(line.rstrip(";"))
                    elif line == rec.sql + ";":
                        results.append(session.run(rec.sql, gpu, 60))
                        if results[-1].status != "ok" and bail:
                            break
                expected_runs = 2 if rec.verdict == "mismatch" else 3
                self.assertEqual(len(results), expected_runs, script)
                if expected_runs == 3:
                    self.assertIn("GPU plan generation failed", results[1].error)
                    self.assertNotIn("GPU plan generation failed", results[2].error)
                self.assertEqual(
                    session.calls[-1], "SET enable_duckdb_fallback = false"
                )

    def test_distinct_fallback_errors_do_not_share_the_plan_rejection_signature(self):
        cfg = load_config(None, ["oracle.on_plan_fallback=count"])
        with tempfile.TemporaryDirectory() as tmp:
            report = Report(pathlib.Path(tmp), cfg, [], 0)
            for reason in ("fallback conversion failed", "fallback storage failed"):
                rec, _ = evaluate(
                    failing_fallback(RunResult("error", error=reason)),
                    **{"oracle.on_plan_fallback": "count"},
                )
                report.add(rec)
            self.assertEqual(len(report.finish()["findings"]), 2)

    def test_plan_fallback_policy_controls_campaign_exit(self):
        def run_campaign(runner):
            session = FakeSession(gpu_plan_then_fallback)
            ev = Evaluator(runner.cfg, session, lambda message: None)
            runner.report.add(ev.evaluate(None, "SELECT 1", 0, "synthetic", 0))
            summary = runner.report.finish()
            self.assertEqual(summary["counts"], {"plan_fallback": 1})
            self.assertEqual(
                summary["plan_fallback_reasons"], [("Window not supported", 1)]
            )
            self.assertEqual(
                len(summary["findings"]),
                int(runner.cfg.oracle.on_plan_fallback == "fail"),
            )
            return summary

        for policy, exit_code in (("fail", 1), ("count", 0), ("skip", 0)):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as tmp:
                args = build_parser().parse_args(
                    [
                        "run",
                        "--seed",
                        "1",
                        "--queries",
                        "1",
                        "--fail-on-findings",
                        "--out",
                        tmp,
                        "--set",
                        f"oracle.on_plan_fallback={policy}",
                    ]
                )
                with patch(
                    "siriusfuzz.cli._engine", return_value=("synthetic-extension", [])
                ), patch("siriusfuzz.cli.provenance", return_value={}), patch.object(
                    Orchestrator, "run", autospec=True, side_effect=run_campaign
                ):
                    self.assertEqual(cmd_run(args), exit_code)

    def test_count_policy_still_reports_fallback_mismatches_and_errors(self):
        def wrong_fallback(sql, cols, rows, session):
            if (
                session.calls
                and session.calls[-1] == "SET enable_duckdb_fallback = true"
            ):
                return gpu_wrong(sql, cols, rows, session)
            return gpu_plan_fallback(sql, cols, rows, session)

        for behaviour, verdict in (
            (wrong_fallback, "fallback_mismatch"),
            (gpu_plan_fallback, "gpu_error"),
        ):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as tmp:
                cfg = load_config(None, ["oracle.on_plan_fallback=count"])
                record, _ = evaluate(behaviour, **{"oracle.on_plan_fallback": "count"})
                report = Report(pathlib.Path(tmp), cfg, [], 0)
                name = report.add(record)
                summary = report.finish()
                self.assertIsNotNone(name)
                self.assertEqual(summary["findings"][0]["verdict"], verdict)

    def test_dedup_known_issue_and_artifacts(self):
        cfg = load_config(None)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp)
            (run_dir / "datasets").mkdir()
            (run_dir / "datasets" / "w0-d0.sql").write_text(
                "CREATE TABLE t(k INTEGER);\n"
            )
            known = [
                KnownIssue(
                    ".",
                    "sirius-db/sirius#1218",
                    verdicts=["gpu_error"],
                    sql_pattern="count\\(DISTINCT",
                )
            ]
            report = Report(run_dir, cfg, known, seed=1)
            rec1, _ = evaluate(gpu_count_distinct_error)
            rec2 = replace(rec1, dataset="w0-d1")
            report.dataset_path(rec2.dataset).write_text("CREATE TABLE t(k BIGINT);\n")
            name1 = report.add(rec1)
            name2 = report.add(rec2)
            self.assertIsNotNone(name1)
            self.assertIsNone(name2, "same signature must dedup")
            self.assertEqual(signature(rec1), signature(rec2))
            d = run_dir / "findings" / name1
            for f in (
                "query.sql",
                "dataset.sql",
                "config.toml",
                "meta.json",
                "detail.txt",
                "repro_catch2.cpp",
            ):
                self.assertTrue((d / f).exists(), f)
            bundles = sorted((run_dir / "findings").rglob("dataset.sql"))
            self.assertEqual(len(bundles), 2)
            self.assertNotEqual(bundles[0].read_text(), bundles[1].read_text())
            for data in bundles:
                verify(data.parent)
            rec3, _ = evaluate(
                gpu_count_distinct_error, sql="SELECT count(DISTINCT k) AS c0 FROM t"
            )
            report.add(rec3)
            self.assertEqual(rec3.verdict, Verdict.KNOWN_ISSUE.value)
            self.assertIn("#1218", rec3.reason)
            summary = report.finish()
            self.assertEqual(summary["counts"]["gpu_error"], 2)
            self.assertEqual(summary["counts"]["known_issue"], 1)
            self.assertTrue((run_dir / "summary.txt").exists())
            self.assertIn("gpu_error", report.render_summary(summary))


if __name__ == "__main__":
    unittest.main()
