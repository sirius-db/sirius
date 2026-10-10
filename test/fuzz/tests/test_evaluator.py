"""Evaluator + Report with a fake session: findings, dedup, known issues, artifacts."""

import json
import pathlib
import random
import tempfile
import time
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import patch

from . import conftest_path  # noqa: F401
from siriusfuzz import sqltypes as st
from siriusfuzz.classify import Verdict
from siriusfuzz.cli import Engine, build_parser, cmd_run
from siriusfuzz.compare import ColumnInfo, ResultSet
from siriusfuzz.config import FUZZ_DIR, load_config
from siriusfuzz.report import (
    KnownIssue,
    QueryRecord,
    Report,
    load_known_issues,
    signature,
)
from siriusfuzz.runner import (
    Evaluator,
    Orchestrator,
    OrchestratorOptions,
    wants_reduction,
)
from siriusfuzz.session import RunResult, Session
from siriusfuzz.sqlast import Alias, ColumnRef, OrderItem, Select, SelectItem, TableRef


class FakeSession:
    """Scripted stand-in for Session: `gpu_behaviour` decides what the GPU run returns."""

    extension = None

    def __init__(self, gpu_behaviour):
        self.gpu_available = True
        self.gpu_behaviour = gpu_behaviour
        self.settings = {}
        self.current_alias = "main"

    # Session API used by Evaluator
    def setting_supported(self, name):
        return name == "hash_partition_bytes"

    def set(self, name, value):
        self.settings[name] = value

    def restore(self, name):
        self.settings.pop(name, None)

    def mark_auxiliary(self, operation, input_sql):
        self.auxiliary = {"operation": operation, "input_sql": input_sql}

    def load_dataset(self, ds, alias, permutation_seed=None):
        pass

    def use(self, alias):
        self.current_alias = alias

    def load_sqlsmith(self):
        return False

    def sqlsmith_candidates(self, sql):
        return []

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


def gpu_variant_wrong(sql, cols, rows, s):
    if s.settings.get("hash_partition_bytes") is not None:
        return RunResult("ok", ResultSet(cols, [(1, 1.5)] * 2))
    return RunResult("ok", ResultSet(cols, list(rows)))


def gpu_count_distinct_error(sql, cols, rows, s):
    return RunResult(
        "error",
        error="Invalid Error: Sirius GPU execution failed: Distinct aggregates not supported in GPU path yet",
    )


UNSUPPORTED_REASON = "Unsupported expression in projection (falling back to CPU): "
UNSUPPORTED_PROJECTION = (
    "Not implemented Error: GPU plan generation failed: " + UNSUPPORTED_REASON
)


def gpu_rejects(expression):
    return lambda *args: RunResult("error", error=UNSUPPORTED_PROJECTION + expression)


def gpu_reversed(sql, cols, rows, s):
    return RunResult("ok", ResultSet(cols, list(reversed(rows))))


ORDERED_QUERY = Select(
    [
        SelectItem(ColumnRef("t", name, typ), name)
        for name, typ in (("c0", st.INTEGER), ("c1", st.DOUBLE))
    ],
    TableRef("t", "t"),
    order_by=[OrderItem(Alias("c0")), OrderItem(Alias("c1"))],
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

    def test_sqlsmith_call_marks_nonquery_reducer_work(self):
        _, ev = evaluate(gpu_wrong)
        with patch.object(ev.s, "mark_auxiliary", wraps=ev.s.mark_auxiliary) as mark:
            ev._sqlsmith_candidates("SELECT candidate")
        mark.assert_called_once_with("sqlsmith_reduction", "SELECT candidate")

    def test_auxiliary_marker_replaces_completed_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            active_path = pathlib.Path(tmp) / "worker.active.json"
            active_path.write_text(
                json.dumps({"sql": "SELECT old_candidate", "status": "ok"})
            )
            session = Session(
                "duckdb",
                None,
                None,
                pathlib.Path(tmp),
                0,
                cpu_only=True,
                evidence_path=active_path,
            )
            session.stage = "reduction"
            session.mark_auxiliary("sqlsmith_reduction", "SELECT candidate")
            active = json.loads(active_path.read_text())
            self.assertEqual(active["operation"], "sqlsmith_reduction")
            self.assertEqual(active["input_sql"], "SELECT candidate")
            self.assertNotIn("sql", active)

    def test_gap_reduction_narrows_to_the_unsupported_function(self):
        rec, ev = evaluate(
            gpu_rejects('regexp_matches(concat("a"."c1", "a"."c2"), \'x\')')
        )
        self.assertEqual(rec.verdict, Verdict.PLAN_FALLBACK.value)
        check = ev._make_still_fails(rec)
        # Dropping the supported concat keeps the rejection: accepted, and the
        # record remembers what the smaller query was rejected for.
        ev.s.gpu_behaviour = gpu_rejects('regexp_matches("a"."c1", \'x\')')
        self.assertTrue(check("SELECT 1"))
        self.assertTrue(rec.reduced_reason.endswith('regexp_matches("a"."c1", \'x\')'))
        # A different unsupported function, a different rejection, or success: rejected.
        for behaviour in (
            gpu_rejects('upper("a"."c1")'),
            lambda *a: RunResult(
                "error",
                error="Not implemented Error: GPU plan generation failed: Window not supported",
            ),
            gpu_ok,
        ):
            ev.s.gpu_behaviour = behaviour
            self.assertFalse(check("SELECT 1"))
        # Other GPU errors still need the exact normalized reason.
        rec, ev = evaluate(
            lambda *a: RunResult(
                "error", error="Sirius GPU execution failed: f(x) broke"
            )
        )
        check = ev._make_still_fails(rec)
        ev.s.gpu_behaviour = lambda *a: RunResult(
            "error", error="Sirius GPU execution failed: f(y) broke"
        )
        self.assertTrue(check("SELECT 1"))
        ev.s.gpu_behaviour = lambda *a: RunResult(
            "error", error="Sirius GPU execution failed: broke"
        )
        self.assertFalse(check("SELECT 1"))

    def test_reduction_runs_once_per_error_signature(self):
        seen = set()
        error = QueryRecord(
            0,
            "d0",
            1,
            "SELECT f(x)",
            "gpu_error",
            reason="Unsupported expression: f(x)",
        )
        same_reason = replace(
            error,
            sql="SELECT f(y)",
            dataset="d1",
            reason="Unsupported expression: f(y)",
        )
        self.assertTrue(wants_reduction(error, "correctness", seen))
        self.assertFalse(wants_reduction(same_reason, "correctness", seen))
        mismatch = QueryRecord(0, "d0", 1, "SELECT 1", "mismatch", reason="1 vs 2")
        self.assertTrue(wants_reduction(mismatch, "correctness", seen))
        self.assertTrue(
            wants_reduction(replace(mismatch, sql="SELECT 2"), "correctness", seen)
        )
        self.assertFalse(
            wants_reduction(replace(mismatch, verdict="ok"), "gaps", set())
        )

    def test_gaps_mode_reduces_runtime_but_not_plan_fallbacks(self):
        plan = QueryRecord(0, "d0", 1, "SELECT 1", "plan_fallback", reason="Window")
        runtime = replace(plan, verdict="runtime_fallback", reason="x not supported")
        self.assertFalse(wants_reduction(plan, "gaps", set()))
        self.assertTrue(wants_reduction(plan, "correctness", set()))
        seen = set()
        self.assertTrue(wants_reduction(runtime, "gaps", seen))
        # Runtime fallbacks dedup by reason, like plan rejections and errors.
        again = replace(runtime, sql="SELECT 2", labels=["Select(group_by)"])
        self.assertFalse(wants_reduction(again, "gaps", seen))
        self.assertEqual(signature(runtime), signature(again))

    def test_a_dead_shell_is_a_crash_finding_with_its_backtrace_reason(self):
        tail = (
            "*** SIGSEGV received\n#0 /x/build/release/duckdb(+0x1)\n"
            "#1 /x/build/release/duckdb(+0x2)\n*** end backtrace ***\n"
        )
        rec, _ = evaluate(lambda *a: RunResult("crash", error=tail, exitcode=-11))
        self.assertEqual(rec.verdict, Verdict.CRASH.value)
        self.assertTrue(rec.reason.startswith("SIGSEGV"), rec.reason)
        self.assertIn("end backtrace", rec.detail)
        # A death during the CPU reference run is marked as such.
        session = FakeSession(gpu_ok)
        session.run = lambda sql, gpu, timeout: RunResult(
            "crash", error="*** SIGABRT\n", exitcode=-6
        )
        rec = Evaluator(load_config(None), session, lambda m: None).evaluate(
            None, "SELECT 1", 0, "d", 0
        )
        self.assertEqual(rec.verdict, Verdict.CRASH.value)
        self.assertTrue(rec.reason.startswith("CPU phase: SIGABRT"), rec.reason)
        # Under a setting variant the variant is recorded with the crash.
        rec, _ = evaluate(
            lambda sql, cols, rows, s: (
                RunResult("crash", error="*** SIGSEGV\n", exitcode=-11)
                if s.settings
                else RunResult("ok", ResultSet(cols, list(rows)))
            ),
            **{"variants.per_query": "1"},
        )
        self.assertEqual(rec.verdict, Verdict.CRASH.value)
        self.assertEqual(list(rec.variant), ["hash_partition_bytes"])

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
    def test_finding_dedup_keys_mismatches_on_their_inputs(self):
        for verdict in ("mismatch", "variant_mismatch", "timeout"):
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
                distinct = [
                    replace(original, **change)
                    for change in (
                        {"sql": "SELECT k + 1 FROM t"},
                        {"dataset": "d1"},
                        {"comparison": "ordered"},
                        {"variant": {"hash_partition_bytes": 456}},
                    )
                ]
                for record in [original, *distinct]:
                    self.assertIsNotNone(report.add(record))
                # The same input seen again, with other detail, is the same finding.
                for change in (
                    {"detail": "row count 1 vs 2"},
                    {"reason": "GPU run exceeded 120s"},
                    {"diffs": ["row 0 col 0: 1 vs 2"]},
                    {"reduced_sql": "SELECT k", "reduced_labels": ["ColumnRef"]},
                ):
                    self.assertIsNone(report.add(replace(original, **change)))
                self.assertEqual(len(report.finish()["findings"]), 1 + len(distinct))

    def test_reduction_compares_each_candidate_in_its_own_order_mode(self):
        cfg = load_config(None, ["oracle.ambiguity_filter=false"])
        ev = Evaluator(cfg, FakeSession(gpu_reversed), lambda m: None)
        check = ev._make_still_fails(
            QueryRecord(0, "d", 0, ORDERED_QUERY.sql(), "mismatch")
        )
        # Same rows, different order: a finding only for a totally ordered candidate.
        self.assertTrue(check(ORDERED_QUERY.sql(), ORDERED_QUERY))
        unordered = replace(ORDERED_QUERY, order_by=[])
        self.assertFalse(check(unordered.sql(), unordered))
        self.assertFalse(check(ORDERED_QUERY.sql(), None))

    def test_reduction_rejects_candidates_that_depend_on_input_order(self):
        class PermutationSensitive(FakeSession):
            def run(self, sql, gpu, timeout):
                res = super().run(sql, gpu, timeout)
                if not gpu and self.current_alias.endswith("p"):
                    res.result.rows = res.result.rows[:1]
                return res

        for ambiguity_filter, accepted in (("true", False), ("false", True)):
            with self.subTest(ambiguity_filter=ambiguity_filter):
                cfg = load_config(None, [f"oracle.ambiguity_filter={ambiguity_filter}"])
                session = PermutationSensitive(gpu_wrong)
                ev = Evaluator(cfg, session, lambda m: None)
                ev.set_dataset(SimpleNamespace(seed=1), "main")
                check = ev._make_still_fails(
                    QueryRecord(0, "d", 0, "SELECT 1", "mismatch")
                )
                self.assertEqual(check("SELECT 0 LIMIT 1", None), accepted)
                self.assertEqual(session.current_alias, "main")

    def test_crash_before_first_run_is_attributed_to_the_new_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            active = root / "w0-s1.active.json"
            # The previous query finished: its last run is still in the active file.
            session = Session(
                "duckdb", None, None, root, 0, cpu_only=True, evidence_path=active
            )
            session.stage = "reduction"
            active.write_text(
                json.dumps({"sql": "SELECT previous", "phase": "cpu", "status": "ok"})
            )
            session.begin_query("SELECT current")
            self.assertEqual(session.evidence, {})
            self.assertEqual(session.stage, "evaluation")

            # The worker dies before its first Session.run() rewrites the active file.
            cfg = load_config(None)
            report = Report(root / "run", cfg, [], 1)
            runner = Orchestrator(cfg, report, 1, OrchestratorOptions(quiet=True))
            runner.active_paths[0] = active
            runner.inflight[0] = ("SELECT current", "w0-d0", time.time(), [])
            process = SimpleNamespace(exitcode=-11)
            with patch.object(runner, "_maybe_respawn"):
                runner._worker_died(0, process, set())
            record = json.loads(report.log_path.read_text().splitlines()[0])
            self.assertEqual(record["verdict"], "crash")
            self.assertEqual(record["sql"], "SELECT current")
            self.assertFalse(record["reason"].startswith("CPU phase"))
            report.finish()

    def test_supervisor_reads_last_messages_of_an_exited_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(None)
            report = Report(pathlib.Path(tmp), cfg, [], 1)
            runner = Orchestrator(cfg, report, 1, OrchestratorOptions(quiet=True))
            rec, _ = evaluate(gpu_ok)

            def post(msg):
                runner.queue.put({**msg, "worker": 0, "spawn": 1})

            class ExitedAfterReporting:
                exitcode = 0
                reported = False

                def is_alive(self):
                    # The worker's last messages land after the supervisor's drain,
                    # just before it observes the exit.
                    if not self.reported:
                        self.reported = True
                        post({"type": "result", "record": asdict(rec)})
                        post({"type": "idle"})
                        post({"type": "done", "reason": "finished"})
                    return False

                def join(self, timeout=None):
                    pass

            def spawn(w):
                runner.worker_spawns[w] = 1
                runner.procs[w] = ExitedAfterReporting()
                runner.last_seen[w] = time.time()
                post({"type": "begin", "sql": rec.sql, "dataset": "w0-d0"})

            with patch.object(runner, "_spawn", side_effect=spawn):
                runner._run()
            self.assertEqual(runner.total_results, 1)
            self.assertEqual(report.status, "complete")
            self.assertEqual(dict(report.counts), {"ok": 1})
            report.finish()

    def test_shipped_count_distinct_known_issue_is_specific(self):
        known = load_known_issues(FUZZ_DIR / "known_issues.toml")
        sql = "SELECT count(DISTINCT k) AS c0 FROM t"
        rec, _ = evaluate(gpu_count_distinct_error, sql=sql)
        self.assertTrue(any(k.matches(rec) for k in known))
        self.assertEqual(rec.verdict, Verdict.RUNTIME_FALLBACK.value)
        other, _ = evaluate(
            lambda *a: RunResult(
                "error", error="Sirius GPU execution failed: unrelated join failure"
            ),
            sql=sql,
        )
        self.assertEqual(other.verdict, Verdict.GPU_ERROR.value)
        self.assertFalse(any(k.matches(other) for k in known))

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

    def test_finding_md_leads_with_the_query_and_the_replay_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec, _ = evaluate(gpu_wrong, sql="SELECT k FROM t")
            report = Report(pathlib.Path(tmp), load_config(None), [], 0)
            report.dataset_path(rec.dataset).write_text("CREATE TABLE t(k INT);\n")
            name = report.add(rec)
            bundle = pathlib.Path(tmp) / "findings" / name
            text = (bundle / "FINDING.md").read_text()
            self.assertIn("- verdict: `mismatch`", text)
            self.assertIn("## Query\n\n```sql\nSELECT k FROM t;\n```", text)
            self.assertIn(f"fuzz replay {bundle.resolve()}", text)
            self.assertNotIn("Reduced query", text)
            # Reduction rewrites the page around the smaller query.
            rec.reduced_sql = "SELECT k"
            rec.reduction_steps = 3
            report.add_reduction(rec)
            text = (bundle / "FINDING.md").read_text()
            self.assertIn("## Query (reduced in 3 steps; original in query.sql)", text)
            self.assertIn("```sql\nSELECT k;\n```", text)
            self.assertEqual((bundle / "reduced.sql").read_text(), "SELECT k;\n")
            # Only crashes and hangs carry the worker log; the rest is in meta.json.
            names = sorted(p.name for p in bundle.iterdir())
            self.assertEqual(
                names,
                [
                    "FINDING.md",
                    "config.toml",
                    "dataset.sql",
                    "meta.json",
                    "query.sql",
                    "reduced.sql",
                    "reduction.json",
                ],
            )
            meta = json.loads((bundle / "meta.json").read_text())
            self.assertEqual(meta["bundle_version"], 2)
            self.assertIn("environment", meta)
            self.assertIn("runtime", meta)

    def test_distinct_runtime_errors_do_not_share_a_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Report(pathlib.Path(tmp), load_config(None), [], 0)
            for reason in ("conversion failed", "storage failed"):
                rec, _ = evaluate(
                    lambda *a, reason=reason: RunResult(
                        "error", error=f"Sirius GPU execution failed: {reason}"
                    )
                )
                report.add(rec)
            self.assertEqual(len(report.finish()["findings"]), 2)

    def test_plan_fallback_fails_a_correctness_campaign_but_not_a_gaps_campaign(self):
        def run_campaign(runner):
            session = FakeSession(gpu_plan_fallback)
            ev = Evaluator(runner.cfg, session, lambda message: None)
            runner.report.add(ev.evaluate(None, "SELECT 1", 0, "synthetic", 0))
            summary = runner.report.finish()
            self.assertEqual(summary["counts"], {"plan_fallback": 1})
            self.assertEqual(len(summary["findings"]), 1)
            self.assertEqual(summary["gaps"]["runtime"], [])
            self.assertEqual(
                summary["gaps"]["plan"][0]["reason"], "Window not supported"
            )
            self.assertEqual(summary["gaps"]["plan"][0]["sql"], "SELECT 1")
            self.assertEqual(runner.opts.mode, summary["mode"])
            return summary

        for mode, exit_code in (("correctness", 1), ("gaps", 0)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                args = build_parser().parse_args(
                    ["run", "--mode", mode, "--seed", "1", "--queries", "1"]
                    + ["--fail-on-findings", "--no-doctor", "--out", tmp]
                )
                with patch(
                    "siriusfuzz.cli._engine",
                    return_value=Engine(
                        "synthetic-shell", "synthetic-extension", False, []
                    ),
                ), patch("siriusfuzz.cli.provenance", return_value={}), patch.object(
                    Orchestrator, "run", autospec=True, side_effect=run_campaign
                ):
                    self.assertEqual(cmd_run(args), exit_code)
                summary = json.loads(
                    next(pathlib.Path(tmp).glob("run-*/summary.json")).read_text()
                )
                self.assertEqual(summary["mode"], mode)

    def test_run_checks_readiness_first_and_defaults_to_a_time_budget(self):
        seen = {}

        def run_campaign(runner):
            seen["opts"] = runner.opts
            return runner.report.finish()

        engine = patch(
            "siriusfuzz.cli._engine",
            return_value=Engine("synthetic-shell", "synthetic-extension", False, []),
        )
        provenance = patch("siriusfuzz.cli.provenance", return_value={})
        campaign = patch.object(
            Orchestrator, "run", autospec=True, side_effect=run_campaign
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = build_parser().parse_args(["run", "--seed", "1", "--out", tmp])
            with engine, provenance, campaign as run, patch(
                "siriusfuzz.cli.run_doctor",
                return_value={"status": "setup_error", "error": "boom"},
            ) as doctor:
                self.assertEqual(cmd_run(args), 2)
                doctor.assert_called_once()
                run.assert_not_called()
            with engine, provenance, campaign, patch(
                "siriusfuzz.cli.run_doctor", return_value={"status": "ok"}
            ) as doctor:
                self.assertEqual(cmd_run(args), 0)
                doctor.assert_called_once()
            self.assertEqual(seen["opts"].duration, 600.0)
            self.assertIsNone(seen["opts"].max_queries)
            args = build_parser().parse_args(
                ["run", "--seed", "1", "--out", tmp, "--no-doctor", "--queries", "5"]
            )
            with engine, provenance, campaign, patch(
                "siriusfuzz.cli.run_doctor"
            ) as doctor:
                self.assertEqual(cmd_run(args), 0)
                doctor.assert_not_called()
            self.assertEqual(seen["opts"].max_queries, 5)
            self.assertIsNone(seen["opts"].duration)

    def test_known_gaps_stay_in_their_fallback_table(self):
        known = [
            KnownIssue(
                "Distinct aggregates not supported in GPU path",
                "sirius-db/sirius#1218",
                verdicts=["runtime_fallback"],
                sql_pattern="count\\(DISTINCT",
            ),
            KnownIssue(".", "sirius-db/sirius#9999", verdicts=["mismatch"]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            report = Report(pathlib.Path(tmp), load_config(None), known, 0, mode="gaps")
            tagged, _ = evaluate(
                gpu_count_distinct_error, sql="SELECT count(DISTINCT c1) FROM t"
            )
            tagged.labels = ["Agg(count,distinct)", "Select"]
            untagged, _ = evaluate(gpu_count_distinct_error, sql="SELECT 1 AS c0")
            untagged.labels = ["Select"]
            wrong, _ = evaluate(gpu_wrong, sql="SELECT c1 FROM t")
            for rec in (tagged, untagged, wrong):
                report.add(rec)
            self.assertEqual(tagged.verdict, Verdict.KNOWN_ISSUE.value)
            self.assertEqual(untagged.verdict, Verdict.RUNTIME_FALLBACK.value)
            self.assertEqual(wrong.verdict, Verdict.KNOWN_ISSUE.value)
            summary = report.finish()
            runtime = summary["gaps"]["runtime"]
            self.assertEqual(len(runtime), 1)
            self.assertEqual(runtime[0]["count"], 2)
            self.assertEqual(runtime[0]["issues"], ["sirius-db/sirius#1218"])
            text = report.render_summary(summary)
            self.assertIn("known: sirius-db/sirius#1218", text)
            # In the table once; not repeated under findings.
            self.assertEqual(text.count("Distinct aggregates not supported"), 1)
            self.assertIn("findings (1 unique)", text)
            self.assertIn("[known_issue]", text)
            self.assertIn("sirius-db/sirius#9999: ", text)

    def test_summary_groups_gaps_by_their_reduced_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Report(pathlib.Path(tmp), load_config(None), [], 0, mode="gaps")
            # Two rejections of different expressions around the same function, one
            # unrelated plan rejection, two runtime fallbacks with one reason but
            # different shapes, and a mismatch.
            records = []
            for expression, sql in (
                (
                    'regexp_matches(concat("a"."c1", "a"."c2"), \'x\')',
                    "SELECT f(g(c1, c2)) FROM t",
                ),
                (
                    'regexp_matches(substring("a"."c1", 1, 2), \'y\')',
                    "SELECT f(h(c1)) FROM t",
                ),
            ):
                rec, _ = evaluate(gpu_rejects(expression), sql=sql)
                rec.labels = ["Func(concat)", "Func(regexp_matches)", "Select"]
                report.add(rec)
                records.append(rec)
            other, _ = evaluate(gpu_plan_fallback, sql="SELECT w() OVER () FROM t")
            report.add(other)
            for sql, seconds, labels in (
                (
                    "SELECT count(DISTINCT c1) FROM t",
                    0.5,
                    ["Agg(count,distinct)", "Select"],
                ),
                (
                    "SELECT count(DISTINCT c2) FROM t WHERE k > 1",
                    0.25,
                    ["Agg(count,distinct)", "Compare(>)", "Select"],
                ),
            ):
                runtime, _ = evaluate(gpu_count_distinct_error, sql=sql)
                runtime.elapsed_gpu = seconds
                runtime.labels = labels
                report.add(runtime)
            wrong, _ = evaluate(gpu_wrong, sql="SELECT c1 FROM t")
            report.add(wrong)
            self.assertEqual(len(report.findings), 5)
            for rec, reduced in zip(
                records, ("SELECT f(c1) FROM t", "SELECT f(c1) FROM t_2")
            ):
                rec.reduced_sql = reduced
                rec.reduced_labels = ["Func(regexp_matches)", "Select"]
                rec.reduced_reason = (
                    UNSUPPORTED_REASON + 'regexp_matches("a"."c1", \'x\')'
                )
                report.add_reduction(rec)
            summary = report.finish()
            plan = summary["gaps"]["plan"]
            self.assertEqual([g["count"] for g in plan], [2, 1])
            self.assertEqual(plan[0]["sql"], "SELECT f(c1) FROM t")
            self.assertEqual(plan[0]["labels"], ["Func(regexp_matches)", "Select"])
            self.assertEqual(len(plan[0]["findings"]), 2)
            self.assertTrue(plan[0]["reduced"])
            self.assertEqual(plan[1]["reason"], "Window not supported")
            self.assertFalse(plan[1]["reduced"])
            runtime = summary["gaps"]["runtime"]
            self.assertEqual(len(runtime), 1)
            self.assertEqual(runtime[0]["count"], 2)
            self.assertAlmostEqual(runtime[0]["gpu_seconds"], 0.75)
            self.assertEqual(runtime[0]["sql"], "SELECT count(DISTINCT c1) FROM t")
            text = report.render_summary(summary)
            self.assertIn(
                "runtime fallbacks (1 reason, 2 queries, 0.8s of GPU work thrown away)",
                text,
            )
            self.assertIn("   0.8s  Distinct aggregates not supported", text)
            self.assertIn("plan-time fallbacks (2 reasons, 3 queries):", text)
            self.assertLess(
                text.index("runtime fallbacks ("), text.index("plan-time fallbacks (")
            )
            # Findings outrank gaps: listed first, with their query and difference.
            self.assertLess(
                text.index("findings (1 unique)"), text.index("runtime fallbacks (")
            )
            self.assertIn("      SELECT c1 FROM t\n", text)
            self.assertRegex(
                text,
                r"findings \(1 unique\).*\n.*mismatch.*\n      SELECT c1 FROM t\n(?:      features: .*\n)?      row 1 col 0",
            )
            # A reduced plan group shows its smallest query; an unreduced one only
            # its normalized reason.
            self.assertIn("SELECT f(c1) FROM t\n", text)
            self.assertIn("features: Func(regexp_matches), Select", text)
            self.assertNotIn("SELECT w() OVER ()", text)
            # The mismatch is listed as a finding; the gaps are not listed twice.
            self.assertIn("findings (1 unique)", text)
            self.assertEqual(text.count("Window not supported"), 1)

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
                    "Distinct aggregates not supported in GPU path",
                    "sirius-db/sirius#1218",
                    verdicts=["runtime_fallback"],
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
                "FINDING.md",
            ):
                self.assertTrue((d / f).exists(), f)
            bundles = sorted((run_dir / "findings").rglob("dataset.sql"))
            self.assertEqual(len(bundles), 2)
            self.assertNotEqual(bundles[0].read_text(), bundles[1].read_text())
            rec3, _ = evaluate(
                gpu_count_distinct_error, sql="SELECT count(DISTINCT k) AS c0 FROM t"
            )
            report.add(rec3)
            self.assertEqual(rec3.verdict, Verdict.KNOWN_ISSUE.value)
            self.assertIn("#1218", rec3.reason)
            summary = report.finish()
            self.assertEqual(summary["counts"]["runtime_fallback"], 2)
            self.assertEqual(summary["counts"]["known_issue"], 1)
            self.assertTrue((run_dir / "summary.txt").exists())
            self.assertIn("runtime_fallback", report.render_summary(summary))


if __name__ == "__main__":
    unittest.main()
