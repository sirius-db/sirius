"""Triage evidence, recovery and human-review contracts; all review fixtures are synthetic."""

import argparse
import copy
import pathlib
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from . import conftest_path  # noqa: F401
from siriusfuzz.artifacts import seal, write_json
from siriusfuzz.cli import build_parser
from siriusfuzz.triage import (
    active_batch,
    describe_outcome,
    discover,
    evidence_hashes,
    failure_matches,
    import_candidate,
    process_candidate,
    read,
    render,
    summarize,
    attempt,
    replay_command,
    suggested_group,
    ReductionBudgetReached,
)
from siriusfuzz.triage_reduce import (
    dataset_candidates,
    permute_dataset,
    query_candidates,
)
from siriusfuzz.triage_review import cmd_group, cmd_issue_draft, cmd_review


def descriptor(kind="mismatch", reference="stable", reason="different values"):
    return {
        "kind": kind,
        "reason": reason,
        "key": {
            "kind": kind,
            "phase": "gpu",
            "reason": reason,
            "variant": None,
            "execution_path": "fallback" if kind == "fallback_mismatch" else "strict",
        },
        "reference": reference,
        "reference_status": "ok",
        "gpu_interception": True,
        "engine": {"sha256": "synthetic-test-engine"},
        "comparison": "multiset",
        "duckdb_binary": {"sha256": "synthetic-duckdb"},
    }


def fixture(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "query.sql").write_text("SELECT k FROM t WHERE k > 0;")
    (path / "dataset.sql").write_text(
        "CREATE TABLE t(k INT); INSERT INTO t VALUES (1),(2),(3);"
    )
    (path / "config.toml").write_text("# synthetic fixture\n")
    (path / "sirius.yaml").write_text("# synthetic fixture\n")
    write_json(path / "meta.json", {"verdict": "mismatch", "comparison": "multiset"})
    write_json(
        path / "runtime.json",
        {
            "session_settings": {"threads": "1"},
            "duckdb_binary": {"sha256": "synthetic-duckdb"},
        },
    )
    write_json(
        path / "environment.json", {"extension": {"sha256": "synthetic-test-engine"}}
    )
    write_json(path / "canary.json", {"ok": True})
    write_json(
        path / "outcome.json",
        {
            "status": "ok",
            "record": {
                "verdict": "mismatch",
                "reason": "different values",
                "comparison": "multiset",
                "evidence": {
                    "operations": [
                        {
                            "phase": "cpu",
                            "status": "ok",
                            "fingerprint_multiset": "stable",
                        },
                        {"phase": "gpu", "status": "ok"},
                    ]
                },
            },
        },
    )
    seal(path)
    return path


class ReductionTests(unittest.TestCase):
    def test_literals_comments_and_nested_commas_survive_row_reduction(self):
        import duckdb

        sql = "CREATE TABLE t(s VARCHAR, n INT); /* VALUES fake */ INSERT INTO t VALUES('a,;(''x'')', 1), ('b', 2), ('c', 3);"
        for candidate in dataset_candidates(sql):
            with duckdb.connect() as con:
                con.execute(candidate)
                self.assertLess(len(con.execute("SELECT * FROM t").fetchall()), 3)
        permuted, available = permute_dataset(sql)
        self.assertTrue(available)
        with duckdb.connect() as con:
            con.execute(permuted)
            self.assertEqual(
                con.execute("SELECT * FROM t").fetchall(),
                [("c", 3), ("b", 2), ("a,;('x')", 1)],
            )

    def test_projection_and_clause_candidates_parse(self):
        import duckdb

        sql = "SELECT k, coalesce(k, 2), 'FROM,WHERE' FROM t WHERE k > 0 ORDER BY k LIMIT 5;"
        with duckdb.connect() as con:
            con.execute("CREATE TABLE t(k INT); INSERT INTO t VALUES (1),(2)")
            candidates = list(query_candidates(sql))
            self.assertEqual(len(candidates), 6)
            for query in candidates:
                con.execute(query).fetchall()

    def test_unsupported_quoting_is_declined(self):
        self.assertEqual(list(query_candidates("SELECT $$a,b$$ FROM t")), [])
        self.assertEqual(
            permute_dataset("INSERT INTO t VALUES (E'a\\nb'),('c');")[1], False
        )

    def test_plain_backslash_literal_can_be_permuted(self):
        import duckdb

        data = "CREATE TABLE t(s VARCHAR); INSERT INTO t VALUES ('\\'), ('other');"
        permuted, available = permute_dataset(data)
        self.assertTrue(available)
        with duckdb.connect() as con:
            con.execute(permuted)
            self.assertEqual(
                con.execute("SELECT s FROM t").fetchall(), [("other",), ("\\",)]
            )


class ClassificationTests(unittest.TestCase):
    def summary(self, *descriptions):
        return summarize([{"description": d} for d in descriptions], "mismatch")[
            "status"
        ]

    def test_observation_is_not_manual_verification(self):
        self.assertEqual(
            self.summary(descriptor(), descriptor()), "automatically_reproduced"
        )
        self.assertEqual(self.summary(descriptor(), descriptor("ok")), "intermittent")
        self.assertEqual(
            self.summary(descriptor(), descriptor(reference="changed")),
            "unstable_reference",
        )
        self.assertEqual(self.summary(descriptor("gpu_error")), "changed_failure")
        self.assertEqual(self.summary(descriptor("ok")), "not_reproduced")
        d = descriptor()
        d["gpu_interception"] = False
        self.assertEqual(self.summary(d), "needs_investigation")

    def test_reducer_rejects_unrelated_failure_or_failed_cpu(self):
        self.assertFalse(failure_matches(descriptor(), descriptor("gpu_error")))
        self.assertFalse(
            failure_matches(descriptor(), descriptor(reason="row count mismatch"))
        )
        d = descriptor()
        d["reference_status"] = "error"
        self.assertFalse(failure_matches(descriptor(), d))

    def test_failure_identity_distinguishes_strict_and_fallback_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(pathlib.Path(tmp))
            outcome = read(path / "outcome.json")
            outcome["record"].update(
                verdict="gpu_error", reason="conversion failed", context={}
            )
            strict = describe_outcome(path, outcome)
            outcome["record"]["context"] = {
                "plan_fallback_reason": "Window not supported"
            }
            fallback = describe_outcome(path, outcome)
            self.assertFalse(failure_matches(fallback, strict))
            self.assertFalse(failure_matches(strict, fallback))
            self.assertTrue(failure_matches(fallback, describe_outcome(path, outcome)))

    def test_timeout_deadline_and_phase_are_part_of_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(pathlib.Path(tmp))
            base = {
                "status": "timeout",
                "deadline_seconds": 90,
                "last_operation": {"phase": "gpu"},
            }
            a = describe_outcome(path, base)
            b = describe_outcome(path, {**base, "deadline_seconds": 1})
            c = describe_outcome(path, {**base, "last_operation": {"phase": "cpu"}})
            self.assertNotEqual(a["key"], b["key"])
            self.assertNotEqual(a["key"], c["key"])

    def test_crash_matching_preserves_offsets_and_build_but_ignores_aslr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = fixture(pathlib.Path(tmp))
            outcome = read(path / "outcome.json")
            outcome.update(
                status="crash", exitcode=-11, last_operation={"phase": "gpu"}
            )
            outcome["record"].update(verdict="crash")

            def observation(offset, address):
                (path / "stderr.log").write_text(
                    "*** SIGSEGV ***\n"
                    "#0 /tmp/sirius.duckdb_extension(+0xaaaaaa) [0x1]\n"
                    f"#2 /tmp/sirius.duckdb_extension(+0x{offset}) [0x{address}]\n"
                    "*** end backtrace ***\n"
                )
                return describe_outcome(path, outcome)

            original = observation("111111", "123456")
            relocated = observation("111111", "789abc")
            different = observation("222222", "123456")
            self.assertTrue(failure_matches(original, relocated))
            self.assertFalse(failure_matches(original, different))
            write_json(
                path / "environment.json", {"extension": {"sha256": "another-build"}}
            )
            self.assertFalse(failure_matches(original, observation("111111", "123456")))


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)
        self.source = fixture(self.base / "finding")
        self.root = self.base / "triage"
        self.directory = import_candidate(self.root, self.source)
        self.identifier = self.directory.name
        self.engine = self.base / "engine"
        self.engine.write_text("synthetic engine v1")
        self.args = build_parser().parse_args(
            ["triage", str(self.source), "--out", str(self.root), "--no-reduce"]
        )
        self.args.timeout = 5
        self.args.runtime_identity = {"duckdb_binary": "synthetic-runtime-v1"}
        self.options = {"extension": str(self.engine)}

    def fake_attempt(
        self, batch, tag, source, options, timeout, query=None, dataset=None
    ):
        work = batch / "attempts" / tag / "run-synthetic"
        if not work.exists():
            shutil.copytree(source, work)
        if query:
            shutil.copy(query, work / "query.sql")
        if dataset:
            shutil.copy(dataset, work / "dataset.sql")
        seal(work)
        return {
            "tag": tag,
            "evidence": work.name,
            "hashes": evidence_hashes(work),
            "description": descriptor(),
        }

    def run_batch(self, effect=None):
        with patch(
            "siriusfuzz.triage.attempt", side_effect=effect or self.fake_attempt
        ):
            process_candidate(
                self.directory, self.args, self.options, time.monotonic() + 60
            )
        return active_batch(self.directory, read(self.directory / "candidate.json"))

    def test_copied_bundle_has_stable_identity_and_preserves_source(self):
        before = evidence_hashes(self.source)
        other = self.base / "copied"
        shutil.copytree(self.source, other)
        self.assertEqual(import_candidate(self.root, other), self.directory)
        self.assertEqual(len(read(self.directory / "candidate.json")["sources"]), 2)
        self.assertEqual(evidence_hashes(self.source), before)
        self.assertEqual(discover([str(other)]), [other.resolve()])

    def test_interrupted_batch_resumes_and_binary_change_starts_new_batch(self):
        def interrupted(batch, tag, *rest):
            if tag == "original-002":
                raise KeyboardInterrupt
            return self.fake_attempt(batch, tag, *rest)

        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(interrupted)
        before, batch = active_batch(
            self.directory, read(self.directory / "candidate.json")
        )
        self.assertEqual(batch["status"], "interrupted")
        self.assertEqual(len(batch["original"]), 1)
        after, batch = self.run_batch()
        self.assertEqual(before, after)
        self.assertEqual(len(batch["original"]), 3)
        self.engine.write_text("synthetic engine v2")
        new, _ = self.run_batch()
        self.assertNotEqual(new, before)
        self.assertTrue((before / "batch.json").exists())

    def test_unrelated_failure_never_accepted_as_reduction(self):
        self.args.no_reduce = False
        self.args.reduce_steps = 2

        def changed(batch, tag, *rest):
            result = self.fake_attempt(batch, tag, *rest)
            if tag.startswith("reduce-"):
                result["description"] = descriptor("gpu_error")
            return result

        _, batch = self.run_batch(changed)
        self.assertEqual(len(batch["reduction"]["trials"]), 2)
        self.assertEqual(batch["reduction"]["accepted"], [])

    def test_reduction_cannot_promote_a_strict_error_for_a_fallback_error(self):
        metadata = read(self.source / "meta.json")
        metadata.update(
            verdict="gpu_error",
            reason="conversion failed",
            context={"plan_fallback_reason": "Window not supported"},
        )
        write_json(self.source / "meta.json", metadata)
        outcome = read(self.source / "outcome.json")
        outcome["record"].update(metadata)
        write_json(self.source / "outcome.json", outcome)
        seal(self.source)
        self.directory = import_candidate(self.root, self.source)
        self.args.no_reduce = False
        self.args.reduce_steps = 2

        def changed(batch, tag, *rest):
            receipt = self.fake_attempt(batch, tag, *rest)
            path = batch / "attempts" / tag / receipt["evidence"]
            result = read(path / "outcome.json")
            if tag.startswith("reduce-"):
                result["record"]["context"] = {}
            write_json(path / "outcome.json", result)
            seal(path)
            receipt["hashes"] = evidence_hashes(path)
            receipt["description"] = describe_outcome(path, result)
            return receipt

        _, batch = self.run_batch(changed)
        self.assertEqual(batch["automatic"]["status"], "automatically_reproduced")
        self.assertEqual(len(batch["reduction"]["trials"]), 2)
        self.assertEqual(batch["reduction"]["accepted"], [])

    def fake_replay(self, args):
        work = pathlib.Path(args.out) / "run-synthetic"
        shutil.copytree(args.target, work, dirs_exist_ok=True)
        if args.query_override:
            shutil.copy(args.query_override, work / "query.sql")
        if args.dataset:
            shutil.copy(args.dataset, work / "dataset.sql")
        seal(work)
        return 1

    def test_actual_receipt_cache_and_missing_or_changed_evidence(self):
        with patch("siriusfuzz.cli.cmd_replay", side_effect=self.fake_replay) as replay:
            process_candidate(
                self.directory, self.args, self.options, time.monotonic() + 60
            )
            self.assertEqual(replay.call_count, 4)
            before, _ = active_batch(
                self.directory, read(self.directory / "candidate.json")
            )
            process_candidate(
                self.directory, self.args, self.options, time.monotonic() + 60
            )
            self.assertEqual(replay.call_count, 4)
            render(self.root)
            self.assertEqual(replay.call_count, 4)
            evidence = before / "attempts/original-001/run-synthetic/query.sql"
            evidence.write_text("changed")
            process_candidate(
                self.directory, self.args, self.options, time.monotonic() + 60
            )
            self.assertEqual(replay.call_count, 4)
            self.assertIn("evidence changed", read(before / "batch.json")["error"])
            evidence.unlink()
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                attempt(before, "original-001", self.source, self.options, 5)

    def test_budget_exhaustion_is_not_an_engine_timeout(self):
        with patch("siriusfuzz.cli.cmd_replay", side_effect=self.fake_replay) as replay:
            with self.assertRaises(TimeoutError):
                process_candidate(
                    self.directory, self.args, self.options, time.monotonic() + 1
                )
            before, batch = active_batch(
                self.directory, read(self.directory / "candidate.json")
            )
            self.assertEqual(batch["original"], [])
            replay.assert_not_called()
            process_candidate(
                self.directory, self.args, self.options, time.monotonic() + 60
            )
            after, batch = active_batch(
                self.directory, read(self.directory / "candidate.json")
            )
            self.assertEqual(before, after)
            self.assertEqual(batch["automatic"]["status"], "automatically_reproduced")
            self.assertTrue(
                all(call.args[0].timeout == 5 for call in replay.call_args_list)
            )

    def test_changed_runtime_invalidates_receipts(self):
        before, _ = self.run_batch()
        self.args.runtime_identity["duckdb_binary"] = "synthetic-runtime-v2"
        after, _ = self.run_batch()
        self.assertNotEqual(before, after)

    def test_cancelled_reduction_verification_resumes_same_trial(self):
        self.args.no_reduce = False
        self.args.reduce_steps = 1

        def interrupted(batch, tag, *rest):
            if tag.endswith("-verify"):
                raise KeyboardInterrupt
            return self.fake_attempt(batch, tag, *rest)

        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(interrupted)
        before, batch = active_batch(
            self.directory, read(self.directory / "candidate.json")
        )
        self.assertFalse(batch["reduction"]["trials"][0]["complete"])
        after, batch = self.run_batch()
        self.assertEqual(before, after)
        self.assertEqual(len(batch["reduction"]["trials"]), 1)
        self.assertTrue(batch["reduction"]["trials"][0]["complete"])
        self.assertEqual(len(batch["reduction"]["accepted"]), 1)

    def test_reduction_budget_resumes_pending_work_and_reuses_receipts(self):
        self.args.no_reduce = False
        self.args.reduce_steps = 1
        for pending in ("reduce-001", "reduce-001-verify", "reduce-001-permutation"):
            with self.subTest(pending=pending):
                self.args.rerun = True
                real_attempt = attempt

                def budget_reached(directory, tag, *args):
                    if tag == pending:
                        raise ReductionBudgetReached
                    return real_attempt(directory, tag, *args)

                with patch(
                    "siriusfuzz.cli.cmd_replay", side_effect=self.fake_replay
                ) as replay, patch(
                    "siriusfuzz.triage.dataset_candidates",
                    side_effect=lambda data: iter(
                        [data.replace("(1),(2),(3)", "(1),(2)")]
                    ),
                ):
                    before, batch = self.run_batch(budget_reached)
                    reduction = batch["reduction"]
                    self.assertEqual(batch["status"], "incomplete")
                    self.assertFalse(reduction["finished"])
                    self.assertTrue(reduction["budget_reached"])
                    self.assertFalse(reduction["trials"][0]["complete"])
                    self.assertEqual(
                        self.assessment(before, batch)["status"], "needs_investigation"
                    )
                    already_run = replay.call_count

                    # Also recover inconsistent flags saved by older code when
                    # the recipe is otherwise unchanged.
                    reduction["finished"] = True
                    write_json(before / "batch.json", batch)
                    self.args.rerun = False
                    after, batch = self.run_batch(real_attempt)
                    self.assertEqual(after, before)
                    self.assertEqual(batch["status"], "complete")
                    self.assertTrue(batch["reduction"]["finished"])
                    self.assertFalse(batch["reduction"]["budget_reached"])
                    self.assertEqual(len(batch["reduction"]["trials"]), 1)
                    self.assertTrue(batch["reduction"]["trials"][0]["complete"])
                    self.assertEqual(replay.call_count, 7)
                    self.assertGreater(replay.call_count, already_run)
                    self.assertEqual(
                        self.assessment(after, batch)["status"],
                        "automatically_validated",
                    )

    def test_reduced_input_integrity_and_faithful_report_command(self):
        import shlex

        self.args.no_reduce = False
        self.args.reduce_steps = 1
        config = self.base / "non default config.yaml"
        config.write_text("# synthetic config")
        self.options.update(sirius_config=str(config), allow_metadata_mismatch=False)
        batch_dir, batch = self.run_batch()
        best = batch["reduction"]["accepted"][-1]
        render(self.root)
        report = (self.directory / "REPORT.md").read_text()
        evidence = (
            batch_dir
            / "attempts"
            / best["attempt"]["tag"]
            / best["attempt"]["evidence"]
        )
        command = replay_command(evidence, batch)
        parsed = build_parser().parse_args(shlex.split(command)[5:])
        self.assertEqual(parsed.target, str(evidence))
        self.assertEqual(parsed.extension, str(self.engine))
        self.assertEqual(parsed.sirius_config, [str(config)])
        self.assertFalse(parsed.allow_metadata_mismatch)
        self.assertIn(command, report)
        cmd_review(self.review_args())
        (batch_dir / best["dataset"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "reduced inputs changed"):
            cmd_issue_draft(self.draft_args())
        render(self.root)
        self.assertIn("evidence integrity error", (self.root / "REPORT.md").read_text())

    def test_malformed_and_incomplete_inputs_are_reported(self):
        for contents in ("{not valid json", "[]"):
            (self.source / "meta.json").write_text(contents)
            self.directory = import_candidate(self.root, self.source)
            _, batch = self.run_batch()
            self.assertEqual(batch["status"], "incomplete")
            render(self.root)
        (self.source / "bundle.json").unlink()
        (self.source / "meta.json").write_text("{}")
        (self.source / "dataset.sql").unlink()
        self.directory = import_candidate(self.root, self.source)
        _, batch = self.run_batch()
        self.assertIn("missing dataset.sql", batch["error"])

    def test_query_shape_refines_suggested_groups(self):
        other = fixture(self.base / "another")
        (other / "query.sql").write_text("SELECT substring(k, 1, 2) FROM t")
        self.assertNotEqual(
            suggested_group(self.source, "row mismatch"),
            suggested_group(other, "row mismatch"),
        )

    def review_args(self):
        # Synthetic evidence only: this test never attests to a real engine finding.
        evidence = fixture(self.base / "synthetic-human-replay")
        return argparse.Namespace(
            workspace=str(self.root),
            candidate=self.identifier,
            disposition="manually_verified",
            reviewer="Synthetic unit-test reviewer",
            notes="Synthetic fixture only",
            acknowledge_manual_verification=True,
            evidence=str(evidence),
            expected="synthetic CPU result",
            actual="synthetic GPU result",
            duplicate_of=None,
        )

    def draft_args(self):
        return argparse.Namespace(
            workspace=str(self.root), candidate=self.identifier, title=None
        )

    def automatic_batch(self):
        self.args.no_reduce = False
        self.args.reduce_steps = 1
        return self.run_batch()

    def assessment(self, batch_dir, batch, review=None):
        from siriusfuzz.triage_drafts import assess

        return assess(
            self.directory,
            read(self.directory / "candidate.json"),
            batch_dir,
            batch,
            review,
        )

    def test_automatic_drafts_use_saved_evidence_without_human_attestation(self):
        self.automatic_batch()
        with patch(
            "siriusfuzz.triage.attempt",
            side_effect=AssertionError("must not execute SQL"),
        ):
            args = build_parser().parse_args(["issue-draft", str(self.root), "--all"])
            self.assertEqual(cmd_issue_draft(args), 0)
            self.assertEqual(cmd_issue_draft(self.draft_args()), 0)
        item = read(self.root / "drafts.json")["candidates"][0]
        self.assertEqual(item["validation"]["status"], "automatically_validated")
        draft = self.root / item["draft"]
        before = draft.stat().st_mtime_ns
        render(self.root)
        self.assertEqual(draft.stat().st_mtime_ns, before)
        self.assertIn("No human verification is claimed", draft.read_text())
        self.assertIn("--original", draft.read_text())
        self.assertFalse((self.directory / "review.json").exists())

    def no_yaml_batch(self):
        source = self.directory / "source"
        (source / "sirius.yaml").unlink()
        (source / "config.toml").write_text("[sirius]\nconfigs = []\n")
        metadata = read(source / "meta.json")
        metadata["execution"] = {"cpu_only": False, "sirius_config_required": False}
        write_json(source / "meta.json", metadata)
        runtime = read(source / "runtime.json")
        runtime["sirius_config_mode"] = "builtin_defaults"
        write_json(source / "runtime.json", runtime)
        seal(source)
        state = read(self.directory / "candidate.json")
        state["source_hashes"] = evidence_hashes(source)
        write_json(self.directory / "candidate.json", state)
        # Keep two rows in the reduction so its permutation is checked as well.
        with patch(
            "siriusfuzz.triage.dataset_candidates",
            side_effect=lambda data: iter([data.replace("(1),(2),(3)", "(1),(2)")]),
        ):
            return self.automatic_batch()

    def test_automatic_no_yaml_finding_exports_draft(self):
        batch_dir, batch = self.no_yaml_batch()
        validation = self.assessment(batch_dir, batch)
        self.assertEqual(validation["status"], "automatically_validated", validation)
        self.assertIn("permutation", batch["reduction"]["accepted"][0])
        self.assertEqual(cmd_issue_draft(self.draft_args()), 0)
        item = read(self.root / "drafts.json")["candidates"][0]
        self.assertEqual(item["validation"]["status"], "automatically_validated")
        self.assertIn(
            "No explicit Sirius YAML was recorded",
            (self.root / item["draft"]).read_text(),
        )
        self.assertFalse(list(self.directory.rglob("sirius.yaml")))

    def test_matching_replay_yaml_cannot_replace_the_source_configuration(self):
        real_attempt = self.fake_attempt

        def changed_yaml(batch, tag, *args):
            receipt = real_attempt(batch, tag, *args)
            path = batch / "attempts" / tag / receipt["evidence"]
            (path / "sirius.yaml").write_text("# same replacement in every replay\n")
            metadata = read(path / "meta.json")
            metadata["execution"] = {"cpu_only": False, "sirius_config_required": True}
            write_json(path / "meta.json", metadata)
            runtime = read(path / "runtime.json")
            runtime["sirius_config_mode"] = "explicit_yaml"
            write_json(path / "runtime.json", runtime)
            seal(path)
            receipt["hashes"] = evidence_hashes(path)
            return receipt

        for source_yaml in (True, False):
            with self.subTest(source_yaml=source_yaml), patch.object(
                self, "fake_attempt", side_effect=changed_yaml
            ):
                self.args.rerun = True
                batch_dir, batch = (
                    self.automatic_batch() if source_yaml else self.no_yaml_batch()
                )
                result = self.assessment(batch_dir, batch)
                self.assertEqual(result["status"], "needs_investigation", result)
                self.assertIn("source sirius.yaml", result["reasons"][0])

    def test_no_yaml_source_without_selection_proof_needs_investigation(self):
        batch_dir, batch = self.no_yaml_batch()
        source = self.directory / "source"
        runtime = read(source / "runtime.json")
        runtime.pop("sirius_config_mode")
        write_json(source / "runtime.json", runtime)
        seal(source)
        state = read(self.directory / "candidate.json")
        state["source_hashes"] = evidence_hashes(source)
        write_json(self.directory / "candidate.json", state)
        result = self.assessment(batch_dir, batch)
        self.assertEqual(result["status"], "needs_investigation", result)
        self.assertIn("source sirius.yaml", result["reasons"][0])

    def test_no_yaml_gate_requires_optional_source_metadata(self):
        batch_dir, batch = self.no_yaml_batch()
        source = self.directory / "source"
        for requirement in (True, "missing"):
            with self.subTest(requirement=requirement):
                metadata = read(source / "meta.json")
                if requirement == "missing":
                    metadata["execution"].pop("sirius_config_required")
                else:
                    metadata["execution"]["sirius_config_required"] = requirement
                write_json(source / "meta.json", metadata)
                seal(source)
                state = read(self.directory / "candidate.json")
                state["source_hashes"] = evidence_hashes(source)
                write_json(self.directory / "candidate.json", state)
                result = self.assessment(batch_dir, batch)
                self.assertEqual(result["status"], "needs_investigation")
                self.assertIn("sirius.yaml", result["reasons"][0])

    def test_no_yaml_gate_rejects_inconsistent_attempts(self):
        batch_dir, batch = self.no_yaml_batch()
        for receipt in (
            batch["original"][0],
            batch["original"][1],
            batch["permutation"],
        ):
            path = batch_dir / "attempts" / receipt["tag"] / receipt["evidence"]
            metadata = (path / "meta.json").read_bytes()
            runtime = (path / "runtime.json").read_bytes()
            for change in (
                "added_yaml",
                "required_yaml",
                "unknown_requirement",
                "unknown_selection",
                "missing_toml",
            ):
                with self.subTest(tag=receipt["tag"], change=change):
                    config = (path / "config.toml").read_bytes()
                    if change == "added_yaml":
                        (path / "sirius.yaml").write_text("# unexpected config\n")
                    elif change == "missing_toml":
                        (path / "config.toml").unlink()
                    elif change == "unknown_selection":
                        modified = read(path / "runtime.json")
                        modified.pop("sirius_config_mode")
                        write_json(path / "runtime.json", modified)
                    else:
                        modified = read(path / "meta.json")
                        if change == "required_yaml":
                            modified["execution"]["sirius_config_required"] = True
                        else:
                            modified["execution"].pop("sirius_config_required")
                        write_json(path / "meta.json", modified)
                    seal(path)
                    receipt["hashes"] = evidence_hashes(path)
                    result = self.assessment(batch_dir, batch)
                    self.assertEqual(result["status"], "needs_investigation", result)
                    name = "config.toml" if change == "missing_toml" else "sirius.yaml"
                    self.assertIn(name, result["reasons"][0])
                    (path / "sirius.yaml").unlink(missing_ok=True)
                    (path / "config.toml").write_bytes(config)
                    (path / "meta.json").write_bytes(metadata)
                    (path / "runtime.json").write_bytes(runtime)
                    seal(path)
                    receipt["hashes"] = evidence_hashes(path)

    def test_large_reproducer_has_copyable_draft_and_full_dataset_attachment(self):
        source = self.directory / "source"
        data = (
            "CREATE TABLE t(k INT, note VARCHAR DEFAULT '"
            + "x" * 20000
            + "'); INSERT INTO t(k) VALUES (1),(2);"
        )
        (source / "dataset.sql").write_text(data)
        seal(source)
        state = read(self.directory / "candidate.json")
        state["source_hashes"] = evidence_hashes(source)
        write_json(self.directory / "candidate.json", state)
        self.automatic_batch()
        render(self.root)
        item = read(self.root / "drafts.json")["candidates"][0]
        self.assertEqual(item["validation"]["status"], "automatically_validated")
        body = (self.root / item["draft"]).read_text()
        attachment = self.root / item["dataset_attachment"]
        self.assertLess(len(body.encode()), 15000)
        self.assertIn(attachment.name, body)
        evidence = self.directory / item["validation"]["evidence"]
        self.assertEqual(
            attachment.read_bytes(), (evidence / "dataset.sql").read_bytes()
        )

    def test_automatic_gate_rejects_incomplete_and_reused_evidence(self):
        batch_dir, batch = self.automatic_batch()
        for case in (
            "two_attempts",
            "duplicate_attempt",
            "no_permutation",
            "no_reduction",
            "unfinished_reduction",
            "duplicate_reduction",
            "disabled_reduction",
        ):
            with self.subTest(case=case):
                changed = copy.deepcopy(batch)
                if case == "two_attempts":
                    changed["original"].pop()
                elif case == "duplicate_attempt":
                    changed["original"][1] = changed["original"][0]
                elif case == "no_permutation":
                    changed.pop("permutation")
                elif case == "no_reduction":
                    changed.pop("reduction")
                elif case == "unfinished_reduction":
                    changed["reduction"]["accepted"][-1]["complete"] = False
                elif case == "disabled_reduction":
                    changed["recipe"]["reduce_steps"] = 0
                else:
                    best = changed["reduction"]["accepted"][-1]
                    best["verification"] = best["attempt"]
                self.assertEqual(
                    self.assessment(batch_dir, changed)["status"], "needs_investigation"
                )

    def test_automatic_gate_recomputes_outcomes_and_checks_environment(self):
        batch_dir, batch = self.automatic_batch()
        receipt = batch["original"][1]
        path = batch_dir / "attempts" / receipt["tag"] / receipt["evidence"]
        originals = {
            name: (path / name).read_bytes()
            for name in ("runtime.json", "outcome.json", "sirius.yaml")
        }
        for case in (
            "unstable_cpu",
            "intermittent",
            "missing_gpu",
            "runtime_changed",
            "yaml_changed",
            "bypassed_metadata",
            "timeout",
            "setup_error",
            "gpu_oom",
            "plan_rejection",
        ):
            with self.subTest(case=case):
                for name, content in originals.items():
                    (path / name).write_bytes(content)
                outcome = read(path / "outcome.json")
                if case == "unstable_cpu":
                    outcome["record"]["evidence"]["operations"][0][
                        "fingerprint_multiset"
                    ] = "different"
                elif case == "missing_gpu":
                    outcome["record"]["evidence"]["operations"].pop()
                elif case in ("runtime_changed", "bypassed_metadata"):
                    runtime = read(path / "runtime.json")
                    if case == "runtime_changed":
                        runtime["duckdb_binary"]["sha256"] = "different"
                    else:
                        runtime["metadata_mismatch_bypassed"] = True
                    write_json(path / "runtime.json", runtime)
                elif case == "yaml_changed":
                    (path / "sirius.yaml").write_text("changed")
                else:
                    outcome["record"]["verdict"] = (
                        "ok" if case == "intermittent" else case
                    )
                write_json(path / "outcome.json", outcome)
                seal(path)
                receipt["hashes"] = evidence_hashes(path)
                # Leave the cached description unchanged: classification must use saved outcomes.
                self.assertEqual(
                    self.assessment(batch_dir, batch)["status"], "needs_investigation"
                )

    def test_automatic_gate_respects_human_dispositions(self):
        batch_dir, batch = self.automatic_batch()
        for disposition in (
            "expected_behavior",
            "needs_investigation",
            "duplicate",
            "harness_problem",
        ):
            with self.subTest(disposition=disposition):
                result = self.assessment(
                    batch_dir, batch, {"disposition": disposition, "binding": "stale"}
                )
                self.assertEqual(result["status"], "needs_investigation")
                self.assertIn("human disposition", result["reasons"][0])

    def test_automatic_draft_preserves_user_edits(self):
        self.automatic_batch()
        render(self.root)
        item = read(self.root / "drafts.json")["candidates"][0]
        draft = self.root / item["draft"]
        draft.write_text("User edited draft")
        render(self.root)
        item = read(self.root / "drafts.json")["candidates"][0]
        self.assertIsNone(item["draft"])
        self.assertIn("preserving it", item["draft_error"])
        self.assertEqual(draft.read_text(), "User edited draft")

    def test_stale_or_tampered_evidence_removes_current_draft_link(self):
        self.automatic_batch()
        render(self.root)
        prior = self.root / read(self.root / "drafts.json")["candidates"][0]["draft"]
        self.args.no_reduce = True
        self.run_batch()
        render(self.root)
        self.assertIsNone(read(self.root / "drafts.json")["candidates"][0]["draft"])
        self.assertTrue(prior.is_file())
        self.automatic_batch()
        (self.directory / "source" / "query.sql").write_text("tampered")
        render(self.root)
        self.assertIsNone(read(self.root / "drafts.json")["candidates"][0]["draft"])

    def test_automatic_semantic_screen_declines_uncertain_sql(self):
        from siriusfuzz.triage_drafts import screen_query

        screen_query(
            "SELECT (substring(s, 2)), regexp_replace(s, 'a', 'x') FROM t WHERE (k > 0)"
        )
        screen_query(
            "(SELECT millisecond(ts) FROM t) UNION ALL (SELECT microsecond(ts) FROM t)"
        )
        screen_query(
            "SELECT k, count(*) FROM (SELECT k FROM t) AS x GROUP BY (k % 3), k"
        )
        screen_query("SELECT x.k FROM t LEFT JOIN (SELECT k FROM u) AS x ON t.k = x.k")
        for sql in (
            "SELECT random()",
            "SELECT k FROM t LIMIT 1",
            "SELECT sum(k) OVER () FROM t",
            "SELECT current_timestamp",
            'SELECT "unknown"(k) FROM t',
        ):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                screen_query(sql)

    def test_manual_gate_and_local_export(self):
        self.run_batch()
        with self.assertRaisesRegex(ValueError, "manual verification"):
            cmd_issue_draft(self.draft_args())
        args = self.review_args()
        args.acknowledge_manual_verification = False
        with self.assertRaisesRegex(ValueError, "attestation"):
            cmd_review(args)
        args.acknowledge_manual_verification = True
        self.assertEqual(cmd_review(args), 0)
        self.assertEqual(cmd_issue_draft(self.draft_args()), 0)
        drafts = list((self.root / "drafts").glob("*.md"))
        self.assertEqual(len(drafts), 1)
        self.assertIn("Synthetic unit-test reviewer", drafts[0].read_text())
        self.assertIn("Local issue draft", drafts[0].read_text())
        drafts[0].write_text("User edited human-reviewed draft")
        with self.assertRaisesRegex(ValueError, "preserving it"):
            cmd_issue_draft(self.draft_args())
        self.assertEqual(drafts[0].read_text(), "User edited human-reviewed draft")

    def test_changed_evidence_prevents_export(self):
        self.run_batch()
        cmd_review(self.review_args())
        review = read(self.directory / "review.json")[-1]
        (self.directory / review["evidence"] / "dataset.sql").write_text("modified")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            cmd_issue_draft(self.draft_args())

    def test_new_batch_makes_review_stale(self):
        self.run_batch()
        cmd_review(self.review_args())
        self.args.rerun = True
        self.run_batch()
        render(self.root)
        self.assertTrue(
            read(self.root / "report.json")["candidates"][0]["review_stale"]
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            cmd_issue_draft(self.draft_args())

    def test_configuration_change_rejects_manual_evidence(self):
        self.run_batch()
        args = self.review_args()
        evidence = pathlib.Path(args.evidence)
        (evidence / "sirius.yaml").write_text("different settings")
        seal(evidence)
        with self.assertRaisesRegex(ValueError, "changed sirius.yaml"):
            cmd_review(args)

    def test_runtime_and_cpu_reference_must_match_manual_evidence(self):
        self.run_batch()
        args = self.review_args()
        evidence = pathlib.Path(args.evidence)
        runtime = read(evidence / "runtime.json")
        write_json(
            evidence / "runtime.json",
            {**runtime, "duckdb_binary": {"sha256": "different"}},
        )
        seal(evidence)
        with self.assertRaisesRegex(ValueError, "DuckDB runtime"):
            cmd_review(args)
        write_json(evidence / "runtime.json", runtime)
        outcome = read(evidence / "outcome.json")
        outcome["record"]["evidence"]["operations"][0][
            "fingerprint_multiset"
        ] = "changed"
        write_json(evidence / "outcome.json", outcome)
        seal(evidence)
        with self.assertRaisesRegex(ValueError, "CPU reference"):
            cmd_review(args)

    def test_duplicate_review_cycles_are_rejected(self):
        self.run_batch()
        (self.source / "query.sql").write_text("SELECT 2")
        seal(self.source)
        other = import_candidate(self.root, self.source)
        with patch("siriusfuzz.triage.attempt", side_effect=self.fake_attempt):
            process_candidate(other, self.args, self.options, time.monotonic() + 60)
        args = self.review_args()
        args.disposition = "duplicate"
        args.duplicate_of = other.name
        cmd_review(args)
        args.candidate, args.duplicate_of = other.name, self.identifier
        with self.assertRaisesRegex(ValueError, "cycle"):
            cmd_review(args)

    def test_grouping_does_not_verify_candidates(self):
        self.run_batch()
        cmd_group(
            argparse.Namespace(
                workspace=str(self.root),
                candidates=[self.identifier],
                name="possible scalar defects",
                reviewer="Synthetic test",
                notes="hypothesis",
            )
        )
        self.assertFalse((self.directory / "review.json").exists())
        with self.assertRaisesRegex(ValueError, "manual verification"):
            cmd_issue_draft(self.draft_args())


if __name__ == "__main__":
    unittest.main()
