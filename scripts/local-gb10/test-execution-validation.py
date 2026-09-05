"""CPU-only retry-placement regressions; optionally check existing SF500 evidence."""

import ast
import copy
import datetime as dt
import json
from pathlib import Path
import tempfile
import unittest

from execution_validation import load_execution_evidence, validate_execution

ROOT = Path(__file__).resolve().parents[2]
BASE = dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc)
Q0 = "00000000-0000-0000-0000-000000000001"
Q1 = "00000000-0000-0000-0000-000000000002"
Q2 = "00000000-0000-0000-0000-000000000003"
Q3 = "00000000-0000-0000-0000-000000000004"


def iso(second):
    return (BASE + dt.timedelta(seconds=second)).isoformat().replace("+00:00", "Z")


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cluster = Path(self.temporary.name)
        for name in ("fe.log", "cn0.log", "cn1.log"):
            (self.cluster / name).write_text("")

    def append(self, name, line):
        with (self.cluster / name).open("a") as output:
            output.write(line + "\n")

    def fragment(self, node, query, second, state="started", role="sender"):
        self.append(
            f"{node}.log",
            f'{iso(second)} INFO fragment run {state} query_id={query} fragment_instance_id={query} role="{role}"',
        )

    def retry(self, old, new, second, blacklist=None):
        if blacklist:
            self.append(
                "fe.log",
                f"{iso(second - 0.01).replace('T', ' ')} WARN [HostBlacklist.add():92] add black list: {blacklist}",
            )
        self.append(
            "fe.log",
            f"{iso(second).replace('T', ' ')} INFO transfer QueryId: {old} to {new}",
        )

    def original(self):
        self.fragment("cn0", Q0, 12)
        self.fragment("cn1", Q0, 12)

    def completed(self, query, nodes, second=40):
        for node in nodes:
            self.fragment(node, query, second)
            self.fragment(node, query, second + 1, "finished")
        self.fragment(nodes[-1], query, second + 2, "finished", "result")

    def audit(self):
        return validate_execution(self.cluster, iso(10), iso(90))

    def test_single_cn_plan_without_retry_is_valid(self):
        self.completed(Q0, ["cn1"])
        result = self.audit()
        self.assertEqual(result["status"], "VALID")
        self.assertFalse(result["detected_retry"])

    def test_distributed_retry_to_one_cn_is_ineligible(self):
        self.original()
        self.retry(Q0, Q1, 30, 10001)
        self.completed(Q1, ["cn1"])
        result = self.audit()
        self.assertEqual(result["failure_class"], "DEGRADED_RETRY_TOPOLOGY")
        self.assertIs(result["eligible_for_two_cn"], False)
        self.assertEqual(result["final_nodes"], ["cn1"])
        self.assertEqual(result["retries"][0]["active_blacklist"][0]["node_id"], 10001)

    def test_multihop_retry_uses_terminal_uuid(self):
        self.original()
        self.retry(Q0, Q1, 20)
        self.fragment("cn0", Q1, 22)
        self.fragment("cn1", Q1, 22)
        self.retry(Q1, Q2, 30, 10002)
        self.completed(Q2, ["cn0"])
        result = self.audit()
        self.assertEqual(result["status"], "INELIGIBLE")
        self.assertEqual(result["chains"][0]["query_ids"], [Q0, Q1, Q2])
        self.assertEqual(result["final_query_ids"], [Q2])

    def test_retry_still_using_two_cns_is_valid(self):
        self.original()
        self.retry(Q0, Q1, 30)
        self.completed(Q1, ["cn0", "cn1"])
        self.assertEqual(self.audit()["status"], "VALID")

    def test_legitimate_one_cn_retry_is_not_degradation(self):
        self.fragment("cn1", Q0, 12)
        self.retry(Q0, Q1, 30, 10001)
        self.completed(Q1, ["cn1"])
        self.assertEqual(self.audit()["status"], "VALID")

    def test_retry_without_completed_result_is_unknown(self):
        self.original()
        self.retry(Q0, Q1, 30)
        self.fragment("cn1", Q1, 40)
        result = self.audit()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertTrue(result["detected_retry"])

    def test_missing_cn_log_does_not_prove_single_cn_retry(self):
        self.original()
        self.retry(Q0, Q1, 30)
        self.completed(Q1, ["cn1"])
        (self.cluster / "cn0.log").unlink()
        result = self.audit()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertIsNone(result["eligible_for_two_cn"])
        self.assertTrue(result["missing_logs"])

    def test_admin_retry_outside_sql_window_is_ignored(self):
        self.retry(Q1, Q2, 5, 10001)
        self.retry(Q2, Q3, 95)
        self.append(
            "fe.log", f"{iso(4).replace('T', ' ')} INFO transfer QueryId: malformed"
        )
        self.completed(Q0, ["cn0"])
        result = self.audit()
        self.assertEqual(result["status"], "VALID")
        self.assertFalse(result["detected_retry"])
        self.assertFalse(result["retries"])

    def test_unrelated_execution_in_retry_window_is_unknown(self):
        self.original()
        self.retry(Q0, Q1, 30)
        self.completed(Q1, ["cn1"])
        self.fragment("cn0", Q2, 50)
        self.assertEqual(self.audit()["status"], "UNKNOWN")

    def test_cycle_is_unknown(self):
        self.retry(Q0, Q1, 20)
        self.retry(Q1, Q0, 30)
        self.assertEqual(self.audit()["issue"], "Cyclic FE retry chain")

    def test_missing_fe_log_and_invalid_interval_are_explicit(self):
        (self.cluster / "fe.log").unlink()
        self.completed(Q0, ["cn0"])
        result = self.audit()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["detected_retry"])
        self.assertEqual(
            validate_execution(self.cluster, None, iso(90))["status"], "UNKNOWN"
        )

    def test_cached_evidence_is_not_mutated(self):
        self.original()
        self.retry(Q0, Q1, 30)
        self.completed(Q1, ["cn1"])
        evidence = load_execution_evidence(self.cluster)
        before = copy.deepcopy(evidence)
        validate_execution(self.cluster, iso(10), iso(90), evidence=evidence)
        self.assertEqual(evidence, before)


class HarnessIntegrationTests(unittest.TestCase):
    def test_ineligible_or_unresolved_retry_stops_warms_but_preserves_raw_pass(self):
        tree = ast.parse(
            (ROOT / "scripts/local-gb10/benchmark-multi-cn.py").read_text()
        )
        loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "repetition"
        )
        start = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "execution"
                for target in node.targets
            )
        )
        stop = next(
            i
            for i, node in enumerate(loop.body[start:], start)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "benchmark_failure_class"
                for target in node.targets
            )
        )
        failed = next(
            node
            for node in reversed(loop.body)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "failed"
                for target in node.targets
            )
        )
        module = ast.fix_missing_locations(
            ast.Module(body=loop.body[start : stop + 1] + [failed], type_ignores=[])
        )
        for status, retry, excluded in (
            ("INELIGIBLE", True, True),
            ("UNKNOWN", True, True),
            ("UNKNOWN", False, False),
            ("VALID", False, False),
        ):
            with self.subTest(status=status, retry=retry):
                sql = {
                    "status": "OK",
                    "elapsed_seconds": 85.1,
                    "started_utc": iso(10),
                    "finished_utc": iso(90),
                }
                row = {"status": "PASS", "starrocks": dict(sql)}
                validation = {
                    "status": status,
                    "detected_retry": retry,
                    "failure_class": "EXAMPLE",
                }
                scope = {
                    "run": row,
                    "cluster": Path("/unused"),
                    "validate_execution": lambda *a: validation,
                }
                exec(compile(module, "harness-fragment", "exec"), scope)
                self.assertEqual(row["status"], "PASS")
                self.assertEqual(row["starrocks"], sql)
                self.assertEqual(scope["failed"], excluded)
                self.assertEqual(row["benchmark_eligible"], not excluded)


class ExistingEvidenceTests(unittest.TestCase):
    def test_q09_and_q21_primary_metadata(self):
        root = ROOT / "results/multi-cn-throughput-ab/optimized"
        if not (root / "q09/manifest.json").is_file():
            self.skipTest("Optional immutable primary benchmark evidence is absent")
        for query in ("q09", "q21"):
            block = root / query
            manifest = json.loads((block / "manifest.json").read_text())
            evidence = load_execution_evidence(block / "cluster-001")
            for row in manifest["runs"]:
                sql = row.get("starrocks")
                if not sql:
                    continue
                with self.subTest(query=query, repetition=row["repetition"]):
                    result = validate_execution(
                        block / "cluster-001",
                        sql["started_utc"],
                        sql["finished_utc"],
                        evidence=evidence,
                    )
                    if query == "q09":
                        self.assertEqual(result["status"], "INELIGIBLE")
                        self.assertEqual(
                            result["final_nodes"],
                            ["cn0" if row["repetition"] == 1 else "cn1"],
                        )
                        self.assertTrue(row["comparison"]["match"])
                        self.assertEqual(row["status"], "PASS")
                    else:
                        self.assertEqual(result["status"], "UNKNOWN")
                        self.assertTrue(result["detected_retry"])
                        self.assertEqual(row["status"], "STARROCKS_ERROR")


if __name__ == "__main__":
    unittest.main(verbosity=2)
