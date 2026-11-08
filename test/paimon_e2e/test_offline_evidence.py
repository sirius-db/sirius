"""Synthetic causal-model controls through the real offline checker CLI."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CHECKER = Path(__file__).with_name("check_offline_evidence.py")


def fixture():
    scope = {
        "run": "r1",
        "runtime": "rt1",
        "process": 123,
        "statement": "s1",
        "sql": "a" * 64,
        "oracle": "b" * 64,
        "inputs": "c" * 64,
        "artifacts": "d" * 64,
    }
    binding = {
        "contract": 8,
        "occurrence": 0,
        "materialization_generation": 9,
        "read_owner": "owned-read-1",
        "producer": "native-owner-1",
        "certified_view": "view-1",
        "scan_operator": 10,
        "pipeline": "pipe-1",
    }
    expected = {
        "scope": scope,
        "window": 7,
        "binding": binding,
        "result": {
            "columns": [["id", "BIGINT"], ["amount", "DECIMAL(12,2)"]],
            "rows": [[1, "10.00"], [1, "10.00"], [2, None], [3, "5.00"]],
        },
    }
    observed = {
        "scope": copy.deepcopy(scope),
        "query": {
            "uuid": "query-1",
            "label": "s1",
            "window": 7,
            "finalize_generation": 0,
            "plan": "plan-1",
        },
        "mapping": {"window": 7, "uuid": "query-1"},
        "runtime": {
            "scope": "whole_runtime",
            "before_active": [],
            "during_windows": [7],
            "after_active": [],
            "other_work": [],
            "drained": True,
        },
        "settings": {
            "gpu_execution": True,
            "enable_duckdb_fallback": False,
            "plan_fallbacks": 0,
            "runtime_fallbacks": 0,
        },
        "sources": [{"query": "query-1", "window": 7, **copy.deepcopy(binding)}],
        "native_origin": {
            "query": "query-1",
            "window": 7,
            "certified_view": "view-1",
            "factory_view": "view-1",
            "sdk_read_view": "view-1",
            "read_owner": "owned-read-1",
            "producer": "native-owner-1",
            "batch_owners": ["batch-1"],
        },
        "validations": [
            {
                "query": "query-1",
                "window": 7,
                "contract": 8,
                "occurrence": 0,
                "generation": 9,
                "unit": 0,
                "read_owner": "owned-read-1",
                "producer": "native-owner-1",
                "batch_owner": "batch-1",
                "rootless": False,
                "rows": 4,
            }
        ],
        "imports": {
            "before": {"batches": 0, "rows": 0},
            "after": {"batches": 1, "rows": 4},
        },
        "scan_tasks": [
            {
                "query": "query-1",
                "plan": "plan-1",
                "pipeline": "pipe-1",
                "operator": 10,
                "task": "task-1",
                "gpu": 0,
                "computing": 1,
                "finalizing": 2,
                "success": True,
            }
        ],
        "transport": {
            "closed": True,
            "dropped": 0,
            "truncated": False,
            "ambiguous": False,
        },
        "process": {"termination": "exited", "exit": 0, "timeout": False},
        "result": {
            "query": "query-1",
            "complete": True,
            "error": None,
            "columns": [["id", "BIGINT"], ["amount", "DECIMAL(12,2)"]],
            "rows": [
                {"id": 1, "amount": "10.00"},
                {"id": 1, "amount": "10.00"},
                {"id": 2, "amount": None},
                {"id": 3, "amount": "5.00"},
            ],
        },
        "cleanup": {
            "query": "query-1",
            "status": "ok",
            "drained": True,
            "probe": [{"probe": "LIVENESS_OK"}],
        },
    }
    return expected, observed


class OfflineEvidenceTests(unittest.TestCase):
    def check(self, change=None, *, accepted=False, raw=None):
        expected, observed = fixture()
        if change:
            change(expected, observed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "expected.json").write_text(json.dumps(expected))
            (root / "trace.json").write_text(
                json.dumps(observed) if raw is None else raw
            )
            output = root / "decision.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--expected",
                    str(root / "expected.json"),
                    "--trace",
                    str(root / "trace.json"),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(
                result.returncode, 0 if accepted else 1, result.stdout + result.stderr
            )
            report = json.loads(output.read_text())
            self.assertEqual(
                report["decision"],
                "CONSISTENT_OFFLINE_MODEL" if accepted else "REJECTED",
            )
            self.assertIs(report["actual_gpu_qualified"], False)
            self.assertEqual(report["scope"], "synthetic normalized causal model only")
            return report

    def test_consistent_model_is_never_actual_gpu_qualification(self):
        self.check(accepted=True)

    def test_cpu_correct_rows_are_rejected(self):
        self.check(lambda e, o: o["settings"].update(gpu_execution=False))

    def test_missing_mandatory_stage_cannot_be_replaced_by_delivery_flag(self):
        for key in (
            "query",
            "mapping",
            "runtime",
            "sources",
            "native_origin",
            "validations",
            "imports",
            "scan_tasks",
            "result",
            "cleanup",
        ):
            with self.subTest(stage=key):
                self.check(lambda e, o: o.pop(key))

    def test_wrong_run_query_nonce_window_and_artifact_are_rejected(self):
        changes = [
            lambda e, o: o["scope"].update(runtime="other-runtime"),
            lambda e, o: o["scope"].update(statement="old-prepared-execution"),
            lambda e, o: o["scope"].update(artifacts="e" * 64),
            lambda e, o: o["query"].update(label="old-nonce"),
            lambda e, o: o["mapping"].update(uuid="another-query"),
            lambda e, o: o["sources"][0].update(window=99),
            lambda e, o: o["scan_tasks"][0].update(query="another-query"),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.check(change)

    def test_generic_stamp_or_claimed_source_is_not_native_origin(self):
        self.check(
            lambda e, o: o.update(native_origin={"source": "paimon", "isolated": True})
        )
        for key in ("factory_view", "sdk_read_view", "read_owner", "producer"):
            with self.subTest(key=key):
                self.check(
                    lambda e, o: o["native_origin"].update({key: "different-origin"})
                )

    def test_another_arrow_input_or_unit_contract_is_rejected(self):
        for key, value in (
            ("contract", 12),
            ("producer", "other-arrow-source"),
            ("read_owner", "other-owner"),
            ("batch_owner", "stamped-only"),
            ("generation", 0),
        ):
            with self.subTest(key=key):
                self.check(lambda e, o: o["validations"][0].update({key: value}))

    def test_repeated_validation_deduplicates_but_contradictory_unit_rejects(self):
        self.check(
            lambda e, o: o["validations"].append(copy.deepcopy(o["validations"][0])),
            accepted=True,
        )

        def contradictory(e, o):
            other = copy.deepcopy(o["validations"][0])
            other["rows"] = 3
            o["validations"].append(other)

        self.check(contradictory)

    def test_eof_and_validation_without_successful_import_are_rejected(self):
        self.check(lambda e, o: o["validations"][0].update(rootless=True, rows=0))
        self.check(lambda e, o: o["imports"]["after"].update(batches=0, rows=0))
        self.check(lambda e, o: o["imports"]["before"].update(batches=2, rows=9))

    def test_single_connection_is_not_whole_runtime_isolation(self):
        self.check(lambda e, o: o["runtime"].update(scope="connection"))
        self.check(lambda e, o: o["runtime"].update(during_windows=[7, 8]))
        self.check(lambda e, o: o["runtime"].update(other_work=["another-connection"]))

    def test_plan_deployment_and_query_finished_cannot_replace_scan_completion(self):
        self.check(lambda e, o: o.update(scan_tasks=[]))
        for key in ("computing", "finalizing"):
            with self.subTest(key=key):
                self.check(lambda e, o: o["scan_tasks"][0].pop(key))
        self.check(lambda e, o: o["scan_tasks"][0].update(success=False))
        self.check(lambda e, o: o["scan_tasks"][0].update(finalizing=1))
        self.check(lambda e, o: o["scan_tasks"][0].update(operator=11))
        self.check(lambda e, o: o["scan_tasks"][0].update(pipeline="other-pipeline"))

    def test_execution_finalize_zero_is_valid_but_arrow_generation_is_distinct(self):
        self.check(accepted=True)
        self.check(lambda e, o: o["query"].update(finalize_generation=9))

    def test_multiscan_cannot_pass_using_one_source_receipt(self):
        self.check(lambda e, o: o["sources"].append(copy.deepcopy(o["sources"][0])))

    def test_late_error_partial_results_cleanup_and_liveness_fail(self):
        changes = [
            lambda e, o: o["result"].update(complete=False),
            lambda e, o: o["result"].update(error="late-import-error"),
            lambda e, o: o["result"]["rows"].pop(),
            lambda e, o: o["result"]["columns"][0].__setitem__(1, "VARCHAR"),
            lambda e, o: o["cleanup"].update(status="error"),
            lambda e, o: o["cleanup"].update(drained=False),
            lambda e, o: o["cleanup"].update(probe=[]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.check(change)

    def test_transport_loss_or_process_failure_overrides_successful_stages(self):
        for key, value in (
            ("closed", False),
            ("dropped", 1),
            ("truncated", True),
            ("ambiguous", True),
        ):
            with self.subTest(key=key):
                self.check(lambda e, o: o["transport"].update({key: value}))
        self.check(lambda e, o: o["process"].update(exit=7))
        self.check(lambda e, o: o["process"].update(termination="signal", exit=-15))
        self.check(
            lambda e, o: o["process"].update(termination="timeout", timeout=True)
        )

    def test_bool_cannot_impersonate_integer_and_invalid_json_rejects(self):
        self.check(lambda e, o: o["process"].update(exit=False))
        self.check(lambda e, o: o["imports"]["after"].update(batches=True))
        for raw in ('{"scope":', '{"scope":{},"scope":{}}', '{"x":NaN}'):
            with self.subTest(raw=raw):
                self.check(raw=raw)

    def test_bad_oracle_values_reject_without_disclosing_row_data(self):
        for value in (True, "credential-sentinel", "NaN", "Infinity"):
            with self.subTest(value=value):
                report = self.check(
                    lambda e, o: o["result"]["rows"][0].update(amount=value)
                )
                self.assertEqual(report["reason"], "result_oracle")
                self.assertNotIn("credential-sentinel", json.dumps(report))

    def test_output_cannot_replace_inputs_or_previous_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected, observed = fixture()
            source = root / "expected.json"
            source.write_text(json.dumps(expected))
            trace = root / "trace.json"
            trace.write_text(json.dumps(observed))
            existing = root / "previous.json"
            existing.write_text("preserved failed attempt")
            for output in (source, trace, existing):
                before = output.read_bytes()
                result = subprocess.run(
                    [
                        sys.executable,
                        str(CHECKER),
                        "--expected",
                        str(source),
                        "--trace",
                        str(trace),
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 2)
                self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
