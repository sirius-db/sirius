"""Exercise the real runner entry point with a deliberately faulty CLI double.

This tests harness decisions, not Paimon or native DuckDB semantics.
"""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_conformance
from corpus_checks import canonical_json, digest
from run_conformance import main

HERE = Path(__file__).resolve().parent


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="paimon reader's; ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.corpus = self.root / "corpus"
        shutil.copytree(HERE / "warehouse", self.corpus / "warehouse")
        shutil.copy2(HERE / "manifest.json", self.corpus / "manifest.json")
        self.extension = self.root / "paimon.duckdb_extension"
        self.extension.write_bytes(b"fake extension for harness testing only")
        self.spec = json.loads((HERE / "expectations.json").read_text())
        self.spec["extension"] = json.loads(
            (HERE / "qualified-artifacts.json").read_text()
        )["artifacts"][0]
        self.spec["extension"]["sha256"] = digest(self.extension)
        self.registry = self.root / "qualified-artifacts.json"
        self.registry.write_text(
            canonical_json({"format_version": 1, "artifacts": [self.spec["extension"]]})
        )
        self.original = copy.deepcopy(self.spec)
        self.trace = self.root / "trace.jsonl"
        self.cli = self.root / "fake-duckdb"
        self.cli.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            f"spec = json.loads({json.dumps(self.original)!r})\n"
            f"manifest = json.loads({(self.corpus / 'manifest.json').read_text()!r})\n"
            f"corpus = {str(self.corpus)!r}\n"
            f"trace = {str(self.trace)!r}\n"
            + """
sql = sys.argv[sys.argv.index('-c') + 1]
mode = 'disabled' if os.environ.get('SIRIUS_DISABLE') == '1' else ('transparent' if 'SET gpu_execution=true;' in sql else 'cpu')
with open(trace, 'a') as f: f.write(json.dumps({'sql':sql,'mode':mode,'argv':sys.argv[1:]})+'\\n')
fault = os.environ.get('PAIMON_TEST_FAULT', '')
if sql == 'SELECT version() AS version; PRAGMA platform;':
    print(json.dumps([{'version':os.environ.get('PAIMON_TEST_VERSION', spec['extension']['duckdb_version'])}]))
    print(json.dumps([{'platform':spec['extension']['platform']}]))
    sys.exit(0)

sirius_path = os.environ.get('PAIMON_TEST_SIRIUS', '')
if sirius_path:
    load = "LOAD '" + sirius_path.replace("'", "''") + "';"
    if load not in sql or sql.index(load) > sql.index('SET gpu_execution='):
        print('Sirius must load before its settings are used', file=sys.stderr)
        sys.exit(1)
    if fault == 'sirius_load_failure':
        print('Sirius extension failed to load', file=sys.stderr)
        sys.exit(1)
if os.environ.get('PAIMON_TEST_REQUIRE_UNSIGNED') == '1' and '-unsigned' not in sys.argv:
    print('Unsigned local extension requires -unsigned', file=sys.stderr)
    sys.exit(1)

print('Engine diagnostic outside the SQL result channel', flush=True)
if '-cmd' in sys.argv:
    assert sys.argv[sys.argv.index('-cmd') + 1] == '.output sql-results.json'
    sys.stdout = open('sql-results.json', 'w')

def fixture_query(candidate):
    table = manifest['tables'][candidate['table']]
    snapshot = table['snapshots'][candidate['snapshot']] if candidate['snapshot'] else None
    scan = "paimon_scan('" + (corpus+'/'+table['path']).replace("'", "''") + "'"
    if snapshot is not None: scan += ', snapshot_from_id='+str(snapshot)
    return candidate['sql'].format(scan=scan+')')

# This double recognizes our finite generated protocol, not arbitrary SQL.
# Compare the whole tail, not a DESCRIBE substring. In particular, do not split
# on semicolons: quoted filesystem paths can contain them.
_, separator, tail = sql.partition("SELECT current_setting('gpu_execution') AS gpu;")
if not separator: raise RuntimeError('Unexpected test SQL: missing setup boundary')
if 'SET enable_duckdb_fallback=false;' in sql:
    candidate = next(c for c in spec['cases'] if c['id']=='append_b')
    if tail != 'SET enable_duckdb_fallback=false;'+fixture_query(candidate):
        raise RuntimeError('Unexpected test SQL: rejection requires the actual Paimon query')
    message = "GPU plan generation failed: Table function 'paimon_scan' is not supported in Sirius"
    if fault == 'rejection_unrelated': message = 'Binder Error: missing table'
    print(message, file=sys.stderr, flush=True)
    if fault == 'rejection_timeout':
        import time
        time.sleep(60)
    if fault == 'rejection_success': sys.exit(0)
    if fault == 'rejection_wrong_exit': sys.exit(2)
    if fault == 'rejection_crash':
        import resource, signal
        resource.setrlimit(resource.RLIMIT_CORE, (0,0))
        os.kill(os.getpid(), signal.SIGSEGV)
    sys.exit(1)
if fault == 'mutate_warehouse':
    with open(corpus+'/warehouse/reader-added-file', 'w') as f: f.write('mutation')
attached = 'ATTACH ' in sql
case = None
for candidate in spec['cases']:
    prefix = ''
    query = fixture_query(candidate)
    if attached:
        if candidate['id'] != 'pk_d': continue
        prefix = "ATTACH '"+(corpus+'/warehouse').replace("'", "''")+"' AS p (TYPE paimon, READ_ONLY);"
        query = 'SELECT id, amount FROM p.reference.orders_pk'
    expected = prefix+'DESCRIBE '+query+';'+query+"; SELECT 'LIVENESS_OK' AS probe;"
    if tail == expected:
        case=candidate
        break
if case is None: raise RuntimeError('Unexpected test SQL: '+sql)
columns=[{'column_name':n,'column_type':t} for n,t in case['columns']]
rows=[dict(zip([n for n,t in case['columns']], row)) for row in case['rows']]
results=[[{'extension_version':spec['extension']['version']}],[{'version':spec['extension']['duckdb_version']}],[{'gpu':mode=='transparent'}],columns,rows,[{'probe':'LIVENESS_OK'}]]
target = os.environ.get('PAIMON_TEST_TARGET', 'disabled/empty_rows')
label = 'attached' if attached else mode+'/'+case['id']
if label == target:
    if fault == 'query_crash':
        import resource, signal
        resource.setrlimit(resource.RLIMIT_CORE, (0,0))
        os.kill(os.getpid(), signal.SIGSEGV)
    if fault == 'missing_file': os.unlink('sql-results.json')
    elif fault == 'empty_file': results = []
    elif fault == 'missing': results.pop(4)
    elif fault == 'extra': results.insert(4, [])
    elif fault == 'reordered': results[3],results[4]=results[4],results[3]
    elif fault == 'wrong_type': results[3][1]['column_type']='BIGINT'
    elif fault == 'wrong_scale': results[3][1]['column_type']='DECIMAL(12,3)'
    elif fault == 'bad_describe': results[3]=[None]
    elif fault == 'mode': results[2]=[{'gpu':0}]
    elif fault == 'identity': results[0]=[{'extension_version':'wrong'}]
    elif fault == 'liveness': results[-1]=[]
    elif fault == 'bool_decimal': results[4][0]['amount']=True
for result in results: print(json.dumps(result))
"""
        )
        self.cli.chmod(0o755)

    def run_suite(
        self,
        selected=(),
        fault="",
        target="disabled/empty_rows",
        source=False,
        sirius=None,
        default_sirius=False,
        sql_mutation=None,
        timeout=90,
    ):
        (self.corpus / "expectations.json").write_text(canonical_json(self.spec))
        args = [
            str(self.corpus),
            "--duckdb",
            str(self.cli),
            "--paimon-extension",
            str(self.extension),
            "--registry",
            str(self.registry),
            "--output",
            str(self.root / "reports"),
            "--timeout",
            str(timeout),
        ]
        if default_sirius:
            sirius = (
                self.root
                / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
            )
        elif sirius is None:
            args.append("--builtin-sirius")
        else:
            args.extend(["--sirius-extension", str(sirius)])
        if source:
            index = args.index("--registry")
            del args[index : index + 2]
        for case in selected:
            args.extend(["--case", case])
        error = None
        execute = run_conformance.execute
        mutated = []

        def execute_with_mutation(command, **kwargs):
            if sql_mutation and kwargs["log"].name == sql_mutation[0]:
                command = list(command)
                index = command.index("-c") + 1
                original = command[index]
                command[index] = sql_mutation[1](original)
                self.assertNotEqual(command[index], original)
                mutated.append(command[index])
            return execute(command, **kwargs)

        with patch.dict(
            os.environ,
            PAIMON_TEST_FAULT=fault,
            PAIMON_TEST_TARGET=target,
            PAIMON_TEST_REQUIRE_UNSIGNED="1" if source or sirius else "0",
            PAIMON_TEST_SIRIUS=str(sirius) if sirius else "",
        ), patch.object(
            run_conformance, "execute", execute_with_mutation
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            try:
                main(args)
            except Exception as caught:
                error = caught
        if sql_mutation:
            self.assertEqual(len(mutated), 1, "Mutation must reach one real CLI call")
        report = json.loads(
            next((self.root / "reports").glob("run-*/report.json")).read_text()
        )
        return report, error

    def test_source_receipt_default_rejects_missing_evidence(self):
        report, error = self.run_suite(selected=("append_b",), source=True)
        self.assertIsInstance(error, FileNotFoundError)
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["executed_cases"], 0)
        self.assertEqual(report["not_run_cases"], report["planned_cases"])

    def test_source_receipt_acceptance_and_tampering_at_real_entry_point(self):
        import build_extension as builder

        recipe = builder.read_recipe()
        self.cli.write_text(
            self.cli.read_text()
            .replace("v1.5.5", recipe["duckdb_version"])
            .replace("5e89198", recipe["extension_version"])
        )
        identity = {"version": recipe["duckdb_version"], "platform": recipe["platform"]}
        receipt = {
            "format_version": 1,
            "state": "built",
            "recipe_sha256": digest(builder.RECIPE),
            "builder_sha256": digest(Path(builder.__file__)),
            "downloader_sha256": digest(builder.DOWNLOADER),
            "pixi_lock_sha256": digest(builder.ROOT / "pixi.lock"),
            "sources": recipe["sources"],
            "cmake_options": recipe["cmake_options"],
            "duckdb_identity": identity,
            "duckdb_sha256": digest(self.cli),
            "duckdb_source_id": recipe["sources"]["duckdb"]["commit"][:8],
            "toolchain": {key: "test tool" for key in ("cc", "cxx", "cmake", "ninja")},
            "build_environment": {key: "" for key in builder.BUILD_ENVIRONMENT_KEYS},
            "sha256": digest(self.extension),
            "size_bytes": self.extension.stat().st_size,
        }
        path = self.root / "build-receipt.json"
        path.write_text(json.dumps(receipt))
        report, error = self.run_suite(selected=("append_b",), source=True)
        self.assertIsNone(error)
        self.assertEqual(report["state"], "passed")
        self.assertEqual(report["executed_cases"], 5)
        self.assertEqual(report["extension"]["kind"], "source_build")
        calls = [json.loads(line) for line in self.trace.read_text().splitlines()]
        self.assertNotIn("-unsigned", calls[0]["argv"])
        self.assertEqual(len(calls), 6)
        for call in calls[1:]:
            self.assertIn("-unsigned", call["argv"])
        shutil.rmtree(self.root / "reports")
        self.extension.write_bytes(b"corrupted after build")
        report, error = self.run_suite(selected=("append_b",), source=True)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["executed_cases"], 0)
        self.assertEqual(report["not_run_cases"], 5)

    def test_historical_registry_cannot_enable_unsigned_loading(self):
        record = self.spec["extension"] | {"kind": "source_build"}
        self.registry.write_text(
            canonical_json({"format_version": 1, "artifacts": [record]})
        )
        report, error = self.run_suite(selected=("append_b",))
        self.assertIsNone(error)
        self.assertEqual(report["state"], "passed")
        for line in self.trace.read_text().splitlines():
            self.assertNotIn("-unsigned", json.loads(line)["argv"])

    def test_complete_suite_one_process_per_case(self):
        report, error = self.run_suite()
        self.assertIsNone(error)
        self.assertEqual(report["state"], "passed")
        self.assertEqual(
            (
                report["planned_cases"],
                report["executed_cases"],
                report["failed_cases"],
                report["not_run_cases"],
            ),
            (53, 53, 0, 0),
        )
        self.assertEqual(len(self.trace.read_text().splitlines()), 54)
        for line in self.trace.read_text().splitlines():
            self.assertNotIn("-unsigned", json.loads(line)["argv"])
        attached = [
            json.loads(line)["sql"]
            for line in self.trace.read_text().splitlines()
            if "ATTACH " in line
        ][0]
        self.assertLess(attached.index("ATTACH "), attached.index("DESCRIBE "))
        run_dir = next((self.root / "reports").glob("run-*"))
        self.assertIn("Engine diagnostic", (run_dir / "transparent.stdout").read_text())
        self.assertNotIn(
            "Engine diagnostic", (run_dir / "transparent.results.json").read_text()
        )

    def assert_sql_mutation_fails(self, label, transform, case_id, mode):
        shutil.rmtree(self.root / "reports", ignore_errors=True)
        report, error = self.run_suite(["append_b"], sql_mutation=(label, transform))
        self.assertIsNotNone(error)
        self.assertEqual(report["state"], "failed")
        self.assertEqual(
            (report["executed_cases"], report["failed_cases"], report["not_run_cases"]),
            (5, 1, 0),
        )
        failures = [case for case in report["cases"] if case["state"] == "FAIL"]
        self.assertEqual(
            [(case["id"], case["mode"]) for case in failures], [(case_id, mode)]
        )
        self.assertIn("Unexpected test SQL", failures[0]["error"])
        self.assertTrue(
            all(
                case["state"] == "PASS"
                for case in report["cases"]
                if case not in failures
            )
        )

    def test_cli_double_requires_ordered_query_and_liveness_in_every_read_path(self):
        # Change the SQL actually sent to the child, not its fabricated JSON.
        # DESCRIBE alone must never be enough to obtain a successful response.
        for label, case_id, mode in (
            ("disabled-append_b", "append_b", "disabled"),
            ("cpu-append_b", "append_b", "cpu"),
            ("transparent", "rows_and_liveness", "transparent"),
            ("attached", "attached_catalog", "cpu"),
        ):
            for fault in (
                "missing_query",
                "missing_describe",
                "missing_liveness",
                "early_liveness",
                "query_before_describe",
                "duplicate_query",
                "different_query",
                "extra_statement",
            ):
                with self.subTest(label=label, fault=fault):

                    def transform(sql):
                        prefix, tail = sql.split("DESCRIBE ", 1)
                        ending = "; SELECT 'LIVENESS_OK' AS probe;"
                        # The generated tail repeats the exact query twice. Avoid
                        # splitting on semicolons, which can occur in quoted paths.
                        repeated = tail.removesuffix(ending)
                        query = repeated[: (len(repeated) - 1) // 2]
                        self.assertEqual(repeated, query + ";" + query)
                        describe = "DESCRIBE " + query + ";"
                        read = query + ";"
                        live = " SELECT 'LIVENESS_OK' AS probe;"
                        tails = {
                            "missing_query": describe + live,
                            "missing_describe": read + live,
                            "missing_liveness": describe + read,
                            "early_liveness": describe + live + read,
                            "query_before_describe": read + describe + live,
                            "duplicate_query": describe + read + read + live,
                            "different_query": describe + "SELECT 42;" + live,
                            "extra_statement": describe + read + live + "SELECT 42;",
                        }
                        return prefix + tails[fault]

                    self.assert_sql_mutation_fails(label, transform, case_id, mode)

    def test_cli_double_requires_actual_paimon_query_for_rejection(self):
        for fault in (
            "missing_query",
            "different_query",
            "wrong_table",
            "wrong_snapshot",
            "query_before_setting",
            "extra_statement",
        ):
            with self.subTest(fault=fault):

                def transform(sql):
                    setting = "SET enable_duckdb_fallback=false;"
                    prefix, query = sql.split(setting, 1)
                    tails = {
                        "missing_query": setting,
                        "different_query": setting + "SELECT 42",
                        "wrong_table": setting
                        + query.replace("append_basic", "empty_table"),
                        "wrong_snapshot": setting
                        + query.replace("snapshot_from_id=2", "snapshot_from_id=1"),
                        "query_before_setting": query + ";" + setting,
                        "extra_statement": setting + query + ";SELECT 42;",
                    }
                    return prefix + tails[fault]

                self.assert_sql_mutation_fails(
                    "rejection", transform, "planning_rejection", "transparent"
                )

    def test_incomplete_and_bad_protocol_fail_but_later_cases_run(self):
        for fault in (
            "missing_file",
            "empty_file",
            "missing",
            "extra",
            "reordered",
            "bad_describe",
            "mode",
            "identity",
            "liveness",
        ):
            with self.subTest(fault=fault):
                shutil.rmtree(self.root / "reports", ignore_errors=True)
                report, error = self.run_suite(["empty_rows", "append_b"], fault)
                self.assertIsNotNone(error)
                self.assertEqual(report["state"], "failed")
                self.assertEqual(report["failed_cases"], 1)
                self.assertEqual(report["not_run_cases"], 0)
                self.assertTrue(report["cases"][-1]["passed"])

    def test_both_smoke_paths_require_types(self):
        for target in ("attached", "transparent/append_b"):
            for fault in ("wrong_type", "wrong_scale", "bool_decimal"):
                with self.subTest(target=target, fault=fault):
                    shutil.rmtree(self.root / "reports", ignore_errors=True)
                    report, error = self.run_suite(["empty_rows"], fault, target)
                    self.assertIsNotNone(error)
                    self.assertEqual(report["failed_cases"], 1)

    def test_case_local_construction_and_decimal_errors_are_recorded(self):
        for mutation in ("snapshot", "decimal"):
            with self.subTest(mutation=mutation):
                shutil.rmtree(self.root / "reports", ignore_errors=True)
                self.spec = copy.deepcopy(self.original)
                case = next(
                    c for c in self.spec["cases"] if c["id"] == "append_b_totals"
                )
                if mutation == "snapshot":
                    case["snapshot"] = "missing-label"
                else:
                    case["rows"][0][-1] = "1O.00"
                report, error = self.run_suite(["append_b_totals", "empty_rows"])
                self.assertIsNotNone(error)
                failures = [c for c in report["cases"] if c["state"] == "FAIL"]
                self.assertEqual(len(failures), 2)
                self.assertTrue(all(c["id"] == "append_b_totals" for c in failures))
                self.assertTrue(
                    all(
                        c["error_type"]
                        == (
                            "KeyError" if mutation == "snapshot" else "InvalidOperation"
                        )
                        for c in failures
                    )
                )
                self.assertEqual(report["not_run_cases"], 0)

    def test_setup_failure_has_report_and_does_not_start_reader(self):
        (self.corpus / "warehouse/extra").write_text("extra")
        report, error = self.run_suite(["empty_rows"])
        self.assertIsNotNone(error)
        self.assertEqual(report["state"], "failed")
        self.assertEqual((report["executed_cases"], report["not_run_cases"]), (0, 5))
        self.assertFalse(self.trace.exists())

    def test_oracle_failure_does_not_start_reader(self):
        self.spec["cases"][0]["rows"][0][1] = "999.00"
        report, error = self.run_suite(["empty_rows"])
        self.assertIsNotNone(error)
        self.assertEqual(report["executed_cases"], 0)
        self.assertFalse(self.trace.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX signals")
    def test_crash_is_not_an_expected_rejection_in_real_runner(self):
        report, error = self.run_suite(["empty_rows"], "rejection_crash")
        self.assertIsNotNone(error)
        failures = [c for c in report["cases"] if c["state"] == "FAIL"]
        self.assertEqual([c["id"] for c in failures], ["planning_rejection"])
        self.assertEqual(failures[0]["error_type"], "ProcessError")
        self.assertEqual(
            failures[0]["process"],
            {
                "termination": "signal",
                "exit_status": -signal.SIGSEGV,
                "timed_out": False,
                "expected_error_match": True,
            },
        )
        self.assert_process_logs(report)

    def assert_process_logs(self, report):
        directory = next((self.root / "reports").glob("run-*"))
        self.assertEqual(
            json.loads((directory / "reader-identity.process.json").read_text()),
            report["reader_process"],
        )
        smoke = {
            "rows_and_liveness": "transparent",
            "planning_rejection": "rejection",
            "attached_catalog": "attached",
        }
        for case in report["cases"]:
            label = smoke.get(case["id"], case["mode"] + "-" + case["id"])
            self.assertEqual(
                json.loads((directory / (label + ".process.json")).read_text()),
                case["process"],
            )

    def test_process_facts_for_success_and_normal_rejection(self):
        report, error = self.run_suite(["append_b"])
        self.assertIsNone(error)
        self.assert_process_logs(report)
        for case in report["cases"]:
            rejection = case["id"] == "planning_rejection"
            self.assertEqual(
                case["process"],
                {
                    "termination": "exited",
                    "exit_status": 1 if rejection else 0,
                    "timed_out": False,
                    "expected_error_match": True if rejection else None,
                },
            )

    def test_wrong_rejection_outcomes_retain_independent_process_facts(self):
        scenarios = [
            ("rejection_success", 0, True, "exited"),
            ("rejection_wrong_exit", 2, True, "exited"),
            ("rejection_unrelated", 1, False, "exited"),
            ("rejection_timeout", None, True, "timeout"),
        ]
        for fault, status, matches, termination in scenarios:
            with self.subTest(fault=fault):
                shutil.rmtree(self.root / "reports", ignore_errors=True)
                report, error = self.run_suite(["append_b"], fault, timeout=2)
                self.assertIsNotNone(error)
                failures = [c for c in report["cases"] if c["state"] == "FAIL"]
                self.assertEqual([c["id"] for c in failures], ["planning_rejection"])
                self.assertEqual(
                    failures[0]["process"],
                    {
                        "termination": termination,
                        "exit_status": status,
                        "timed_out": termination == "timeout",
                        "expected_error_match": matches,
                    },
                )
                self.assertEqual(
                    (report["executed_cases"], report["not_run_cases"]), (5, 0)
                )
                self.assert_process_logs(report)

    @unittest.skipUnless(os.name == "posix", "POSIX signals")
    def test_query_crash_has_no_expected_rejection_attribution(self):
        report, error = self.run_suite(["append_b"], "query_crash", "cpu/append_b")
        self.assertIsNotNone(error)
        failures = [c for c in report["cases"] if c["state"] == "FAIL"]
        self.assertEqual([c["mode"] for c in failures], ["cpu"])
        self.assertEqual(
            failures[0]["process"],
            {
                "termination": "signal",
                "exit_status": -signal.SIGSEGV,
                "timed_out": False,
                "expected_error_match": None,
            },
        )
        self.assert_process_logs(report)

    def test_identity_launch_failure_is_distinct_from_reader_exit(self):
        self.cli.write_text("not an executable file format\n")
        report, error = self.run_suite(["append_b"])
        self.assertIsInstance(error, OSError)
        self.assertEqual((report["executed_cases"], report["not_run_cases"]), (0, 5))
        self.assertEqual(
            report["reader_process"],
            {
                "termination": "launch_error",
                "exit_status": None,
                "timed_out": False,
                "expected_error_match": None,
            },
        )
        directory = next((self.root / "reports").glob("run-*"))
        self.assertEqual(
            json.loads((directory / "reader-identity.process.json").read_text()),
            report["reader_process"],
        )

    def test_unqualified_reader_version_fails_before_load(self):
        with patch.dict(os.environ, PAIMON_TEST_VERSION="v1.5.6"):
            report, error = self.run_suite(["empty_rows"])
        self.assertIn("No qualified Paimon artifact", str(error))
        self.assertEqual(report["not_run_cases"], 5)
        self.assertEqual(len(self.trace.read_text().splitlines()), 1)

    def test_tampered_extension_fails_before_load(self):
        self.extension.write_bytes(b"tampered after registry creation")
        report, error = self.run_suite(["empty_rows"])
        self.assertIn("Unqualified Paimon artifact", str(error))
        self.assertEqual((report["executed_cases"], report["not_run_cases"]), (0, 5))
        self.assertEqual(len(self.trace.read_text().splitlines()), 1)

    def test_reader_side_warehouse_mutation_fails_postflight(self):
        report, error = self.run_suite(["empty_rows"], "mutate_warehouse")
        self.assertIn("Warehouse inventory differs", str(error))
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["executed_cases"], 5)
        self.assertTrue(all(c["passed"] for c in report["cases"]))

    def test_loadable_sirius_all_modes_and_smoke_paths(self):
        sirius = self.root / "Sirius reader's local.duckdb_extension"
        sirius.write_bytes(b"fake Sirius wrapper")
        report, error = self.run_suite(["append_b"], sirius=sirius)
        self.assertIsNone(error)
        self.assertEqual(report["state"], "passed")
        self.assertEqual(report["executed_cases"], 5)
        self.assertEqual(
            report["sirius"],
            {"mode": "loadable", "path": str(sirius), "sha256": digest(sirius)},
        )
        calls = [json.loads(line) for line in self.trace.read_text().splitlines()]
        self.assertNotIn("-unsigned", calls[0]["argv"])
        self.assertEqual(len(calls), 6)
        for call in calls[1:]:
            self.assertIn("-unsigned", call["argv"])
            self.assertIn("LOAD '" + str(sirius).replace("'", "''") + "';", call["sql"])

    def test_default_sirius_uses_wrapper_build(self):
        sirius = (
            self.root
            / "sirius-duckdb/build/release/extension/sirius/sirius.duckdb_extension"
        )
        sirius.parent.mkdir(parents=True)
        sirius.write_bytes(b"default local wrapper")
        with patch("run_conformance.ROOT", self.root):
            report, error = self.run_suite(["append_b"], default_sirius=True)
        self.assertIsNone(error)
        self.assertEqual(report["state"], "passed")
        self.assertEqual(report["sirius"]["path"], str(sirius))

    def test_missing_sirius_fails_setup_without_running_cases(self):
        report, error = self.run_suite(
            ["append_b"], sirius=self.root / "missing-sirius"
        )
        self.assertIsInstance(error, FileNotFoundError)
        self.assertEqual(report["state"], "failed")
        self.assertEqual((report["executed_cases"], report["not_run_cases"]), (0, 5))
        self.assertFalse(self.trace.exists())

    def test_sirius_load_failure_cannot_pass_any_case(self):
        sirius = self.root / "sirius.duckdb_extension"
        sirius.write_bytes(b"unloadable wrapper")
        report, error = self.run_suite(
            ["append_b"], "sirius_load_failure", sirius=sirius
        )
        self.assertIsNotNone(error)
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["failed_cases"], 5)
        self.assertTrue(all(case["state"] == "FAIL" for case in report["cases"]))

    def test_loadable_sirius_does_not_bypass_paimon_qualification(self):
        sirius = self.root / "sirius.duckdb_extension"
        sirius.write_bytes(b"fake wrapper")
        self.extension.write_bytes(b"tampered reader")
        report, error = self.run_suite(["append_b"], sirius=sirius)
        self.assertIn("Unqualified Paimon artifact", str(error))
        self.assertEqual((report["executed_cases"], report["not_run_cases"]), (0, 5))
        self.assertEqual(len(self.trace.read_text().splitlines()), 1)

    def test_output_cannot_be_inside_corpus(self):
        (self.corpus / "expectations.json").write_text(canonical_json(self.spec))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as error:
            main(
                [
                    str(self.corpus),
                    "--duckdb",
                    str(self.cli),
                    "--paimon-extension",
                    str(self.extension),
                    "--registry",
                    str(self.registry),
                    "--case",
                    "empty_rows",
                    "--output",
                    str(self.corpus / "reports"),
                ]
            )
        self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.corpus / "reports").exists())
        self.assertFalse(self.trace.exists())
