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
import sys
import tempfile
import unittest
from unittest.mock import patch

from corpus_checks import canonical_json, digest
from run_conformance import main

HERE = Path(__file__).resolve().parent


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
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
with open(trace, 'a') as f: f.write(json.dumps({'sql':sql,'mode':mode})+'\\n')
fault = os.environ.get('PAIMON_TEST_FAULT', '')
if sql == 'SELECT version() AS version; PRAGMA platform;':
    print(json.dumps([{'version':os.environ.get('PAIMON_TEST_VERSION', spec['extension']['duckdb_version'])}]))
    print(json.dumps([{'platform':spec['extension']['platform']}]))
    sys.exit(0)

if 'SET enable_duckdb_fallback=false;' in sql:
    print("GPU plan generation failed: Table function 'paimon_scan' is not supported in Sirius", file=sys.stderr, flush=True)
    if fault == 'rejection_crash':
        import resource, signal
        resource.setrlimit(resource.RLIMIT_CORE, (0,0))
        os.kill(os.getpid(), signal.SIGSEGV)
    sys.exit(1)
attached = 'ATTACH ' in sql
case = None
for candidate in spec['cases']:
    table = manifest['tables'][candidate['table']]
    snapshot = table['snapshots'][candidate['snapshot']] if candidate['snapshot'] else None
    scan = "paimon_scan('" + (corpus+'/'+table['path']).replace("'", "''") + "'"
    if snapshot is not None: scan += ', snapshot_from_id='+str(snapshot)
    scan += ')'
    query = candidate['sql'].format(scan=scan)
    if (attached and candidate['id']=='pk_d') or (not attached and 'DESCRIBE '+query+';' in sql):
        case=candidate
        break
if case is None: raise RuntimeError('Unexpected test SQL: '+sql)
columns=[{'column_name':n,'column_type':t} for n,t in case['columns']]
rows=[dict(zip([n for n,t in case['columns']], row)) for row in case['rows']]
results=[[{'extension_version':spec['extension']['version']}],[{'version':spec['extension']['duckdb_version']}],[{'gpu':mode=='transparent'}],columns,rows,[{'probe':'LIVENESS_OK'}]]
target = os.environ.get('PAIMON_TEST_TARGET', 'disabled/empty_rows')
label = 'attached' if attached else mode+'/'+case['id']
if label == target:
    if fault == 'missing': results.pop(4)
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

    def run_suite(self, selected=(), fault="", target="disabled/empty_rows"):
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
        ]
        for case in selected:
            args.extend(["--case", case])
        error = None
        with patch.dict(
            os.environ, PAIMON_TEST_FAULT=fault, PAIMON_TEST_TARGET=target
        ), contextlib.redirect_stdout(io.StringIO()):
            try:
                main(args)
            except Exception as caught:
                error = caught
        report = json.loads(
            next((self.root / "reports").glob("run-*/report.json")).read_text()
        )
        return report, error

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
        attached = [
            json.loads(line)["sql"]
            for line in self.trace.read_text().splitlines()
            if "ATTACH " in line
        ][0]
        self.assertLess(attached.index("ATTACH "), attached.index("DESCRIBE "))

    def test_incomplete_and_bad_protocol_fail_but_later_cases_run(self):
        for fault in (
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

    def test_unqualified_reader_version_fails_before_load(self):
        with patch.dict(os.environ, PAIMON_TEST_VERSION="v1.5.6"):
            report, error = self.run_suite(["empty_rows"])
        self.assertIn("No qualified Paimon artifact", str(error))
        self.assertEqual(report["not_run_cases"], 5)
        self.assertEqual(len(self.trace.read_text().splitlines()), 1)
