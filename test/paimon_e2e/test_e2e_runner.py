"""Exercise the launcher's real CLI; child fixtures do not execute a GPU."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


RUNNER = Path(__file__).with_name("run_e2e.py")
XML = """<Catch2TestRun catch2-version="3.16.1" xml-format-version="3">
<TestCase name="Paimon E2E append snapshot GPU"><OverallResult success="true" skips="0"/></TestCase>
<TestCase name="Paimon E2E forced CPU rejects GPU qualification"><OverallResult success="true" skips="0"/></TestCase>
<OverallResults successes="8" failures="0" expectedFailures="0" skips="0"/>
<OverallResultsCases successes="2" failures="0" expectedFailures="0" skips="0"/>
</Catch2TestRun>"""


class E2ERunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.binary = self.root / "test-binary"

    def invoke(self, xml=XML, ending="", *, expected=0, bad_hash=False, timeout=3):
        self.binary.write_text(
            f"#!{sys.executable}\n"
            "import os, pathlib, signal, sys, time\n"
            "assert sys.argv[1] == '[paimon_e2e]'\n"
            "assert sys.argv[2:4] == ['--reporter', 'xml']\n"
            "pathlib.Path('launched').write_text('yes')\n"
            + (
                f"pathlib.Path(sys.argv[sys.argv.index('--out')+1]).write_text({xml!r})\n"
                if xml is not None
                else ""
            )
            + ending
        )
        self.binary.chmod(0o755)
        digest = (
            "0" * 64
            if bad_hash
            else hashlib.sha256(self.binary.read_bytes()).hexdigest()
        )
        output = self.root / "attempts"
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--binary",
                str(self.binary),
                "--sha256",
                digest,
                "--output",
                str(output),
                "--timeout",
                str(timeout),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        reports = list(output.glob("run-*/report.json"))
        self.assertEqual(len(reports), 1, result.stdout + result.stderr)
        self.attempt = reports[0].parent
        self.report = json.loads(reports[0].read_text())
        return self.report

    def test_complete_pair_passes_and_preserves_actual_argv_and_hash(self):
        r = self.invoke()
        self.assertEqual(r["state"], "PASS")
        self.assertEqual(r["process"]["exit_status"], 0)
        self.assertEqual(
            r["binary_sha256"], hashlib.sha256(self.binary.read_bytes()).hexdigest()
        )
        self.assertEqual((self.attempt / "tests.xml").read_text(), XML)

    def test_missing_report_and_truncated_xml_fail(self):
        for xml in (None, "", XML[:-20]):
            with self.subTest(xml=xml), tempfile.TemporaryDirectory() as tmp:
                self.root = Path(tmp)
                self.binary = self.root / "child"
                self.assertEqual(self.invoke(xml, expected=1)["state"], "FAIL")

    def test_wrong_duplicate_missing_or_skipped_case_cannot_pass(self):
        bad = [
            XML.replace("append snapshot GPU", "unrelated query"),
            XML.replace(
                "Paimon E2E forced CPU rejects GPU qualification",
                "Paimon E2E append snapshot GPU",
            ),
            XML.replace('success="true"', 'success="false"', 1),
            XML.replace('skips="0"', 'skips="1"', 1),
            XML.replace('successes="2"', 'successes="0"'),
            XML.replace('expectedFailures="0"', 'expectedFailures="1"'),
            XML.replace('successes="8"', 'successes="0"'),
            XML.replace('failures="0"', 'failures="1"'),
        ]
        for index, xml in enumerate(bad):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                self.root = Path(tmp)
                self.binary = self.root / "child"
                self.assertEqual(self.invoke(xml, expected=1)["state"], "FAIL")

    def test_status_failure_wins_over_complete_report_and_expected_text(self):
        r = self.invoke(
            ending="print('expected GPU refusal', flush=True)\nsys.exit(7)\n",
            expected=1,
        )
        self.assertEqual(r["process"]["exit_status"], 7)
        self.assertIn("expected GPU refusal", (self.attempt / "stdout.log").read_text())

    @unittest.skipUnless(os.name == "posix", "process signal status")
    def test_signal_cannot_pass_with_successful_report(self):
        r = self.invoke(ending="os.kill(os.getpid(), signal.SIGTERM)\n", expected=1)
        self.assertEqual(r["process"]["exit_status"], -signal.SIGTERM)
        self.assertEqual(r["process"]["termination"], "signal")

    def test_no_test_exit_is_failure(self):
        self.assertEqual(
            self.invoke(xml=None, ending="sys.exit(2)\n", expected=1)["state"], "FAIL"
        )

    def test_timeout_preserves_partial_output_and_report_but_fails(self):
        r = self.invoke(
            ending="print('partial output', flush=True)\ntime.sleep(30)\n",
            expected=1,
            timeout=2,
        )
        self.assertEqual(r["process"]["termination"], "timeout")
        self.assertIn("partial output", (self.attempt / "stdout.log").read_text())
        self.assertTrue((self.attempt / "tests.xml").exists())

    def test_hash_mismatch_prevents_launch(self):
        r = self.invoke(expected=1, bad_hash=True)
        self.assertEqual(r["process"]["termination"], "not_started")
        self.assertFalse((self.attempt / "launched").exists())

    def test_executable_mutation_during_run_fails(self):
        self.invoke(
            ending="with open(sys.argv[0], 'a') as out: out.write('# drift')\n",
            expected=1,
        )
        self.assertIn("changed", self.report["error"])


if __name__ == "__main__":
    unittest.main()
