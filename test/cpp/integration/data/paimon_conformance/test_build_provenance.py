"""Check the workflow identity contract without dispatching any GPU jobs."""

import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from build_provenance import verify_run, verify_checkout


class BuildProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.head = "a" * 40
        self.run = {
            "id": 123,
            "head_sha": self.head,
            "event": "workflow_dispatch",
            "status": "completed",
            "path": ".github/workflows/test.yml",
        }

    def test_plain_and_ref_suffixed_workflow_paths(self):
        for path in (".github/workflows/test.yml", ".github/workflows/test.yml@main"):
            verify_run({**self.run, "path": path}, self.head)

    def test_bad_run_identity_fails(self):
        for key, value in (
            ("head_sha", "b" * 40),
            ("path", ".github/workflows/other.yml@main"),
            ("status", "in_progress"),
            ("event", "pull_request_target"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_run({**self.run, key: value}, self.head)

    def test_pr_merge_checkout_is_labeled_separately(self):
        result = verify_checkout(
            {**self.run, "event": "pull_request"}, self.head, "b" * 40
        )
        self.assertEqual(result["source_head_sha"], self.head)
        self.assertEqual(result["build_checkout_sha"], "b" * 40)
        self.assertEqual(result["description"], "PR merge build at run time")

    def test_direct_build_requires_exact_checkout(self):
        for event in ("workflow_dispatch", "merge_group"):
            run = {**self.run, "event": event}
            self.assertEqual(
                verify_checkout(run, self.head, self.head)["build_checkout_sha"],
                self.head,
            )
            with self.assertRaisesRegex(ValueError, "Non-PR build"):
                verify_checkout(run, self.head, "b" * 40)

    def test_missing_or_malformed_checkout_fails(self):
        for sha in ("", "abc", "g" * 40):
            with self.subTest(sha=sha), self.assertRaisesRegex(
                ValueError, "checkout SHA"
            ):
                verify_checkout(self.run, self.head, sha)

    def test_cli_checks_remain_active_with_python_optimization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_file, sha_file = root / "run.json", root / "sha.txt"
            sha_file.write_text(self.head + "\n")
            command = [
                sys.executable,
                "-O",
                str(Path(__file__).with_name("build_provenance.py")),
                "--run",
                str(run_file),
                "--expected-sha",
                self.head,
                "--checkout-sha-file",
                str(sha_file),
            ]
            run_file.write_text(json.dumps(self.run))
            good = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(json.loads(good.stdout)["build_checkout_sha"], self.head)
            run_file.write_text(json.dumps({**self.run, "head_sha": "b" * 40}))
            bad = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("source head does not match", bad.stderr)
