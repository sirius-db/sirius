#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from step_r import Qualification


class StepRControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.binding = {"build": "A", "config": "A", "domain": "loopback-test"}
        self.g = Qualification(
            self.root / "registry/db", self.root / "control", lambda: self.binding
        )

    def capture(self):
        self.g.begin()
        for name in ("location", "reference"):
            (self.g.control / name).write_text(name)
        with patch("step_r.time.monotonic", return_value=1):
            self.g.captured(
                self.g.control / "location", self.g.control / "reference", {}
            )

    def passed(self):
        self.capture()
        with patch("step_r.time.monotonic", return_value=601):
            self.assertEqual(self.g.finish(lambda p: None), 600)

    def test_age_boundary(self):
        self.capture()
        with patch("step_r.time.monotonic", return_value=600.99), self.assertRaises(
            RuntimeError
        ):
            self.g.finish(lambda p: self.fail("replay before age gate"))
        self.assertTrue(self.g.broken)

    def test_success_and_restart_denial(self):
        self.passed()
        with patch("step_r.time.monotonic", return_value=602):
            self.g.require()
        other = Qualification(self.g.registry, self.g.control, lambda: self.binding)
        with self.assertRaises(RuntimeError):
            other.require()

    def test_mutation_revokes(self):
        for mutation in ("config", "location", "reference", "capture", "registry"):
            with self.subTest(mutation=mutation):
                self.setUp()
                self.passed()
                if mutation == "config":
                    self.binding["config"] = "changed"
                elif mutation == "registry":
                    self.g.registry.write_bytes(b"broken db")
                else:
                    (
                        self.g.control
                        / ("capture.json" if mutation == "capture" else mutation)
                    ).write_text("tampered")
                with self.assertRaises(Exception):
                    self.g.require()
                self.assertTrue(self.g.broken)
                self.assertFalse(self.g.allowed)

    def test_failed_requalification_revokes_even_when_storage_recovers(self):
        self.passed()
        old = self.g.generation
        with patch.object(
            self.g, "storage_probe", side_effect=OSError("local storage unavailable")
        ), self.assertRaises(OSError):
            self.g.begin()
        self.assertGreater(self.g.generation, old)
        with self.assertRaises(RuntimeError):
            self.g.require()

    def test_registration_failure_revokes_in_memory(self):
        self.passed()
        with patch.object(
            self.g, "connect", side_effect=OSError("registry unavailable")
        ), self.assertRaises(OSError):
            self.g.begin()
        with self.assertRaises(RuntimeError):
            self.g.require()

    def test_new_generation_invalidates_prior_pass(self):
        self.passed()
        other = Qualification(
            self.g.registry, self.root / "other-control", lambda: self.binding
        )
        other.begin()
        with self.assertRaises(RuntimeError):
            self.g.require()

    def test_failed_replay_and_consumer_revoke(self):
        self.capture()
        with patch("step_r.time.monotonic", return_value=601), self.assertRaises(
            ValueError
        ):
            self.g.finish(lambda p: (_ for _ in ()).throw(ValueError("truncated")))
        with self.assertRaises(RuntimeError):
            self.g.require()
        self.setUp()
        self.passed()
        with patch("step_r.time.monotonic", return_value=602), self.assertRaises(
            ValueError
        ):
            self.g.consume(lambda: (_ for _ in ()).throw(ValueError("consumer failed")))
        with self.assertRaises(RuntimeError):
            self.g.require()

    def test_expiry(self):
        self.passed()
        with patch("step_r.time.monotonic", return_value=3602), self.assertRaises(
            RuntimeError
        ):
            self.g.require()


if __name__ == "__main__":
    unittest.main(verbosity=2)
