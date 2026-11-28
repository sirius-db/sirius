"""Exercise launcher cleanup and report publication with injected failures."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import struct
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import run_e2e


class AnchoredLifecycleTests(unittest.TestCase):
    def exercise(self, payload, *, group_error=None, anchor_exit=-signal.SIGKILL):
        facts = {"termination": "not_started", "exit_status": None}
        order = []
        actual_write = os.write

        def accept_control_byte(descriptor, payload):
            # The mock anchor has no retained control reader. Model its
            # acceptance so BrokenPipeError cannot mask the cleanup observer.
            return 1 if payload == b"x" else actual_write(descriptor, payload)

        def spawn(argv, **kwargs):
            # No real process or signal: only the private channel uses actual FDs.
            status_fd = int(argv[4])
            os.write(status_fd, payload)

            def wait(timeout):
                self.assertEqual(order, ["group_signal"])
                order.append("reap")
                return anchor_exit

            return SimpleNamespace(pid=12345, wait=wait, returncode=None)

        def kill_group(pid, sig):
            self.assertEqual((pid, sig), (12345, signal.SIGKILL))
            order.append("group_signal")
            if group_error:
                raise group_error

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            run_e2e.subprocess, "Popen", side_effect=spawn
        ), mock.patch.object(
            run_e2e.os, "killpg", side_effect=kill_group
        ), mock.patch.object(
            run_e2e.os, "write", side_effect=accept_control_byte
        ):
            error = None
            try:
                run_e2e.run_child(
                    ["test-binary"], Path(directory), 1, facts, run_e2e.Interrupts()
                )
            except (
                Exception,
                KeyboardInterrupt,
                run_e2e.TerminationRequested,
            ) as caught:
                error = caught
        return facts, error, order

    def test_unreaped_anchor_signalled_before_wait(self):
        facts, error, order = self.exercise(struct.pack("!i", 0))
        self.assertIsNone(error)
        self.assertEqual(order, ["group_signal", "reap"])
        self.assertEqual(facts["exit_status"], 0)
        self.assertEqual(facts["cleanup"]["anchor_exit_status"], -signal.SIGKILL)

    def test_status_eof_after_deadline_is_not_success(self):
        data = bytearray(struct.pack("!i", 0))
        with mock.patch.object(
            run_e2e.time, "monotonic", side_effect=[0.0, 0.003]
        ), mock.patch.object(
            run_e2e.select, "select", return_value=([42], [], [])
        ) as ready, mock.patch.object(
            run_e2e.os, "read", return_value=b""
        ):
            status = run_e2e.read_child_status(42, data, deadline=0.002)

        self.assertIsNone(status)
        ready.assert_called_once_with([42], [], [], 0.002)

    def test_cpp_failure_and_raw_survive_group_error(self):
        facts, error, _ = self.exercise(
            struct.pack("!i", 7), group_error=PermissionError("denied")
        )
        self.assertEqual(facts["exit_status"], 7)
        self.assertIn("status 7", str(error))
        self.assertEqual(facts["cleanup"]["state"], "UNKNOWN")
        self.assertTrue(facts["cleanup"]["errors"])

    def test_absent_short_or_invalid_status_never_uses_anchor_exit_as_cpp_raw(self):
        for payload in (
            b"",
            b"\0\0",
            struct.pack("!i", 256),
            struct.pack("!i", 0) + b"x",
        ):
            with self.subTest(payload=payload):
                facts, error, order = self.exercise(payload)
                self.assertIsNotNone(error)
                self.assertIsNone(facts["exit_status"])
                self.assertEqual(order, ["group_signal", "reap"])
                self.assertEqual(facts["cleanup"]["state"], "UNKNOWN")

    def test_unexpected_anchor_exit_is_not_pair_success(self):
        facts, error, _ = self.exercise(struct.pack("!i", 0), anchor_exit=0)
        self.assertEqual(facts["exit_status"], 0)
        self.assertIsNotNone(error)
        self.assertEqual(facts["cleanup"]["state"], "UNKNOWN")

    def test_interrupts_remember_first_signal_without_raising_in_handler(self):
        interrupts = run_e2e.Interrupts()
        interrupts.request(signal.SIGTERM, None)
        interrupts.request(signal.SIGINT, None)
        with self.assertRaises(run_e2e.TerminationRequested):
            interrupts.check()

    def test_observation_exception_does_not_skip_group_cleanup(self):
        with mock.patch.object(
            run_e2e, "read_child_status", side_effect=ValueError("observer failed")
        ) as observer:
            facts, error, order = self.exercise(struct.pack("!i", 0))
        self.assertIsInstance(error, ValueError)
        self.assertEqual(order, ["group_signal", "reap"])
        self.assertEqual(observer.call_count, 3)
        self.assertEqual(facts["cleanup"]["state"], "UNKNOWN")
        self.assertIsNone(facts["exit_status"])

    def test_publication_failure_preserves_primary_interrupt_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"not executed")
            stderr = io.StringIO()
            with mock.patch.object(
                run_e2e, "run_child", side_effect=run_e2e.TerminationRequested()
            ), mock.patch.object(
                Path, "write_bytes", side_effect=OSError("publication failed")
            ), redirect_stdout(
                io.StringIO()
            ), redirect_stderr(
                stderr
            ):
                status = run_e2e.main(
                    [
                        "--binary",
                        str(binary),
                        "--sha256",
                        hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "--output",
                        str(root / "attempts"),
                    ]
                )
            self.assertEqual(status, 143)
            fallback = json.loads(stderr.getvalue())
            self.assertEqual(fallback["error_type"], "TerminationRequested")
            self.assertIn("publication failed", fallback["persistence_error"])

    def run_cancellation(self, signum, *, after_child):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"not executed")
            output = root / "attempts"
            interrupts = run_e2e.Interrupts()

            def exit_then_cancel(argv, directory, timeout, facts, requests):
                facts.update(termination="exited", exit_status=0)
                requests.request(signum, None)

            if not after_child:
                interrupts.request(signum, None)
            with mock.patch.object(
                run_e2e, "Interrupts", return_value=interrupts
            ), mock.patch.object(
                run_e2e, "run_child", side_effect=exit_then_cancel
            ) as child, redirect_stdout(
                io.StringIO()
            ):
                status = run_e2e.main(
                    [
                        "--binary",
                        str(binary),
                        "--sha256",
                        hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(child.called, after_child)
            report = json.loads(next(output.glob("run-*/report.json")).read_text())
            self.assertEqual(
                (status, report["state"], report["interrupt_signal"]),
                (128 + signum, "FAIL", signum),
            )
            return report["process"]

    def test_cancellation_preserves_cpp_process_state(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum, point="before launch"):
                self.assertEqual(
                    self.run_cancellation(signum, after_child=False),
                    {"termination": "not_started", "exit_status": None},
                )
            with self.subTest(signal=signum, point="after child exit"):
                self.assertEqual(
                    self.run_cancellation(signum, after_child=True),
                    {"termination": "exited", "exit_status": 0},
                )


if __name__ == "__main__":
    unittest.main()
