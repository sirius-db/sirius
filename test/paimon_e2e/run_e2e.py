#!/usr/bin/env python3
"""Run the first dedicated Paimon E2E test pair; retain every attempt.

This verifies execution/report integrity, not independent build provenance. The
selected C++ tests must implement the SQL, route and independent-result checks.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

from process_anchor import LAUNCH_FAILED


CASES = {
    "Paimon E2E append snapshot GPU",
    "Paimon E2E forced CPU rejects GPU qualification",
}


class TerminationRequested(BaseException):
    """Route SIGTERM through the owned-child cleanup and retained report."""


class Interrupts:
    """Handlers record requests; only explicit checkpoints unwind execution."""

    def __init__(self):
        self.signum = None

    def request(self, signum, frame):
        if self.signum is None:
            self.signum = signum

    def check(self):
        if self.signum == signal.SIGTERM:
            raise TerminationRequested()
        if self.signum == signal.SIGINT:
            raise KeyboardInterrupt()


@contextmanager
def regular_reader(path):
    def opener(name, flags):
        return os.open(name, flags | os.O_NONBLOCK | os.O_NOFOLLOW)

    with open(path, "rb", opener=opener) as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        yield stream


def digest(path):
    with regular_reader(path) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_xml(raw):
    root = ET.fromstring(raw)
    if root.tag != "Catch2TestRun" or root.get("xml-format-version") != "3":
        raise ValueError("Expected Catch2 XML format 3")
    cases = root.findall("TestCase")
    names = [case.get("name") for case in cases]
    if len(names) != len(CASES) or set(names) != CASES:
        raise ValueError(f"Incomplete or unexpected E2E case discovery: {names}")
    for case in cases:
        results = case.findall("OverallResult")
        if len(results) != 1 or results[0].attrib != {"success": "true", "skips": "0"}:
            raise ValueError("E2E case failed, skipped or lacks a complete result")
    for tag in ("Exception", "FatalErrorCondition", "Failure", "Skip"):
        if root.findall(".//" + tag):
            raise ValueError(f"Unexpected {tag} in E2E report")
    for expression in root.findall(".//Expression"):
        if expression.get("success") != "true":
            raise ValueError("Unsuccessful assertion in E2E report")
    for result in root.findall(".//OverallResults"):
        if result.get("failures") != "0" or result.get("expectedFailures") != "0":
            raise ValueError("Failed or expected-failure assertions in E2E report")
        if result.get("skipped", "false") != "false" or result.get("skips", "0") != "0":
            raise ValueError("Skipped assertions in E2E report")
    for tag, expected_successes in (
        ("OverallResults", None),
        ("OverallResultsCases", len(CASES)),
    ):
        results = root.findall(tag)
        if len(results) != 1:
            raise ValueError(f"Missing or duplicate {tag} aggregate")
        value = results[0]
        if any(
            value.get(field) != "0"
            for field in ("failures", "expectedFailures", "skips")
        ):
            raise ValueError(f"Unsuccessful {tag} aggregate")
        successes = int(value.attrib["successes"])
        if successes <= 0 or (
            expected_successes is not None and successes != expected_successes
        ):
            raise ValueError(f"Empty or inconsistent {tag} aggregate")
    return names


def read_child_status(descriptor, data, deadline, interrupts=None):
    """Read a closed launch-failure record or a signed C++ exit status."""
    while (remaining := deadline - time.monotonic()) > 0:
        if interrupts is not None:
            interrupts.check()
        if len(data) > 4:
            raise RuntimeError("Invalid C++ status length")
        if select.select([descriptor], [], [], min(remaining, 0.02))[0]:
            chunk = os.read(descriptor, 5 - len(data))
            if not chunk:
                if time.monotonic() >= deadline:
                    return None
                if data == LAUNCH_FAILED:
                    return LAUNCH_FAILED
                if len(data) != 4:
                    raise RuntimeError("Missing or truncated C++ status")
                code = struct.unpack("!i", data)[0]
                if not -signal.NSIG < code <= 255:
                    raise RuntimeError("Invalid C++ status")
                return code
            data.extend(chunk)
    return None


def run_child(argv, directory, timeout, facts, interrupts):
    process = None
    code = None
    primary = None
    channels = []
    data = bytearray()
    cleanup = facts["cleanup"] = {
        "state": "NOT_STARTED",
        "scope": "own_process_group",
        "errors": [],
    }
    # This standalone CLI has one reaper. SIGCHLD must not auto-reap the anchor
    # before the last group signal; no Popen.poll/wait precedes that signal.
    previous_sigchld = None
    logs = []
    try:
        stdout = (directory / "stdout.log").open("wb")
        logs.append(stdout)
        stderr = (directory / "stderr.log").open("wb")
        logs.append(stderr)
        previous_sigchld = signal.signal(signal.SIGCHLD, signal.SIG_DFL)
        control_read, control_write = os.pipe()
        channels.extend([control_read, control_write])
        status_read, status_write = os.pipe()
        channels.extend([status_read, status_write])
        process = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("process_anchor.py")),
                str(control_read),
                str(status_write),
                *argv,
            ],
            cwd=directory,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
            pass_fds=(control_read, status_write),
        )
        facts["anchor_pid"] = process.pid
        os.close(control_read)
        channels.remove(control_read)
        os.close(status_write)
        channels.remove(status_write)
        facts["termination"] = "running"
        code = read_child_status(
            status_read, data, time.monotonic() + timeout, interrupts
        )
        if code is None:
            facts["termination"] = "timeout"
            raise RuntimeError(f"E2E process timed out after {timeout}s")
        if code == LAUNCH_FAILED:
            facts["termination"] = "launch_error"
            raise RuntimeError("Could not launch the C++ test binary. See stderr.log")
        facts["termination"] = "signal" if code < 0 else "exited"
        if code:
            raise RuntimeError(f"E2E test binary exited with status {code}")
    except (Exception, KeyboardInterrupt, TerminationRequested) as error:
        primary = error
        if isinstance(error, OSError):
            facts["termination"] = (
                "launch_error" if process is None else "channel_error"
            )
        elif isinstance(error, KeyboardInterrupt):
            facts["termination"] = "interrupted"
        elif isinstance(error, TerminationRequested):
            facts["termination"] = "terminated"
    finally:
        if process is not None:
            # Capture the active exception without replacing it with a later
            # cleanup error. A cancellation asks the anchor to publish the
            # actual C++ status before we stop the retained group.
            if code is None:
                try:
                    os.write(control_write, b"x")
                    code = read_child_status(status_read, data, time.monotonic() + 3)
                    if code is None:
                        cleanup["errors"].append("C++ status remains unknown")
                except Exception as error:
                    cleanup["errors"].append(type(error).__name__ + ": " + str(error))
            try:
                # The anchor has not been reaped, even if it died early.
                # Its reserved PID prevents group-number reassignment.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as error:
                cleanup["errors"].append(type(error).__name__ + ": " + str(error))
            try:
                cleanup["anchor_exit_status"] = process.wait(timeout=2)
                if cleanup["anchor_exit_status"] != -signal.SIGKILL:
                    cleanup["errors"].append("Anchor ended before group cleanup")
            except Exception as error:
                cleanup["errors"].append(type(error).__name__ + ": " + str(error))
            if code is None:
                # Closing the anchor can make a previously incomplete
                # status channel terminal. Retain a newly observed C++ raw
                # status even though the attempt already failed.
                try:
                    code = read_child_status(status_read, data, time.monotonic() + 0.02)
                except Exception as error:
                    cleanup["errors"].append(type(error).__name__ + ": " + str(error))
        for descriptor in channels:
            try:
                os.close(descriptor)
            except Exception as error:
                cleanup["errors"].append(type(error).__name__ + ": " + str(error))
        for stream in reversed(logs):
            try:
                stream.close()
            except Exception as error:
                cleanup["errors"].append(type(error).__name__ + ": " + str(error))
        if previous_sigchld is not None:
            try:
                signal.signal(signal.SIGCHLD, previous_sigchld)
            except Exception as error:
                cleanup["errors"].append(type(error).__name__ + ": " + str(error))
        if process is not None:
            cleanup["state"] = (
                "UNKNOWN" if cleanup["errors"] else "GROUP_SIGNALLED_ANCHOR_REAPED"
            )
        if facts["termination"] == "running":
            facts["termination"] = "unknown"
        if code != LAUNCH_FAILED:
            facts["exit_status"] = code
    if primary is not None:
        raise primary
    if cleanup["errors"]:
        raise RuntimeError("E2E group cleanup failed: " + "; ".join(cleanup["errors"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    if os.name != "posix":
        parser.error("This runner requires POSIX process-group cleanup")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output)).resolve()
    report = {
        "state": "NOT RUN",
        "scope": "selected C++ test pair; not a complete E2E or build-provenance qualification",
        "expected_cases": sorted(CASES),
        "process": {"termination": "not_started", "exit_status": None},
    }
    status = 1
    interrupts = Interrupts()
    previous = {
        sig: signal.signal(sig, interrupts.request)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        interrupts.check()
        binary = args.binary.resolve(strict=True)
        observed = digest(binary)
        report.update(
            binary=str(binary), binary_sha256=observed, requested_sha256=args.sha256
        )
        if observed != args.sha256:
            raise ValueError("Test binary SHA256 mismatch")
        command = [
            str(binary),
            "[paimon_e2e]",
            "--reporter",
            "xml",
            "--out",
            str(directory / "tests.xml"),
        ]
        report["argv"] = command
        interrupts.check()
        run_child(command, directory, args.timeout, report["process"], interrupts)
        interrupts.check()
        if digest(binary) != observed:
            raise ValueError("Test binary changed during execution")
        with regular_reader(directory / "tests.xml") as stream:
            raw_xml = stream.read()
        report["cases"] = verify_xml(raw_xml)
        report["xml_sha256"] = hashlib.sha256(raw_xml).hexdigest()
        interrupts.check()
        report["state"] = "PASS"
        status = 0
    except KeyboardInterrupt:
        report.update(state="FAIL", error="Interrupted", error_type="KeyboardInterrupt")
        status = 130
    except TerminationRequested:
        report.update(
            state="FAIL", error="SIGTERM received", error_type="TerminationRequested"
        )
        status = 143
    except Exception as error:
        report.update(state="FAIL", error=str(error), error_type=type(error).__name__)
    finally:
        try:
            if interrupts.signum is not None:
                report["interrupt_signal"] = interrupts.signum
                if report["state"] == "PASS":
                    report.update(
                        state="FAIL",
                        error="Interrupted before publication",
                        error_type="TerminationRequested",
                    )
                    status = 128 + interrupts.signum
            temporary = directory / "report.json.tmp"
            raw_report = (json.dumps(report, indent=2) + "\n").encode()
            temporary.write_bytes(raw_report)
            temporary.replace(directory / "report.json")
            # The C++ command has redirected stdout/stderr and no inherited
            # launcher stream FD. Outer retains this original single-writer
            # publication record, not a hash reconstructed after drainage.
            print(
                json.dumps(
                    {
                        "event": "inner_report_persisted",
                        "state": report["state"],
                        "attempt": str(directory),
                        "report_sha256": hashlib.sha256(raw_report).hexdigest(),
                    }
                ),
                flush=True,
            )
        except OSError as error:
            report.update(
                state="FAIL", persistence_error=type(error).__name__ + ": " + str(error)
            )
            print(json.dumps(report), file=sys.stderr, flush=True)
            status = status or 1
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    # Requests recorded during publication cannot silently return launcher0.
    # Later signals use the restored handler; outer records the real launcher
    # status separately from the already-persisted scoped C++ pair result.
    if interrupts.signum is not None and status == 0:
        print(
            json.dumps(
                {"event": "interrupt_after_report", "signal": interrupts.signum}
            ),
            file=sys.stderr,
            flush=True,
        )
        status = 128 + interrupts.signum
    return status


if __name__ == "__main__":
    raise SystemExit(main())
